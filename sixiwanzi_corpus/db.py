"""SQLite 数据库连接、建库、事务封装与增量写入工具."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from .tokenizer import tokenize_for_fts

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


@dataclass
class VideoRecord:
    bvid: str
    page: int
    up_name: str
    pub_date: str
    cid: Optional[int] = None
    up_uid: Optional[int] = None
    raw_up_name: str = ""
    title: str = ""
    part_title: str = ""
    pub_ts: int = 0
    duration_sec: int = 0
    subtitle_lan: str = ""
    subtitle_status: str = "pending"
    line_count: int = 0
    scene_type: str = "unknown"
    scene_confidence: float = 0.0
    scene_reason: str = ""
    scene_locked: int = 0
    is_visible: int = 1
    voiceprint_status: str = "none"
    extra_meta: Dict[str, Any] | str = "{}"


@dataclass
class SubtitleRow:
    up_name: str
    bvid: str
    page: int
    pub_date: str
    start_sec: float
    end_sec: float
    content: str
    scene_type: str = "unknown"
    is_visible: int = 1
    source: str = "bilibili_ai"
    speaker_label: str = "unknown"
    speaker_name: str = ""
    speaker_sim: Optional[float] = None
    content_locked: int = 0
    speaker_locked: int = 0


class CorpusDB:
    """SQLite 语料库访问层（保持纯标准 SQL，便于后续抽象为 PostgreSQL 适配层）."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)

    def connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=60.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 60000;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        return conn

    @contextmanager
    def session(self) -> Iterator[sqlite3.Connection]:
        conn = self.connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def init_db(self) -> None:
        schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self.session() as conn:
            conn.executescript(schema_sql)
            self._ensure_schema_migrations(conn)

    @staticmethod
    def _ensure_schema_migrations(conn: sqlite3.Connection) -> None:
        """对已有旧库执行幂等增量列迁移（保证云端与本地平滑升级）."""
        video_cols = {
            str(r["name"]) for r in conn.execute("PRAGMA table_info(videos)").fetchall()
        }
        if "voiceprint_status" not in video_cols:
            conn.execute(
                "ALTER TABLE videos ADD COLUMN voiceprint_status TEXT NOT NULL DEFAULT 'none'"
            )

        sub_cols = {
            str(r["name"]) for r in conn.execute("PRAGMA table_info(subtitles)").fetchall()
        }
        if "speaker_label" not in sub_cols:
            conn.execute(
                "ALTER TABLE subtitles ADD COLUMN speaker_label TEXT NOT NULL DEFAULT 'unknown'"
            )
        if "speaker_name" not in sub_cols:
            conn.execute(
                "ALTER TABLE subtitles ADD COLUMN speaker_name TEXT NOT NULL DEFAULT ''"
            )
        if "speaker_sim" not in sub_cols:
            conn.execute("ALTER TABLE subtitles ADD COLUMN speaker_sim REAL")
        if "content_locked" not in sub_cols:
            conn.execute(
                "ALTER TABLE subtitles ADD COLUMN content_locked INTEGER NOT NULL DEFAULT 0"
            )
        if "speaker_locked" not in sub_cols:
            conn.execute(
                "ALTER TABLE subtitles ADD COLUMN speaker_locked INTEGER NOT NULL DEFAULT 0"
            )

    def get_video(self, bvid: str, page: int = 1) -> Optional[Dict[str, Any]]:
        with self.session() as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE bvid = ? AND page = ?",
                (bvid, page),
            ).fetchone()
            return dict(row) if row else None

    def is_video_ingested(self, bvid: str, page: int = 1) -> bool:
        with self.session() as conn:
            row = conn.execute(
                "SELECT subtitle_status FROM videos WHERE bvid = ? AND page = ?",
                (bvid, page),
            ).fetchone()
            return bool(row and row["subtitle_status"] == "ok")

    def is_bvid_completed(self, bvid: str, *, allow_no_subtitle: bool = True) -> bool:
        """判断某个 BV 号是否已完整入库（避免断点续抓时重复请求 /x/web-interface/view）."""
        done_statuses = ("ok", "no_subtitle", "asr_empty") if allow_no_subtitle else ("ok", "asr_empty")
        placeholders = ",".join("?" for _ in done_statuses)
        with self.session() as conn:
            row = conn.execute(
                f"""
                SELECT COUNT(*) AS total,
                       SUM(CASE WHEN subtitle_status IN ({placeholders}) THEN 1 ELSE 0 END) AS done
                  FROM videos
                 WHERE bvid = ?
                """,
                (*done_statuses, bvid),
            ).fetchone()
            total = int(row["total"] or 0)
            done = int(row["done"] or 0)
            return total > 0 and total == done

    def list_retry_bvids(self, include_no_subtitle: bool = False) -> List[str]:
        """查询需要重试的 BV 号列表（pending / failed，可选包含 no_subtitle）."""
        statuses = ["pending", "failed"]
        if include_no_subtitle:
            statuses.append("no_subtitle")
        placeholders = ",".join("?" for _ in statuses)
        with self.session() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT bvid
                  FROM videos
                 WHERE subtitle_status IN ({placeholders})
                UNION
                SELECT DISTINCT l.bvid
                  FROM ingest_logs l
                 WHERE l.status = 'failed'
                   AND NOT EXISTS (
                       SELECT 1 FROM videos v
                        WHERE v.bvid = l.bvid AND v.subtitle_status = 'ok'
                   )
                """,
                statuses,
            ).fetchall()
            return [str(r[0]) for r in rows if r[0]]

    def optimize_db(self) -> Dict[str, Any]:
        """大批量入库完成后压缩 FTS5 倒排段、更新统计信息并回收 WAL，保持百万级检索毫秒响应."""
        with self.session() as conn:
            conn.execute("INSERT INTO subtitles_fts(subtitles_fts) VALUES('optimize');")
            conn.execute("ANALYZE;")
        # 事务提交后再执行 WAL checkpoint，避免与未提交写事务自锁
        conn = self.connect()
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        finally:
            conn.close()
        size_mb = (
            round(self.db_path.stat().st_size / (1024 * 1024), 3)
            if self.db_path.exists()
            else 0.0
        )
        return {"status": "ok", "db_path": str(self.db_path), "db_size_mb": size_mb}

    def upsert_video(self, conn: sqlite3.Connection, rec: VideoRecord) -> None:
        extra_str = (
            rec.extra_meta
            if isinstance(rec.extra_meta, str)
            else json.dumps(rec.extra_meta, ensure_ascii=False)
        )
        conn.execute(
            """
            INSERT INTO videos (
                bvid, page, cid, up_uid, up_name, raw_up_name,
                title, part_title, pub_date, pub_ts, duration_sec,
                subtitle_lan, subtitle_status, line_count,
                scene_type, scene_confidence, scene_reason, scene_locked,
                is_visible, voiceprint_status, extra_meta
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?
            )
            ON CONFLICT(bvid, page) DO UPDATE SET
                cid              = COALESCE(excluded.cid, videos.cid),
                up_uid           = COALESCE(excluded.up_uid, videos.up_uid),
                up_name          = excluded.up_name,
                raw_up_name      = CASE WHEN excluded.raw_up_name != '' THEN excluded.raw_up_name ELSE videos.raw_up_name END,
                title            = CASE WHEN excluded.title != '' THEN excluded.title ELSE videos.title END,
                part_title       = CASE WHEN excluded.part_title != '' THEN excluded.part_title ELSE videos.part_title END,
                pub_date         = excluded.pub_date,
                pub_ts           = CASE WHEN excluded.pub_ts > 0 THEN excluded.pub_ts ELSE videos.pub_ts END,
                duration_sec     = CASE WHEN excluded.duration_sec > 0 THEN excluded.duration_sec ELSE videos.duration_sec END,
                subtitle_lan     = CASE WHEN excluded.subtitle_lan != '' THEN excluded.subtitle_lan ELSE videos.subtitle_lan END,
                subtitle_status  = CASE
                                       WHEN videos.subtitle_status = 'asr_empty' AND excluded.subtitle_status = 'no_subtitle'
                                       THEN 'asr_empty'
                                       ELSE excluded.subtitle_status
                                   END,
                line_count       = excluded.line_count,
                -- 若已有非 unknown 的锁定场景标签，增量入库不破坏它
                scene_type       = CASE
                                       WHEN videos.scene_locked = 1 THEN videos.scene_type
                                       WHEN excluded.scene_type != 'unknown' THEN excluded.scene_type
                                       ELSE videos.scene_type
                                   END,
                voiceprint_status = CASE
                                        WHEN excluded.voiceprint_status != 'none' THEN excluded.voiceprint_status
                                        ELSE videos.voiceprint_status
                                    END,
                extra_meta       = CASE WHEN excluded.extra_meta != '{}' THEN excluded.extra_meta ELSE videos.extra_meta END,
                updated_at       = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
            """,
            (
                rec.bvid,
                rec.page,
                rec.cid,
                rec.up_uid,
                rec.up_name,
                rec.raw_up_name,
                rec.title,
                rec.part_title,
                rec.pub_date,
                rec.pub_ts,
                rec.duration_sec,
                rec.subtitle_lan,
                rec.subtitle_status,
                rec.line_count,
                rec.scene_type,
                rec.scene_confidence,
                rec.scene_reason,
                rec.scene_locked,
                rec.is_visible,
                rec.voiceprint_status,
                extra_str,
            ),
        )

    def save_video_and_subtitles(
        self,
        video: VideoRecord,
        subtitles: Sequence[SubtitleRow],
        *,
        replace_existing: bool = False,
    ) -> Tuple[int, int]:
        """事务性写入视频分 P 元数据与字幕列表，返回 (新增条数, 忽略重复条数)."""
        with self.session() as conn:
            existing = conn.execute(
                "SELECT scene_type, is_visible FROM videos WHERE bvid = ? AND page = ?",
                (video.bvid, video.page),
            ).fetchone()

            effective_scene = video.scene_type
            effective_visible = video.is_visible
            if existing:
                if existing["scene_type"] != "unknown" and video.scene_type == "unknown":
                    effective_scene = existing["scene_type"]
                effective_visible = existing["is_visible"]

            if replace_existing:
                conn.execute(
                    "DELETE FROM subtitles WHERE bvid = ? AND page = ?",
                    (video.bvid, video.page),
                )

            before_count = int(
                conn.execute(
                    "SELECT COUNT(*) FROM subtitles WHERE bvid = ? AND page = ?",
                    (video.bvid, video.page),
                ).fetchone()[0]
            )
            rows_to_insert = [
                (
                    s.up_name,
                    s.bvid,
                    s.page,
                    s.pub_date,
                    round(float(s.start_sec), 2),
                    round(float(s.end_sec), 2),
                    s.content,
                    tokenize_for_fts(s.content),
                    effective_scene if s.scene_type == "unknown" else s.scene_type,
                    effective_visible,
                    s.source,
                    s.speaker_label,
                    s.speaker_name,
                    round(float(s.speaker_sim), 4) if s.speaker_sim is not None else None,
                )
                for s in subtitles
            ]

            conn.executemany(
                """
                INSERT OR IGNORE INTO subtitles (
                    up_name, bvid, page, pub_date,
                    start_sec, end_sec, content, content_fts,
                    scene_type, is_visible, source, speaker_label, speaker_name, speaker_sim
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows_to_insert,
            )

            total_in_db = int(
                conn.execute(
                    "SELECT COUNT(*) FROM subtitles WHERE bvid = ? AND page = ?",
                    (video.bvid, video.page),
                ).fetchone()[0]
            )
            inserted = total_in_db - before_count
            skipped = len(rows_to_insert) - inserted
            video.line_count = total_in_db
            self.upsert_video(conn, video)
            return inserted, skipped

    def log_ingest(
        self, run_id: str, bvid: str, page: int, status: str, detail: str = ""
    ) -> None:
        with self.session() as conn:
            conn.execute(
                """
                INSERT INTO ingest_logs (run_id, bvid, page, status, detail)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, bvid, page, status, detail[:1000]),
            )

    def update_visibility(
        self, bvid: str, page: Optional[int], is_visible: int
    ) -> int:
        val = 1 if is_visible else 0
        with self.session() as conn:
            if page is None:
                cur = conn.execute(
                    """
                    UPDATE videos
                       SET is_visible = ?,
                           updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                     WHERE bvid = ?
                    """,
                    (val, bvid),
                )
                conn.execute(
                    "UPDATE subtitles SET is_visible = ? WHERE bvid = ?",
                    (val, bvid),
                )
            else:
                cur = conn.execute(
                    """
                    UPDATE videos
                       SET is_visible = ?,
                           updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                     WHERE bvid = ? AND page = ?
                    """,
                    (val, bvid, page),
                )
                conn.execute(
                    "UPDATE subtitles SET is_visible = ? WHERE bvid = ? AND page = ?",
                    (val, bvid, page),
                )
            return cur.rowcount

    def get_stats(self) -> Dict[str, Any]:
        with self.session() as conn:
            # 全部聚合走轻量 videos 表（仅数千行，<3ms），避免在云端小内存机器上全表扫描 579 万行 subtitles
            summary = conn.execute(
                """
                SELECT COUNT(*) AS video_total,
                       COALESCE(SUM(CASE WHEN subtitle_status = 'ok' THEN 1 ELSE 0 END), 0) AS video_ok,
                       COALESCE(SUM(line_count), 0) AS sub_total,
                       COALESCE(SUM(CASE WHEN is_visible = 1 THEN line_count ELSE 0 END), 0) AS sub_visible
                  FROM videos
                """
            ).fetchone()
            video_total = int(summary["video_total"] or 0)
            video_ok = int(summary["video_ok"] or 0)
            sub_total = int(summary["sub_total"] or 0)
            sub_visible = int(summary["sub_visible"] or 0)

            by_up = [
                dict(r)
                for r in conn.execute(
                    """
                    SELECT up_name,
                           COUNT(*) AS video_parts,
                           SUM(CASE WHEN subtitle_status = 'ok' THEN 1 ELSE 0 END) AS ok_parts,
                           COALESCE(SUM(line_count), 0) AS subtitle_lines
                      FROM videos
                     GROUP BY up_name
                     ORDER BY subtitle_lines DESC, up_name ASC
                    """
                ).fetchall()
            ]
            date_range = conn.execute(
                """
                SELECT MIN(pub_date) AS min_date,
                       MAX(pub_date) AS max_date
                  FROM videos
                 WHERE subtitle_status = 'ok'
                   AND is_visible = 1
                   AND pub_date != ''
                """
            ).fetchone()
            if not date_range or not date_range["min_date"]:
                date_range = conn.execute(
                    "SELECT MIN(pub_date) AS min_date, MAX(pub_date) AS max_date "
                    "FROM videos WHERE pub_date != ''"
                ).fetchone()
            latest_updated = conn.execute(
                "SELECT MAX(updated_at) AS latest_updated_at FROM videos"
            ).fetchone()
            latest_video_row = conn.execute(
                """
                SELECT bvid, page, up_name, title, pub_date, pub_ts
                  FROM videos
                 WHERE subtitle_status = 'ok'
                   AND is_visible = 1
                   AND pub_date != ''
                 ORDER BY pub_date DESC, pub_ts DESC, updated_at DESC, page ASC
                 LIMIT 1
                """
            ).fetchone()

        latest_video: Optional[Dict[str, Any]] = None
        if latest_video_row:
            lv_bvid = str(latest_video_row["bvid"] or "")
            lv_page = int(latest_video_row["page"] or 1)
            lv_url = (
                f"https://www.bilibili.com/video/{lv_bvid}/?p={lv_page}"
                if lv_page > 1
                else f"https://www.bilibili.com/video/{lv_bvid}/"
            )
            latest_video = {
                "bvid": lv_bvid,
                "page": lv_page,
                "up_name": str(latest_video_row["up_name"] or ""),
                "title": str(latest_video_row["title"] or ""),
                "pub_date": str(latest_video_row["pub_date"] or ""),
                "video_url": lv_url,
            }

        db_size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
        return {
            "db_path": str(self.db_path),
            "db_size_mb": round(db_size_bytes / (1024 * 1024), 3),
            "video_parts_total": video_total,
            "video_parts_ok": video_ok,
            "subtitles_total": sub_total,
            "subtitles_visible": sub_visible,
            "date_min": date_range["min_date"] if date_range else None,
            "date_max": date_range["max_date"] if date_range else None,
            "latest_updated_at": (
                latest_updated["latest_updated_at"] if latest_updated else None
            ),
            "latest_video": latest_video,
            "by_up": by_up,
        }

    def update_part_voiceprint_scores(
        self,
        bvid: str,
        page: int,
        scores: Sequence[Tuple[Any, ...]],
        *,
        status: str = "done",
        overwrite_locked: bool = False,
    ) -> int:
        """批量写回某个分 P 的声纹标签、匹配成员名与相似度，默认跳过已人工锁定的条目."""
        with self.session() as conn:
            affected = 0
            if scores:
                params = []
                for row in scores:
                    sid = int(row[0])
                    label = str(row[1])
                    sim = round(float(row[2]), 4) if row[2] is not None else None
                    spk_name = str(row[3]) if len(row) > 3 and row[3] else ""
                    params.append((label, spk_name, sim, sid))
                sql = (
                    "UPDATE subtitles SET speaker_label = ?, speaker_name = ?, speaker_sim = ? WHERE id = ?"
                    if overwrite_locked
                    else "UPDATE subtitles SET speaker_label = ?, speaker_name = ?, speaker_sim = ? WHERE id = ? AND speaker_locked = 0"
                )
                cur = conn.executemany(sql, params)
                affected = cur.rowcount if cur.rowcount is not None and cur.rowcount >= 0 else len(params)
            conn.execute(
                """
                UPDATE videos
                   SET voiceprint_status = ?,
                       updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                 WHERE bvid = ? AND page = ?
                """,
                (status, bvid, int(page)),
            )
            return affected

    @staticmethod
    def _normalize_speaker_fields(
        speaker_label: str, speaker_name: Optional[str], fallback_up_name: str = ""
    ) -> Tuple[str, str]:
        lbl = (speaker_label or "").strip().lower()
        if lbl not in ("unknown", "host", "mixed", "external"):
            raise ValueError(f"无效的声音来源标签: {speaker_label!r}")
        if lbl in ("external", "unknown"):
            return lbl, ""
        name = (speaker_name or "").strip() or (fallback_up_name or "").strip()
        return lbl, name

    def edit_subtitle(
        self,
        subtitle_id: int,
        *,
        content: Optional[str] = None,
        speaker_label: Optional[str] = None,
        speaker_name: Optional[str] = None,
        note: str = "",
    ) -> Dict[str, Any]:
        """人工纠错单条字幕正文和/或声音来源，同步 FTS5、置锁定标记并写入审计日志."""
        with self.session() as conn:
            row = conn.execute(
                """
                SELECT id, up_name, bvid, page, pub_date, start_sec, end_sec,
                       content, speaker_label, speaker_name, speaker_sim,
                       content_locked, speaker_locked
                  FROM subtitles
                 WHERE id = ?
                """,
                (int(subtitle_id),),
            ).fetchone()
            if not row:
                raise KeyError(f"未找到字幕记录: id={subtitle_id}")

            old_content = str(row["content"])
            old_label = str(row["speaker_label"] or "unknown")
            old_name = str(row["speaker_name"] or "")

            new_content = old_content
            text_changed = False
            if content is not None:
                cleaned = " ".join(str(content).strip().split())
                if not cleaned:
                    raise ValueError("字幕内容不能为空")
                new_content = cleaned
                text_changed = new_content != old_content

            new_label = old_label
            new_name = old_name
            speaker_changed = False
            if speaker_label is not None:
                new_label, new_name = self._normalize_speaker_fields(
                    speaker_label, speaker_name, fallback_up_name=str(row["up_name"] or "")
                )
                speaker_changed = (new_label != old_label) or (new_name != old_name)

            new_content_locked = 1 if (content is not None and (text_changed or int(row["content_locked"] or 0))) else int(row["content_locked"] or 0)
            new_speaker_locked = 1 if (speaker_label is not None) else int(row["speaker_locked"] or 0)

            if not text_changed and not speaker_changed and new_speaker_locked == int(row["speaker_locked"] or 0):
                return {
                    "changed": False,
                    "edit_id": None,
                    "subtitle": dict(row),
                }

            try:
                if text_changed:
                    conn.execute(
                        """
                        UPDATE subtitles
                           SET content = ?,
                               content_fts = ?,
                               speaker_label = ?,
                               speaker_name = ?,
                               content_locked = ?,
                               speaker_locked = ?
                         WHERE id = ?
                        """,
                        (
                            new_content,
                            tokenize_for_fts(new_content),
                            new_label,
                            new_name,
                            new_content_locked,
                            new_speaker_locked,
                            int(subtitle_id),
                        ),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE subtitles
                           SET speaker_label = ?,
                               speaker_name = ?,
                               speaker_locked = ?
                         WHERE id = ?
                        """,
                        (new_label, new_name, new_speaker_locked, int(subtitle_id)),
                    )
            except sqlite3.IntegrityError as exc:
                raise ValueError(
                    "修改后的字幕文本与同分 P 相同时间段的另一条字幕完全重复，请检查或保留原句"
                ) from exc

            if text_changed and speaker_changed:
                edit_type = "both"
            elif text_changed:
                edit_type = "text"
            else:
                edit_type = "speaker"

            cur = conn.execute(
                """
                INSERT INTO subtitle_edits (
                    subtitle_id, bvid, page, start_sec, end_sec, edit_type,
                    old_content, new_content,
                    old_speaker_label, old_speaker_name,
                    new_speaker_label, new_speaker_name, note
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    int(subtitle_id),
                    str(row["bvid"]),
                    int(row["page"]),
                    float(row["start_sec"]),
                    float(row["end_sec"]),
                    edit_type,
                    old_content,
                    new_content,
                    old_label,
                    old_name,
                    new_label,
                    new_name,
                    str(note or "").strip()[:300],
                ),
            )
            edit_id = int(cur.lastrowid)

            conn.execute(
                """
                UPDATE videos
                   SET updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                 WHERE bvid = ? AND page = ?
                """,
                (str(row["bvid"]), int(row["page"])),
            )

            updated_row = conn.execute(
                """
                SELECT id, up_name, bvid, page, pub_date, start_sec, end_sec,
                       content, speaker_label, speaker_name, speaker_sim,
                       content_locked, speaker_locked
                  FROM subtitles
                 WHERE id = ?
                """,
                (int(subtitle_id),),
            ).fetchone()
            return {
                "changed": True,
                "edit_id": edit_id,
                "edit_type": edit_type,
                "subtitle": dict(updated_row),
            }

    def batch_edit_speakers(
        self,
        subtitle_ids: Sequence[int],
        *,
        speaker_label: str,
        speaker_name: Optional[str] = None,
        note: str = "",
    ) -> Dict[str, Any]:
        """批量修正多条字幕的声音来源并锁定，适合 Re/观影/团播连续时间段纠错."""
        clean_ids = sorted({int(x) for x in subtitle_ids if int(x) > 0})
        if not clean_ids:
            raise ValueError("请至少选择一条字幕")

        clean_note = str(note or "").strip()[:300]
        updated_count = 0
        touched_parts: set[Tuple[str, int]] = set()
        edit_ids: List[int] = []

        with self.session() as conn:
            ph = ",".join("?" for _ in clean_ids)
            rows = conn.execute(
                f"""
                SELECT id, up_name, bvid, page, start_sec, end_sec, content,
                       speaker_label, speaker_name, speaker_locked
                  FROM subtitles
                 WHERE id IN ({ph})
                 ORDER BY bvid ASC, page ASC, start_sec ASC, id ASC
                """,
                clean_ids,
            ).fetchall()

            for r in rows:
                sid = int(r["id"])
                old_label = str(r["speaker_label"] or "unknown")
                old_name = str(r["speaker_name"] or "")
                new_label, new_name = self._normalize_speaker_fields(
                    speaker_label, speaker_name, fallback_up_name=str(r["up_name"] or "")
                )
                if new_label == old_label and new_name == old_name and int(r["speaker_locked"] or 0) == 1:
                    continue
                conn.execute(
                    """
                    UPDATE subtitles
                       SET speaker_label = ?, speaker_name = ?, speaker_locked = 1
                     WHERE id = ?
                    """,
                    (new_label, new_name, sid),
                )
                cur = conn.execute(
                    """
                    INSERT INTO subtitle_edits (
                        subtitle_id, bvid, page, start_sec, end_sec, edit_type,
                        old_content, new_content,
                        old_speaker_label, old_speaker_name,
                        new_speaker_label, new_speaker_name, note
                    ) VALUES (?, ?, ?, ?, ?, 'speaker', ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sid,
                        str(r["bvid"]),
                        int(r["page"]),
                        float(r["start_sec"]),
                        float(r["end_sec"]),
                        str(r["content"]),
                        str(r["content"]),
                        old_label,
                        old_name,
                        new_label,
                        new_name,
                        clean_note or "批量校对声音来源",
                    ),
                )
                edit_ids.append(int(cur.lastrowid))
                touched_parts.add((str(r["bvid"]), int(r["page"])))
                updated_count += 1

            for bvid, page in touched_parts:
                conn.execute(
                    """
                    UPDATE videos
                       SET updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                     WHERE bvid = ? AND page = ?
                    """,
                    (bvid, page),
                )

        return {
            "status": "ok",
            "requested": len(clean_ids),
            "updated": updated_count,
            "touched_parts": len(touched_parts),
            "edit_ids": edit_ids,
        }

    def list_subtitle_edits(
        self,
        *,
        bvid: Optional[str] = None,
        subtitle_id: Optional[int] = None,
        edit_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        clauses: List[str] = []
        params: List[Any] = []
        if bvid:
            clauses.append("e.bvid = ?")
            params.append(bvid.strip())
        if subtitle_id is not None:
            clauses.append("e.subtitle_id = ?")
            params.append(int(subtitle_id))
        if edit_type:
            clauses.append("e.edit_type = ?")
            params.append(edit_type.strip())
        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        lim = max(1, min(200, int(limit)))
        off = max(0, int(offset))

        with self.session() as conn:
            total = int(
                conn.execute(f"SELECT COUNT(*) FROM subtitle_edits e {where_sql}", params).fetchone()[0]
            )
            rows = conn.execute(
                f"""
                SELECT e.*,
                       COALESCE(v.up_name, '') AS up_name,
                       COALESCE(v.title, '') AS title,
                       COALESCE(v.pub_date, '') AS pub_date
                  FROM subtitle_edits e
                  LEFT JOIN videos v ON v.bvid = e.bvid AND v.page = e.page
                {where_sql}
                 ORDER BY e.id DESC
                 LIMIT ? OFFSET ?
                """,
                [*params, lim, off],
            ).fetchall()
        return {
            "total": total,
            "limit": lim,
            "offset": off,
            "items": [dict(r) for r in rows],
        }

    def revert_subtitle_edit(self, edit_id: int) -> Dict[str, Any]:
        """根据一条审计记录将对应字幕回滚到修改前状态，并写入一条 revert 审计记录."""
        with self.session() as conn:
            ev = conn.execute(
                "SELECT * FROM subtitle_edits WHERE id = ?",
                (int(edit_id),),
            ).fetchone()
            if not ev:
                raise KeyError(f"未找到校对记录: edit_id={edit_id}")

            sid = int(ev["subtitle_id"])
            cur_sub = conn.execute(
                """
                SELECT id, up_name, bvid, page, start_sec, end_sec,
                       content, speaker_label, speaker_name,
                       content_locked, speaker_locked
                  FROM subtitles
                 WHERE id = ?
                """,
                (sid,),
            ).fetchone()
            if not cur_sub:
                raise KeyError(f"对应字幕已不存在: subtitle_id={sid}")

            ev_type = str(ev["edit_type"])
            target_content = str(cur_sub["content"])
            target_content_locked = int(cur_sub["content_locked"] or 0)
            target_label = str(cur_sub["speaker_label"] or "unknown")
            target_name = str(cur_sub["speaker_name"] or "")
            target_speaker_locked = int(cur_sub["speaker_locked"] or 0)

            revert_text = ev_type in ("text", "both") or (ev_type == "revert" and ev["old_content"] != ev["new_content"])
            revert_speaker = ev_type in ("speaker", "both") or (
                ev_type == "revert"
                and (ev["old_speaker_label"] != ev["new_speaker_label"] or ev["old_speaker_name"] != ev["new_speaker_name"])
            )

            if revert_text:
                target_content = str(ev["old_content"])
                target_content_locked = 0
            if revert_speaker:
                target_label = str(ev["old_speaker_label"] or "unknown")
                target_name = str(ev["old_speaker_name"] or "")
                target_speaker_locked = 0

            try:
                if revert_text:
                    conn.execute(
                        """
                        UPDATE subtitles
                           SET content = ?,
                               content_fts = ?,
                               speaker_label = ?,
                               speaker_name = ?,
                               content_locked = ?,
                               speaker_locked = ?
                         WHERE id = ?
                        """,
                        (
                            target_content,
                            tokenize_for_fts(target_content),
                            target_label,
                            target_name,
                            target_content_locked,
                            target_speaker_locked,
                            sid,
                        ),
                    )
                else:
                    conn.execute(
                        """
                        UPDATE subtitles
                           SET speaker_label = ?,
                               speaker_name = ?,
                               speaker_locked = ?
                         WHERE id = ?
                        """,
                        (target_label, target_name, target_speaker_locked, sid),
                    )
            except sqlite3.IntegrityError as exc:
                raise ValueError("回滚后的字幕与同时间点另一条字幕冲突，无法自动回滚") from exc

            new_cur = conn.execute(
                """
                INSERT INTO subtitle_edits (
                    subtitle_id, bvid, page, start_sec, end_sec, edit_type,
                    old_content, new_content,
                    old_speaker_label, old_speaker_name,
                    new_speaker_label, new_speaker_name, note
                ) VALUES (?, ?, ?, ?, ?, 'revert', ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    sid,
                    str(cur_sub["bvid"]),
                    int(cur_sub["page"]),
                    float(cur_sub["start_sec"]),
                    float(cur_sub["end_sec"]),
                    str(cur_sub["content"]),
                    target_content,
                    str(cur_sub["speaker_label"] or "unknown"),
                    str(cur_sub["speaker_name"] or ""),
                    target_label,
                    target_name,
                    f"回滚校对记录 #{edit_id}",
                ),
            )
            conn.execute(
                """
                UPDATE videos
                   SET updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                 WHERE bvid = ? AND page = ?
                """,
                (str(cur_sub["bvid"]), int(cur_sub["page"])),
            )

            updated = conn.execute(
                """
                SELECT id, up_name, bvid, page, pub_date, start_sec, end_sec,
                       content, speaker_label, speaker_name, speaker_sim,
                       content_locked, speaker_locked
                  FROM subtitles
                 WHERE id = ?
                """,
                (sid,),
            ).fetchone()
            return {
                "status": "ok",
                "reverted_edit_id": int(edit_id),
                "new_edit_id": int(new_cur.lastrowid),
                "subtitle": dict(updated),
            }

    def list_review_candidates(
        self,
        *,
        mode: str = "uncertain",
        up_name: Optional[str] = None,
        bvid: Optional[str] = None,
        page: Optional[int] = None,
        sim_min: float = 0.40,
        sim_max: float = 0.56,
        start_sec: Optional[float] = None,
        end_sec: Optional[float] = None,
        include_locked: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """查询集中校对队列：支持声纹置信度灰色边界、含混音、高分无关音轨、按场次时间轴、已校对记录."""
        clauses: List[str] = ["s.is_visible = 1"]
        params: List[Any] = []
        eff_mode = (mode or "uncertain").strip().lower()

        if up_name:
            clauses.append("s.up_name = ?")
            params.append(up_name.strip())
        if bvid:
            clauses.append("s.bvid = ?")
            params.append(bvid.strip())
        if page is not None:
            clauses.append("s.page = ?")
            params.append(int(page))
        if start_sec is not None:
            clauses.append("s.end_sec >= ?")
            params.append(float(start_sec))
        if end_sec is not None:
            clauses.append("s.start_sec <= ?")
            params.append(float(end_sec))

        if eff_mode == "uncertain":
            clauses.append(
                "(s.speaker_label = 'mixed' OR (s.speaker_sim IS NOT NULL AND s.speaker_sim >= ? AND s.speaker_sim <= ?))"
            )
            params.extend([float(sim_min), float(sim_max)])
            if not include_locked:
                clauses.append("s.speaker_locked = 0")
            order_sql = "ORDER BY ABS(COALESCE(s.speaker_sim, 0.50) - 0.50) ASC, s.pub_date DESC, s.start_sec ASC"
        elif eff_mode == "mixed":
            clauses.append("s.speaker_label = 'mixed'")
            if not include_locked:
                clauses.append("s.speaker_locked = 0")
            order_sql = "ORDER BY s.pub_date DESC, s.bvid DESC, s.page ASC, s.start_sec ASC"
        elif eff_mode == "external":
            clauses.append("s.speaker_label = 'external'")
            if not include_locked:
                clauses.append("s.speaker_locked = 0")
            order_sql = "ORDER BY COALESCE(s.speaker_sim, -1.0) DESC, s.pub_date DESC, s.start_sec ASC"
        elif eff_mode == "locked":
            clauses.append("(s.speaker_locked = 1 OR s.content_locked = 1)")
            order_sql = "ORDER BY s.pub_date DESC, s.bvid DESC, s.page ASC, s.start_sec ASC"
        elif eff_mode == "timeline":
            if not bvid:
                raise ValueError("按时间轴顺序校对模式必须指定 BV 号")
            order_sql = "ORDER BY s.page ASC, s.start_sec ASC, s.id ASC"
        else:
            raise ValueError(f"不支持的审校模式: {mode!r}")

        where_sql = f"WHERE {' AND '.join(clauses)}"
        lim = max(1, min(200, int(limit)))
        off = max(0, int(offset))

        with self.session() as conn:
            total = int(
                conn.execute(f"SELECT COUNT(*) FROM subtitles s {where_sql}", params).fetchone()[0]
            )
            rows = conn.execute(
                f"""
                SELECT s.id, s.up_name, s.bvid, s.page, s.pub_date,
                       s.start_sec, s.end_sec, s.content, s.source,
                       s.speaker_label, s.speaker_name, s.speaker_sim,
                       s.content_locked, s.speaker_locked,
                       COALESCE(v.title, '') AS title,
                       COALESCE(v.part_title, '') AS part_title
                  FROM subtitles s
                  LEFT JOIN videos v ON v.bvid = s.bvid AND v.page = s.page
                {where_sql}
                {order_sql}
                 LIMIT ? OFFSET ?
                """,
                [*params, lim, off],
            ).fetchall()

        return {
            "mode": eff_mode,
            "total": total,
            "limit": lim,
            "offset": off,
            "items": [dict(r) for r in rows],
        }

    def rebuild_fts(self) -> None:
        """如手动修改过字幕或迁移导入数据，可一键重建 FTS5 索引."""
        with self.session() as conn:
            conn.execute("INSERT INTO subtitles_fts(subtitles_fts) VALUES('rebuild');")
