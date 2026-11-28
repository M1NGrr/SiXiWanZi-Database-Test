"""端到端自动化测试：建库、清洗、增量去重、FTS5 中文检索、场景初筛、切片规划与 FastAPI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
from fastapi.testclient import TestClient

from sixiwanzi_corpus.api import create_app
from sixiwanzi_corpus.asr_backfill import (
    _is_whisper_hallucination,
    _sanitize_asr_text,
    _split_text_proportionally,
    list_no_subtitle_targets,
    resplit_legacy_asr,
)
from sixiwanzi_corpus.bilibili_client import SubtitleTrack, VideoInfo, VideoPageInfo
from sixiwanzi_corpus.cleaner import clean_bcc_payload, clean_srt_text
from sixiwanzi_corpus.clip_planner import plan_clip_job
from sixiwanzi_corpus.config import load_config
from sixiwanzi_corpus.db import CorpusDB
from sixiwanzi_corpus.ingest import ingest_bvids, ingest_local_file, seed_demo_data
from sixiwanzi_corpus.normalizer import (
    apply_domain_corrections,
    build_fuzzy_fts_query,
    run_corpus_normalization,
)
from sixiwanzi_corpus.search import SearchQuery, SearchService
from sixiwanzi_corpus.wbi import extract_wbi_keys, sign_wbi_params


def test_wbi_signing_deterministic() -> None:
    nav = {
        "wbi_img": {
            "img_url": "https://i0.hdslb.com/bfs/wbi/7cd084941338484aae1ad9425b84077c.png",
            "sub_url": "https://i0.hdslb.com/bfs/wbi/4932caff0ff746eab6f01bf08b70ac45.png",
        }
    }
    img_key, sub_key = extract_wbi_keys(nav)
    assert img_key == "7cd084941338484aae1ad9425b84077c"
    assert sub_key == "4932caff0ff746eab6f01bf08b70ac45"
    signed = sign_wbi_params(
        {"bvid": "BV1xx411c7mD", "cid": 123456, "note": "a!b'c(d)e*"},
        img_key,
        sub_key,
        timestamp=1700000000,
    )
    assert signed["wts"] == "1700000000"
    assert signed["note"] == "abcde"
    assert len(signed["w_rid"]) == 32


def test_cleaner_bcc_and_srt() -> None:
    raw_bcc = {
        "body": [
            {"from": 10.126, "to": 12.5, "content": "  嘉然\u200b今天  吃什么 &amp; 唱歌  "},
            {"from": 10.13, "to": 12.5, "content": "嘉然今天 吃什么 & 唱歌"},  # 完全重复去重
            {"from": 13.0, "to": 14.0, "content": "[音乐]"},
            {"from": 15.0, "to": 14.0, "content": "..."},  # 纯标点噪声
            {"from": -2.0, "to": -1.0, "content": "负时间校正"},
        ]
    }
    cleaned = clean_bcc_payload(raw_bcc)
    assert len(cleaned) == 2
    assert cleaned[0].start_sec == 0.0
    assert cleaned[0].end_sec == 0.0
    assert cleaned[0].content == "负时间校正"
    assert cleaned[1].start_sec == 10.13
    assert cleaned[1].content == "嘉然今天 吃什么 & 唱歌"

    # 校验繁转简归一化
    trad = clean_bcc_payload({"body": [{"from": 1.0, "to": 2.0, "content": "元宵節快樂，看我手語！"}]})
    assert trad[0].content == "元宵节快乐，看我手语！"

    srt = (
        "1\n00:01:02,500 --> 00:01:05,250\n你好，枝江！\n\n"
        "2\n00:01:06,000 --> 00:01:08,000\n（掌声）\n"
    )
    srt_items = clean_srt_text(srt)
    assert len(srt_items) == 1
    assert srt_items[0].start_sec == 62.5
    assert srt_items[0].end_sec == 65.25
    assert srt_items[0].content == "你好，枝江！"


def test_full_pipeline_search_tagging_dedup_and_clips(tmp_path: Path) -> None:
    db_file = tmp_path / "test_corpus.db"
    cfg = load_config(db_path=db_file)
    db = CorpusDB(db_file)

    # 1. 首次写入 demo 数据 + 重复写入验证去重
    first = seed_demo_data(db)
    assert first["videos"] == 4
    assert first["lines_inserted"] >= 20
    assert first["lines_duplicate"] == 0

    second = seed_demo_data(db)
    assert second["lines_inserted"] == 0
    assert second["lines_duplicate"] == first["lines_inserted"]

    # 2. 中文 1字 / 2字 / 长短语 / 英文 / 多关键词 / 过滤 / 分页 / 上下文检索
    svc = SearchService(db)

    # 2字热词 + 跨 UP 命中
    r_heihua = svc.search(SearchQuery(q="黑话"))
    assert r_heihua["pagination"]["total"] == 4

    # 3字中文精确检索 + UP 过滤
    r_xiu = svc.search(SearchQuery(q="羞死了", up_names=["梨安"]))
    assert r_xiu["pagination"]["total"] == 1
    assert r_xiu["items"][0]["bvid"] == "BV1DEMO000004"

    # 1字中文检索
    r_one = svc.search(SearchQuery(q="笑", limit=2, offset=0))
    assert r_one["pagination"]["total"] >= 3
    assert len(r_one["items"]) == 2
    assert r_one["pagination"]["has_more"] is True

    # 英文混排检索
    r_en = svc.search(SearchQuery(q="MUMU", fuzzy=False))
    assert r_en["pagination"]["total"] == 1
    assert r_en["items"][0]["up_name"] == "恬豆"

    # 多词 AND + 日期过滤 + 上下文
    r_multi = svc.search(
        SearchQuery(
            q="卧底 大梨老师",
            date_from="2025-08-01",
            date_to="2025-08-31",
            context_lines=1,
        )
    )
    assert r_multi["pagination"]["total"] == 1
    hit = r_multi["items"][0]
    assert len(hit["context_before"]) == 1
    assert len(hit["context_after"]) == 1

    # 软隐藏 is_visible = 0 默认不可见，include_hidden=True 可见
    db.update_visibility("BV1DEMO000004", 1, 0)
    assert svc.search(SearchQuery(q="羞死了", up_names=["梨安"]))["pagination"]["total"] == 0
    assert (
        svc.search(SearchQuery(q="羞死了", up_names=["梨安"], include_hidden=True))[
            "pagination"
        ]["total"]
        == 1
    )

    # 5. FFmpeg 切片清单规划（相邻两句自动合并为一段）
    win = svc.get_subtitle_window("BV1DEMO000001", 1)
    first_two_ids = [win["items"][0]["id"], win["items"][1]["id"]]
    manifest = plan_clip_job(
        db,
        subtitle_ids=first_two_ids,
        pad_before_sec=1.0,
        pad_after_sec=1.0,
        merge_gap_sec=3.0,
    )
    assert manifest["segment_count"] == 1
    assert manifest["segments"][0]["subtitle_ids"] == first_two_ids
    assert manifest["segments"][0]["ffmpeg_cut_cmd"][0] == "ffmpeg"


def test_local_import_and_mocked_bvid_incremental_ingest(tmp_path: Path) -> None:
    db_file = tmp_path / "ingest.db"
    cfg = load_config(db_path=db_file)
    db = CorpusDB(db_file)

    # 本地 BCC JSON 导入
    bcc_path = tmp_path / "sample_bcc.json"
    bcc_path.write_text(
        json.dumps(
            {
                "body": [
                    {"from": 1.0, "to": 3.5, "content": "欢迎来到测试直播回放"},
                    {"from": 4.0, "to": 6.5, "content": "今天状态很不错"},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    ins, dup = ingest_local_file(
        db,
        bcc_path,
        up_name="沐霂",
        bvid="BV1LOCAL00001",
        page=1,
        pub_date="2025-02-01",
        title="本地测试回放",
    )
    assert (ins, dup) == (2, 0)

    # 模拟 B站客户端抓取多 P 视频并验证增量跳过
    class FakeBiliClient:
        has_sessdata = True

        def get_video_info(self, bvid: str) -> VideoInfo:
            return VideoInfo(
                bvid=bvid,
                aid=99999,
                title="【直播回放】模拟抓取测试",
                desc="测试简介",
                up_uid=1878154667,
                raw_up_name="沐霂是MUMU呀",
                pub_ts=1739500000,
                pub_date="2025-02-14",
                duration_sec=3600,
                pages=[
                    VideoPageInfo(page=1, cid=111, part_title="P1 上半场", duration_sec=1800),
                    VideoPageInfo(page=2, cid=222, part_title="P2 下半场", duration_sec=1800),
                ],
            )

        def list_subtitle_tracks(self, bvid: str, cid: int) -> List[SubtitleTrack]:
            if cid == 111:
                return [
                    SubtitleTrack(
                        lan="ai-zh",
                        lan_doc="中文（自动生成）",
                        subtitle_url="https://example.com/p1.json",
                        ai_type=1,
                    )
                ]
            return []  # P2 模拟无字幕

        @staticmethod
        def pick_subtitle_track(tracks: Any, prefer_lans: Any = None) -> Any:
            return tracks[0] if tracks else None

        def fetch_subtitle_json(self, subtitle_url: str) -> Dict[str, Any]:
            return {
                "body": [
                    {"from": 10.0, "to": 13.0, "content": "情人节快乐嘉心糖！"},
                    {"from": 14.0, "to": 17.5, "content": "我们一起唱情歌吧！"},
                ]
            }

    fake = FakeBiliClient()
    s1 = ingest_bvids(db, cfg, ["BV1MOCK000001"], client=fake)  # type: ignore[arg-type]
    assert s1.parts_ok == 1
    assert s1.parts_no_subtitle == 1
    assert s1.lines_inserted == 2

    # 再次增量运行：整部 BV 已被 is_bvid_completed 秒级预判跳过，不重复发任何网络请求
    s2 = ingest_bvids(db, cfg, ["BV1MOCK000001"], client=fake)  # type: ignore[arg-type]
    assert s2.parts_skipped == 1
    assert s2.lines_inserted == 0

    # 若无字幕分 P 已被 Whisper 判定为 asr_empty，后续增量抓取不得将其重置为 no_subtitle 或重复转录
    with db.session() as conn:
        conn.execute(
            "UPDATE videos SET subtitle_status = 'asr_empty' WHERE bvid = 'BV1MOCK000001' AND page = 2"
        )
    assert db.is_bvid_completed("BV1MOCK000001", allow_no_subtitle=True) is True
    assert db.is_bvid_completed("BV1MOCK000001", allow_no_subtitle=False) is True
    s3 = ingest_bvids(db, cfg, ["BV1MOCK000001"], client=fake)  # type: ignore[arg-type]
    assert s3.parts_skipped == 1
    assert db.get_video("BV1MOCK000001", 2)["subtitle_status"] == "asr_empty"

    assert db.optimize_db()["status"] == "ok"


def test_fastapi_endpoints_public_and_admin_separation(tmp_path: Path) -> None:
    db_file = tmp_path / "api.db"
    db = CorpusDB(db_file)
    seed_demo_data(db)

    # 1. 云端公网只读模式（enable_admin=False）：不暴露任何 admin 接口或调试页
    pub_app = create_app(db_path=db_file, enable_admin=False)
    pub_client = TestClient(pub_app)
    assert pub_client.get("/api/v1/health").status_code == 200
    assert pub_client.get("/api/v1/members").status_code == 200
    assert pub_client.get("/debug").status_code == 404
    assert pub_client.get("/api/v1/admin/auth/status").status_code == 404
    assert pub_client.patch(
        "/api/v1/admin/videos/BV1DEMO000001/visibility",
        json={"page": 1, "is_visible": 0},
    ).status_code == 404

    st = pub_client.get("/api/v1/stats").json()
    assert "db_path" not in st  # 公网统计不暴露服务器本地磁盘路径
    assert st["video_parts_total"] == 4
    assert st["subtitles_total"] >= 20
    assert st["latest_video"] is not None
    assert st["latest_video"]["bvid"].startswith("BV1")
    assert st["latest_video"]["video_url"].startswith("https://www.bilibili.com/video/")
    assert isinstance(st["announcements"], list)
    ann_res = pub_client.get("/api/v1/announcements")
    assert ann_res.status_code == 200
    assert any("四禧丸子" in item["content"] for item in ann_res.json()["items"])

    sr = pub_client.get("/api/v1/search", params={"q": "黑话", "up_name": "沐霂,梨安"}).json()
    assert sr["pagination"]["total"] == 2
    # 校验规范化的 B 站空降 URL（带尾部斜杠、单 P 不带冗余 p=1）
    first_item = sr["items"][0]
    assert first_item["jump_url"].startswith(f"https://www.bilibili.com/video/{first_item['bvid']}/?t=")
    assert first_item["video_url"] == f"https://www.bilibili.com/video/{first_item['bvid']}/"

    # 2. 本地运维模式（enable_admin=True）：允许调用 /api/v1/admin/* 与 /debug
    loc_app = create_app(db_path=db_file, enable_admin=True)
    loc_client = TestClient(loc_app)
    assert loc_client.get("/debug").status_code == 200

    patch = loc_client.patch(
        "/api/v1/admin/videos/BV1DEMO000001/visibility",
        json={"page": 1, "is_visible": 0},
    )
    assert patch.status_code == 200

    sr2 = loc_client.get("/api/v1/search", params={"q": "黑话", "up_name": "沐霂"}).json()
    assert sr2["pagination"]["total"] == 0
    loc_client.patch(
        "/api/v1/admin/videos/BV1DEMO000001/visibility",
        json={"page": 1, "is_visible": 1},
    )

    hit_id = sr["items"][0]["id"]
    plan = loc_client.post("/api/v1/admin/clips/plan", json={"subtitle_ids": [hit_id]}).json()
    assert plan["segment_count"] == 1
    assert plan["segments"][0]["subtitle_ids"] == [hit_id]


def test_asr_backfill_helpers(tmp_path: Path) -> None:
    db_file = tmp_path / "asr_helper.db"
    db = CorpusDB(db_file)
    db.init_db()
    from sixiwanzi_corpus.db import VideoRecord

    # 写入一个 3 秒碎片和一个 600 秒正常无字幕分 P
    db.save_video_and_subtitles(
        VideoRecord(
            bvid="BV1SHORT0001",
            page=1,
            cid=1,
            up_name="心宜",
            pub_date="2026-01-01",
            duration_sec=3,
            subtitle_status="no_subtitle",
        ),
        [],
    )
    db.save_video_and_subtitles(
        VideoRecord(
            bvid="BV1LONG00001",
            page=1,
            cid=2,
            up_name="心宜",
            pub_date="2026-01-02",
            duration_sec=600,
            subtitle_status="no_subtitle",
        ),
        [],
    )
    targets = list_no_subtitle_targets(db, min_duration=15, limit=10)
    assert [t["bvid"] for t in targets] == ["BV1LONG00001"]
    assert _is_whisper_hallucination("以下是B站虚拟偶像嘉然") is True
    assert _is_whisper_hallucination("今天晚上吃什么呀") is False

    # 校验 ASR 异常标点清洗与多级断句切分
    assert _sanitize_asr_text("我们去玩吧﹐然后回家﹒") == "我们去玩吧，然后回家。"
    long_text = (
        "今天晚上大家一起来枝江看直播吧，我觉得这个新衣服真的特别好看，"
        "而且背景音乐也非常带感，你们觉得呢？如果喜欢的话记得点个关注呀！"
    )
    pieces = _split_text_proportionally(10.0, 40.0, long_text, max_chars=22)
    assert len(pieces) >= 4
    assert all(len(text) <= 28 for _, _, text in pieces)
    assert pieces[0][0] == pytest.approx(10.0, abs=0.05)
    assert pieces[-1][1] == pytest.approx(40.0, abs=0.05)
    for i in range(len(pieces) - 1):
        assert pieces[i][0] <= pieces[i][1] <= pieces[i + 1][0] + 0.01

    # 无标点超长连续口语也必须强制软切分
    unpunct = "其实我今天本来是想早点起床准备直播的但是闹钟响了之后我又赖床了半个小时结果差点迟到"
    pieces_u = _split_text_proportionally(0.0, 20.0, unpunct, max_chars=22)
    assert len(pieces_u) >= 2
    assert all(len(text) <= 28 for _, _, text in pieces_u)

    # 校验 resplit_legacy_asr 仅重切旧版 ASR 且保留 asr-large-v3-turbo 标记供 GPU 重录识别
    from sixiwanzi_corpus.db import SubtitleRow

    db.save_video_and_subtitles(
        VideoRecord(
            bvid="BV1LEGACY001",
            page=1,
            cid=3,
            up_name="嘉然",
            pub_date="2026-01-03",
            duration_sec=120,
            subtitle_status="ok",
            subtitle_lan="asr-large-v3-turbo",
        ),
        [
            SubtitleRow(
                up_name="嘉然",
                bvid="BV1LEGACY001",
                page=1,
                pub_date="2026-01-03",
                start_sec=0.0,
                end_sec=30.0,
                content=long_text,
                source="asr-large-v3-turbo",
            )
        ],
    )
    db.save_video_and_subtitles(
        VideoRecord(
            bvid="BV1NEWASR001",
            page=1,
            cid=4,
            up_name="嘉然",
            pub_date="2026-01-04",
            duration_sec=120,
            subtitle_status="ok",
            subtitle_lan="asr-v2-large-v3-turbo",
        ),
        [
            SubtitleRow(
                up_name="嘉然",
                bvid="BV1NEWASR001",
                page=1,
                pub_date="2026-01-04",
                start_sec=0.0,
                end_sec=3.0,
                content="短句无需重切",
                source="asr-v2-large-v3-turbo",
            )
        ],
    )
    db.save_video_and_subtitles(
        VideoRecord(
            bvid="BV1OFFICIAL1",
            page=1,
            cid=5,
            up_name="嘉然",
            pub_date="2026-01-05",
            duration_sec=120,
            subtitle_status="ok",
            subtitle_lan="ai-zh",
        ),
        [
            SubtitleRow(
                up_name="嘉然",
                bvid="BV1OFFICIAL1",
                page=1,
                pub_date="2026-01-05",
                start_sec=0.0,
                end_sec=3.0,
                content="官方字幕不动",
                source="ai-zh",
            )
        ],
    )

    res = resplit_legacy_asr(db, max_chars=22, dry_run=False)
    assert res["parts_modified"] == 1
    assert res["rows_after"] > res["rows_before"]

    with db.connect() as conn:
        vrow = conn.execute(
            "SELECT subtitle_lan, line_count FROM videos WHERE bvid='BV1LEGACY001' AND page=1"
        ).fetchone()
        assert vrow["subtitle_lan"] == "asr-large-v3-turbo"
        assert vrow["line_count"] >= 4

    # 校验 --retranscribe-legacy-asr 目标筛选只选中旧版 asr-large-v3-turbo，跳过 v2 与官方字幕
    legacy_targets = list_no_subtitle_targets(db, min_duration=15, limit=10, retranscribe_legacy_asr=True)
    assert [t["bvid"] for t in legacy_targets] == ["BV1LEGACY001"]


def test_domain_normalization_and_fuzzy_search(tmp_path: Path) -> None:
    # 1. Tier 1 & Tier 2 纠错与保护校验
    assert apply_domain_corrections("欢迎思西丸子、沐沐、田豆、黎安、苏苏又、甜豆包、向心离、肥狸、佑神和王幼仪！") == (
        "欢迎四禧丸子、沐霂、恬豆、梨安、酥酥又、恬豆包、向心梨、肥梨、又神和王又一！"
    )
    # 排除 左右神经，替换 右神 -> 又神；保留 甜豆 / 李安 / 沐霖 / 小目标 / 向心力 等日常语境
    assert apply_domain_corrections("左右神经没问题，右神厉害，先定个小目标，吃甜豆，看李安电影，再看李安不迷路直播！") == (
        "左右神经没问题，又神厉害，先定个小目标，吃甜豆，看李安电影，再看梨安不迷路直播！"
    )

    # 2. 同音模糊查询表达式校验
    expr, expanded = build_fuzzy_fts_query("四禧丸子", fuzzy=True)
    assert '"四 禧 丸 子"' in expr and '"四 喜 丸 子"' in expr
    assert "四禧丸子" in expanded

    # 4. 存量批量修复 + 模糊搜索端到端校验
    db_file = tmp_path / "norm.db"
    db = CorpusDB(db_file)
    db.init_db()
    from sixiwanzi_corpus.db import SubtitleRow, VideoRecord

    db.save_video_and_subtitles(
        VideoRecord(bvid="BV1NORM00001", page=1, up_name="沐霂", pub_date="2026-01-01", subtitle_status="ok"),
        [
            SubtitleRow(up_name="沐霂", bvid="BV1NORM00001", page=1, pub_date="2026-01-01", start_sec=1.0, end_sec=3.0, content="沐沐和田豆欢迎苏苏又和小目标"),
            SubtitleRow(up_name="沐霂", bvid="BV1NORM00001", page=1, pub_date="2026-01-01", start_sec=4.0, end_sec=6.0, content="今天聊聊四喜丸子的向心力"),
        ],
    )
    # 模拟历史遗留未清洗入库后执行批量修复
    res = run_corpus_normalization(db, dry_run=False)
    assert res["updated"] == 1

    svc = SearchService(db)
    # 搜 小沐标（模糊模式）能同时搜到保留原文的 小目标！
    r_fuzzy = svc.search(SearchQuery(q="小沐标", fuzzy=True))
    assert r_fuzzy["pagination"]["total"] == 1
    assert "小目标" in r_fuzzy["items"][0]["content"]
    # 搜 四禧丸子（模糊模式）能同时搜到 四喜丸子！
    r_group = svc.search(SearchQuery(q="四禧丸子", fuzzy=True))
    assert r_group["pagination"]["total"] == 1
    # 精确模式不扩展
    r_exact = svc.search(SearchQuery(q="小沐标", fuzzy=False))
    assert r_exact["pagination"]["total"] == 0
    # 搜 恬豆 能搜到被修复后的 田豆 -> 恬豆
    r_td = svc.search(SearchQuery(q="恬豆", fuzzy=False))
    assert r_td["pagination"]["total"] == 1


def test_incremental_delta_sync_and_live_stats(tmp_path: Path) -> None:
    from sixiwanzi_corpus.db import SubtitleRow, VideoRecord
    from sixiwanzi_corpus.sync_delta import apply_delta, build_manifest, export_delta

    local_db = CorpusDB(tmp_path / "local.db")
    cloud_db = CorpusDB(tmp_path / "cloud.db")
    local_db.init_db()
    cloud_db.init_db()

    # 初始共同数据：旧分 P（2026-11-20）
    v_old = VideoRecord(bvid="BV1SYNC00001", page=1, up_name="嘉然", pub_date="2026-11-20", subtitle_status="ok", scene_type="solo")
    s_old = [SubtitleRow(up_name="嘉然", bvid="BV1SYNC00001", page=1, pub_date="2026-11-20", start_sec=10.0, end_sec=12.0, content="旧直播玉米粑粑")]
    local_db.save_video_and_subtitles(v_old, s_old)
    cloud_db.save_video_and_subtitles(v_old, s_old)

    # 本地新入库 1 场最新直播（2026-11-28），云端尚未同步
    v_new = VideoRecord(bvid="BV1SYNC00002", page=1, up_name="思诺", pub_date="2026-11-28", subtitle_status="ok", scene_type="solo")
    s_new = [
        SubtitleRow(up_name="思诺", bvid="BV1SYNC00002", page=1, pub_date="2026-11-28", start_sec=5.0, end_sec=8.0, content="小海诺增量同步成功啦"),
        SubtitleRow(up_name="思诺", bvid="BV1SYNC00002", page=1, pub_date="2026-11-28", start_sec=9.0, end_sec=12.0, content="首页时间跨度也会自动更新"),
    ]
    local_db.save_video_and_subtitles(v_new, s_new)

    # 1. 拉取云端清单并仅导出增量补丁
    remote_manifest = build_manifest(cloud_db)
    delta_file = tmp_path / "delta.json.gz"
    exp = export_delta(local_db, remote_manifest, delta_file)
    assert exp["parts_new"] == 1
    assert exp["parts_changed"] == 1
    assert exp["subtitles_exported"] == 2

    # 2. 在云端公网只读接口保持挂载的同时热应用补丁
    pub_app = create_app(db_path=cloud_db.db_path, enable_admin=False)
    client = TestClient(pub_app)
    before_stats = client.get("/api/v1/stats").json()
    assert before_stats["date_max"] == "2026-11-20"
    assert before_stats["subtitles_total"] == 1

    res = apply_delta(cloud_db, delta_file)
    assert res["parts_applied"] == 1
    assert res["subtitles_inserted"] == 2

    # 3. 不重启服务，公网 /api/v1/stats 与 /api/v1/search 立刻生效
    after_stats = client.get("/api/v1/stats").json()
    assert after_stats["date_max"] == "2026-11-28"
    assert after_stats["subtitles_total"] == 3
    assert after_stats["video_parts_ok"] == 2

    sr = client.get("/api/v1/search", params={"q": "增量同步成功"}).json()
    assert sr["pagination"]["total"] == 1
    assert sr["items"][0]["bvid"] == "BV1SYNC00002"

    # 4. 再次导出增量应为 0 变动
    exp2 = export_delta(local_db, build_manifest(cloud_db), delta_file)
    assert exp2["parts_changed"] == 0


def test_voiceprint_classification_host_only_search_migration_and_sync(tmp_path: Path) -> None:
    import sqlite3
    from sixiwanzi_corpus.db import SubtitleRow, VideoRecord
    from sixiwanzi_corpus.sync_delta import apply_delta, build_manifest, export_delta
    from sixiwanzi_corpus.voiceprint import (
        DEFAULT_VOICEPRINTS_PATH,
        classify_utterance_scores,
        is_re_watch_title,
        load_voiceprints,
        parse_time_to_sec,
    )

    # 1. 校验内置 4 位成员声纹库与工具函数
    vp_cfg = load_voiceprints(DEFAULT_VOICEPRINTS_PATH)
    assert set(vp_cfg["members"].keys()) == {"又一", "沐霂", "梨安", "恬豆"}
    for member, info in vp_cfg["members"].items():
        assert len(info["embedding"]) == vp_cfg["dim"]
        assert info["clean_p50"] >= 0.65

    assert parse_time_to_sec("0:15:00") == 900.0
    assert parse_time_to_sec("04:30") == 270.0
    assert is_re_watch_title("【直播回放】Re:略略略略 2026年11月25日20点场") is True
    assert is_re_watch_title("【直播回放】一起看迪迦奥特曼！！") is True
    assert is_re_watch_title("【直播回放】做个雪花酥【做完了】") is False

    # 2. 校验四成员三段式分类规则（直判 host + 对应成员名 / 滑窗峰值 mixed / 邻接平滑 host / 边界 external 保持过滤）
    seq = [
        {"id": 1, "start_sec": 10.0, "end_sec": 14.0, "sim_mean": 0.16, "sim_peak": 0.19, "best_member": "又一"},
        {"id": 2, "start_sec": 14.4, "end_sec": 17.8, "sim_mean": 0.49, "sim_peak": 0.546, "best_member": "又一", "peak_member": "又一"},
        {"id": 3, "start_sec": 583.4, "end_sec": 584.5, "sim_mean": 0.58, "sim_peak": 0.58, "best_member": "又一"},
        {"id": 4, "start_sec": 584.5, "end_sec": 586.1, "sim_mean": 0.46, "sim_peak": 0.46, "best_member": "又一", "member_sims": {"又一": 0.46, "沐霂": 0.22}},
        {"id": 5, "start_sec": 586.1, "end_sec": 594.3, "sim_mean": 0.76, "sim_peak": 0.76, "best_member": "梨安"},
        {"id": 6, "start_sec": 739.2, "end_sec": 741.4, "sim_mean": 0.31, "sim_peak": 0.31, "best_member": "沐霂"},
        {"id": 7, "start_sec": 741.5, "end_sec": 744.1, "sim_mean": 0.491, "sim_peak": 0.491, "best_member": "又一"},
    ]
    classify_utterance_scores(seq)
    assert [x["speaker_label"] for x in seq] == [
        "external",
        "mixed",
        "host",
        "host",
        "host",
        "external",
        "external",
    ]
    assert [x["speaker_name"] for x in seq] == [
        "",
        "又一",
        "又一",
        "又一",
        "梨安",
        "",
        "",
    ]
    assert seq[3]["reason"] == "smooth"

    # 3. 校验旧库自动增量列迁移（模拟缺少 speaker_label / speaker_sim / voiceprint_status 的存量库）
    legacy_db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(legacy_db_path))
    conn.executescript(
        """
        CREATE TABLE videos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            bvid TEXT NOT NULL,
            page INTEGER NOT NULL DEFAULT 1,
            cid INTEGER,
            up_uid INTEGER,
            up_name TEXT NOT NULL,
            raw_up_name TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            part_title TEXT NOT NULL DEFAULT '',
            pub_date TEXT NOT NULL,
            pub_ts INTEGER NOT NULL DEFAULT 0,
            duration_sec INTEGER NOT NULL DEFAULT 0,
            subtitle_lan TEXT NOT NULL DEFAULT '',
            subtitle_status TEXT NOT NULL DEFAULT 'pending',
            line_count INTEGER NOT NULL DEFAULT 0,
            scene_type TEXT NOT NULL DEFAULT 'unknown',
            scene_confidence REAL NOT NULL DEFAULT 0.0,
            scene_reason TEXT NOT NULL DEFAULT '',
            scene_locked INTEGER NOT NULL DEFAULT 0,
            is_visible INTEGER NOT NULL DEFAULT 1,
            extra_meta TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            UNIQUE (bvid, page)
        );
        CREATE TABLE subtitles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            up_name TEXT NOT NULL,
            bvid TEXT NOT NULL,
            page INTEGER NOT NULL DEFAULT 1,
            pub_date TEXT NOT NULL,
            start_sec REAL NOT NULL,
            end_sec REAL NOT NULL,
            content TEXT NOT NULL,
            content_fts TEXT NOT NULL DEFAULT '',
            scene_type TEXT NOT NULL DEFAULT 'unknown',
            is_visible INTEGER NOT NULL DEFAULT 1,
            source TEXT NOT NULL DEFAULT 'bilibili_ai',
            created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
            UNIQUE (bvid, page, start_sec, end_sec, content)
        );
        """
    )
    conn.commit()
    conn.close()

    local_db = CorpusDB(legacy_db_path)
    local_db.init_db()  # 应自动补齐 3 个新列

    # 4. 写入 1 场已打分 Re 场次 + 1 场未打分普通场次，验证 host_only 检索与增量同步
    local_db.save_video_and_subtitles(
        VideoRecord(
            bvid="BV1RETEST0001",
            page=1,
            up_name="又一",
            title="【直播回放】Re:测试综艺",
            pub_date="2026-11-25",
            subtitle_status="ok",
        ),
        [
            SubtitleRow(up_name="又一", bvid="BV1RETEST0001", page=1, pub_date="2026-11-25", start_sec=1.0, end_sec=3.0, content="又一测试台词本人发言"),
            SubtitleRow(up_name="又一", bvid="BV1RETEST0001", page=1, pub_date="2026-11-25", start_sec=4.0, end_sec=7.0, content="又一测试台词半句插嘴"),
            SubtitleRow(up_name="又一", bvid="BV1RETEST0001", page=1, pub_date="2026-11-25", start_sec=8.0, end_sec=11.0, content="又一测试台词综艺原声"),
        ],
    )
    local_db.save_video_and_subtitles(
        VideoRecord(
            bvid="BV1SOLOTEST1",
            page=1,
            up_name="又一",
            title="【直播回放】普通杂谈",
            pub_date="2026-11-24",
            subtitle_status="ok",
        ),
        [
            SubtitleRow(up_name="又一", bvid="BV1SOLOTEST1", page=1, pub_date="2026-11-24", start_sec=1.0, end_sec=3.0, content="又一测试台词未打分单播"),
        ],
    )

    svc = SearchService(local_db)
    win = svc.get_subtitle_window("BV1RETEST0001", 1)
    ids = [x["id"] for x in win["items"]]
    local_db.update_part_voiceprint_scores(
        "BV1RETEST0001",
        1,
        [(ids[0], "host", 0.742, "又一"), (ids[1], "mixed", 0.546, "梨安"), (ids[2], "external", 0.182, "")],
        status="done",
    )

    pub_app = create_app(db_path=legacy_db_path, enable_admin=False)
    client = TestClient(pub_app)

    all_hits = client.get("/api/v1/search", params={"q": "又一测试台词", "host_only": "false"}).json()
    assert all_hits["pagination"]["total"] == 4

    host_hits = client.get("/api/v1/search", params={"q": "又一测试台词", "host_only": "true"}).json()
    # host_only 过滤掉 external，同时保留 host / mixed / 未打分 unknown
    assert host_hits["pagination"]["total"] == 3
    labels = {x["speaker_label"] for x in host_hits["items"]}
    assert labels == {"host", "mixed", "unknown"}

    # speaker_source 精细筛选：host / host_verified / external
    src_verified = client.get(
        "/api/v1/search", params={"q": "又一测试台词", "speaker_source": "host_verified"}
    ).json()
    assert src_verified["pagination"]["total"] == 2
    assert {x["speaker_label"] for x in src_verified["items"]} == {"host", "mixed"}
    assert {x["speaker_name"] for x in src_verified["items"]} == {"又一", "梨安"}

    src_ext = client.get(
        "/api/v1/search", params={"q": "又一测试台词", "speaker_source": "external"}
    ).json()
    assert src_ext["pagination"]["total"] == 1
    assert src_ext["items"][0]["speaker_label"] == "external"

    # 5. 验证 sync_delta 同步 voiceprint_status 与 speaker_label / speaker_sim
    cloud_db = CorpusDB(tmp_path / "cloud_vp.db")
    cloud_db.init_db()
    delta_file = tmp_path / "vp_delta.json.gz"
    exp = export_delta(local_db, build_manifest(cloud_db), delta_file)
    assert exp["parts_new"] == 2
    apply_delta(cloud_db, delta_file)

    cloud_svc = SearchService(cloud_db)
    c_host = cloud_svc.search(SearchQuery(q="又一测试台词", host_only=True))
    assert c_host["pagination"]["total"] == 3
    c_all = cloud_svc.search(SearchQuery(q="又一测试台词", host_only=False))
    ext_items = [x for x in c_all["items"] if x["speaker_label"] == "external"]
    assert len(ext_items) == 1
    assert ext_items[0]["speaker_sim"] == pytest.approx(0.182, abs=1e-4)
    assert ext_items[0]["speaker_name"] == ""
    assert {x["speaker_name"] for x in c_all["items"] if x["speaker_label"] in ("host", "mixed")} == {
        "又一",
        "梨安",
    }
    assert cloud_db.get_video("BV1RETEST0001", 1)["voiceprint_status"] == "done"


def test_manual_correction_locks_batch_review_revert_sync_and_public_isolation(tmp_path: Path) -> None:
    from sixiwanzi_corpus.db import SubtitleRow, VideoRecord
    from sixiwanzi_corpus.normalizer import append_literal_replace_rule, preview_literal_replace_rule
    from sixiwanzi_corpus.sync_delta import apply_delta, build_manifest, export_delta

    db_file = tmp_path / "manual_edit.db"
    db = CorpusDB(db_file)
    db.init_db()

    db.save_video_and_subtitles(
        VideoRecord(
            bvid="BV1EDIT00001",
            page=1,
            up_name="恬豆",
            title="【直播回放】四人团播测试",
            pub_date="2026-11-26",
            subtitle_status="ok",
        ),
        [
            SubtitleRow(
                up_name="恬豆",
                bvid="BV1EDIT00001",
                page=1,
                pub_date="2026-11-26",
                start_sec=10.0,
                end_sec=13.0,
                content="今天田豆包和大家一起玩游戏",
            ),
            SubtitleRow(
                up_name="恬豆",
                bvid="BV1EDIT00001",
                page=1,
                pub_date="2026-11-26",
                start_sec=14.0,
                end_sec=17.0,
                content="这句其实是梨安在说话",
            ),
            SubtitleRow(
                up_name="恬豆",
                bvid="BV1EDIT00001",
                page=1,
                pub_date="2026-11-26",
                start_sec=18.0,
                end_sec=21.0,
                content="这句也是电影原声台词",
            ),
        ],
    )

    svc = SearchService(db)
    win = svc.get_subtitle_window("BV1EDIT00001", 1)
    s0, s1, s2 = [x["id"] for x in win["items"]]
    db.update_part_voiceprint_scores(
        "BV1EDIT00001",
        1,
        [
            (s0, "host", 0.72, "恬豆"),
            (s1, "mixed", 0.51, "恬豆"),
            (s2, "host", 0.48, "恬豆"),
        ],
        status="done",
    )

    # 1. 公网只读模式：/admin 及所有人工纠错写接口必须严格返回 404
    pub_client = TestClient(create_app(db_path=db_file, enable_admin=False))
    assert pub_client.get("/admin").status_code == 404
    assert pub_client.patch(f"/api/v1/admin/subtitles/{s0}", json={"content": "非法写入"}).status_code == 404
    assert pub_client.post(
        "/api/v1/admin/subtitles/batch-speaker",
        json={"subtitle_ids": [s1], "speaker_label": "external"},
    ).status_code == 404
    assert pub_client.get("/api/v1/admin/review/candidates").status_code == 404
    assert pub_client.get("/api/v1/admin/edits").status_code == 404

    # 2. 本地后台模式：/admin 可访问，单条修正字幕正文与声音来源
    loc_client = TestClient(create_app(db_path=db_file, enable_admin=True))
    assert loc_client.get("/admin").status_code == 200

    # 边界校对队列应优先命中最接近 0.50 的 s1 (mixed, 0.51) 与 s2 (0.48)
    rv = loc_client.get("/api/v1/admin/review/candidates", params={"mode": "uncertain"}).json()
    assert rv["total"] == 2
    assert {x["id"] for x in rv["items"]} == {s1, s2}

    # 修改 s0 字幕正文（故意改成一个容易被规则命中的词，校验后续 normalize 不会覆盖人工锁定行）
    r_edit0 = loc_client.patch(
        f"/api/v1/admin/subtitles/{s0}",
        json={"content": "今天田豆包精校保留测试", "note": "人工精校正文"},
    )
    assert r_edit0.status_code == 200
    d_edit0 = r_edit0.json()
    assert d_edit0["changed"] is True
    assert d_edit0["subtitle"]["content_locked"] == 1
    assert d_edit0["subtitle"]["speaker_locked"] == 0

    # FTS5 触发器应已同步，立刻能搜到修订后的新词，且搜不到旧词
    assert loc_client.get("/api/v1/search", params={"q": "精校保留测试", "fuzzy": "false"}).json()["pagination"]["total"] == 1
    assert loc_client.get("/api/v1/search", params={"q": "一起玩游戏", "fuzzy": "false"}).json()["pagination"]["total"] == 0

    # 批量修改 s1 -> 梨安原声，s2 -> external
    r_edit1 = loc_client.patch(
        f"/api/v1/admin/subtitles/{s1}",
        json={"speaker_label": "host", "speaker_name": "梨安", "note": "团播实际是梨安"},
    )
    assert r_edit1.status_code == 200
    assert r_edit1.json()["subtitle"]["speaker_locked"] == 1
    assert r_edit1.json()["subtitle"]["speaker_name"] == "梨安"

    r_batch = loc_client.post(
        "/api/v1/admin/subtitles/batch-speaker",
        json={"subtitle_ids": [s2], "speaker_label": "external", "note": "电影台词"},
    )
    assert r_batch.status_code == 200
    assert r_batch.json()["updated"] == 1

    # 3. 自动化保护锁校验：
    # (a) 重跑声纹打分默认不得覆盖 s1 / s2 的人工锁定来源
    db.update_part_voiceprint_scores(
        "BV1EDIT00001",
        1,
        [
            (s0, "host", 0.75, "恬豆"),
            (s1, "external", 0.12, ""),
            (s2, "host", 0.88, "恬豆"),
        ],
        status="done",
    )
    win_after_vp = {x["id"]: x for x in svc.get_subtitle_window("BV1EDIT00001", 1)["items"]}
    assert win_after_vp[s1]["speaker_label"] == "host"
    assert win_after_vp[s1]["speaker_name"] == "梨安"
    assert win_after_vp[s2]["speaker_label"] == "external"

    # (b) 批量词表清洗不得覆盖 s0 的人工锁定正文（田豆包在默认规则里本来会被改成恬豆包）
    run_corpus_normalization(db, dry_run=False)
    win_after_norm = {x["id"]: x for x in svc.get_subtitle_window("BV1EDIT00001", 1)["items"]}
    assert win_after_norm[s0]["content"] == "今天田豆包精校保留测试"

    # 4. 增量同步保留人工锁定状态与修正结果
    cloud_db = CorpusDB(tmp_path / "cloud_edit.db")
    cloud_db.init_db()
    delta_file = tmp_path / "edit_delta.json.gz"
    exp = export_delta(local_db := db, build_manifest(cloud_db), delta_file)
    assert exp["parts_changed"] == 1
    apply_delta(cloud_db, delta_file)
    cloud_win = {x["id"]: x for x in SearchService(cloud_db).get_subtitle_window("BV1EDIT00001", 1)["items"]}
    by_start = {round(x["start_sec"], 1): x for x in cloud_win.values()}
    assert by_start[10.0]["content"] == "今天田豆包精校保留测试"
    assert by_start[10.0]["content_locked"] == 1
    assert by_start[14.0]["speaker_name"] == "梨安"
    assert by_start[14.0]["speaker_locked"] == 1
    assert by_start[18.0]["speaker_label"] == "external"

    # 5. 审计日志查询与一键回滚
    edits = loc_client.get("/api/v1/admin/edits", params={"bvid": "BV1EDIT00001"}).json()
    assert edits["total"] == 3
    edit1_id = r_edit1.json()["edit_id"]
    rev = loc_client.post(f"/api/v1/admin/edits/{edit1_id}/revert")
    assert rev.status_code == 200
    assert rev.json()["subtitle"]["speaker_label"] == "mixed"
    assert rev.json()["subtitle"]["speaker_locked"] == 0

    # 6. 全局规则预览与写入测试（使用临时 rules 文件避免污染仓库配置）
    tmp_rules = tmp_path / "rules.json"
    prev = preview_literal_replace_rule(db, "电影原声", "影视原声")
    assert prev["matched_count"] == 1
    add_res = append_literal_replace_rule("电影原声", "影视原声", rules_path=tmp_rules)
    assert add_res["status"] == "added"
    norm2 = run_corpus_normalization(db, dry_run=False, rules_path=tmp_rules)
    assert norm2["updated"] == 1
    assert SearchService(db).search(SearchQuery(q="影视原声", fuzzy=False))["pagination"]["total"] == 1
