"""预留扩展模块：FFmpeg 切片/合并任务清单规划器（不预先囤积视频本体）.

二阶段接入视频剪辑工作流时：
  1. 前端/用户从检索结果勾选若干 subtitle_id（或自定义起止秒数）；
  2. 调用本模块生成标准化的 ClipJobManifest（自动前后留白 + 合并相邻重叠片段）；
  3. 异步 Worker 按需用 yt-dlp / BBDown 仅拉取目标区间或临时缓存目标分 P，再执行 FFmpeg 切片与 concat 合并。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .db import CorpusDB
from .search import build_bilibili_jump_url, format_sec_hms


@dataclass
class RawClipRequest:
    bvid: str
    page: int
    start_sec: float
    end_sec: float
    up_name: str = ""
    pub_date: str = ""
    scene_type: str = "unknown"
    text_preview: str = ""
    subtitle_ids: List[int] = field(default_factory=list)


def plan_clip_job(
    db: CorpusDB,
    *,
    subtitle_ids: Optional[Sequence[int]] = None,
    custom_segments: Optional[Sequence[Dict[str, Any]]] = None,
    pad_before_sec: float = 1.5,
    pad_after_sec: float = 1.5,
    merge_gap_sec: float = 3.0,
    output_name: str = "sixiwanzi_compilation.mp4",
) -> Dict[str, Any]:
    """根据命中的字幕 ID 或自定义区间，生成可交由 FFmpeg Worker 执行的切片/合并清单."""
    raw_items: List[RawClipRequest] = []

    if subtitle_ids:
        id_list = [int(x) for x in subtitle_ids]
        placeholders = ",".join("?" for _ in id_list)
        with db.session() as conn:
            rows = conn.execute(
                f"""
                SELECT s.id, s.up_name, s.bvid, s.page, s.pub_date,
                       s.start_sec, s.end_sec, s.content, s.scene_type,
                       COALESCE(v.duration_sec, 0) AS video_duration_sec
                  FROM subtitles s
                  LEFT JOIN videos v ON v.bvid = s.bvid AND v.page = s.page
                 WHERE s.id IN ({placeholders})
                """,
                id_list,
            ).fetchall()
        by_id = {int(r["id"]): dict(r) for r in rows}
        for sid in id_list:
            r = by_id.get(sid)
            if not r:
                continue
            s_sec = round(max(0.0, float(r["start_sec"]) - pad_before_sec), 2)
            e_sec = round(float(r["end_sec"]) + pad_after_sec, 2)
            dur = int(r.get("video_duration_sec") or 0)
            if dur > 0:
                e_sec = min(float(dur), e_sec)
            raw_items.append(
                RawClipRequest(
                    bvid=r["bvid"],
                    page=int(r["page"]),
                    start_sec=s_sec,
                    end_sec=e_sec,
                    up_name=r["up_name"],
                    pub_date=r["pub_date"],
                    scene_type=r["scene_type"],
                    text_preview=r["content"],
                    subtitle_ids=[sid],
                )
            )

    if custom_segments:
        for seg in custom_segments:
            s_sec = round(max(0.0, float(seg["start_sec"]) - pad_before_sec), 2)
            e_sec = round(max(s_sec, float(seg["end_sec"]) + pad_after_sec), 2)
            raw_items.append(
                RawClipRequest(
                    bvid=str(seg["bvid"]),
                    page=int(seg.get("page", 1)),
                    start_sec=s_sec,
                    end_sec=e_sec,
                    up_name=str(seg.get("up_name", "")),
                    pub_date=str(seg.get("pub_date", "")),
                    scene_type=str(seg.get("scene_type", "unknown")),
                    text_preview=str(seg.get("label", "")),
                )
            )

    # 合并同视频同分 P 下重叠或间隔 <= merge_gap_sec 的相邻片段
    merged: List[RawClipRequest] = []
    for item in raw_items:
        if (
            merged
            and merged[-1].bvid == item.bvid
            and merged[-1].page == item.page
            and item.start_sec <= merged[-1].end_sec + merge_gap_sec
            and item.end_sec >= merged[-1].start_sec
        ):
            prev = merged[-1]
            prev.start_sec = min(prev.start_sec, item.start_sec)
            prev.end_sec = max(prev.end_sec, item.end_sec)
            prev.subtitle_ids.extend(item.subtitle_ids)
            if item.text_preview:
                prev.text_preview = f"{prev.text_preview} / {item.text_preview}".strip(" /")
        else:
            merged.append(item)

    job_id = f"clip_{uuid.uuid4().hex[:10]}"
    segments_out: List[Dict[str, Any]] = []
    total_duration = 0.0

    for idx, seg in enumerate(merged, start=1):
        dur = round(max(0.0, seg.end_sec - seg.start_sec), 2)
        total_duration += dur
        clip_filename = f"{job_id}_seg{idx:03d}_{seg.bvid}_p{seg.page}.mp4"
        source_placeholder = f"cache/{seg.bvid}_p{seg.page}.mp4"
        segments_out.append(
            {
                "index": idx,
                "bvid": seg.bvid,
                "page": seg.page,
                "up_name": seg.up_name,
                "pub_date": seg.pub_date,
                "scene_type": seg.scene_type,
                "start_sec": seg.start_sec,
                "end_sec": seg.end_sec,
                "duration_sec": dur,
                "start_hms": format_sec_hms(seg.start_sec),
                "end_hms": format_sec_hms(seg.end_sec),
                "subtitle_ids": seg.subtitle_ids,
                "text_preview": seg.text_preview,
                "source_url": build_bilibili_jump_url(seg.bvid, seg.page, seg.start_sec),
                "clip_file": clip_filename,
                "ffmpeg_cut_cmd": [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{seg.start_sec:.2f}",
                    "-to",
                    f"{seg.end_sec:.2f}",
                    "-i",
                    source_placeholder,
                    "-c:v",
                    "libx264",
                    "-c:a",
                    "aac",
                    clip_filename,
                ],
            }
        )

    concat_list_lines = [f"file '{s['clip_file']}'" for s in segments_out]
    return {
        "schema_version": "1.0",
        "job_id": job_id,
        "output_name": output_name,
        "options": {
            "pad_before_sec": pad_before_sec,
            "pad_after_sec": pad_after_sec,
            "merge_gap_sec": merge_gap_sec,
        },
        "segment_count": len(segments_out),
        "estimated_duration_sec": round(total_duration, 2),
        "segments": segments_out,
        "ffmpeg_concat_manifest": concat_list_lines,
        "ffmpeg_concat_cmd": [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            f"{job_id}_concat.txt",
            "-c",
            "copy",
            output_name,
        ],
    }
