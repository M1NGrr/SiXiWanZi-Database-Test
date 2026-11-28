"""公网只读检索 API（部署至云端对外开放时仅挂载本路由，无任何写操作或 Cookie 暴露）."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query, Response

from sixiwanzi_corpus.config import AppConfig, PROJECT_ROOT
from sixiwanzi_corpus.db import CorpusDB
from sixiwanzi_corpus.search import SearchQuery, SearchService


def _load_announcements() -> List[Dict[str, Any]]:
    """实时读取 config/announcements.json 中的有效公告，改文件即刻生效无需重启服务."""
    env_path = os.getenv("ZJ_ANNOUNCEMENTS_PATH", "").strip()
    path = Path(env_path) if env_path else (PROJECT_ROOT / "config" / "announcements.json")
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = raw.get("items", []) if isinstance(raw, dict) else (raw if isinstance(raw, list) else [])
    active_items: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("active", True) is False:
            continue
        content = str(item.get("content", "")).strip()
        if not content:
            continue
        level = str(item.get("level", "info")).strip().lower()
        if level not in ("info", "change", "warn"):
            level = "info"
        active_items.append(
            {
                "id": str(item.get("id", "")).strip(),
                "tag": str(item.get("tag", "公告")).strip() or "公告",
                "level": level,
                "date": str(item.get("date", "")).strip(),
                "pinned": bool(item.get("pinned", False)),
                "content": content,
            }
        )
    # 置顶优先，同级按日期倒序
    pinned = sorted([x for x in active_items if x["pinned"]], key=lambda x: x["date"], reverse=True)
    normal = sorted([x for x in active_items if not x["pinned"]], key=lambda x: x["date"], reverse=True)
    return pinned + normal


def build_public_router(db: CorpusDB, cfg: AppConfig) -> APIRouter:
    router = APIRouter(prefix="/api/v1", tags=["Public Read-Only API"])
    search_svc = SearchService(db)

    @router.get("/health", summary="服务健康检查")
    def health() -> Dict[str, Any]:
        return {"status": "ok", "mode": "read_only_api"}

    @router.get("/members", summary="获取成员应援色、粉丝名与站点元数据（供前端渲染）")
    def list_members() -> Dict[str, Any]:
        return {
            "members": [
                {
                    "up_name": m.up_name,
                    "uid": m.uid,
                    "official_name": m.official_name,
                    "group": m.group,
                    "fan_name": m.fan_name,
                    "color": m.color,
                }
                for m in cfg.members
            ],
            "site": {
                "author_name": os.getenv("ZJ_AUTHOR_NAME", "M1NG"),
                "author_bilibili_url": os.getenv("ZJ_AUTHOR_BILI_URL", ""),
                "github_url": os.getenv(
                    "ZJ_GITHUB_URL", "https://github.com/M1NGrr/SiXiWanZi-Database-Test"
                ),
                "upstream_author_name": "没品人类之一",
                "upstream_author_url": "https://space.bilibili.com/396896138",
                "upstream_repo_url": "https://github.com/RicojrH/zhijiang-database",
            },
        }

    @router.get("/announcements", summary="站点公告与更新动态（实时读取 config/announcements.json）")
    def announcements(response: Response) -> Dict[str, Any]:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        return {"items": _load_announcements()}

    @router.get("/stats", summary="语料库公开统计信息（实时读取数据库，禁止缓存）")
    def stats(response: Response) -> Dict[str, Any]:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        raw = db.get_stats()
        # 公网统计不暴露服务器本地磁盘绝对路径
        return {
            "db_size_mb": raw["db_size_mb"],
            "video_parts_total": raw["video_parts_total"],
            "video_parts_ok": raw["video_parts_ok"],
            "subtitles_total": raw["subtitles_total"],
            "subtitles_visible": raw["subtitles_visible"],
            "date_min": raw["date_min"],
            "date_max": raw["date_max"],
            "latest_updated_at": raw.get("latest_updated_at"),
            "latest_video": raw.get("latest_video"),
            "announcements": _load_announcements(),
            "by_up": raw["by_up"],
        }

    @router.get("/search", summary="关键词检索 AI 字幕语料")
    def search(
        q: str = Query(default="", description="检索关键词（空格分隔多词 AND）"),
        up_name: Optional[str] = Query(
            default=None, description="UP 主简称过滤，支持逗号分隔多个，如 沐霂,恬豆"
        ),
        date_from: Optional[str] = Query(default=None, description="起始日期 YYYY-MM-DD"),
        date_to: Optional[str] = Query(default=None, description="截止日期 YYYY-MM-DD"),
        bvid: Optional[str] = Query(default=None, description="限定单个 BV 号"),
        context_lines: int = Query(default=0, ge=0, le=10, description="附带前后上下文字幕条数"),
        fuzzy: bool = Query(
            default=True,
            description="开启同音/别称/Tier 3 歧义词模糊扩展（如搜 四禧丸子 同时命中 四喜丸子，搜 小沐标 同时命中 小目标）",
        ),
        host_only: bool = Query(
            default=False,
            description="兼容参数：过滤声纹识别出的非主播原声（等价于 speaker_source=host）",
        ),
        speaker_source: Optional[str] = Query(
            default=None,
            pattern="^(all|host|host_verified|external)$",
            description="声音来源筛选：all=全部来源，host=仅主播原声（过滤无关音轨），host_verified=仅已识别主播原声，external=仅主播无关音轨",
        ),
        sort: str = Query(default="date_desc", pattern="^(date_desc|date_asc|relevance)$"),
        limit: int = Query(default=20, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> Dict[str, Any]:
        up_list = [x.strip() for x in up_name.split(",") if x.strip()] if up_name else None
        req = SearchQuery(
            q=q,
            up_names=up_list,
            date_from=date_from,
            date_to=date_to,
            bvid=bvid,
            include_hidden=False,  # 公网接口始终只返回 is_visible=1
            host_only=host_only,
            speaker_source=speaker_source,
            fuzzy=fuzzy,
            context_lines=context_lines,
            sort=sort,
            limit=limit,
            offset=offset,
        )
        return search_svc.search(req)

    @router.get("/videos", summary="浏览已收录直播回放分 P 列表")
    def list_videos(
        up_name: Optional[str] = None,
        status: Optional[str] = Query(default="ok", description="默认仅展示已入库字幕的分 P"),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> Dict[str, Any]:
        return search_svc.list_videos(
            up_name=up_name,
            status=status or None,
            include_hidden=False,
            limit=limit,
            offset=offset,
        )

    @router.get("/videos/{bvid}/subtitles", summary="按时间窗口读取某场直播的连续上下文字幕")
    def video_subtitles(
        bvid: str,
        page: int = Query(default=1, ge=1),
        start_sec: Optional[float] = Query(default=None, ge=0.0),
        end_sec: Optional[float] = Query(default=None, ge=0.0),
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> Dict[str, Any]:
        return search_svc.get_subtitle_window(
            bvid=bvid,
            page=page,
            start_sec=start_sec,
            end_sec=end_sec,
            limit=limit,
        )

    return router
