"""本地私有运维管理 API（登录态配置、合集抓取任务、场景修正、数据库优化）.

重要：仅在本地管理模式（ZJ_ENABLE_ADMIN=1）下挂载，云端对外服务默认完全关闭此路由。
"""

from __future__ import annotations

import base64
import io
import threading
import time
from typing import Any, Dict, List, Optional

import qrcode
import qrcode.image.svg
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from zhijiang_corpus.bilibili_client import BilibiliClient
from zhijiang_corpus.clip_planner import plan_clip_job
from zhijiang_corpus.config import VALID_SCENE_TYPES, load_config, save_cookie_to_env
from zhijiang_corpus.db import CorpusDB
from zhijiang_corpus.ingest import ingest_bvids, ingest_from_up
from zhijiang_corpus.scene_tagger import run_scene_tagging


class ScenePatchRequest(BaseModel):
    page: Optional[int] = Field(default=None, description="为空则更新该 BV 号全部分 P")
    scene_type: Optional[str] = Field(
        default=None,
        description=f"可选值: {', '.join(VALID_SCENE_TYPES)}",
    )
    lock: bool = Field(default=True, description="是否锁定标签避免自动初筛覆盖")
    reason: str = Field(default="manual_admin", description="备注说明")
    is_visible: Optional[int] = Field(default=None, description="1=可见，0=隐藏")


class CookieSaveRequest(BaseModel):
    cookie: str = Field(min_length=1, description="B 站 Cookie 字符串或单独的 SESSDATA 值")


class IngestStartRequest(BaseModel):
    target: str = Field(default="all", description="UP 简称（如 嘉然）、UID 或 all")
    limit: int = Field(default=5, ge=0, le=5000, description="每位 UP 抓取条数，0 表示抓取该合集全部")
    since_date: Optional[str] = Field(default=None, description="起始日期 YYYY-MM-DD")
    until_date: Optional[str] = Field(default=None, description="截止日期 YYYY-MM-DD")
    bvids: List[str] = Field(default_factory=list, description="可选：直接指定若干 BV 号抓取")
    auto_tag_scenes: bool = Field(default=True, description="抓取结束后自动执行轻量场景初筛")
    force: bool = Field(default=False, description="是否覆盖已入库分 P")


class ClipPlanRequest(BaseModel):
    subtitle_ids: List[int] = Field(default_factory=list)
    custom_segments: List[Dict[str, Any]] = Field(default_factory=list)
    pad_before_sec: float = Field(default=1.5, ge=0.0, le=30.0)
    pad_after_sec: float = Field(default=1.5, ge=0.0, le=30.0)
    merge_gap_sec: float = Field(default=3.0, ge=0.0, le=60.0)
    output_name: str = Field(default="zhijiang_compilation.mp4")


def _normalize_cookie_input(raw: str) -> str:
    text = (raw or "").strip()
    if "=" not in text and text:
        return f"SESSDATA={text}"
    return text


def _make_qr_svg_data_uri(url: str) -> str:
    qr = qrcode.QRCode(box_size=8, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(image_factory=qrcode.image.svg.SvgPathImage)
    buf = io.BytesIO()
    img.save(buf)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


def build_admin_router(db: CorpusDB) -> APIRouter:
    router = APIRouter(prefix="/api/v1/admin", tags=["Local Admin API (Private)"])

    ingest_lock = threading.Lock()
    ingest_state: Dict[str, Any] = {
        "running": False,
        "current_target": "",
        "logs": [],
        "last_summary": "",
        "last_finished_at": None,
    }

    def _append_log(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        with ingest_lock:
            ingest_state["logs"].append(f"[{ts}] {msg}")
            if len(ingest_state["logs"]) > 500:
                ingest_state["logs"] = ingest_state["logs"][-500:]

    @router.get("/auth/status", summary="检查本地 B 站 Cookie 登录态")
    def auth_status() -> Dict[str, Any]:
        cfg = load_config(db_path=db.db_path)
        if not cfg.bilibili_cookie:
            return {"is_login": False, "uname": "", "mid": 0, "has_sessdata": False}
        try:
            with BilibiliClient(cookie=cfg.bilibili_cookie, delay=0.0, jitter=0.0) as bili:
                return bili.verify_login()
        except Exception as exc:
            return {
                "is_login": False,
                "uname": "",
                "mid": 0,
                "has_sessdata": bool("SESSDATA=" in cfg.bilibili_cookie),
                "error": str(exc),
            }

    @router.post("/auth/cookie", summary="保存手动粘贴的 B 站 Cookie 至 .env")
    def auth_save_cookie(body: CookieSaveRequest) -> Dict[str, Any]:
        cookie_str = _normalize_cookie_input(body.cookie)
        save_cookie_to_env(cookie_str)
        with BilibiliClient(cookie=cookie_str, delay=0.0, jitter=0.0) as bili:
            info = bili.verify_login()
        return {"saved": True, **info}

    @router.post("/auth/qrcode", summary="生成 B 站官方扫码登录二维码")
    def auth_generate_qrcode() -> Dict[str, Any]:
        try:
            qr = BilibiliClient.generate_login_qrcode()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"生成 B 站登录二维码失败: {exc}")
        return {
            "qrcode_key": qr["qrcode_key"],
            "url": qr["url"],
            "qr_svg_data_uri": _make_qr_svg_data_uri(qr["url"]),
        }

    @router.get("/auth/qrcode/poll", summary="轮询 B 站扫码登录状态")
    def auth_poll_qrcode(qrcode_key: str = Query(..., min_length=4)) -> Dict[str, Any]:
        res = BilibiliClient.poll_login_qrcode(qrcode_key)
        code = int(res.get("code", -1))
        status_map = {
            0: "登录成功！Cookie 已自动保存到本地 .env",
            86101: "等待手机 B 站 App 扫码...",
            86090: "已扫码，请在手机上点击确认登录...",
            86038: "二维码已失效，请重新生成",
        }
        if code == 0 and res.get("cookie"):
            save_cookie_to_env(res["cookie"])
        return {
            "code": code,
            "message": res.get("message", ""),
            "status_text": status_map.get(code, res.get("message", "")),
        }

    @router.post("/ingest/start", summary="启动后台增量抓取任务")
    def start_ingest(body: IngestStartRequest) -> Dict[str, Any]:
        with ingest_lock:
            if ingest_state["running"]:
                raise HTTPException(status_code=409, detail="当前已有抓取任务正在运行")
            ingest_state["running"] = True
            ingest_state["logs"] = []
            ingest_state["current_target"] = body.target
            ingest_state["last_summary"] = ""

        def _worker() -> None:
            try:
                cfg = load_config(db_path=db.db_path)
                total_ok = 0
                total_lines = 0
                total_skip = 0
                total_pending_login = 0
                total_fail = 0

                if body.bvids:
                    _append_log(f"开始按指定 BV 号抓取 ({len(body.bvids)} 个)...")
                    s = ingest_bvids(
                        db,
                        cfg,
                        body.bvids,
                        force=body.force,
                        progress_cb=_append_log,
                    )
                    total_ok += s.parts_ok
                    total_lines += s.lines_inserted
                    total_skip += s.parts_skipped
                    total_pending_login += s.parts_pending_login
                    total_fail += s.parts_failed
                else:
                    targets = (
                        [m.up_name for m in cfg.members]
                        if body.target.lower() == "all"
                        else [body.target]
                    )
                    eff_limit = None if body.limit == 0 else body.limit
                    for t in targets:
                        with ingest_lock:
                            ingest_state["current_target"] = t
                        s = ingest_from_up(
                            db,
                            cfg,
                            t,
                            limit=eff_limit,
                            source="auto",
                            since_date=body.since_date,
                            until_date=body.until_date,
                            force=body.force,
                            progress_cb=_append_log,
                        )
                        total_ok += s.parts_ok
                        total_lines += s.lines_inserted
                        total_skip += s.parts_skipped
                        total_pending_login += s.parts_pending_login
                        total_fail += s.parts_failed

                if body.auto_tag_scenes and total_ok > 0:
                    _append_log("正在执行轻量场景初筛...")
                    tagged = run_scene_tagging(db, cfg, only_unknown=True, dry_run=False)
                    _append_log(f"场景初筛完成，更新 {len(tagged)} 个分 P。")

                summary_str = (
                    f"成功分P {total_ok}，新增字幕 {total_lines} 条，"
                    f"跳过 {total_skip}，待登录 {total_pending_login}，失败 {total_fail}"
                )
                _append_log(f"任务结束：{summary_str}")
                with ingest_lock:
                    ingest_state["last_summary"] = summary_str
            except Exception as exc:
                _append_log(f"抓取任务异常中止：{exc}")
                with ingest_lock:
                    ingest_state["last_summary"] = f"异常: {exc}"
            finally:
                with ingest_lock:
                    ingest_state["running"] = False
                    ingest_state["current_target"] = ""
                    ingest_state["last_finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        threading.Thread(target=_worker, daemon=True).start()
        return {"started": True, "target": body.target, "limit": body.limit}

    @router.get("/ingest/status", summary="查询后台抓取实时进度与日志")
    def ingest_status() -> Dict[str, Any]:
        with ingest_lock:
            return dict(ingest_state)

    @router.post("/scenes/tag", summary="触发轻量规则场景初筛")
    def api_tag_scenes(only_unknown: bool = False) -> Dict[str, Any]:
        cfg = load_config(db_path=db.db_path)
        res = run_scene_tagging(db, cfg, only_unknown=only_unknown, dry_run=False)
        skipped = sum(1 for r in res if r.skipped_locked)
        return {"processed": len(res), "skipped_locked": skipped}

    @router.patch("/videos/{bvid}/scene", summary="人工修正并锁定场景标签或可见性")
    def patch_video_scene(bvid: str, body: ScenePatchRequest) -> Dict[str, Any]:
        updated = 0
        if body.scene_type is not None:
            if body.scene_type not in VALID_SCENE_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"无效 scene_type，可选值: {VALID_SCENE_TYPES}",
                )
            updated += db.update_scene_type(
                bvid,
                body.page,
                body.scene_type,
                confidence=1.0,
                reason=body.reason,
                lock=body.lock,
            )
        if body.is_visible is not None:
            updated += db.update_visibility(bvid, body.page, body.is_visible)
        if updated == 0:
            raise HTTPException(status_code=404, detail=f"未找到视频记录: {bvid}")
        return {"status": "ok", "bvid": bvid, "page": body.page, "affected_rows": updated}

    @router.post("/db/optimize", summary="大批量抓取后优化 FTS5 索引与回收 WAL")
    def optimize_db() -> Dict[str, Any]:
        return db.optimize_db()

    @router.post("/clips/plan", summary="预留：二阶段 FFmpeg 切片清单规划")
    def create_clip_plan(body: ClipPlanRequest) -> Dict[str, Any]:
        if not body.subtitle_ids and not body.custom_segments:
            raise HTTPException(status_code=400, detail="请至少提供 subtitle_ids 或 custom_segments")
        return plan_clip_job(
            db,
            subtitle_ids=body.subtitle_ids,
            custom_segments=body.custom_segments,
            pad_before_sec=body.pad_before_sec,
            pad_after_sec=body.pad_after_sec,
            merge_gap_sec=body.merge_gap_sec,
            output_name=body.output_name,
        )

    return router
