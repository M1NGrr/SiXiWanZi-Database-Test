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
from zhijiang_corpus.config import load_config, save_cookie_to_env
from zhijiang_corpus.db import CorpusDB
from zhijiang_corpus.ingest import ingest_bvids, ingest_from_up
from zhijiang_corpus.normalizer import (
    append_literal_replace_rule,
    preview_literal_replace_rule,
    run_corpus_normalization,
)
from zhijiang_corpus.search import build_bilibili_jump_url, format_sec_hms


class VisibilityPatchRequest(BaseModel):
    page: Optional[int] = Field(default=None, description="为空则更新该 BV 号全部分 P")
    is_visible: int = Field(..., ge=0, le=1, description="1=可见，0=隐藏")


class CookieSaveRequest(BaseModel):
    cookie: str = Field(min_length=1, description="B 站 Cookie 字符串或单独的 SESSDATA 值")


class IngestStartRequest(BaseModel):
    target: str = Field(default="all", description="UP 简称（如 沐霂）、UID 或 all")
    limit: int = Field(default=5, ge=0, le=5000, description="每位 UP 抓取条数，0 表示抓取该合集全部")
    since_date: Optional[str] = Field(default=None, description="起始日期 YYYY-MM-DD")
    until_date: Optional[str] = Field(default=None, description="截止日期 YYYY-MM-DD")
    bvids: List[str] = Field(default_factory=list, description="可选：直接指定若干 BV 号抓取")
    force: bool = Field(default=False, description="是否覆盖已入库分 P")


class ClipPlanRequest(BaseModel):
    subtitle_ids: List[int] = Field(default_factory=list)
    custom_segments: List[Dict[str, Any]] = Field(default_factory=list)
    pad_before_sec: float = Field(default=1.5, ge=0.0, le=30.0)
    pad_after_sec: float = Field(default=1.5, ge=0.0, le=30.0)
    merge_gap_sec: float = Field(default=3.0, ge=0.0, le=60.0)
    output_name: str = Field(default="zhijiang_compilation.mp4")


class SubtitleEditRequest(BaseModel):
    content: Optional[str] = Field(default=None, description="修订后的字幕正文（不传则保持不变）")
    speaker_label: Optional[str] = Field(
        default=None,
        pattern="^(unknown|host|mixed|external)$",
        description="修订后的声音来源标签：host | mixed | external | unknown",
    )
    speaker_name: Optional[str] = Field(
        default=None,
        description="修订后的成员名（沐霂/又一/恬豆/梨安），external/unknown 时自动清空",
    )
    note: str = Field(default="", max_length=300, description="可选校对备注")


class BatchSpeakerEditRequest(BaseModel):
    subtitle_ids: List[int] = Field(..., min_length=1, description="要批量修正声音来源的字幕 ID 列表")
    speaker_label: str = Field(
        ...,
        pattern="^(unknown|host|mixed|external)$",
        description="目标声音来源标签：host | mixed | external | unknown",
    )
    speaker_name: Optional[str] = Field(
        default=None,
        description="目标成员名（host/mixed 时必填或按单条 up_name 兜底）",
    )
    note: str = Field(default="", max_length=300, description="可选批量校对备注")


class LiteralRuleRequest(BaseModel):
    wrong_term: str = Field(..., min_length=1, max_length=60, description="ASR 高频误识别错词")
    right_term: str = Field(..., min_length=1, max_length=60, description="规范目标词")
    note: str = Field(default="", max_length=200, description="规则备注")
    apply_corpus: bool = Field(default=True, description="写入规则后是否立即执行全库批量修复")


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

    @router.patch("/videos/{bvid}/visibility", summary="设置视频分 P 可见性（1=可见，0=隐藏）")
    def patch_video_visibility(bvid: str, body: VisibilityPatchRequest) -> Dict[str, Any]:
        updated = db.update_visibility(bvid, body.page, body.is_visible)
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

    @router.patch("/subtitles/{subtitle_id}", summary="人工校对单条字幕正文与/或声音来源")
    def patch_subtitle(subtitle_id: int, body: SubtitleEditRequest) -> Dict[str, Any]:
        if body.content is None and body.speaker_label is None:
            raise HTTPException(status_code=400, detail="请至少修改字幕内容或声音来源之一")
        try:
            res = db.edit_subtitle(
                subtitle_id,
                content=body.content,
                speaker_label=body.speaker_label,
                speaker_name=body.speaker_name,
                note=body.note,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        sub = res["subtitle"]
        sub["start_hms"] = format_sec_hms(float(sub["start_sec"]))
        sub["end_hms"] = format_sec_hms(float(sub["end_sec"]))
        sub["jump_url"] = build_bilibili_jump_url(
            str(sub["bvid"]), int(sub["page"]), float(sub["start_sec"])
        )
        return res

    @router.post("/subtitles/batch-speaker", summary="批量修正多条字幕的声音来源并锁定")
    def batch_patch_speaker(body: BatchSpeakerEditRequest) -> Dict[str, Any]:
        try:
            return db.batch_edit_speakers(
                body.subtitle_ids,
                speaker_label=body.speaker_label,
                speaker_name=body.speaker_name,
                note=body.note,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.get("/review/candidates", summary="获取集中校对队列（声纹灰色边界 / 混音 / 高分无关音轨 / 场次时间轴）")
    def list_review_candidates(
        mode: str = Query(
            default="uncertain",
            pattern="^(uncertain|mixed|external|timeline|locked)$",
            description="uncertain=灰色边界与混音，mixed=仅含混音，external=高分无关音轨，timeline=按BV时间轴顺序，locked=已人工校对",
        ),
        up_name: Optional[str] = Query(default=None),
        bvid: Optional[str] = Query(default=None),
        page: Optional[int] = Query(default=None, ge=1),
        sim_min: float = Query(default=0.40, ge=-1.0, le=1.0),
        sim_max: float = Query(default=0.56, ge=-1.0, le=1.0),
        start_sec: Optional[float] = Query(default=None, ge=0.0),
        end_sec: Optional[float] = Query(default=None, ge=0.0),
        include_locked: bool = Query(default=False),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> Dict[str, Any]:
        try:
            data = db.list_review_candidates(
                mode=mode,
                up_name=up_name,
                bvid=bvid,
                page=page,
                sim_min=sim_min,
                sim_max=sim_max,
                start_sec=start_sec,
                end_sec=end_sec,
                include_locked=include_locked,
                limit=limit,
                offset=offset,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        for it in data["items"]:
            st = float(it["start_sec"])
            ed = float(it["end_sec"])
            it["start_hms"] = format_sec_hms(st)
            it["end_hms"] = format_sec_hms(ed)
            it["jump_url"] = build_bilibili_jump_url(str(it["bvid"]), int(it["page"]), st)
        return data

    @router.get("/edits", summary="查询人工校对审计历史")
    def list_edits(
        bvid: Optional[str] = Query(default=None),
        subtitle_id: Optional[int] = Query(default=None),
        edit_type: Optional[str] = Query(default=None, pattern="^(text|speaker|both|revert)$"),
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> Dict[str, Any]:
        data = db.list_subtitle_edits(
            bvid=bvid,
            subtitle_id=subtitle_id,
            edit_type=edit_type,
            limit=limit,
            offset=offset,
        )
        for it in data["items"]:
            st = float(it["start_sec"])
            it["start_hms"] = format_sec_hms(st)
            it["jump_url"] = build_bilibili_jump_url(str(it["bvid"]), int(it["page"]), st)
        return data

    @router.post("/edits/{edit_id}/revert", summary="一键回滚某次人工校对")
    def revert_edit(edit_id: int) -> Dict[str, Any]:
        try:
            return db.revert_subtitle_edit(edit_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/rules/preview-add", summary="预览新增高频错词全局替换规则对当前语料库的影响")
    def preview_add_rule(body: LiteralRuleRequest) -> Dict[str, Any]:
        try:
            return preview_literal_replace_rule(db, body.wrong_term, body.right_term)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @router.post("/rules/apply-add", summary="沉淀高频错词到 config/correction_rules.json 并可选全库批量修复")
    def apply_add_rule(body: LiteralRuleRequest) -> Dict[str, Any]:
        try:
            rule_res = append_literal_replace_rule(
                body.wrong_term, body.right_term, note=body.note
            )
            norm_res = (
                run_corpus_normalization(db, dry_run=False)
                if body.apply_corpus
                else None
            )
            return {"status": "ok", "rule": rule_res, "normalization": norm_res}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    return router
