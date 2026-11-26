"""抓取、清洗、标准化与增量入库编排模块（支持直接抓取「直播回放」合集/系列）."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .bilibili_client import BilibiliAPIError, BilibiliClient, DEFAULT_LAN_PRIORITY
from .cleaner import clean_bcc_payload, clean_srt_text
from .config import AppConfig
from .db import CorpusDB, SubtitleRow, VideoRecord

ProgressCallback = Callable[[str], None]


@dataclass
class IngestSummary:
    run_id: str
    videos_seen: int = 0
    parts_processed: int = 0
    parts_ok: int = 0
    parts_skipped: int = 0
    parts_no_subtitle: int = 0
    parts_pending_login: int = 0
    parts_failed: int = 0
    lines_inserted: int = 0
    lines_duplicate: int = 0
    errors: List[str] = field(default_factory=list)


def ingest_bvids(
    db: CorpusDB,
    config: AppConfig,
    bvids: Sequence[str],
    *,
    up_name_override: Optional[str] = None,
    pages_filter: Optional[Sequence[int]] = None,
    prefer_lans: Sequence[str] = DEFAULT_LAN_PRIORITY,
    force: bool = False,
    retry_no_sub: bool = False,
    cooldown_every: int = 30,
    cooldown_sec: float = 8.0,
    client: Optional[BilibiliClient] = None,
    progress_cb: Optional[ProgressCallback] = None,
) -> IngestSummary:
    """对给定 BV 号列表执行分 P 抓取、AI 字幕清洗与增量入库（支持秒级断点跳过与礼貌冷却）."""
    db.init_db()
    run_id = uuid.uuid4().hex[:12]
    summary = IngestSummary(run_id=run_id)
    page_allow = set(pages_filter) if pages_filter else None

    def emit(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    own_client = client is None
    bili = client or BilibiliClient(
        cookie=config.bilibili_cookie,
        delay=config.request_delay,
        jitter=config.request_jitter,
        max_retries=config.max_retries,
    )

    fetched_network_videos = 0
    try:
        total_bvids = len(bvids)
        for idx, raw_bvid in enumerate(bvids, start=1):
            bvid = raw_bvid.strip()
            if not bvid:
                continue
            summary.videos_seen += 1

            # 断点续抓秒级预判：整部视频全部分 P 均已入库（且无强制/指定分 P）时直接跳过，不发任何 HTTP 请求
            if (
                not force
                and page_allow is None
                and db.is_bvid_completed(bvid, allow_no_subtitle=not retry_no_sub)
            ):
                summary.parts_skipped += 1
                emit(f"[{idx}/{total_bvids}] 已完整入库，跳过网络请求: {bvid}")
                continue

            if cooldown_every > 0 and fetched_network_videos > 0 and fetched_network_videos % cooldown_every == 0:
                emit(f"[限速冷却] 已连续抓取 {fetched_network_videos} 场，休息 {cooldown_sec:.0f} 秒降低风控风险...")
                time.sleep(cooldown_sec)
            fetched_network_videos += 1

            try:
                vinfo = bili.get_video_info(bvid)
            except Exception as exc:
                summary.parts_failed += 1
                msg = f"[{idx}/{total_bvids}] {bvid} 获取视频详情失败: {exc}"
                summary.errors.append(msg)
                emit(msg)
                db.log_ingest(run_id, bvid, 0, "failed", msg)
                continue

            up_name = up_name_override or config.resolve_up_name(
                uid=vinfo.up_uid, raw_name=vinfo.raw_up_name
            )
            extra_meta = {
                "aid": vinfo.aid,
                "desc_snippet": (vinfo.desc or "")[:500],
            }

            for p in vinfo.pages:
                if page_allow is not None and p.page not in page_allow:
                    continue
                summary.parts_processed += 1

                existing = db.get_video(vinfo.bvid, p.page)
                if existing and not force:
                    st = existing.get("subtitle_status")
                    if st == "ok":
                        summary.parts_skipped += 1
                        emit(
                            f"[{idx}/{total_bvids}] 跳过已入库: {up_name} {vinfo.bvid} P{p.page} ({vinfo.pub_date})"
                        )
                        db.log_ingest(run_id, vinfo.bvid, p.page, "skipped", "already ok")
                        continue
                    if st == "no_subtitle" and not retry_no_sub:
                        summary.parts_skipped += 1
                        db.log_ingest(
                            run_id, vinfo.bvid, p.page, "skipped", "previously no_subtitle"
                        )
                        continue

                try:
                    tracks = bili.list_subtitle_tracks(vinfo.bvid, p.cid)
                    chosen = bili.pick_subtitle_track(tracks, prefer_lans=prefer_lans)
                    if chosen is None:
                        if not bili.has_sessdata:
                            # 未配置 SESSDATA 时不误判为 no_subtitle，保持 pending 便于登录后直接增量补抓
                            vid_rec = VideoRecord(
                                bvid=vinfo.bvid,
                                page=p.page,
                                cid=p.cid,
                                up_uid=vinfo.up_uid,
                                up_name=up_name,
                                raw_up_name=vinfo.raw_up_name,
                                title=vinfo.title,
                                part_title=p.part_title,
                                pub_date=vinfo.pub_date,
                                pub_ts=vinfo.pub_ts,
                                duration_sec=p.duration_sec or vinfo.duration_sec,
                                subtitle_lan="",
                                subtitle_status="pending",
                                line_count=0,
                                extra_meta=extra_meta,
                            )
                            db.save_video_and_subtitles(vid_rec, [], replace_existing=False)
                            summary.parts_pending_login += 1
                            msg = (
                                f"[{idx}/{total_bvids}] {up_name} {vinfo.bvid} P{p.page} "
                                "未获取到字幕轨（未配置 B 站登录 Cookie SESSDATA，已暂存为 pending）"
                            )
                            emit(msg)
                            db.log_ingest(run_id, vinfo.bvid, p.page, "pending_login", msg)
                            continue

                        vid_rec = VideoRecord(
                            bvid=vinfo.bvid,
                            page=p.page,
                            cid=p.cid,
                            up_uid=vinfo.up_uid,
                            up_name=up_name,
                            raw_up_name=vinfo.raw_up_name,
                            title=vinfo.title,
                            part_title=p.part_title,
                            pub_date=vinfo.pub_date,
                            pub_ts=vinfo.pub_ts,
                            duration_sec=p.duration_sec or vinfo.duration_sec,
                            subtitle_lan="",
                            subtitle_status="no_subtitle",
                            line_count=0,
                            extra_meta=extra_meta,
                        )
                        db.save_video_and_subtitles(vid_rec, [], replace_existing=False)
                        summary.parts_no_subtitle += 1
                        emit(
                            f"[{idx}/{total_bvids}] 无字幕轨: {up_name} {vinfo.bvid} P{p.page} {vinfo.title[:30]}"
                        )
                        db.log_ingest(
                            run_id, vinfo.bvid, p.page, "no_subtitle", "该分 P 暂无 B 站字幕轨"
                        )
                        continue

                    raw_sub = bili.fetch_subtitle_json(chosen.subtitle_url)
                    cleaned_items = clean_bcc_payload(raw_sub, up_name=up_name)
                    source_tag = (
                        "bilibili_ai"
                        if chosen.lan.startswith("ai-") or chosen.ai_type > 0
                        else "bilibili_cc"
                    )
                    rows = [
                        SubtitleRow(
                            up_name=up_name,
                            bvid=vinfo.bvid,
                            page=p.page,
                            pub_date=vinfo.pub_date,
                            start_sec=item.start_sec,
                            end_sec=item.end_sec,
                            content=item.content,
                            scene_type="unknown",
                            is_visible=1,
                            source=source_tag,
                        )
                        for item in cleaned_items
                    ]
                    vid_rec = VideoRecord(
                        bvid=vinfo.bvid,
                        page=p.page,
                        cid=p.cid,
                        up_uid=vinfo.up_uid,
                        up_name=up_name,
                        raw_up_name=vinfo.raw_up_name,
                        title=vinfo.title,
                        part_title=p.part_title,
                        pub_date=vinfo.pub_date,
                        pub_ts=vinfo.pub_ts,
                        duration_sec=p.duration_sec or vinfo.duration_sec,
                        subtitle_lan=chosen.lan,
                        subtitle_status="ok" if rows else "no_subtitle",
                        line_count=len(rows),
                        extra_meta=extra_meta,
                    )
                    inserted, dup = db.save_video_and_subtitles(
                        vid_rec, rows, replace_existing=force
                    )
                    summary.lines_inserted += inserted
                    summary.lines_duplicate += dup
                    if rows:
                        summary.parts_ok += 1
                        emit(
                            f"[{idx}/{total_bvids}] 入库成功: {up_name} {vinfo.bvid} P{p.page} "
                            f"({vinfo.pub_date}) [{chosen.lan}] +{inserted}条 | {vinfo.title[:28]}"
                        )
                        db.log_ingest(
                            run_id,
                            vinfo.bvid,
                            p.page,
                            "ok",
                            f"lan={chosen.lan} inserted={inserted} dup={dup}",
                        )
                    else:
                        summary.parts_no_subtitle += 1
                        db.log_ingest(
                            run_id, vinfo.bvid, p.page, "no_subtitle", "cleaned lines=0"
                        )
                except (BilibiliAPIError, Exception) as exc:
                    summary.parts_failed += 1
                    msg = f"[{idx}/{total_bvids}] {vinfo.bvid} P{p.page} 抓取字幕失败: {exc}"
                    summary.errors.append(msg)
                    emit(msg)
                    db.log_ingest(run_id, vinfo.bvid, p.page, "failed", msg)
    finally:
        if own_client:
            bili.close()

    return summary


def ingest_from_up(
    db: CorpusDB,
    config: AppConfig,
    up_or_uid: str,
    *,
    limit: Optional[int] = 5,
    source: str = "auto",  # auto | series | season | space
    series_id: Optional[int] = None,
    season_id: Optional[int] = None,
    keyword: str = "回放",
    title_contains: Optional[str] = None,
    since_date: Optional[str] = None,
    until_date: Optional[str] = None,
    force: bool = False,
    retry_no_sub: bool = False,
    progress_cb: Optional[ProgressCallback] = None,
) -> IngestSummary:
    """按 UP 主简称或 UID 拉取「直播回放」合集/系列并增量入库.

    默认 source='auto'：优先使用 config/up_list.json 中已核实的「直播回放」series_id，
    直接精准拉取该成员全部回放，不混入切片/Vlog/MV！
    """
    member = config.resolve_member(up_or_uid)
    if member is not None:
        mid = member.uid
        up_name = member.up_name
        eff_series_id = series_id if series_id is not None else member.replay_series_id
        eff_season_id = season_id if season_id is not None else member.replay_season_id
    elif str(up_or_uid).strip().isdigit():
        mid = int(str(up_or_uid).strip())
        up_name = config.resolve_up_name(uid=mid)
        eff_series_id = series_id
        eff_season_id = season_id
    else:
        raise ValueError(f"无法识别 UP 主或 UID: {up_or_uid!r}")

    def emit(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    with BilibiliClient(
        cookie=config.bilibili_cookie,
        delay=config.request_delay,
        jitter=config.request_jitter,
        max_retries=config.max_retries,
    ) as bili:
        if source == "series" or (source == "auto" and eff_series_id is not None):
            if eff_series_id is None:
                raise ValueError(f"{up_name} 未配置 replay_series_id，请通过 --series-id 指定")
            emit(
                f"正在从 [{up_name}] 的「直播回放」合集/列表 (series_id={eff_series_id}) 拉取视频清单..."
            )
            briefs = bili.list_series_videos(
                mid=mid,
                series_id=eff_series_id,
                limit=limit,
                since_date=since_date,
                until_date=until_date,
            )
        elif source == "season" or (source == "auto" and eff_season_id is not None):
            if eff_season_id is None:
                raise ValueError(f"{up_name} 未配置 replay_season_id，请通过 --season-id 指定")
            emit(f"正在从 [{up_name}] 的合集 (season_id={eff_season_id}) 拉取视频清单...")
            briefs = bili.list_season_videos(
                mid=mid,
                season_id=eff_season_id,
                limit=limit,
                since_date=since_date,
                until_date=until_date,
            )
        else:
            emit(f"正在从 [{up_name}] 投稿空间按关键词 {keyword!r} 拉取视频清单...")
            briefs = bili.list_up_videos(
                mid=mid,
                limit=limit,
                keyword=keyword,
                since_date=since_date,
                until_date=until_date,
            )

        if title_contains:
            briefs = [b for b in briefs if title_contains in b.title]
        bvids = [b.bvid for b in briefs if b.bvid]
        emit(f"[{up_name}] 获取到 {len(bvids)} 个回放视频，开始逐分 P 抓取 AI 字幕...")
        return ingest_bvids(
            db,
            config,
            bvids,
            up_name_override=up_name,
            force=force,
            retry_no_sub=retry_no_sub,
            client=bili,
            progress_cb=progress_cb,
        )


def ingest_local_file(
    db: CorpusDB,
    file_path: str | Path,
    *,
    up_name: str,
    bvid: str,
    page: int = 1,
    pub_date: str = "1970-01-01",
    title: str = "",
    part_title: str = "",
    scene_type: str = "unknown",
    force: bool = False,
) -> tuple[int, int]:
    """导入本地 BCC JSON 或 SRT 字幕文件（用于离线验证、历史归档或 Whisper 补全）."""
    db.init_db()
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"字幕文件不存在: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        cleaned = clean_bcc_payload(payload, up_name=up_name)
        lan = "local-bcc"
    elif suffix == ".srt":
        cleaned = clean_srt_text(path.read_text(encoding="utf-8"), up_name=up_name)
        lan = "local-srt"
    else:
        raise ValueError(f"暂不支持的文件扩展名 {suffix}，请提供 .json (BCC) 或 .srt")

    rows = [
        SubtitleRow(
            up_name=up_name,
            bvid=bvid,
            page=page,
            pub_date=pub_date,
            start_sec=item.start_sec,
            end_sec=item.end_sec,
            content=item.content,
            scene_type=scene_type,
            is_visible=1,
            source="local_import",
        )
        for item in cleaned
    ]
    duration = int(cleaned[-1].end_sec) if cleaned else 0
    vid = VideoRecord(
        bvid=bvid,
        page=page,
        up_name=up_name,
        title=title or path.stem,
        part_title=part_title,
        pub_date=pub_date,
        duration_sec=duration,
        subtitle_lan=lan,
        subtitle_status="ok" if rows else "no_subtitle",
        line_count=len(rows),
        scene_type=scene_type,
    )
    return db.save_video_and_subtitles(vid, rows, replace_existing=force)


def seed_demo_data(db: CorpusDB) -> Dict[str, Any]:
    """写入覆盖 6 位成员、单播/双播/团播/生日会的演示语料（仅用于单元测试或无网验证）."""
    db.init_db()
    samples: List[Dict[str, Any]] = [
        {
            "video": VideoRecord(
                bvid="BV1DEMO000001",
                page=1,
                cid=100001,
                up_uid=672328094,
                up_name="嘉然",
                raw_up_name="嘉然今天吃什么",
                title="【直播回放】晚上好嘉心糖们！今天杂谈唱歌吃夜宵 2024年05月12日",
                part_title="P1 杂谈与点歌",
                pub_date="2024-05-12",
                duration_sec=7200,
                subtitle_lan="ai-zh",
                subtitle_status="ok",
                extra_meta={"demo": True},
            ),
            "lines": [
                (12.40, 15.80, "哈喽大家好，我是嘉然，欢迎来到今天的晚间杂谈直播！"),
                (16.10, 19.50, "嘉心糖们晚上好呀，今天你们晚饭吃什么了？"),
                (185.00, 189.20, "刚才看到弹幕问我为什么笑这么开心，因为我学会了一首新歌！"),
                (620.50, 625.10, "等一下我调一下麦，这首猫中毒准备好了吗？"),
                (1420.00, 1424.30, "哈哈哈哈你们怎么都在刷小作文，我都看不过来啦。"),
                (2600.20, 2604.90, "今天一个人直播感觉时间过得好快，再唱最后两首歌就下播啦。"),
            ],
        },
        {
            "video": VideoRecord(
                bvid="BV1DEMO000002",
                page=1,
                cid=100002,
                up_uid=672353429,
                up_name="贝拉",
                raw_up_name="贝拉kira",
                title="【直播回放】周末练舞房日常 2024年06月18日",
                part_title="P1 练舞与聊天",
                pub_date="2024-06-18",
                duration_sec=6800,
                subtitle_lan="ai-zh",
                subtitle_status="ok",
                extra_meta={"demo": True},
            ),
            "lines": [
                (35.00, 38.40, "大家好，今天我们两个一起在练舞房排练，晚晚你快过来打个招呼！"),
                (39.10, 41.80, "来啦来啦，向晚大魔王上线！拉姐你今天这个动作好难啊。"),
                (640.00, 643.50, "晚晚你刚才那一拍抢早了，来我们对着镜子再跳一遍。"),
                (644.00, 647.20, "哈哈哈哈贝拉救命，我腿已经软了，咱们先歇两分钟喝口水吧。"),
                (1320.50, 1324.00, "拉姐你看弹幕都在说我们俩像相声演员，谁先笑谁请吃夜宵！"),
                (1980.00, 1984.40, "向晚别跑，轮到你了，刚才抽签抽到你来读这段小作文！"),
                (2750.30, 2754.10, "贝拉你别光笑呀，下一首双人合唱你起个头！"),
            ],
        },
        {
            "video": VideoRecord(
                bvid="BV1DEMO000003",
                page=1,
                cid=100003,
                up_uid=672342685,
                up_name="乃琳",
                raw_up_name="乃琳Queen",
                title="【A-SOUL团播回放】枝江游戏夜大乱斗！谁是卧底终极对决 2024年08月20日",
                part_title="P1 谁是卧底",
                pub_date="2024-08-20",
                duration_sec=7600,
                subtitle_lan="ai-zh",
                subtitle_status="ok",
                extra_meta={"demo": True},
            ),
            "lines": [
                (10.00, 14.50, "大家好我们是 A-SOUL，欢迎来到今晚的枝江游戏夜团播！"),
                (15.20, 18.80, "今天嘉然、贝拉、向晚还有乃琳全员到齐，我们先抽签分组。"),
                (720.00, 723.60, "然然你刚才眼神躲闪，我严重怀疑你就是这轮的卧底！"),
                (724.10, 727.90, "冤枉啊乃宝！明明是晚晚和拉姐一直在偷偷交换眼神！"),
                (1510.20, 1514.00, "贝拉快投票，向晚这一波发言直接破防了哈哈哈哈。"),
                (2240.00, 2244.50, "恭喜裁判宣布本轮平局，全员一起唱一首传说的世界庆祝一下！"),
            ],
        },
        {
            "video": VideoRecord(
                bvid="BV1DEMO000004",
                page=1,
                cid=100004,
                up_uid=672346917,
                up_name="向晚",
                raw_up_name="向晚大魔王",
                title="【直播回放】向晚的深夜电台与游戏实况 2024年09月05日",
                part_title="P1 深夜电台",
                pub_date="2024-09-05",
                duration_sec=5900,
                subtitle_lan="ai-zh",
                subtitle_status="ok",
                extra_meta={"demo": True},
            ),
            "lines": [
                (22.00, 26.20, "晚上好顶碗人们，欢迎收听今天的深夜电台。"),
                (310.50, 315.00, "这关怎么又掉下去了，我真的要破防了，弹幕别笑！"),
                (940.00, 944.80, "给大家念一段粉丝写的小作文，写得太感人了谢谢你们一直陪伴。"),
                (1805.10, 1809.60, "今天状态超好，最后清唱一段晚安曲送给还没睡的大家。"),
            ],
        },
        {
            "video": VideoRecord(
                bvid="BV1DEMO000005",
                page=1,
                cid=100005,
                up_uid=3537115310721181,
                up_name="心宜",
                raw_up_name="心宜不是心仪",
                title="【直播回放】心宜思诺小心思联动茶话会 2025年01月15日",
                part_title="P1 双人茶话会",
                pub_date="2025-01-15",
                duration_sec=6400,
                subtitle_lan="ai-zh",
                subtitle_status="ok",
                extra_meta={"demo": True},
            ),
            "lines": [
                (18.00, 22.00, "哈喽大家晚上好！今天是心宜和思诺的冬季茶话会连麦！"),
                (680.00, 684.50, "思诺你快看这个热可可棉花糖，看起来超好吃！"),
                (1350.00, 1354.20, "心宜别光顾着吃，轮到你回答观众提问啦，今年最想唱什么歌？"),
                (2100.00, 2104.80, "我们俩一起合唱一首吧，谢谢大家送的小作文和祝福！"),
            ],
        },
        {
            "video": VideoRecord(
                bvid="BV1DEMO000006",
                page=1,
                cid=100006,
                up_uid=3537115310721781,
                up_name="思诺",
                raw_up_name="思诺snow",
                title="【直播回放】思诺生日会特别纪念晚会 2025年03月09日",
                part_title="P1 生日会主舞台",
                pub_date="2025-03-09",
                duration_sec=7100,
                subtitle_lan="ai-zh",
                subtitle_status="ok",
                extra_meta={"demo": True},
            ),
            "lines": [
                (40.00, 45.00, "欢迎大家来到思诺的生日会特别舞台，今晚准备了超多惊喜！"),
                (520.00, 524.80, "谢谢大家在弹幕里刷生日快乐，看到你们的小作文我眼眶都湿了。"),
                (1480.00, 1485.00, "下一首是专为今年生日会准备的原创歌，希望能一直陪在枝江大家身边。"),
            ],
        },
    ]

    total_inserted = 0
    total_dup = 0
    for item in samples:
        vid: VideoRecord = item["video"]
        rows = [
            SubtitleRow(
                up_name=vid.up_name,
                bvid=vid.bvid,
                page=vid.page,
                pub_date=vid.pub_date,
                start_sec=s,
                end_sec=e,
                content=c,
                scene_type="unknown",
                is_visible=1,
                source="demo_seed",
            )
            for s, e, c in item["lines"]
        ]
        ins, dup = db.save_video_and_subtitles(vid, rows, replace_existing=False)
        total_inserted += ins
        total_dup += dup

    return {
        "videos": len(samples),
        "lines_inserted": total_inserted,
        "lines_duplicate": total_dup,
    }
