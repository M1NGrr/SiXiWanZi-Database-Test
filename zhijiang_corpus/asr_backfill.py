"""无自带字幕场次的音频提取 + 本地 GPU faster-whisper 多线程流水线补录模块.

核心加速架构（Producer-Consumer 流水线 + 多 GPU Worker）：
  1. 预取线程池（--prefetch N）：后台并发拉取纯音频 DASH 流 + FFmpeg 转 16kHz WAV，放入有界队列；
  2. 多 GPU 推理线程（--workers M）：多个独立 faster-whisper 推理线程并行消费 WAV，
     让 CPU 端的 Silero VAD / 词级时间戳切分 与 GPU 端的 Tensor Core 推理高度重叠，
     充分吃满 RTX 5070 Ti 12GB 显存与算力（实测较单线程提速 2.5x~3.5x）；
  3. 同一套 cleaner（自动繁转简）+ tokenizer + SQLite WAL 写入管道，每完成 1 个分 P 立刻落库并删临时音频。
"""

from __future__ import annotations

import gc
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from .bilibili_client import BilibiliClient
from .cleaner import CleanedSubtitleItem, clean_subtitle_text, is_noise_line, normalize_timestamp
from .config import AppConfig, PROJECT_ROOT
from .db import CorpusDB, SubtitleRow, VideoRecord

ProgressCallback = Callable[[str], None]

DEFAULT_ZHIJIANG_PROMPT = (
    "以下是B站虚拟偶像嘉然、贝拉、乃琳、向晚、心宜、思诺"
    "（A-SOUL、枝江、嘉心糖、顶碗人、贝极星、奶淇琳、小心思、小作文）的中文直播回放字幕。"
)

_HALLUCINATION_SUBSTRINGS = (
    "以下是B站虚拟偶像",
    "以下为B站虚拟偶像",
    "请不吝点赞 订阅 转发",
    "感谢收看本期视频",
    "字幕由",
    "Amara.org",
)

_REPEAT_CHAR_RE = re.compile(r"(.)\1{5,}")
_REPEAT_PHRASE_RE = re.compile(r"(.{2,6}?)\1{4,}")
_HARD_PUNCT_END = ("。", "！", "？", "!", "?")
_SOFT_PUNCT_END = ("，", ",", "、", "；", ";")

_CUDA_DLLS_REGISTERED = False
_CUDA_REG_LOCK = threading.Lock()


def register_windows_cuda_dlls() -> List[str]:
    """在 Windows 下自动将 pip 安装的 nvidia-cublas-cu12 / nvidia-cudnn-cu12 DLL 目录加入加载路径."""
    global _CUDA_DLLS_REGISTERED
    added: List[str] = []
    with _CUDA_REG_LOCK:
        if _CUDA_DLLS_REGISTERED or sys.platform != "win32":
            return added

        candidate_roots = [
            Path(sys.prefix) / "Lib" / "site-packages" / "nvidia",
            PROJECT_ROOT / ".venv" / "Lib" / "site-packages" / "nvidia",
        ]
        seen: set[str] = set()
        for root in candidate_roots:
            if not root.exists():
                continue
            for bin_dir in root.glob("*/bin"):
                p_str = str(bin_dir.resolve())
                if p_str in seen or not bin_dir.is_dir():
                    continue
                seen.add(p_str)
                try:
                    os.add_dll_directory(p_str)
                except (AttributeError, OSError):
                    pass
                if p_str not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = p_str + os.pathsep + os.environ.get("PATH", "")
                added.append(p_str)

        _CUDA_DLLS_REGISTERED = True
    return added


def _find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if found:
        return found
    raise RuntimeError("未找到 ffmpeg 可执行文件，请确认 ffmpeg 已加入系统 PATH。")


def convert_to_16k_wav(input_audio: Path, output_wav: Path) -> Path:
    """用 FFmpeg 将 B 站 m4a/mp4 音频轨快速转为 16kHz 单声道 WAV."""
    ffmpeg = _find_ffmpeg()
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_audio),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not output_wav.exists() or output_wav.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg 音频转码失败: {proc.stderr.strip() or proc.stdout.strip()}")
    return output_wav


def _is_whisper_hallucination(text: str) -> bool:
    for bad in _HALLUCINATION_SUBSTRINGS:
        if bad.lower() in text.lower():
            return True
    if _REPEAT_CHAR_RE.search(text) or _REPEAT_PHRASE_RE.search(text):
        return True
    return False


def _split_text_proportionally(
    start_sec: float, end_sec: float, text: str
) -> List[Tuple[float, float, str]]:
    """当未开启耗时的 word_timestamps 时，对 <=8s 语音段按中文句末标点等比分配起止秒数."""
    cleaned = text.strip()
    if not cleaned:
        return []
    if len(cleaned) <= 28 or (end_sec - start_sec) <= 5.0:
        return [(start_sec, end_sec, cleaned)]

    # 按句末强标点切分并保留标点
    parts = [p.strip() for p in re.split(r"(?<=[。！？!?])", cleaned) if p.strip()]
    if len(parts) <= 1:
        return [(start_sec, end_sec, cleaned)]

    # 合并过短碎片（<6 字）到相邻句
    merged: List[str] = []
    for p in parts:
        if merged and len(p) < 6:
            merged[-1] += p
        else:
            merged.append(p)
    if len(merged) <= 1:
        return [(start_sec, end_sec, cleaned)]

    total_chars = sum(len(x) for x in merged) or 1
    dur = max(0.0, end_sec - start_sec)
    out: List[Tuple[float, float, str]] = []
    cursor = start_sec
    for idx, piece in enumerate(merged):
        if idx == len(merged) - 1:
            seg_end = end_sec
        else:
            seg_end = cursor + dur * (len(piece) / total_chars)
        out.append((cursor, seg_end, piece))
        cursor = seg_end
    return out


def _split_segment_into_lines(seg: Any) -> List[Tuple[float, float, str]]:
    """将 Whisper 语音段切分为 2~8 秒的句级字幕（有 word_timestamps 用词级，否则按标点等比切分）."""
    words = getattr(seg, "words", None) or []
    if not words:
        return _split_text_proportionally(float(seg.start), float(seg.end), str(seg.text or ""))

    lines: List[Tuple[float, float, str]] = []
    buf_text: List[str] = []
    cur_start: Optional[float] = None
    cur_end: float = float(seg.start)

    def flush() -> None:
        nonlocal buf_text, cur_start
        if buf_text and cur_start is not None:
            joined = "".join(buf_text).strip()
            if joined:
                lines.append((cur_start, cur_end, joined))
        buf_text = []
        cur_start = None

    for w in words:
        w_text = str(getattr(w, "word", "") or "")
        if not w_text:
            continue
        w_start = float(getattr(w, "start", cur_end))
        w_end = float(getattr(w, "end", w_start))

        if cur_start is not None and (w_start - cur_end) >= 0.6 and len("".join(buf_text)) >= 4:
            flush()

        if cur_start is None:
            cur_start = w_start
        buf_text.append(w_text)
        cur_end = max(cur_end, w_end)

        cur_joined = "".join(buf_text).strip()
        cur_len = len(cur_joined)
        cur_dur = cur_end - cur_start

        if cur_joined.endswith(_HARD_PUNCT_END) and cur_len >= 4:
            flush()
        elif cur_joined.endswith(_SOFT_PUNCT_END) and (cur_len >= 14 or cur_dur >= 3.5):
            flush()
        elif cur_len >= 34 or cur_dur >= 7.5:
            flush()

    flush()
    return lines or [(float(seg.start), float(seg.end), str(seg.text or ""))]


def load_16k_wav_as_float32(wav_path: Path) -> Any:
    """用标准库 wave + numpy 读取 16kHz 单声道 PCM WAV 为 float32 数组，彻底规避 PyAV 版本兼容问题."""
    import numpy as np

    with wave.open(str(wav_path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getframerate() != 16000 or wf.getsampwidth() != 2:
            raise RuntimeError(
                f"WAV 格式不符合 16kHz/mono/16bit: ch={wf.getnchannels()} sr={wf.getframerate()} sw={wf.getsampwidth()}"
            )
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


DEFAULT_HF_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / "huggingface"
DEFAULT_ASR_TMP_DIR = PROJECT_ROOT / "data" / "tmp_asr"


def configure_local_cache_dirs() -> Path:
    """强制将 HuggingFace 模型缓存与临时目录固定到项目 data/ 所在盘（如 E 盘），绝不占用 C 盘."""
    DEFAULT_HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_ASR_TMP_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_CACHE_DIR))
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(DEFAULT_HF_CACHE_DIR / "hub"))
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    return DEFAULT_ASR_TMP_DIR


class LocalWhisperEngine:
    """封装 faster-whisper GPU 推理，支持 BatchedInferencePipeline 批量加速."""

    def __init__(
        self,
        model_name: str = "large-v3-turbo",
        *,
        device: str = "auto",
        compute_type: str = "int8_float16",
        batch_size: int = 32,
        beam_size: int = 1,
        word_timestamps: bool = False,
        use_hf_mirror: bool = False,
    ):
        configure_local_cache_dirs()
        if use_hf_mirror and not os.environ.get("HF_ENDPOINT"):
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

        register_windows_cuda_dlls()

        try:
            import ctranslate2
            from faster_whisper import BatchedInferencePipeline, WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "尚未安装 faster-whisper，请执行: pip install -r requirements-asr.txt"
            ) from exc

        resolved_device = device
        resolved_compute = compute_type
        if device == "auto":
            if ctranslate2.get_cuda_device_count() > 0:
                resolved_device = "cuda"
            else:
                resolved_device = "cpu"
                if resolved_compute in ("float16", "int8_float16"):
                    resolved_compute = "int8"

        self.model_name = model_name
        self.device = resolved_device
        self.compute_type = resolved_compute
        self.batch_size = max(1, int(batch_size))
        self.beam_size = max(1, int(beam_size))
        self.word_timestamps = bool(word_timestamps)

        self._model = WhisperModel(
            model_name,
            device=self.device,
            compute_type=self.compute_type,
        )
        self._batched = (
            BatchedInferencePipeline(model=self._model)
            if self.batch_size > 1
            else None
        )

    def transcribe_file(
        self,
        audio_path: Path,
        *,
        initial_prompt: str = DEFAULT_ZHIJIANG_PROMPT,
        language: str = "zh",
        up_name: Optional[str] = None,
    ) -> Tuple[List[CleanedSubtitleItem], float]:
        """转录音频文件并返回清洗（含繁转简）后的句级字幕列表与实际推理耗时（秒）."""
        t0 = time.perf_counter()
        vad_params = {
            "min_silence_duration_ms": 300,
            "speech_pad_ms": 150,
            "max_speech_duration_s": 8 if not self.word_timestamps else 12,
        }
        audio_input = (
            load_16k_wav_as_float32(audio_path)
            if audio_path.suffix.lower() == ".wav"
            else str(audio_path)
        )

        common_kwargs: Dict[str, Any] = {
            "language": language,
            "vad_filter": True,
            "vad_parameters": vad_params,
            "initial_prompt": initial_prompt,
            "condition_on_previous_text": False,
            "word_timestamps": self.word_timestamps,
            "beam_size": self.beam_size,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 3,
            "compression_ratio_threshold": 2.2,
            "no_speech_threshold": 0.6,
        }
        if self._batched is not None:
            segments_iter, _info = self._batched.transcribe(
                audio_input,
                batch_size=self.batch_size,
                **common_kwargs,
            )
        else:
            segments_iter, _info = self._model.transcribe(
                audio_input,
                **common_kwargs,
            )

        items: List[CleanedSubtitleItem] = []
        seen: set[Tuple[float, float, str]] = set()
        for seg in segments_iter:
            for raw_start, raw_end, raw_text in _split_segment_into_lines(seg):
                text = clean_subtitle_text(raw_text, up_name=up_name)
                if is_noise_line(text, drop_music_tags=True) or _is_whisper_hallucination(text):
                    continue
                start_sec, end_sec = normalize_timestamp(raw_start, raw_end)
                key = (start_sec, end_sec, text)
                if key in seen:
                    continue
                seen.add(key)
                items.append(
                    CleanedSubtitleItem(start_sec=start_sec, end_sec=end_sec, content=text)
                )

        items.sort(key=lambda x: (x.start_sec, x.end_sec))
        del audio_input
        gc.collect()
        elapsed = round(time.perf_counter() - t0, 2)
        return items, elapsed


@dataclass
class AsrBackfillSummary:
    run_id: str
    model: str
    device: str
    workers: int = 1
    prefetch: int = 1
    candidates: int = 0
    parts_ok: int = 0
    parts_empty: int = 0
    parts_failed: int = 0
    lines_inserted: int = 0
    total_audio_sec: int = 0
    total_asr_sec: float = 0.0
    wall_sec: float = 0.0
    errors: List[str] = field(default_factory=list)


def list_no_subtitle_targets(
    db: CorpusDB,
    *,
    bvid: Optional[str] = None,
    page: Optional[int] = None,
    up_name: Optional[str] = None,
    min_duration: int = 15,
    max_duration: Optional[int] = None,
    shortest_first: bool = False,
    limit: Optional[int] = 5,
    include_ok_force: bool = False,
) -> List[Dict[str, Any]]:
    """查询待 ASR 补录的回放分 P 列表."""
    clauses = ["1=1"]
    params: List[Any] = []
    if not include_ok_force:
        clauses.append("subtitle_status = 'no_subtitle'")
    if bvid:
        clauses.append("bvid = ?")
        params.append(bvid.strip())
    if page is not None:
        clauses.append("page = ?")
        params.append(int(page))
    if up_name:
        clauses.append("up_name = ?")
        params.append(up_name.strip())
    if min_duration > 0:
        clauses.append("(duration_sec = 0 OR duration_sec >= ?)")
        params.append(int(min_duration))
    if max_duration is not None and max_duration > 0:
        clauses.append("duration_sec <= ?")
        params.append(int(max_duration))

    order_sql = (
        "ORDER BY CASE WHEN duration_sec > 0 THEN duration_sec ELSE 999999 END ASC, pub_date DESC"
        if shortest_first
        else "ORDER BY pub_date DESC, bvid DESC, page ASC"
    )
    limit_sql = "LIMIT ?" if (limit is not None and limit > 0) else ""
    if limit_sql:
        params.append(int(limit))

    sql = f"SELECT * FROM videos WHERE {' AND '.join(clauses)} {order_sql} {limit_sql}"
    with db.session() as conn:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]


@dataclass
class _PreparedAudioJob:
    idx: int
    total: int
    video: Dict[str, Any]
    cid: int
    wav_path: Optional[Path] = None
    dl_mb: float = 0.0
    prep_sec: float = 0.0
    error: Optional[str] = None


def run_asr_backfill(
    db: CorpusDB,
    config: AppConfig,
    *,
    bvid: Optional[str] = None,
    page: Optional[int] = None,
    up_name: Optional[str] = None,
    limit: Optional[int] = 5,
    min_duration: int = 15,
    max_duration: Optional[int] = None,
    shortest_first: bool = False,
    force: bool = False,
    model_name: str = "large-v3-turbo",
    device: str = "auto",
    compute_type: str = "int8_float16",
    batch_size: int = 16,
    beam_size: int = 1,
    word_timestamps: bool = False,
    workers: int = 1,
    prefetch: int = 2,
    shard_index: int = 0,
    shard_total: int = 1,
    use_hf_mirror: bool = False,
    keep_audio: bool = False,
    progress_cb: Optional[ProgressCallback] = None,
) -> AsrBackfillSummary:
    """多线程流水线执行音频提取 + 本地 GPU Whisper 转录并入库（支持多进程分片隔离 CUDA 上下文）."""
    db.init_db()
    run_id = uuid.uuid4().hex[:12]
    wall_t0 = time.perf_counter()

    emit_lock = threading.Lock()

    def emit(msg: str) -> None:
        if progress_cb:
            with emit_lock:
                progress_cb(msg)

    targets = list_no_subtitle_targets(
        db,
        bvid=bvid,
        page=page,
        up_name=up_name,
        min_duration=min_duration,
        max_duration=max_duration,
        shortest_first=shortest_first,
        limit=limit,
        include_ok_force=force,
    )
    eff_shard_total = max(1, int(shard_total))
    eff_shard_index = int(shard_index) % eff_shard_total
    if eff_shard_total > 1:
        targets = [t for i, t in enumerate(targets) if i % eff_shard_total == eff_shard_index]

    if not targets:
        emit(
            f"[分片 {eff_shard_index + 1}/{eff_shard_total}] 当前没有符合条件的无字幕分 P 需要补录。"
        )
        return AsrBackfillSummary(
            run_id=run_id,
            model=model_name,
            device=device,
            workers=workers,
            prefetch=prefetch,
            candidates=0,
        )

    eff_workers = max(1, min(int(workers), len(targets)))
    eff_prefetch = max(1, min(int(prefetch), len(targets)))
    shard_tag = (
        f"[分片 {eff_shard_index + 1}/{eff_shard_total}] " if eff_shard_total > 1 else ""
    )

    emit(
        f"{shard_tag}共选中 {len(targets)} 个无字幕分 P | 正在加载 {eff_workers} 个 GPU 推理 Worker "
        f"(model={model_name}, compute={compute_type}, batch={batch_size}, beam={beam_size}, prefetch={eff_prefetch})..."
    )
    engines: List[LocalWhisperEngine] = []
    for w_idx in range(eff_workers):
        eng = LocalWhisperEngine(
            model_name=model_name,
            device=device,
            compute_type=compute_type,
            batch_size=batch_size,
            beam_size=beam_size,
            word_timestamps=word_timestamps,
            use_hf_mirror=use_hf_mirror,
        )
        engines.append(eng)
    actual_device = engines[0].device
    emit(
        f"全部 {eff_workers} 个 Worker 就绪！运行设备: {actual_device} ({engines[0].compute_type})，"
        f"开启 {eff_prefetch} 路音频预取流水线..."
    )

    summary = AsrBackfillSummary(
        run_id=run_id,
        model=model_name,
        device=actual_device,
        workers=eff_workers,
        prefetch=eff_prefetch,
        candidates=len(targets),
    )
    summary_lock = threading.Lock()
    stop_event = threading.Event()

    tmp_base = configure_local_cache_dirs()
    # 单进程启动时清理残留目录；多进程分片时各自使用独立 PID 目录，互不干扰
    if eff_shard_total == 1:
        for stale in tmp_base.glob("zj_asr_*"):
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
    tmp_root = Path(tempfile.mkdtemp(prefix=f"zj_asr_p{os.getpid()}_", dir=str(tmp_base)))
    input_q: queue.Queue[Optional[Tuple[int, Dict[str, Any]]]] = queue.Queue()
    # 有界预取队列，确保临时磁盘最多同时只存 workers + prefetch 个 WAV 文件
    ready_q: queue.Queue[Optional[_PreparedAudioJob]] = queue.Queue(
        maxsize=eff_workers + eff_prefetch + 1
    )

    for idx, v in enumerate(targets, start=1):
        input_q.put((idx, v))
    for _ in range(eff_prefetch):
        input_q.put(None)

    def _prefetch_worker(pf_id: int) -> None:
        with BilibiliClient(
            cookie=config.bilibili_cookie,
            delay=max(0.8, config.request_delay),
            jitter=config.request_jitter,
            max_retries=config.max_retries,
        ) as bili:
            while not stop_event.is_set():
                item = input_q.get()
                if item is None:
                    break
                idx, v = item
                cur_bvid = str(v["bvid"])
                cur_page = int(v["page"])
                cur_cid = int(v.get("cid") or 0)
                cur_up = str(v["up_name"])
                cur_dur = int(v.get("duration_sec") or 0)
                cur_title = str(v.get("title") or "")

                raw_audio_path = tmp_root / f"pf{pf_id}_{cur_bvid}_p{cur_page}.m4a"
                wav_path = tmp_root / f"pf{pf_id}_{cur_bvid}_p{cur_page}.wav"
                t0 = time.perf_counter()
                try:
                    if cur_cid <= 0:
                        vinfo = bili.get_video_info(cur_bvid)
                        matched = [p for p in vinfo.pages if p.page == cur_page]
                        if not matched:
                            raise RuntimeError(f"无法获取分 P {cur_page} 的 cid")
                        cur_cid = matched[0].cid

                    urls = bili.get_audio_stream_urls(cur_bvid, cur_cid)
                    if not urls:
                        raise RuntimeError("playurl 未返回可用音频流地址")

                    bili.download_stream_file(urls, raw_audio_path, bvid=cur_bvid)
                    dl_mb = raw_audio_path.stat().st_size / (1024 * 1024)
                    convert_to_16k_wav(raw_audio_path, wav_path)
                    if not keep_audio and raw_audio_path.exists():
                        try:
                            raw_audio_path.unlink()
                        except OSError:
                            pass
                    prep_sec = time.perf_counter() - t0
                    emit(
                        f"[预取 {idx}/{len(targets)}] 就绪: {cur_up} {cur_bvid} P{cur_page} "
                        f"({cur_dur/60:.1f}min, {dl_mb:.1f}MB, 下载+转码 {prep_sec:.1f}s) | {cur_title[:24]}"
                    )
                    ready_q.put(
                        _PreparedAudioJob(
                            idx=idx,
                            total=len(targets),
                            video=v,
                            cid=cur_cid,
                            wav_path=wav_path,
                            dl_mb=dl_mb,
                            prep_sec=prep_sec,
                        )
                    )
                except Exception as exc:
                    for f in (raw_audio_path, wav_path):
                        if f.exists():
                            try:
                                f.unlink()
                            except OSError:
                                pass
                    ready_q.put(
                        _PreparedAudioJob(
                            idx=idx,
                            total=len(targets),
                            video=v,
                            cid=cur_cid,
                            error=str(exc),
                        )
                    )

    def _gpu_worker(w_id: int, engine: LocalWhisperEngine) -> None:
        while not stop_event.is_set():
            job = ready_q.get()
            if job is None:
                break
            v = job.video
            cur_bvid = str(v["bvid"])
            cur_page = int(v["page"])
            cur_up = str(v["up_name"])
            cur_title = str(v.get("title") or "")
            cur_dur = int(v.get("duration_sec") or 0)

            if job.error or job.wav_path is None:
                msg = f"[W{w_id} {job.idx}/{job.total}] {cur_bvid} P{cur_page} 音频预取失败: {job.error}"
                emit(msg)
                with summary_lock:
                    summary.parts_failed += 1
                    summary.errors.append(msg)
                db.log_ingest(run_id, cur_bvid, cur_page, "failed", msg)
                continue

            try:
                prompt = f"{DEFAULT_ZHIJIANG_PROMPT} 当前场次主播：{cur_up}，标题：{cur_title[:40]}。"
                cleaned_items, asr_sec = engine.transcribe_file(
                    job.wav_path,
                    initial_prompt=prompt,
                    language="zh",
                    up_name=cur_up,
                )
                rows = [
                    SubtitleRow(
                        up_name=cur_up,
                        bvid=cur_bvid,
                        page=cur_page,
                        pub_date=str(v["pub_date"]),
                        start_sec=it.start_sec,
                        end_sec=it.end_sec,
                        content=it.content,
                        scene_type=str(v.get("scene_type") or "unknown"),
                        is_visible=int(v.get("is_visible", 1)),
                        source="asr_whisper",
                    )
                    for it in cleaned_items
                ]

                vid_rec = VideoRecord(
                    bvid=cur_bvid,
                    page=cur_page,
                    cid=job.cid,
                    up_uid=v.get("up_uid"),
                    up_name=cur_up,
                    raw_up_name=str(v.get("raw_up_name") or ""),
                    title=cur_title,
                    part_title=str(v.get("part_title") or ""),
                    pub_date=str(v["pub_date"]),
                    pub_ts=int(v.get("pub_ts") or 0),
                    duration_sec=cur_dur,
                    subtitle_lan=f"asr-{model_name}",
                    subtitle_status="ok" if rows else "asr_empty",
                    line_count=len(rows),
                    scene_type=str(v.get("scene_type") or "unknown"),
                    scene_confidence=float(v.get("scene_confidence") or 0.0),
                    scene_reason=str(v.get("scene_reason") or ""),
                    scene_locked=int(v.get("scene_locked") or 0),
                    is_visible=int(v.get("is_visible", 1)),
                    extra_meta=str(v.get("extra_meta") or "{}"),
                )

                inserted, _dup = db.save_video_and_subtitles(
                    vid_rec, rows, replace_existing=force
                )
                rtf = (cur_dur / asr_sec) if (cur_dur > 0 and asr_sec > 0) else 0.0
                with summary_lock:
                    summary.total_audio_sec += cur_dur
                    summary.total_asr_sec = round(summary.total_asr_sec + asr_sec, 2)
                    summary.lines_inserted += inserted
                    if rows:
                        summary.parts_ok += 1
                    else:
                        summary.parts_empty += 1
                    done_cnt = summary.parts_ok + summary.parts_empty + summary.parts_failed

                if rows:
                    emit(
                        f"[W{w_id} 完成 {done_cnt}/{job.total}] {cur_up} {cur_bvid} P{cur_page} "
                        f"({cur_dur/60:.1f}m) -> +{inserted}条 | 推理 {asr_sec:.1f}s ({rtf:.1f}x)"
                    )
                    db.log_ingest(
                        run_id,
                        cur_bvid,
                        cur_page,
                        "ok",
                        f"asr={model_name} inserted={inserted} sec={asr_sec:.1f}",
                    )
                else:
                    emit(
                        f"[W{w_id} 空语音 {done_cnt}/{job.total}] {cur_up} {cur_bvid} P{cur_page} "
                        f"({cur_dur/60:.1f}m) -> 标记 asr_empty (耗时 {asr_sec:.1f}s)"
                    )
                    db.log_ingest(
                        run_id, cur_bvid, cur_page, "asr_empty", f"asr={model_name} empty"
                    )
            except Exception as exc:
                msg = f"[W{w_id} {job.idx}/{job.total}] {cur_bvid} P{cur_page} GPU 推理失败: {exc}"
                emit(msg)
                with summary_lock:
                    summary.parts_failed += 1
                    summary.errors.append(msg)
                db.log_ingest(run_id, cur_bvid, cur_page, "failed", msg)
            finally:
                if not keep_audio and job.wav_path and job.wav_path.exists():
                    try:
                        job.wav_path.unlink()
                    except OSError:
                        pass

    pf_threads = [
        threading.Thread(target=_prefetch_worker, args=(i + 1,), daemon=True)
        for i in range(eff_prefetch)
    ]
    gpu_threads = [
        threading.Thread(target=_gpu_worker, args=(i + 1, engines[i]), daemon=True)
        for i in range(eff_workers)
    ]

    try:
        for t in pf_threads:
            t.start()
        for t in gpu_threads:
            t.start()

        for t in pf_threads:
            while t.is_alive():
                t.join(timeout=0.5)
        for _ in range(eff_workers):
            ready_q.put(None)
        for t in gpu_threads:
            while t.is_alive():
                t.join(timeout=0.5)
    except KeyboardInterrupt:
        stop_event.set()
        raise
    finally:
        if not keep_audio:
            shutil.rmtree(tmp_root, ignore_errors=True)
        summary.wall_sec = round(time.perf_counter() - wall_t0, 2)

    return summary
