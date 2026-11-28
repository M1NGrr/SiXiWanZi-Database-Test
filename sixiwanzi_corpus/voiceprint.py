"""基于阿里 3D-Speaker CAM++（ONNX）的成员声纹注册与 Re/观影场次主播人声打分模块.

架构设计：
  - 仅在真正执行声纹提取/打分时按需导入 numpy 与 sherpa_onnx，云端公网只读服务器零 ML 依赖；
  - 4 位成员声纹模板以轻量 JSON 保存在 config/voiceprints.json（约 12 KB）；
  - 27 MB CAM++ ONNX 模型默认缓存在 data/models/ 下，缺失时支持自动下载；
  - 采用三段式判定规则（不重跑 Whisper 重切字幕）：
      1) 主阈值：sim_mean >= 0.50 -> host（主播原声）
      2) 拼接句召回：sim_peak >= 0.52 -> mixed（半句主播插嘴 + 半句外部原声）
      3) 上下文平滑：sim_mean >= 0.45 且 1.0s 内紧邻 >= 0.54 的高置信主播句 -> host
      4) 其余 -> external（Re 视频/电影/综艺/BGM 非主播原声）
"""

from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import httpx

from .asr_backfill import register_windows_cuda_dlls
from .bilibili_client import BilibiliClient, DEFAULT_UA
from .cleaner import clean_bcc_payload
from .config import AppConfig, PROJECT_ROOT
from .db import CorpusDB

ProgressCallback = Callable[[str], None]

VALID_SPEAKER_LABELS = ("unknown", "host", "mixed", "external")

DEFAULT_VOICEPRINTS_PATH = PROJECT_ROOT / "config" / "voiceprints.json"
DEFAULT_MODEL_FILENAME = "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "models" / DEFAULT_MODEL_FILENAME

DEFAULT_MODEL_URLS = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
    "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
    "https://ghfast.top/https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    "speaker-recongition-models/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
)

DEFAULT_THRESHOLDS: Dict[str, float] = {
    "host_thr": 0.50,
    "peak_thr": 0.52,
    "smooth_floor": 0.45,
    "strong_neighbor": 0.54,
    "neighbor_gap": 1.0,
}

_RE_WATCH_PATTERNS = (
    re.compile(r"(?<![a-zA-Z])re\s*[:：]", re.IGNORECASE),
    re.compile(r"观影|一起看|看电影|电影回|看番|看剧|追剧|看综艺|鉴赏|反应回|拉片|陪看"),
)


def parse_time_to_sec(val: str | int | float) -> float:
    """将 'HH:MM:SS'、'MM:SS' 或秒数字符串解析为浮点秒."""
    if isinstance(val, (int, float)):
        return max(0.0, float(val))
    s = str(val).strip()
    if ":" not in s:
        return max(0.0, float(s))
    parts = [float(p) for p in s.split(":")]
    if len(parts) == 2:
        return max(0.0, parts[0] * 60.0 + parts[1])
    if len(parts) == 3:
        return max(0.0, parts[0] * 3600.0 + parts[1] * 60.0 + parts[2])
    raise ValueError(f"无法解析时间格式: {val!r}（支持秒数、MM:SS 或 HH:MM:SS）")


def is_re_watch_title(title: str) -> bool:
    """启发式判断视频标题是否属于 Re/观影/一起看/电影/剧集类混合人声场次."""
    text = (title or "").strip()
    if not text:
        return False
    return any(p.search(text) for p in _RE_WATCH_PATTERNS)


def load_voiceprints(path: Path = DEFAULT_VOICEPRINTS_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {
            "model": DEFAULT_MODEL_FILENAME,
            "dim": 192,
            "thresholds": dict(DEFAULT_THRESHOLDS),
            "members": {},
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    if "thresholds" not in data or not isinstance(data["thresholds"], dict):
        data["thresholds"] = dict(DEFAULT_THRESHOLDS)
    else:
        merged = dict(DEFAULT_THRESHOLDS)
        merged.update({k: float(v) for k, v in data["thresholds"].items() if k in DEFAULT_THRESHOLDS})
        data["thresholds"] = merged
    data.setdefault("members", {})
    return data


def save_voiceprints(data: Dict[str, Any], path: Path = DEFAULT_VOICEPRINTS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def classify_utterance_scores(
    items: List[Dict[str, Any]],
    *,
    host_thr: float = DEFAULT_THRESHOLDS["host_thr"],
    peak_thr: float = DEFAULT_THRESHOLDS["peak_thr"],
    smooth_floor: float = DEFAULT_THRESHOLDS["smooth_floor"],
    strong_neighbor: float = DEFAULT_THRESHOLDS["strong_neighbor"],
    neighbor_gap: float = DEFAULT_THRESHOLDS["neighbor_gap"],
) -> List[Dict[str, Any]]:
    """纯 Python 四成员三段式声纹判定（直接原地更新并返回 items，不依赖 numpy/onnx）."""
    for it in items:
        sim_mean = float(it.get("sim_mean", 0.0))
        sim_peak = float(it.get("sim_peak", sim_mean))
        best_member = str(it.get("best_member") or "")
        peak_member = str(it.get("peak_member") or best_member)
        if sim_mean >= host_thr:
            it["speaker_label"] = "host"
            it["speaker_name"] = best_member
            it["reason"] = "direct"
        elif sim_peak >= peak_thr:
            it["speaker_label"] = "mixed"
            it["speaker_name"] = peak_member
            it["reason"] = "peak"
        else:
            it["speaker_label"] = "external"
            it["speaker_name"] = ""
            it["reason"] = "low"

    n = len(items)
    for i, it in enumerate(items):
        if it["speaker_label"] != "external":
            continue
        start_i = float(it["start_sec"])
        end_i = float(it["end_sec"])
        member_sims = it.get("member_sims")

        def _neighbor_match(nb: Dict[str, Any], gap: float) -> Optional[Tuple[float, str]]:
            if gap > neighbor_gap:
                return None
            nb_sim = float(nb.get("sim_mean", 0.0))
            if nb_sim < strong_neighbor:
                return None
            nb_name = str(nb.get("speaker_name") or nb.get("best_member") or "")
            if isinstance(member_sims, dict) and nb_name:
                cur_sim_for_nb = float(member_sims.get(nb_name, 0.0))
            else:
                cur_best = str(it.get("best_member") or "")
                if cur_best and nb_name and cur_best != nb_name:
                    return None
                cur_sim_for_nb = float(it.get("sim_mean", 0.0))
            if cur_sim_for_nb >= smooth_floor:
                return (nb_sim, nb_name)
            return None

        cand_prev = _neighbor_match(items[i - 1], start_i - float(items[i - 1]["end_sec"])) if i > 0 else None
        cand_next = _neighbor_match(items[i + 1], float(items[i + 1]["start_sec"]) - end_i) if i + 1 < n else None
        chosen = None
        if cand_prev and cand_next:
            chosen = cand_prev if cand_prev[0] >= cand_next[0] else cand_next
        else:
            chosen = cand_prev or cand_next
        if chosen is not None:
            it["speaker_label"] = "host"
            it["speaker_name"] = chosen[1]
            it["reason"] = "smooth"

    return items


def _require_numpy():
    try:
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "声纹识别需要安装本地可选依赖：pip install -r requirements-sv.txt"
        ) from exc
    return np


def _require_sv_deps():
    np = _require_numpy()
    try:
        import sherpa_onnx  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "声纹识别需要安装本地可选依赖：pip install -r requirements-sv.txt"
        ) from exc
    return np, sherpa_onnx


def _extract_kaldi_fbank(mono: Any, sr: int = 16000) -> Any:
    """用与 sherpa-onnx 完全一致的 Kaldi 80 维 Fbank 参数提取整段特征并做全局均值归一化."""
    np = _require_numpy()
    import kaldi_native_fbank as knf  # type: ignore

    if len(mono) < int(0.4 * sr):
        return None
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0.0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = float(sr)
    opts.mel_opts.num_bins = 80
    opts.mel_opts.low_freq = 20.0
    opts.mel_opts.high_freq = -400.0
    fb = knf.OnlineFbank(opts)
    fb.accept_waveform(sr, mono)
    fb.input_finished()
    n_frames = fb.num_frames_ready
    if n_frames <= 0:
        return None
    feats = np.asarray([fb.get_frame(i) for i in range(n_frames)], dtype=np.float32)
    feats -= feats.mean(axis=0, keepdims=True)
    return feats


_FIXED_GPU_BUCKETS = (140, 150, 180, 220, 260, 320, 400, 500, 650, 850)


def _pick_gpu_bucket(n_frames: int) -> int:
    if n_frames in (140, 150):
        return n_frames
    for b in _FIXED_GPU_BUCKETS:
        if n_frames <= b + 15:
            return b
    return _FIXED_GPU_BUCKETS[-1]


def _fit_fbank_to_bucket(feat: Any, target_t: int) -> Any:
    np = _require_numpy()
    cur_t = int(feat.shape[0])
    if cur_t == target_t:
        return feat
    if cur_t > target_t:
        off = (cur_t - target_t) // 2
        out = feat[off : off + target_t].copy()
    else:
        reps = (target_t + cur_t - 1) // cur_t
        out = np.tile(feat, (reps, 1))[:target_t].copy()
    out -= out.mean(axis=0, keepdims=True)
    return out


class _OnnxBatchExtractor:
    """基于 onnxruntime-gpu + kaldi-native-fbank 的高精度 GPU 批量声纹提取器（与 sherpa-onnx 同分布）."""

    def __init__(self, model_path: Path, *, device: str = "auto", num_threads: int = 4) -> None:
        register_windows_cuda_dlls()
        import kaldi_native_fbank  # noqa: F401
        import onnxruntime as ort  # type: ignore

        if hasattr(ort, "preload_dlls") and device != "cpu":
            try:
                ort.preload_dlls(cuda=True, cudnn=True)
            except Exception:
                pass

        sess_opts = ort.SessionOptions()
        sess_opts.log_severity_level = 3
        sess_opts.intra_op_num_threads = max(1, int(num_threads))
        providers: List[Any] = (
            ["CPUExecutionProvider"]
            if device == "cpu"
            else [
                ("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}),
                "CPUExecutionProvider",
            ]
        )
        self.session = ort.InferenceSession(str(model_path), sess_options=sess_opts, providers=providers)
        self.active_providers = list(self.session.get_providers())
        self.using_cuda = "CUDAExecutionProvider" in self.active_providers
        self.input_name = self.session.get_inputs()[0].name

    def compute_batch_from_fbanks(self, fbanks: Sequence[Any], *, batch_size: int = 64) -> Any:
        """按固定长度桶 + 固定 batch 尺寸批量送入 GPU（避免 cuDNN 动态 shape 反复重编译，速度提升 30x）."""
        np = _require_numpy()
        n = len(fbanks)
        if n == 0:
            return np.zeros((0, 192), dtype=np.float32)

        buckets: Dict[int, List[int]] = defaultdict(list)
        for idx, feat in enumerate(fbanks):
            buckets[_pick_gpu_bucket(int(feat.shape[0]))].append(idx)

        out = np.zeros((n, 192), dtype=np.float32)
        bs = max(1, int(batch_size))
        for t_len, idxs in buckets.items():
            for off in range(0, len(idxs), bs):
                chunk_idx = idxs[off : off + bs]
                fitted = [_fit_fbank_to_bucket(fbanks[i], t_len) for i in chunk_idx]
                real_k = len(fitted)
                if self.using_cuda and real_k < bs:
                    # 补齐固定 batch_size 避免尾批触发 cuDNN 新 shape 重建
                    fitted.extend([fitted[-1]] * (bs - real_k))
                x = np.stack(fitted, axis=0)
                emb = self.session.run(None, {self.input_name: x})[0][:real_k].astype(np.float32)
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                emb = emb / np.maximum(norms, 1e-8)
                out[chunk_idx] = emb
        return out

    def compute_single(self, mono: Any, sr: int = 16000) -> Any:
        feat = _extract_kaldi_fbank(mono, sr=sr)
        if feat is None:
            return None
        return self.compute_batch_from_fbanks([feat], batch_size=1)[0]


def _create_sv_engine(
    model_path: Path,
    *,
    device: str = "auto",
    num_threads: int = 4,
    progress_cb: Optional[ProgressCallback] = None,
) -> Tuple[str, Any]:
    """优先创建 GPU 批量提取器；若无 CUDA 或缺少可选包则自动回退 sherpa-onnx CPU."""
    if device in ("auto", "cuda"):
        try:
            eng = _OnnxBatchExtractor(model_path, device=device, num_threads=num_threads)
            if eng.using_cuda:
                if progress_cb:
                    progress_cb("声纹引擎已就绪: onnxruntime-gpu (CUDAExecutionProvider, 批量 GPU 推理)")
                return "onnx_gpu", eng
            if device == "cuda":
                raise RuntimeError("指定了 --device cuda，但 onnxruntime 未能启用 CUDAExecutionProvider")
        except Exception as exc:
            if device == "cuda":
                raise
            if progress_cb:
                progress_cb(f"未启用 CUDA 批量加速（{exc}），自动回退 CPU 模式...")

    _, sherpa_onnx = _require_sv_deps()
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model_path),
        num_threads=max(1, int(num_threads)),
        debug=False,
        provider="cpu",
    )
    if progress_cb:
        progress_cb(f"声纹引擎已就绪: sherpa-onnx (CPU, threads={num_threads})")
    return "sherpa_cpu", sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


def _find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError("未找到 ffmpeg 可执行文件，请确认 ffmpeg 已加入系统 PATH。")
    return found


def ensure_campplus_model(
    model_path: Path = DEFAULT_MODEL_PATH,
    *,
    progress_cb: Optional[ProgressCallback] = None,
) -> Path:
    """确认本地存在 CAM++ ONNX 模型，缺失时自动从官方 Release（含镜像兜底）下载."""
    env_override = os.getenv("ZJ_SV_MODEL_PATH", "").strip()
    if env_override:
        p = Path(env_override)
        if p.exists():
            return p
    if model_path.exists() and model_path.stat().st_size > 10_000_000:
        return model_path

    model_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = model_path.with_suffix(".onnx.part")
    last_err: Optional[Exception] = None
    for url in DEFAULT_MODEL_URLS:
        try:
            if progress_cb:
                progress_cb(f"正在下载 CAM++ 声纹模型: {url}")
            with httpx.stream("GET", url, timeout=120.0, follow_redirects=True) as resp:
                resp.raise_for_status()
                with tmp_path.open("wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=131072):
                        if chunk:
                            f.write(chunk)
            if tmp_path.exists() and tmp_path.stat().st_size > 10_000_000:
                tmp_path.replace(model_path)
                return model_path
        except Exception as exc:
            last_err = exc
            tmp_path.unlink(missing_ok=True)
    raise RuntimeError(f"无法下载 CAM++ 声纹模型到 {model_path}: {last_err}")


def _create_extractor(model_path: Path, num_threads: int = 4):
    _, sherpa_onnx = _require_sv_deps()
    cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=str(model_path),
        num_threads=max(1, int(num_threads)),
        debug=False,
        provider="cpu",
    )
    return sherpa_onnx.SpeakerEmbeddingExtractor(cfg)


def _compute_embedding(extractor: Any, mono: Any, sr: int):
    np, _ = _require_sv_deps()
    if len(mono) < int(0.4 * sr):
        return None
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=sr, waveform=mono.tolist())
    stream.input_finished()
    if not extractor.is_ready(stream):
        return None
    emb = np.array(extractor.compute(stream), dtype=np.float32)
    norm = float(np.linalg.norm(emb))
    if norm < 1e-8:
        return None
    return emb / norm


def _slice_mono(mono: Any, sr: int, rel_s: float, rel_e: float, min_dur: float = 1.4):
    total_dur = len(mono) / sr
    s = max(0.0, rel_s)
    e = min(total_dur, rel_e)
    raw_seg = mono[int(s * sr) : int(e * sr)]
    dur = e - s
    if dur < min_dur:
        pad = 0.5 * (min_dur - dur)
        s_p = max(0.0, s - pad)
        e_p = min(total_dur, e + pad)
        emb_seg = mono[int(s_p * sr) : int(e_p * sr)]
    else:
        emb_seg = raw_seg
    return raw_seg, emb_seg


def _load_mono_wav(wav_path: Path):
    np, _ = _require_sv_deps()
    with wave.open(str(wav_path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        if sr != 16000 or ch != 1:
            raise RuntimeError(f"WAV 格式不符，期望 16kHz 单声道，实际 sr={sr}, ch={ch}")
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return arr, sr


def _extract_bili_audio_wav(
    bili: BilibiliClient,
    bvid: str,
    cid: int,
    out_wav: Path,
    *,
    start_sec: Optional[float] = None,
    end_sec: Optional[float] = None,
) -> Path:
    """下载 B 站纯音频 DASH 流并转为 16kHz 单声道 WAV（指定时间段时优先远端快速 seek）."""
    ffmpeg = _find_ffmpeg()
    urls = bili.get_audio_stream_urls(bvid, cid)
    if not urls:
        raise RuntimeError(f"未获取到音频流 URL: {bvid} (cid={cid})")

    if start_sec is not None and end_sec is not None and end_sec > start_sec:
        dur = end_sec - start_sec
        headers = (
            f"User-Agent: {DEFAULT_UA}\r\n"
            f"Referer: https://www.bilibili.com/video/{bvid}/\r\n"
            "Origin: https://www.bilibili.com\r\n"
        )
        for url in urls:
            cmd = [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-headers",
                headers,
                "-ss",
                f"{start_sec:.2f}",
                "-t",
                f"{dur:.2f}",
                "-i",
                url,
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(out_wav),
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode == 0 and out_wav.exists() and out_wav.stat().st_size > 16000:
                return out_wav

    tmp_raw = out_wav.with_suffix(".m4a")
    try:
        bili.download_stream_file(urls, tmp_raw, bvid=bvid)
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        if start_sec is not None and end_sec is not None and end_sec > start_sec:
            cmd.extend(["-ss", f"{start_sec:.2f}", "-t", f"{end_sec - start_sec:.2f}"])
        cmd.extend(
            [
                "-i",
                str(tmp_raw),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(out_wav),
            ]
        )
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg 音频转码失败: {proc.stderr.strip()}")
        return out_wav
    finally:
        tmp_raw.unlink(missing_ok=True)


def enroll_member_voiceprint(
    db: CorpusDB,
    cfg: AppConfig,
    up_name: str,
    bvid: str,
    page: int,
    start_sec: float,
    end_sec: float,
    *,
    voiceprints_path: Path = DEFAULT_VOICEPRINTS_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    num_threads: int = 4,
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """用指定成员的一段干净单播片段提取并更新其 CAM++ 声纹模板."""
    np, _ = _require_sv_deps()
    resolved_up = cfg.resolve_up_name(raw_name=up_name)
    model_file = ensure_campplus_model(model_path, progress_cb=progress_cb)
    extractor = _create_extractor(model_file, num_threads=num_threads)

    with BilibiliClient(
        cookie=cfg.bilibili_cookie,
        delay=cfg.request_delay,
        jitter=cfg.request_jitter,
        max_retries=cfg.max_retries,
    ) as bili:
        vinfo = bili.get_video_info(bvid)
        pinfo = next((p for p in vinfo.pages if p.page == page), None)
        if pinfo is None:
            raise ValueError(f"{bvid} 不存在分 P {page}")

        # 优先读取本地库字幕，不存在则直接拉取 B 站字幕轨
        subs: List[Tuple[float, float]] = []
        if db.db_path.exists():
            db.init_db()
            with db.session() as conn:
                rows = conn.execute(
                    """
                    SELECT start_sec, end_sec FROM subtitles
                     WHERE bvid = ? AND page = ? AND start_sec >= ? AND end_sec <= ?
                     ORDER BY start_sec ASC
                    """,
                    (bvid, page, start_sec, end_sec),
                ).fetchall()
                subs = [(float(r["start_sec"]), float(r["end_sec"])) for r in rows]

        if not subs:
            tracks = bili.list_subtitle_tracks(bvid, pinfo.cid)
            chosen = bili.pick_subtitle_track(tracks)
            if chosen is None:
                raise RuntimeError(f"{bvid} P{page} 无本地字幕且无 B 站字幕轨，无法切分参考句")
            cleaned = clean_bcc_payload(
                bili.fetch_subtitle_json(chosen.subtitle_url),
                up_name=resolved_up,
                pub_date=vinfo.pub_date,
            )
            subs = [
                (it.start_sec, it.end_sec)
                for it in cleaned
                if it.start_sec >= start_sec and it.end_sec <= end_sec
            ]

        if len(subs) < 5:
            raise RuntimeError(f"{bvid} P{page} [{start_sec:.0f}s~{end_sec:.0f}s] 有效字幕不足 5 条")

        with tempfile.TemporaryDirectory(prefix="sv_enroll_") as td:
            wav_path = Path(td) / "clean.wav"
            if progress_cb:
                progress_cb(
                    f"提取参考音频: {resolved_up} {bvid} P{page} ({start_sec:.0f}s~{end_sec:.0f}s)..."
                )
            _extract_bili_audio_wav(
                bili, bvid, pinfo.cid, wav_path, start_sec=start_sec, end_sec=end_sec
            )
            mono, sr = _load_mono_wav(wav_path)

    embs: List[Any] = []
    durs: List[float] = []
    rmss: List[float] = []
    for s, e in subs:
        raw_seg, emb_seg = _slice_mono(mono, sr, s - start_sec, e - start_sec)
        if len(raw_seg) == 0:
            continue
        rms = float(np.sqrt(np.mean(raw_seg * raw_seg) + 1e-12))
        emb = _compute_embedding(extractor, emb_seg, sr)
        if emb is not None:
            embs.append(emb)
            durs.append(e - s)
            rmss.append(rms)

    if len(embs) < 5:
        raise RuntimeError(f"可提取声纹的有效片段不足 5 条（实际 {len(embs)} 条）")

    mat = np.stack(embs, axis=0)
    durs_arr = np.array(durs)
    rmss_arr = np.array(rmss)
    mask = (durs_arr >= 1.8) & (rmss_arr >= 0.015)
    ref_mat = mat[mask] if int(mask.sum()) >= 20 else mat
    mean_v = ref_mat.mean(axis=0)
    mean_v /= np.linalg.norm(mean_v) + 1e-8
    sims = ref_mat @ mean_v
    cutoff = float(np.percentile(sims, 20))
    core = ref_mat[sims >= cutoff]
    centroid = core.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-8

    self_sims = mat @ centroid
    p50 = round(float(np.percentile(self_sims, 50)), 4)
    p10 = round(float(np.percentile(self_sims, 10)), 4)

    vp_data = load_voiceprints(voiceprints_path)
    vp_data["model"] = model_file.name
    vp_data["dim"] = int(centroid.shape[0])
    vp_data["members"][resolved_up] = {
        "bvid": bvid,
        "page": int(page),
        "start_sec": round(float(start_sec), 2),
        "end_sec": round(float(end_sec), 2),
        "ref_count": int(len(core)),
        "clean_p50": p50,
        "clean_p10": p10,
        "embedding": [round(float(x), 6) for x in centroid.tolist()],
    }
    save_voiceprints(vp_data, voiceprints_path)

    return {
        "status": "ok",
        "up_name": resolved_up,
        "bvid": bvid,
        "page": page,
        "lines_used": len(embs),
        "core_count": int(len(core)),
        "clean_p50": p50,
        "clean_p10": p10,
        "voiceprints_path": str(voiceprints_path),
    }


@dataclass
class VoiceprintScoreSummary:
    parts_selected: int = 0
    parts_scored: int = 0
    parts_skipped: int = 0
    parts_failed: int = 0
    lines_total: int = 0
    lines_host: int = 0
    lines_mixed: int = 0
    lines_external: int = 0
    elapsed_sec: float = 0.0
    errors: List[str] = field(default_factory=list)


def list_voiceprint_targets(
    db: CorpusDB,
    *,
    bvid: Optional[str] = None,
    page: Optional[int] = None,
    up_name: Optional[str] = None,
    re_only: bool = False,
    force: bool = False,
    limit: Optional[int] = 10,
) -> List[Dict[str, Any]]:
    """查询待执行声纹打分的已入库视频分 P."""
    db.init_db()
    clauses = ["subtitle_status = 'ok'", "line_count > 0"]
    params: List[Any] = []
    if bvid:
        clauses.append("bvid = ?")
        params.append(bvid.strip())
    if page is not None:
        clauses.append("page = ?")
        params.append(int(page))
    if up_name:
        clauses.append("up_name = ?")
        params.append(up_name.strip())
    if not force:
        clauses.append("(voiceprint_status IS NULL OR voiceprint_status = '' OR voiceprint_status = 'none')")

    with db.session() as conn:
        rows = conn.execute(
            f"""
            SELECT bvid, page, cid, up_name, title, part_title, pub_date,
                   duration_sec, line_count, voiceprint_status
              FROM videos
             WHERE {' AND '.join(clauses)}
             ORDER BY pub_date DESC, bvid DESC, page ASC
            """,
            params,
        ).fetchall()

    targets = [dict(r) for r in rows]
    if re_only and not bvid:
        targets = [t for t in targets if is_re_watch_title(str(t.get("title") or ""))]
    if limit is not None and limit > 0:
        targets = targets[:limit]
    return targets


def _score_prefetched_part(
    db: CorpusDB,
    bvid: str,
    page: int,
    up_name: str,
    sub_rows: Sequence[Any],
    *,
    engine_kind: str,
    engine: Any,
    vp_mat: Any,
    member_names: Sequence[str],
    thrs: Dict[str, Any],
    batch_size: int = 64,
    precomputed_pack: Optional[Dict[str, Any]] = None,
    mono: Any = None,
    sr: int = 16000,
    elapsed_dl: float = 0.0,
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """执行单分 P 的批量声纹推理、三段式分类与原子落库."""
    np = _require_numpy()
    t_inf = time.perf_counter()
    scored_items: List[Dict[str, Any]] = []

    if engine_kind == "onnx_gpu":
        if precomputed_pack is None:
            precomputed_pack = _prepare_part_fbanks(mono, sr, sub_rows)
        whole_fbanks: List[Any] = precomputed_pack["whole_fbanks"]
        whole_meta: List[Tuple[int, float, float]] = precomputed_pack["whole_meta"]
        skip_meta: List[Tuple[int, float, float]] = precomputed_pack["skip_meta"]
        win_fbanks: List[Any] = precomputed_pack["win_fbanks"]
        win_owners: List[int] = precomputed_pack["win_owners"]

        whole_embs = engine.compute_batch_from_fbanks(whole_fbanks, batch_size=batch_size)
        win_embs = (
            engine.compute_batch_from_fbanks(win_fbanks, batch_size=batch_size)
            if win_fbanks
            else np.zeros((0, 192), dtype=np.float32)
        )

        # 向量化四成员点积
        whole_sims = whole_embs @ vp_mat.T if len(whole_embs) else np.zeros((0, len(member_names)), dtype=np.float32)
        win_sims = win_embs @ vp_mat.T if len(win_embs) else np.zeros((0, len(member_names)), dtype=np.float32)

        # 聚合每个整句的滑窗峰值
        peak_by_owner: Dict[int, Tuple[float, str]] = {}
        if len(win_sims):
            w_best_idxs = np.argmax(win_sims, axis=1)
            w_best_vals = win_sims[np.arange(len(win_sims)), w_best_idxs]
            for owner_idx, val, m_idx in zip(win_owners, w_best_vals.tolist(), w_best_idxs.tolist()):
                prev = peak_by_owner.get(owner_idx)
                if prev is None or val > prev[0]:
                    peak_by_owner[owner_idx] = (val, member_names[m_idx])

        item_by_sid: Dict[int, Dict[str, Any]] = {}
        for pos, (sid, s, e) in enumerate(whole_meta):
            sims_row = whole_sims[pos]
            best_idx = int(np.argmax(sims_row))
            sim_mean = float(sims_row[best_idx])
            best_member = member_names[best_idx]
            member_sims_map = {member_names[j]: float(sims_row[j]) for j in range(len(member_names))}
            sim_peak = sim_mean
            peak_member = best_member
            pk = peak_by_owner.get(pos)
            if pk is not None and pk[0] > sim_peak:
                sim_peak, peak_member = pk
            item_by_sid[sid] = {
                "id": sid,
                "start_sec": s,
                "end_sec": e,
                "sim_mean": round(sim_mean, 4),
                "sim_peak": round(sim_peak, 4),
                "best_member": best_member,
                "peak_member": peak_member,
                "member_sims": member_sims_map,
                "skip": False,
            }
        for sid, s, e in skip_meta:
            item_by_sid[sid] = {
                "id": sid,
                "start_sec": s,
                "end_sec": e,
                "sim_mean": 0.0,
                "sim_peak": 0.0,
                "best_member": "",
                "peak_member": "",
                "skip": True,
            }
        scored_items = [item_by_sid[int(r["id"])] for r in sub_rows]
    else:
        win_n = int(1.5 * sr)
        hop_n = int(0.5 * sr)
        for r in sub_rows:
            sid = int(r["id"])
            s = float(r["start_sec"])
            e = float(r["end_sec"])
            raw_seg, emb_seg = _slice_mono(mono, sr, s, e, min_dur=1.4)
            emb = _compute_embedding(engine, emb_seg, sr) if len(emb_seg) > 0 else None
            if emb is None:
                scored_items.append(
                    {
                        "id": sid,
                        "start_sec": s,
                        "end_sec": e,
                        "sim_mean": 0.0,
                        "sim_peak": 0.0,
                        "best_member": "",
                        "peak_member": "",
                        "skip": True,
                    }
                )
                continue
            sims_all = emb @ vp_mat.T
            best_idx = int(np.argmax(sims_all))
            sim_mean = float(sims_all[best_idx])
            best_member = member_names[best_idx]
            member_sims_map = {member_names[j]: float(sims_all[j]) for j in range(len(member_names))}
            sim_peak = sim_mean
            peak_member = best_member
            dur = e - s
            if dur >= 2.0 and len(raw_seg) >= win_n:
                for off in range(0, len(raw_seg) - win_n + 1, hop_n):
                    w_emb = _compute_embedding(engine, raw_seg[off : off + win_n], sr)
                    if w_emb is not None:
                        w_sims = w_emb @ vp_mat.T
                        w_idx = int(np.argmax(w_sims))
                        w_val = float(w_sims[w_idx])
                        if w_val > sim_peak:
                            sim_peak = w_val
                            peak_member = member_names[w_idx]
            scored_items.append(
                {
                    "id": sid,
                    "start_sec": s,
                    "end_sec": e,
                    "sim_mean": round(sim_mean, 4),
                    "sim_peak": round(sim_peak, 4),
                    "best_member": best_member,
                    "peak_member": peak_member,
                    "member_sims": member_sims_map,
                    "skip": False,
                }
            )

    valid_items = [x for x in scored_items if not x["skip"]]
    classify_utterance_scores(
        valid_items,
        host_thr=float(thrs.get("host_thr", DEFAULT_THRESHOLDS["host_thr"])),
        peak_thr=float(thrs.get("peak_thr", DEFAULT_THRESHOLDS["peak_thr"])),
        smooth_floor=float(thrs.get("smooth_floor", DEFAULT_THRESHOLDS["smooth_floor"])),
        strong_neighbor=float(thrs.get("strong_neighbor", DEFAULT_THRESHOLDS["strong_neighbor"])),
        neighbor_gap=float(thrs.get("neighbor_gap", DEFAULT_THRESHOLDS["neighbor_gap"])),
    )

    updates: List[Tuple[int, str, Optional[float], str]] = []
    n_host = 0
    n_mixed = 0
    n_ext = 0
    by_member: Dict[str, int] = {}
    for it in scored_items:
        if it["skip"]:
            updates.append((int(it["id"]), "unknown", None, ""))
            continue
        lbl = str(it["speaker_label"])
        spk_name = str(it.get("speaker_name") or "")
        if lbl == "host":
            n_host += 1
            if spk_name:
                by_member[spk_name] = by_member.get(spk_name, 0) + 1
        elif lbl == "mixed":
            n_mixed += 1
            if spk_name:
                by_member[spk_name] = by_member.get(spk_name, 0) + 1
        else:
            n_ext += 1
        sim_out = float(it["sim_peak"]) if lbl == "mixed" else float(it["sim_mean"])
        updates.append((int(it["id"]), lbl, sim_out, spk_name))

    db.update_part_voiceprint_scores(bvid, page, updates, status="done")
    elapsed_inf = round(time.perf_counter() - t_inf, 2)
    if progress_cb:
        m_str = ", ".join(f"{k}={v}" for k, v in sorted(by_member.items(), key=lambda kv: kv[1], reverse=True))
        progress_cb(
            f"[OK] {up_name} {bvid} P{page}: host={n_host}, mixed={n_mixed}, external={n_ext} "
            f"({m_str or '无成员命中'} | 共 {len(sub_rows)} 条, 预取 {elapsed_dl:.1f}s, 推理+落库 {elapsed_inf:.2f}s)"
        )
    return {
        "bvid": bvid,
        "page": page,
        "up_name": up_name,
        "total": len(sub_rows),
        "host": n_host,
        "mixed": n_mixed,
        "external": n_ext,
        "by_member": by_member,
        "elapsed_sec": round(elapsed_dl + elapsed_inf, 2),
    }


def _prepare_part_fbanks(mono: Any, sr: int, sub_rows: Sequence[Any]) -> Dict[str, Any]:
    """在预取线程中并行提取单分 P 全部字幕与滑窗的 Kaldi 80 维 Fbank，随即释放大音频数组."""
    win_n = int(1.5 * sr)
    hop_n = int(0.5 * sr)
    whole_fbanks: List[Any] = []
    whole_meta: List[Tuple[int, float, float]] = []
    skip_meta: List[Tuple[int, float, float]] = []
    win_fbanks: List[Any] = []
    win_owners: List[int] = []

    for r in sub_rows:
        sid = int(r["id"])
        s = float(r["start_sec"])
        e = float(r["end_sec"])
        raw_seg, emb_seg = _slice_mono(mono, sr, s, e, min_dur=1.4)
        feat = _extract_kaldi_fbank(emb_seg, sr=sr) if len(emb_seg) > 0 else None
        if feat is None:
            skip_meta.append((sid, s, e))
            continue
        owner_pos = len(whole_fbanks)
        whole_fbanks.append(feat)
        whole_meta.append((sid, s, e))
        dur = e - s
        if dur >= 2.0 and len(raw_seg) >= win_n:
            for off in range(0, len(raw_seg) - win_n + 1, hop_n):
                wf = _extract_kaldi_fbank(raw_seg[off : off + win_n], sr=sr)
                if wf is not None:
                    win_fbanks.append(wf)
                    win_owners.append(owner_pos)

    return {
        "whole_fbanks": whole_fbanks,
        "whole_meta": whole_meta,
        "skip_meta": skip_meta,
        "win_fbanks": win_fbanks,
        "win_owners": win_owners,
    }


def score_video_part(
    db: CorpusDB,
    cfg: AppConfig,
    bvid: str,
    page: int,
    *,
    extractor: Any,
    voiceprints_data: Dict[str, Any],
    bili: BilibiliClient,
    engine_kind: str = "sherpa_cpu",
    batch_size: int = 64,
    keep_audio: bool = False,
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """对单个分 P 的所有字幕执行四成员 CAM++ 声纹打分并写回 SQLite."""
    np = _require_numpy()
    vid = db.get_video(bvid, page)
    if not vid:
        raise ValueError(f"数据库中未找到视频记录: {bvid} P{page}")

    up_name = str(vid["up_name"])
    members_vp = voiceprints_data.get("members") or {}
    if not members_vp:
        raise ValueError("声纹库为空，请先执行 enroll-voiceprint")

    member_names = list(members_vp.keys())
    vp_mat = np.stack(
        [np.array(members_vp[m]["embedding"], dtype=np.float32) for m in member_names],
        axis=0,
    )
    vp_mat /= np.linalg.norm(vp_mat, axis=1, keepdims=True) + 1e-8
    thrs = voiceprints_data.get("thresholds") or DEFAULT_THRESHOLDS

    with db.session() as conn:
        sub_rows = conn.execute(
            """
            SELECT id, start_sec, end_sec, content
              FROM subtitles
             WHERE bvid = ? AND page = ?
             ORDER BY start_sec ASC, id ASC
            """,
            (bvid, page),
        ).fetchall()

    if not sub_rows:
        db.update_part_voiceprint_scores(bvid, page, [], status="done")
        return {"bvid": bvid, "page": page, "total": 0, "host": 0, "mixed": 0, "external": 0}

    cid = vid.get("cid")
    if not cid:
        vinfo = bili.get_video_info(bvid)
        pinfo = next((p for p in vinfo.pages if p.page == page), None)
        if not pinfo:
            raise ValueError(f"B 站未找到 {bvid} P{page}")
        cid = pinfo.cid

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="sv_score_") as td:
        wav_path = Path(td) / f"{bvid}_p{page}.wav"
        if progress_cb:
            progress_cb(
                f"下载音频并声纹打分: {up_name} {bvid} P{page} ({len(sub_rows)} 条字幕) | {str(vid.get('title') or '')[:28]}"
            )
        _extract_bili_audio_wav(bili, bvid, int(cid), wav_path)
        mono, sr = _load_mono_wav(wav_path)
        if keep_audio:
            save_copy = PROJECT_ROOT / "data" / "tmp_asr" / wav_path.name
            save_copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wav_path, save_copy)
    elapsed_dl = time.perf_counter() - t0

    return _score_prefetched_part(
        db,
        bvid,
        page,
        up_name,
        sub_rows,
        engine_kind=engine_kind,
        engine=extractor,
        vp_mat=vp_mat,
        member_names=member_names,
        thrs=thrs,
        batch_size=batch_size,
        mono=mono,
        sr=sr,
        elapsed_dl=elapsed_dl,
        progress_cb=progress_cb,
    )


def run_voiceprint_scoring(
    db: CorpusDB,
    cfg: AppConfig,
    *,
    bvid: Optional[str] = None,
    page: Optional[int] = None,
    up_name: Optional[str] = None,
    re_only: bool = False,
    force: bool = False,
    limit: Optional[int] = 10,
    voiceprints_path: Path = DEFAULT_VOICEPRINTS_PATH,
    model_path: Path = DEFAULT_MODEL_PATH,
    device: str = "auto",
    batch_size: int = 64,
    prefetch: int = 3,
    shard_index: int = 0,
    shard_total: int = 1,
    num_threads: int = 4,
    keep_audio: bool = False,
    progress_cb: Optional[ProgressCallback] = None,
) -> VoiceprintScoreSummary:
    """批量对指定视频分 P 执行离线四成员声纹打分（支持 GPU 批量推理、多路预取与随时 Ctrl+C 断点续跑）."""
    np = _require_numpy()
    t_all = time.perf_counter()
    summary = VoiceprintScoreSummary()
    resolved_up = cfg.resolve_up_name(raw_name=up_name) if up_name else None
    targets = list_voiceprint_targets(
        db,
        bvid=bvid,
        page=page,
        up_name=resolved_up,
        re_only=re_only,
        force=force,
        limit=limit,
    )
    if shard_total > 1:
        targets = [t for idx, t in enumerate(targets) if idx % shard_total == shard_index]
    summary.parts_selected = len(targets)
    if not targets:
        if progress_cb:
            progress_cb("没有待声纹打分的目标分 P。")
        summary.elapsed_sec = round(time.perf_counter() - t_all, 2)
        return summary

    vp_data = load_voiceprints(voiceprints_path)
    members_vp = vp_data.get("members") or {}
    if not members_vp:
        raise RuntimeError(f"声纹模板为空: {voiceprints_path}，请先运行 enroll-voiceprint")

    member_names = list(members_vp.keys())
    vp_mat = np.stack(
        [np.array(members_vp[m]["embedding"], dtype=np.float32) for m in member_names],
        axis=0,
    )
    vp_mat /= np.linalg.norm(vp_mat, axis=1, keepdims=True) + 1e-8
    thrs = vp_data.get("thresholds") or DEFAULT_THRESHOLDS

    model_file = ensure_campplus_model(model_path, progress_cb=progress_cb)
    engine_kind, engine = _create_sv_engine(
        model_file, device=device, num_threads=num_threads, progress_cb=progress_cb
    )

    n_prefetch = max(1, min(int(prefetch), len(targets), 8))
    if progress_cb:
        progress_cb(
            f"共选中 {len(targets)} 个待打分分 P | 引擎={engine_kind}, batch={batch_size}, prefetch={n_prefetch}"
        )

    # 生产者-消费者流水线：预取线程负责下载+转码+提 Fbank 并立刻删临时音频；主线程独占 GPU 批推理与 SQLite 原子落库
    work_q: queue.Queue[Optional[Tuple[int, Dict[str, Any]]]] = queue.Queue()
    for idx, t in enumerate(targets, start=1):
        work_q.put((idx, t))
    for _ in range(n_prefetch):
        work_q.put(None)

    ready_q: queue.Queue[Dict[str, Any]] = queue.Queue(maxsize=max(2, n_prefetch + 1))
    stop_event = threading.Event()
    active_workers = [n_prefetch]
    workers_lock = threading.Lock()

    def _prefetch_worker() -> None:
        bili = BilibiliClient(
            cookie=cfg.bilibili_cookie,
            delay=cfg.request_delay,
            jitter=cfg.request_jitter,
            max_retries=cfg.max_retries,
        )
        try:
            while not stop_event.is_set():
                try:
                    item = work_q.get_nowait()
                except queue.Empty:
                    break
                if item is None:
                    break
                idx, t = item
                t_bvid = str(t["bvid"])
                t_page = int(t["page"])
                t_up = str(t["up_name"])
                try:
                    t0 = time.perf_counter()
                    with db.session() as conn:
                        sub_rows = conn.execute(
                            """
                            SELECT id, start_sec, end_sec, content
                              FROM subtitles
                             WHERE bvid = ? AND page = ?
                             ORDER BY start_sec ASC, id ASC
                            """,
                            (t_bvid, t_page),
                        ).fetchall()
                    if not sub_rows:
                        ready_q.put(
                            {
                                "kind": "empty",
                                "idx": idx,
                                "bvid": t_bvid,
                                "page": t_page,
                                "up_name": t_up,
                            }
                        )
                        continue

                    cid = t.get("cid")
                    if not cid:
                        vinfo = bili.get_video_info(t_bvid)
                        pinfo = next((p for p in vinfo.pages if p.page == t_page), None)
                        if not pinfo:
                            raise ValueError(f"B 站未找到 {t_bvid} P{t_page}")
                        cid = pinfo.cid

                    with tempfile.TemporaryDirectory(prefix="sv_score_") as td:
                        wav_path = Path(td) / f"{t_bvid}_p{t_page}.wav"
                        _extract_bili_audio_wav(bili, t_bvid, int(cid), wav_path)
                        mono, sr = _load_mono_wav(wav_path)
                        if keep_audio:
                            save_copy = PROJECT_ROOT / "data" / "tmp_asr" / wav_path.name
                            save_copy.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(wav_path, save_copy)

                    pack = _prepare_part_fbanks(mono, sr, sub_rows) if engine_kind == "onnx_gpu" else None
                    payload: Dict[str, Any] = {
                        "kind": "ready",
                        "idx": idx,
                        "bvid": t_bvid,
                        "page": t_page,
                        "up_name": t_up,
                        "title": str(t.get("title") or ""),
                        "sub_rows": sub_rows,
                        "pack": pack,
                        "mono": None if engine_kind == "onnx_gpu" else mono,
                        "sr": sr,
                        "elapsed_dl": time.perf_counter() - t0,
                    }
                    while not stop_event.is_set():
                        try:
                            ready_q.put(payload, timeout=0.5)
                            break
                        except queue.Full:
                            continue
                except Exception as exc:
                    if not stop_event.is_set():
                        ready_q.put(
                            {
                                "kind": "error",
                                "idx": idx,
                                "bvid": t_bvid,
                                "page": t_page,
                                "error": f"{t_bvid} P{t_page}: {exc}",
                            }
                        )
        finally:
            bili.close()
            with workers_lock:
                active_workers[0] -= 1
                if active_workers[0] == 0:
                    ready_q.put({"kind": "done"})

    threads = [
        threading.Thread(target=_prefetch_worker, name=f"sv-prefetch-{i}", daemon=True)
        for i in range(n_prefetch)
    ]
    for th in threads:
        th.start()

    pause_flag = PROJECT_ROOT / "data" / "tmp_asr" / "PAUSE_SV"
    pause_flag.unlink(missing_ok=True)
    try:
        while True:
            if pause_flag.exists():
                stop_event.set()
                pause_flag.unlink(missing_ok=True)
                if progress_cb:
                    progress_cb(
                        f"\n[PAUSED] 收到暂停标记，已安全落库 {summary.parts_scored}/{len(targets)} 个分 P，下次重跑自动续接。"
                    )
                break
            msg = ready_q.get()
            kind = msg["kind"]
            if kind == "done":
                break
            if kind == "error":
                summary.parts_failed += 1
                summary.errors.append(msg["error"])
                if progress_cb:
                    progress_cb(f"[ERROR {msg['idx']}/{len(targets)}] {msg['error']}")
                continue
            if kind == "empty":
                db.update_part_voiceprint_scores(msg["bvid"], msg["page"], [], status="done")
                summary.parts_scored += 1
                continue

            res = _score_prefetched_part(
                db,
                msg["bvid"],
                msg["page"],
                msg["up_name"],
                msg["sub_rows"],
                engine_kind=engine_kind,
                engine=engine,
                vp_mat=vp_mat,
                member_names=member_names,
                thrs=thrs,
                batch_size=batch_size,
                precomputed_pack=msg["pack"],
                mono=msg["mono"],
                sr=msg["sr"],
                elapsed_dl=msg["elapsed_dl"],
                progress_cb=lambda line, i=msg["idx"]: progress_cb(f"[{i}/{len(targets)}] {line}") if progress_cb else None,
            )
            summary.parts_scored += 1
            summary.lines_total += int(res["total"])
            summary.lines_host += int(res["host"])
            summary.lines_mixed += int(res["mixed"])
            summary.lines_external += int(res["external"])
    except KeyboardInterrupt:
        stop_event.set()
        if progress_cb:
            progress_cb(
                f"\n[PAUSED] 收到中断信号，已安全落库 {summary.parts_scored}/{len(targets)} 个分 P，下次重跑将自动断点续接。"
            )
        raise
    finally:
        stop_event.set()
        for th in threads:
            th.join(timeout=3.0)

    summary.elapsed_sec = round(time.perf_counter() - t_all, 2)
    return summary
