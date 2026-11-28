"""本地部署的 Qwen3-ASR-1.7B（含 Qwen3-ForcedAligner-0.6B）评测与推理脚本.

用法（使用独立环境 .venv-qwen-asr）：
  1) 跑完 8 段真实直播回放评测（对比 B 站 ai-zh 与本地 faster-whisper）：
     .\.venv-qwen-asr\Scripts\python.exe scripts\eval_qwen3_asr.py
  2) 转写任意本地音频文件：
     .\.venv-qwen-asr\Scripts\python.exe scripts\eval_qwen3_asr.py --audio path/to/file.wav --timestamps
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import wave
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import zhconv
from qwen_asr import Qwen3ASRModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASR_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "Qwen3-ASR-1.7B"
ALIGNER_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "Qwen3-ForcedAligner-0.6B"
EVAL_DIR = PROJECT_ROOT / "data" / "tmp_asr" / "qwen3_eval"

SIXIWANZI_CONTEXT = (
    "四禧丸子 沐霂 又一 恬豆 梨安 小沐标 酥酥又 恬豆包 向心梨 "
    "大沐博士 王又一 又神 陈恬豆 肥梨 大梨老师 虚环 GTA6 B站 电池 猫猫币 弹幕 切片"
)

_PUNCT_RE = re.compile(r"[\s\.,!?~`@#$%^&*()_\-+=\[\]{};:'\"<>/\\|，。！？、；：（）【】《》“”‘’…—·♪♫～]+")


def normalize_for_diff(text: str) -> str:
    t = zhconv.convert(str(text or ""), "zh-cn")
    return _PUNCT_RE.sub("", t).lower()


def load_wav_16k(path: str | Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1)
    return arr, sr


def serialize_timestamps(ts_obj: Any) -> List[Dict[str, Any]]:
    if not ts_obj:
        return []
    out: List[Dict[str, Any]] = []
    items = getattr(ts_obj, "items", ts_obj)
    for it in items:
        if isinstance(it, dict):
            out.append(it)
        else:
            out.append(
                {
                    "text": getattr(it, "text", str(it)),
                    "start_time": round(float(getattr(it, "start_time", 0.0)), 3),
                    "end_time": round(float(getattr(it, "end_time", 0.0)), 3),
                }
            )
    return out


def load_qwen3_model(*, with_aligner: bool = True) -> Qwen3ASRModel:
    asr_path = str(ASR_MODEL_DIR) if ASR_MODEL_DIR.exists() else "Qwen/Qwen3-ASR-1.7B"
    kwargs: Dict[str, Any] = dict(
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=16,
        max_new_tokens=2048,
    )
    if with_aligner:
        align_path = str(ALIGNER_MODEL_DIR) if ALIGNER_MODEL_DIR.exists() else "Qwen/Qwen3-ForcedAligner-0.6B"
        kwargs["forced_aligner"] = align_path
        kwargs["forced_aligner_kwargs"] = dict(dtype=torch.bfloat16, device_map="cuda:0")
    return Qwen3ASRModel.from_pretrained(asr_path, **kwargs)


def run_single_audio(audio_path: str, *, use_context: bool, timestamps: bool) -> None:
    t0 = time.perf_counter()
    model = load_qwen3_model(with_aligner=timestamps)
    print(f"[Loaded] {time.perf_counter() - t0:.2f}s | VRAM={torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    t1 = time.perf_counter()
    res = model.transcribe(
        audio=audio_path,
        context=SIXIWANZI_CONTEXT if use_context else "",
        language="Chinese",
        return_time_stamps=timestamps,
    )[0]
    dt = time.perf_counter() - t1
    print(f"[Done] {dt:.2f}s | Language={res.language}")
    print("Text:", res.text)
    if timestamps and getattr(res, "time_stamps", None):
        ts = serialize_timestamps(res.time_stamps)
        print(f"Timestamps ({len(ts)} items), first 10:", ts[:10])


def run_benchmark() -> None:
    clips_file = EVAL_DIR / "clips.json"
    if not clips_file.exists():
        raise FileNotFoundError(f"找不到评测片段清单: {clips_file}")
    clips = json.loads(clips_file.read_text(encoding="utf-8"))
    whisper_bench_file = EVAL_DIR / "whisper_bench.json"
    whisper_bench = (
        json.loads(whisper_bench_file.read_text(encoding="utf-8"))
        if whisper_bench_file.exists()
        else {"clips": {}}
    )

    torch.cuda.reset_peak_memory_stats()
    t_load = time.perf_counter()
    model = load_qwen3_model(with_aligner=True)
    load_sec = round(time.perf_counter() - t_load, 2)
    load_vram_gb = round(torch.cuda.memory_allocated() / (1024 ** 3), 2)
    print(f"QWEN3_LOAD_SEC: {load_sec} | LOAD_VRAM_GB: {load_vram_gb}")

    # Warmup (short 3s slice) so first clip timing isn't skewed by CUDA kernel warmup
    w_arr, w_sr = load_wav_16k(clips[0]["wav_path"])
    _ = model.transcribe(audio=(w_arr[: w_sr * 3], w_sr), language="Chinese", return_time_stamps=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    total_audio_sec = 0.0
    total_raw_sec = 0.0
    total_ctx_ts_sec = 0.0
    total_batch30_sec = 0.0
    eval_results: List[Dict[str, Any]] = []

    for clip in clips:
        cid = clip["id"]
        arr, sr = load_wav_16k(clip["wav_path"])
        dur = round(len(arr) / sr, 2)
        total_audio_sec += dur

        # 1) Full 90s clip, no context, no timestamps
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r_raw = model.transcribe(
            audio=(arr, sr),
            context="",
            language="Chinese",
            return_time_stamps=False,
        )[0]
        torch.cuda.synchronize()
        t_raw = round(time.perf_counter() - t0, 3)
        total_raw_sec += t_raw

        # 2) Full 90s clip, with SiXiWanZi context + ForcedAligner timestamps
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r_ctx = model.transcribe(
            audio=(arr, sr),
            context=SIXIWANZI_CONTEXT,
            language="Chinese",
            return_time_stamps=True,
        )[0]
        torch.cuda.synchronize()
        t_ctx = round(time.perf_counter() - t0, 3)
        total_ctx_ts_sec += t_ctx

        # 3) 3x30s batched inference with context (measuring batch throughput & stability)
        chunk_len = sr * 30
        chunks = [(arr[i : i + chunk_len], sr) for i in range(0, len(arr), chunk_len) if len(arr[i : i + chunk_len]) > sr]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r_chunks = model.transcribe(
            audio=chunks,
            context=[SIXIWANZI_CONTEXT] * len(chunks),
            language=["Chinese"] * len(chunks),
            return_time_stamps=False,
        )
        torch.cuda.synchronize()
        t_batch = round(time.perf_counter() - t0, 3)
        total_batch30_sec += t_batch
        batch_text = "".join(zhconv.convert(x.text.strip(), "zh-cn") for x in r_chunks)

        raw_text = zhconv.convert(r_raw.text.strip(), "zh-cn")
        ctx_text = zhconv.convert(r_ctx.text.strip(), "zh-cn")
        ts_list = serialize_timestamps(getattr(r_ctx, "time_stamps", None))

        db_full = "".join(str(x["content"]).strip() for x in clip["db_lines"])
        w_lines = whisper_bench.get("clips", {}).get(cid, {}).get("lines", [])
        w_full = "".join(zhconv.convert(str(x["text"]).strip(), "zh-cn") for x in w_lines)

        n_db = normalize_for_diff(db_full)
        n_qctx = normalize_for_diff(ctx_text)
        n_qraw = normalize_for_diff(raw_text)
        n_w = normalize_for_diff(w_full)

        sim_qctx_db = round(SequenceMatcher(None, n_qctx, n_db).ratio(), 4)
        sim_qraw_qctx = round(SequenceMatcher(None, n_qraw, n_qctx).ratio(), 4)
        sim_qctx_w = round(SequenceMatcher(None, n_qctx, n_w).ratio(), 4) if n_w else None

        rec = {
            "id": cid,
            "group": clip["group"],
            "up_name": clip["up_name"],
            "bvid": clip["bvid"],
            "page": clip["page"],
            "start_sec": clip["start_sec"],
            "end_sec": clip["end_sec"],
            "subtitle_lan": clip["subtitle_lan"],
            "desc": clip["desc"],
            "title": clip["title"],
            "dur_sec": dur,
            "timing": {
                "qwen_raw_sec": t_raw,
                "qwen_raw_rtf": round(t_raw / dur, 4),
                "qwen_ctx_ts_sec": t_ctx,
                "qwen_ctx_ts_rtf": round(t_ctx / dur, 4),
                "qwen_batch30_sec": t_batch,
                "qwen_batch30_rtf": round(t_batch / dur, 4),
                "whisper_sec": whisper_bench.get("clips", {}).get(cid, {}).get("infer_sec"),
            },
            "char_counts": {
                "db": len(n_db),
                "whisper_live": len(n_w),
                "qwen_raw": len(n_qraw),
                "qwen_ctx": len(n_qctx),
                "qwen_batch30": len(normalize_for_diff(batch_text)),
            },
            "similarity": {
                "qwen_ctx_vs_db": sim_qctx_db,
                "qwen_raw_vs_ctx": sim_qraw_qctx,
                "qwen_ctx_vs_whisper": sim_qctx_w,
            },
            "timestamps_count": len(ts_list),
            "timestamps_sample": ts_list[:8],
            "db_text": db_full,
            "db_lines": clip["db_lines"],
            "whisper_live_text": w_full,
            "qwen_raw_text": raw_text,
            "qwen_ctx_text": ctx_text,
            "qwen_batch30_text": batch_text,
        }
        eval_results.append(rec)
        print(
            f"[{cid}] raw={t_raw:.2f}s ctx+ts={t_ctx:.2f}s batch30={t_batch:.2f}s | "
            f"chars(db={len(n_db)}, qctx={len(n_qctx)}, w={len(n_w)}) | sim(q,db)={sim_qctx_db:.3f} | ts={len(ts_list)}"
        )

    peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
    out = {
        "model": "Qwen/Qwen3-ASR-1.7B",
        "aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
        "gpu": torch.cuda.get_device_name(0),
        "load_sec": load_sec,
        "load_vram_gb": load_vram_gb,
        "peak_vram_gb": peak_vram_gb,
        "total_audio_sec": round(total_audio_sec, 2),
        "avg_rtf": {
            "qwen_raw_90s": round(total_raw_sec / total_audio_sec, 4),
            "qwen_ctx_ts_90s": round(total_ctx_ts_sec / total_audio_sec, 4),
            "qwen_batch30": round(total_batch30_sec / total_audio_sec, 4),
            "whisper_turbo": whisper_bench.get("avg_rtf"),
        },
        "total_infer_sec": {
            "qwen_raw_90s": round(total_raw_sec, 2),
            "qwen_ctx_ts_90s": round(total_ctx_ts_sec, 2),
            "qwen_batch30": round(total_batch30_sec, 2),
            "whisper_turbo": whisper_bench.get("total_infer_sec"),
        },
        "clips": eval_results,
    }
    out_path = EVAL_DIR / "qwen3_eval_report.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SAVED_REPORT: {out_path}")
    print(f"PEAK_VRAM_GB: {peak_vram_gb}")
    print("AVG_RTF:", json.dumps(out["avg_rtf"], ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-ASR-1.7B 本地推理与评测脚本")
    parser.add_argument("--audio", help="可选：转写单个音频文件")
    parser.add_argument("--no-context", action="store_true", help="不使用四禧丸子领域热词")
    parser.add_argument("--timestamps", action="store_true", help="启用 ForcedAligner 输出时间戳")
    args = parser.parse_args()
    if args.audio:
        run_single_audio(args.audio, use_context=not args.no_context, timestamps=args.timestamps)
    else:
        run_benchmark()


if __name__ == "__main__":
    main()
