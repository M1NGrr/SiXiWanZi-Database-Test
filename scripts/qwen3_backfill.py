"""基于 Qwen3-ASR-1.7B + Silero VAD + Qwen3-ForcedAligner-0.6B 的高性能多进程分片校正引擎.

性能优化（面向 RTX 5090 D v2 24GB）：
  1. 预取线程下沉全套 CPU/IO 工作：下载 B 站 DASH 音频 -> FFmpeg 转 16kHz WAV -> Silero VAD 切段 -> 按时长排序 -> 即刻删 WAV；
     GPU 主线程零 CPU 等待，连续吃满 Tensor Core；
  2. 批处理合并重试：整场触发热词泄露/异常复读的音频块汇总后一次性 batch 无热词重试（告别串行 batch=1 拖慢）；
  3. 支持多进程 GPU 分片并行（--shard-index / --shard-total，推荐 2 分片 × batch_size=28，合计显存 ~15~18 GB / 24 GB）；
  4. 按场次成员动态裁剪热词 + 歌回/外语自适应语种 + 人工锁定防覆盖 + PAUSE_QWEN3 优雅断点续跑。
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import zhconv
from qwen_asr import Qwen3ASRModel

from sixiwanzi_corpus.bilibili_client import BilibiliClient
from sixiwanzi_corpus.cleaner import clean_subtitle_text, is_noise_line, normalize_timestamp
from sixiwanzi_corpus.config import AppConfig, load_config
from sixiwanzi_corpus.db import CorpusDB, SubtitleRow, VideoRecord

ASR_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "Qwen3-ASR-1.7B"
ALIGNER_MODEL_DIR = PROJECT_ROOT / "data" / "models" / "Qwen3-ForcedAligner-0.6B"
AUDIT_JSON_PATH = PROJECT_ROOT / "data" / "tmp_asr" / "qwen3_audit.json"
WAV_TMP_DIR = PROJECT_ROOT / "data" / "tmp_asr" / "qwen3_wav"
PAUSE_SENTINEL = PROJECT_ROOT / "data" / "tmp_asr" / "PAUSE_QWEN3"

QWEN3_LAN_TAG = "asr-qwen3-1.7b"

BASE_BILI_CONTEXT = "四禧丸子 上舰 舰长 提督 虚拟区 SC 电池 弹幕 虚环"
MEMBER_SPECIFIC_CONTEXT: Dict[str, str] = {
    "沐霂": "沐霂 小沐标 大沐博士 沐霖 又一 恬豆 梨安",
    "又一": "又一 酥酥又 又神 王又一 又爸 沐霂 恬豆 梨安",
    "恬豆": "恬豆 恬豆包 陈恬豆 唐豆人 猫猫币 沐霂 又一 梨安",
    "梨安": "梨安 向心梨 肥梨 大梨老师 沐霂 又一 恬豆",
}
DOMAIN_CONTEXT = (
    "四禧丸子 沐霂 又一 恬豆 梨安 小沐标 酥酥又 恬豆包 向心梨 "
    "上舰 舰长 提督 虚拟区 SC 电池"
)
_CONTEXT_TOKENS = tuple(
    sorted(
        {
            t
            for t in (DOMAIN_CONTEXT + " " + " ".join(MEMBER_SPECIFIC_CONTEXT.values())).split()
            if len(t) >= 2
        }
    )
)
_MULTILINGUAL_TITLE_RE = re.compile(r"歌回|弹唱|唱歌|点歌|KTV|英文|英语|外语|日语|翻唱", re.IGNORECASE)

_SPACED_UPPER_RE = re.compile(r"(?<![A-Za-z])([A-Z])(?:\s+([A-Z]))+(?![A-Za-z])")
_REPEAT_CHAR_RE = re.compile(r"([^哈呵嘿嘻啊哦哇呜嗯哼诶哎\.…!\?！？~～])\1{6,}")
_REPEAT_PHRASE_RE = re.compile(r"(.{2,6}?)\1{4,}")
_NON_WORD_RE = re.compile(r"[^\u4e00-\u9fffA-Za-z0-9]+")
_CLAUSE_HINT_RE = re.compile(
    r"(?=(?:就是|然后|但是|因为|所以|不过|其实|而且|如果|哈哈+|呜呜+|啊啊+|"
    r"不是|这个|那个|你们|我们|大家|感觉|觉得|好像|真的|可以|怎么|什么|为什么))"
)


def _ts() -> str:
    return time.strftime("%H:%M:%S")


def build_member_context(up_name: str) -> str:
    m_ctx = MEMBER_SPECIFIC_CONTEXT.get((up_name or "").strip(), "")
    return f"{m_ctx} {BASE_BILI_CONTEXT}".strip() if m_ctx else DOMAIN_CONTEXT


def pick_language_for_title(title: str) -> Optional[str]:
    if _MULTILINGUAL_TITLE_RE.search(title or ""):
        return None
    return "Chinese"


def _find_ffmpeg() -> str:
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError("未找到 ffmpeg，请确认已加入系统 PATH")
    return found


def _download_and_convert_wav(
    bili: BilibiliClient, bvid: str, cid: int, out_wav: Path
) -> Path:
    ffmpeg = _find_ffmpeg()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    tmp_m4a = out_wav.with_suffix(".m4a")
    try:
        urls = bili.get_audio_stream_urls(bvid, cid)
        if not urls:
            raise RuntimeError(f"未获取到音频流 URL: {bvid} (cid={cid})")
        bili.download_stream_file(urls, tmp_m4a, bvid=bvid)
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(tmp_m4a),
            "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(out_wav),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size < 3200:
            raise RuntimeError(f"FFmpeg 转码失败: {proc.stderr.strip()}")
        return out_wav
    finally:
        tmp_m4a.unlink(missing_ok=True)


def _load_wav_16k(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        raw = wf.readframes(wf.getnframes())
    arr = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        arr = arr.reshape(-1, ch).mean(axis=1)
    return arr, sr


def _split_long_span_at_energy_dip(
    arr: np.ndarray, sr: int, start_i: int, end_i: int, max_sec: float = 26.0
) -> List[Tuple[int, int]]:
    max_n = int(max_sec * sr)
    min_n = int(14.0 * sr)
    out: List[Tuple[int, int]] = []
    cur = start_i
    win = int(0.2 * sr)
    hop = int(0.1 * sr)
    while end_i - cur > max_n:
        search_l = cur + min_n
        search_r = min(end_i - int(2.0 * sr), cur + max_n)
        if search_r <= search_l + win:
            cut = cur + max_n
        else:
            seg = arr[search_l:search_r]
            best_pos = search_l + (search_r - search_l) // 2
            best_e = float("inf")
            for off in range(0, len(seg) - win, hop):
                w = seg[off : off + win]
                e = float(np.mean(w * w))
                if e < best_e:
                    best_e = e
                    best_pos = search_l + off + win // 2
            cut = best_pos
        out.append((cur, cut))
        cur = cut
    if end_i > cur:
        out.append((cur, end_i))
    return out


def build_vad_chunks(
    arr: np.ndarray,
    sr: int,
    vad_model: Any = None,
    *,
    max_chunk_sec: float = 25.0,
    max_merge_gap_sec: float = 1.4,
    pad_sec: float = 0.15,
) -> List[Tuple[float, float, np.ndarray]]:
    """向量化 NumPy 自适应能量 + 静音谷底切段（2.3 小时音频仅耗时 ~0.08s，比 PyTorch 逐帧 Silero VAD 快 1000x）."""
    if len(arr) < int(0.35 * sr):
        return []

    hop = int(0.02 * sr)  # 20ms 帧移 (320 samples @ 16kHz)
    n_frames = len(arr) // hop
    if n_frames <= 0:
        return []
    frames = arr[: n_frames * hop].reshape(n_frames, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)

    # 自适应静音底噪门限：兼顾轻声说话与高噪游戏场次
    noise_floor = float(np.percentile(rms, 15))
    peak_ref = float(np.percentile(rms, 95))
    if peak_ref < 0.003:
        return []
    thr = max(0.0035, min(noise_floor * 1.8 + 0.002, peak_ref * 0.18))
    active = rms >= thr

    # 对活跃掩码做 300ms 形态学闭运算（填平词间短停顿）
    gap_frames = int(0.30 / 0.02)
    min_speech_frames = int(0.24 / 0.02)
    idx_active = np.flatnonzero(active)
    if len(idx_active) == 0:
        return []

    # 找出连续活跃段（间隔 <= gap_frames 合并）
    breaks = np.flatnonzero(np.diff(idx_active) > gap_frames)
    starts = np.concatenate(([idx_active[0]], idx_active[breaks + 1]))
    ends = np.concatenate((idx_active[breaks], [idx_active[-1]]))

    atomic: List[Tuple[int, int]] = []
    max_n = int(max_chunk_sec * sr)
    for sf, ef in zip(starts.tolist(), ends.tolist()):
        if (ef - sf + 1) < min_speech_frames:
            continue
        s_i = sf * hop
        e_i = min(len(arr), (ef + 1) * hop)
        if e_i - s_i > max_n:
            atomic.extend(_split_long_span_at_energy_dip(arr, sr, s_i, e_i, max_sec=max_chunk_sec))
        else:
            atomic.append((s_i, e_i))

    if not atomic:
        return []

    merged: List[Tuple[int, int]] = []
    gap_n = int(max_merge_gap_sec * sr)
    for s_i, e_i in atomic:
        if not merged:
            merged.append((s_i, e_i))
            continue
        prev_s, prev_e = merged[-1]
        if (s_i - prev_e) <= gap_n and (e_i - prev_s) <= max_n:
            merged[-1] = (prev_s, e_i)
        else:
            merged.append((s_i, e_i))

    pad_n = int(pad_sec * sr)
    total_n = len(arr)
    chunks: List[Tuple[float, float, np.ndarray]] = []
    for s_i, e_i in merged:
        ps = max(0, s_i - pad_n)
        pe = min(total_n, e_i + pad_n)
        if pe - ps < int(0.35 * sr):
            continue
        sub = arr[ps:pe]
        if float(np.sqrt(np.mean(sub * sub) + 1e-12)) < 0.003:
            continue
        chunks.append((round(ps / sr, 3), round(pe / sr, 3), sub))
    return chunks


def _has_context_leak_or_hallucination(text: str) -> bool:
    if not text:
        return False
    hits = sum(1 for tok in _CONTEXT_TOKENS if tok in text)
    if hits >= 4 and ("四禧丸子" in text and ("上舰" in text or "虚拟区" in text or "小沐标" in text)):
        return True
    if _REPEAT_CHAR_RE.search(text) or _REPEAT_PHRASE_RE.search(text):
        return True
    return False


def _normalize_qwen_text(text: str) -> str:
    s = zhconv.convert(str(text or "").strip(), "zh-cn")
    s = _SPACED_UPPER_RE.sub(lambda m: m.group(0).replace(" ", ""), s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _split_sentences_clean(text: str, max_chars: int = 24) -> List[str]:
    s = text.strip()
    if not s:
        return []
    stage1 = [p.strip() for p in re.split(r"(?<=[。！？!?])", s) if p.strip()]
    stage2: List[str] = []
    for piece in stage1:
        if len(piece) <= max_chars:
            stage2.append(piece)
        else:
            sub = [p.strip() for p in re.split(r"(?<=[，,、；;：:…])", piece) if p.strip()]
            buf = ""
            for p in sub:
                if not buf:
                    buf = p
                elif len(buf) + len(p) <= max_chars:
                    buf += p
                else:
                    stage2.append(buf)
                    buf = p
            if buf:
                stage2.append(buf)

    final: List[str] = []
    for piece in stage2:
        rem = piece
        while len(rem) > max_chars + 4:
            cut = -1
            for m in _CLAUSE_HINT_RE.finditer(rem):
                pos = m.start()
                if 10 <= pos <= max_chars:
                    cut = pos
                elif pos > max_chars:
                    break
            if cut < 0:
                sp = rem.rfind(" ", 10, max_chars + 1)
                if sp >= 10:
                    cut = sp
            if cut < 0:
                cut = min(len(rem), max_chars)
                while (
                    cut < len(rem)
                    and cut < max_chars + 5
                    and rem[cut - 1].isascii()
                    and rem[cut - 1].isalnum()
                    and rem[cut].isascii()
                    and rem[cut].isalnum()
                ):
                    cut += 1
            head = rem[:cut].strip()
            if head:
                final.append(head)
            rem = rem[cut:].strip()
        if rem:
            final.append(rem)
    return final


def align_sentences_to_timestamps(
    raw_text: str,
    ts_obj: Any,
    chunk_start: float,
    chunk_end: float,
    *,
    up_name: str,
    pub_date: str,
    duration_sec: float,
) -> List[Tuple[float, float, str]]:
    norm_text = _normalize_qwen_text(raw_text)
    if not norm_text:
        return []
    sentences = _split_sentences_clean(norm_text, max_chars=24)
    if not sentences:
        return []

    ts_items = getattr(ts_obj, "items", ts_obj) if ts_obj else []
    char_spans: List[Tuple[str, float, float]] = []
    for it in ts_items or []:
        t_str = _NON_WORD_RE.sub("", zhconv.convert(str(getattr(it, "text", "") or ""), "zh-cn")).lower()
        if not t_str:
            continue
        st = float(getattr(it, "start_time", 0.0))
        ed = float(getattr(it, "end_time", st))
        if len(t_str) == 1:
            char_spans.append((t_str, st, ed))
        else:
            step = (ed - st) / len(t_str)
            for idx, ch in enumerate(t_str):
                char_spans.append((ch, st + idx * step, st + (idx + 1) * step))

    out: List[Tuple[float, float, str]] = []
    total_chars = sum(len(_NON_WORD_RE.sub("", s)) for s in sentences) or 1
    chunk_dur = max(0.2, chunk_end - chunk_start)
    cursor = 0
    n_spans = len(char_spans)
    prop_acc = 0

    for sent in sentences:
        cleaned = clean_subtitle_text(sent, to_simplified=True, apply_corrections=True, up_name=up_name, pub_date=pub_date)
        if is_noise_line(cleaned, drop_music_tags=True):
            prop_acc += len(_NON_WORD_RE.sub("", sent))
            continue

        key_chars = _NON_WORD_RE.sub("", sent).lower()
        k_len = len(key_chars)
        matched_s: Optional[float] = None
        matched_e: Optional[float] = None

        if k_len > 0 and cursor < n_spans:
            temp_cur = cursor
            hits: List[Tuple[float, float]] = []
            for ch in key_chars:
                found_idx = -1
                limit_idx = min(n_spans, temp_cur + 8)
                for j in range(temp_cur, limit_idx):
                    if char_spans[j][0] == ch:
                        found_idx = j
                        break
                if found_idx >= 0:
                    hits.append((char_spans[found_idx][1], char_spans[found_idx][2]))
                    temp_cur = found_idx + 1
            if hits:
                matched_s = chunk_start + hits[0][0]
                matched_e = chunk_start + hits[-1][1]
                cursor = temp_cur

        if matched_s is None or matched_e is None:
            r0 = prop_acc / total_chars
            r1 = (prop_acc + max(1, k_len)) / total_chars
            matched_s = chunk_start + r0 * chunk_dur
            matched_e = chunk_start + r1 * chunk_dur
        prop_acc += max(1, k_len)

        s_sec, e_sec = normalize_timestamp(matched_s, max(matched_e, matched_s + 0.25))
        if duration_sec > 0:
            s_sec = min(s_sec, duration_sec)
            e_sec = min(max(e_sec, s_sec), duration_sec)
        out.append((s_sec, e_sec, cleaned))

    return out


def list_targets(
    db: CorpusDB,
    *,
    mode: str,
    bvid: Optional[str] = None,
    page: Optional[int] = None,
    limit: int = 0,
    force: bool = False,
    shard_index: int = 0,
    shard_total: int = 1,
) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    seen: set[Tuple[str, int]] = set()

    def _add(row: Dict[str, Any], tier: str, reason: str) -> None:
        key = (str(row["bvid"]), int(row["page"]))
        if key in seen:
            return
        if not force and str(row.get("subtitle_lan") or "") == QWEN3_LAN_TAG:
            return
        seen.add(key)
        item = dict(row)
        item["tier"] = tier
        item["reason"] = reason
        targets.append(item)

    with db.session() as conn:
        if bvid:
            clauses = ["bvid = ?"]
            params: List[Any] = [bvid.strip()]
            if page is not None:
                clauses.append("page = ?")
                params.append(int(page))
            rows = conn.execute(
                f"SELECT * FROM videos WHERE {' AND '.join(clauses)} ORDER BY page ASC",
                params,
            ).fetchall()
            for r in rows:
                _add(dict(r), "manual", "explicit_bvid")
        else:
            if mode in ("p0", "both"):
                if AUDIT_JSON_PATH.exists():
                    audit = json.loads(AUDIT_JSON_PATH.read_text(encoding="utf-8"))
                    for it in audit.get("p0", []):
                        r = conn.execute(
                            "SELECT * FROM videos WHERE bvid = ? AND page = ?",
                            (it["bvid"], int(it["page"])),
                        ).fetchone()
                        if r:
                            _add(dict(r), "P0", str(it.get("reason") or "p0_audit"))
            if mode in ("p1", "both"):
                rows = conn.execute(
                    """
                    SELECT * FROM videos
                     WHERE subtitle_status = 'ok'
                       AND (subtitle_lan = 'asr-v2-large-v3-turbo' OR subtitle_lan = 'asr-large-v3-turbo')
                     ORDER BY (line_count * 60.0 / NULLIF(duration_sec, 0)) ASC, duration_sec DESC, bvid ASC, page ASC
                    """
                ).fetchall()
                for r in rows:
                    dur = int(r["duration_sec"] or 0)
                    lpm = round(int(r["line_count"] or 0) * 60.0 / dur, 2) if dur > 0 else 0.0
                    _add(dict(r), "P1", f"whisper({lpm}/min)")
            if mode == "quality-flagged":
                rows = conn.execute(
                    """
                    SELECT * FROM videos
                     WHERE (subtitle_status = 'no_subtitle' AND duration_sec >= 15)
                        OR (subtitle_status = 'ok' AND subtitle_lan = 'ai-zh' AND extra_meta LIKE '%"needs_asr_review": true%')
                     ORDER BY pub_date DESC, bvid DESC, page ASC
                    """
                ).fetchall()
                for r in rows:
                    reason = "no_subtitle" if r["subtitle_status"] == "no_subtitle" else "ingest_quality_flagged"
                    _add(dict(r), "QF", reason)

    if shard_total > 1:
        targets = [t for i, t in enumerate(targets) if (i % shard_total) == shard_index]
    if limit and limit > 0:
        targets = targets[:limit]
    return targets


def part_has_manual_locks(db: CorpusDB, bvid: str, page: int) -> int:
    with db.session() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM subtitles WHERE bvid = ? AND page = ? AND (content_locked = 1 OR speaker_locked = 1)",
            (bvid, int(page)),
        ).fetchone()
        return int(row[0] or 0)


def _iter_dynamic_batches(
    sorted_chunks: List[Tuple[float, float, np.ndarray]],
    *,
    max_batch_size: int = 24,
    max_padded_sec: float = 200.0,
) -> List[List[Tuple[float, float, np.ndarray]]]:
    """按补零后的批次总音频秒数做动态分桶，避免 20×25s 长段集中触发 Windows WDDM 23GB 显存换页墙。"""
    batches: List[List[Tuple[float, float, np.ndarray]]] = []
    cur: List[Tuple[float, float, np.ndarray]] = []
    cur_max_sec = 0.0
    cap_n = max(1, int(max_batch_size))
    cap_sec = max(30.0, float(max_padded_sec))

    for ch in sorted_chunks:
        dur = max(0.5, float(ch[1] - ch[0]))
        new_max = max(cur_max_sec, dur)
        if cur and (len(cur) >= cap_n or (len(cur) + 1) * new_max > cap_sec):
            batches.append(cur)
            cur = [ch]
            cur_max_sec = dur
        else:
            cur.append(ch)
            cur_max_sec = new_max
    if cur:
        batches.append(cur)
    return batches


def _iter_align_batches(
    items: List[Tuple[Tuple[float, float, np.ndarray], str, str]],
    *,
    max_batch_size: int = 16,
    max_padded_tokens: int = 3600,
) -> List[List[Tuple[Tuple[float, float, np.ndarray], str, str]]]:
    """按 ForcedAligner 估算序列长度做动态分桶，避免长文本与大 batch 叠加撑爆 [B, L, 152064] logits 显存。"""
    def _est_tokens(entry: Tuple[Tuple[float, float, np.ndarray], str, str]) -> int:
        chunk_meta, txt, _ = entry
        dur = max(0.5, float(chunk_meta[1] - chunk_meta[0]))
        return int(dur * 14.0 + len(txt) * 3 + 40)

    ordered = sorted(items, key=_est_tokens)
    batches: List[List[Tuple[Tuple[float, float, np.ndarray], str, str]]] = []
    cur: List[Tuple[Tuple[float, float, np.ndarray], str, str]] = []
    cur_max_tok = 0
    cap_n = max(1, int(max_batch_size))
    cap_tok = max(800, int(max_padded_tokens))

    for entry in ordered:
        tlen = _est_tokens(entry)
        new_max = max(cur_max_tok, tlen)
        if cur and (len(cur) >= cap_n or (len(cur) + 1) * new_max > cap_tok):
            batches.append(cur)
            cur = [entry]
            cur_max_tok = tlen
        else:
            cur.append(entry)
            cur_max_tok = new_max
    if cur:
        batches.append(cur)
    return batches


def transcribe_chunks_qwen3(
    model: Qwen3ASRModel,
    sorted_chunks: List[Tuple[float, float, np.ndarray]],
    sr: int,
    *,
    actual_dur: float,
    up_name: str,
    bvid: str,
    page: int,
    pub_date: str,
    duration_sec: float,
    title: str = "",
    batch_size: int = 32,
    max_padded_sec: float = 280.0,
    max_align_tokens: int = 3600,
) -> Tuple[List[SubtitleRow], Dict[str, Any]]:
    """两段解耦式高速推理：
    1) 纯 ASR 批处理 + 批处理重试（先过滤幻觉/热词泄露，不让异常长文本进入 ForcedAligner）；
    2) 有效文本按序列长度排序后做严格 token 预算分桶 ForcedAligner 逐字对齐（彻底消除 23GB WDDM 显存爆峰）。
    """
    t_inf0 = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    eff_dur = float(duration_sec) if duration_sec and duration_sec > 0 else actual_dur
    if not sorted_chunks:
        return [], {"chunks": 0, "speech_sec": 0.0, "infer_sec": 0.0, "rtf": 0.0, "retries": 0}

    speech_sec = round(sum(c[1] - c[0] for c in sorted_chunks), 1)
    ctx_str = build_member_context(up_name)
    eff_lang = pick_language_for_title(title)

    valid_items: List[Tuple[Tuple[float, float, np.ndarray], str, str]] = []
    retry_queue: List[Tuple[float, float, np.ndarray]] = []

    supported_align_langs: Optional[set[str]] = None
    if getattr(model, "forced_aligner", None) is not None:
        get_langs = getattr(model.forced_aligner, "get_supported_languages", None)
        if callable(get_langs):
            try:
                supported_align_langs = set(get_langs() or [])
            except Exception:
                supported_align_langs = None

    def _pick_align_lang(pred_lang: Optional[str]) -> str:
        cand = str(pred_lang or eff_lang or "Chinese").strip() or "Chinese"
        if supported_align_langs and cand.lower() not in supported_align_langs:
            return "Chinese"
        return cand

    with torch.inference_mode():
        # 阶段 1：纯 ASR 批处理（return_time_stamps=False，无 aligner logits 显存尖峰，可放心开大 batch）
        for batch in _iter_dynamic_batches(sorted_chunks, max_batch_size=batch_size, max_padded_sec=max_padded_sec):
            res_list = model.transcribe(
                audio=[(c[2], sr) for c in batch],
                context=[ctx_str] * len(batch),
                language=[eff_lang] * len(batch),
                return_time_stamps=False,
            )
            for chunk_meta, res in zip(batch, res_list):
                txt = str(res.text or "").strip()
                if _has_context_leak_or_hallucination(txt):
                    retry_queue.append(chunk_meta)
                    continue
                if txt:
                    valid_items.append((chunk_meta, txt, _pick_align_lang(getattr(res, "language", None))))

        # 阶段 1.5：仅对幻觉/热词泄露块做一次性无热词批处理重试
        retries = len(retry_queue)
        if retry_queue:
            retry_queue.sort(key=lambda c: len(c[2]))
            for rb in _iter_dynamic_batches(retry_queue, max_batch_size=batch_size, max_padded_sec=max_padded_sec):
                res2_list = model.transcribe(
                    audio=[(c[2], sr) for c in rb],
                    context=[""] * len(rb),
                    language=[eff_lang] * len(rb),
                    return_time_stamps=False,
                )
                for chunk_meta, res2 in zip(rb, res2_list):
                    txt2 = str(res2.text or "").strip()
                    if not txt2 or _has_context_leak_or_hallucination(txt2):
                        continue
                    valid_items.append((chunk_meta, txt2, _pick_align_lang(getattr(res2, "language", None))))

        # 阶段 2：对全部已清洗有效文本做 token 预算分桶 ForcedAligner 对齐
        all_lines: List[Tuple[float, float, str]] = []
        aligner = getattr(model, "forced_aligner", None)
        if valid_items and aligner is not None:
            align_batches = _iter_align_batches(
                valid_items,
                max_batch_size=min(16, max(4, batch_size // 2)),
                max_padded_tokens=max_align_tokens,
            )
            for ab in align_batches:
                try:
                    ts_list = aligner.align(
                        audio=[(entry[0][2], sr) for entry in ab],
                        text=[entry[1] for entry in ab],
                        language=[entry[2] for entry in ab],
                    )
                except Exception:
                    # 极端情况（罕见语言/符号异常）降级逐条对齐或按字数线性插值
                    ts_list = []
                    for entry in ab:
                        try:
                            ts_list.append(
                                aligner.align(
                                    audio=[(entry[0][2], sr)],
                                    text=[entry[1]],
                                    language=[entry[2]],
                                )[0]
                            )
                        except Exception:
                            ts_list.append(None)

                for (chunk_meta, txt, _), ts_obj in zip(ab, ts_list):
                    c_start, c_end, _ = chunk_meta
                    all_lines.extend(
                        align_sentences_to_timestamps(
                            txt,
                            ts_obj,
                            c_start,
                            c_end,
                            up_name=up_name,
                            pub_date=pub_date,
                            duration_sec=eff_dur,
                        )
                    )
        else:
            for chunk_meta, txt, _ in valid_items:
                c_start, c_end, _ = chunk_meta
                all_lines.extend(
                    align_sentences_to_timestamps(
                        txt,
                        None,
                        c_start,
                        c_end,
                        up_name=up_name,
                        pub_date=pub_date,
                        duration_sec=eff_dur,
                    )
                )

    torch.cuda.synchronize()
    peak_vram_gb = round(torch.cuda.max_memory_allocated() / (1024 ** 3), 2) if torch.cuda.is_available() else 0.0
    infer_sec = round(time.perf_counter() - t_inf0, 2)

    dedup: List[SubtitleRow] = []
    seen_keys: set[Tuple[float, float, str]] = set()
    for s_sec, e_sec, content in sorted(all_lines, key=lambda x: (x[0], x[1])):
        k = (s_sec, e_sec, content)
        if k in seen_keys:
            continue
        seen_keys.add(k)
        dedup.append(
            SubtitleRow(
                up_name=up_name,
                bvid=bvid,
                page=page,
                pub_date=pub_date,
                start_sec=s_sec,
                end_sec=e_sec,
                content=content,
                source=QWEN3_LAN_TAG,
            )
        )

    return dedup, {
        "chunks": len(sorted_chunks),
        "speech_sec": speech_sec,
        "infer_sec": infer_sec,
        "rtf": round(infer_sec / max(1.0, actual_dur), 4),
        "retries": retries,
        "peak_vram_gb": peak_vram_gb,
    }


def _load_qwen3_gpu_model(batch_size: int) -> Qwen3ASRModel:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    asr_path = str(ASR_MODEL_DIR) if ASR_MODEL_DIR.exists() else "Qwen/Qwen3-ASR-1.7B"
    align_path = str(ALIGNER_MODEL_DIR) if ALIGNER_MODEL_DIR.exists() else "Qwen/Qwen3-ForcedAligner-0.6B"
    base_kwargs: Dict[str, Any] = dict(
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_inference_batch_size=max(batch_size, 48),
        max_new_tokens=280,
        forced_aligner=align_path,
        forced_aligner_kwargs=dict(
            dtype=torch.bfloat16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        ),
    )
    try:
        return Qwen3ASRModel.from_pretrained(asr_path, attn_implementation="sdpa", **base_kwargs)
    except TypeError:
        base_kwargs["forced_aligner_kwargs"] = dict(dtype=torch.bfloat16, device_map="cuda:0")
        return Qwen3ASRModel.from_pretrained(asr_path, **base_kwargs)


def run_backfill(
    *,
    mode: str,
    bvid: Optional[str],
    page: Optional[int],
    limit: int,
    batch_size: int,
    max_batch_sec: float,
    max_align_tokens: int,
    prefetch: int,
    shard_index: int,
    shard_total: int,
    force: bool,
    force_locked: bool,
    log_file: Optional[str] = None,
) -> int:
    log_fp = None
    if log_file:
        lp = Path(log_file)
        lp.parent.mkdir(parents=True, exist_ok=True)
        log_fp = lp.open("a", encoding="utf-8", buffering=1)

    shard_tag = f"S{shard_index}/{shard_total}" if shard_total > 1 else "S0"

    def emit(msg: str) -> None:
        line = f"[{_ts()}][{shard_tag}] {msg}"
        print(line, flush=True)
        if log_fp is not None:
            log_fp.write(line + "\n")

    cfg: AppConfig = load_config()
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass
    db = CorpusDB(cfg.db_path)
    db.init_db()

    targets = list_targets(
        db,
        mode=mode,
        bvid=bvid,
        page=page,
        limit=limit,
        force=force,
        shard_index=shard_index,
        shard_total=shard_total,
    )
    total_hours = sum(int(t.get("duration_sec") or 0) for t in targets) / 3600.0
    emit(
        f"模式={mode} | 待处理分 P: {len(targets)} 个（总时长 {total_hours:.2f} 小时）| "
        f"batch_size={batch_size} | max_batch_sec={max_batch_sec:g}s | align_tok={max_align_tokens} | prefetch={prefetch}"
    )
    if not targets:
        emit("没有待处理的分 P。")
        if log_fp is not None:
            log_fp.close()
        return 0

    WAV_TMP_DIR.mkdir(parents=True, exist_ok=True)
    t_load = time.perf_counter()
    model = _load_qwen3_gpu_model(batch_size)
    emit(
        f"GPU 模型就绪: Qwen3-ASR-1.7B + ForcedAligner-0.6B (SDPA/TF32) "
        f"(加载 {time.perf_counter()-t_load:.2f}s, 显存 {torch.cuda.memory_allocated()/1024**3:.2f} GB)"
    )

    filtered_targets: List[Dict[str, Any]] = []
    skipped_locked = 0
    for t in targets:
        if not force_locked:
            lock_cnt = part_has_manual_locks(db, str(t["bvid"]), int(t["page"]))
            if lock_cnt > 0:
                skipped_locked += 1
                emit(f"[SKIP_LOCKED] {t['bvid']} P{t['page']} 含 {lock_cnt} 条人工校对锁定记录，跳过保护")
                continue
        filtered_targets.append(t)

    if not filtered_targets:
        emit(f"全部目标均已跳过（locked={skipped_locked}）。")
        if log_fp is not None:
            log_fp.close()
        return 0

    # 多预取线程（下载 + FFmpeg + Silero VAD + 时长排序全部在后台 CPU 线程完成，立刻删临时 WAV）
    n_prefetch = max(1, min(int(prefetch), len(filtered_targets), 4))
    work_q: "queue.Queue[Optional[Tuple[int, Dict[str, Any]]]]" = queue.Queue()
    for idx, item in enumerate(filtered_targets, start=1):
        work_q.put((idx, item))
    for _ in range(n_prefetch):
        work_q.put(None)

    ready_q: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max(2, n_prefetch + 1))
    stop_event = threading.Event()
    active_workers = [n_prefetch]
    workers_lock = threading.Lock()

    def _prefetch_worker(wid: int) -> None:
        bili = BilibiliClient(
            cookie=cfg.bilibili_cookie,
            delay=max(0.5, cfg.request_delay),
            jitter=cfg.request_jitter,
            max_retries=cfg.max_retries,
        )
        try:
            while not stop_event.is_set() and not PAUSE_SENTINEL.exists():
                try:
                    pkt = work_q.get_nowait()
                except queue.Empty:
                    break
                if pkt is None:
                    break
                idx, item = pkt
                b = str(item["bvid"])
                p = int(item["page"])
                cid = item.get("cid")
                wav_path = WAV_TMP_DIR / f"{b}_p{p}_s{shard_index}_w{wid}.wav"
                t_pre0 = time.perf_counter()
                try:
                    if not cid:
                        vinfo = bili.get_video_info(b)
                        pinfo = next((x for x in vinfo.pages if x.page == p), None)
                        if not pinfo:
                            raise RuntimeError(f"B 站未找到分 P {p}")
                        cid = pinfo.cid
                    _download_and_convert_wav(bili, b, int(cid), wav_path)
                    arr, sr = _load_wav_16k(wav_path)
                    actual_dur = round(len(arr) / sr, 2)
                    chunks = build_vad_chunks(arr, sr, max_chunk_sec=25.0)
                    sorted_chunks = sorted(chunks, key=lambda c: len(c[2]))
                    del arr
                    payload = {
                        "kind": "ready",
                        "idx": idx,
                        "item": item,
                        "chunks": sorted_chunks,
                        "sr": sr,
                        "actual_dur": actual_dur,
                        "prep_sec": round(time.perf_counter() - t_pre0, 1),
                    }
                    while not stop_event.is_set():
                        try:
                            ready_q.put(payload, timeout=0.5)
                            break
                        except queue.Full:
                            continue
                except Exception as exc:
                    if not stop_event.is_set():
                        ready_q.put({"kind": "error", "idx": idx, "item": item, "error": str(exc)})
                finally:
                    wav_path.unlink(missing_ok=True)
        finally:
            bili.close()
            with workers_lock:
                active_workers[0] -= 1
                if active_workers[0] == 0:
                    ready_q.put({"kind": "done"})

    threads = [
        threading.Thread(target=_prefetch_worker, args=(w,), name=f"qwen-pf-{shard_index}-{w}", daemon=True)
        for w in range(n_prefetch)
    ]
    for th in threads:
        th.start()

    ok_parts = 0
    kept_existing = 0
    failed_parts = 0
    total_old_lines = 0
    total_new_lines = 0
    t_all = time.perf_counter()

    try:
        while True:
            msg = ready_q.get()
            kind = msg["kind"]
            if kind == "done":
                break
            idx = msg["idx"]
            item = msg["item"]
            b = str(item["bvid"])
            p = int(item["page"])
            up = str(item["up_name"])
            old_lan = str(item.get("subtitle_lan") or "")
            old_lines = int(item.get("line_count") or 0)
            dur = int(item.get("duration_sec") or 0)

            if kind == "error":
                failed_parts += 1
                emit(f"[{idx}/{len(filtered_targets)}] [FAIL] {up} {b} P{p}: {msg['error']}")
                continue

            try:
                sub_rows, st = transcribe_chunks_qwen3(
                    model,
                    msg["chunks"],
                    msg["sr"],
                    actual_dur=msg["actual_dur"],
                    up_name=up,
                    bvid=b,
                    page=p,
                    pub_date=str(item["pub_date"]),
                    duration_sec=float(dur),
                    title=str(item.get("title") or ""),
                    batch_size=batch_size,
                    max_padded_sec=max_batch_sec,
                    max_align_tokens=max_align_tokens,
                )
            except Exception as exc:
                failed_parts += 1
                emit(f"[{idx}/{len(filtered_targets)}] [ERR_INFER] {up} {b} P{p}: {exc}")
                continue
            finally:
                msg["chunks"] = None
                torch.cuda.empty_cache()

            if old_lan == "ai-zh" and old_lines >= 80 and len(sub_rows) < int(old_lines * 0.6):
                kept_existing += 1
                emit(
                    f"[{idx}/{len(filtered_targets)}] [KEEP_AI_ZH] {up} {b} P{p} "
                    f"(old={old_lines}, qwen3={len(sub_rows)}，保留原字幕 | gpu={st['infer_sec']}s)"
                )
                continue

            if not sub_rows:
                if old_lines == 0:
                    with db.session() as conn:
                        conn.execute(
                            """
                            UPDATE videos
                               SET subtitle_lan = ?, subtitle_status = 'asr_empty',
                                   updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                             WHERE bvid = ? AND page = ?
                            """,
                            (QWEN3_LAN_TAG, b, p),
                        )
                kept_existing += 1
                emit(f"[{idx}/{len(filtered_targets)}] [EMPTY] {up} {b} P{p} 无有效人声 (prep={msg['prep_sec']}s)")
                continue

            vid_rec = VideoRecord(
                bvid=b,
                page=p,
                cid=item.get("cid"),
                up_uid=item.get("up_uid"),
                up_name=up,
                raw_up_name=str(item.get("raw_up_name") or ""),
                title=str(item.get("title") or ""),
                part_title=str(item.get("part_title") or ""),
                pub_date=str(item["pub_date"]),
                pub_ts=int(item.get("pub_ts") or 0),
                duration_sec=dur,
                subtitle_lan=QWEN3_LAN_TAG,
                subtitle_status="ok",
                line_count=len(sub_rows),
                is_visible=int(item.get("is_visible") if item.get("is_visible") is not None else 1),
                voiceprint_status="none",
                extra_meta=str(item.get("extra_meta") or "{}"),
            )
            ins, _ = db.save_video_and_subtitles(vid_rec, sub_rows, replace_existing=True)
            with db.session() as conn:
                conn.execute(
                    "UPDATE videos SET voiceprint_status = 'none' WHERE bvid = ? AND page = ?",
                    (b, p),
                )

            ok_parts += 1
            total_old_lines += old_lines
            total_new_lines += ins
            emit(
                f"[{idx}/{len(filtered_targets)}] [OK/{item['tier']}] {up} {b} P{p} "
                f"({old_lan or 'empty'} {old_lines} -> qwen3 {ins} 条 | "
                f"dur={dur}s, speech={st['speech_sec']}s, chunks={st['chunks']}, "
                f"prep={msg['prep_sec']}s, gpu={st['infer_sec']}s, RTF={st['rtf']}, "
                f"vram={st.get('peak_vram_gb', 0)}G, retry={st['retries']}) | {str(item.get('title') or '')[:20]}"
            )

            if PAUSE_SENTINEL.exists():
                stop_event.set()
                emit(f"检测到 {PAUSE_SENTINEL}，已安全落库并优雅暂停。")
                break
    finally:
        stop_event.set()
        for th in threads:
            th.join(timeout=3.0)

    elapsed_min = round((time.perf_counter() - t_all) / 60.0, 2)
    emit(
        f"分片完成: ok={ok_parts}, kept_existing={kept_existing}, failed={failed_parts}, "
        f"locked_skipped={skipped_locked}, lines {total_old_lines} -> {total_new_lines} (耗时 {elapsed_min} min)"
    )
    if log_fp is not None:
        log_fp.close()
    return 0 if failed_parts == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-ASR-1.7B + Silero VAD + ForcedAligner 高速分片校正引擎")
    parser.add_argument("--mode", choices=("p0", "p1", "both", "quality-flagged"), default="p1")
    parser.add_argument("--bvid", help="可选：指定单个 BV 号测试/重录")
    parser.add_argument("--page", type=int, help="可选：配合 --bvid 指定单个分 P")
    parser.add_argument("--limit", type=int, default=0, help="限制处理分 P 数量，0 表示不限制")
    parser.add_argument("--batch-size", type=int, default=32, help="GPU 推理最大批大小（短段上限）")
    parser.add_argument("--max-batch-sec", type=float, default=280.0, help="单批次补零后音频总秒数上限（防 WDDM 显存换页）")
    parser.add_argument("--max-align-tokens", type=int, default=3200, help="ForcedAligner 单批估算 token 上限（防 logits 显存尖峰）")
    parser.add_argument("--prefetch", type=int, default=3, help="每个 GPU 进程的后台下载+VAD 预取线程数")
    parser.add_argument("--shard-index", type=int, default=0, help="分片序号（0 开始）")
    parser.add_argument("--shard-total", type=int, default=1, help="总分片数（24GB 挂机激进模式推荐 2，日常桌面推荐 1）")
    parser.add_argument("--force", action="store_true", help="强制覆盖已经是 asr-qwen3-1.7b 的分 P")
    parser.add_argument("--force-locked", action="store_true", help="强制覆盖含人工校对锁定的分 P（默认跳过保护）")
    parser.add_argument("--log-file", help="可选：同步追加 UTF-8 日志到指定文件路径")
    args = parser.parse_args()
    raise SystemExit(
        run_backfill(
            mode=args.mode,
            bvid=args.bvid,
            page=args.page,
            limit=args.limit,
            batch_size=args.batch_size,
            max_batch_sec=args.max_batch_sec,
            max_align_tokens=args.max_align_tokens,
            prefetch=args.prefetch,
            shard_index=args.shard_index,
            shard_total=args.shard_total,
            force=args.force,
            force_locked=args.force_locked,
            log_file=args.log_file,
        )
    )


if __name__ == "__main__":
    main()
