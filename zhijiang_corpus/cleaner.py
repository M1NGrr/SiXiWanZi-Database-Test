"""B 站 AI 字幕（BCC JSON / SRT）解析、清洗与标准化模块."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

try:
    import zhconv as _zhconv
except ImportError:
    _zhconv = None

# 零宽字符与不可见控制字符
_INVISIBLE_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff\x00-\x08\x0b\x0c\x0e-\x1f]")
_MULTISPACE_RE = re.compile(r"\s+")
# 仅含标点/空白/音乐符号等无语义噪声行
_NOISE_ONLY_RE = re.compile(
    r"^[\s\.,!?~`@#$%^&*()_\-+=\[\]{};:'\"<>/\\|，。！？、；：（）【】《》“”‘’…—·♪♫～]+$"
)
# 纯音频占位标签，如 [音乐]、(掌声)
_PURE_TAG_RE = re.compile(r"^[\[\(（【]\s*(音乐|掌声|笑声|鼓掌|bgm|music)\s*[\]\)）】]$", re.I)

_SRT_TIME_RE = re.compile(
    r"(?P<sh>\d{1,2}):(?P<sm>\d{2}):(?P<ss>\d{2})[,\.](?P<sms>\d{1,3})\s*-->\s*"
    r"(?P<eh>\d{1,2}):(?P<em>\d{2}):(?P<es>\d{2})[,\.](?P<ems>\d{1,3})"
)


@dataclass(frozen=True)
class CleanedSubtitleItem:
    start_sec: float
    end_sec: float
    content: str


def clean_subtitle_text(
    raw: str,
    *,
    to_simplified: bool = True,
    apply_corrections: bool = True,
    up_name: str | None = None,
) -> str:
    """清洗单条字幕文本：繁转简 + 去除脏字符 + 枝江专有名词自动纠错."""
    if not raw:
        return ""
    text = html.unescape(str(raw))
    text = _INVISIBLE_RE.sub("", text)
    text = text.replace("\r", " ").replace("\n", " ")
    text = _MULTISPACE_RE.sub(" ", text).strip()
    if to_simplified and text and _zhconv is not None:
        text = _zhconv.convert(text, "zh-cn")
    if apply_corrections and text:
        from .normalizer import apply_domain_corrections

        text = apply_domain_corrections(text, up_name=up_name)
    return text


def is_noise_line(text: str, drop_music_tags: bool = True) -> bool:
    if not text:
        return True
    if _NOISE_ONLY_RE.match(text):
        return True
    if drop_music_tags and _PURE_TAG_RE.match(text):
        return True
    return False


def normalize_timestamp(start: Any, end: Any) -> Tuple[float, float]:
    """将起止秒数标准化为非负且 end >= start 的两位小数浮点."""
    try:
        s = float(start)
    except (TypeError, ValueError):
        s = 0.0
    try:
        e = float(end)
    except (TypeError, ValueError):
        e = s
    s = max(0.0, round(s, 2))
    e = max(s, round(e, 2))
    return s, e


def clean_bcc_payload(
    payload: Dict[str, Any] | Sequence[Dict[str, Any]],
    *,
    drop_music_tags: bool = True,
    up_name: str | None = None,
) -> List[CleanedSubtitleItem]:
    """清洗 B 站 BCC JSON 字幕结构.

    兼容格式:
      - 标准 BCC: {"body": [{"from": 1.2, "to": 3.4, "content": "..."}, ...]}
      - 直接列表: [{"from": ..., "to": ..., "content": ...}, ...]
      - 扩展字段名: start / start_sec / end / end_sec / text
    """
    body = payload.get("body", []) if isinstance(payload, dict) else list(payload)
    cleaned: List[CleanedSubtitleItem] = []
    seen: set[Tuple[float, float, str]] = set()

    for item in body:
        if not isinstance(item, dict):
            continue
        raw_text = item.get("content") if "content" in item else item.get("text", "")
        text = clean_subtitle_text(str(raw_text or ""), up_name=up_name)
        if is_noise_line(text, drop_music_tags=drop_music_tags):
            continue
        raw_start = item.get("from", item.get("start_sec", item.get("start", 0.0)))
        raw_end = item.get("to", item.get("end_sec", item.get("end", raw_start)))
        start_sec, end_sec = normalize_timestamp(raw_start, raw_end)

        dedup_key = (start_sec, end_sec, text)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        cleaned.append(
            CleanedSubtitleItem(
                start_sec=start_sec,
                end_sec=end_sec,
                content=text,
            )
        )

    cleaned.sort(key=lambda x: (x.start_sec, x.end_sec))
    return cleaned


def _hms_to_sec(h: str, m: str, s: str, ms: str) -> float:
    ms_val = int(ms.ljust(3, "0")[:3])
    return round(int(h) * 3600 + int(m) * 60 + int(s) + ms_val / 1000.0, 2)


def clean_srt_text(
    srt_content: str,
    *,
    drop_music_tags: bool = True,
    up_name: str | None = None,
) -> List[CleanedSubtitleItem]:
    """解析并清洗标准 SRT 字幕文件（便于离线导入或未来接 Whisper ASR 补全）."""
    blocks = re.split(r"\r?\n\s*\r?\n", srt_content.strip())
    cleaned: List[CleanedSubtitleItem] = []
    seen: set[Tuple[float, float, str]] = set()

    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if len(lines) < 2:
            continue
        time_idx = 0
        match = _SRT_TIME_RE.search(lines[0])
        if not match and len(lines) >= 3:
            time_idx = 1
            match = _SRT_TIME_RE.search(lines[1])
        if not match:
            continue

        start_sec = _hms_to_sec(
            match.group("sh"), match.group("sm"), match.group("ss"), match.group("sms")
        )
        end_sec = _hms_to_sec(
            match.group("eh"), match.group("em"), match.group("es"), match.group("ems")
        )
        start_sec, end_sec = normalize_timestamp(start_sec, end_sec)
        text = clean_subtitle_text(" ".join(lines[time_idx + 1 :]), up_name=up_name)
        if is_noise_line(text, drop_music_tags=drop_music_tags):
            continue

        dedup_key = (start_sec, end_sec, text)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        cleaned.append(
            CleanedSubtitleItem(start_sec=start_sec, end_sec=end_sec, content=text)
        )

    cleaned.sort(key=lambda x: (x.start_sec, x.end_sec))
    return cleaned
