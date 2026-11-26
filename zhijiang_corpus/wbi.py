"""B 站 WBI 请求签名工具（用于 x/player/wbi/v2、x/space/wbi/arc/search 等接口）."""

from __future__ import annotations

import hashlib
import time
import urllib.parse
from typing import Any, Dict, Tuple

MIXIN_KEY_ENC_TAB = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
    27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
    37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
    22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52,
]

_FILTER_CHARS = "!'()*"


def get_mixin_key(orig: str) -> str:
    """对 img_key + sub_key 做固定表重排并截取前 32 位."""
    return "".join(orig[i] for i in MIXIN_KEY_ENC_TAB)[:32]


def extract_wbi_keys(nav_data: Dict[str, Any]) -> Tuple[str, str]:
    """从 /x/web-interface/nav 返回的 data 中提取 img_key 与 sub_key."""
    wbi_img = nav_data.get("wbi_img") or {}
    img_url = str(wbi_img.get("img_url", ""))
    sub_url = str(wbi_img.get("sub_url", ""))
    if not img_url or not sub_url:
        raise ValueError("未能从 /x/web-interface/nav 获取 wbi_img 密钥")
    img_key = img_url.rsplit("/", 1)[-1].split(".")[0]
    sub_key = sub_url.rsplit("/", 1)[-1].split(".")[0]
    return img_key, sub_key


def sign_wbi_params(
    params: Dict[str, Any],
    img_key: str,
    sub_key: str,
    *,
    timestamp: int | None = None,
) -> Dict[str, str]:
    """为查询参数附加 wts 与 w_rid 签名."""
    mixin_key = get_mixin_key(img_key + sub_key)
    wts = int(time.time()) if timestamp is None else int(timestamp)
    signed: Dict[str, str] = {}
    for k, v in params.items():
        if v is None:
            continue
        val_str = "".join(ch for ch in str(v) if ch not in _FILTER_CHARS)
        signed[str(k)] = val_str
    signed["wts"] = str(wts)

    ordered = dict(sorted(signed.items()))
    query = urllib.parse.urlencode(ordered)
    w_rid = hashlib.md5((query + mixin_key).encode("utf-8")).hexdigest()
    ordered["w_rid"] = w_rid
    return ordered
