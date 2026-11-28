"""B 站 API 客户端：获取合集/系列回放列表、视频分 P、AI 字幕与官方二维码登录."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from .wbi import extract_wbi_keys, sign_wbi_params

DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/131.0.0.0 Safari/537.36"
)

DEFAULT_LAN_PRIORITY = ("ai-zh", "zh-CN", "zh-Hans", "zh-Hant", "zh-HK", "zh-TW")


class BilibiliAPIError(RuntimeError):
    """B 站接口返回业务错误或风控拦截."""

    def __init__(self, code: int, message: str, endpoint: str = ""):
        self.code = code
        self.endpoint = endpoint
        super().__init__(f"B站接口错误 [{code}] {endpoint}: {message}")


@dataclass
class VideoPageInfo:
    page: int
    cid: int
    part_title: str
    duration_sec: int


@dataclass
class VideoInfo:
    bvid: str
    aid: int
    title: str
    desc: str
    up_uid: int
    raw_up_name: str
    pub_ts: int
    pub_date: str
    duration_sec: int
    pages: List[VideoPageInfo] = field(default_factory=list)


@dataclass
class SpaceVideoBrief:
    bvid: str
    title: str
    pub_ts: int
    pub_date: str
    duration_sec: int
    author: str
    mid: int


@dataclass
class CollectionBrief:
    kind: str  # series | season
    collection_id: int
    name: str
    total: int


@dataclass
class SubtitleTrack:
    lan: str
    lan_doc: str
    subtitle_url: str
    ai_type: int = 0
    ai_status: int = 0


def _ts_to_iso_date(ts: int) -> str:
    """将 Unix 时间戳转为北京时间（UTC+8）日期 YYYY-MM-DD."""
    if not ts or ts <= 0:
        return "1970-01-01"
    dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    beijing_ts = dt.timestamp() + 8 * 3600
    return datetime.fromtimestamp(beijing_ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _parse_cookie_str(cookie_str: str) -> Dict[str, str]:
    cookies: Dict[str, str] = {}
    for item in (cookie_str or "").split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        k, v = item.split("=", 1)
        cookies[k.strip()] = v.strip()
    return cookies


def _parse_duration_str(val: Any) -> int:
    if isinstance(val, (int, float)):
        return int(val)
    text = str(val or "").strip()
    if text.isdigit():
        return int(text)
    parts = text.split(":")
    try:
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except ValueError:
        pass
    return 0


class BilibiliClient:
    """封装 B 站公开/WBI 接口，支持直播回放合集抓取、礼貌限速、重试与官方登录校验."""

    def __init__(
        self,
        cookie: str = "",
        *,
        delay: float = 1.5,
        jitter: float = 0.6,
        max_retries: int = 3,
        timeout: float = 20.0,
    ):
        self.delay = max(0.0, delay)
        self.jitter = max(0.0, jitter)
        self.max_retries = max(1, max_retries)
        self._cookies = _parse_cookie_str(cookie)
        self._wbi_keys: Optional[Tuple[str, str]] = None
        self._wbi_fetched_at: float = 0.0
        self._last_req_ts: float = 0.0
        self._buvid_checked: bool = False

        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={
                "User-Agent": DEFAULT_UA,
                "Referer": "https://www.bilibili.com/",
                "Origin": "https://www.bilibili.com",
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
        )
        for k, v in self._cookies.items():
            self._client.cookies.set(k, v, domain=".bilibili.com")

    @property
    def has_sessdata(self) -> bool:
        return bool(self._cookies.get("SESSDATA"))

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BilibiliClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _throttle(self) -> None:
        if self.delay <= 0:
            return
        now = time.monotonic()
        elapsed = now - self._last_req_ts
        target = self.delay + random.uniform(0.0, self.jitter)
        if self._last_req_ts > 0 and elapsed < target:
            time.sleep(target - elapsed)
        self._last_req_ts = time.monotonic()

    def _request_json(
        self,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        check_bili_code: bool = True,
        send_cookies: bool = True,
        referer: Optional[str] = None,
    ) -> Dict[str, Any]:
        last_exc: Optional[Exception] = None
        extra_headers = {"Referer": referer} if referer else {}
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                if send_cookies:
                    resp = self._client.get(url, params=params, headers=extra_headers)
                else:
                    resp = httpx.get(
                        url,
                        params=params,
                        timeout=self._client.timeout,
                        follow_redirects=True,
                        headers={
                            "User-Agent": DEFAULT_UA,
                            "Referer": referer or "https://www.bilibili.com/",
                        },
                    )
                if resp.status_code in (412, 429, 502, 503, 504):
                    # 412/429 风控或限流时拉长退避时间，保护账号与批量任务稳定
                    base_wait = 15.0 if resp.status_code in (412, 429) else 2.0
                    backoff = (base_wait * attempt) + random.uniform(1.0, 4.0)
                    time.sleep(backoff)
                    last_exc = httpx.HTTPStatusError(
                        f"HTTP {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                    continue
                resp.raise_for_status()
                payload = resp.json()
                if check_bili_code and isinstance(payload, dict) and "code" in payload:
                    code = int(payload.get("code", 0))
                    if code != 0:
                        msg = str(payload.get("message") or payload.get("msg") or "")
                        if code in (-352, -412, -509, -799) and attempt < self.max_retries:
                            # WBI 风控 (-352) 时顺便使缓存密钥失效，下次自动刷新
                            if code == -352:
                                self._wbi_keys = None
                            base_wait = 12.0 if code in (-352, -412) else 3.0
                            time.sleep((base_wait * attempt) + random.uniform(1.0, 3.0))
                            last_exc = BilibiliAPIError(code, msg, url)
                            continue
                        raise BilibiliAPIError(code, msg, url)
                return payload
            except BilibiliAPIError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep((1.5**attempt) + random.uniform(0.2, 0.8))
                    continue
        if last_exc:
            raise last_exc
        raise RuntimeError(f"请求失败: {url}")

    def ensure_buvid(self) -> None:
        """若未提供 buvid3，尝试从 spi 接口自动获取以降低 WBI 风控概率."""
        if self._buvid_checked or self._cookies.get("buvid3"):
            self._buvid_checked = True
            return
        self._buvid_checked = True
        try:
            payload = self._request_json(
                "https://api.bilibili.com/x/frontend/finger/spi",
                check_bili_code=True,
            )
            data = payload.get("data") or {}
            b3 = data.get("b_3")
            b4 = data.get("b_4")
            if b3:
                self._cookies["buvid3"] = b3
                self._client.cookies.set("buvid3", b3, domain=".bilibili.com")
            if b4:
                self._cookies["buvid4"] = b4
                self._client.cookies.set("buvid4", b4, domain=".bilibili.com")
        except Exception:
            pass

    def verify_login(self) -> Dict[str, Any]:
        """校验当前 Cookie 是否为有效的 B 站登录态."""
        payload = self._request_json(
            "https://api.bilibili.com/x/web-interface/nav",
            check_bili_code=False,
        )
        data = payload.get("data") or {}
        is_login = bool(data.get("isLogin"))
        return {
            "is_login": is_login,
            "uname": str(data.get("uname") or ""),
            "mid": int(data.get("mid") or 0),
            "has_sessdata": self.has_sessdata,
        }

    @staticmethod
    def generate_login_qrcode() -> Dict[str, str]:
        """调用 B 站官方 Web 扫码登录接口生成二维码 URL 与轮询 key."""
        resp = httpx.get(
            "https://passport.bilibili.com/x/passport-login/web/qrcode/generate",
            headers={"User-Agent": DEFAULT_UA, "Referer": "https://www.bilibili.com/"},
            timeout=15.0,
        )
        resp.raise_for_status()
        payload = resp.json()
        if int(payload.get("code", -1)) != 0:
            raise BilibiliAPIError(
                int(payload.get("code", -1)),
                str(payload.get("message", "生成二维码失败")),
                "qrcode/generate",
            )
        data = payload.get("data") or {}
        return {
            "url": str(data.get("url") or ""),
            "qrcode_key": str(data.get("qrcode_key") or ""),
        }

    @staticmethod
    def poll_login_qrcode(qrcode_key: str) -> Dict[str, Any]:
        """轮询官方扫码登录状态；登录成功时返回提取好的 Cookie 字符串."""
        with httpx.Client(
            headers={"User-Agent": DEFAULT_UA, "Referer": "https://www.bilibili.com/"},
            timeout=15.0,
            follow_redirects=True,
        ) as c:
            resp = c.get(
                "https://passport.bilibili.com/x/passport-login/web/qrcode/poll",
                params={"qrcode_key": qrcode_key},
            )
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data") or {}
            status_code = int(data.get("code", -1))
            message = str(data.get("message") or "")
            cookie_str = ""
            if status_code == 0:
                # 优先从客户端 Cookie Jar 提取完整登录态
                kv = dict(c.cookies.items())
                # 若返回体 URL 含查询参数（部分版本），也做补充解析
                if kv:
                    cookie_str = "; ".join(f"{k}={v}" for k, v in kv.items())
            return {
                "code": status_code,
                "message": message,
                "cookie": cookie_str,
            }

    def get_wbi_keys(self, force_refresh: bool = False) -> Tuple[str, str]:
        """获取并缓存当前 WBI 签名密钥（30 分钟缓存）."""
        now = time.monotonic()
        if (
            not force_refresh
            and self._wbi_keys is not None
            and (now - self._wbi_fetched_at) < 1800
        ):
            return self._wbi_keys
        self.ensure_buvid()
        payload = self._request_json(
            "https://api.bilibili.com/x/web-interface/nav",
            check_bili_code=False,
        )
        data = payload.get("data") or {}
        keys = extract_wbi_keys(data)
        self._wbi_keys = keys
        self._wbi_fetched_at = now
        return keys

    def discover_collections(self, mid: int) -> List[CollectionBrief]:
        """查询 UP 主空间的所有合集 (season) 与视频列表 (series)."""
        payload = self._request_json(
            "https://api.bilibili.com/x/polymer/web-space/seasons_series_list",
            params={"mid": mid, "page_num": 1, "page_size": 20},
            referer=f"https://space.bilibili.com/{mid}/lists",
        )
        lists_obj = (payload.get("data") or {}).get("items_lists") or {}
        out: List[CollectionBrief] = []
        for s in lists_obj.get("series_list") or []:
            meta = s.get("meta") or {}
            out.append(
                CollectionBrief(
                    kind="series",
                    collection_id=int(meta.get("series_id") or 0),
                    name=str(meta.get("name") or ""),
                    total=int(meta.get("total") or 0),
                )
            )
        for s in lists_obj.get("seasons_list") or []:
            meta = s.get("meta") or {}
            out.append(
                CollectionBrief(
                    kind="season",
                    collection_id=int(meta.get("season_id") or 0),
                    name=str(meta.get("name") or ""),
                    total=int(meta.get("total") or 0),
                )
            )
        return out

    def list_series_videos(
        self,
        mid: int,
        series_id: int,
        *,
        limit: Optional[int] = 20,
        since_date: Optional[str] = None,
        until_date: Optional[str] = None,
        page_size: int = 30,
    ) -> List[SpaceVideoBrief]:
        """从 UP 主的「直播回放」视频列表 (series) 按时间倒序分页拉取视频."""
        results: List[SpaceVideoBrief] = []
        pn = 1
        ps = min(100, max(1, page_size))

        while limit is None or len(results) < limit:
            payload = self._request_json(
                "https://api.bilibili.com/x/series/archives",
                params={
                    "mid": mid,
                    "series_id": series_id,
                    "only_normal": "true",
                    "sort": "desc",
                    "pn": pn,
                    "ps": ps,
                },
                referer=f"https://space.bilibili.com/{mid}/lists/{series_id}?type=series",
            )
            data = payload.get("data") or {}
            archives = data.get("archives") or []
            if not archives:
                break

            reached_older = False
            for item in archives:
                pub_ts = int(item.get("pubdate") or 0)
                pub_date = _ts_to_iso_date(pub_ts)
                if until_date and pub_date > until_date:
                    continue
                if since_date and pub_date < since_date:
                    reached_older = True
                    break
                results.append(
                    SpaceVideoBrief(
                        bvid=str(item.get("bvid") or ""),
                        title=str(item.get("title") or ""),
                        pub_ts=pub_ts,
                        pub_date=pub_date,
                        duration_sec=int(item.get("duration") or 0),
                        author="",
                        mid=mid,
                    )
                )
                if limit is not None and len(results) >= limit:
                    break

            total = int((data.get("page") or {}).get("total") or 0)
            if reached_older or pn * ps >= total:
                break
            pn += 1

        return results

    def list_season_videos(
        self,
        mid: int,
        season_id: int,
        *,
        limit: Optional[int] = 20,
        since_date: Optional[str] = None,
        until_date: Optional[str] = None,
        page_size: int = 30,
    ) -> List[SpaceVideoBrief]:
        """从 UP 主的合集 (season) 分页拉取视频."""
        results: List[SpaceVideoBrief] = []
        pn = 1
        ps = min(100, max(1, page_size))

        while limit is None or len(results) < limit:
            payload = self._request_json(
                "https://api.bilibili.com/x/polymer/web-space/seasons_archives_list",
                params={
                    "mid": mid,
                    "season_id": season_id,
                    "sort_reverse": "false",
                    "page_num": pn,
                    "page_size": ps,
                },
                referer=f"https://space.bilibili.com/{mid}/lists/{season_id}?type=season",
            )
            data = payload.get("data") or {}
            archives = data.get("archives") or []
            if not archives:
                break

            for item in archives:
                pub_ts = int(item.get("pubdate") or 0)
                pub_date = _ts_to_iso_date(pub_ts)
                if until_date and pub_date > until_date:
                    continue
                if since_date and pub_date < since_date:
                    continue
                results.append(
                    SpaceVideoBrief(
                        bvid=str(item.get("bvid") or ""),
                        title=str(item.get("title") or ""),
                        pub_ts=pub_ts,
                        pub_date=pub_date,
                        duration_sec=int(item.get("duration") or 0),
                        author="",
                        mid=mid,
                    )
                )
                if limit is not None and len(results) >= limit:
                    break

            total = int((data.get("page") or {}).get("total") or 0)
            if pn * ps >= total:
                break
            pn += 1

        return results

    def get_video_info(self, bvid: str) -> VideoInfo:
        """获取视频详情及全部分 P（cid / 分 P 标题 / 时长）."""
        payload = self._request_json(
            "https://api.bilibili.com/x/web-interface/view",
            params={"bvid": bvid},
        )
        data = payload.get("data") or {}
        owner = data.get("owner") or {}
        pub_ts = int(data.get("pubdate") or 0)
        pages: List[VideoPageInfo] = []
        for p in data.get("pages") or []:
            pages.append(
                VideoPageInfo(
                    page=int(p.get("page", 1)),
                    cid=int(p.get("cid", 0)),
                    part_title=str(p.get("part") or ""),
                    duration_sec=int(p.get("duration") or 0),
                )
            )
        if not pages and data.get("cid"):
            pages.append(
                VideoPageInfo(
                    page=1,
                    cid=int(data["cid"]),
                    part_title=str(data.get("title") or ""),
                    duration_sec=int(data.get("duration") or 0),
                )
            )
        return VideoInfo(
            bvid=str(data.get("bvid") or bvid),
            aid=int(data.get("aid") or 0),
            title=str(data.get("title") or ""),
            desc=str(data.get("desc") or ""),
            up_uid=int(owner.get("mid") or 0),
            raw_up_name=str(owner.get("name") or ""),
            pub_ts=pub_ts,
            pub_date=_ts_to_iso_date(pub_ts),
            duration_sec=int(data.get("duration") or 0),
            pages=pages,
        )

    def list_up_videos(
        self,
        mid: int,
        *,
        limit: Optional[int] = 20,
        keyword: str = "",
        since_date: Optional[str] = None,
        until_date: Optional[str] = None,
        page_size: int = 30,
    ) -> List[SpaceVideoBrief]:
        """按发布时间倒序拉取 UP 主投稿列表（WBI 签名，作为无合集 ID 时的备选方式）."""
        img_key, sub_key = self.get_wbi_keys()
        results: List[SpaceVideoBrief] = []
        pn = 1
        ps = min(50, max(1, page_size))

        while limit is None or len(results) < limit:
            raw_params: Dict[str, Any] = {
                "mid": mid,
                "ps": ps,
                "pn": pn,
                "order": "pubdate",
                "tid": 0,
            }
            if keyword:
                raw_params["keyword"] = keyword
            signed = sign_wbi_params(raw_params, img_key, sub_key)
            payload = self._request_json(
                "https://api.bilibili.com/x/space/wbi/arc/search",
                params=signed,
                referer=f"https://space.bilibili.com/{mid}/video",
            )
            data = payload.get("data") or {}
            vlist = (data.get("list") or {}).get("vlist") or []
            if not vlist:
                break

            reached_older_than_since = False
            for item in vlist:
                pub_ts = int(item.get("created") or 0)
                pub_date = _ts_to_iso_date(pub_ts)
                if until_date and pub_date > until_date:
                    continue
                if since_date and pub_date < since_date:
                    reached_older_than_since = True
                    break
                results.append(
                    SpaceVideoBrief(
                        bvid=str(item.get("bvid") or ""),
                        title=str(item.get("title") or ""),
                        pub_ts=pub_ts,
                        pub_date=pub_date,
                        duration_sec=_parse_duration_str(item.get("length")),
                        author=str(item.get("author") or ""),
                        mid=int(item.get("mid") or mid),
                    )
                )
                if limit is not None and len(results) >= limit:
                    break

            page_info = data.get("page") or {}
            total_count = int(page_info.get("count") or 0)
            if reached_older_than_since or pn * ps >= total_count:
                break
            pn += 1

        return results

    def list_subtitle_tracks(self, bvid: str, cid: int) -> List[SubtitleTrack]:
        """调用 /x/player/wbi/v2 获取指定分 P 的全部字幕轨（含 B 站 AI 字幕 ai-zh）."""
        img_key, sub_key = self.get_wbi_keys()
        signed = sign_wbi_params({"bvid": bvid, "cid": cid}, img_key, sub_key)
        payload = self._request_json(
            "https://api.bilibili.com/x/player/wbi/v2",
            params=signed,
            referer=f"https://www.bilibili.com/video/{bvid}/",
        )
        data = payload.get("data") or {}
        sub_obj = data.get("subtitle") or {}
        raw_list = sub_obj.get("subtitles") or []
        tracks: List[SubtitleTrack] = []
        for item in raw_list:
            url = str(item.get("subtitle_url") or "").strip()
            if not url:
                continue
            if url.startswith("//"):
                url = "https:" + url
            tracks.append(
                SubtitleTrack(
                    lan=str(item.get("lan") or ""),
                    lan_doc=str(item.get("lan_doc") or ""),
                    subtitle_url=url,
                    ai_type=int(item.get("ai_type") or 0),
                    ai_status=int(item.get("ai_status") or 0),
                )
            )
        return tracks

    @staticmethod
    def pick_subtitle_track(
        tracks: Sequence[SubtitleTrack],
        prefer_lans: Sequence[str] = DEFAULT_LAN_PRIORITY,
    ) -> Optional[SubtitleTrack]:
        """按语言优先级挑选最合适的字幕轨（默认优先 ai-zh，其次人工中文字幕）."""
        if not tracks:
            return None
        by_lan = {t.lan: t for t in tracks}
        for lan in prefer_lans:
            if lan in by_lan:
                return by_lan[lan]
        for t in tracks:
            if "zh" in t.lan.lower():
                return t
        return tracks[0]

    def fetch_subtitle_json(self, subtitle_url: str) -> Dict[str, Any]:
        """下载 B 站字幕 CDN 上的 BCC JSON 文件."""
        url = "https:" + subtitle_url if subtitle_url.startswith("//") else subtitle_url
        return self._request_json(url, check_bili_code=False, send_cookies=False)

    def get_audio_stream_urls(self, bvid: str, cid: int) -> List[str]:
        """调用 /x/player/wbi/playurl 获取纯音频 DASH 流地址（不下载视频，省流量且极快）."""
        img_key, sub_key = self.get_wbi_keys()
        signed = sign_wbi_params(
            {
                "bvid": bvid,
                "cid": cid,
                "fnval": 16,
                "fnver": 0,
                "fourk": 1,
            },
            img_key,
            sub_key,
        )
        payload = self._request_json(
            "https://api.bilibili.com/x/player/wbi/playurl",
            params=signed,
            referer=f"https://www.bilibili.com/video/{bvid}/",
        )
        data = payload.get("data") or {}
        dash = data.get("dash") or {}
        audios = dash.get("audio") or []
        if audios:
            # 语音识别只需 16kHz 单声道，优先挑最小码率的清晰音频轨（如 64k/132k AAC），大幅节省下载时间
            audios_sorted = sorted(audios, key=lambda a: int(a.get("bandwidth") or 0))
            chosen = audios_sorted[0]
            urls: List[str] = []
            for key in ("baseUrl", "base_url"):
                if chosen.get(key):
                    urls.append(str(chosen[key]))
            for key in ("backupUrl", "backup_url"):
                for u in chosen.get(key) or []:
                    if u and str(u) not in urls:
                        urls.append(str(u))
            if urls:
                return urls

        # 极少数早期老视频无 DASH 时，回退传统 durl
        durls = data.get("durl") or []
        fallback_urls: List[str] = []
        if durls:
            first = durls[0]
            if first.get("url"):
                fallback_urls.append(str(first["url"]))
            for u in first.get("backup_url") or []:
                if u and str(u) not in fallback_urls:
                    fallback_urls.append(str(u))
        return fallback_urls

    def download_stream_file(
        self,
        urls: Sequence[str],
        out_path: Any,
        *,
        bvid: str,
        timeout: float = 180.0,
    ) -> Any:
        """流式下载 B 站音频流到本地临时文件（自动尝试备用 CDN 节点）."""
        from pathlib import Path

        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = dest.with_suffix(dest.suffix + ".part")
        last_exc: Optional[Exception] = None

        headers = {
            "User-Agent": DEFAULT_UA,
            "Referer": f"https://www.bilibili.com/video/{bvid}/",
            "Origin": "https://www.bilibili.com",
        }
        for url in urls:
            try:
                with httpx.stream(
                    "GET",
                    url,
                    headers=headers,
                    timeout=timeout,
                    follow_redirects=True,
                ) as resp:
                    resp.raise_for_status()
                    with tmp_path.open("wb") as f:
                        for chunk in resp.iter_bytes(chunk_size=65536):
                            if chunk:
                                f.write(chunk)
                if tmp_path.exists() and tmp_path.stat().st_size > 0:
                    tmp_path.replace(dest)
                    return dest
            except Exception as exc:
                last_exc = exc
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass
                continue
        if last_exc:
            raise last_exc
        raise RuntimeError(f"未能下载音频流: {bvid}")
