"""项目配置加载：环境变量、数据库路径与 6 位成员元数据（含直播回放合集 ID）."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "up_list.json"
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "sixiwanzi_subtitles.db"
ENV_FILE_PATH = PROJECT_ROOT / ".env"

# 启动时自动加载项目根目录 .env（如存在）
load_dotenv(ENV_FILE_PATH, override=False)


@dataclass(frozen=True)
class MemberConfig:
    up_name: str
    uid: int
    official_name: str
    group: str = ""
    fan_name: str = ""
    color: str = "#576690"
    replay_series_id: Optional[int] = None
    replay_season_id: Optional[int] = None
    replay_series_name: str = "直播回放"
    aliases: tuple[str, ...] = field(default_factory=tuple)


@dataclass
class AppConfig:
    db_path: Path
    bilibili_cookie: str
    request_delay: float
    request_jitter: float
    max_retries: int
    api_host: str
    api_port: int
    members: List[MemberConfig]
    members_by_name: Dict[str, MemberConfig]
    members_by_uid: Dict[int, MemberConfig]

    def resolve_up_name(self, uid: Optional[int] = None, raw_name: str = "") -> str:
        """将 B 站 UID 或原始昵称归一化为标准简称（如 嘉然今天吃什么 -> 嘉然）."""
        if uid is not None and uid in self.members_by_uid:
            return self.members_by_uid[uid].up_name
        cleaned = (raw_name or "").strip()
        if cleaned in self.members_by_name:
            return cleaned
        lowered = cleaned.lower()
        for m in self.members:
            if lowered == m.official_name.lower():
                return m.up_name
            for alias in m.aliases:
                if lowered == alias.lower():
                    return m.up_name
        return cleaned or "unknown"

    def resolve_member(self, name_or_uid: str) -> Optional[MemberConfig]:
        """根据简称、昵称或 UID 查找成员配置."""
        text = str(name_or_uid).strip()
        if text.isdigit():
            return self.members_by_uid.get(int(text))
        if text in self.members_by_name:
            return self.members_by_name[text]
        lowered = text.lower()
        for m in self.members:
            if lowered == m.official_name.lower() or any(
                lowered == a.lower() for a in m.aliases
            ):
                return m
        return None


def _build_cookie_from_env() -> str:
    raw = os.getenv("BILIBILI_COOKIE", "").strip()
    if raw:
        return raw
    parts = []
    mapping = [
        ("SESSDATA", os.getenv("BILIBILI_SESSDATA", "").strip()),
        ("bili_jct", os.getenv("BILIBILI_BILI_JCT", "").strip()),
        ("DedeUserID", os.getenv("BILIBILI_DEDEUSERID", "").strip()),
        ("buvid3", os.getenv("BILIBILI_BUVID3", "").strip()),
    ]
    for k, v in mapping:
        if v:
            parts.append(f"{k}={v}")
    return "; ".join(parts)


def save_cookie_to_env(cookie_str: str, env_path: Path = ENV_FILE_PATH) -> Path:
    """将用户授权登录得到的 B 站 Cookie 写入本地 .env 并刷新当前进程环境变量."""
    cleaned = (cookie_str or "").strip().replace("\r", "").replace("\n", "")
    lines: List[str] = []
    if env_path.exists():
        lines = env_path.read_text(encoding="utf-8").splitlines()

    replaced = False
    new_lines: List[str] = []
    for ln in lines:
        if ln.strip().startswith("BILIBILI_COOKIE="):
            new_lines.append(f"BILIBILI_COOKIE={cleaned}")
            replaced = True
        else:
            new_lines.append(ln)
    if not replaced:
        if new_lines and new_lines[-1].strip() != "":
            new_lines.append("")
        new_lines.append(f"BILIBILI_COOKIE={cleaned}")

    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    os.environ["BILIBILI_COOKIE"] = cleaned
    return env_path


def load_members(config_path: Path = DEFAULT_CONFIG_PATH) -> List[MemberConfig]:
    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    members: List[MemberConfig] = []
    for item in data.get("members", []):
        members.append(
            MemberConfig(
                up_name=item["up_name"],
                uid=int(item["uid"]),
                official_name=item.get("official_name", item["up_name"]),
                group=item.get("group", ""),
                fan_name=item.get("fan_name", ""),
                color=item.get("color", "#576690"),
                replay_series_id=(
                    int(item["replay_series_id"])
                    if item.get("replay_series_id") is not None
                    else None
                ),
                replay_season_id=(
                    int(item["replay_season_id"])
                    if item.get("replay_season_id") is not None
                    else None
                ),
                replay_series_name=str(item.get("replay_series_name", "直播回放")),
                aliases=tuple(item.get("aliases", [item["up_name"]])),
            )
        )
    return members


def load_config(
    db_path: Optional[str | Path] = None,
    cookie: Optional[str] = None,
    config_path: Path = DEFAULT_CONFIG_PATH,
) -> AppConfig:
    # 每次加载时轻量重读 .env 以便感知最新的 Cookie 写入
    load_dotenv(ENV_FILE_PATH, override=False)

    env_db = os.getenv("ZJ_DB_PATH", "").strip()
    if db_path is not None:
        resolved_db = Path(db_path)
    elif env_db:
        p = Path(env_db)
        resolved_db = p if p.is_absolute() else (PROJECT_ROOT / p)
    else:
        resolved_db = DEFAULT_DB_PATH

    members = load_members(config_path)
    return AppConfig(
        db_path=resolved_db,
        bilibili_cookie=cookie if cookie is not None else _build_cookie_from_env(),
        request_delay=float(os.getenv("ZJ_REQUEST_DELAY", "1.5")),
        request_jitter=float(os.getenv("ZJ_REQUEST_JITTER", "0.6")),
        max_retries=int(os.getenv("ZJ_MAX_RETRIES", "3")),
        api_host=os.getenv("ZJ_API_HOST", "127.0.0.1"),
        api_port=int(os.getenv("ZJ_API_PORT", "8000")),
        members=members,
        members_by_name={m.up_name: m for m in members},
        members_by_uid={m.uid: m for m in members},
    )
