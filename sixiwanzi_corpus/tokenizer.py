"""中英混排 FTS5 切分器：让 SQLite 原生 FTS5 高速支持中文 1~N 字精确子串匹配."""

from __future__ import annotations

import re
from typing import List

# 匹配单个 CJK/假名/谚文字符，或连续的英文字母/数字
_TOKEN_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff\uac00-\ud7af]|[A-Za-z0-9]+"
)


def tokenize_list(text: str) -> List[str]:
    """将中英混排文本切分为统一小写的 token 列表."""
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def tokenize_for_fts(text: str) -> str:
    """生成写入 subtitles.content_fts 的空格分隔索引文本.

    示例:
        '嘉然今天吃什么？A-SOUL太棒了！'
        -> '嘉 然 今 天 吃 什 么 a soul 太 棒 了'
    """
    return " ".join(tokenize_list(text))


def build_fts_query(raw_query: str) -> str:
    """将用户输入关键词安全转换为 SQLite FTS5 MATCH 表达式.

    规则:
      - 以空格分隔的多个关键词按 AND 组合，例如 '嘉然 唱歌' -> '"嘉 然" AND "唱 歌"'
      - 每个关键词内部转为 FTS5 精确短语，天然支持 1 字、2 字、长句、中英混排（如 'B站'、'A-SOUL'）
      - 所有特殊字符均被过滤或引号转义，杜绝 FTS5 语法报错或注入
    """
    if not raw_query or not raw_query.strip():
        return ""
    clauses: List[str] = []
    for part in raw_query.strip().split():
        tokens = tokenize_list(part)
        if not tokens:
            continue
        phrase = " ".join(tokens).replace('"', '""')
        clauses.append(f'"{phrase}"')
    return " AND ".join(clauses)
