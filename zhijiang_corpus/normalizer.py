"""枝江专有名词纠错归一化引擎 + 搜索同音/别名模糊扩展器."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from .config import PROJECT_ROOT
from .db import CorpusDB
from .tokenizer import build_fts_query, tokenize_for_fts, tokenize_list

DEFAULT_RULES_PATH = PROJECT_ROOT / "config" / "correction_rules.json"
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class CompiledReplaceRule:
    rule_id: str
    tier: int
    target: str
    regex: re.Pattern[str]
    fts_probes: Tuple[str, ...]
    up_names: Optional[frozenset[str]]
    min_pub_date: Optional[str] = None
    max_pub_date: Optional[str] = None


@dataclass(frozen=True)
class RulesBundle:
    replace_rules: Tuple[CompiledReplaceRule, ...]
    fuzzy_groups: Tuple[Tuple[str, ...], ...]
    fuzzy_lookup: Dict[str, Tuple[str, ...]]


@lru_cache(maxsize=4)
def load_rules(rules_path: Path = DEFAULT_RULES_PATH) -> RulesBundle:
    if not rules_path.exists():
        return RulesBundle(replace_rules=(), fuzzy_groups=(), fuzzy_lookup={})
    raw = json.loads(rules_path.read_text(encoding="utf-8"))

    compiled: List[CompiledReplaceRule] = []
    for item in raw.get("replace_rules", []):
        flags = re.IGNORECASE if item.get("ignore_case") else 0
        up_list = item.get("up_names")
        compiled.append(
            CompiledReplaceRule(
                rule_id=str(item["id"]),
                tier=int(item.get("tier", 1)),
                target=str(item["target"]),
                regex=re.compile(str(item["pattern"]), flags),
                fts_probes=tuple(str(p) for p in item.get("fts_probes", []) if p),
                up_names=frozenset(up_list) if up_list else None,
                min_pub_date=str(item["min_pub_date"]).strip() if item.get("min_pub_date") else None,
                max_pub_date=str(item["max_pub_date"]).strip() if item.get("max_pub_date") else None,
            )
        )

    groups: List[Tuple[str, ...]] = []
    lookup: Dict[str, Set[str]] = {}
    for grp in raw.get("fuzzy_groups", []):
        cleaned_grp = tuple(str(x).strip() for x in grp if str(x).strip())
        if len(cleaned_grp) < 2:
            continue
        groups.append(cleaned_grp)
        for member in cleaned_grp:
            key = member.lower()
            bucket = lookup.setdefault(key, set())
            bucket.update(cleaned_grp)

    final_lookup = {k: tuple(sorted(v)) for k, v in lookup.items()}
    return RulesBundle(
        replace_rules=tuple(compiled),
        fuzzy_groups=tuple(groups),
        fuzzy_lookup=final_lookup,
    )


def apply_domain_corrections(
    text: str,
    *,
    up_name: Optional[str] = None,
    pub_date: Optional[str] = None,
    rules_path: Path = DEFAULT_RULES_PATH,
) -> str:
    """对单条字幕正文执行枝江专有名词 Tier 1 & Tier 2 纠错（支持按 UP 主与日期范围约束，幂等安全）."""
    if not text:
        return text
    bundle = load_rules(rules_path)
    out = text
    for rule in bundle.replace_rules:
        if rule.up_names is not None:
            if not up_name or up_name not in rule.up_names:
                continue
        if rule.min_pub_date is not None:
            if not pub_date or pub_date < rule.min_pub_date:
                continue
        if rule.max_pub_date is not None:
            if not pub_date or pub_date > rule.max_pub_date:
                continue
        if rule.regex.search(out):
            out = rule.regex.sub(rule.target, out)
    return out


def build_fuzzy_fts_query(
    raw_query: str,
    *,
    fuzzy: bool = True,
    rules_path: Path = DEFAULT_RULES_PATH,
) -> Tuple[str, Dict[str, List[str]]]:
    """构造 FTS5 查询表达式；fuzzy=True 时自动对同音/别称/Tier 3 歧义词做 OR 扩展.

    示例：
      - raw_query='小海诺' -> '("小 海 诺" OR "小 海 螺" OR "小 螺")'
      - raw_query='心宜 唱歌' -> '("心 宜" OR "心 仪" OR "欣 怡" OR "心 怡") AND "唱 歌"'
    """
    if not raw_query or not raw_query.strip():
        return "", {}
    if not fuzzy:
        return build_fts_query(raw_query), {}

    bundle = load_rules(rules_path)
    clauses: List[str] = []
    expanded_map: Dict[str, List[str]] = {}

    for part in raw_query.strip().split():
        variants = bundle.fuzzy_lookup.get(part.lower())
        if not variants:
            # 也尝试匹配去标点/小写后的形式（如 A-SOUL / a-soul）
            norm_key = "".join(tokenize_list(part))
            variants = bundle.fuzzy_lookup.get(norm_key)

        if variants and len(variants) > 1:
            # 保持用户输入的原词排在第一位
            ordered = [part] + [v for v in variants if v.lower() != part.lower()]
            sub_phrases: List[str] = []
            seen_phrases: Set[str] = set()
            for v in ordered:
                toks = tokenize_list(v)
                if not toks:
                    continue
                ph = " ".join(toks).replace('"', '""')
                if ph not in seen_phrases:
                    seen_phrases.add(ph)
                    sub_phrases.append(f'"{ph}"')
            if len(sub_phrases) > 1:
                clauses.append("(" + " OR ".join(sub_phrases) + ")")
                expanded_map[part] = ordered
                continue
            if len(sub_phrases) == 1:
                clauses.append(sub_phrases[0])
                continue

        tokens = tokenize_list(part)
        if not tokens:
            continue
        phrase = " ".join(tokens).replace('"', '""')
        clauses.append(f'"{phrase}"')

    return " AND ".join(clauses), expanded_map


def run_corpus_normalization(
    db: CorpusDB,
    *,
    dry_run: bool = False,
    batch_size: int = 500,
    rules_path: Path = DEFAULT_RULES_PATH,
    progress_cb: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """利用现有 FTS5 倒排索引快速定位并批量修复存量数据库中的专有名词错词.

    特点：
      - 不全表扫描，只通过 FTS5 精确拉取候选行 ID（570 万行库仅需数秒定位）；
      - 按小批次（默认 500 行/事务）温和提交并让出锁，可与正在运行的 GPU 补录安全并发；
      - 幂等：支持反复执行（如补录结束后再跑一次，仅处理新增未修正行）。
    """
    bundle = load_rules(rules_path)
    t0 = time.perf_counter()

    def emit(msg: str) -> None:
        if progress_cb:
            progress_cb(msg)

    # 1. 利用 FTS5 倒排索引收集所有可能命中的 rowid
    candidate_ids: Set[int] = set()
    probe_stats: Dict[str, int] = {}

    with db.session() as conn:
        for rule in bundle.replace_rules:
            rule_hits: Set[int] = set()
            for probe in rule.fts_probes:
                fts_q = build_fts_query(probe)
                if not fts_q:
                    continue
                clauses = ["subtitles_fts MATCH ?"]
                qparams: List[Any] = [fts_q]
                if rule.up_names:
                    ups = list(rule.up_names)
                    ph = ",".join("?" for _ in ups)
                    clauses.append(f"s.up_name IN ({ph})")
                    qparams.extend(ups)
                if rule.min_pub_date:
                    clauses.append("s.pub_date >= ?")
                    qparams.append(rule.min_pub_date)
                if rule.max_pub_date:
                    clauses.append("s.pub_date <= ?")
                    qparams.append(rule.max_pub_date)

                if len(clauses) > 1:
                    rows = conn.execute(
                        f"""
                        SELECT s.id
                          FROM subtitles_fts f
                          CROSS JOIN subtitles s ON s.id = f.rowid
                         WHERE {' AND '.join(clauses)}
                        """,
                        qparams,
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT rowid FROM subtitles_fts WHERE subtitles_fts MATCH ?",
                        (fts_q,),
                    ).fetchall()
                for r in rows:
                    rule_hits.add(int(r[0]))
            if rule_hits:
                probe_stats[f"{rule.rule_id}->{rule.target}"] = len(rule_hits)
                candidate_ids.update(rule_hits)

    sorted_ids = sorted(candidate_ids)
    emit(f"FTS5 快速初筛完成：共定位 {len(sorted_ids)} 条候选字幕（耗时 {time.perf_counter()-t0:.1f}s）")

    # 2. 逐批校验并执行（或 dry-run 预览）
    updated_count = 0
    dedup_deleted = 0
    unchanged_count = 0
    samples: List[Dict[str, str]] = []

    for offset in range(0, len(sorted_ids), batch_size):
        chunk = sorted_ids[offset : offset + batch_size]
        ph = ",".join("?" for _ in chunk)

        with db.session() as conn:
            rows = conn.execute(
                f"SELECT id, up_name, bvid, page, pub_date, content FROM subtitles WHERE id IN ({ph})",
                chunk,
            ).fetchall()

            to_update: List[Tuple[str, str, int, str, int]] = []
            for r in rows:
                sid = int(r["id"])
                up_name = str(r["up_name"])
                pub_date = str(r["pub_date"] or "")
                old_text = str(r["content"])
                new_text = apply_domain_corrections(
                    old_text, up_name=up_name, pub_date=pub_date, rules_path=rules_path
                )
                if new_text == old_text:
                    unchanged_count += 1
                    continue
                if len(samples) < 15:
                    samples.append({"up": up_name, "old": old_text, "new": new_text})
                to_update.append((new_text, tokenize_for_fts(new_text), sid, str(r["bvid"]), int(r["page"])))

            if not dry_run and to_update:
                touched_parts = set()
                for new_text, new_fts, sid, bvid, page in to_update:
                    touched_parts.add((bvid, page))
                    try:
                        conn.execute(
                            "UPDATE subtitles SET content = ?, content_fts = ? WHERE id = ?",
                            (new_text, new_fts, sid),
                        )
                        updated_count += 1
                    except sqlite3.IntegrityError:
                        # 修正后若与同分 P 同时间点已存在的标准字幕完全重复，则安全去重删除
                        conn.execute("DELETE FROM subtitles WHERE id = ?", (sid,))
                        dedup_deleted += 1
                # 同步刷新受影响分 P 的 updated_at 与 line_count，确保增量同步能自动感知词表纠错改动
                for bvid, page in touched_parts:
                    conn.execute(
                        """
                        UPDATE videos
                           SET line_count = (SELECT COUNT(*) FROM subtitles WHERE bvid = ? AND page = ?),
                               updated_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                         WHERE bvid = ? AND page = ?
                        """,
                        (bvid, page, bvid, page),
                    )
            else:
                updated_count += len(to_update)

        # 温和让出写锁，不干扰后台并行的 GPU 补录进程
        if not dry_run and (offset + batch_size) < len(sorted_ids):
            time.sleep(0.01)

    elapsed = round(time.perf_counter() - t0, 2)
    emit(
        f"批量清洗{'预览' if dry_run else '落库'}完成："
        f"修正 {updated_count} 条，去重删除 {dedup_deleted} 条，保护未改 {unchanged_count} 条（总耗时 {elapsed}s）"
    )
    return {
        "dry_run": dry_run,
        "candidates": len(sorted_ids),
        "updated": updated_count,
        "dedup_deleted": dedup_deleted,
        "unchanged_protected": unchanged_count,
        "elapsed_sec": elapsed,
        "rule_candidate_counts": probe_stats,
        "samples": samples,
    }
