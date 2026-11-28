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


_CN_DIGITS = "零一二三四五六七八九"
_CN_DIGIT_MAP = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "俩": 2,
    "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}
_CN_UNIT_MAP = {"十": 10, "百": 100, "千": 1000, "万": 10000}
_ARABIC_PERCENT_RE = re.compile(r"(\d{1,3})%")
_CN_PERCENT_RE = re.compile(r"百分之([零〇一二两三四五六七八九十百]{1,6})")
_ARABIC_INT_RE = re.compile(r"(?<!\d)(\d{1,5})(?!\d)")
_CN_NUM_RE = re.compile(r"[零〇一二两三四五六七八九十百千万]{1,7}")


def _int_to_cn_spoken(n: int) -> str:
    """将 0~99999 整数转为常见中文口语写法（如 12->十二, 1500->一千五百, 39->三十九）."""
    if n < 0 or n > 99999:
        return ""
    if n == 0:
        return "零"
    if n < 10:
        return _CN_DIGITS[n]
    if 10 <= n < 20:
        return "十" + ("" if n == 10 else _CN_DIGITS[n % 10])

    parts: List[str] = []
    units = ((10000, "万"), (1000, "千"), (100, "百"), (10, "十"), (1, ""))
    rem = n
    need_zero = False
    for val, unit in units:
        d = rem // val
        rem %= val
        if d > 0:
            if need_zero and parts:
                parts.append("零")
            parts.append(_CN_DIGITS[d] + unit)
            need_zero = False
        elif parts:
            need_zero = True
    return "".join(parts)


def _int_to_cn_digits(s: str) -> str:
    """逐位转中文数字（适用于年份 2026->二零二六 或编号 1128->一一二八）."""
    return "".join(_CN_DIGITS[int(ch)] for ch in s if ch.isdigit())


def _cn_to_int(cn: str) -> Optional[int]:
    """解析常见中文数字串（如 十二->12、一千五百->1500、二零二六->2026），无效则返回 None."""
    if not cn:
        return None
    # 纯逐位数字（无十百千万），如 二零二六 / 一一二八
    if not any(u in cn for u in _CN_UNIT_MAP):
        if len(cn) > 5 or not all(ch in _CN_DIGIT_MAP for ch in cn):
            return None
        if len(cn) == 1:
            return _CN_DIGIT_MAP[cn]
        return int("".join(str(_CN_DIGIT_MAP[ch]) for ch in cn))

    total = 0
    section = 0
    num = 0
    has_val = False
    for ch in cn:
        if ch in _CN_DIGIT_MAP:
            num = _CN_DIGIT_MAP[ch]
            has_val = True
        elif ch in _CN_UNIT_MAP:
            unit = _CN_UNIT_MAP[ch]
            if unit == 10000:
                section = (section + (num if has_val else 1)) * unit
                total += section
                section = 0
            else:
                section += (num if has_val else 1) * unit
            num = 0
            has_val = False
        else:
            return None
    val = total + section + num
    return val if 0 <= val <= 99999 else None


def expand_numeral_variants(term: str) -> List[str]:
    """生成中阿数字等价扩展词（如 39% <-> 百分之三十九，12点 <-> 十二点，1500 <-> 一千五百）.

    仅在词条包含数字或中文数词时触发，上限 4 个变体以防止 FTS5 表达式膨胀。
    """
    raw = (term or "").strip()
    if not raw:
        return []
    out: List[str] = [raw]
    seen: Set[str] = {raw.lower()}

    def _push(v: str) -> None:
        v = v.strip()
        if v and v.lower() not in seen and len(out) < 4:
            seen.add(v.lower())
            out.append(v)

    # 1. 百分比双向转换：39% <-> 百分之三十九
    m_pct = _ARABIC_PERCENT_RE.search(raw)
    if m_pct:
        val = int(m_pct.group(1))
        cn = _int_to_cn_spoken(val)
        if cn:
            _push(_ARABIC_PERCENT_RE.sub(f"百分之{cn}", raw, count=1))
    m_cnpct = _CN_PERCENT_RE.search(raw)
    if m_cnpct:
        val_opt = _cn_to_int(m_cnpct.group(1))
        if val_opt is not None:
            _push(_CN_PERCENT_RE.sub(f"{val_opt}%", raw, count=1))

    # 2. 阿拉伯整数 -> 中文口语 / 逐位读法（如 12点 -> 十二点，2026年 -> 二零二六年）
    m_ar = _ARABIC_INT_RE.search(raw)
    if m_ar and not m_pct:
        num_str = m_ar.group(1)
        val = int(num_str)
        if 1900 <= val <= 2099 or (len(num_str) >= 3 and num_str.startswith("0")):
            _push(_ARABIC_INT_RE.sub(_int_to_cn_digits(num_str), raw, count=1))
        spoken = _int_to_cn_spoken(val)
        if spoken:
            _push(_ARABIC_INT_RE.sub(spoken, raw, count=1))
        if len(num_str) == 4 and not (1900 <= val <= 2099):
            _push(_ARABIC_INT_RE.sub(_int_to_cn_digits(num_str), raw, count=1))

    _cn_unit_follow = "点分秒天年月日号个次遍块元毛角斤米岁级名层楼集期场倍成折"

    # 3. 中文数词 -> 阿拉伯数字（如 十二点 -> 12点，第一名 -> 第1名，一千五百 -> 1500）
    # 单字中文数字仅在带量词/序数词语境下转换，避免误伤“四禧丸子 / 又一 / 一起”等专有名词与日常词
    if not m_cnpct and not m_ar:
        m_cn = _CN_NUM_RE.search(raw)
        if m_cn:
            cn_sub = m_cn.group(0)
            s_pos, e_pos = m_cn.span()
            prev_ch = raw[s_pos - 1] if s_pos > 0 else ""
            next_ch = raw[e_pos] if e_pos < len(raw) else ""
            is_multi = len(cn_sub) >= 2
            has_quant_ctx = (prev_ch == "第") or (bool(next_ch) and next_ch in _cn_unit_follow)
            if is_multi or has_quant_ctx:
                val_opt = _cn_to_int(cn_sub)
                if val_opt is not None:
                    _push(raw[:s_pos] + str(val_opt) + raw[e_pos:])

    return out


def build_fuzzy_fts_query(
    raw_query: str,
    *,
    fuzzy: bool = True,
    rules_path: Path = DEFAULT_RULES_PATH,
) -> Tuple[str, Dict[str, List[str]]]:
    """构造 FTS5 查询表达式；fuzzy=True 时自动对同音/别称/Tier 3 歧义词及中阿数字做 OR 扩展.

    示例：
      - raw_query='四禧丸子' -> '("四 禧 丸 子" OR "四 喜 丸 子" ...)'
      - raw_query='39%' -> '("39" OR "百 分 之 三 十 九")'
      - raw_query='12点' -> '("12 点" OR "十 二 点")'
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

        combined: List[str] = [part]
        seen_raw: Set[str] = {part.lower()}
        if variants:
            for v in variants:
                if v.lower() not in seen_raw:
                    seen_raw.add(v.lower())
                    combined.append(v)
        for nv in expand_numeral_variants(part):
            if nv.lower() not in seen_raw:
                seen_raw.add(nv.lower())
                combined.append(nv)

        if len(combined) > 1:
            sub_phrases: List[str] = []
            seen_phrases: Set[str] = set()
            for v in combined:
                toks = tokenize_list(v)
                if not toks:
                    continue
                ph = " ".join(toks).replace('"', '""')
                if ph not in seen_phrases:
                    seen_phrases.add(ph)
                    sub_phrases.append(f'"{ph}"')
            if len(sub_phrases) > 1:
                clauses.append("(" + " OR ".join(sub_phrases) + ")")
                expanded_map[part] = combined
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
                f"SELECT id, up_name, bvid, page, pub_date, content, content_locked FROM subtitles WHERE id IN ({ph})",
                chunk,
            ).fetchall()

            to_update: List[Tuple[str, str, int, str, int]] = []
            for r in rows:
                sid = int(r["id"])
                if int(r["content_locked"] or 0) == 1:
                    unchanged_count += 1
                    continue
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
        "rule_candidate_counts": probe_stats,
        "samples": samples,
        "elapsed_sec": elapsed,
    }


def preview_literal_replace_rule(
    db: CorpusDB,
    wrong_term: str,
    right_term: str,
    *,
    sample_limit: int = 12,
) -> Dict[str, Any]:
    """预览一条「错词 -> 正确词」全局字面替换规则在当前数据库中的命中情况（跳过人工锁定行）."""
    wrong = (wrong_term or "").strip()
    right = (right_term or "").strip()
    if not wrong or not right:
        raise ValueError("错词与目标正确词均不能为空")
    if wrong == right:
        raise ValueError("错词与目标正确词不能相同")

    fts_q = build_fts_query(wrong)
    if not fts_q:
        raise ValueError("错词不包含可索引的有效字符")

    with db.session() as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.up_name, s.bvid, s.page, s.pub_date, s.content, s.content_locked
              FROM subtitles_fts f
              CROSS JOIN subtitles s ON s.id = f.rowid
             WHERE subtitles_fts MATCH ?
            """,
            (fts_q,),
        ).fetchall()

    matched_unlocked = 0
    locked_skipped = 0
    samples: List[Dict[str, str]] = []
    for r in rows:
        old_text = str(r["content"])
        if wrong not in old_text:
            continue
        if int(r["content_locked"] or 0) == 1:
            locked_skipped += 1
            continue
        matched_unlocked += 1
        if len(samples) < sample_limit:
            samples.append(
                {
                    "up": str(r["up_name"]),
                    "bvid": str(r["bvid"]),
                    "old": old_text,
                    "new": old_text.replace(wrong, right),
                }
            )

    return {
        "wrong_term": wrong,
        "right_term": right,
        "matched_count": matched_unlocked,
        "locked_skipped": locked_skipped,
        "samples": samples,
    }


def append_literal_replace_rule(
    wrong_term: str,
    right_term: str,
    *,
    note: str = "",
    rules_path: Path = DEFAULT_RULES_PATH,
) -> Dict[str, Any]:
    """将高频错词安全追加到 config/correction_rules.json（若同 target 已存在自定义规则则合并）."""
    wrong = (wrong_term or "").strip()
    right = (right_term or "").strip()
    if not wrong or not right:
        raise ValueError("错词与目标正确词均不能为空")
    if wrong == right:
        raise ValueError("错词与目标正确词不能相同")

    if rules_path.exists():
        raw = json.loads(rules_path.read_text(encoding="utf-8"))
    else:
        raw = {"version": "1.1", "description": "人工纠错规则表", "replace_rules": [], "fuzzy_groups": []}

    rules = raw.setdefault("replace_rules", [])
    escaped_wrong = re.escape(wrong)

    # 检查是否已存在完全一致的规则探针
    for r in rules:
        if str(r.get("target")) == right and wrong in (r.get("fts_probes") or []):
            return {"status": "exists", "rule_id": r.get("id"), "wrong_term": wrong, "right_term": right}

    # 查找是否已有同 target 的 manual 规则可以合并
    manual_id = f"manual_{abs(hash(right)) % 1000000}"
    merged = False
    for r in rules:
        if str(r.get("id", "")).startswith("manual_") and str(r.get("target")) == right:
            probes = list(r.get("fts_probes") or [])
            if wrong not in probes:
                probes.append(wrong)
            r["fts_probes"] = probes
            existing_pat = str(r.get("pattern") or "")
            parts = [p for p in existing_pat.split("|") if p]
            if escaped_wrong not in parts:
                parts.append(escaped_wrong)
            r["pattern"] = "|".join(parts)
            if note:
                r["note"] = note
            manual_id = str(r["id"])
            merged = True
            break

    if not merged:
        existing_ids = {str(r.get("id")) for r in rules}
        idx = 1
        while manual_id in existing_ids:
            manual_id = f"manual_{abs(hash(right)) % 1000000}_{idx}"
            idx += 1
        rules.append(
            {
                "id": manual_id,
                "tier": 1,
                "target": right,
                "pattern": escaped_wrong,
                "fts_probes": [wrong],
                "note": note or f"人工校对沉淀：{wrong} -> {right}",
            }
        )

    rules_path.parent.mkdir(parents=True, exist_ok=True)
    rules_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    load_rules.cache_clear()
    return {"status": "added" if not merged else "merged", "rule_id": manual_id, "wrong_term": wrong, "right_term": right}
