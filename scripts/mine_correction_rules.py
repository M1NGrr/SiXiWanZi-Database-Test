"""全库只读拼音/近音错词自动挖掘脚本（WAL 只读模式，可与后台写入安全并发）.

用法：
  .\.venv\Scripts\python.exe scripts\mine_correction_rules.py --min-hits 15
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sixiwanzi_corpus.config import load_config
from sixiwanzi_corpus.normalizer import DEFAULT_RULES_PATH
from sixiwanzi_corpus.tokenizer import build_fts_query

DEFAULT_PROBES: Dict[str, List[str]] = {
    "四禧丸子": ["四喜丸子", "四西丸子", "思西丸子", "四喜玩子", "四禧玩子", "四喜湾子"],
    "沐霂": ["木木", "沐沐", "沐霖", "幕幕", "穆穆", "慕慕", "目目", "沐木", "慕木"],
    "小沐标": ["小目标", "小木标", "小目表", "小木表", "小沐表", "小慕标"],
    "大沐博士": ["大木博士", "大幕博士", "大目博士", "大慕博士"],
    "又神": ["佑神", "右神", "幼神", "柚神", "诱神", "游神", "优神", "友神"],
    "酥酥又": ["苏苏又", "酥酥油", "苏苏油", "素素又", "素素友", "酥酥友", "苏苏有", "酥酥有", "叔叔又"],
    "恬豆": ["甜豆", "田豆", "添豆", "恬逗", "甜逗", "天豆", "舔豆"],
    "恬豆包": ["甜豆包", "田豆包", "添豆包", "天豆包", "舔豆包", "甜豆宝", "天豆宝"],
    "梨安": ["李安", "黎安", "离安", "丽安", "利安", "莉安", "立安", "里安", "璃安"],
    "向心梨": ["向心力", "向心里", "向心理", "向心离", "相信里", "相心里", "像心理", "象心里", "想心里"],
    "肥梨": ["肥狸", "肥里", "肥离", "肥力"],
    "大梨老师": ["大理老师", "大力老师", "大李老师", "大黎老师"],
    "上舰": ["上剑", "上见", "上建", "上健"],
    "舰长": ["建长", "健长", "剑长", "见长"],
    "虚环": ["虚幻", "徐环", "续环"],
    "虚拟区": ["虚拟去", "去年去"],
    "猫猫币": ["猫猫臂", "猫猫壁"],
    "提督": ["提都", "提度"],
    "弹幕": ["弹目", "弹木"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="全库只读同音错词自动挖掘工具")
    parser.add_argument("--min-hits", type=int, default=15, help="最小命中条数阈值")
    parser.add_argument("--out", help="可选：导出 JSON 报告路径")
    args = parser.parse_args()

    cfg = load_config()
    rules_raw = json.loads(DEFAULT_RULES_PATH.read_text(encoding="utf-8")) if DEFAULT_RULES_PATH.exists() else {}
    existing_probes = {
        p for r in rules_raw.get("replace_rules", []) for p in (r.get("fts_probes") or [])
    }
    existing_fuzzy = {w for grp in rules_raw.get("fuzzy_groups", []) for w in grp}

    conn = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)
    report: Dict[str, List[Dict[str, object]]] = {}

    for target, variants in DEFAULT_PROBES.items():
        items: List[Dict[str, object]] = []
        for v in variants:
            fq = build_fts_query(v)
            if not fq:
                continue
            cnt = int(
                conn.execute(
                    "SELECT COUNT(*) FROM subtitles_fts WHERE subtitles_fts MATCH ?", (fq,)
                ).fetchone()[0]
            )
            if cnt < args.min_hits:
                continue
            samples = [
                str(r[0])
                for r in conn.execute(
                    "SELECT s.content FROM subtitles_fts f CROSS JOIN subtitles s ON s.id = f.rowid WHERE subtitles_fts MATCH ? LIMIT 3",
                    (fq,),
                ).fetchall()
            ]
            status = (
                "in_replace"
                if v in existing_probes
                else ("in_fuzzy" if v in existing_fuzzy else "NEW_CANDIDATE")
            )
            items.append({"variant": v, "hits": cnt, "status": status, "samples": samples})
        if items:
            items.sort(key=lambda x: int(x["hits"]), reverse=True)
            report[target] = items
            print(f"\n=== [{target}] ===")
            for it in items:
                print(
                    f"  {it['variant']:<8s} {it['hits']:>6d}  [{it['status']:<13s}]  "
                    f"{' || '.join(it['samples'])[:96]}"
                )

    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[OK] 报告已写入: {args.out}")


if __name__ == "__main__":
    main()
