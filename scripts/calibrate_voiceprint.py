"""基于 /admin 人工校对锁定样本（speaker_locked=1）的声纹阈值自动校准工具.

功能：
  1. 只读扫描数据库中已人工核验的声纹样本（speaker_locked=1 且 speaker_sim 非空）；
  2. 在 [0.40, 0.62] 区间内为每位成员网格搜索最优 F1 的独立 host_thr；
  3. 样本充足（默认 >=20 条/人）且显式传 --apply 时，将结果写入 config/voiceprints.json 的 thresholds.per_member。

用法：
  .\.venv\Scripts\python.exe scripts\calibrate_voiceprint.py
  .\.venv\Scripts\python.exe scripts\calibrate_voiceprint.py --min-samples 20 --apply
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sixiwanzi_corpus.config import load_config
from sixiwanzi_corpus.voiceprint import (
    DEFAULT_THRESHOLDS,
    DEFAULT_VOICEPRINTS_PATH,
    load_voiceprints,
    save_voiceprints,
)


def evaluate_threshold(samples: List[Tuple[float, bool]], thr: float) -> Tuple[float, float, float, int]:
    tp = fp = fn = 0
    for sim, is_host in samples:
        pred = sim >= thr
        if pred and is_host:
            tp += 1
        elif pred and not is_host:
            fp += 1
        elif not pred and is_host:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return f1, prec, rec, tp + fp + fn


def main() -> None:
    parser = argparse.ArgumentParser(description="利用人工校对样本自动校准四成员声纹阈值")
    parser.add_argument("--min-samples", type=int, default=20, help="每位成员触发校准的最低人工样本数")
    parser.add_argument("--apply", action="store_true", help="将校准出的独立阈值写入 config/voiceprints.json")
    args = parser.parse_args()

    cfg = load_config()
    conn = sqlite3.connect(f"file:{cfg.db_path}?mode=ro", uri=True)
    rows = conn.execute(
        """
        SELECT up_name, speaker_label, speaker_name, speaker_sim
          FROM subtitles
         WHERE speaker_locked = 1
           AND speaker_sim IS NOT NULL
           AND speaker_label IN ('host', 'mixed', 'external')
        """
    ).fetchall()

    by_member: Dict[str, List[Tuple[float, bool]]] = defaultdict(list)
    for up_name, label, spk_name, sim in rows:
        is_host = label in ("host", "mixed")
        key = (spk_name or "").strip() or (up_name or "").strip()
        if key:
            by_member[key].append((float(sim), is_host))

    print(f"已读取人工锁定声纹样本总计: {len(rows)} 条")
    if not rows:
        print("暂无足够的人工校对样本；请在本地 /admin 后台校对一批边界样本后再运行本脚本。")
        return

    vp_data = load_voiceprints(DEFAULT_VOICEPRINTS_PATH)
    global_thr = float(vp_data["thresholds"].get("host_thr", DEFAULT_THRESHOLDS["host_thr"]))
    recommended: Dict[str, Dict[str, float]] = {}

    for member, samples in sorted(by_member.items()):
        pos = sum(1 for _, y in samples if y)
        neg = len(samples) - pos
        base_f1, base_p, base_r, _ = evaluate_threshold(samples, global_thr)
        best = (base_f1, global_thr, base_p, base_r)
        for step in range(40, 63):
            thr = round(step / 100.0, 2)
            f1, p, r, _ = evaluate_threshold(samples, thr)
            if f1 > best[0] + 1e-6 or (abs(f1 - best[0]) <= 1e-6 and abs(thr - global_thr) < abs(best[1] - global_thr)):
                best = (f1, thr, p, r)
        print(
            f"  [{member}] 样本={len(samples)} (主播={pos}, 无关={neg}) | "
            f"当前({global_thr:.2f}) F1={base_f1:.3f} -> 建议({best[1]:.2f}) F1={best[0]:.3f} (P={best[2]:.3f}, R={best[3]:.3f})"
        )
        if len(samples) >= args.min_samples and pos >= 3 and neg >= 3:
            recommended[member] = {
                "host_thr": best[1],
                "peak_thr": round(min(0.70, best[1] + 0.02), 2),
                "smooth_floor": round(max(0.35, best[1] - 0.05), 2),
                "strong_neighbor": round(min(0.75, best[1] + 0.04), 2),
            }

    if args.apply:
        if not recommended:
            print(f"[SKIP] 尚无成员达到最低校准样本要求（min_samples={args.min_samples} 且正负样本各 >=3），未修改配置文件。")
            return
        per_m = dict(vp_data["thresholds"].get("per_member") or {})
        per_m.update(recommended)
        vp_data["thresholds"]["per_member"] = per_m
        save_voiceprints(vp_data, DEFAULT_VOICEPRINTS_PATH)
        print(f"[OK] 已写入 {len(recommended)} 位成员独立阈值至 {DEFAULT_VOICEPRINTS_PATH}")


if __name__ == "__main__":
    main()
