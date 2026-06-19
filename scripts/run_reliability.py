#!/usr/bin/env python
"""Phase 0b -- compute the test-retest reliability ceiling.

    python scripts/run_reliability.py --config configs/default.yaml

Writes results/reliability_dv.csv and results/reliability_task.csv.
Needs only the two scalar-DV CSVs (no GPU, no NL data).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sro_transfer.diagnostics import load_dvs, per_task_reliability, test_retest
from sro_transfer.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)

    complete = load_dvs(cfg["paths"]["dv_complete"])
    retest = load_dvs(cfg["paths"]["dv_retest"])
    print(f"complete: {complete.shape}, retest: {retest.shape}")

    rel = test_retest(complete, retest)
    by_task = per_task_reliability(rel)

    out = Path(cfg["paths"]["results"])
    out.mkdir(parents=True, exist_ok=True)
    rel.to_csv(out / "reliability_dv.csv", index=False)
    by_task.to_csv(out / "reliability_task.csv")

    print(f"\n{len(rel)} DVs scored. Per-task mean ICC (top/bottom):")
    print(by_task.head(12).round(3).to_string())
    print("...")
    print(by_task.tail(6).round(3).to_string())
    print(f"\nsaved -> {out/'reliability_dv.csv'} , {out/'reliability_task.csv'}")


if __name__ == "__main__":
    main()
