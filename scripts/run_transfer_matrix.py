#!/usr/bin/env python
"""Phase 0c -- the handcrafted directed transfer matrix T[A, B] (the decision gate).

    python scripts/run_transfer_matrix.py --config configs/default.yaml --subset starting_subset

Writes results/transfer_matrix.csv, results/transfer_hubs.csv. Reuses
results/reliability_task.csv if present (run run_reliability.py first) to
reliability-normalize. No GPU.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sro_transfer.diagnostics import load_dvs
from sro_transfer.diagnostics.transfer_matrix import (
    domain_structure,
    hub_ranking,
    transfer_matrix,
)
from sro_transfer.utils import load_config, load_tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--subset", default="starting_subset",
                    help="named subset in configs/tasks.yaml, or 'all'")
    args = ap.parse_args()
    cfg = load_config(args.config)
    taxonomy = load_tasks()

    dvs = load_dvs(cfg["paths"]["dv_complete"])
    present = {c.split(".")[0] for c in dvs.columns}

    if args.subset == "all":
        tasks = sorted(taxonomy["tasks"])
    else:
        tasks = taxonomy["subsets"][args.subset]
    tasks = [t for t in tasks if t in present]
    missing = [t for t in (taxonomy["subsets"].get(args.subset, []) if args.subset != "all" else [])
               if t not in present]
    if missing:
        print(f"NOTE: no DV columns for {missing} (check DV prefix names)")
    print(f"transfer matrix over {len(tasks)} tasks: {tasks}")

    # reliability normalization (optional)
    reliability = {}
    rel_fp = Path(cfg["paths"]["results"]) / "reliability_task.csv"
    if rel_fp.exists():
        rt = pd.read_csv(rel_fp, index_col=0)
        col = "mean_icc" if "mean_icc" in rt.columns else rt.columns[-1]
        reliability = rt[col].to_dict()

    T = transfer_matrix(dvs, tasks, reliability=reliability, value="norm")
    hubs = hub_ranking(T)
    domain = {t: taxonomy["tasks"][t]["domain"] for t in tasks}
    struct = domain_structure(T, domain)

    out = Path(cfg["paths"]["results"])
    out.mkdir(parents=True, exist_ok=True)
    T.to_csv(out / "transfer_matrix.csv")
    hubs.to_csv(out / "transfer_hubs.csv")

    print("\nHub ranking (mean transfer in/out, reliability-normalized):")
    print(hubs.round(3).to_string())
    print(f"\nWithin-domain mean transfer:  {struct['within_domain_mean']:.3f} (n={struct['n_within']})")
    print(f"Across-domain mean transfer:  {struct['across_domain_mean']:.3f} (n={struct['n_across']})")
    print("\nDECISION GATE: if the whole matrix ~ 0, task-based individual")
    print("differences barely transfer -> shrink scope before the 8B work.")
    print(f"\nsaved -> {out/'transfer_matrix.csv'} , {out/'transfer_hubs.csv'}")


if __name__ == "__main__":
    main()
