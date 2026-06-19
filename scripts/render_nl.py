#!/usr/bin/env python
"""Phase 0a -- (re)generate the Centaur-format NL from the raw SRO release.

The encoders live in src/sro_transfer/data/centaur_render.py. They expect the
SRO ``Individual_Measures`` CSVs. Point --sro-data-root at a checkout of the
Self_Regulation_Ontology repo's Data/ folder.

    python scripts/render_nl.py --sro-data-root /path/to/Self_Regulation_Ontology/Data \
        --source complete --full

Then build the correctness sidecars and copy output_nl/ to your Drive data_root.
Most users will not need this: the NL is generated once and lives on Drive.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "sro_transfer" / "data"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sro-data-root", required=True,
                    help="path to Self_Regulation_Ontology/Data")
    ap.add_argument("--source", choices=["complete", "retest"], default="complete")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--tasks", nargs="*", default=None)
    ap.add_argument("--out", default=None, help="output dir (default: ./output_nl)")
    args = ap.parse_args()

    import centaur_render as cr

    root = Path(args.sro_data_root)
    cr.DATA_ROOTS = {
        "complete": root / "Complete_02-16-2019" / "Individual_Measures",
        "retest": root / "Retest_02-16-2019" / "Individual_Measures",
    }
    if args.out:
        cr.OUT_DIR = Path(args.out)
        cr.OUT_DIR.mkdir(parents=True, exist_ok=True)

    tasks = args.tasks or list(cr.TASKS)
    for task in tasks:
        try:
            s = cr.write_task(task, args.source, args.full)
            print(f"  {s['task']:<46} {s['subjects']:>4} subj  -> {s['out']}")
        except Exception as e:  # noqa: BLE001
            print(f"  {task}: ERROR {e}")


if __name__ == "__main__":
    main()
