from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hoi_latent.evaluation.stage2 import summarize_stage2_repeat_runs, write_stage2_repeat_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize repeated Stage2 InterAct-style evaluation runs into mean +/- 95% CI tables."
    )
    parser.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--require-interact-comparable", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = summarize_stage2_repeat_runs(args.run_dirs)
    if args.require_interact_comparable and not report.all_interact_comparable:
        raise RuntimeError(
            "At least one repeat is not InterAct-comparable: "
            + ", ".join(report.blocking_reasons or ["unknown"])
        )
    write_stage2_repeat_report(report, args.out_dir)
    print(f"Wrote {args.out_dir / 'metrics' / 'report_table.json'}")


if __name__ == "__main__":
    main()
