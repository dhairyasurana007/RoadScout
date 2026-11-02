"""Eval v1: ablation (SAM only vs SAM + Muse Spark), review precision, dedup accuracy, cost and processing time.

    python scripts/eval_v1.py [--labels DIR] [--db PATH] [--cost-log PATH] [--tolerance S] [--out FILE]

Prints the markdown report to stdout; a file is written only with --out (and only when the evaluation succeeded).
Read-only: the database is opened read-only, no provider is called and no key is needed. Defaults: labels folder
``labels/`` (missing or empty is fine), the database from ROADSCOUT_DB, the cost log from ROADSCOUT_COST_LOG
(default data/cost_log.jsonl), tolerance 2.0 s. Exit code 0 on success; 1 for a missing database or --out that
cannot be written; 2 for a bad label file (the message names the file and line) or a bad argument.
See ``backend/eval_v1.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.eval_v1 import evaluate_v1, render_markdown  # noqa: E402
from backend.evaluation import DEFAULT_TOLERANCE, LabelError, UnknownDriveError, load_labels  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RoadScout eval v1 (read-only).")
    ap.add_argument("--labels", default=str(ROOT / "labels"), help="folder with one CSV per drive (default: labels/)")
    ap.add_argument("--db", default=None, help="SQLite file (default: ROADSCOUT_DB or data/roadscout.sqlite)")
    ap.add_argument("--cost-log", default=None, help="JSONL cost log (default: ROADSCOUT_COST_LOG or data/cost_log.jsonl)")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help="seconds added around a labeled hazard when matching (default: 2.0)")
    ap.add_argument("--out", default=None, help="also write the markdown to this file")
    args = ap.parse_args(argv)
    db = Path(args.db) if args.db else config.db_path()
    cost_log = Path(args.cost_log) if args.cost_log else config.cost_log_path()
    try:
        labels = load_labels(args.labels)
        report = evaluate_v1(db, labels, tolerance=args.tolerance, cost_log=cost_log)
        markdown = render_markdown(report)
    except (LabelError, UnknownDriveError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    if args.out:
        try:
            Path(args.out).write_text(markdown, encoding="utf-8")
        except OSError as e:
            print(f"error: cannot write the output file ({type(e).__name__})", file=sys.stderr)
            return 1
    sys.stdout.write(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
