"""Eval v0: precision and recall per hazard class from hand labels and the stored results (read-only).

    python scripts/eval.py [--labels DIR] [--db PATH] [--tolerance SECONDS] [--drive ID]

Defaults: labels folder ``labels/`` (missing or empty is fine: recall is then "not measured"), the database
from ROADSCOUT_DB (``backend.config``), tolerance 2.0 s. It never writes the database, never calls a model
provider and never needs a key. Exit code 0 on success; 1 for a missing database or an unknown --drive; 2
for a bad label file (the message names the file and line) or a bad argument. See ``backend/evaluation.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.evaluation import (  # noqa: E402
    DEFAULT_TOLERANCE,
    LabelError,
    UnknownDriveError,
    evaluate,
    format_report,
    load_labels,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RoadScout eval v0 (read-only).")
    ap.add_argument("--labels", default=str(ROOT / "labels"), help="folder with one CSV per drive (default: labels/)")
    ap.add_argument("--db", default=None, help="SQLite file (default: ROADSCOUT_DB or data/roadscout.sqlite)")
    ap.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                    help="seconds added around a labeled hazard when matching (default: 2.0)")
    ap.add_argument("--drive", default=None, help="evaluate only this drive id")
    args = ap.parse_args(argv)
    db = Path(args.db) if args.db else config.db_path()
    try:
        labels = load_labels(args.labels)
        report = evaluate(db, labels, tolerance=args.tolerance, drive_id=args.drive)
    except LabelError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except UnknownDriveError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(format_report(report), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
