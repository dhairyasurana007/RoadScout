"""Build the read-only demo dataset (demo/demo.sqlite and demo/storage/...) from the real database and storage.

    python scripts/build_demo_dataset.py [--db PATH] [--storage DIR] [--out DIR] [--drive ID ...] [--force]
                                         [--skip-identical | --allow-identical]

Keeps only approved and reported hazards whose photo is a privacy-blurred file, copies only those blurred
photos, and writes a fresh database without any real path, detection, raw-video or unblurred-frame reference.
A privacy scan runs before anything is put in place; on a hit nothing is written. The source database is opened
read-only. Defaults: the database from ROADSCOUT_DB, the storage folder from ROADSCOUT_STORAGE_DIR, output
``demo/`` at the repo root. A non-empty output folder needs --force (it then replaces only demo.sqlite and
storage/ inside it). Exit code 0 on success, 1 for a runtime error (nothing is left behind), 2 for bad arguments.
See ``backend/demo_dataset.py``.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config  # noqa: E402
from backend.demo_dataset import DemoBuildError, build_demo_dataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build the privacy-safe read-only demo dataset.")
    ap.add_argument("--db", default=None, help="source SQLite file (default: ROADSCOUT_DB or data/roadscout.sqlite)")
    ap.add_argument("--storage", default=None, help="source storage folder (default: ROADSCOUT_STORAGE_DIR or data/storage)")
    ap.add_argument("--out", default=str(ROOT / "demo"), help="output folder (default: demo/ at the repo root)")
    ap.add_argument("--drive", action="append", default=None, metavar="ID",
                    help="keep only hazards seen on this drive id (repeat for several)")
    ap.add_argument("--force", action="store_true", help="replace demo.sqlite and storage/ in a non-empty output folder")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--skip-identical", action="store_true",
                       help="skip hazards whose photo is byte-identical to an unblurred original (nothing was "
                            "blurred) instead of failing the build")
    group.add_argument("--allow-identical", action="store_true",
                       help="accept and copy published images that are byte-identical to an unblurred original "
                            "(the blur step found no faces or plates); LOOK AT THEM before committing")
    args = ap.parse_args(argv)
    db = Path(args.db) if args.db else config.db_path()
    storage = Path(args.storage) if args.storage else config.storage_dir()
    try:
        report = build_demo_dataset(db, storage, args.out, drive_ids=args.drive, overwrite=args.force,
                                    skip_identical=args.skip_identical, allow_identical=args.allow_identical)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (DemoBuildError, OSError, sqlite3.Error) as e:  # includes FileNotFoundError, FileExistsError, DemoPrivacyError
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"demo dataset written to {args.out}")
    print(f"  hazards kept: {report.hazards_kept}")
    print(f"  hazards skipped: {len(report.hazards_skipped)}")
    for hazard_id, reason in report.hazards_skipped:
        print(f"    hazard {hazard_id}: {reason}")
    print(f"  candidates kept: {report.candidates_kept}")
    print(f"  drives kept: {report.drives_kept}")
    print(f"  images copied: {report.files_copied} ({report.bytes_copied} bytes)")
    if report.identical_copies:
        n = len({name for _, name in report.identical_copies})
        print(f"  {n} published images are unchanged by the blur step (no faces or plates detected); "
              "look at them before committing:")
        for hazard_id, name in report.identical_copies:
            print(f"    hazard {hazard_id}: {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
