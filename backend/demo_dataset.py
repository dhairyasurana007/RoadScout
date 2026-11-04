"""Demo dataset builder (task 23b): a privacy-safe, read-only copy of reviewed hazards for the hosted demo.

    from backend.demo_dataset import build_demo_dataset
    report = build_demo_dataset("data/roadscout.sqlite", "data/storage", "demo")

Command line: ``python scripts/build_demo_dataset.py [--db PATH] [--storage DIR] [--out DIR] [--drive ID ...]
[--force] [--skip-identical]``.

What is built (``<out>/demo.sqlite`` and ``<out>/storage/<drive id>/candidates/<file>``), nothing else:

- The source database is opened read-only and never modified. The output database is created fresh with the
  project's own ``init_db`` schema, then filled.
- Kept hazards: status in ``statuses`` (default approved and reported), optionally only those with a candidate in
  ``drive_ids``, AND whose ``photo_path`` is a servable BLURRED file: it exists, lies directly in
  ``<storage>/<drive id>/candidates/`` and is not (resolved) the ``best_frame_path`` or ``crop_path`` of any candidate
  (the rule of ``backend/hazards.py`` ``_servable``). Any other hazard is skipped with a reason in
  ``hazards_skipped`` and nothing of it is copied. A blurred crop is kept only when it passes the same test.
- Files copied: only the kept hazards' blurred photo (and crop). Never anything else from storage.
- Rows copied (ids and other columns preserved): the kept hazards; the candidates whose ``hazard_id`` is a kept
  hazard, with their verdicts; the drives of those candidates; ``app_meta``. Not copied: detections,
  detection_chunks, pipeline_runs, exports, other drives, candidates, verdicts and hazards.
- Paths: in the output ``hazards.photo_path`` / ``crop_path`` (and ``candidates.blurred_frame_path`` /
  ``blurred_crop_path`` of the copied files) are relative strings ``<path_prefix>/storage/<drive id>/candidates/<file>``
  with forward slashes (the photo routes resolve them against the working directory, the repo root locally and
  /app in the image). A candidate's blurred path that was not copied becomes NULL (and its ``blur_status`` NULL).
  The unblurred ``best_frame_path`` is the placeholder ``(removed)`` and ``crop_path`` NULL. In drives
  ``video_path`` and ``gpx_path`` are the placeholder ``(removed)``, ``video_filename`` NULL, ``video_deleted_at`` the source
  drive's finished_at, else updated_at, else created_at, else 1970-01-01T00:00:00+00:00, or ``now=`` when given (any
  non-empty value makes the UI say the raw video is gone), ``error`` and ``warning`` NULL.
- Privacy scan before anything is put in place; on a hit the staging folder is deleted and ``DemoPrivacyError`` is
  raised: no text value of the output database contains the source storage folder, the source database path, a
  drive-letter path (``C:\\``), ``/Users/`` or ``\\Users\\``; every file is ``demo.sqlite`` or
  ``storage/<drive id>/candidates/*.jpg|jpeg|png``; every image is referenced by the database; no copied image is
  byte-identical (sha256) to an original best-frame or crop file of the source. A frame with nothing to blur can
  come out byte-identical to its original: ``skip_identical=True`` skips such hazards (reason in
  ``hazards_skipped``) instead of failing, and ``allow_identical=True`` (a human-reviewed override) accepts and copies
  them, listing them in ``identical_copies`` as (hazard id, file name). It still copies only the blurred path, never
  an original path, and every other scan check stays. The two options together raise ``ValueError``.
- The build happens in a staging folder next to ``out_dir`` and is moved into place at the end, so a failure leaves
  nothing behind. A non-empty ``out_dir`` is refused unless ``overwrite=True``, which replaces only ``demo.sqlite``
  and ``storage/`` inside it. Rows and files are written in id order and nothing reads the clock, so two builds of the same
  source give byte-identical ``demo.sqlite`` files and photos.
"""
from __future__ import annotations

import hashlib
import re
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

from .db import HAZARD_STATUSES, get_connection, init_db
from .evaluation import _connect_readonly
from .hazards import CANDIDATES_DIR, _resolve, _servable

PLACEHOLDER = "(removed)"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
_DRIVE_LETTER_PATH = re.compile(r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]")


class DemoBuildError(RuntimeError):
    """The dataset could not be built (nothing is left behind)."""


class DemoPrivacyError(DemoBuildError):
    """The privacy scan found something that must not be in the demo (nothing is left behind)."""


@dataclass
class DemoBuildReport:
    hazards_kept: int = 0
    hazards_skipped: list[tuple[int, str]] = field(default_factory=list)  # (hazard id, reason)
    candidates_kept: int = 0
    drives_kept: int = 0
    files_copied: int = 0
    bytes_copied: int = 0
    out_dir: str = ""
    # (hazard id, file name) of published images that are byte-identical to an unblurred original: only filled when
    # allow_identical=True (the blur step found nothing to blur). A human should look at them before committing.
    identical_copies: list[tuple[int, str]] = field(default_factory=list)


# ----- helpers -----------------------------------------------------------------------------


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [r["name"] for r in conn.execute(f"PRAGMA table_info({table})")]


def _tables(conn: sqlite3.Connection) -> set[str]:
    return {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _clean_prefix(prefix: str) -> str:
    text = str(prefix).replace("\\", "/").strip("/")
    if not text or ".." in text.split("/") or _DRIVE_LETTER_PATH.search(text + "/") or text.startswith("~"):
        raise ValueError(f"path_prefix must be a short relative path like 'demo', got {prefix!r}")
    return text


def _path_forms(path: str | Path) -> set[str]:
    """Spellings of a path to look for in text: as given, resolved, with / and with \\."""
    forms: set[str] = set()
    given = Path(path)
    for p in ((given, given.resolve()) if given.is_absolute() else (given.resolve(),)):
        text = str(p)
        forms.update({text, text.replace("\\", "/"), text.replace("/", "\\")})
    return {f for f in forms if len(f) > 3}


def _skip_reason(stored: Any, root: Path, originals: set[Path]) -> str:
    """Why ``stored`` is not a servable blurred file (the file is never named)."""
    path = _resolve(stored)
    if path is None:
        return "no blurred photo is recorded"
    if path in originals:
        return "the photo is an unblurred original frame or crop"
    try:
        relative = path.relative_to(root)
    except ValueError:
        return "the photo is outside the storage folder"
    if len(relative.parts) != 3 or relative.parts[1] != CANDIDATES_DIR:
        return "the photo is not directly in a drive's candidates folder"
    return "the photo file is missing"


EPOCH_STAMP = "1970-01-01T00:00:00+00:00"


def _deleted_stamp(drive: dict[str, Any]) -> str:
    """``video_deleted_at`` for a copied drive when no ``now`` is given: the source row's own finished_at, else
    updated_at, else created_at, else a constant. It never reads the clock, so a rebuild gives identical bytes."""
    for column in ("finished_at", "updated_at", "created_at"):
        value = drive.get(column)
        if isinstance(value, str) and value.strip():
            return value
    return EPOCH_STAMP


def _insert(conn: sqlite3.Connection, table: str, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    cols = list(rows[0])
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})",
        [tuple(r[c] for c in cols) for r in rows],
    )


def _rel(prefix: str, drive: str, name: str) -> str:
    return f"{prefix}/storage/{drive}/{CANDIDATES_DIR}/{name}"


# ----- the privacy scan --------------------------------------------------------------------


def _scan(stage: Path, source_storage: Path, source_db: Path, originals_sha: set[str], prefix: str,
          allow_identical: bool = False) -> None:
    db_file = stage / "demo.sqlite"
    needles = _path_forms(source_storage) | _path_forms(source_db)
    referenced: set[Path] = set()
    conn = sqlite3.connect(f"{db_file.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        for table in sorted(_tables(conn)):
            if table.startswith("sqlite_"):
                continue
            for row in conn.execute(f"SELECT * FROM {table}"):
                for key in row.keys():
                    value = row[key]
                    if not isinstance(value, str):
                        continue
                    if any(n in value for n in needles):
                        raise DemoPrivacyError(f"privacy scan: {table}.{key} contains a source path")
                    if _DRIVE_LETTER_PATH.search(value) or "/Users/" in value or "\\Users\\" in value:
                        raise DemoPrivacyError(f"privacy scan: {table}.{key} contains a local path")
        wanted = f"{prefix}/storage/"
        for table, column in (("hazards", "photo_path"), ("hazards", "crop_path"),
                              ("candidates", "blurred_frame_path"), ("candidates", "blurred_crop_path")):
            for r in conn.execute(f"SELECT {column} AS v FROM {table} WHERE {column} IS NOT NULL"):
                value = str(r["v"])
                if not value.startswith(wanted):
                    raise DemoPrivacyError(f"privacy scan: {table}.{column} is not a demo-relative path")
                target = stage / "storage" / value[len(wanted):]
                if not target.is_file():
                    raise DemoPrivacyError(f"privacy scan: {table}.{column} points at a file that was not copied")
                referenced.add(target.resolve())
    finally:
        conn.close()

    allowed_parent = "candidates"
    for path in sorted(stage.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(stage)
        if rel.parts == ("demo.sqlite",):
            continue
        if not (len(rel.parts) == 4 and rel.parts[0] == "storage" and rel.parts[2] == allowed_parent
                and path.suffix.lower() in IMAGE_SUFFIXES):
            raise DemoPrivacyError("privacy scan: an unexpected file is in the output")
        if path.resolve() not in referenced:
            raise DemoPrivacyError("privacy scan: an image in the output is not referenced by the database")
        if not allow_identical and _sha256(path) in originals_sha:
            raise DemoPrivacyError(
                f"privacy scan: {path.name} is byte-identical to an unblurred original (look at it, then use allow_identical to accept it or skip_identical to skip it)")


# ----- main --------------------------------------------------------------------------------


def build_demo_dataset(
    db_path: str | Path,
    storage_dir: str | Path,
    out_dir: str | Path,
    *,
    drive_ids: Iterable[str] | None = None,
    statuses: Sequence[str] = ("approved", "reported"),
    path_prefix: str = "demo",
    overwrite: bool = False,
    skip_identical: bool = False,
    allow_identical: bool = False,
    now: datetime | str | None = None,
) -> DemoBuildReport:
    """Build ``<out_dir>/demo.sqlite`` and ``<out_dir>/storage/...`` from a source database and storage folder.

    Raises ``ValueError`` for bad arguments (unknown status, unknown drive id, bad prefix), ``FileNotFoundError`` for a
    missing source database, ``FileExistsError`` for a non-empty ``out_dir`` without ``overwrite``, and
    ``DemoBuildError`` / ``DemoPrivacyError`` when the build fails (the output is then not touched)."""
    if skip_identical and allow_identical:
        raise ValueError("skip_identical and allow_identical contradict each other; choose one")
    statuses = tuple(statuses)
    bad = [s for s in statuses if s not in HAZARD_STATUSES]
    if bad:
        raise ValueError(f"unknown hazard status {bad[0]!r}; choose from {', '.join(HAZARD_STATUSES)}")
    prefix = _clean_prefix(path_prefix)
    wanted_drives = None if drive_ids is None else [str(d) for d in drive_ids]
    fixed_stamp = now.isoformat() if isinstance(now, datetime) else (str(now) if now else None)

    source_db = Path(db_path)
    storage = Path(storage_dir)
    root = storage.resolve()
    out = Path(out_dir)
    o = out.resolve()
    replaced = (o / "storage").resolve()
    if o == root or root in o.parents or replaced == root or replaced in root.parents:
        raise ValueError("the output folder must not be the storage folder, inside it, or hold it as storage/")
    if out.exists() and not out.is_dir():
        raise DemoBuildError("the output path exists and is not a folder")
    if out.is_dir() and any(out.iterdir()) and not overwrite:
        raise FileExistsError("the output folder is not empty; use overwrite (--force) to replace demo.sqlite and storage/")

    src = _connect_readonly(source_db)
    try:
        tables = _tables(src)
        report = DemoBuildReport(out_dir=str(out))
        if wanted_drives is not None:
            known = {r["id"] for r in src.execute("SELECT id FROM drives")} if "drives" in tables else set()
            unknown = [d for d in wanted_drives if d not in known]
            if unknown:
                raise ValueError(f"unknown drive: {unknown[0]}")

        originals: set[Path] = set()
        if "candidates" in tables:
            for r in src.execute("SELECT best_frame_path, crop_path FROM candidates"):
                for value in (r["best_frame_path"], r["crop_path"]):
                    p = _resolve(value)
                    if p is not None:
                        originals.add(p)
        originals_sha = {_sha256(p) for p in sorted(originals) if p.is_file()}

        hazards = [dict(r) for r in src.execute("SELECT * FROM hazards ORDER BY id")] if "hazards" in tables else []
        drive_of_hazard: dict[int, set[str]] = {}
        if "candidates" in tables:
            for r in src.execute("SELECT DISTINCT hazard_id, drive_id FROM candidates WHERE hazard_id IS NOT NULL"):
                drive_of_hazard.setdefault(r["hazard_id"], set()).add(r["drive_id"])

        kept: list[dict[str, Any]] = []
        copies: dict[Path, tuple[str, str]] = {}  # resolved source file -> (drive id, file name)
        for h in hazards:
            if h["status"] not in statuses:
                continue
            if wanted_drives is not None and not (drive_of_hazard.get(h["id"], set()) & set(wanted_drives)):
                continue
            photo = _servable(h.get("photo_path"), root, originals)
            if photo is None:
                report.hazards_skipped.append((h["id"], _skip_reason(h.get("photo_path"), root, originals)))
                continue
            if skip_identical and _sha256(photo) in originals_sha:
                report.hazards_skipped.append((h["id"], "the photo is byte-identical to an unblurred original"))
                continue
            crop = _servable(h.get("crop_path"), root, originals)
            if crop is not None and skip_identical and _sha256(crop) in originals_sha:
                crop = None
            if allow_identical:  # a human reviewed these: list them, copy them (still the blurred path, never the original)
                for f in (photo, crop):
                    if f is not None and _sha256(f) in originals_sha:
                        report.identical_copies.append((h["id"], f.name))
            kept.append({"hazard": h, "photo": photo, "crop": crop})
            for f in (photo, crop):
                if f is not None:
                    copies[f] = (f.relative_to(root).parts[0], f.name)

        kept_ids = [k["hazard"]["id"] for k in kept]
        candidates: list[dict[str, Any]] = []
        verdicts: list[dict[str, Any]] = []
        drives: list[dict[str, Any]] = []
        app_meta: list[dict[str, Any]] = []
        if kept_ids and "candidates" in tables:
            marks = ",".join("?" for _ in kept_ids)
            candidates = [dict(r) for r in src.execute(
                f"SELECT * FROM candidates WHERE hazard_id IN ({marks}) ORDER BY id", kept_ids)]
            cand_ids = [c["id"] for c in candidates]
            if cand_ids and "verdicts" in tables:
                cm = ",".join("?" for _ in cand_ids)
                verdicts = [dict(r) for r in src.execute(
                    f"SELECT * FROM verdicts WHERE candidate_id IN ({cm}) ORDER BY candidate_id", cand_ids)]
            drive_list = sorted({c["drive_id"] for c in candidates})
            if drive_list and "drives" in tables:
                dm = ",".join("?" for _ in drive_list)
                drives = [dict(r) for r in src.execute(f"SELECT * FROM drives WHERE id IN ({dm}) ORDER BY id", drive_list)]
        if "app_meta" in tables:
            app_meta = [dict(r) for r in src.execute("SELECT * FROM app_meta ORDER BY key")]
    finally:
        src.close()

    out.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".demo-build-", dir=out.parent))
    try:
        # files first (blurred photos and crops only)
        for source_file, (drive, name) in sorted(copies.items(), key=lambda kv: (kv[1], str(kv[0]))):
            target = stage / "storage" / drive / CANDIDATES_DIR / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_file, target)
            report.files_copied += 1
            report.bytes_copied += target.stat().st_size

        rel_of = {f: _rel(prefix, drive, name) for f, (drive, name) in copies.items()}
        db_file = stage / "demo.sqlite"
        init_db(db_file)
        out_conn = get_connection(db_file)
        try:
            with out_conn:
                have = {t: set(_columns(out_conn, t)) for t in ("drives", "hazards", "candidates", "verdicts", "app_meta")}

                def only(table: str, row: dict[str, Any]) -> dict[str, Any]:
                    return {k: v for k, v in row.items() if k in have[table]}

                if app_meta:  # init_db wrote the default row; the source's rows replace it
                    out_conn.execute("DELETE FROM app_meta")
                    _insert(out_conn, "app_meta", [only("app_meta", r) for r in app_meta])

                drive_rows = []
                for d in drives:
                    row = only("drives", d)
                    row.update(video_path=PLACEHOLDER, gpx_path=PLACEHOLDER, video_deleted_at=fixed_stamp or _deleted_stamp(d),
                               error=None, warning=None)
                    if "video_filename" in have["drives"]:
                        row["video_filename"] = None
                    drive_rows.append(row)
                _insert(out_conn, "drives", drive_rows)

                hazard_rows = []
                for k in kept:
                    row = only("hazards", k["hazard"])
                    row["photo_path"] = rel_of[k["photo"]]
                    row["crop_path"] = rel_of[k["crop"]] if k["crop"] is not None else None
                    hazard_rows.append(row)
                _insert(out_conn, "hazards", hazard_rows)

                cand_rows = []
                for c in candidates:
                    row = only("candidates", c)
                    frame = rel_of.get(_resolve(c.get("blurred_frame_path"))) if c.get("blurred_frame_path") else None
                    crop = rel_of.get(_resolve(c.get("blurred_crop_path"))) if c.get("blurred_crop_path") else None
                    row["best_frame_path"] = PLACEHOLDER
                    row["crop_path"] = None
                    row["blurred_frame_path"] = frame
                    row["blurred_crop_path"] = crop
                    if frame is None:
                        row["blur_status"] = None
                        row["blur_note"] = None
                    cand_rows.append(row)
                _insert(out_conn, "candidates", cand_rows)
                _insert(out_conn, "verdicts", [only("verdicts", v) for v in verdicts])
        finally:
            out_conn.close()

        report.hazards_kept = len(kept)
        report.candidates_kept = len(candidates)
        report.drives_kept = len(drives)
        _scan(stage, storage, source_db, originals_sha, prefix, allow_identical)

        # put it in place: only demo.sqlite and storage/ are replaced in an existing folder
        if not out.exists():
            stage.replace(out)
        else:
            if overwrite:
                (out / "demo.sqlite").unlink(missing_ok=True)
                if (out / "storage").exists():
                    shutil.rmtree(out / "storage")
            elif any(out.iterdir()):
                raise FileExistsError("the output folder is not empty")
            (stage / "demo.sqlite").replace(out / "demo.sqlite")
            if (stage / "storage").exists():
                (stage / "storage").replace(out / "storage")
            shutil.rmtree(stage, ignore_errors=True)
        return report
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
