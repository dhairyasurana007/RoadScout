"""Eval v0 (task 12): precision and recall per hazard class from hand labels and the stored results.

    from backend.evaluation import load_labels, evaluate, format_report
    labels = load_labels("labels")              # a missing folder or no CSVs is fine: no labels
    report = evaluate(db_path, labels, tolerance=2.0)
    print(format_report(report))

Command line: ``python scripts/eval.py [--labels DIR] [--db PATH] [--tolerance SECONDS] [--drive ID]``.
Everything here is read-only: the database is opened read-only, no provider is called, no key is needed.

Labels (optional, partial is fine). One CSV per drive in the labels folder, named ``<video filename
stem>.csv`` (``IMG_6513.csv`` for ``IMG_6513.MOV``, case-insensitive). Columns ``row_type, segment_id,
start, end, class, severity, hazard_id, note``. ``row_type`` is ``segment`` (a stretch you labelled
carefully) or ``hazard`` (one hazard inside a segment). ``start``/``end`` are ``mm:ss`` or ``h:mm:ss``
(seconds may have a fraction) of video time. Classes: pothole, debris in lane, damaged guardrail, fallen
or damaged sign, faded lane markings (case and spacing are ignored). Severity 1 to 3 (may be left empty).
Rows whose note starts with ``EXAMPLE ROW`` are ignored. A bad row raises ``LabelError`` (a
``ValueError``) naming the file and line: unknown row type or class, bad time or severity, end before
start, a duplicate or missing segment id, or a hazard outside the segment with its segment id.

``LabelSet.drives`` maps the lower-case file stem to ``DriveLabels(segments, hazards)``; ``Segment`` and
``LabelHazard`` hold times in seconds. ``LabelSet.has_hazard_labels`` is true when any hazard row exists.

Which drive a label file belongs to: the file stem equals the stem of the drive's ``video_filename`` (the
upload's file name, column added in task 12) or the drive id, ignoring case. Drives without a label file
are listed in ``EvalReport.skipped_drives``: skipped for label metrics, still counted for review metrics.

Matching. The candidates that count are the accepted ones (a ``verdicts`` row with ``accepted = 1``); their
class is ``candidates.hazard_class`` and their time is ``candidates.best_timestamp`` (seconds of video
time, no clock offset). A candidate matches a labeled hazard of the same class when its time is within
``[start - tolerance, end + tolerance]``. A label matched by several candidates counts once for recall;
every candidate counts once for precision.

Label metrics (only inside labeled segments):

- recall per class = labeled hazards matched / labeled hazards (``None``, printed n/a, with no labeled
  hazards: never 0);
- precision per class = accepted candidates matched to a label / accepted candidates inside labeled
  segments (a segment's exact [start, end] counts as inside; a candidate matched through the tolerance
  always counts as inside);
- false alarms = accepted candidates inside clean segments (segments with no hazard rows), also reported
  per hour of clean segment time (``None`` with no clean segment).

Review metrics (work without labels): per hazard class, approved / (approved + rejected) from the hazards'
review status. ``reported`` counts as approved; ``possibly_fixed`` counts by the status it had before
(``previous_status``); everything undecided (new) is left out of the precision and counted in
``EvalReport.unreviewed``. A hazard belongs to the drives of its candidates (``candidates.hazard_id``);
with ``drive_id`` only hazards with a candidate in that drive are counted.
"""
from __future__ import annotations

import csv
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

CLASSES = (
    "pothole",
    "debris in lane",
    "damaged guardrail",
    "fallen or damaged sign",
    "faded lane markings",
)
DEFAULT_TOLERANCE = 2.0
EXAMPLE_PREFIX = "EXAMPLE ROW"
REQUIRED_COLUMNS = ("row_type", "segment_id", "start", "end", "class", "severity", "hazard_id", "note")

_CLASS_BY_KEY = {" ".join(c.split()).casefold(): c for c in CLASSES}


class LabelError(ValueError):
    """A label file has an invalid row. The message names the file and line."""


class UnknownDriveError(ValueError):
    """``evaluate(drive_id=...)`` named a drive that is not in the database."""


# ----- labels ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Segment:
    segment_id: str
    start: float  # seconds of video time
    end: float


@dataclass(frozen=True)
class LabelHazard:
    segment_id: str
    start: float
    end: float
    hazard_class: str
    severity: int | None = None
    hazard_id: str | None = None


@dataclass
class DriveLabels:
    """The labels of one drive (one CSV file)."""

    segments: list[Segment] = field(default_factory=list)
    hazards: list[LabelHazard] = field(default_factory=list)

    def clean_segments(self) -> list[Segment]:
        used = {h.segment_id for h in self.hazards}
        return [s for s in self.segments if s.segment_id not in used]


@dataclass
class LabelSet:
    """All label files of a folder, keyed by lower-case file stem."""

    drives: dict[str, DriveLabels] = field(default_factory=dict)

    @property
    def has_hazard_labels(self) -> bool:
        return any(d.hazards for d in self.drives.values())

    @property
    def has_labels(self) -> bool:
        return any(d.segments or d.hazards for d in self.drives.values())


def normalise_class(text: str) -> str | None:
    """The canonical class name for ``text`` (case and spacing ignored), or None if it is not a class."""
    return _CLASS_BY_KEY.get(" ".join(str(text).split()).casefold())


def parse_time(text: str) -> float:
    """``mm:ss`` or ``h:mm:ss`` (seconds may have a fraction) as seconds. ValueError if malformed."""
    parts = str(text).strip().split(":")
    if len(parts) not in (2, 3) or not all(p.strip() for p in parts):
        raise ValueError(f"time {text!r} must look like mm:ss or h:mm:ss")
    try:
        whole = [int(p) for p in parts[:-1]]
        seconds = float(parts[-1])
    except ValueError:
        raise ValueError(f"time {text!r} must look like mm:ss or h:mm:ss") from None
    if any(n < 0 for n in whole) or not 0 <= seconds < 60:  # also rejects NaN
        raise ValueError(f"time {text!r} has a value out of range")
    if len(parts) == 3 and whole[1] >= 60:
        raise ValueError(f"time {text!r} has 60 or more minutes in the middle field")
    total = 0
    for n in whole:
        total = total * 60 + n
    return total * 60 + seconds


def _read_label_file(path: Path) -> DriveLabels:
    name = path.name

    def fail(line: int, message: str) -> LabelError:
        return LabelError(f"{name}, line {line}: {message}")

    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as e:
        raise LabelError(f"{name}: cannot read the file ({type(e).__name__})") from None
    labels = DriveLabels()
    header: list[str] | None = None
    ids: set[str] = set()
    pending: list[tuple[int, LabelHazard]] = []
    for lineno, row in enumerate(csv.reader(text.splitlines()), start=1):
        if not any(cell.strip() for cell in row):
            continue
        if header is None:
            header = [c.strip().lower() for c in row]
            missing = [c for c in REQUIRED_COLUMNS if c not in header]
            if missing:
                raise fail(lineno, f"the header is missing the column(s) {', '.join(missing)}")
            continue
        cells = {h: (row[i].strip() if i < len(row) else "") for i, h in enumerate(header)}
        if cells["note"].upper().startswith(EXAMPLE_PREFIX):
            continue
        kind = cells["row_type"].lower()
        seg_id = cells["segment_id"]
        if kind not in ("segment", "hazard"):
            raise fail(lineno, f"row_type must be 'segment' or 'hazard', got {cells['row_type']!r}")
        if not seg_id:
            raise fail(lineno, "segment_id is empty")
        try:
            start, end = parse_time(cells["start"]), parse_time(cells["end"])
        except ValueError as e:
            raise fail(lineno, str(e)) from None
        if end < start:
            raise fail(lineno, "end is before start")
        if kind == "segment":
            if seg_id in ids:
                raise fail(lineno, f"segment_id {seg_id!r} is used by two segment rows")
            ids.add(seg_id)
            labels.segments.append(Segment(seg_id, start, end))
            continue
        cls = normalise_class(cells["class"])
        if cls is None:
            raise fail(lineno, f"class {cells['class']!r} is not one of: {', '.join(CLASSES)}")
        severity: int | None = None
        if cells["severity"]:
            try:
                severity = int(cells["severity"])
            except ValueError:
                raise fail(lineno, f"severity must be 1, 2 or 3, got {cells['severity']!r}") from None
            if not 1 <= severity <= 3:
                raise fail(lineno, f"severity must be 1, 2 or 3, got {severity}")
        pending.append((lineno, LabelHazard(seg_id, start, end, cls, severity, cells["hazard_id"] or None)))
    by_id = {s.segment_id: s for s in labels.segments}
    for lineno, h in pending:
        seg = by_id.get(h.segment_id)
        if seg is None:
            raise fail(lineno, f"hazard row has segment_id {h.segment_id!r} but there is no segment row with it")
        if h.start < seg.start or h.end > seg.end:
            raise fail(lineno, f"the hazard lies outside segment {h.segment_id!r}")
        labels.hazards.append(h)
    return labels


def load_labels(labels_dir: str | Path | None = "labels") -> LabelSet:
    """Read every ``*.csv`` in ``labels_dir``. A missing folder or no CSVs gives an empty ``LabelSet``."""
    out = LabelSet()
    if labels_dir is None or not Path(labels_dir).is_dir():
        return out
    for path in sorted(Path(labels_dir).glob("*.csv"), key=lambda p: p.name.casefold()):
        out.drives[path.stem.casefold()] = _read_label_file(path)
    return out


# ----- report ------------------------------------------------------------------------------


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


@dataclass
class ClassMetrics:
    """Label-based numbers for one class (summed over the labeled drives)."""

    labeled: int = 0  # labeled hazards
    matched: int = 0  # of those, matched by an accepted candidate
    recall: float | None = None  # matched / labeled, None when nothing is labeled
    accepted_in_segments: int = 0  # accepted candidates inside labeled segments
    matched_accepted: int = 0  # of those, matched to a label
    precision: float | None = None  # matched_accepted / accepted_in_segments


@dataclass
class ReviewMetrics:
    """Review-based numbers for one class."""

    approved: int = 0
    rejected: int = 0
    precision: float | None = None  # approved / (approved + rejected), None when neither


@dataclass
class EvalReport:
    per_class: dict[str, ClassMetrics] = field(default_factory=dict)
    review: dict[str, ReviewMetrics] = field(default_factory=dict)
    unreviewed: int = 0  # hazards not approved or rejected, left out of the review precision
    false_alarms: int = 0  # accepted candidates inside clean segments
    clean_seconds: float = 0.0
    false_alarms_per_hour: float | None = None
    has_labels: bool = False  # at least one drive was evaluated against labels
    skipped_drives: list[str] = field(default_factory=list)  # drive ids without a label file
    tolerance: float = DEFAULT_TOLERANCE
    labeled_drive_ids: list[str] = field(default_factory=list)
    labeled_segment_count: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def totals(self) -> ClassMetrics:
        t = ClassMetrics()
        for m in self.per_class.values():
            t.labeled += m.labeled
            t.matched += m.matched
            t.accepted_in_segments += m.accepted_in_segments
            t.matched_accepted += m.matched_accepted
        t.recall = _ratio(t.matched, t.labeled)
        t.precision = _ratio(t.matched_accepted, t.accepted_in_segments)
        return t

    @property
    def review_totals(self) -> ReviewMetrics:
        t = ReviewMetrics()
        for m in self.review.values():
            t.approved += m.approved
            t.rejected += m.rejected
        t.precision = _ratio(t.approved, t.approved + t.rejected)
        return t


# ----- database (read-only) ----------------------------------------------------------------


def _connect_readonly(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"database not found: {path}")
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


def _stem(name: str | None) -> str:
    """File name minus its last extension, lower case (directories are ignored)."""
    base = str(name or "").replace("\\", "/").rsplit("/", 1)[-1]
    return (base.rsplit(".", 1)[0] if "." in base else base).casefold()


def _label_for_drive(labels: LabelSet, drive_id: str, filename: str | None) -> DriveLabels | None:
    for key in (_stem(filename), str(drive_id).casefold()):
        if key and key in labels.drives:
            return labels.drives[key]
    return None


def _inside(t: float, start: float, end: float, slack: float = 0.0) -> bool:
    return start - slack <= t <= end + slack


def _score_drive(lab: DriveLabels, cands: list[tuple[float, str]], tolerance: float,
                 per_class: dict[str, ClassMetrics]) -> tuple[int, float]:
    """Add one labeled drive's counts to ``per_class``. Returns (false alarms, clean seconds)."""
    clean = lab.clean_segments()
    matched_labels: set[int] = set()
    false_alarms = 0
    for t, cls in cands:
        m = per_class.setdefault(cls, ClassMetrics())
        hits = [i for i, h in enumerate(lab.hazards)
                if h.hazard_class == cls and _inside(t, h.start, h.end, tolerance)]
        if hits:
            matched_labels.update(hits)
            m.accepted_in_segments += 1
            m.matched_accepted += 1
        elif any(_inside(t, s.start, s.end) for s in lab.segments):
            m.accepted_in_segments += 1
        if any(_inside(t, s.start, s.end) for s in clean):
            false_alarms += 1
    for i, h in enumerate(lab.hazards):
        m = per_class.setdefault(h.hazard_class, ClassMetrics())
        m.labeled += 1
        if i in matched_labels:
            m.matched += 1
    return false_alarms, sum(s.end - s.start for s in clean)


def _review_kind(row: sqlite3.Row, has_previous: bool) -> str:
    status = str(row["status"])
    if status == "possibly_fixed" and has_previous:
        status = str(row["previous_status"] or "new")
    if status in ("approved", "reported"):
        return "approved"
    if status == "rejected":
        return "rejected"
    return "unreviewed"


def evaluate(
    db_path: str | Path,
    labels: LabelSet | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    drive_id: str | None = None,
) -> EvalReport:
    """Compute the report from the stored results and the labels (both optional). Read-only.

    ``drive_id`` limits both the label metrics and the review metrics to that drive. Raises
    ``FileNotFoundError`` for a missing database, ``UnknownDriveError`` for a ``drive_id`` that is not in
    it and ``ValueError`` for a bad tolerance."""
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not 0 <= tolerance < float("inf"):
        raise ValueError(f"tolerance must be a number of 0 or more, got {tolerance!r}")
    labels = labels if labels is not None else LabelSet()
    report = EvalReport(per_class={c: ClassMetrics() for c in CLASSES},
                        review={c: ReviewMetrics() for c in CLASSES}, tolerance=float(tolerance))
    conn = _connect_readonly(db_path)
    try:
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        drives = conn.execute("SELECT * FROM drives ORDER BY created_at, id").fetchall() if "drives" in tables else []
        has_filename = "drives" in tables and "video_filename" in _columns(conn, "drives")
        if drive_id is not None:
            drives = [d for d in drives if d["id"] == drive_id]
            if not drives:
                raise UnknownDriveError(f"unknown drive: {drive_id}")
        can_score = {"candidates", "verdicts"} <= tables

        used: set[str] = set()
        for d in drives:
            lab = _label_for_drive(labels, d["id"], d["video_filename"] if has_filename else None)
            if lab is None or not (lab.segments or lab.hazards):
                report.skipped_drives.append(d["id"])
                continue
            used.update(k for k, v in labels.drives.items() if v is lab)
            cands: list[tuple[float, str]] = []
            if can_score:
                for r in conn.execute(
                    "SELECT c.best_timestamp AS t, c.hazard_class AS cls FROM candidates c "
                    "JOIN verdicts v ON v.candidate_id = c.id WHERE c.drive_id = ? AND v.accepted = 1",
                    (d["id"],),
                ):
                    cands.append((float(r["t"]), normalise_class(r["cls"]) or str(r["cls"])))
            alarms, clean_s = _score_drive(lab, cands, report.tolerance, report.per_class)
            report.false_alarms += alarms
            report.clean_seconds += clean_s
            report.labeled_drive_ids.append(d["id"])
            report.labeled_segment_count += len(lab.segments)
        report.has_labels = bool(report.labeled_drive_ids)

        if drive_id is None:
            for key, lab in labels.drives.items():
                if key not in used and (lab.segments or lab.hazards):
                    report.notes.append(
                        f"label file {key}.csv matches no drive (name it after the uploaded video file or the drive id)")

        if "hazards" in tables:
            has_prev = "previous_status" in _columns(conn, "hazards")
            allowed: set[int] | None = None
            if drive_id is not None:
                allowed = {r[0] for r in conn.execute(
                    "SELECT DISTINCT hazard_id FROM candidates WHERE drive_id = ? AND hazard_id IS NOT NULL",
                    (drive_id,))} if "candidates" in tables else set()
            for h in conn.execute("SELECT * FROM hazards").fetchall():
                if allowed is not None and h["id"] not in allowed:
                    continue
                cls = normalise_class(h["hazard_class"]) or str(h["hazard_class"])
                m = report.review.setdefault(cls, ReviewMetrics())
                kind = _review_kind(h, has_prev)
                if kind == "approved":
                    m.approved += 1
                elif kind == "rejected":
                    m.rejected += 1
                else:
                    report.unreviewed += 1
    finally:
        conn.close()

    for m in report.per_class.values():
        m.recall = _ratio(m.matched, m.labeled)
        m.precision = _ratio(m.matched_accepted, m.accepted_in_segments)
    for m in report.review.values():
        m.precision = _ratio(m.approved, m.approved + m.rejected)
    report.false_alarms_per_hour = (report.false_alarms / (report.clean_seconds / 3600.0)
                                    if report.clean_seconds > 0 else None)
    return report


# ----- text --------------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]

    def fmt(cells: list[str]) -> str:
        return "  ".join(c.ljust(widths[0]) if i == 0 else c.rjust(widths[i]) for i, c in enumerate(cells))

    return [fmt(header), "  ".join("-" * w for w in widths), *(fmt(r) for r in rows)]


def format_report(report: EvalReport) -> str:
    """The report as plain text (tables and notes). Never raises for an empty report."""
    lines: list[str] = ["RoadScout eval v0", ""]
    lines.append(f"Drives with labels: {len(report.labeled_drive_ids)}, labeled segments: "
                 f"{report.labeled_segment_count}, tolerance: {report.tolerance:g} s")
    lines.append("")

    lines.append("Label-based metrics (only inside labeled segments)")
    rows = []
    for cls in list(CLASSES) + sorted(c for c in report.per_class if c not in CLASSES):
        m = report.per_class.get(cls) or ClassMetrics()
        rows.append([cls, str(m.labeled), str(m.matched), _pct(m.recall), str(m.accepted_in_segments),
                     str(m.matched_accepted), _pct(m.precision)])
    t = report.totals
    rows.append(["all classes", str(t.labeled), str(t.matched), _pct(t.recall), str(t.accepted_in_segments),
                 str(t.matched_accepted), _pct(t.precision)])
    lines += _table(["class", "labeled", "matched", "recall", "accepted", "correct", "precision"], rows)
    lines.append("")
    if t.labeled == 0:
        lines.append("recall: not measured (no labels)")
    else:
        lines.append(f"recall: {_pct(t.recall)} ({t.matched} of {t.labeled} labeled hazards)")
        for cls in CLASSES:
            if (report.per_class.get(cls) or ClassMetrics()).labeled == 0:
                lines.append(f"recall ({cls}): not measured (no labeled hazards of this class)")
    if not report.has_labels:
        lines.append("precision (labels): not measured (no labels)")
    else:
        lines.append(f"precision (labels): {_pct(t.precision)} ({t.matched_accepted} of "
                     f"{t.accepted_in_segments} accepted candidates inside labeled segments)")
    if report.false_alarms_per_hour is not None:
        lines.append(f"false alarms in clean segments: {report.false_alarms} over {report.clean_seconds / 60:.1f} min "
                     f"= {report.false_alarms_per_hour:.2f} per hour")
    else:
        lines.append(f"false alarms per hour: not measured (no clean segments; {report.false_alarms} counted)")
    lines.append("")

    lines.append("Review-based precision (approved / (approved + rejected))")
    rows = []
    for cls in list(CLASSES) + sorted(c for c in report.review if c not in CLASSES):
        m = report.review.get(cls) or ReviewMetrics()
        rows.append([cls, str(m.approved), str(m.rejected), _pct(m.precision)])
    rt = report.review_totals
    rows.append(["all classes", str(rt.approved), str(rt.rejected), _pct(rt.precision)])
    lines += _table(["class", "approved", "rejected", "precision"], rows)
    if rt.approved + rt.rejected == 0:
        lines.append("review precision: not measured (no approved or rejected hazards)")
    lines.append(f"not reviewed (new, left out of precision): {report.unreviewed}")
    lines.append("")

    if report.skipped_drives:
        lines.append(f"Drives without a label file (skipped for label metrics, counted in the review table): "
                     f"{len(report.skipped_drives)}")
        lines += [f"  {d}" for d in report.skipped_drives]
        lines.append("")
    for note in report.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines).rstrip() + "\n"
