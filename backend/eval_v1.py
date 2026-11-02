"""Eval v1 (task 20): ablation, review precision, dedup accuracy, cost and processing time, as a markdown report.

    from backend.evaluation import load_labels
    from backend.eval_v1 import evaluate_v1, render_markdown
    report = evaluate_v1(db_path, load_labels("labels"), tolerance=2.0, cost_log="data/cost_log.jsonl")
    print(render_markdown(report))

Command line: ``python scripts/eval_v1.py [--labels DIR] [--db PATH] [--cost-log PATH] [--tolerance S] [--out FILE]``.
Read-only: the database is opened read-only (never created, never written), no provider is called, no key is
needed, and a file is written only with ``--out``. Labels are optional; this builds on ``backend/evaluation.py``
(label files, drive matching, the matching rule and ``ReviewMetrics`` are shared with eval v0).

Ablation (SAM 3.1 alone vs SAM 3.1 plus the Muse Spark verdict), inside the labeled segments of the drives that
have a label file. "SAM only" counts every candidate of the drive as a detection, with or without a verdict.
"SAM + Muse Spark" counts only the candidates whose verdict was accepted (``verdicts.accepted = 1``). A
candidate's class is ``candidates.hazard_class`` and its time ``candidates.best_timestamp``; it matches a labeled
hazard of the same class within ``[start - tolerance, end + tolerance]``. Per class and overall (overall = summed
counts, not averaged rates): ``labeled``, ``matched`` (labels matched by at least one detection), ``recall``,
``detections`` (inside labeled segments; a matched detection always counts as inside), ``matched_detections``,
``precision`` = matched_detections / detections, ``false_alarms`` (detections inside clean segments, those with no
hazard row) and ``false_alarms_per_hour`` = false alarms / clean segment hours. Every ratio is None when its
denominator is 0. ``removed_by_verifier`` (on the SAM + Muse Spark row) is SAM-only detections minus verified ones.

Review: per class approved / (approved + rejected) from the hazards' review status (``reported`` counts as
approved, ``possibly_fixed`` counts by the status it had before, new is "unreviewed" and left out). It needs no
labels. ``review_rejected_per_hour`` is the rejected hazards per hour of video of the drives in scope.

Dedup accuracy: each labeled hazard that has a ``hazard_id`` is followed to its matching accepted candidate (the
nearest in time, same class) and from there to ``candidates.hazard_id`` (the hazard row). Labels with no match, or
whose candidate has no hazard row, are left out. A label id seen on at least two different drives is
``merged_correctly`` when all its candidates share one hazard row and ``split_wrongly`` when they are spread over two
or more. A hazard row that holds candidates of at least two distinct label ids is ``merged_wrongly`` (label ids
seen on a single drive only count here).

Cost and processing time: ``cost.total_usd`` is the sum of ``cost_usd`` over the cost log (JSONL; bad lines are
skipped), ``cost.minutes_of_video`` the summed ``drives.duration_seconds`` of the drives in scope, and
``usd_per_minute`` their quotient (None when the log is absent or empty, there are no video minutes, or ``drive_id``
limits the scope, because the log is not per drive). ``timing.seconds_per_minute`` is the sum of
``finished_at - created_at`` over the drives that have a ``finished_at``, divided by the video minutes of those
same finished drives (None when no drive finished or they have no video minutes).
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .evaluation import (
    CLASSES,
    DEFAULT_TOLERANCE,
    ClassMetrics,
    DriveLabels,
    LabelSet,
    ReviewMetrics,
    UnknownDriveError,
    _columns,
    _connect_readonly,
    _inside,
    _label_for_drive,
    _ratio,
    _review_kind,
    _score_drive,
    normalise_class,
)

OVERALL = "overall"


# ----- report ------------------------------------------------------------------------------


@dataclass
class AblationRow:
    labeled: int = 0
    matched: int = 0
    recall: float | None = None
    detections: int = 0  # detections inside labeled segments
    matched_detections: int = 0  # of those, matched to a label
    precision: float | None = None
    false_alarms: int = 0  # detections inside clean segments
    false_alarms_per_hour: float | None = None
    removed_by_verifier: int | None = None  # only on the "sam_muse" row


@dataclass
class DedupReport:
    merged_correctly: int = 0
    split_wrongly: int = 0
    merged_wrongly: int = 0
    repeated_labels: int = 0  # label ids seen on at least two drives with a matched candidate (the denominator)


@dataclass
class CostReport:
    total_usd: float | None = None
    minutes_of_video: float = 0.0
    usd_per_minute: float | None = None
    entries: int = 0
    by_operation: dict[str, float] = field(default_factory=dict)


@dataclass
class TimingReport:
    seconds_per_minute: float | None = None
    total_seconds: float = 0.0
    finished_drives: int = 0


@dataclass
class EvalV1Report:
    ablation: dict[str, dict[str, AblationRow]] = field(default_factory=dict)
    review: dict[str, ReviewMetrics] = field(default_factory=dict)
    unreviewed: int = 0
    dedup: DedupReport = field(default_factory=DedupReport)
    cost: CostReport = field(default_factory=CostReport)
    timing: TimingReport = field(default_factory=TimingReport)
    has_labels: bool = False
    skipped_drives: list[str] = field(default_factory=list)
    clean_seconds: float = 0.0
    review_rejected_per_hour: float | None = None
    tolerance: float = DEFAULT_TOLERANCE
    drive_count: int = 0
    labeled_drive_ids: list[str] = field(default_factory=list)
    labeled_segment_count: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def labeled_hazards(self) -> int:
        return self.ablation[OVERALL]["sam_only"].labeled if OVERALL in self.ablation else 0

    @property
    def review_totals(self) -> ReviewMetrics:
        t = ReviewMetrics()
        for m in self.review.values():
            t.approved += m.approved
            t.rejected += m.rejected
        t.precision = _ratio(t.approved, t.approved + t.rejected)
        return t


# ----- computing ---------------------------------------------------------------------------


def _row(m: ClassMetrics, false_alarms: int, clean_seconds: float) -> AblationRow:
    return AblationRow(
        labeled=m.labeled,
        matched=m.matched,
        recall=_ratio(m.matched, m.labeled),
        detections=m.accepted_in_segments,
        matched_detections=m.matched_accepted,
        precision=_ratio(m.matched_accepted, m.accepted_in_segments),
        false_alarms=false_alarms,
        false_alarms_per_hour=(false_alarms / (clean_seconds / 3600.0)) if clean_seconds > 0 else None,
    )


def _parse_time(text: Any) -> datetime | None:
    try:
        return datetime.fromisoformat(str(text).strip().replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _read_cost_log(path: str | Path | None) -> tuple[float, int, dict[str, float]]:
    """(total USD, entries, USD per operation) of a JSONL cost log; (0, 0, {}) when it is absent or unreadable.
    Lines that are not JSON objects with a non-negative numeric ``cost_usd`` are skipped."""
    if path is None:
        return 0.0, 0, {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return 0.0, 0, {}
    total, count, by_op = 0.0, 0, {}
    for line in lines:
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if not isinstance(entry, dict) or not _is_number(entry.get("cost_usd")) or entry["cost_usd"] < 0:
            continue
        total += float(entry["cost_usd"])
        count += 1
        op = str(entry.get("operation") or "other")
        by_op[op] = by_op.get(op, 0.0) + float(entry["cost_usd"])
    return total, count, by_op


def _dedup(
    labeled: list[tuple[str, DriveLabels, list[dict[str, Any]]]], tolerance: float
) -> DedupReport:
    """Count merged_correctly / split_wrongly / merged_wrongly (see the module docstring)."""
    # label id -> list of (drive id, hazard row id) for its matched candidates
    by_label: dict[str, list[tuple[str, int]]] = {}
    for drive_id, lab, cands in labeled:
        accepted = [c for c in cands if c["accepted"] and c["hazard_id"] is not None]
        for h in lab.hazards:
            if not h.hazard_id:
                continue
            best = None
            for c in accepted:
                if c["cls"] != h.hazard_class or not _inside(c["t"], h.start, h.end, tolerance):
                    continue
                dist = max(h.start - c["t"], c["t"] - h.end, 0.0)
                if best is None or (dist, c["id"]) < (best[0], best[1]["id"]):
                    best = (dist, c)
            if best is not None:
                by_label.setdefault(h.hazard_id, []).append((drive_id, int(best[1]["hazard_id"])))
    out = DedupReport()
    rows: dict[int, set[str]] = {}
    for label_id, pairs in by_label.items():
        for _, row_id in pairs:
            rows.setdefault(row_id, set()).add(label_id)
        if len({d for d, _ in pairs}) >= 2:
            out.repeated_labels += 1
            if len({r for _, r in pairs}) == 1:
                out.merged_correctly += 1
            else:
                out.split_wrongly += 1
    out.merged_wrongly = sum(1 for ids in rows.values() if len(ids) >= 2)
    return out


def evaluate_v1(
    db_path: str | Path,
    labels: LabelSet | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
    cost_log: str | Path | None = None,
    drive_id: str | None = None,
) -> EvalV1Report:
    """Compute the eval v1 report (read-only). See the module docstring.

    Raises ``FileNotFoundError`` for a missing database, ``UnknownDriveError`` for a ``drive_id`` that is not in it
    and ``ValueError`` for a bad tolerance."""
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not 0 <= tolerance < float("inf"):
        raise ValueError(f"tolerance must be a number of 0 or more, got {tolerance!r}")
    labels = labels if labels is not None else LabelSet()
    report = EvalV1Report(tolerance=float(tolerance))
    conn = _connect_readonly(db_path)
    try:
        tables = {r["name"] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        drives: list[sqlite3.Row] = (
            conn.execute("SELECT * FROM drives ORDER BY created_at, id").fetchall() if "drives" in tables else []
        )
        drive_cols = _columns(conn, "drives") if "drives" in tables else set()
        if drive_id is not None:
            drives = [d for d in drives if d["id"] == drive_id]
            if not drives:
                raise UnknownDriveError(f"unknown drive: {drive_id}")
        report.drive_count = len(drives)
        has_filename = "video_filename" in drive_cols
        has_cands = "candidates" in tables
        has_verdicts = "verdicts" in tables

        sam_counts: dict[str, ClassMetrics] = {}
        muse_counts: dict[str, ClassMetrics] = {}
        sam_fa: dict[str, int] = {}
        muse_fa: dict[str, int] = {}
        labeled_inputs: list[tuple[str, DriveLabels, list[dict[str, Any]]]] = []
        used: set[str] = set()

        for d in drives:
            lab = _label_for_drive(labels, d["id"], d["video_filename"] if has_filename else None)
            if lab is None or not (lab.segments or lab.hazards):
                report.skipped_drives.append(d["id"])
                continue
            used.update(k for k, v in labels.drives.items() if v is lab)
            cands: list[dict[str, Any]] = []
            if has_cands:
                join = "LEFT JOIN verdicts v ON v.candidate_id = c.id" if has_verdicts else ""
                acc = "v.accepted" if has_verdicts else "NULL"
                for r in conn.execute(
                    f"SELECT c.id AS id, c.best_timestamp AS t, c.hazard_class AS cls, c.hazard_id AS hazard_id, "
                    f"{acc} AS accepted FROM candidates c {join} WHERE c.drive_id = ? ORDER BY c.id", (d["id"],)
                ):
                    cands.append({
                        "id": r["id"], "t": float(r["t"]), "cls": normalise_class(r["cls"]) or str(r["cls"]),
                        "hazard_id": r["hazard_id"], "accepted": r["accepted"] == 1,
                    })
            both = [(c["t"], c["cls"]) for c in cands]
            verified = [(c["t"], c["cls"]) for c in cands if c["accepted"]]
            _, clean_s = _score_drive(lab, both, report.tolerance, sam_counts)
            _score_drive(lab, verified, report.tolerance, muse_counts)
            clean = lab.clean_segments()
            for t, cls in both:
                if any(_inside(t, s.start, s.end) for s in clean):
                    sam_fa[cls] = sam_fa.get(cls, 0) + 1
            for t, cls in verified:
                if any(_inside(t, s.start, s.end) for s in clean):
                    muse_fa[cls] = muse_fa.get(cls, 0) + 1
            report.clean_seconds += clean_s
            report.labeled_drive_ids.append(d["id"])
            report.labeled_segment_count += len(lab.segments)
            labeled_inputs.append((d["id"], lab, cands))
        report.has_labels = bool(report.labeled_drive_ids)
        if drive_id is None:
            for key, lab in labels.drives.items():
                if key not in used and (lab.segments or lab.hazards):
                    report.notes.append(
                        f"label file {key}.csv matches no drive (name it after the uploaded video file or the drive id)")

        # ablation rows: the five classes plus the overall row (summed counts)
        names = list(CLASSES) + sorted((set(sam_counts) | set(muse_counts)) - set(CLASSES))
        for name in names:
            sam_row = _row(sam_counts.get(name, ClassMetrics()), sam_fa.get(name, 0), report.clean_seconds)
            muse_row = _row(muse_counts.get(name, ClassMetrics()), muse_fa.get(name, 0), report.clean_seconds)
            muse_row.removed_by_verifier = max(0, sam_row.detections - muse_row.detections)
            if name in CLASSES:
                report.ablation[name] = {"sam_only": sam_row, "sam_muse": muse_row}
            else:  # a class outside the five only shows up in the overall sums
                continue
        for counts, fa, key in ((sam_counts, sam_fa, "sam_only"), (muse_counts, muse_fa, "sam_muse")):
            total = ClassMetrics()
            for m in counts.values():
                total.labeled += m.labeled
                total.matched += m.matched
                total.accepted_in_segments += m.accepted_in_segments
                total.matched_accepted += m.matched_accepted
            row = _row(total, sum(fa.values()), report.clean_seconds)
            report.ablation.setdefault(OVERALL, {})[key] = row
        overall = report.ablation[OVERALL]
        overall["sam_muse"].removed_by_verifier = max(0, overall["sam_only"].detections - overall["sam_muse"].detections)

        report.dedup = _dedup(labeled_inputs, report.tolerance)

        # review precision (no labels needed)
        report.review = {c: ReviewMetrics() for c in CLASSES}
        if "hazards" in tables:
            has_prev = "previous_status" in _columns(conn, "hazards")
            allowed: set[int] | None = None
            if drive_id is not None:
                allowed = {r[0] for r in conn.execute(
                    "SELECT DISTINCT hazard_id FROM candidates WHERE drive_id = ? AND hazard_id IS NOT NULL",
                    (drive_id,))} if has_cands else set()
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
        for m in report.review.values():
            m.precision = _ratio(m.approved, m.approved + m.rejected)

        # video minutes, cost and processing time
        minutes = 0.0
        seconds = 0.0
        finished = 0
        finished_minutes = 0.0  # video minutes of the drives that finished; the processing-time denominator
        if "duration_seconds" in drive_cols:
            for d in drives:
                if _is_number(d["duration_seconds"]) and d["duration_seconds"] > 0:
                    minutes += float(d["duration_seconds"]) / 60.0
        if "finished_at" in drive_cols and "created_at" in drive_cols:
            for d in drives:
                start, end = _parse_time(d["created_at"]), _parse_time(d["finished_at"]) if d["finished_at"] else None
                if start is None or end is None:
                    continue
                try:
                    taken = (end - start).total_seconds()
                except TypeError:  # naive and aware timestamps mixed
                    continue
                if taken >= 0:
                    seconds += taken
                    finished += 1
                    dur = d["duration_seconds"] if "duration_seconds" in drive_cols else None
                    if _is_number(dur) and dur > 0:
                        finished_minutes += float(dur) / 60.0
    finally:
        conn.close()

    report.cost.minutes_of_video = minutes
    total, entries, by_op = _read_cost_log(cost_log)
    report.cost.entries = entries
    if entries:
        report.cost.total_usd = total
        report.cost.by_operation = by_op
        if minutes > 0 and drive_id is None:
            report.cost.usd_per_minute = total / minutes
    report.timing.total_seconds = seconds
    report.timing.finished_drives = finished
    if finished and finished_minutes > 0:
        report.timing.seconds_per_minute = seconds / finished_minutes
    rejected = report.review_totals.rejected
    report.review_rejected_per_hour = rejected / (minutes / 60.0) if minutes > 0 else None
    return report


# ----- markdown ----------------------------------------------------------------------------


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _md_table(header: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(" --- " if i == 0 else " ---: " for i in range(len(header))) + "|",
        *("| " + " | ".join(r) + " |" for r in rows),
    ]


def _ablation_table(report: EvalV1Report, key: str) -> list[str]:
    rows = []
    for name in list(CLASSES) + [OVERALL]:
        r = report.ablation[name][key]
        rows.append([name, str(r.labeled), str(r.matched), _pct(r.recall), str(r.detections),
                     str(r.matched_detections), _pct(r.precision), str(r.false_alarms), _num(r.false_alarms_per_hour)])
    return _md_table(["class", "labeled", "matched", "recall", "detections", "correct", "precision",
                      "false alarms", "false alarms/hour"], rows)


def render_markdown(report: EvalV1Report) -> str:
    """The report as markdown (plain ASCII-safe text: no paths, no image data, ``n/a`` for anything not measured)."""
    o = report.ablation[OVERALL]
    lines: list[str] = ["# RoadScout eval v1", ""]
    lines.append(f"- drives in scope: {report.drive_count} ({len(report.labeled_drive_ids)} with labels used, "
                 f"{len(report.skipped_drives)} without)")
    lines.append(f"- labeled segments: {report.labeled_segment_count}, matching tolerance: {report.tolerance:g} s")
    if o["sam_only"].labeled == 0:
        lines.append("- labeled hazards: 0 -> recall not measured")
    else:
        lines.append(f"- labeled hazards: {o['sam_only'].labeled} -> recall {_pct(o['sam_only'].recall)} (SAM only), "
                     f"{_pct(o['sam_muse'].recall)} (SAM + Muse Spark)")
    if not report.has_labels:
        lines.append("- precision and false alarms against labels: not measured (no labels)")
    lines.append(f"- clean segment time: {report.clean_seconds / 60:.1f} min")
    lines.append("")

    lines += ["## Ablation: SAM only vs SAM + Muse Spark", "",
              "Counts are inside the labeled segments of the drives that have a label file. A detection is a "
              "candidate; SAM + Muse Spark keeps only the candidates whose verdict was accepted.", ""]
    lines += ["### SAM only", ""] + _ablation_table(report, "sam_only") + [""]
    lines += ["### SAM + Muse Spark", ""] + _ablation_table(report, "sam_muse") + [""]
    removed = o["sam_muse"].removed_by_verifier or 0
    lost = max(0, o["sam_only"].matched_detections - o["sam_muse"].matched_detections)
    lines.append(f"Removed by the verifier: {removed} detections inside labeled segments, "
                 f"{lost} of them matched a labeled hazard.")
    if o["sam_only"].false_alarms_per_hour is None:
        lines.append("False alarms per hour: not measured (no clean segments).")
    lines.append("")

    lines += ["## Review precision (approved / (approved + rejected))", ""]
    rows = []
    for name in list(CLASSES) + sorted(c for c in report.review if c not in CLASSES):
        m = report.review.get(name) or ReviewMetrics()
        rows.append([name, str(m.approved), str(m.rejected), _pct(m.precision)])
    t = report.review_totals
    rows.append([OVERALL, str(t.approved), str(t.rejected), _pct(t.precision)])
    lines += _md_table(["class", "approved", "rejected", "precision"], rows) + [""]
    lines.append(f"Not reviewed (new, left out of the precision): {report.unreviewed}")
    lines.append(f"Rejected hazards per hour of video: {_num(report.review_rejected_per_hour)}")
    if t.approved + t.rejected == 0:
        lines.append("Review precision: not measured (no approved or rejected hazards).")
    lines.append("")

    d = report.dedup
    lines += ["## Dedup accuracy (hazards seen on repeated drives)", ""]
    lines.append(f"- merged correctly: {d.merged_correctly}")
    lines.append(f"- split wrongly: {d.split_wrongly}")
    lines.append(f"- merged wrongly: {d.merged_wrongly}")
    if d.repeated_labels == 0 and d.merged_wrongly == 0:
        lines.append("")
        lines.append("Dedup accuracy: not measured (no labeled hazard was matched on two or more drives).")
    lines.append("")

    c = report.cost
    lines += ["## Cost and processing time", ""]
    lines.append(f"- video minutes in scope: {c.minutes_of_video:.1f}")
    lines.append(f"- total cost (cost log): {_usd(c.total_usd)}")
    lines.append(f"- cost per minute of video: {_usd(c.usd_per_minute)}")
    for op, usd in sorted(c.by_operation.items()):
        lines.append(f"  - cost of {op}: {_usd(usd)}")
    if c.total_usd is None:
        lines.append("- cost: not measured (no cost log entries)")
    lines.append(f"- processing time per minute of video: {_num(report.timing.seconds_per_minute, 1)} s "
                 f"({report.timing.finished_drives} finished drives)")
    if report.timing.seconds_per_minute is None:
        lines.append("- processing time: not measured (no finished drive or no video minutes)")
    lines.append("")

    if report.skipped_drives:
        lines.append(f"Drives without a label file (skipped for the ablation and dedup, still counted for review, "
                     f"cost and processing): {len(report.skipped_drives)}")
        lines += [f"- {drive}" for drive in report.skipped_drives]
        lines.append("")
    for note in report.notes:
        lines.append(f"Note: {note}")
    text = "\n".join(lines).rstrip() + "\n"
    return text.encode("cp1252", "replace").decode("cp1252")
