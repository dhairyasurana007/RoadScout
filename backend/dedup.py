"""Hazard deduplication (task 14): merge candidates of one real-world hazard, within and across drives.

    from backend.dedup import deduplicate_drive, load_hazards
    summary = deduplicate_drive(drive_id)        # after verification; also runs as pipeline stage 'deduplicating'
    hazards = load_hazards()                     # every hazard, ordered by id

``deduplicate_drive(drive_id, *, db_path=None, distance_m=None, heading_tolerance_deg=None)``
returns a ``DedupSummary``. ``rebuild_hazards(*, db_path=None, distance_m=None,
heading_tolerance_deg=None)`` does the same work without naming a drive (``deduplicate_drive`` is
``rebuild_hazards`` plus a summary about one drive). No network calls, no keys.

Which candidates take part (everything else is ignored and has ``hazard_id`` NULL): the verdict is
``accepted`` and ``is_hazard`` (task 9) AND ``geotag_status = 'ok'`` with a finite ``lat`` in
[-90, 90] and ``lon`` in [-180, 180] (task 8) AND the privacy blur did not fail (task 15: ``blur_status`` is not ``'failed'``; such a candidate is
SKIPPED, never merged, never shown; it counts in ``skipped_count`` and keeps ``hazard_id`` NULL).
Candidates are not required to have a blurred image, nor do the files have to exist: the hazard takes the
blurred paths as they are (NULL when there are none) and NEVER falls back to the unblurred ones. The class used is the verdict's ``hazard_class`` (the
class Muse Spark confirmed, as the results page shows it).

Merge rule. Two participating candidates are *linked* when all hold:

- same class;
- great-circle (haversine, Earth radius 6 371 008.8 m, accurate to well under a metre over 25 m at
  Virginia/Maryland latitudes) distance <= ``distance_m`` (inclusive; default 25, env
  ``ROADSCOUT_DEDUP_DISTANCE_M``);
- both have a heading and the difference in direction of travel <= ``heading_tolerance_deg``
  (inclusive, wrap-around aware: 350 and 10 differ by 20; default 45, env
  ``ROADSCOUT_DEDUP_HEADING_TOLERANCE_DEG``, 0 to 180, 180 means "any direction"). A candidate whose
  heading is unknown (NULL or not finite) is never linked to anything: it is a hazard on its own.
  This keeps the other side of a divided road (about 180 degrees apart) separate.

A hazard is a *connected component* of this link relation (single-link clustering over ALL
participating candidates of the class, every drive at once). Consequences, by design: the result does
not depend on processing order or on the order of rows; and chaining is allowed, so a row of
candidates 20 m apart forms one hazard even though its ends are more than 25 m apart (and a heading
can drift by up to 45 degrees per link). The members are always re-clustered from scratch.

Hazard values (table ``hazards``, ``backend/db.py``):

- ``sightings``: the number of DISTINCT drives among the members. Several candidates of one drive
  (an object the tracker lost at a chunk boundary, task 8) are one sighting. ``candidate_count`` is
  the number of member candidates.
- ``severity``: the largest verdict severity among the members.
- ``confidence``: noisy-OR over drives of each drive's best member confidence, ``1 - prod(1 - c_d)``,
  limited to [best single confidence, 1]. One drive gives exactly its best confidence; each further
  drive that saw it raises the value, and it never reaches above 1 or below the best single one.
- ``direction``: circular mean of the members' headings in degrees, [0, 360), rounded to 6
  decimals (the best-scoring member's heading in the degenerate case of opposite headings; NULL if
  no member has a heading, i.e. a lone candidate without heading).
- ``lat`` / ``lon``: centroid of the members weighted by verdict confidence (plain mean if all
  weights are 0), rounded to 9 decimals; exactly the member's position for a lone candidate.
- ``photo_path`` / ``crop_path``: the BLURRED ``blurred_frame_path`` / ``blurred_crop_path`` of the member
  with the highest candidate ``score`` (ties: higher verdict confidence, then drive id, track key).
- ``first_seen`` / ``last_seen``: earliest and latest member time as UTC ISO text. A member's time is
  the drive's ``started_at`` plus the candidate's ``best_timestamp`` seconds; if the drive record or
  its ``started_at`` is unusable, the verdict's ``created_at``, else the time of the run.
- ``status``: ``new`` for a new hazard; one of ``new, approved, rejected, reported, possibly_fixed``
  (a CHECK constraint enforces it; ``set_hazard_status`` validates). ``created_at`` is set once,
  ``updated_at`` changes only when something else about the hazard changes.
- Every member candidate gets ``candidates.hazard_id`` = the hazard id; all other candidates get NULL.

Re-running (idempotent). Every run recomputes the clusters, then matches each cluster to an existing
hazard through the members' ``hazard_id`` (only an existing hazard of the same class counts; a dangling
id is ignored): hazards with a user-set status (anything but ``new``) are matched first, then by the
number of shared members, then by cluster order and id. A matched hazard keeps its id, status and
``created_at`` and only its derived values are rewritten (when they changed); an unmatched cluster
becomes a new hazard; an existing hazard that matches no cluster (no members left: its candidates
were deleted or replaced by ``aggregate_drive``, their verdicts were rejected, ...) is deleted, even
if it has a user-set status (unless it is adopted, see "adoption" below). If two hazards merge, the one with a user-set status survives (see the
order above); if one splits, the part with more of its members keeps the id. Candidates replaced by
a re-run of ``aggregate_drive`` come back with NULL ``hazard_id`` and are re-linked here; a hazard
whose only members were the replaced candidates is therefore deleted and recreated with a new id
(unless it has user decisions and is adopted, see "adoption" below).
Running twice in a row changes nothing (no new rows, no double counting, no ``updated_at`` change).
A drive with no participating candidates just makes sure the table matches the remaining members
(hazards of other drives with members are untouched; hazards with no members are removed).

Review edits are sticky (task 17). ``review_hazard`` (the PATCH route) can set a hazard's class and
severity by hand; the flags ``class_edited`` / ``severity_edited`` then stay 1 and every later run keeps the
user's value while still rewriting everything else (sightings, confidence, position, direction, photo, first and
last seen, candidate_count). Clustering is unaffected by an edit: it always uses the members' own verdict
classes, and a hazard is matched to its cluster through its AUTOMATIC class (``original_hazard_class`` while
the class is edited), so a re-run never duplicates an edited hazard and never crashes. The automatic values
are kept current in ``original_hazard_class`` / ``original_severity`` on every run. Deduplication NEVER clears
an edit flag, not even when the automatic value momentarily equals the edited one (re-processing one drive
leaves the hazard supported by the other drives alone for a while); only ``review_hazard`` clears a flag, when
the user sets the value back to the automatic one. If the cluster's automatic class changes (a member was
verified again as another class), the old hazard no longer matches its cluster by class.

The agency chosen on the report screen (task 18, ``hazards.agency_override``: ``VDOT``, ``MDOT SHA`` or ``none``;
NULL = automatic) is a user decision too: deduplication never writes it, so it survives every re-run, and a hazard
that has one counts as carrying review work for the adoption rules below. ``review_hazard`` also sets
``reported`` (only from an approved or reported hazard, else ``ReviewConflict``).

Review work survives re-processing (adoption). A hazard that carries user decisions (a status other than
``new``, an edited class or severity, or a chosen agency) and has lost all its members (its drive was re-processed, so its
candidates were deleted and are being recreated) is not deleted straight away:

- ``refresh_hazards_quietly`` (run when candidates are deleted) leaves such hazards in place ("parked");
  hazards without decisions are removed as before.
- Every deduplication run then tries to ADOPT each memberless decided hazard (an "orphan") into a cluster
  that no existing hazard claimed: same automatic class (``original_hazard_class`` while edited), the
  cluster's position within ``distance_m`` of the orphan's stored position and the cluster's direction within
  ``heading_tolerance_deg`` of the orphan's stored direction (both directions must be known). Pairs are taken
  nearest first (ties: lower hazard id, then cluster order); each orphan adopts at most one cluster and each
  cluster is adopted by at most one orphan. The adopting cluster keeps the orphan's id, status,
  ``created_at``, ``reviewed_at`` and edits; everything else is recomputed from its members.
- An orphan with decisions that no cluster adopts is deleted by that run (a hazard re-detected more than
  ``distance_m`` away, in another direction or as another class is treated as a different hazard), and so is
  every orphan without decisions. Until the next run a parked hazard stays in the table, so a re-run that
  fails before deduplication leaves it without candidates.

Everything happens in ONE transaction. ``clear_results`` (pipeline) and ``aggregate_drive`` call
``rebuild_hazards`` after deleting candidates, so ``hazards`` does not keep rows for deleted candidates.

Bad arguments (blank ``drive_id``, ``distance_m`` not a number >= 0, ``heading_tolerance_deg`` not
in 0..180, including bool/NaN, or such a value in the environment) raise ``ValueError`` before any work.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import config
from .db import AGENCY_OVERRIDES, HAZARD_STATUSES, get_connection, init_db
from .hazard_prompts import PromptConfigError, load_hazard_prompts

logger = logging.getLogger(__name__)

EARTH_RADIUS_M = 6_371_008.8
# One degree of latitude is at least this many metres anywhere on Earth (used to prune the search).
_MIN_M_PER_DEG_LAT = 110_000.0
_EPS = 1e-9

_DERIVED = ("hazard_class", "lat", "lon", "direction", "severity", "confidence", "sightings",
            "candidate_count", "first_seen", "last_seen", "photo_path", "crop_path")
# Review bookkeeping rewritten together with the derived values (see _apply_edits).
_REVIEW_KEPT = ("class_edited", "severity_edited", "original_hazard_class", "original_severity")
_WRITTEN = _DERIVED + _REVIEW_KEPT
REVIEW_STATUSES = ("new", "approved", "rejected", "reported")  # what review_hazard / PATCH may set
REPORTABLE_STATUSES = ("approved", "reported")  # "reported" can only be set from one of these (task 18)
SEVERITIES = (1, 2, 3)


class DedupError(RuntimeError):
    """Deduplication failed; the transaction was rolled back."""


class ReviewConflict(ValueError):
    """The review is well-formed but the hazard's current state does not allow it (the route answers 409)."""


@dataclass
class DedupSummary:
    """What a ``deduplicate_drive`` / ``rebuild_hazards`` call did.

    ``candidate_count`` / ``skipped_count`` / ``hazard_count`` / ``hazards`` are about the named drive
    (0 / 0 / 0 / [] for ``rebuild_hazards``); ``created_count`` / ``updated_count`` / ``removed_count``
    / ``total_hazards`` are about the whole table.
    """

    drive_id: str | None
    distance_m: float
    heading_tolerance_deg: float
    candidate_count: int = 0  # the drive's accepted, geotagged candidates that took part
    skipped_count: int = 0  # the drive's accepted candidates left out (no usable geotag, or the blur failed)
    hazard_count: int = 0  # distinct hazards those candidates belong to
    created_count: int = 0
    updated_count: int = 0
    removed_count: int = 0
    total_hazards: int = 0
    hazards: list[dict[str, Any]] = field(default_factory=list)  # the drive's hazards, as load_hazards returns


# ----- geometry ----------------------------------------------------------------------------


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in metres between two points given in degrees."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def heading_diff_deg(a: float, b: float) -> float:
    """Smallest angle in degrees (0 to 180) between two headings, wrap-around aware."""
    d = abs(a - b) % 360.0
    return min(d, 360.0 - d)


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ----- arguments ---------------------------------------------------------------------------


def _number_in(value: Any, name: str, low: float, high: float) -> float:
    if not _finite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a number from {low:g} to {high:g}, got {value!r}")
    return float(value)


def _resolve_params(distance_m: Any, heading_tolerance_deg: Any) -> tuple[float, float]:
    dist = config.dedup_distance_m() if distance_m is None else _number_in(
        distance_m, "distance_m", 0.0, 1e7)
    head = config.dedup_heading_tolerance_deg() if heading_tolerance_deg is None else _number_in(
        heading_tolerance_deg, "heading_tolerance_deg", 0.0, 180.0)
    return dist, head


# ----- loading -----------------------------------------------------------------------------

_ELIGIBLE_SQL = (
    "SELECT c.id, c.drive_id, c.track_key, c.best_timestamp, c.blurred_frame_path, c.blurred_crop_path, "
    "c.lat, c.lon, c.heading, c.score, v.hazard_class AS cls, v.severity, v.confidence, "
    "v.created_at AS verdict_created, d.started_at "
    "FROM candidates c JOIN verdicts v ON v.candidate_id = c.id "
    "LEFT JOIN drives d ON d.id = c.drive_id "
    "WHERE v.accepted = 1 AND v.is_hazard = 1 AND c.geotag_status = 'ok' "
    "AND c.lat IS NOT NULL AND c.lon IS NOT NULL "
    "AND (c.blur_status IS NULL OR c.blur_status != 'failed')"
)


def _parse_time(text: Any) -> datetime | None:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        parsed = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _seen_time(row: Any, now: datetime) -> datetime:
    start = _parse_time(row["started_at"])
    if start is not None and _finite(row["best_timestamp"]):
        try:
            return start + timedelta(seconds=float(row["best_timestamp"]))
        except OverflowError:
            pass
    return _parse_time(row["verdict_created"]) or now


def _load_members(conn: Any, now: datetime) -> list[dict[str, Any]]:
    members = []
    for row in conn.execute(_ELIGIBLE_SQL):
        lat, lon = row["lat"], row["lon"]
        if not (_finite(lat) and _finite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
            continue
        heading = float(row["heading"]) % 360.0 if _finite(row["heading"]) else None
        confidence = float(row["confidence"]) if _finite(row["confidence"]) else 0.0
        members.append({
            "id": row["id"],
            "drive_id": row["drive_id"],
            "track_key": row["track_key"],
            "best_timestamp": row["best_timestamp"],
            "photo_path": row["blurred_frame_path"],  # only blurred images ever reach a hazard
            "crop_path": row["blurred_crop_path"],
            "lat": float(lat),
            "lon": float(lon),
            "heading": heading,
            "score": float(row["score"]) if _finite(row["score"]) else 0.0,
            "cls": row["cls"],
            "severity": int(row["severity"]),
            "confidence": confidence,
            "seen": _seen_time(row, now),
        })
    # Canonical order: everything below is independent of how the rows came out of the database.
    members.sort(key=_member_key)
    return members


def _member_key(m: dict[str, Any]) -> tuple:
    return (m["cls"], m["lat"], m["lon"], m["drive_id"], m["track_key"], m["best_timestamp"], m["id"])


# ----- clustering --------------------------------------------------------------------------


def _clusters(members: list[dict[str, Any]], distance_m: float, heading_tolerance: float) -> list[list[dict[str, Any]]]:
    """Connected components of the link relation, members and clusters in canonical order."""
    parent = list(range(len(members)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_class: dict[str, list[int]] = {}
    for i, m in enumerate(members):
        if m["heading"] is not None:  # an unknown heading never links
            by_class.setdefault(m["cls"], []).append(i)
    for indexes in by_class.values():
        indexes.sort(key=lambda i: (members[i]["lat"], i))
        for pos, i in enumerate(indexes):
            a = members[i]
            for j in indexes[pos + 1:]:
                b = members[j]
                if (b["lat"] - a["lat"]) * _MIN_M_PER_DEG_LAT > distance_m + 1.0:
                    break  # sorted by latitude: nothing further can be close enough
                if heading_diff_deg(a["heading"], b["heading"]) > heading_tolerance + _EPS:
                    continue
                if haversine_m(a["lat"], a["lon"], b["lat"], b["lon"]) > distance_m + _EPS:
                    continue
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    groups: dict[int, list[dict[str, Any]]] = {}
    for i, m in enumerate(members):  # members are in canonical order, so each group is too
        groups.setdefault(find(i), []).append(m)
    return sorted(groups.values(), key=lambda g: _member_key(g[0]))


def _summarise(cluster: list[dict[str, Any]]) -> dict[str, Any]:
    """The derived hazard values of one cluster (members in canonical order)."""
    best = min(cluster, key=lambda m: (-m["score"], -m["confidence"], m["drive_id"], m["track_key"],
                                       m["best_timestamp"], m["id"]))
    per_drive: dict[str, float] = {}
    for m in cluster:
        per_drive[m["drive_id"]] = max(per_drive.get(m["drive_id"], 0.0), m["confidence"])
    top = max(per_drive.values())
    if len(per_drive) == 1:
        confidence = top
    else:
        miss = 1.0
        for c in sorted(per_drive.values()):
            miss *= 1.0 - min(1.0, max(0.0, c))
        confidence = min(1.0, max(top, 1.0 - miss))

    if len(cluster) == 1:
        lat, lon = cluster[0]["lat"], cluster[0]["lon"]
    else:
        total = math.fsum(m["confidence"] for m in cluster)
        weights = [m["confidence"] for m in cluster] if total > 0 else [1.0] * len(cluster)
        total = math.fsum(weights)
        lat = round(math.fsum(w * m["lat"] for w, m in zip(weights, cluster)) / total, 9)
        lon = round(math.fsum(w * m["lon"] for w, m in zip(weights, cluster)) / total, 9)

    headings = [m["heading"] for m in cluster if m["heading"] is not None]
    direction: float | None = None
    if headings:
        sin = math.fsum(math.sin(math.radians(h)) for h in headings)
        cos = math.fsum(math.cos(math.radians(h)) for h in headings)
        if math.hypot(sin, cos) < 1e-9 * len(headings):
            direction = best["heading"]  # opposite headings cancel: fall back to the best member
        else:
            direction = round(math.degrees(math.atan2(sin, cos)) % 360.0, 6) % 360.0
        if direction is None:
            direction = headings[0]

    seen = sorted(m["seen"] for m in cluster)
    return {
        "hazard_class": cluster[0]["cls"],
        "lat": lat,
        "lon": lon,
        "direction": direction,
        "severity": max(m["severity"] for m in cluster),
        "confidence": confidence,
        "sightings": len(per_drive),
        "candidate_count": len(cluster),
        "first_seen": seen[0].isoformat(),
        "last_seen": seen[-1].isoformat(),
        "photo_path": best["photo_path"],
        "crop_path": best["crop_path"],
    }


# ----- matching to existing hazards ---------------------------------------------------------


def _auto_class(hazard: dict[str, Any]) -> str:
    """The class deduplication gave the hazard: its stored class unless the user edited it."""
    if hazard.get("class_edited") and hazard.get("original_hazard_class"):
        return hazard["original_hazard_class"]
    return hazard["hazard_class"]


def _has_decision(hazard: dict[str, Any]) -> bool:
    """Does the hazard carry review work (a status other than new, an edited class or severity, or an agency
    chosen by the user)?"""
    return (hazard["status"] != "new" or bool(hazard.get("class_edited")) or bool(hazard.get("severity_edited"))
            or hazard.get("agency_override") is not None)


def _apply_edits(values: dict[str, Any], existing: dict[str, Any] | None) -> dict[str, Any]:
    """Add the review bookkeeping to a cluster's derived values: an edited class or severity of the
    matched hazard wins over the automatic one, which is recorded in ``original_*``. A flag is never
    cleared here (only ``review_hazard`` does that); a new hazard has no edits."""
    out = dict(values)
    for flag, field, orig in (("class_edited", "hazard_class", "original_hazard_class"),
                              ("severity_edited", "severity", "original_severity")):
        if existing is not None and existing.get(flag):
            out[flag], out[orig] = 1, values[field]
            out[field] = existing[field]
        else:
            out[flag], out[orig] = 0, None
    return out


def _adopt(clusters: list[list[dict[str, Any]]], summaries: list[dict[str, Any]], matched: dict[int, int],
           existing: dict[int, dict[str, Any]], distance_m: float, heading_tolerance: float) -> dict[int, int]:
    """cluster index -> id of a memberless hazard with review work that takes the cluster over."""
    claimed = set(matched.values())
    orphans = [h for hid, h in sorted(existing.items()) if hid not in claimed and _has_decision(h)]
    free = [ci for ci in range(len(clusters)) if ci not in matched]
    pairs = []
    for h in orphans:
        if not (_finite(h["lat"]) and _finite(h["lon"]) and _finite(h["direction"])):
            continue
        auto = _auto_class(h)
        for ci in free:
            sm = summaries[ci]
            if sm["hazard_class"] != auto or sm["direction"] is None:
                continue
            if heading_diff_deg(h["direction"], sm["direction"]) > heading_tolerance + _EPS:
                continue
            d = haversine_m(h["lat"], h["lon"], sm["lat"], sm["lon"])
            if d <= distance_m + _EPS:
                pairs.append((d, h["id"], ci))
    pairs.sort()
    taken: set[int] = set()
    adopted: dict[int, int] = {}
    for _d, hid, ci in pairs:
        if hid in taken or ci in adopted:
            continue
        taken.add(hid)
        adopted[ci] = hid
    return adopted


def _match(clusters: list[list[dict[str, Any]]], links: dict[int, int | None],
           existing: dict[int, dict[str, Any]]) -> dict[int, int]:
    """cluster index -> existing hazard id it keeps (see the module docstring for the order)."""
    pairs: dict[tuple[int, int], int] = {}
    for ci, cluster in enumerate(clusters):
        for m in cluster:
            hid = links.get(m["id"])
            if hid is not None and hid in existing and _auto_class(existing[hid]) == m["cls"]:
                pairs[(ci, hid)] = pairs.get((ci, hid), 0) + 1
    order = sorted(pairs.items(), key=lambda kv: (
        0 if existing[kv[0][1]]["status"] != "new" else 1, -kv[1], kv[0][0], kv[0][1]))
    taken_clusters: dict[int, int] = {}
    taken_hazards: set[int] = set()
    for (ci, hid), _count in order:
        if ci in taken_clusters or hid in taken_hazards:
            continue
        taken_clusters[ci] = hid
        taken_hazards.add(hid)
    return taken_clusters


# ----- main --------------------------------------------------------------------------------


def deduplicate_drive(
    drive_id: str,
    *,
    db_path: str | Path | None = None,
    distance_m: float | None = None,
    heading_tolerance_deg: float | None = None,
) -> DedupSummary:
    """Merge the drive's accepted, geotagged candidates with the others into ``hazards``, then return
    a summary about the drive. See the module docstring."""
    if drive_id is None or not str(drive_id).strip():
        raise ValueError("drive_id must not be blank")
    drive_id = str(drive_id).strip()
    return _run(drive_id, db_path, distance_m, heading_tolerance_deg)


def rebuild_hazards(
    *,
    db_path: str | Path | None = None,
    distance_m: float | None = None,
    heading_tolerance_deg: float | None = None,
    keep_decided: bool = False,
) -> DedupSummary:
    """Recompute every hazard from the participating candidates of all drives (no drive named).
    ``keep_decided`` leaves memberless hazards with review work in place (see the module docstring)."""
    return _run(None, db_path, distance_m, heading_tolerance_deg, keep_decided)


def _run(drive_id: str | None, db_path: Any, distance_m: Any, heading_tolerance_deg: Any,
         keep_decided: bool = False) -> DedupSummary:
    dist, head = _resolve_params(distance_m, heading_tolerance_deg)
    target = init_db(db_path)
    now = datetime.now(timezone.utc)
    now_text = now.isoformat()
    summary = DedupSummary(drive_id=drive_id, distance_m=dist, heading_tolerance_deg=head)

    conn = get_connection(target)
    conn.isolation_level = None  # explicit transaction below: read, decide and write atomically
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            members = _load_members(conn, now)
            clusters = _clusters(members, dist, head)
            links = {r["id"]: r["hazard_id"] for r in conn.execute("SELECT id, hazard_id FROM candidates")}
            existing = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM hazards")}
            summaries = [_summarise(c) for c in clusters]
            matched = _match(clusters, links, existing)
            matched.update(_adopt(clusters, summaries, matched, existing, dist, head))

            new_links: dict[int, int | None] = {cid: None for cid in links}
            for ci, cluster in enumerate(clusters):
                hid = matched.get(ci)
                values = _apply_edits(summaries[ci], existing[hid] if hid is not None else None)
                if hid is None:
                    cur = conn.execute(
                        "INSERT INTO hazards (" + ", ".join(_WRITTEN) + ", status, created_at, updated_at) "
                        "VALUES (" + ", ".join("?" for _ in _WRITTEN) + ", 'new', ?, ?)",
                        (*(values[k] for k in _WRITTEN), now_text, now_text),
                    )
                    hid = int(cur.lastrowid)
                    summary.created_count += 1
                elif any(existing[hid][k] != values[k] for k in _WRITTEN):
                    conn.execute(
                        "UPDATE hazards SET " + ", ".join(f"{k} = ?" for k in _WRITTEN) + ", updated_at = ? "
                        "WHERE id = ?",
                        (*(values[k] for k in _WRITTEN), now_text, hid),
                    )
                    summary.updated_count += 1
                for m in cluster:
                    new_links[m["id"]] = hid

            kept = set(matched.values())
            stale = [hid for hid in existing
                     if hid not in kept and not (keep_decided and _has_decision(existing[hid]))]
            for hid in stale:
                conn.execute("DELETE FROM hazards WHERE id = ?", (hid,))
            summary.removed_count = len(stale)

            changed = [(new, cid) for cid, new in new_links.items() if links.get(cid) != new]
            if changed:
                conn.executemany("UPDATE candidates SET hazard_id = ? WHERE id = ?", changed)

            if drive_id is not None:
                mine = [m for m in members if m["drive_id"] == drive_id]
                summary.candidate_count = len(mine)
                summary.hazard_count = len({new_links[m["id"]] for m in mine})
                summary.skipped_count = conn.execute(
                    "SELECT COUNT(*) FROM candidates c JOIN verdicts v ON v.candidate_id = c.id "
                    "WHERE c.drive_id = ? AND v.accepted = 1 AND v.is_hazard = 1", (drive_id,)
                ).fetchone()[0] - summary.candidate_count
                ids = sorted({new_links[m["id"]] for m in mine})
            else:
                ids = []
            summary.total_hazards = conn.execute("SELECT COUNT(*) FROM hazards").fetchone()[0]
            if ids:
                marks = ", ".join("?" for _ in ids)
                summary.hazards = [
                    dict(r) for r in conn.execute(f"SELECT * FROM hazards WHERE id IN ({marks}) ORDER BY id", ids)
                ]
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    except (ValueError, KeyboardInterrupt, SystemExit):
        raise
    except Exception as exc:
        raise DedupError(f"deduplication failed; nothing was changed: {type(exc).__name__}: {exc}") from exc
    finally:
        conn.close()
    logger.info("deduplication%s: %d candidates, %d created, %d updated, %d removed, %d hazards in total",
                f" for drive {drive_id}" if drive_id else "", summary.candidate_count,
                summary.created_count, summary.updated_count, summary.removed_count, summary.total_hazards)
    return summary


def refresh_hazards_quietly(db_path: str | Path | None = None) -> None:
    """``rebuild_hazards`` for callers that just deleted candidates (``clear_results``,
    ``aggregate_drive``): hazards of the deleted candidates are recomputed from the remaining members or
    removed, except hazards with review work, which stay until the next deduplication run adopts or deletes
    them. Never raises; a failure is logged and the next ``deduplicate_drive`` repairs the table."""
    try:
        rebuild_hazards(db_path=db_path, keep_decided=True)
    except Exception:
        logger.warning("could not refresh hazards after deleting candidates", exc_info=True)


# ----- reading and review ------------------------------------------------------------------


def load_hazards(
    *, db_path: str | Path | None = None, status: str | None = None, hazard_class: str | None = None
) -> list[dict[str, Any]]:
    """The stored hazards as dicts (column names as keys), ordered by id. ``status`` and
    ``hazard_class`` filter when given; an unknown status raises ``ValueError``."""
    if status is not None:
        _check_status(status)
    conn = get_connection(init_db(db_path))
    try:
        sql, args = "SELECT * FROM hazards", []
        where = []
        if status is not None:
            where.append("status = ?")
            args.append(status)
        if hazard_class is not None:
            where.append("hazard_class = ?")
            args.append(hazard_class)
        if where:
            sql += " WHERE " + " AND ".join(where)
        return [dict(r) for r in conn.execute(sql + " ORDER BY id", args)]
    finally:
        conn.close()


def get_hazard(hazard_id: int, *, db_path: str | Path | None = None) -> dict[str, Any] | None:
    """One hazard as a dict, or None if there is none with that id."""
    conn = get_connection(init_db(db_path))
    try:
        row = conn.execute("SELECT * FROM hazards WHERE id = ?", (hazard_id,)).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def _check_status(status: Any) -> str:
    if not isinstance(status, str) or status not in HAZARD_STATUSES:
        raise ValueError(f"status must be one of {', '.join(HAZARD_STATUSES)}; got {status!r}")
    return status


def set_hazard_status(hazard_id: int, status: str, *, db_path: str | Path | None = None) -> dict[str, Any]:
    """Set a hazard's status (new, approved, rejected, reported, possibly_fixed) and return the row.
    An invalid status raises ``ValueError``, an unknown id ``LookupError``. Deduplication keeps it.
    ``reviewed_at`` is set for approved / rejected and cleared for new (left alone for the other two)."""
    _check_status(status)
    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection(init_db(db_path))
    try:
        with conn:
            if status in ("approved", "rejected"):
                sql, args = "UPDATE hazards SET status = ?, updated_at = ?, reviewed_at = ? WHERE id = ?", (status, now, now, hazard_id)
            elif status == "new":
                sql, args = "UPDATE hazards SET status = ?, updated_at = ?, reviewed_at = NULL WHERE id = ?", (status, now, hazard_id)
            else:
                sql, args = "UPDATE hazards SET status = ?, updated_at = ? WHERE id = ?", (status, now, hazard_id)
            cur = conn.execute(sql, args)
            if cur.rowcount != 1:
                raise LookupError(f"hazard not found: {hazard_id}")
        return dict(conn.execute("SELECT * FROM hazards WHERE id = ?", (hazard_id,)).fetchone())
    finally:
        conn.close()


_UNSET: Any = object()


def review_hazard(
    hazard_id: int,
    *,
    status: Any = _UNSET,
    hazard_class: Any = _UNSET,
    severity: Any = _UNSET,
    agency: Any = _UNSET,
    db_path: str | Path | None = None,
) -> dict[str, Any]:
    """Apply a human review to a hazard in ONE transaction and return the updated row.

    At least one of ``status`` (``new``, ``approved``, ``rejected`` or ``reported``; ``possibly_fixed`` is
    set by a later task and refused here), ``hazard_class`` (one of the classes in the hazard prompts config),
    ``severity`` (an int 1 to 3; bool and float are refused) and ``agency`` (``"VDOT"``, ``"MDOT SHA"``,
    ``"none"`` = no agency, or ``None`` / ``"auto"`` = back to the automatic agency) must be given. Anything else
    raises ``ValueError`` and changes nothing; an unknown id raises ``LookupError``. ``reported`` means "the user
    says they submitted the report themselves" and is only allowed while the hazard is approved or already
    reported, else ``ReviewConflict`` (a ``ValueError``); it leaves ``reviewed_at`` alone.

    Changing the status of a ``possibly_fixed`` hazard (task 19) also records ``fix_check_dismissed_at`` = now and
    clears ``previous_status``, ``possibly_fixed_at`` and ``fix_misses`` (see ``backend/possibly_fixed.py``).

    Bookkeeping: ``updated_at`` always changes; ``reviewed_at`` is set when the status becomes approved or
    rejected and cleared when it becomes new. A class or severity that differs from the automatic one sets
    ``class_edited`` / ``severity_edited`` to 1 and records the automatic value in ``original_hazard_class``
    / ``original_severity``; setting it back to the automatic value clears the flag and the original.
    Deduplication keeps edited values (see the module docstring)."""
    if isinstance(hazard_id, bool) or not isinstance(hazard_id, int):
        raise ValueError(f"hazard id must be an integer, got {hazard_id!r}")
    if status is _UNSET and hazard_class is _UNSET and severity is _UNSET and agency is _UNSET:
        raise ValueError("nothing to change: give status, hazard_class, severity and/or agency")
    if status is not _UNSET and (not isinstance(status, str) or status not in REVIEW_STATUSES):
        raise ValueError(
            f"status must be one of {', '.join(REVIEW_STATUSES)} (possibly_fixed cannot be set "
            f"by a review); got {status!r}")
    if agency is not _UNSET and agency is not None and (not isinstance(agency, str) or (
            agency != "auto" and agency not in AGENCY_OVERRIDES)):
        raise ValueError(f"agency must be one of {', '.join(AGENCY_OVERRIDES)}, auto or null; got {agency!r}")
    if severity is not _UNSET and (
        isinstance(severity, bool) or not isinstance(severity, int) or severity not in SEVERITIES
    ):
        raise ValueError(f"severity must be an integer from 1 to 3; got {severity!r}")
    if hazard_class is not _UNSET:
        try:
            known = load_hazard_prompts().class_names
        except PromptConfigError as exc:
            raise DedupError(f"cannot check the hazard class: {exc}") from exc
        if not isinstance(hazard_class, str) or hazard_class not in known:
            raise ValueError(f"hazard_class must be one of {', '.join(known)}; got {hazard_class!r}")

    now = datetime.now(timezone.utc).isoformat()
    conn = get_connection(init_db(db_path))
    conn.isolation_level = None  # explicit transaction: read the row, decide and write atomically
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            row = conn.execute("SELECT * FROM hazards WHERE id = ?", (hazard_id,)).fetchone()
            if row is None:
                raise LookupError(f"hazard not found: {hazard_id}")
            sets: dict[str, Any] = {"updated_at": now}
            if status is not _UNSET:
                if status == "reported" and row["status"] not in REPORTABLE_STATUSES:
                    raise ReviewConflict(
                        f"only an approved hazard can be marked as reported; this one is {row['status']}")
                sets["status"] = status
                if row["status"] == "possibly_fixed":
                    # The user answered the possibly-fixed flag (task 19): passes before now no longer count.
                    sets["fix_check_dismissed_at"] = now
                    sets["previous_status"], sets["possibly_fixed_at"], sets["fix_misses"] = None, None, 0
                if status != "reported":  # reporting an approved hazard keeps the time it was approved
                    sets["reviewed_at"] = now if status in ("approved", "rejected") else None
            if agency is not _UNSET:
                sets["agency_override"] = None if agency in (None, "auto") else agency
            for value, field, flag, orig in (
                (hazard_class, "hazard_class", "class_edited", "original_hazard_class"),
                (severity, "severity", "severity_edited", "original_severity"),
            ):
                if value is _UNSET:
                    continue
                auto = row[orig] if row[flag] and row[orig] is not None else row[field]
                sets[field] = value
                sets[flag], sets[orig] = (0, None) if value == auto else (1, auto)
            conn.execute(
                "UPDATE hazards SET " + ", ".join(f"{k} = ?" for k in sets) + " WHERE id = ?",
                (*sets.values(), hazard_id),
            )
            result = dict(conn.execute("SELECT * FROM hazards WHERE id = ?", (hazard_id,)).fetchone())
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
    return result
