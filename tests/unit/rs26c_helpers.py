"""Helpers for the task 26c hidden tests (resumable pipeline).  Built on the rs10 pipeline helpers."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from datetime import timedelta
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, Path(r"C:\Users\dhair\RoadScout\tests\unit"), _HERE.parent / "task-10",
           _HERE.parent / "task-09", _HERE.parent / "task-07"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.append(str(_p))

from rs10_helpers import M_PER_DEG, T0  # noqa: E402

CHUNK_FILES = ["chunk_0000.mp4", "chunk_0001.mp4", "chunk_0002.mp4"]


def run(e, did, provider, **kw):
    """Call run_pipeline directly. Returns (result, exception_or_None)."""
    from backend.pipeline import run_pipeline

    try:
        return run_pipeline(did, provider=provider, db_path=str(e.db), **kw), None
    except Exception as ex:  # noqa: BLE001
        return None, ex


def chunk_names(p):
    """Chunk file names the spy provider was asked to detect, in call order."""
    return [Path(str(x)).name for x in p.detect_calls]


def work_dir(e, did=None):
    found = sorted({p.parent for p in Path(e.storage).rglob("frame_timestamps.json")})
    assert len(found) == 1, "expected exactly one work folder, got %s" % found
    return found[0]


def work_chunks(e):
    return sorted(p.name for p in work_dir(e).glob("chunk_*.mp4"))


def fail_detect_at(n):
    from rs10_helpers import spy

    return spy(fail_detect_on=n)


def fail_verify_at(n):
    from rs10_helpers import spy

    return spy(fail_verify_on=n)


def post_path(e, restart=None):
    path = re.sub(r"\{[^}/]+\}", "DID", e.process_route())
    return path, ({} if restart is None else {"restart": restart})


def http_process(e, did, restart=None):
    path, params = post_path(e, restart)
    return e.client.post(path.replace("DID", did), params=params)


# ------------------------------------------------------------------ snapshots

DROP = {"id", "drive_id", "created_at", "updated_at", "hazard_id", "candidate_id"}


def _norm(row):
    out = {}
    for k, v in row.items():
        if k in DROP:
            continue
        if k.endswith("_path") and v:
            v = Path(str(v)).name
        out[k] = v
    return out


def _dump(rows):
    return sorted(json.dumps(_norm(r), sort_keys=True, default=str) for r in rows)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def snapshot(e, did):
    """Id-free, time-free picture of everything a run produced, to compare resumed with uninterrupted."""
    verdicts = e.q("SELECT v.*, c.track_key AS _tk, c.hazard_class AS _hc FROM verdicts v "
                   "JOIN candidates c ON c.id = v.candidate_id WHERE v.drive_id = ?", (did,))
    blurs = {p.name: sha(p) for p in Path(e.storage).rglob("*_blur.jpg")}
    return {
        "detections": _dump(e.q("SELECT * FROM detections WHERE drive_id = ?", (did,))),
        "candidates": _dump(e.candidates(did)),
        "verdicts": _dump(verdicts),
        "hazards": _dump(e.q("SELECT * FROM hazards")),
        "blurred": blurs,
    }


def assert_same_snapshot(a, b):
    for key in ("detections", "candidates", "verdicts", "hazards", "blurred"):
        assert a[key] == b[key], "%s differ between the resumed and the uninterrupted run:\n%s\n----\n%s" % (
            key, a[key], b[key])


def stationary_counts(d):
    return {k: d.get(k) for k in ("frames_total", "stationary_frames_skipped", "stationary_chunks_skipped",
                                   "chunks_sent", "stationary_note")}


def gpx_with_a_stop(still_seconds=5, seconds=60, start=T0):
    """Synthetic GPX (made-up coordinates): parked for the first ``still_seconds``, then ~10 m/s north."""
    pts, lat = [], 38.88
    for s in range(seconds + 1):
        t = (start + timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%SZ")
        pts.append('<trkpt lat="%.7f" lon="-77.3000000"><ele>100</ele><time>%s</time></trkpt>' % (lat, t))
        if s >= still_seconds:
            lat += 10.0 / M_PER_DEG
    body = ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<gpx version="1.1" creator="roadscout-tests" xmlns="http://www.topografix.com/GPX/1/1">'
            "<trk><name>synthetic</name><trkseg>" + "".join(pts) + "</trkseg></trk></gpx>\n")
    return body.encode("utf-8")


def duplicates(rows, keys):
    seen, dup = set(), []
    for r in rows:
        k = tuple(r[x] for x in keys)
        if k in seen:
            dup.append(k)
        seen.add(k)
    return dup
