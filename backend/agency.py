"""Which road agency takes a report for a point (task 18). No network, no keys, no model.

    from backend.agency import agency_for_point
    agency_for_point(38.84, -77.43)    # 'VDOT'      (Centreville, Virginia)
    agency_for_point(39.08, -77.15)    # 'MDOT SHA'  (Rockville, Maryland)
    agency_for_point(38.90, -77.03)    # None        (the District of Columbia)

``agency_for_point(lat, lon)`` returns ``'VDOT'`` for a point in Virginia, ``'MDOT SHA'`` for a point in
Maryland and ``None`` everywhere else (DC, other states, open water far from both). It never raises: a
latitude or longitude that is None, NaN, infinite, a bool, not a number (including numeric text) or out of range
(latitude outside -90..90, longitude outside -180..180) gives ``None``. ``jurisdiction_for_point`` gives the
polygon id (``'VA'``, ``'MD'``, ``'DC'``) behind the answer.

Data: the outlines are in ``config/jurisdictions.json`` (rebuilt by ``scripts/build_jurisdictions.py`` from the US
Census Bureau TIGER/Line 2023 state boundaries, public domain). Points are ``[longitude, latitude]``; the three
outlines are simplified with a 50 m tolerance. A point is inside a jurisdiction when a ray cast from it crosses an
odd number of edges (even-odd rule, plain ray casting on longitude/latitude, no projection needed at this size).
DC is listed first so that it wins over its neighbours, which makes DC "no agency".

Accuracy, honestly: the legal boundaries follow rivers, the shore of the Potomac and survey lines, and the
simplification moves them by up to about 50 m on top of the source data, so expect an error of roughly 100 m at
the Potomac, the DC line and the Virginia/Maryland/West Virginia/Pennsylvania lines, and treat a point within a
few hundred metres of a border as uncertain. Water belongs to whoever the Census file gives it to (the Potomac
is Maryland's up to the Virginia shore; the Chesapeake Bay is split between Maryland and Virginia). Points exactly
on an edge may fall either way. This is a default only: the report screen lets the user change the agency, because
the app also cannot tell who maintains a given street (MDOT SHA covers state roads only; counties and cities look
after their own).
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from numbers import Real
from typing import Any

from .config import ROOT

JURISDICTIONS_PATH = ROOT / "config" / "jurisdictions.json"

VDOT = "VDOT"
MDOT_SHA = "MDOT SHA"
AGENCIES = (VDOT, MDOT_SHA)


@lru_cache(maxsize=1)
def _load() -> tuple[tuple[str, str | None, tuple[float, float, float, float], tuple], ...]:
    """(id, agency, bounding box, rings) per jurisdiction, in evaluation order."""
    data = json.loads(JURISDICTIONS_PATH.read_text(encoding="utf-8"))
    out = []
    for item in data["jurisdictions"]:
        rings = tuple(tuple((float(x), float(y)) for x, y in ring) for ring in item["rings"])
        xs = [x for ring in rings for x, _ in ring]
        ys = [y for ring in rings for _, y in ring]
        out.append((item["id"], item["agency"], (min(xs), min(ys), max(xs), max(ys)), rings))
    return tuple(out)


def _inside(x: float, y: float, rings: tuple) -> bool:
    """Even-odd ray casting: a ray towards +longitude crosses an odd number of edges for an inner point."""
    crossings = 0
    for ring in rings:
        j = len(ring) - 1
        for i in range(len(ring)):
            xi, yi = ring[i]
            xj, yj = ring[j]
            if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                crossings += 1
            j = i
    return crossings % 2 == 1


def _valid(lat: Any, lon: Any) -> bool:
    for value, limit in ((lat, 90.0), (lon, 180.0)):
        if isinstance(value, bool) or not isinstance(value, Real):
            return False
        try:
            if not math.isfinite(value) or abs(value) > limit:
                return False
        except (TypeError, ValueError, OverflowError):
            return False
    return True


def jurisdiction_for_point(lat: Any, lon: Any) -> str | None:
    """``'DC'``, ``'VA'`` or ``'MD'`` for a point inside that outline, else ``None`` (bad input included)."""
    if not _valid(lat, lon):
        return None
    x, y = float(lon), float(lat)
    for ident, _agency, (x0, y0, x1, y1), rings in _load():
        if x0 <= x <= x1 and y0 <= y <= y1 and _inside(x, y, rings):
            return ident
    return None


def agency_for_point(lat: Any, lon: Any) -> str | None:
    """``'VDOT'`` (Virginia), ``'MDOT SHA'`` (Maryland) or ``None`` (anywhere else, DC, bad input)."""
    ident = jurisdiction_for_point(lat, lon)
    if ident is None:
        return None
    for known, agency, _box, _rings in _load():
        if known == ident:
            return agency
    return None
