"""The prefilled agency report of one approved hazard (task 18). Nothing here sends anything anywhere.

    from backend.report import build_report
    report = build_report(hazard_row, verdict_row_or_None, has_photo=True)

``build_report`` is pure (no database, no network, no files): the routes in ``backend/hazards.py`` load the
hazard, the verdict of its best member candidate (``load_report_verdict``) and whether a blurred photo can be
served, and call it. RoadScout NEVER submits a report to VDOT or MDOT SHA: this module has no HTTP client, the
only outbound thing is the ``form_url`` the user may open themselves, and ``submitted`` is always ``false``.

The report (what ``GET /api/hazards/{id}/report`` returns), keys:

- ``hazard_id``, ``hazard_status`` (``approved`` or ``reported``), ``hazard_class``
- ``agency`` (``"VDOT"``, ``"MDOT SHA"`` or ``null`` = no agency), ``agency_name`` (the long name or ``null``),
  ``agency_source`` (``"auto"``: worked out from the position by ``backend/agency.py``; ``"user"``: chosen on the
  report screen, stored in ``hazards.agency_override``), ``automatic_agency`` (what the position says)
- ``category``: the agency's category for the hazard class (``null`` with no agency), see the tables below
- ``sub_topic_note``: MDOT SHA's form has a required Sub-Topic that depends on the Topic and whose options are not
  published, so it is left to the user ("Choose on the form ..."); ``null`` with no agency
- ``fields``: the agency's form fields, key to prefilled text or ``null`` (left to the user); ``field_info``: key to
  ``{label, required, note}``; ``verified_fields``: ``true`` when the field list was checked against the live form
  (MDOT SHA: topic, sub-topic, description, county, route, intersecting route, lane/ramp/shoulder, direction and
  contact method, from https://mdotsha.my.salesforce-sites.com/customercare), ``false`` for VDOT (its form is a
  guided JavaScript flow whose field names could NOT be verified) and for no agency; ``verification_note`` says so
- ``description``: the prefilled free text (class, severity, direction of travel, sightings, first/last seen in UTC,
  lane and the model's reason when available, coordinates and an OpenStreetMap link)
- ``location``: ``{lat, lon, compass, heading, map_url}`` (``compass`` is the 8-point direction of travel such as
  ``"NE"``, ``null`` when unknown); ``lat``, ``lon``, ``compass`` and ``map_url`` are repeated at the top level.
  The model's lane, ``report_description`` and ``reason`` (from the best member candidate's verdict) are left out
  when the user changed the hazard's class by hand; each is clipped to 240 characters (the description stays well
  under 1200)
- ``photo_url`` (``/api/hazards/{id}/photo``, the BLURRED photo; ``null`` when none can be served),
  ``photo_download_url`` (``...?download=1``), ``text_download_url`` (``.../report?format=txt``), ``photo_note``
- ``form_url``: the agency's official form (``null`` with no agency)
- ``text``: the whole report as plain text (what Copy and Download put out)
- ``submitted`` (always ``false``), ``note`` (nothing was submitted; the user submits it), ``agency_note``

Mappings (hazard class of ``config/hazard_prompts.json`` to the agency's category):

=========================== =========================================== =========================
class                       MDOT SHA Topic (verified)                    VDOT category (UNVERIFIED)
=========================== =========================================== =========================
pothole                     Potholes / Holes / Sinkholes                 Pothole repair
debris in lane              Debris, Graffiti, & Litter                   Debris in the road
damaged guardrail           Guardrail/Roadway Safety Devices             Damaged guardrail
fallen or damaged sign      Sign / Digital Message Sign                  Damaged or missing sign
faded lane markings         Line Striping / Roadway & Cross Walk Markings Faded pavement markings
=========================== =========================================== =========================

An unknown class (the prompts file was changed) falls back to MDOT SHA "Other" and VDOT "Other road problem".
The MDOT SHA form has NO photo upload and NO GPS/map field, so the coordinates and the map link are part of the
description and the user attaches nothing; VDOT's photo step could not be checked, so it is also left to the user.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agency import MDOT_SHA, VDOT, agency_for_point
from .db import get_connection, init_db

REPORTABLE = ("approved", "reported")

AGENCY_NAMES = {VDOT: "Virginia Department of Transportation", MDOT_SHA: "Maryland Department of Transportation State Highway Administration"}
FORM_URLS = {VDOT: "https://my.vdot.virginia.gov/", MDOT_SHA: "https://mdotsha.my.salesforce-sites.com/customercare"}

MDOT_TOPICS = {
    "pothole": "Potholes / Holes / Sinkholes",
    "debris in lane": "Debris, Graffiti, & Litter",
    "damaged guardrail": "Guardrail/Roadway Safety Devices",
    "fallen or damaged sign": "Sign / Digital Message Sign",
    "faded lane markings": "Line Striping / Roadway & Cross Walk Markings",
}
MDOT_FALLBACK_TOPIC = "Other"
# UNVERIFIED, check on the live form: VDOT's report flow is guided JavaScript and its real category names could not
# be read. These are best-effort labels for "kind of repair/problem".
VDOT_CATEGORIES = {
    "pothole": "Pothole repair",
    "debris in lane": "Debris in the road",
    "damaged guardrail": "Damaged guardrail",
    "fallen or damaged sign": "Damaged or missing sign",
    "faded lane markings": "Faded pavement markings",
}
VDOT_FALLBACK_CATEGORY = "Other road problem"

NOTHING_SUBMITTED = (
    "Nothing was submitted. RoadScout never sends a report to VDOT or MDOT SHA: you submit this report yourself on the "
    "agency's own form."
)
SUB_TOPIC_NOTE = (
    "Choose on the form: the required Sub-Topic depends on the Topic and its options are not published, so pick the "
    "closest one yourself."
)
VDOT_SUB_NOTE = "Choose on the form: VDOT's guided flow asks follow-up questions that could not be checked in advance."
VDOT_UNVERIFIED = "unverified, check on the live form"
MDOT_VERIFICATION = (
    "The field list was read from the live MDOT SHA service request form. Its Sub-Topic options and the contact "
    "method details are not published, so those are left to you."
)
VDOT_VERIFICATION = (
    "Unverified: VDOT's report flow is a guided JavaScript form and its field names and categories could not be "
    "checked. Treat every field name and category below as a suggestion and check it on the live form."
)
NONE_VERIFICATION = "There is no agency form for this hazard, so no field list applies."
PHOTO_NOTE_MDOT = "The MDOT SHA form has no photo upload. Download the blurred photo if you want to keep it or send it another way."
PHOTO_NOTE_VDOT = "Download the blurred photo and attach it on the VDOT form if it asks for one (unverified)."
PHOTO_NOTE_NONE = "Download the blurred photo if you want to send it to the road's owner yourself."
NO_PHOTO_NOTE = "There is no photo for this hazard."

SEVERITY_LABELS = {1: "low", 2: "medium", 3: "high"}
COMPASS_8 = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
COMPASS_NAMES = {"N": "north", "NE": "northeast", "E": "east", "SE": "southeast", "S": "south", "SW": "southwest",
                 "W": "west", "NW": "northwest"}
COMPASS_4 = ("N", "E", "S", "W")


class ReportNotAvailable(Exception):
    """The hazard is not approved (or reported) yet, so there is no report. The route answers 409."""


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def effective_agency(row: dict[str, Any]) -> tuple[str | None, str | None, str]:
    """(agency in use, automatic agency, source) of a hazard row: ``'user'`` when ``agency_override`` is set."""
    automatic = agency_for_point(row.get("lat"), row.get("lon"))
    override = row.get("agency_override")
    if override is None:
        return automatic, automatic, "auto"
    return (None if override == "none" else override), automatic, "user"


def compass_8(direction: Any) -> str | None:
    if not _finite(direction):
        return None
    return COMPASS_8[int(((float(direction) % 360.0) + 22.5) // 45) % 8]


def compass_4(direction: Any) -> str | None:
    if not _finite(direction):
        return None
    return COMPASS_4[int(((float(direction) % 360.0) + 45.0) // 90) % 4]


def map_url(lat: float, lon: float) -> str:
    return f"https://www.openstreetmap.org/?mlat={lat:.6f}&mlon={lon:.6f}#map=18/{lat:.6f}/{lon:.6f}"


def _utc_text(text: Any) -> str:
    if isinstance(text, str):
        try:
            parsed = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
            parsed = parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
            return parsed.strftime("%Y-%m-%d %H:%M UTC")
        except ValueError:
            return text.strip()
    return "unknown time"


def _clip(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _sentence(text: Any, limit: int = 240) -> str:
    text = _clip(" ".join(str(text).split()), limit)
    return text if text.endswith((".", "!", "?")) else text + "."


def load_report_verdict(hazard_id: int, *, db_path: str | Path | None = None) -> dict[str, Any] | None:
    """The accepted verdict of the hazard's best member candidate (the one whose photo the hazard shows), or None."""
    conn = get_connection(init_db(db_path))
    try:
        row = conn.execute(
            "SELECT v.hazard_class, v.severity, v.lane, v.reason, v.report_description "
            "FROM candidates c JOIN verdicts v ON v.candidate_id = c.id "
            "WHERE c.hazard_id = ? AND v.accepted = 1 AND v.is_hazard = 1 "
            "ORDER BY c.score DESC, v.confidence DESC, c.drive_id, c.track_key, c.best_timestamp, c.id LIMIT 1",
            (hazard_id,),
        ).fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()


def _description(row: dict[str, Any], verdict: dict[str, Any] | None, lat: float | None, lon: float | None) -> str:
    cls = str(row["hazard_class"])
    sev = row.get("severity")
    sev_text = f"{SEVERITY_LABELS[sev]} ({sev} of 3)" if sev in SEVERITY_LABELS else str(sev)
    parts = [f"Road hazard: {cls}. Severity: {sev_text}."]
    compass = compass_8(row.get("direction"))
    if compass:
        parts.append(f"Direction of travel: {COMPASS_NAMES[compass]} ({float(row['direction']) % 360.0:.0f} degrees).")
    sightings = row.get("sightings")
    if isinstance(sightings, int) and sightings > 0:
        first, last = _utc_text(row.get("first_seen")), _utc_text(row.get("last_seen"))
        when = first if first == last else f"first {first}, last {last}"
        parts.append(f"Seen on {sightings} {'drive' if sightings == 1 else 'drives'} ({when}).")
    # (build_report passes no verdict when the user changed the class by hand: the model's words would not fit.)
    if verdict:
        if isinstance(verdict.get("lane"), str) and verdict["lane"].strip():
            parts.append(f"Lane: {_sentence(verdict['lane'])}")
        if isinstance(verdict.get("report_description"), str) and verdict["report_description"].strip():
            parts.append(f"Details: {_sentence(verdict['report_description'])}")
        if isinstance(verdict.get("reason"), str) and verdict["reason"].strip():
            parts.append(f"Why it was flagged: {_sentence(verdict['reason'])}")
    if lat is not None and lon is not None:
        parts.append(f"Location: {lat:.6f}, {lon:.6f} (latitude, longitude). Map: {map_url(lat, lon)}")
    else:
        parts.append("Location: unknown.")
    return " ".join(parts)


def _mdot_fields(category: str, description: str, direction: str | None, verdict: dict[str, Any] | None,
                 cls: str) -> tuple[dict[str, Any], dict[str, Any]]:
    lane = None
    if verdict and isinstance(verdict.get("lane"), str) and verdict["lane"].strip():
        lane = _clip(" ".join(verdict["lane"].split()), 60)
    lane_note = ("Choose one of All Lanes, Driveway, HOV Lane, Lane 1 (Fast), Lane 2 (Middle), Lane 3 (Slow), Lane 4, "
                 "Ramp, Shoulder, Single Lane Road, Sound Barrier."
                 + (f" The lane the model saw it in: {lane}." if lane else ""))
    fields = {
        "topic": category,
        "sub_topic": None,
        "description": description,
        "county": None,
        "route": None,
        "closest_intersecting_route": None,
        "lane_ramp_shoulder": None,
        "direction": direction,
        "contact_method": None,
    }
    info = {
        "topic": {"label": "Topic", "required": True, "note": None},
        "sub_topic": {"label": "Sub-Topic", "required": True, "note": SUB_TOPIC_NOTE},
        "description": {"label": "Description", "required": False,
                        "note": "Includes the coordinates and a map link because the form has no location or photo field."},
        "county": {"label": "County", "required": False, "note": "Left to you: pick the Maryland county or Baltimore City."},
        "route": {"label": "Route Name or Route Number", "required": False, "note": "Left to you: the app does not know the road name."},
        "closest_intersecting_route": {"label": "Closest Intersecting Route/Road", "required": False,
                                       "note": "Left to you."},
        "lane_ramp_shoulder": {"label": "Lane/Ramp/Shoulder", "required": False, "note": lane_note},
        "direction": {"label": "Direction", "required": False,
                      "note": "From the direction of travel when it was seen; options are N, S, E, W, N&S and E&W."
                      if direction else "Direction of travel unknown; choose on the form."},
        "contact_method": {"label": "Contact method", "required": False,
                           "note": "Left to you: Email, Call/Phone or Letter, and your contact details."},
    }
    return fields, info


def _vdot_fields(category: str, description: str, lat: float | None, lon: float | None,
                 cls: str) -> tuple[dict[str, Any], dict[str, Any]]:
    where = f"{lat:.6f}, {lon:.6f} (latitude, longitude); map: {map_url(lat, lon)}" if lat is not None and lon is not None else None
    fields = {
        "problem_type": category,
        "location": where,
        "description": description,
        "blocks_traffic": None,
    }
    info = {
        "problem_type": {"label": "Kind of repair or problem", "required": True, "note": VDOT_UNVERIFIED},
        "location": {"label": "Where it is", "required": True,
                     "note": f"{VDOT_UNVERIFIED}; the form may let you drop a pin or type an address instead."},
        "description": {"label": "Description", "required": False, "note": VDOT_UNVERIFIED},
        "blocks_traffic": {"label": "Does it block traffic?", "required": False,
                           "note": f"Left to you; {VDOT_UNVERIFIED}."},
    }
    return fields, info


def _plain_text(report: dict[str, Any], row: dict[str, Any]) -> str:
    lines = [
        f"RoadScout prefilled report for hazard {report['hazard_id']} (NOT SUBMITTED)",
        report["note"],
        "",
    ]
    if report["agency"]:
        source = "chosen by you" if report["agency_source"] == "user" else "worked out from the location"
        lines += [f"Agency: {report['agency']} ({report['agency_name']}), {source}",
                  f"Official form: {report['form_url']}", report["agency_note"]]
        if not report["verified_fields"]:
            lines.append("WARNING: " + report["verification_note"])
        lines.append("")
        lines.append(f"Category: {report['category']}")
        if report["sub_topic_note"]:
            lines.append(f"Sub-Topic: {report['sub_topic_note']}")
        lines += ["", "Form fields:"]
        for key, value in report["fields"].items():
            info = report["field_info"][key]
            flag = " (required)" if info["required"] else ""
            if key == "description":
                shown = "see Description below"
            elif value:
                shown = value
            else:
                shown = "(choose on the form)" if key == "sub_topic" else "(leave to you)"
            lines.append(f"- {info['label']}{flag}: {shown}")
    else:
        source = "chosen by you" if report["agency_source"] == "user" else "worked out from the location"
        lines += [f"Agency: none ({source})", report["agency_note"]]
    lines += ["", "Description:", report["description"], ""]
    loc = report["location"]
    if loc["lat"] is not None:
        lines.append(f"Location: {loc['lat']:.6f}, {loc['lon']:.6f}")
        lines.append(f"Map: {loc['map_url']}")
    else:
        lines.append("Location: unknown")
    lines.append(f"Photo: {report['photo_note']}")
    if report["photo_url"]:
        lines.append(f"Blurred photo: {report['photo_download_url']} (served by RoadScout on your computer)")
    return "\n".join(lines) + "\n"


def build_report(row: dict[str, Any], verdict: dict[str, Any] | None = None, *, has_photo: bool = True) -> dict[str, Any]:
    """The report of an approved or reported hazard row (dict with the ``hazards`` columns).
    Raises ``ReportNotAvailable`` for any other status."""
    if row["status"] not in REPORTABLE:
        state = "possibly fixed" if row["status"] == "possibly_fixed" else row["status"]
        raise ReportNotAvailable(
            f"A report is only available for approved hazards, and this hazard is {state}. Approve it first.")
    hazard_id = int(row["id"])
    if row.get("class_edited"):
        verdict = None
    lat = float(row["lat"]) if _finite(row.get("lat")) else None
    lon = float(row["lon"]) if _finite(row.get("lon")) else None
    agency, automatic, source = effective_agency(row)
    cls = str(row["hazard_class"])
    description = _description(row, verdict, lat, lon)
    direction4 = compass_4(row.get("direction"))

    if agency == MDOT_SHA:
        category = MDOT_TOPICS.get(cls, MDOT_FALLBACK_TOPIC)
        fields, info = _mdot_fields(category, description, direction4, verdict, cls)
        verified, verification, sub_note = True, MDOT_VERIFICATION, SUB_TOPIC_NOTE
        photo_note = PHOTO_NOTE_MDOT
        agency_note = ("MDOT SHA looks after Maryland state roads only. County and city streets belong to the county "
                       "or city, so change the agency if this is not a state road.")
    elif agency == VDOT:
        category = VDOT_CATEGORIES.get(cls, VDOT_FALLBACK_CATEGORY)
        fields, info = _vdot_fields(category, description, lat, lon, cls)
        verified, verification, sub_note = False, VDOT_VERIFICATION, VDOT_SUB_NOTE
        photo_note = PHOTO_NOTE_VDOT
        agency_note = ("VDOT looks after state-maintained roads. Some Virginia cities, towns and two counties "
                       "(Arlington and Henrico) maintain their own streets, so change the agency if this is not a "
                       "VDOT road.")
    else:
        category, fields, info, sub_note = None, {}, {}, None
        verified, verification, photo_note = False, NONE_VERIFICATION, PHOTO_NOTE_NONE
        if source == "user":
            agency_note = "No agency: you chose not to send this hazard to VDOT or MDOT SHA."
        else:
            agency_note = ("No agency: this location is not in Virginia or Maryland (for example the District of "
                           "Columbia), so there is no VDOT or MDOT SHA form for it. You can choose an agency "
                           "yourself.")
    if not has_photo:
        photo_note = NO_PHOTO_NOTE

    report: dict[str, Any] = {
        "hazard_id": hazard_id,
        "hazard_status": row["status"],
        "hazard_class": cls,
        "agency": agency,
        "agency_name": AGENCY_NAMES.get(agency),
        "agency_source": source,
        "automatic_agency": automatic,
        "category": category,
        "sub_topic_note": sub_note,
        "fields": fields,
        "field_info": info,
        "verified_fields": verified,
        "verification_note": verification,
        "description": description,
        "lat": lat,
        "lon": lon,
        "compass": compass_8(row.get("direction")),
        "map_url": map_url(lat, lon) if lat is not None and lon is not None else None,
        "location": {
            "lat": lat,
            "lon": lon,
            "compass": compass_8(row.get("direction")),
            "heading": float(row["direction"]) % 360.0 if _finite(row.get("direction")) else None,
            "map_url": map_url(lat, lon) if lat is not None and lon is not None else None,
        },
        "photo_url": f"/api/hazards/{hazard_id}/photo" if has_photo else None,
        "photo_download_url": f"/api/hazards/{hazard_id}/photo?download=1" if has_photo else None,
        "text_download_url": f"/api/hazards/{hazard_id}/report?format=txt",
        "photo_note": photo_note,
        "form_url": FORM_URLS.get(agency),
        "submitted": False,
        "note": NOTHING_SUBMITTED,
        "agency_note": agency_note,
    }
    report["text"] = _plain_text(report, row)
    return report
