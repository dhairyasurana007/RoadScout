"""Task 17 backend: the update response and the list never expose filesystem paths."""
import json
import re

from rs17_helpers import rs17_env  # noqa: F401

PATH_KEYS = {"photo_path", "crop_path", "best_frame_path", "blurred_frame_path", "blurred_crop_path", "video_path",
             "gpx_path", "path", "file", "filename", "original_photo_path"}


def walk(o, found, key=""):
    if isinstance(o, dict):
        for k, v in o.items():
            walk(v, found, k)
    elif isinstance(o, list):
        for v in o:
            walk(v, found, key)
    else:
        found.append((key, o))


def check(env, payload):
    text = json.dumps(payload)
    leaves = []
    walk(payload, leaves)
    for k, v in leaves:
        assert k not in PATH_KEYS, "path-like key %r exposed" % k
        if isinstance(v, str):
            assert str(env.tmp) not in v and str(env.tmp).replace("\\", "/") not in v, "%s=%r leaks a local path" % (k, v)
            assert not re.match(r"^[A-Za-z]:[\\/]", v), "%s=%r looks like a drive path" % (k, v)
            assert "\\" not in v or k in ("reason",), "%s=%r contains a backslash" % (k, v)
            if k.endswith("_url"):
                assert v.startswith("/api/"), "%s=%r" % (k, v)
    assert str(env.tmp).replace("\\", "/") not in text.replace("\\\\", "/")


def test_update_response_and_list_have_no_filesystem_paths(rs17_env):
    env = rs17_env
    trio = env.trio()
    for hid, _ in trio:
        crop = env.tmp / "photos" / ("c%d.jpg" % hid)
        crop.write_bytes(b"\xff\xd8x\xff\xd9")
        env.x("UPDATE hazards SET crop_path = ? WHERE id = ?", (str(crop), hid))
    for hid, _ in trio:
        for fields in ({"status": "approved"}, {"severity": 3}, {"hazard_class": "faded lane markings"},
                       {"status": "new", "severity": 1}):
            check(env, env.ok(hid, **fields))
    check(env, env.hazards())
    check(env, env.hazards(status="approved") if "status" in env.list_params() else [])


def test_error_responses_do_not_leak_paths(rs17_env):
    env = rs17_env
    (hid, _), *_ = env.trio()
    for body in ({"status": "banana"}, {"photo_path": "x"}, {}):
        r = env.send(hid, json=body)
        assert str(env.tmp) not in r.text
    r = env.update(9999, status="approved")
    assert str(env.tmp) not in r.text
