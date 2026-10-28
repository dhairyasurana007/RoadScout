"""Throwaway spike (task 26): ask Muse Spark to verify a SAM 3.1 candidate on one real frame.

Grabs the 720p frame at T seconds, asks SAM 3.1 (image mode) for PHRASE to get a box, crops the
best box (with margin), then sends the frame and the crop to Muse Spark with the prompt from
config/verification_prompt.json. Prints the verdict text and the token usage. Reads META_API_KEY
from .env and never prints it. Saves the crop to OUT_DIR so a human can look at it.
Usage: python spike/probe_verify.py VIDEO T "phrase" "detector class" OUT_DIR [--model muse-spark-1.3]
"""
from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path

import cv2
import httpx
import numpy as np
from dotenv import dotenv_values

from probe_sam_image import grab_frame
from probe_sam_video import BASE, ROOT, fields

CLASSES = ["pothole", "debris in lane", "damaged guardrail", "fallen or damaged sign", "faded lane markings"]


def data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()


def text_of(payload: dict) -> str:
    parts = []
    for item in payload.get("output", []):
        for c in item.get("content", []) or []:
            if "text" in c:
                parts.append(c["text"])
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("t", type=float)
    ap.add_argument("phrase")
    ap.add_argument("detector_class")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default="muse-spark-1.3")
    args = ap.parse_args()

    key = (dotenv_values(ROOT / ".env").get("META_API_KEY") or "").strip()
    if not key:
        raise SystemExit("META_API_KEY is not set in .env")
    auth = {"Authorization": f"Bearer {key}"}
    cfg = json.loads((ROOT / "config" / "verification_prompt.json").read_text(encoding="utf-8"))

    jpeg = grab_frame(args.video, args.t)
    sam = httpx.post(f"{BASE}/responses", headers=auth, timeout=120, json={
        "model": "sam-3.1",
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": args.phrase},
            {"type": "input_image", "image_url": data_url(jpeg)}]}],
        "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
    })
    sam.raise_for_status()
    boxes = [fields(b) for b in re.findall(r"<\|box;([^|]*)\|>", text_of(sam.json()))]
    print(f"SAM boxes for {args.phrase!r}: {len(boxes)}")
    if not boxes:
        return
    best = max(boxes, key=lambda b: b.get("c", 0))
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    mx, my = int((best["x2"] - best["x1"]) * 0.5) + 20, int((best["y2"] - best["y1"]) * 0.5) + 20
    x1, y1 = max(0, int(best["x1"]) - mx), max(0, int(best["y1"]) - my)
    x2, y2 = min(w, int(best["x2"]) + mx), min(h, int(best["y2"]) + my)
    crop = cv2.imencode(".jpg", img[y1:y2, x1:x2])[1].tobytes()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"{Path(args.video).stem}_{int(args.t)}s"
    (out / f"{tag}_crop.jpg").write_bytes(crop)
    (out / f"{tag}_frame.jpg").write_bytes(jpeg)

    prompt = cfg["prompt"].format(
        detector_class=args.detector_class, schema=json.dumps(cfg["schema"]), classes=", ".join(CLASSES))
    body = {
        "model": args.model,
        "input": [{"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": prompt},
            {"type": "input_image", "image_url": data_url(jpeg)},
            {"type": "input_image", "image_url": data_url(crop)}]}],
        "text": {"format": {"type": "json_schema", "name": "verdict", "schema": cfg["schema"]}},
    }
    resp = httpx.post(f"{BASE}/responses", headers=auth, json=body, timeout=180)
    print("Muse Spark:", resp.status_code)
    if resp.status_code != 200:
        print(resp.text[:600])
        return
    payload = resp.json()
    print("usage:", payload.get("usage"))
    print("verdict:", text_of(payload))


if __name__ == "__main__":
    main()
