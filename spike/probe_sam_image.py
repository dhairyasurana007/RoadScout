"""Throwaway spike (task 26): one real frame -> SAM 3.1, to check the key and the response format.

Reads META_API_KEY from the git-ignored .env and never prints it. Prints a short summary of
each response (status, how many objects, first box, mask header), not the raw masks.
Usage: python spike/probe_sam_image.py spike/footage/IMG_6503.MOV 30 car pothole
"""
from __future__ import annotations

import base64
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
URL = "https://api.meta.ai/v1/responses"


def grab_frame(video: str, seconds: float) -> bytes:
    # ffmpeg applies the rotation metadata, so the frame is upright.
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", str(seconds), "-i", video, "-frames:v", "1",
         "-vf", "scale=-2:720", "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", "3", "-"],
        capture_output=True, check=True,
    )
    return out.stdout


def main() -> None:
    video, seconds, prompts = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
    key = (dotenv_values(ROOT / ".env").get("META_API_KEY") or "").strip()
    if not key:
        sys.exit("META_API_KEY is not set in .env")
    jpeg = grab_frame(video, seconds)
    data_url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    print(f"frame: {len(jpeg)} bytes")
    for prompt in prompts:
        body = {
            "model": "sam-3.1",
            "input": [{
                "type": "message", "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
            "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
        }
        started = time.monotonic()
        resp = httpx.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=120)
        took = time.monotonic() - started
        print(f"\n== {prompt!r}: HTTP {resp.status_code} in {took:.1f}s")
        if resp.status_code != 200:
            print(resp.text[:500])
            continue
        payload = resp.json()
        print("top-level keys:", sorted(payload))
        if payload.get("usage"):
            print("usage:", payload["usage"])
        text = payload.get("output_text")
        if text is None:  # fall back to walking the Responses output items
            parts = []
            for item in payload.get("output", []):
                for c in item.get("content", []) or []:
                    if "text" in c:
                        parts.append(c["text"])
            text = "".join(parts)
        print("output_text length:", len(text))
        print("objects:", len(re.findall(r"<\|box;", text)))
        m = re.search(r"<\|box;[^|]*\|>", text)
        print("first box:", m.group(0) if m else None)
        m = re.search(r"<\|mask;[^,]*,[^,]*,[^,]*,.{0,12}", text)
        print("mask header:", m.group(0) if m else None)


if __name__ == "__main__":
    main()
