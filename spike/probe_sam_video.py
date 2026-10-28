"""Throwaway spike (task 26): a short real clip -> SAM 3.1 video segmentation.

Cuts a sampled (default 3 fps, 720p, H.264, faststart) clip with ffmpeg, uploads it with
POST /v1/files, runs one SAM 3.1 request per prompt (one concept per request), prints a
summary and deletes the uploaded file. Reads META_API_KEY from .env and never prints it.
Usage: python spike/probe_sam_video.py VIDEO START_S DURATION_S [--fps 3] prompt [prompt ...]
"""
from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://api.meta.ai/v1"
FRAME = re.compile(r"<(\d+)f>")
RECORD = re.compile(r"(\d+)<\|box;([^|]*)\|><\|mask;([^|]*)\|>")


def cut_clip(video: str, start: float, duration: float, fps: float, out: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", str(duration), "-i", video,
         "-an", "-vf", f"fps={fps},scale=-2:720", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(out)],
        check=True,
    )


def fields(box: str) -> dict[str, float]:
    return {k: float(v) for k, v in (p.split("=") for p in box.split(";") if "=" in p)}


def output_text(payload: dict) -> str:
    parts = []
    for item in payload.get("output", []):
        for c in item.get("content", []) or []:
            if "text" in c:
                parts.append(c["text"])
    return "".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("start", type=float)
    ap.add_argument("duration", type=float)
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("prompts", nargs="+")
    args = ap.parse_args()

    key = (dotenv_values(ROOT / ".env").get("META_API_KEY") or "").strip()
    if not key:
        raise SystemExit("META_API_KEY is not set in .env")
    auth = {"Authorization": f"Bearer {key}"}

    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "clip.mp4"
        cut_clip(args.video, args.start, args.duration, args.fps, clip)
        print(f"clip: {clip.stat().st_size / 1e6:.2f} MB")
        with clip.open("rb") as f:
            up = httpx.post(f"{BASE}/files", headers=auth, data={"purpose": "user_data"},
                            files={"file": ("clip.mp4", f, "video/mp4")}, timeout=300)
        print("upload:", up.status_code, {k: v for k, v in up.json().items() if k in ("id", "status", "bytes")}
              if up.status_code == 200 else up.text[:300])
        if up.status_code != 200:
            return
        file_id = up.json()["id"]
        try:
            for prompt in args.prompts:
                body = {
                    "model": "sam-3.1",
                    "input": [{"type": "message", "role": "user", "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_video", "file_id": file_id},
                    ]}],
                    "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
                }
                started = time.monotonic()
                resp = httpx.post(f"{BASE}/responses", json=body, headers=auth, timeout=600)
                took = time.monotonic() - started
                print(f"\n== {prompt!r}: HTTP {resp.status_code} in {took:.1f}s")
                if resp.status_code != 200:
                    print(resp.text[:400])
                    continue
                payload = resp.json()
                print("usage:", payload.get("usage"))
                text = output_text(payload)
                # one line per frame: "<Nf>id<|box..|><|mask..|>,id<|box..|><|mask..|>"
                lines = [ln for ln in re.split(r"(?=<\d+f>)", text) if FRAME.match(ln)]
                per_track: dict[str, list[float]] = defaultdict(list)
                frames_with = 0
                for ln in lines:
                    recs = RECORD.findall(ln)
                    frames_with += bool(recs)
                    for oid, box, _mask in recs:
                        per_track[oid].append(fields(box).get("c", -1.0))
                print(f"frames in output: {len(lines)}  frames with objects: {frames_with}  tracks: {len(per_track)}")
                for oid, confs in sorted(per_track.items(), key=lambda kv: -len(kv[1]))[:8]:
                    print(f"  track {oid}: {len(confs)} frames, conf {min(confs):.2f}-{max(confs):.2f}")
        finally:
            d = httpx.delete(f"{BASE}/files/{file_id}", headers=auth, timeout=60)
            print("\ndeleted upload:", d.status_code)


if __name__ == "__main__":
    main()
