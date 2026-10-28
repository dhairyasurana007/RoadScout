"""Throwaway spike (task 26): draw SAM 3.1 boxes on the matched frames so a human can judge them.

Same clip cutting and upload as probe_sam_video.py; for the longest tracks it saves one
contact sheet (a JPEG with the box and the track id drawn on the frame where the track is most
confident, or the middle frame when confidences tie). Output stays local (git-ignored or the
scratchpad); the uploaded copy is deleted afterwards.
Usage: python spike/show_matches.py VIDEO START_S DURATION_S OUT.jpg "prompt" [--fps 3] [--tracks 8]
"""
from __future__ import annotations

import argparse
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import httpx
import numpy as np
from dotenv import dotenv_values

from probe_sam_video import BASE, RECORD, ROOT, cut_clip, fields, output_text


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("start", type=float)
    ap.add_argument("duration", type=float)
    ap.add_argument("out")
    ap.add_argument("prompt")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--tracks", type=int, default=8)
    args = ap.parse_args()

    key = (dotenv_values(ROOT / ".env").get("META_API_KEY") or "").strip()
    if not key:
        raise SystemExit("META_API_KEY is not set in .env")
    auth = {"Authorization": f"Bearer {key}"}

    with tempfile.TemporaryDirectory() as tmp:
        clip = Path(tmp) / "clip.mp4"
        cut_clip(args.video, args.start, args.duration, args.fps, clip)
        with clip.open("rb") as f:
            up = httpx.post(f"{BASE}/files", headers=auth, data={"purpose": "user_data"},
                            files={"file": ("clip.mp4", f, "video/mp4")}, timeout=300)
        up.raise_for_status()
        file_id = up.json()["id"]
        try:
            body = {
                "model": "sam-3.1",
                "input": [{"type": "message", "role": "user", "content": [
                    {"type": "input_text", "text": args.prompt},
                    {"type": "input_video", "file_id": file_id},
                ]}],
                "metadata": {"mask_encoding": "one_bit", "include_confidence": "true"},
            }
            resp = httpx.post(f"{BASE}/responses", json=body, headers=auth, timeout=600)
            resp.raise_for_status()
        finally:
            httpx.delete(f"{BASE}/files/{file_id}", headers=auth, timeout=60)
        text = output_text(resp.json())
        # frame index -> list of (track id, fields)
        per_frame: dict[int, list[tuple[str, dict[str, float]]]] = defaultdict(list)
        for ln in re.split(r"(?=<\d+f>)", text):
            m = re.match(r"<(\d+)f>", ln)
            if not m:
                continue
            for oid, box, _mask in RECORD.findall(ln):
                per_frame[int(m.group(1))].append((oid, fields(box)))
        frames_of: dict[str, list[int]] = defaultdict(list)
        for idx, recs in per_frame.items():
            for oid, _ in recs:
                frames_of[oid].append(idx)
        top = sorted(frames_of, key=lambda o: -len(frames_of[o]))[: args.tracks]
        print(f"{len(per_frame)} frames with objects, {len(frames_of)} tracks, showing {len(top)}")
        cap = cv2.VideoCapture(str(clip))
        tiles = []
        for oid in top:
            idxs = sorted(frames_of[oid])
            idx = idxs[len(idxs) // 2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, img = cap.read()
            if not ok:
                continue
            for o2, fld in per_frame[idx]:
                x1, y1, x2, y2 = (int(fld[k]) for k in ("x1", "y1", "x2", "y2"))
                colour = (0, 0, 255) if o2 == oid else (0, 200, 255)
                cv2.rectangle(img, (x1, y1), (x2, y2), colour, 3)
                cv2.putText(img, o2, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)
            t = args.start + idx / args.fps
            label = f"track {oid}  t={t:.0f}s  frames={len(idxs)}"
            cv2.putText(img, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3)
            cv2.putText(img, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1)
            tiles.append(cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2)))
        cap.release()  # Windows cannot delete the temp clip while OpenCV holds it open
        if not tiles:
            print("nothing to draw")
            return
        cols = 4
        while len(tiles) % cols:
            tiles.append(np.zeros_like(tiles[0]))
        rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
        cv2.imwrite(args.out, np.vstack(rows))
        print("saved", args.out)


if __name__ == "__main__":
    main()
