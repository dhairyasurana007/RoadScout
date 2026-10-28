"""Record live Meta responses as provider fixtures (manual only; never run in CI).

    ROADSCOUT_PROVIDER=live python scripts/record_fixtures.py CHUNK_VIDEO OUT_DIR

Runs the live provider on ONE preprocessed chunk with the configured hazard prompts (the first
phrasing of each class, as the pipeline sends them) and on one sample verification, then writes
``detect.json`` and ``verify.json`` into OUT_DIR in the FixtureProvider format (see
``backend/providers/fixture.py``). The verification sample is one frame cut from the chunk (the
best-scoring detection's frame, else the middle frame), with the configured verification prompt.
The frame itself is not saved.

It refuses to run unless ROADSCOUT_PROVIDER=live and META_API_KEY are set (environment or the
git-ignored .env), and it never prints the key. It spends real money (a few cents): the spend cap
and cost log apply as usual. The files hold masks and boxes derived from your footage, so review
them before committing anything, and never commit the footage or frames.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

from backend import config  # noqa: E402
from backend.hazard_prompts import load_hazard_prompts  # noqa: E402
from backend.providers import get_provider  # noqa: E402
from backend.verification_prompt import load_verification_prompt  # noqa: E402


def _sample_frame(video: Path, wanted: int | None) -> bytes:
    """JPEG bytes of frame ``wanted`` of the chunk (the middle frame when None)."""
    cap = cv2.VideoCapture(str(video))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        index = wanted if wanted is not None else max(0, total // 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("could not read a frame from the chunk")
        done, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not done:
            raise SystemExit("could not encode the sample frame")
        return buf.tobytes()
    finally:
        cap.release()


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: ROADSCOUT_PROVIDER=live python scripts/record_fixtures.py CHUNK_VIDEO OUT_DIR", file=sys.stderr)
        return 2
    if config.provider_name() != "live":
        print("refusing to run: set ROADSCOUT_PROVIDER=live (this script makes real, paid Meta calls)", file=sys.stderr)
        return 1
    if not config.meta_api_key():
        print("refusing to run: META_API_KEY is not set (put it in the environment or the git-ignored .env)",
              file=sys.stderr)
        return 1
    video = Path(argv[1])
    out_dir = Path(argv[2])
    if not video.is_file():
        print("CHUNK_VIDEO is not a file", file=sys.stderr)
        return 1

    provider = get_provider("live")
    hazard_prompts = load_hazard_prompts()
    prompts = hazard_prompts.default_prompts()

    print(f"detect: {len(prompts)} prompts on {video.name}")
    result = provider.detect(str(video), prompts)
    detections = [(f, d) for f, d in result.all_detections()]
    print(f"  {len(result.frames)} frames with matches, {len(detections)} detections")

    best = max(detections, key=lambda fd: fd[1].score, default=None)
    frame_index = best[0].frame_index if best else None
    detector_class = (hazard_prompts.class_for_prompt(best[1].prompt) if best else None) or (
        hazard_prompts.class_names[0]
    )
    prompt = load_verification_prompt()
    jpeg = _sample_frame(video, frame_index)
    print(f"verify: one frame ({'best detection' if best else 'middle of the chunk'}), class {detector_class!r}")
    verdict = provider.verify(
        [jpeg],
        class_prompt=detector_class,
        prompt_version=prompt.version,
        prompt_text=prompt.render(detector_class, hazard_prompts.class_names),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "detect.json").write_text(json.dumps(result.to_dict()), encoding="utf-8")
    (out_dir / "verify.json").write_text(json.dumps(verdict.to_dict(), indent=2), encoding="utf-8")
    print(f"wrote {out_dir / 'detect.json'} and {out_dir / 'verify.json'}")
    print("review them before committing; never commit footage or frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
