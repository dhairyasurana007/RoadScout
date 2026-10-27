"""Settings read from the environment (a git-ignored .env is loaded if present)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")


def db_path() -> Path:
    """SQLite file location. Override with ROADSCOUT_DB."""
    return Path(os.environ.get("ROADSCOUT_DB", ROOT / "data" / "roadscout.sqlite"))


def storage_dir() -> Path:
    """Where LocalDiskStorage keeps uploaded files. Override with ROADSCOUT_STORAGE_DIR."""
    return Path(os.environ.get("ROADSCOUT_STORAGE_DIR") or ROOT / "data" / "storage")


def meta_api_key() -> str | None:
    """The Meta API key, or None when unset. Never log or print this value."""
    return os.environ.get("META_API_KEY") or None


# --- Preprocessing settings (task 5) ----------------------------------------

DEFAULT_PREPROCESS_FPS = 3.0
DEFAULT_PREPROCESS_MAX_HEIGHT = 720
DEFAULT_CHUNK_SECONDS = 10.0


def _positive_number(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None
    if not value > 0 or value == float("inf"):
        raise ValueError(f"{name} must be a positive number, got {raw!r}")
    return value


def preprocess_fps() -> float:
    """Sampled frames per second. Set ROADSCOUT_PREPROCESS_FPS (default 3; the task 26 spike tunes it)."""
    return _positive_number("ROADSCOUT_PREPROCESS_FPS", DEFAULT_PREPROCESS_FPS)


def preprocess_max_height() -> int:
    """Maximum output height in pixels (never upscales). Set ROADSCOUT_PREPROCESS_MAX_HEIGHT (default 720)."""
    return int(_positive_number("ROADSCOUT_PREPROCESS_MAX_HEIGHT", DEFAULT_PREPROCESS_MAX_HEIGHT))


def chunk_seconds() -> float:
    """Target chunk length in seconds. Set ROADSCOUT_CHUNK_SECONDS (default 10)."""
    return _positive_number("ROADSCOUT_CHUNK_SECONDS", DEFAULT_CHUNK_SECONDS)


# --- Detection settings (task 7) --------------------------------------------


def hazard_prompts_path() -> Path:
    """JSON file holding the SAM 3.1 hazard class prompts.
    Set ROADSCOUT_HAZARD_PROMPTS (default: config/hazard_prompts.json)."""
    raw = (os.environ.get("ROADSCOUT_HAZARD_PROMPTS") or "").strip()
    return Path(raw) if raw else ROOT / "config" / "hazard_prompts.json"


# --- Stationary-stretch settings (task 13) ----------------------------------

DEFAULT_STATIONARY_SPEED_MPS = 0.5

_FALSE_WORDS = ("0", "false", "no", "off")


def skip_stationary_enabled() -> bool:
    """Whether detection drops frames recorded while the car was stopped (default: on).
    Set ROADSCOUT_SKIP_STATIONARY to 0/false/off/no (off) or 1/true/on/yes (on), case-insensitive.
    Anything else falls back to on. Read at call time."""
    word = (os.environ.get("ROADSCOUT_SKIP_STATIONARY") or "").strip().lower()
    return word not in _FALSE_WORDS


def stationary_speed_mps() -> float:
    """GPS speed in m/s below which a frame counts as stationary (a speed exactly at the value counts
    as moving). Set ROADSCOUT_STATIONARY_SPEED_MPS to a number of 0 or more (default 0.5); an unset,
    empty, non-numeric, negative or non-finite value falls back to the default. Read at call time."""
    raw = os.environ.get("ROADSCOUT_STATIONARY_SPEED_MPS")
    try:
        value = float(raw) if raw is not None and raw.strip() else DEFAULT_STATIONARY_SPEED_MPS
    except ValueError:
        return DEFAULT_STATIONARY_SPEED_MPS
    if not (0 <= value < float("inf")):  # also rejects NaN
        return DEFAULT_STATIONARY_SPEED_MPS
    return value


# --- Track aggregation settings (task 8) ------------------------------------

DEFAULT_MIN_TRACK_FRAMES = 3


def min_track_frames() -> int:
    """Fewest frames a tracked object must be seen in to become a candidate (a track with exactly
    this many is kept). Set ROADSCOUT_MIN_TRACK_FRAMES to a whole number of 1 or more (default 3;
    the task 26 spike sets the real value)."""
    raw = os.environ.get("ROADSCOUT_MIN_TRACK_FRAMES")
    if raw is None or not raw.strip():
        return DEFAULT_MIN_TRACK_FRAMES
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"ROADSCOUT_MIN_TRACK_FRAMES must be a whole number, got {raw!r}") from None
    if value != value or value in (float("inf"), float("-inf")) or value != int(value) or value < 1:
        raise ValueError(f"ROADSCOUT_MIN_TRACK_FRAMES must be a whole number of 1 or more, got {raw!r}")
    return int(value)


# --- Verification settings (task 9) -----------------------------------------

DEFAULT_VERIFY_MIN_CONFIDENCE = 0.5


def verification_prompt_path() -> Path:
    """JSON file holding the versioned Muse Spark verification prompt.
    Set ROADSCOUT_VERIFY_PROMPT (default: config/verification_prompt.json)."""
    raw = (os.environ.get("ROADSCOUT_VERIFY_PROMPT") or "").strip()
    return Path(raw) if raw else ROOT / "config" / "verification_prompt.json"


def verify_min_confidence() -> float:
    """Minimum confidence for a verdict to be accepted (inclusive). Set ROADSCOUT_VERIFY_MIN_CONFIDENCE
    to a number from 0 to 1 (default 0.5)."""
    raw = os.environ.get("ROADSCOUT_VERIFY_MIN_CONFIDENCE")
    if raw is None or not raw.strip():
        return DEFAULT_VERIFY_MIN_CONFIDENCE
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"ROADSCOUT_VERIFY_MIN_CONFIDENCE must be a number from 0 to 1, got {raw!r}") from None
    if not 0.0 <= value <= 1.0:  # also rejects NaN
        raise ValueError(f"ROADSCOUT_VERIFY_MIN_CONFIDENCE must be a number from 0 to 1, got {raw!r}")
    return value


# --- Deduplication settings (task 14) ---------------------------------------

DEFAULT_DEDUP_DISTANCE_M = 25.0
DEFAULT_DEDUP_HEADING_TOLERANCE_DEG = 45.0


def _bounded_number(name: str, default: float, low: float, high: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number from {low:g} to {high:g}, got {raw!r}") from None
    if not low <= value <= high:  # also rejects NaN
        raise ValueError(f"{name} must be a number from {low:g} to {high:g}, got {raw!r}")
    return value


def dedup_distance_m() -> float:
    """Largest distance in metres between two candidates of one hazard (inclusive). Set
    ROADSCOUT_DEDUP_DISTANCE_M to a number of 0 or more (default 25); an invalid value raises
    ValueError. Read at call time."""
    return _bounded_number("ROADSCOUT_DEDUP_DISTANCE_M", DEFAULT_DEDUP_DISTANCE_M, 0.0, 1e7)


def dedup_heading_tolerance_deg() -> float:
    """Largest difference in direction of travel in degrees between two candidates of one hazard
    (inclusive, wrap-around aware, so 350 and 10 differ by 20). Set
    ROADSCOUT_DEDUP_HEADING_TOLERANCE_DEG to a number from 0 to 180 (default 45; 180 ignores direction
    for candidates that have a heading); an invalid value raises ValueError. Read at call time."""
    return _bounded_number(
        "ROADSCOUT_DEDUP_HEADING_TOLERANCE_DEG", DEFAULT_DEDUP_HEADING_TOLERANCE_DEG, 0.0, 180.0
    )


# --- Model provider settings (task 3) ---------------------------------------

DEFAULT_SPEND_CAP_USD = 20.0


def provider_name() -> str:
    """Which provider to use: fake (default), fixture or live. Set ROADSCOUT_PROVIDER."""
    return (os.environ.get("ROADSCOUT_PROVIDER") or "fake").strip().lower()


def fake_scenario() -> str:
    """Canned scenario for the fake provider. Set ROADSCOUT_FAKE_SCENARIO."""
    return (os.environ.get("ROADSCOUT_FAKE_SCENARIO") or "default").strip().lower()


def fixtures_dir() -> Path:
    """Directory holding detect.json and verify.json for the fixture provider.
    Set ROADSCOUT_FIXTURES_DIR (default: fixtures/providers/tiny)."""
    return Path(os.environ.get("ROADSCOUT_FIXTURES_DIR") or ROOT / "fixtures" / "providers" / "tiny")


def spend_cap_usd() -> float:
    """Hard cap on live spend in USD. Set ROADSCOUT_SPEND_CAP_USD (default 20)."""
    raw = os.environ.get("ROADSCOUT_SPEND_CAP_USD")
    if raw is None or not raw.strip():
        return DEFAULT_SPEND_CAP_USD
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"ROADSCOUT_SPEND_CAP_USD must be a number, got {raw!r}") from None
    if value < 0:
        raise ValueError("ROADSCOUT_SPEND_CAP_USD must not be negative")
    return value


def cost_log_path() -> Path | None:
    """JSONL cost log for live calls. Set ROADSCOUT_COST_LOG; "off" disables the file."""
    raw = (os.environ.get("ROADSCOUT_COST_LOG") or "").strip()
    if raw.lower() in ("off", "none", "false", "0"):
        return None
    return Path(raw) if raw else ROOT / "data" / "cost_log.jsonl"


def cost_per_call_usd(operation: str) -> float:
    """Estimated USD per live call (operation: "detect" or "verify").

    Set ROADSCOUT_COST_PER_DETECT_USD / ROADSCOUT_COST_PER_VERIFY_USD. Default 0:
    real pricing is unknown until the task 26 spike measures it.
    """
    raw = os.environ.get(f"ROADSCOUT_COST_PER_{operation.upper()}_USD")
    if raw is None or not raw.strip():
        return 0.0
    value = float(raw)
    if value < 0:
        raise ValueError(f"ROADSCOUT_COST_PER_{operation.upper()}_USD must not be negative")
    return value
