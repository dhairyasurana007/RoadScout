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
