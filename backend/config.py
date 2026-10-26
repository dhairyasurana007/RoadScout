"""Settings read from the environment (a git-ignored .env is loaded if present)."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")


def db_path() -> Path:
    """SQLite file location. Override with ROADSCOUT_DB."""
    return Path(os.environ.get("ROADSCOUT_DB", ROOT / "data" / "roadscout.sqlite"))


def meta_api_key() -> str | None:
    """The Meta API key, or None when unset. Never log or print this value."""
    return os.environ.get("META_API_KEY") or None


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
