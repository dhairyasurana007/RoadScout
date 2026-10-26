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
