"""Path setup for the task 26d hidden tests (standalone, or copied into tests/unit)."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, Path(r"C:\Users\dhair\RoadScout\tests\unit"), _HERE.parent / "task-26", _HERE.parent / "task-15"):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.append(str(_p))
