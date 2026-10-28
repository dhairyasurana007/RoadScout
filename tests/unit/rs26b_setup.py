"""Path setup for the task 26b hidden tests (works standalone and when copied into tests/unit)."""
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE.parent / "task-26", _HERE.parent / "task-09", _HERE.parent / "task-10",
           Path(r"C:\Users\dhair\RoadScout\tests\unit")):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.append(str(_p))
