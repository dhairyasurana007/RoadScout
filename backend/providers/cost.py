"""Spend tracker: running total in USD, a hard cap and an append-only JSONL cost log.

``SpendTracker(cap_usd=None, log_path=None)``; a ``None`` cap uses
``config.spend_cap_usd()`` (default $20, override with ROADSCOUT_SPEND_CAP_USD).

- ``charge(cost_usd, *, model=None, operation=None, **extra)`` records spend, raising
  ``SpendCapExceeded`` (and changing nothing) if ``spent + cost`` would exceed the cap.
  Landing exactly on the cap is allowed. Negative costs raise ``ValueError``.
- ``check(estimated_cost)`` raises ``SpendCapExceeded`` if a call of that cost would not
  fit; it spends nothing. Call it before a live call.
- Each successful charge appends one JSON line to ``log_path`` (parent directories are
  created): ``timestamp`` (UTC ISO 8601), ``cost_usd``, ``model``, ``operation``,
  ``spent_usd`` (running total) plus any ``extra`` fields. Refused charges write
  nothing. The same entries are kept in memory in ``.entries``.
- If ``log_path`` already holds entries, ``spent`` starts from their total, so the cap
  survives restarts. Callers must never put a key in ``extra``.
- Thread-safe.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .. import config
from .errors import SpendCapExceeded

_EPS = 1e-9  # float tolerance so that landing exactly on the cap is allowed


class SpendTracker:
    def __init__(self, cap_usd: float | None = None, log_path: str | Path | None = None):
        self.cap_usd = float(config.spend_cap_usd() if cap_usd is None else cap_usd)
        if self.cap_usd < 0:
            raise ValueError("spend cap must not be negative")
        self.log_path = Path(log_path) if log_path is not None else None
        self._lock = threading.Lock()
        self._entries: list[dict[str, Any]] = []
        self._spent = 0.0
        if self.log_path is not None and self.log_path.exists():
            for line in self.log_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._entries.append(entry)
                    self._spent += float(entry.get("cost_usd", 0.0))

    @property
    def spent(self) -> float:
        with self._lock:
            return self._spent

    @property
    def remaining(self) -> float:
        with self._lock:
            return max(0.0, self.cap_usd - self._spent)

    @property
    def entries(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._entries)

    def _refuse(self, cost: float) -> SpendCapExceeded:
        return SpendCapExceeded(
            f"spend cap ${self.cap_usd:.2f} would be exceeded: spent ${self._spent:.2f}, "
            f"next call ${cost:.2f}. Raise ROADSCOUT_SPEND_CAP_USD to continue."
        )

    def check(self, estimated_cost: float = 0.0) -> None:
        if estimated_cost < 0:
            raise ValueError("estimated cost must not be negative")
        with self._lock:
            if self._spent + estimated_cost > self.cap_usd + _EPS:
                raise self._refuse(estimated_cost)

    def charge(
        self,
        cost_usd: float,
        *,
        model: str | None = None,
        operation: str | None = None,
        **extra: Any,
    ) -> float:
        """Record spend and return the new total."""
        if cost_usd < 0:
            raise ValueError("cost must not be negative")
        with self._lock:
            if self._spent + cost_usd > self.cap_usd + _EPS:
                raise self._refuse(cost_usd)
            self._spent += cost_usd
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "cost_usd": cost_usd,
                "model": model,
                "operation": operation,
                "spent_usd": self._spent,
                **extra,
            }
            self._entries.append(entry)
            if self.log_path is not None:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open("a", encoding="utf-8") as f:  # append-only
                    f.write(json.dumps(entry) + "\n")
            return self._spent
