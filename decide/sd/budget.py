"""Session time budget.  `ok(next_cost)` answers: can I still afford a step that costs X seconds?"""
from __future__ import annotations

import time
from typing import Callable, Optional


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Wall-clock budget for one notebook session (Kaggle: 12 h max, we plan for 11.5 h).

    `start` is a clock value (default: now).  Pass the kernel start time to account for work
    done before the object was created.  `clock` is injectable for tests.
    """

    def __init__(self, session_hours: float = 11.5, reserve_minutes: float = 20.0,
                 clock: Callable[[], float] = time.monotonic, start: Optional[float] = None) -> None:
        if session_hours <= 0 or reserve_minutes < 0:
            raise ValueError("invalid budget")
        self.total = float(session_hours) * 3600.0
        self.reserve = float(reserve_minutes) * 60.0
        if self.reserve >= self.total:
            raise ValueError("reserve must be smaller than the session")
        self._clock = clock
        self.start = clock() if start is None else float(start)

    @classmethod
    def from_config(cls, cfg, **kw) -> "Budget":
        return cls(cfg["budget"]["session_hours"], cfg["budget"]["reserve_minutes"], **kw)

    @classmethod
    def from_process_start(cls, session_hours: float = 11.5, reserve_minutes: float = 20.0) -> "Budget":  # pragma: no cover
        """Count from the start of this process (Kaggle kernel = session) when psutil is available."""
        try:
            import psutil

            age = time.time() - psutil.Process().create_time()
            return cls(session_hours, reserve_minutes, start=time.monotonic() - max(0.0, age))
        except Exception:
            return cls(session_hours, reserve_minutes)

    def elapsed(self) -> float:
        return self._clock() - self.start

    def remaining(self) -> float:
        """Seconds left before the hard session limit."""
        return self.total - self.elapsed()

    def available(self) -> float:
        """Seconds that can still be spent on work (remaining minus the reserve for syncing/shutdown)."""
        return max(0.0, self.remaining() - self.reserve)

    def ok(self, next_cost: float = 0.0) -> bool:
        return self.available() >= float(next_cost)

    def guard(self, next_cost: float, what: str = "step") -> None:
        if not self.ok(next_cost):
            raise BudgetExceeded(
                f"{what}: needs ~{next_cost / 60:.1f} min but only {self.available() / 60:.1f} min are left "
                f"(reserve {self.reserve / 60:.0f} min)"
            )

    def fraction(self, f: float) -> float:
        """f * available seconds (for sizing a loop to a share of what is left)."""
        return max(0.0, f) * self.available()

    def __repr__(self) -> str:
        return f"Budget(elapsed={self.elapsed() / 3600:.2f}h, available={self.available() / 3600:.2f}h)"
