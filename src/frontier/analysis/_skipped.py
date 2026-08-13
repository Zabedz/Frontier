"""One reason a variant produced no row in a derived table."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Skipped:
    """A variant left out of a derived table, and why."""

    variant: str
    task: str
    reason: str
