"""Periodic-deposit scheduling for the trade simulator.

A `DepositSchedule` describes a recurring contribution: a fixed dollar amount
paid every N months on a chosen day-of-month (1, 15, or end-of-month).

`get_deposit_dates()` maps that schedule onto the actual trading calendar:
- For day=1 or day=15: snap forward to the next available trading day in the
  same month or later (handles weekends and market holidays).
- For day="EOM": snap backward to the last trading day of the same month
  (matches how payroll/brokerages settle month-end transfers and avoids
  spilling December into January).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import pandas as pd


DayOfMonth = Union[int, str]  # 1, 15, or "EOM"


@dataclass(frozen=True)
class DepositSchedule:
    amount: float
    period_months: int
    day_of_month: DayOfMonth

    def __post_init__(self) -> None:
        if self.amount < 0:
            raise ValueError("amount must be non-negative")
        if self.period_months < 1:
            raise ValueError("period_months must be >= 1")
        if isinstance(self.day_of_month, str):
            if self.day_of_month.upper() != "EOM":
                raise ValueError("day_of_month string must be 'EOM'")
        elif isinstance(self.day_of_month, int):
            if self.day_of_month < 1 or self.day_of_month > 28:
                raise ValueError("day_of_month int must be in 1..28")
        else:
            raise TypeError("day_of_month must be int or 'EOM'")


def _is_eom(day: DayOfMonth) -> bool:
    return isinstance(day, str) and day.upper() == "EOM"


def get_deposit_dates(
    trading_days: list[pd.Timestamp],
    schedule: DepositSchedule,
) -> list[pd.Timestamp]:
    """Return the trading-day dates on which deposits should be applied."""
    if not trading_days or schedule.amount <= 0:
        return []

    trading_index = pd.DatetimeIndex(sorted(trading_days))
    start = trading_index[0]
    end = trading_index[-1]

    # Walk calendar months from the start month to the end month, stepping
    # `period_months`. The first deposit lands in the start month.
    months: list[pd.Period] = []
    cur = pd.Period(year=start.year, month=start.month, freq="M")
    last = pd.Period(year=end.year, month=end.month, freq="M")
    while cur <= last:
        months.append(cur)
        cur = cur + schedule.period_months

    out: list[pd.Timestamp] = []
    for m in months:
        if _is_eom(schedule.day_of_month):
            month_mask = (trading_index.year == m.year) & (trading_index.month == m.month)
            month_days = trading_index[month_mask]
            if len(month_days) == 0:
                continue
            target = month_days[-1]  # last trading day of the month
        else:
            target_calendar = pd.Timestamp(year=m.year, month=m.month, day=int(schedule.day_of_month))
            after = trading_index[trading_index >= target_calendar]
            if len(after) == 0:
                continue
            target = after[0]  # first trading day on or after the calendar target

        if target < start or target > end:
            continue
        out.append(target)

    # De-dupe (defensive — period stepping shouldn't collide, but EOM snapping
    # could in degenerate calendars) and keep sorted.
    return sorted(set(out))
