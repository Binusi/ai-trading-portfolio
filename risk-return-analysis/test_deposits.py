"""Smoke test for periodic deposits in the simulation engine.

Run from the project root:

    python test_deposits.py

Covers:
  - get_deposit_dates snap-forward for day-of-month=1 (handles Jan-1 holiday)
  - get_deposit_dates snap-backward for EOM (handles Nov-30-2024 = Saturday)
  - get_deposit_dates spacing for every-3-months
  - cash injection on deposit day in run_simulation
  - cashflow-adjusted daily return excludes the deposit
  - lump-sum path is unchanged when deposit_schedule=None
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.deposits import DepositSchedule, get_deposit_dates
from src.simulation import compute_simulation_metrics, run_simulation


def _flat_price_data(start: str, end: str, tickers: list[str]) -> pd.DataFrame:
    """Multi-index price data, all tickers held constant at $100. Skips Jan 1 2024
    so we can verify the day-1 snap-forward behavior at the year boundary."""
    dates = pd.bdate_range(start, end)
    dates = dates[dates != pd.Timestamp("2024-01-01")]  # simulate market holiday
    n = len(dates)

    arrays = []
    cols = []
    for field in ["Close", "Open", "High", "Low", "Volume"]:
        for t in tickers:
            cols.append((field, t))
            if field == "Volume":
                arrays.append(np.full(n, 1_000_000.0))
            else:
                arrays.append(np.full(n, 100.0))

    df = pd.DataFrame(
        np.column_stack(arrays),
        index=dates,
        columns=pd.MultiIndex.from_tuples(cols, names=["Price", "Ticker"]),
    )
    df.index.name = "Date"
    return df


def _all_cash_allocation(start_date: str, tickers: list[str]) -> pd.DataFrame:
    """An allocation timeline that targets 0% in every ticker (all cash).
    Lets us verify deposits without strategy-driven trades muddying the totals.
    """
    rows = [
        {"Date": pd.Timestamp(start_date), "Ticker": t, "ticker_weight": 0.0}
        for t in tickers
    ]
    return pd.DataFrame(rows)


def test_deposit_dates_first_of_month_snaps_forward() -> None:
    days = pd.bdate_range("2024-01-01", "2024-03-31").tolist()
    days = [d for d in days if d != pd.Timestamp("2024-01-01")]
    schedule = DepositSchedule(amount=100.0, period_months=1, day_of_month=1)
    dates = get_deposit_dates(days, schedule)
    print(f"  Deposit dates (monthly, day=1): {[d.strftime('%Y-%m-%d') for d in dates]}")
    assert pd.Timestamp("2024-01-02") in dates, "Jan deposit should snap from Jan-1 holiday to Jan-2"
    assert pd.Timestamp("2024-02-01") in dates, "Feb-1 2024 is a Thursday, no snap needed"
    assert pd.Timestamp("2024-03-01") in dates, "Mar-1 2024 is a Friday, no snap needed"
    assert len(dates) == 3, f"expected 3 monthly deposits, got {len(dates)}"


def test_deposit_dates_eom_snaps_backward() -> None:
    days = pd.bdate_range("2024-11-01", "2024-12-31").tolist()
    schedule = DepositSchedule(amount=100.0, period_months=1, day_of_month="EOM")
    dates = get_deposit_dates(days, schedule)
    print(f"  Deposit dates (monthly, EOM): {[d.strftime('%Y-%m-%d') for d in dates]}")
    assert pd.Timestamp("2024-11-29") in dates, (
        "Nov 30 2024 is a Saturday; EOM should snap back to Nov 29 (Fri)"
    )
    assert pd.Timestamp("2024-12-31") in dates, "Dec 31 2024 is a Tuesday, no snap needed"


def test_deposit_dates_every_3_months() -> None:
    days = pd.bdate_range("2024-01-02", "2024-12-31").tolist()
    schedule = DepositSchedule(amount=250.0, period_months=3, day_of_month=15)
    dates = get_deposit_dates(days, schedule)
    print(f"  Deposit dates (quarterly, day=15): {[d.strftime('%Y-%m-%d') for d in dates]}")
    months = sorted({d.month for d in dates})
    assert months == [1, 4, 7, 10], f"expected months [1,4,7,10], got {months}"
    assert len(dates) == 4, f"expected 4 quarterly deposits, got {len(dates)}"


def test_cash_injected_on_deposit_day_and_return_is_cashflow_adjusted() -> None:
    """Run a flat-price 3-month simulation with all-cash allocation. The deposit
    should add exactly $100 to cash on the snapped Jan deposit day, and the
    daily return on that day must be 0.0 (not +10%) because prices are flat."""
    tickers = ["AAPL", "MSFT"]
    price_data = _flat_price_data("2024-01-01", "2024-03-31", tickers)

    allocation = _all_cash_allocation("2024-01-02", tickers)
    schedule = DepositSchedule(amount=100.0, period_months=1, day_of_month=1)

    daily_log, _ = run_simulation(
        allocation_timeline=allocation,
        price_data=price_data,
        tickers=tickers,
        start_date="2024-01-02",
        end_date="2024-03-31",
        initial_capital=1000.0,
        rebalance_dates=[],  # never rebalance — keep everything in cash
        deposit_schedule=schedule,
    )

    deposits_only = daily_log[daily_log["deposit"] > 0]
    print(f"  Deposit rows: {len(deposits_only)} (expected 3)")
    assert len(deposits_only) == 3, f"expected 3 deposit days, got {len(deposits_only)}"

    # Cash should monotonically increase by $100 on each deposit day; everything
    # is in cash because allocation targets are zero.
    assert daily_log["total_value"].iloc[-1] == 1300.0, (
        f"expected $1,300 final (1000 + 3 * 100), got {daily_log['total_value'].iloc[-1]}"
    )

    # Cashflow-adjusted daily returns must be zero on every day (flat prices,
    # deposit excluded from the numerator).
    max_abs_return = daily_log["daily_return"].abs().max()
    print(f"  Max |daily_return| over flat-price + deposits run: {max_abs_return:.6f}")
    assert max_abs_return < 1e-9, (
        f"flat prices + deposits should give zero daily return; max |r| = {max_abs_return}"
    )

    # contributions_to_date final should be $1,300
    final_contrib = daily_log["contributions_to_date"].iloc[-1]
    assert final_contrib == 1300.0, f"expected contributions=1300, got {final_contrib}"


def test_lump_sum_path_unchanged_when_deposit_schedule_none() -> None:
    """Calling run_simulation with deposit_schedule=None should match calling it
    with an explicit zero-amount schedule (and both should leave the deposit
    column at 0 throughout)."""
    tickers = ["AAPL", "MSFT"]
    price_data = _flat_price_data("2024-01-02", "2024-03-31", tickers)
    allocation = _all_cash_allocation("2024-01-02", tickers)

    log_none, _ = run_simulation(
        allocation_timeline=allocation,
        price_data=price_data,
        tickers=tickers,
        start_date="2024-01-02",
        end_date="2024-03-31",
        initial_capital=1000.0,
        rebalance_dates=[],
        deposit_schedule=None,
    )
    log_zero, _ = run_simulation(
        allocation_timeline=allocation,
        price_data=price_data,
        tickers=tickers,
        start_date="2024-01-02",
        end_date="2024-03-31",
        initial_capital=1000.0,
        rebalance_dates=[],
        deposit_schedule=DepositSchedule(amount=0.0, period_months=1, day_of_month=1),
    )

    # Total value series should match exactly.
    assert (log_none["total_value"].values == log_zero["total_value"].values).all(), (
        "lump-sum and zero-deposit runs should produce identical total_value series"
    )
    assert log_none["deposit"].sum() == 0.0
    assert log_zero["deposit"].sum() == 0.0

    metrics = compute_simulation_metrics(log_none, initial_capital=1000.0)
    print(f"  Lump-sum total_contributions: ${metrics['total_contributions']:.2f}")
    assert metrics["total_contributions"] == 1000.0, (
        f"lump-sum mode should report initial capital as total contributions"
    )
    assert metrics["money_in_minus_out"] == 0.0, (
        "flat prices + no deposits should give zero money_in_minus_out"
    )


if __name__ == "__main__":
    print("\n[1] get_deposit_dates: day-1 snap forward (Jan 1 holiday → Jan 2)")
    test_deposit_dates_first_of_month_snaps_forward()
    print("\n[2] get_deposit_dates: EOM snap backward (Nov 30 Sat → Nov 29 Fri)")
    test_deposit_dates_eom_snaps_backward()
    print("\n[3] get_deposit_dates: every 3 months on day 15")
    test_deposit_dates_every_3_months()
    print("\n[4] Cash injected + cashflow-adjusted return on deposit day")
    test_cash_injected_on_deposit_day_and_return_is_cashflow_adjusted()
    print("\n[5] Lump-sum path unchanged when deposit_schedule=None")
    test_lump_sum_path_unchanged_when_deposit_schedule_none()
    print("\nAll deposit smoke tests passed.")
