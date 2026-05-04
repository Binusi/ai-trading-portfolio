"""Quick smoke test for the new profile-based strategy module.

Run from the project root:

    python test_profile_strategy.py

This does NOT touch the ML pipeline — it only exercises the allocation logic
with synthetic ML scores so we can verify the math, the rationales, and the
allocation timeline structure before the full integration.
"""

from __future__ import annotations

import pandas as pd

from src.profile_strategy import (
    build_profile_allocation_timeline,
    compute_equity_tilts,
    get_quarterly_rebalance_dates,
)
from src.profiles import (
    DEFAULT_EQUITY_UNIVERSE,
    INDEX_PROXIES,
    RISK_PROFILES,
    get_profile,
)


def test_quarterly_dates() -> None:
    days = pd.bdate_range("2024-01-01", "2025-12-31").tolist()
    rb = get_quarterly_rebalance_dates(days)
    print(f"  Quarterly rebalance dates over 2024–2025: {len(rb)} (expected 8)")
    print(f"  Dates: {[d.strftime('%Y-%m-%d') for d in rb]}")
    assert len(rb) == 8, f"expected 8 quarters, got {len(rb)}"


def test_tilts_zero_sum_and_capped() -> None:
    tickers = ["AAPL", "MSFT", "NVDA"]
    scores = {"AAPL": -0.02, "MSFT": 0.005, "NVDA": 0.04}
    tilts = compute_equity_tilts(
        tickers, base_equity_weight=0.65, scores=scores, tilt_cap=0.05
    )
    total = sum(t.final_weight for t in tilts)
    print(f"  Equity sleeve total after tilt: {total:.4f} (expected 0.6500)")
    print(f"  Per-ticker breakdown:")
    for t in tilts:
        print(
            f"    {t.ticker:<5s} score={t.score:+.4f} base={t.base_weight:.4f} "
            f"tilt={t.tilt:+.4f} final={t.final_weight:.4f} ({t.direction})"
        )
    assert abs(total - 0.65) < 1e-6, "equity sleeve weight drifted"
    assert all(abs(t.tilt) <= 0.05 + 1e-9 for t in tilts), "tilt exceeded cap"
    nvda = next(t for t in tilts if t.ticker == "NVDA")
    aapl = next(t for t in tilts if t.ticker == "AAPL")
    assert nvda.tilt > 0, "NVDA (highest score) should be overweight"
    assert aapl.tilt < 0, "AAPL (lowest score) should be underweight"


def test_full_timeline_with_synthetic_scores() -> None:
    days = pd.bdate_range("2024-01-01", "2025-12-31").tolist()
    quarters = get_quarterly_rebalance_dates(days)

    rows = []
    for i, q in enumerate(quarters):
        signal_date = q - pd.Timedelta(days=3)
        rows.append({"Date": signal_date, "Ticker": "AAPL", "score": -0.01 + 0.005 * i})
        rows.append({"Date": signal_date, "Ticker": "MSFT", "score": 0.0})
        rows.append({"Date": signal_date, "Ticker": "NVDA", "score": 0.02 - 0.003 * i})
    prediction_df = pd.DataFrame(rows)

    profile = get_profile("aggressive")
    alloc, rationales = build_profile_allocation_timeline(
        profile=profile,
        trading_days=days,
        equity_tickers=DEFAULT_EQUITY_UNIVERSE,
        prediction_df=prediction_df,
        tilt_cap=0.05,
        use_tilt=True,
    )

    print(f"  Allocation rows: {len(alloc)} (expected {len(quarters)} quarters * 7 tickers = {len(quarters) * 7})")
    print(f"  Distinct tickers in allocation: {sorted(alloc['Ticker'].unique())}")
    expected_tickers = set(DEFAULT_EQUITY_UNIVERSE) | set(INDEX_PROXIES.values())
    actual_tickers = set(alloc["Ticker"].unique())
    assert actual_tickers == expected_tickers, (
        f"Expected {expected_tickers}, got {actual_tickers}"
    )

    # Check that weights at each rebalance date sum to 1
    for date, sub in alloc.groupby("Date"):
        total = sub["ticker_weight"].sum()
        assert abs(total - 1.0) < 1e-6, f"weights on {date} sum to {total}"

    print(f"\n  First rebalance ({rationales[0]['date'].date()}):")
    print(f"  {rationales[0]['rationale_text']}")
    print(f"\n  Last rebalance ({rationales[-1]['date'].date()}):")
    print(f"  {rationales[-1]['rationale_text']}")


def test_no_tilt_path() -> None:
    days = pd.bdate_range("2024-01-01", "2025-12-31").tolist()
    profile = get_profile("conservative")
    alloc, rationales = build_profile_allocation_timeline(
        profile=profile,
        trading_days=days,
        equity_tickers=DEFAULT_EQUITY_UNIVERSE,
        prediction_df=None,
        use_tilt=False,
    )
    print(f"  Allocation rows (no-tilt conservative): {len(alloc)}")
    print(f"  Rationale (no tilt path):")
    print(f"  {rationales[0]['rationale_text']}")
    # Equity weight should be exactly profile.targets["us_equity"] / N per equity ticker
    first_date = alloc["Date"].min()
    first = alloc[alloc["Date"] == first_date]
    equity_rows = first[first["AssetGroup"] == "us_equity"]
    expected_per = profile.targets["us_equity"] / len(DEFAULT_EQUITY_UNIVERSE)
    for _, row in equity_rows.iterrows():
        assert abs(row["ticker_weight"] - expected_per) < 1e-6, (
            f"{row['Ticker']} weight {row['ticker_weight']} != expected {expected_per}"
        )


def test_profile_definitions_sum_to_one() -> None:
    for key, profile in RISK_PROFILES.items():
        total = sum(profile.targets.values())
        print(f"  {key}: targets sum to {total:.6f}")
        assert abs(total - 1.0) < 1e-6


if __name__ == "__main__":
    print("\n[1] Quarterly rebalance dates")
    test_quarterly_dates()
    print("\n[2] Profile definitions sum to 1.0")
    test_profile_definitions_sum_to_one()
    print("\n[3] Tilts are zero-sum and capped at ±5%")
    test_tilts_zero_sum_and_capped()
    print("\n[4] Full allocation timeline with synthetic ML scores")
    test_full_timeline_with_synthetic_scores()
    print("\n[5] No-tilt code path (use_tilt=False)")
    test_no_tilt_path()
    print("\nAll smoke tests passed.")
