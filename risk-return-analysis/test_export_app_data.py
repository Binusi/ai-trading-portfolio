"""Smoke test for the JSON export module.

Builds synthetic profile_results that mimic what main.py produces, runs
export_app_data into a temp dir, and verifies the resulting JSON files
are well-formed and contain the keys the app will look for.

Does NOT exercise the ML training pipeline.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.export_app_data import export_app_data
from src.profile_strategy import build_profile_allocation_timeline
from src.profiles import DEFAULT_EQUITY_UNIVERSE, get_profile, get_required_tickers
from src.simulation import compute_simulation_metrics, run_simulation


def _synthetic_price_data(start: str, end: str, tickers: list[str]) -> pd.DataFrame:
    """Build a fake yfinance-style multi-index DataFrame with deterministic prices."""
    dates = pd.bdate_range(start, end)
    rng = np.random.default_rng(42)

    # Mean daily return + volatility per ticker (very rough)
    profile_per: dict[str, tuple[float, float]] = {
        "AAPL": (0.0006, 0.018),
        "MSFT": (0.0006, 0.017),
        "NVDA": (0.0010, 0.030),
        "SPY":  (0.0004, 0.010),
        "TLT":  (0.0001, 0.008),
        "GLD":  (0.0003, 0.009),
        "EFA":  (0.0003, 0.011),
    }

    closes = {}
    for t in tickers:
        mu, sigma = profile_per.get(t, (0.0003, 0.012))
        rets = rng.normal(mu, sigma, len(dates))
        closes[t] = 100.0 * np.exp(np.cumsum(rets))

    arrays = []
    cols = []
    for field in ["Close", "Open", "High", "Low", "Volume"]:
        for t in tickers:
            cols.append((field, t))
            if field == "Volume":
                arrays.append(np.full(len(dates), 1_000_000.0))
            elif field == "Close":
                arrays.append(closes[t])
            else:
                arrays.append(closes[t] * (1 + rng.normal(0, 0.005, len(dates))))

    df = pd.DataFrame(np.column_stack(arrays), index=dates, columns=pd.MultiIndex.from_tuples(cols, names=["Price", "Ticker"]))
    df.index.name = "Date"
    return df


def _synthetic_predictions(start: str, end: str, equity_tickers: list[str]) -> pd.DataFrame:
    """Generate synthetic ML scores per equity ticker per quarter."""
    rng = np.random.default_rng(7)
    quarter_starts = pd.date_range(start, end, freq="QS")
    rows = []
    for q in quarter_starts:
        for t in equity_tickers:
            rows.append({"Date": q - pd.Timedelta(days=2), "Ticker": t, "score": rng.normal(0, 0.02)})
    return pd.DataFrame(rows)


def main() -> None:
    start, end = "2024-01-01", "2024-12-31"
    equity_universe = list(DEFAULT_EQUITY_UNIVERSE)
    sim_tickers = get_required_tickers(equity_universe)

    price_data = _synthetic_price_data(start, end, sim_tickers + ["^GSPC"])
    predictions = _synthetic_predictions(start, end, equity_universe)

    # Build profile_results structure exactly like main.py
    from src.utils import get_trading_days
    from src.profile_strategy import get_quarterly_rebalance_dates

    trading_days = get_trading_days(start, end, price_data)
    quarterly = get_quarterly_rebalance_dates(trading_days)

    profile_results = {}
    for profile_key in ["conservative", "balanced", "aggressive"]:
        profile = get_profile(profile_key)
        for use_tilt in [False, True]:
            allocation, rationales = build_profile_allocation_timeline(
                profile=profile,
                trading_days=trading_days,
                equity_tickers=equity_universe,
                prediction_df=predictions if use_tilt else None,
                use_tilt=use_tilt,
            )
            daily_log, trade_log = run_simulation(
                allocation_timeline=allocation,
                price_data=price_data,
                tickers=sim_tickers,
                start_date=start,
                end_date=end,
                initial_capital=1000.0,
                transaction_cost_bps=10.0,
                rebalance_dates=quarterly,
            )
            metrics = compute_simulation_metrics(daily_log, initial_capital=1000.0)
            if not trade_log.empty:
                metrics["total_trades"] = len(trade_log)
                metrics["total_transaction_costs"] = float(trade_log["transaction_cost"].sum())
            profile_results[(profile_key, use_tilt)] = {
                "profile": profile,
                "use_tilt": use_tilt,
                "allocation": allocation,
                "daily_log": daily_log,
                "trade_log": trade_log,
                "rationale_log": rationales,
                "metrics": metrics,
            }

    tmp = Path(tempfile.mkdtemp(prefix="export_test_"))
    print(f"Writing JSON to {tmp}")

    written = export_app_data(
        profile_results=profile_results,
        price_data=price_data,
        ml_model_info={"model_name": "test_model", "target": "test_target", "validation_sharpe": 1.0},
        simulation_config={
            "start_date": start,
            "end_date": end,
            "initial_capital": 1000.0,
            "transaction_cost_bps": 10.0,
            "rebalance_cadence": "quarterly",
            "tilt_cap_pct": 5.0,
        },
        output_dir=tmp,
    )

    print(f"\nWrote {len(written)} files:")
    for p in written:
        print(f"  - {p.name}  ({p.stat().st_size / 1024:.1f} KB)")

    # ---- Validate summary
    summary = json.loads((tmp / "summary.json").read_text())
    assert summary["schema_version"] == 1
    assert "disclaimer" in summary
    assert len(summary["profiles"]) == 3
    assert any(p["default"] for p in summary["profiles"]), "exactly one default profile expected"
    assert summary["default_view"]["profile_key"] == "balanced"
    assert summary["default_view"]["tilt_enabled"] is False
    assert len(summary["comparison"]) == 6
    assert all("data_file" in c for c in summary["comparison"])
    assert "benchmark" in summary and summary["benchmark"]
    assert len(summary["benchmark"]["daily"]) > 0
    print(f"\nsummary.json validated.")
    print(f"  Profiles: {[p['key'] for p in summary['profiles']]}")
    print(f"  Default view: {summary['default_view']}")
    print(f"  Comparison rows: {len(summary['comparison'])}")
    print(f"  Benchmark: {summary['benchmark']['name']}, "
          f"final={summary['benchmark']['metrics']['final_portfolio_value']:.2f}")

    # ---- Validate one profile file
    sample = json.loads((tmp / "balanced_no_tilt.json").read_text())
    assert sample["profile"]["key"] == "balanced"
    assert sample["tilt_enabled"] is False
    assert len(sample["daily"]) > 200, f"daily series too short: {len(sample['daily'])}"
    assert len(sample["rebalance_events"]) >= 4, "expected at least 4 quarterly rebalances in 2024"
    first_event = sample["rebalance_events"][0]
    assert "rationale_text" in first_event
    assert "trades" in first_event
    assert "asset_class_weights" in first_event
    assert sum(first_event["asset_class_weights"].values()) == \
        sum(sample["profile"]["targets"].values()), "asset class weights should match profile targets"
    print(f"\nbalanced_no_tilt.json validated.")
    print(f"  Daily rows: {len(sample['daily'])}")
    print(f"  Rebalance events: {len(sample['rebalance_events'])}")
    print(f"  First rationale (truncated): {first_event['rationale_text'][:140]}...")
    print(f"  First event trades: {first_event['trade_count']}, "
          f"cost ${first_event['transaction_cost']:.2f}")

    # ---- Validate tilt file has tilts populated
    sample_tilt = json.loads((tmp / "balanced_tilt.json").read_text())
    assert sample_tilt["tilt_enabled"] is True
    assert any(ev["tilt_applied"] for ev in sample_tilt["rebalance_events"]), "expected at least one tilted event"
    tilted_event = next(ev for ev in sample_tilt["rebalance_events"] if ev["tilt_applied"])
    assert len(tilted_event["tilts"]) == len(DEFAULT_EQUITY_UNIVERSE)
    print(f"\nbalanced_tilt.json validated. Sample tilts:")
    for t in tilted_event["tilts"]:
        print(f"  {t['ticker']}: tilt={t['tilt']*100:+.2f}% direction={t['direction']}")

    print(f"\nKeeping output for inspection: {tmp}")


if __name__ == "__main__":
    main()
