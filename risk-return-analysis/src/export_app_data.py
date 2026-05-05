"""Export simulation results as JSON files for the Expo app to consume.

The app does not run the Python pipeline live. Instead, after `main.py`
finishes the ML training and the six profile simulations, this module writes
a small set of JSON files into `app/assets/data/`. The app bundles those
files at build time and renders them.

All dollar values are stored against a $1,000 starting capital with NO
periodic deposits — i.e. the canonical lump-sum baseline. The app scales
the lump-sum case linearly by `(user_capital / 1000)`. For deposit mode,
the app reconstructs the user's dollar trajectory client-side by replaying
the exported `daily_return` series against the user's initial capital and
deposit schedule. This works because `daily_return` is cashflow-adjusted
in the simulation engine; in lump-sum exports it equals the strategy's
per-day return assuming 100% deployment.

File layout produced:

    summary.json                  — top-level index, profile descriptions,
                                    cross-profile comparison, SPY benchmark
    conservative_no_tilt.json     ┐
    conservative_tilt.json        │
    balanced_no_tilt.json         │  per-(profile, tilt) detail files:
    balanced_tilt.json            │  daily values + rebalance events with
    aggressive_no_tilt.json       │  trades + rationale + allocation snapshots
    aggressive_tilt.json          ┘
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.features import ASSET_GROUP_MAP
from src.profiles import FRIENDLY_CLASS_NAMES, RISK_PROFILES, RiskProfile

EXPORT_INITIAL_CAPITAL = 1000.0
DISCLAIMER = (
    "Simulation only — not investment advice. Past performance does not "
    "predict future results. Returns shown are based on historical price "
    "data through 2024-2025 (a strong bull market for US equities) and may "
    "not generalize to other periods."
)


def _iso_date(d: Any) -> str:
    return pd.Timestamp(d).strftime("%Y-%m-%d")


def _safe_number(v: Any) -> Any:
    """Convert pandas/numpy numbers to native Python; replace NaN/Inf with None."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    return v


def _clean(obj: Any) -> Any:
    """Recursively make a value JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (pd.Timestamp, np.datetime64)):
        return _iso_date(obj)
    return _safe_number(obj)


def _profile_meta(profile: RiskProfile, default: bool = False) -> dict:
    return {
        "key": profile.key,
        "name": profile.name,
        "description": profile.description,
        "targets": dict(profile.targets),
        "default": default,
    }


def _asset_class_label_map() -> dict[str, str]:
    return dict(FRIENDLY_CLASS_NAMES)


def _used_asset_classes() -> set[str]:
    """Asset classes actually referenced by any profile target."""
    used: set[str] = set()
    for profile in RISK_PROFILES.values():
        for cls, weight in profile.targets.items():
            if weight > 0:
                used.add(cls)
    return used


def _build_daily_series(daily_log: pd.DataFrame) -> list[dict]:
    asset_classes = sorted(_used_asset_classes())
    rows: list[dict] = []
    for _, r in daily_log.iterrows():
        asset_class_weights = {
            cls: float(r.get(f"{cls}_weight", 0.0) or 0.0)
            for cls in asset_classes
            if f"{cls}_weight" in daily_log.columns
        }
        total_value = _safe_number(r["total_value"])
        rows.append(
            {
                "date": _iso_date(r["Date"]),
                "value": total_value,
                # Intent-revealing alias used by the app's deposit-mode
                # reconstruction code. Always equals `value` because the
                # canonical export runs at $1,000 with no deposits.
                "value_per_1000": total_value,
                "daily_return": _safe_number(r.get("daily_return", 0.0) or 0.0),
                "asset_class_weights": _clean(asset_class_weights),
            }
        )
    return rows


def _build_rebalance_events(
    rationale_log: list[dict],
    trade_log: pd.DataFrame,
    daily_log: pd.DataFrame,
) -> list[dict]:
    daily_by_date: dict[str, dict] = {
        _iso_date(r["Date"]): r.to_dict() for _, r in daily_log.iterrows()
    }
    sorted_dates = sorted(daily_by_date.keys())

    trades_by_date: dict[str, list[dict]] = {}
    if not trade_log.empty:
        for _, t in trade_log.iterrows():
            key = _iso_date(t["Date"])
            trades_by_date.setdefault(key, []).append(
                {
                    "ticker": t["ticker"],
                    "action": t["action"],
                    "shares": _safe_number(t["shares"]),
                    "price": _safe_number(t["price"]),
                    "dollar_value": _safe_number(t["dollar_value"]),
                    "transaction_cost": _safe_number(t["transaction_cost"]),
                }
            )

    events: list[dict] = []
    for entry in rationale_log:
        date_str = _iso_date(entry["date"])
        trades = trades_by_date.get(date_str, [])
        transaction_cost = sum(t.get("transaction_cost", 0.0) or 0.0 for t in trades)

        # Snap to the nearest trading day on or after the rebalance date.
        # (Q1 may begin on a market holiday.)
        snap_date = next((d for d in sorted_dates if d >= date_str), None)
        post_value = (
            _safe_number(daily_by_date[snap_date]["total_value"])
            if snap_date else None
        )

        # Pre-rebalance value approximated as the previous trading day's close.
        pre_value: Optional[float] = None
        if snap_date and snap_date != sorted_dates[0]:
            idx = sorted_dates.index(snap_date)
            pre_value = _safe_number(
                daily_by_date[sorted_dates[idx - 1]]["total_value"]
            )

        # Asset-class roll-up of the post-rebalance target weights
        target_weights = entry.get("target_weights", {}) or {}
        asset_class_weights: dict[str, float] = {}
        for ticker, w in target_weights.items():
            cls = ASSET_GROUP_MAP.get(ticker, "other")
            asset_class_weights[cls] = asset_class_weights.get(cls, 0.0) + float(w)

        events.append(
            _clean(
                {
                    "date": date_str,
                    "rationale_text": entry.get("rationale_text", ""),
                    "tilt_applied": bool(entry.get("tilt_applied", False)),
                    "target_weights": target_weights,
                    "asset_class_weights": asset_class_weights,
                    "tilts": entry.get("tilts", []),
                    "trades": trades,
                    "trade_count": len(trades),
                    "transaction_cost": transaction_cost,
                    "portfolio_value_before_rebalance": pre_value,
                    "portfolio_value_after_rebalance": post_value,
                }
            )
        )
    return events


def _build_spy_benchmark(
    price_data: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_capital: float = EXPORT_INITIAL_CAPITAL,
) -> dict:
    try:
        spy = price_data["Close"]["SPY"].dropna()
    except (KeyError, TypeError):
        return {}

    spy = spy[(spy.index >= pd.Timestamp(start_date)) & (spy.index <= pd.Timestamp(end_date))]
    if spy.empty:
        return {}

    base_price = float(spy.iloc[0])
    shares = initial_capital / base_price
    daily_values = spy * shares

    daily_returns = daily_values.pct_change().fillna(0.0)
    final_value = float(daily_values.iloc[-1])
    total_return = final_value / initial_capital - 1.0
    n_days = len(daily_values)
    years = n_days / 252.0
    annualized = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    equity_curve = (1 + daily_returns).cumprod()
    drawdown = equity_curve / equity_curve.cummax() - 1.0
    max_dd = float(drawdown.min())

    daily_series = [
        {
            "date": _iso_date(d),
            "value": _safe_number(v),
        }
        for d, v in daily_values.items()
    ]

    return {
        "name": "SPY buy-and-hold",
        "description": (
            "Bought $1,000 of SPY on the simulation start date and held it "
            "through the end. Useful as a reality check: did the active "
            "strategy beat doing nothing?"
        ),
        "metrics": _clean(
            {
                "initial_capital": initial_capital,
                "final_portfolio_value": final_value,
                "total_return": total_return,
                "annualized_return": annualized,
                "max_drawdown": max_dd,
            }
        ),
        "daily": daily_series,
    }


def _profile_filename(profile_key: str, use_tilt: bool) -> str:
    return f"{profile_key}_{'tilt' if use_tilt else 'no_tilt'}.json"


def _write_profile_file(
    profile: RiskProfile,
    use_tilt: bool,
    daily_log: pd.DataFrame,
    trade_log: pd.DataFrame,
    rationale_log: list[dict],
    metrics: dict,
    output_path: Path,
) -> None:
    payload = {
        "profile": _profile_meta(profile),
        "tilt_enabled": bool(use_tilt),
        "metrics": _clean(metrics),
        "daily": _build_daily_series(daily_log),
        "rebalance_events": _build_rebalance_events(rationale_log, trade_log, daily_log),
    }
    output_path.write_text(json.dumps(payload, indent=2))


def export_app_data(
    profile_results: dict,
    price_data: pd.DataFrame,
    ml_model_info: dict,
    simulation_config: dict,
    output_dir: Path | str,
    default_profile_key: str = "balanced",
    default_use_tilt: bool = False,
) -> list[Path]:
    """Write summary + per-profile JSON files. Returns the list of written paths."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # --- Per-profile files
    comparison: list[dict] = []
    for (profile_key, use_tilt), result in profile_results.items():
        profile: RiskProfile = result["profile"]
        filename = _profile_filename(profile_key, use_tilt)
        path = out / filename
        _write_profile_file(
            profile=profile,
            use_tilt=use_tilt,
            daily_log=result["daily_log"],
            trade_log=result["trade_log"],
            rationale_log=result["rationale_log"],
            metrics=result["metrics"],
            output_path=path,
        )
        written.append(path)

        m = result["metrics"]
        comparison.append(
            _clean(
                {
                    "profile_key": profile_key,
                    "profile_name": profile.name,
                    "tilt_enabled": bool(use_tilt),
                    "data_file": filename,
                    "final_portfolio_value": m.get("final_portfolio_value"),
                    "total_return": m.get("total_return"),
                    "annualized_return": m.get("annualized_return"),
                    "sharpe_ratio": m.get("sharpe_ratio"),
                    "sortino_ratio": m.get("sortino_ratio"),
                    "max_drawdown": m.get("max_drawdown"),
                    "max_drawdown_date": m.get("max_drawdown_date"),
                    "total_trades": m.get("total_trades"),
                    "total_transaction_costs": m.get("total_transaction_costs"),
                }
            )
        )

    # --- Summary file
    profiles_meta = [
        _profile_meta(p, default=(p.key == default_profile_key))
        for p in RISK_PROFILES.values()
    ]
    benchmark = _build_spy_benchmark(
        price_data=price_data,
        start_date=simulation_config["start_date"],
        end_date=simulation_config["end_date"],
        initial_capital=EXPORT_INITIAL_CAPITAL,
    )
    summary = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "disclaimer": DISCLAIMER,
        "simulation": _clean(simulation_config),
        "ml_model": _clean(ml_model_info),
        "profiles": profiles_meta,
        "asset_class_labels": _asset_class_label_map(),
        "comparison": comparison,
        "benchmark": _clean(benchmark),
        "default_view": {
            "profile_key": default_profile_key,
            "tilt_enabled": default_use_tilt,
        },
    }
    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    written.append(summary_path)

    return written
