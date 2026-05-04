from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.profiles import (
    FRIENDLY_CLASS_NAMES,
    INDEX_PROXIES,
    RiskProfile,
)


@dataclass
class TickerTilt:
    """Records the AI tilt applied to a single ticker in the equity sleeve."""

    ticker: str
    score: float
    base_weight: float
    tilt: float
    final_weight: float
    direction: str  # "overweight" | "underweight" | "neutral"


def get_quarterly_rebalance_dates(
    trading_days: list[pd.Timestamp],
) -> list[pd.Timestamp]:
    """Return the first available trading day of each calendar quarter."""
    if not trading_days:
        return []
    days = pd.DatetimeIndex(sorted(set(pd.to_datetime(d) for d in trading_days)))
    df = pd.DataFrame({"day": days})
    df["yq"] = df["day"].dt.to_period("Q")
    first_per_q = df.groupby("yq", as_index=False)["day"].first()
    return [pd.Timestamp(d) for d in first_per_q["day"].tolist()]


def compute_equity_tilts(
    equity_tickers: list[str],
    base_equity_weight: float,
    scores: dict[str, float],
    tilt_cap: float = 0.05,
) -> list[TickerTilt]:
    """Distribute the equity sleeve across names, with an optional capped ML tilt.

    Steps:
    1. Start at equal weight per name = base_equity_weight / N.
    2. Convert ML scores to z-scores, clip to [-1, 1], scale by tilt_cap.
    3. Force the tilts to sum to zero so the total equity sleeve weight is
       preserved (no drift away from the profile target).
    4. Floor at 0 (no shorting) and renormalize.
    """
    n = len(equity_tickers)
    if n == 0 or base_equity_weight <= 0:
        return []

    base_per = base_equity_weight / n
    score_arr = np.array([scores.get(t, np.nan) for t in equity_tickers], dtype=float)

    if np.all(np.isnan(score_arr)):
        return [
            TickerTilt(t, 0.0, base_per, 0.0, base_per, "neutral")
            for t in equity_tickers
        ]

    mean_known = float(np.nanmean(score_arr))
    score_arr = np.where(np.isnan(score_arr), mean_known, score_arr)

    std = float(np.std(score_arr))
    if std < 1e-9:
        z = np.zeros_like(score_arr)
    else:
        z = (score_arr - score_arr.mean()) / std

    tilts_arr = np.clip(z, -1.0, 1.0) * tilt_cap
    tilts_arr = tilts_arr - tilts_arr.mean()  # zero-sum so sleeve stays constant
    tilts_arr = np.clip(tilts_arr, -tilt_cap, tilt_cap)  # enforce hard cap post zero-sum

    results: list[TickerTilt] = []
    for ticker, score, tilt in zip(equity_tickers, score_arr, tilts_arr):
        final = max(0.0, base_per + tilt)
        if tilt > 1e-4:
            direction = "overweight"
        elif tilt < -1e-4:
            direction = "underweight"
        else:
            direction = "neutral"
        results.append(
            TickerTilt(
                ticker=ticker,
                score=float(score),
                base_weight=base_per,
                tilt=float(tilt),
                final_weight=float(final),
                direction=direction,
            )
        )

    total = sum(r.final_weight for r in results)
    if total > 0:
        scale = base_equity_weight / total
        for r in results:
            r.final_weight *= scale

    return results


def build_profile_target_weights(
    profile: RiskProfile,
    equity_tickers: list[str],
    equity_scores: Optional[dict[str, float]] = None,
    tilt_cap: float = 0.05,
) -> tuple[dict[str, float], list[TickerTilt]]:
    """Compute target ticker weights for a single rebalance date."""
    weights: dict[str, float] = {}
    tilts: list[TickerTilt] = []

    for asset_class, target in profile.targets.items():
        if asset_class == "us_equity":
            scores = equity_scores or {}
            tilts = compute_equity_tilts(equity_tickers, target, scores, tilt_cap)
            for t in tilts:
                weights[t.ticker] = weights.get(t.ticker, 0.0) + t.final_weight
        else:
            proxy = INDEX_PROXIES.get(asset_class)
            if proxy is None:
                continue
            weights[proxy] = weights.get(proxy, 0.0) + target

    return weights, tilts


def _asset_group_for_ticker(ticker: str, equity_tickers: list[str]) -> str:
    if ticker in equity_tickers:
        return "us_equity"
    for asset_class, proxy in INDEX_PROXIES.items():
        if proxy == ticker:
            return asset_class
    return "other"


def _format_quarter(date: pd.Timestamp) -> str:
    return f"Q{((date.month - 1) // 3) + 1} {date.year}"


def _format_alloc_summary(targets: dict[str, float]) -> str:
    parts = []
    for asset_class, weight in targets.items():
        if weight <= 0:
            continue
        friendly = FRIENDLY_CLASS_NAMES.get(asset_class, asset_class)
        parts.append(f"{int(round(weight * 100))}% {friendly}")
    return ", ".join(parts)


def generate_rationale(
    profile: RiskProfile,
    tilts: list[TickerTilt],
    rebalance_date: pd.Timestamp,
    tilt_applied: bool,
) -> str:
    """Build a human-readable explanation for one rebalance event."""
    quarter = _format_quarter(rebalance_date)
    alloc_summary = _format_alloc_summary(profile.targets)
    sentences = [
        f"{quarter} rebalance — {profile.name} profile.",
        f"Target allocation: {alloc_summary}.",
    ]

    if not tilt_applied or not tilts:
        sentences.append(
            "Equity sleeve held at equal weight across selected names. "
            "No AI tilt applied this period."
        )
        return " ".join(sentences)

    sleeve_total = sum(t.final_weight for t in tilts)
    names = ", ".join(t.ticker for t in tilts)
    sentences.append(
        f"Equity sleeve ({sleeve_total * 100:.0f}% of portfolio) spread across "
        f"{names}; base equal-weight before AI tilt."
    )

    moved = [t for t in tilts if t.direction != "neutral"]
    if not moved:
        sentences.append(
            "AI tilt: predicted-return signals were too similar across names "
            "to justify any tilt this quarter."
        )
        return " ".join(sentences)

    ranked = sorted(tilts, key=lambda t: -t.score)
    rank_label: dict[str, str] = {}
    if len(ranked) >= 2:
        rank_label[ranked[0].ticker] = "highest score in sleeve"
        rank_label[ranked[-1].ticker] = "lowest score in sleeve"
    if len(ranked) >= 3:
        for t in ranked[1:-1]:
            rank_label[t.ticker] = "mid score"

    moved.sort(key=lambda t: -abs(t.tilt))
    phrases: list[str] = []
    for t in moved:
        sign = "+" if t.tilt > 0 else ""
        label = rank_label.get(t.ticker, "")
        suffix = f", {label}" if label else ""
        phrases.append(
            f"{t.direction} {t.ticker} ({sign}{t.tilt * 100:.1f}%, "
            f"ML score={t.score:+.4f}{suffix})"
        )
    sentences.append(
        "AI tilt within equity sleeve (rankings are relative — the sleeve total "
        "stays at the profile target): " + "; ".join(phrases) + "."
    )
    return " ".join(sentences)


def build_profile_allocation_timeline(
    profile: RiskProfile,
    trading_days: list[pd.Timestamp],
    equity_tickers: list[str],
    prediction_df: Optional[pd.DataFrame] = None,
    tilt_cap: float = 0.05,
    use_tilt: bool = True,
) -> tuple[pd.DataFrame, list[dict]]:
    """Build a quarterly-rebalance allocation timeline plus a rationale log.

    The returned DataFrame has columns Date, Ticker, AssetGroup, group_weight,
    ticker_weight — matching the format consumed by `run_simulation`.
    """
    rebalance_dates = get_quarterly_rebalance_dates(trading_days)

    score_lookup: dict[pd.Timestamp, dict[str, float]] = {}
    if prediction_df is not None and not prediction_df.empty and "score" in prediction_df.columns:
        pdf = prediction_df.copy()
        pdf["Date"] = pd.to_datetime(pdf["Date"])
        for date, sub in pdf.groupby("Date"):
            score_lookup[pd.Timestamp(date)] = dict(zip(sub["Ticker"], sub["score"]))

    rows: list[dict] = []
    rationale_log: list[dict] = []

    for rb_date in rebalance_dates:
        equity_scores: Optional[dict[str, float]] = None
        if use_tilt and score_lookup:
            available = [d for d in score_lookup if d <= rb_date]
            if available:
                latest = max(available)
                equity_scores = {
                    t: s
                    for t, s in score_lookup[latest].items()
                    if t in equity_tickers
                }
                if not equity_scores:
                    equity_scores = None

        tilt_applied = use_tilt and equity_scores is not None

        weights, tilts = build_profile_target_weights(
            profile=profile,
            equity_tickers=equity_tickers,
            equity_scores=equity_scores,
            tilt_cap=tilt_cap,
        )

        for ticker, weight in weights.items():
            asset_group = _asset_group_for_ticker(ticker, equity_tickers)
            rows.append(
                {
                    "Date": rb_date,
                    "Ticker": ticker,
                    "AssetGroup": asset_group,
                    "group_weight": profile.targets.get(asset_group, 0.0),
                    "ticker_weight": float(weight),
                }
            )

        rationale_log.append(
            {
                "date": rb_date,
                "profile": profile.key,
                "tilt_applied": tilt_applied,
                "target_weights": {k: float(v) for k, v in weights.items()},
                "tilts": [
                    {
                        "ticker": t.ticker,
                        "score": t.score,
                        "base_weight": t.base_weight,
                        "tilt": t.tilt,
                        "final_weight": t.final_weight,
                        "direction": t.direction,
                    }
                    for t in tilts
                ],
                "rationale_text": generate_rationale(
                    profile=profile,
                    tilts=tilts,
                    rebalance_date=rb_date,
                    tilt_applied=tilt_applied,
                ),
            }
        )

    allocation_df = (
        pd.DataFrame(rows)
        .sort_values(["Date", "Ticker"])
        .reset_index(drop=True)
    )
    return allocation_df, rationale_log
