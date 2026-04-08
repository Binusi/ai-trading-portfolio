import numpy as np
import pandas as pd

from src.asset_config import ASSET_UNIVERSE


BASE_ALLOCATIONS = {
    "conservative": {
        "us_equity": 0.20,
        "equity_index_etf": 0.10,
        "bond_etf": 0.35,
        "commodity_etf": 0.10,
        "reit_etf": 0.10,
        "international_etf": 0.10,
        "crypto_etf": 0.05,
    },
    "moderate": {
        "us_equity": 0.30,
        "equity_index_etf": 0.10,
        "bond_etf": 0.20,
        "commodity_etf": 0.10,
        "reit_etf": 0.10,
        "international_etf": 0.10,
        "crypto_etf": 0.10,
    },
    "aggressive": {
        "us_equity": 0.40,
        "equity_index_etf": 0.15,
        "bond_etf": 0.10,
        "commodity_etf": 0.10,
        "reit_etf": 0.05,
        "international_etf": 0.10,
        "crypto_etf": 0.10,
    },
}

MIN_CLASS_WEIGHT = 0.05
MAX_CLASS_WEIGHT = 0.50
TILT_FACTOR = 0.3


def aggregate_model_signals(
    all_predictions: list[pd.DataFrame],
    results_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine predictions from multiple models into a consensus score per ticker per date.

    Each model's predictions are weighted by its validation Sharpe ratio (clamped >= 0).
    """
    if not all_predictions:
        return pd.DataFrame(columns=["Date", "Ticker", "AssetGroup", "consensus_score", "n_models", "signal_agreement"])

    # Build a weight lookup: (target_name, group_name, model_name) -> sharpe weight
    weight_lookup = {}
    if "val_backtest_sharpe" in results_df.columns:
        for _, row in results_df.iterrows():
            key = (row["target_name"], row["group_name"], row["model_name"])
            sharpe = row["val_backtest_sharpe"]
            weight_lookup[key] = max(0.0, sharpe) if not np.isnan(sharpe) else 0.0

    weighted_rows = []
    for pred_df in all_predictions:
        if pred_df.empty:
            continue

        model_name = pred_df["model_name"].iloc[0]
        target_name = pred_df["target_name"].iloc[0] if "target_name" in pred_df.columns else "unknown"
        group_name = pred_df["group_name"].iloc[0] if "group_name" in pred_df.columns else "all_assets"

        key = (target_name, group_name, model_name)
        weight = weight_lookup.get(key, 1.0)

        subset = pred_df[["Date", "Ticker", "AssetGroup", "score"]].copy()
        subset["weight"] = weight
        subset["direction"] = np.where(subset["score"] > 0, 1, -1)
        weighted_rows.append(subset)

    if not weighted_rows:
        return pd.DataFrame(columns=["Date", "Ticker", "AssetGroup", "consensus_score", "n_models", "signal_agreement"])

    all_scores = pd.concat(weighted_rows, ignore_index=True)
    all_scores["weighted_score"] = all_scores["score"] * all_scores["weight"]

    grouped = all_scores.groupby(["Date", "Ticker", "AssetGroup"]).agg(
        total_weighted_score=("weighted_score", "sum"),
        total_weight=("weight", "sum"),
        n_models=("score", "count"),
        positive_signals=("direction", lambda x: (x > 0).sum()),
    ).reset_index()

    grouped["consensus_score"] = np.where(
        grouped["total_weight"] > 0,
        grouped["total_weighted_score"] / grouped["total_weight"],
        0.0,
    )
    grouped["signal_agreement"] = grouped["positive_signals"] / grouped["n_models"]

    return grouped[["Date", "Ticker", "AssetGroup", "consensus_score", "n_models", "signal_agreement"]]


def classify_signals(
    consensus_df: pd.DataFrame,
    buy_threshold: float = 0.05,
    sell_threshold: float = -0.03,
) -> pd.DataFrame:
    """Add a BUY / HOLD / SELL signal column based on consensus score thresholds."""
    df = consensus_df.copy()

    conditions = [
        df["consensus_score"] >= buy_threshold,
        df["consensus_score"] <= sell_threshold,
    ]
    choices = ["BUY", "SELL"]
    df["signal"] = np.select(conditions, choices, default="HOLD")

    return df


def compute_asset_class_allocation(
    consensus_df: pd.DataFrame,
    risk_profile: str = "moderate",
) -> dict:
    """Compute recommended asset class weights by tilting base allocations with ML signals.

    Returns dict with keys: weights, conviction, base_weights.
    """
    base = BASE_ALLOCATIONS.get(risk_profile, BASE_ALLOCATIONS["moderate"]).copy()

    if consensus_df.empty:
        return {"weights": base, "conviction": {}, "base_weights": base.copy()}

    latest_date = consensus_df["Date"].max()
    latest = consensus_df[consensus_df["Date"] == latest_date]

    class_scores = latest.groupby("AssetGroup")["consensus_score"].mean().to_dict()

    # Normalize scores to [-1, 1] range for tilting
    all_scores = list(class_scores.values())
    if all_scores:
        max_abs = max(abs(s) for s in all_scores) or 1.0
        normalized = {k: v / max_abs for k, v in class_scores.items()}
    else:
        normalized = {}

    # Apply tilts
    tilted = {}
    for asset_class, base_weight in base.items():
        signal = normalized.get(asset_class, 0.0)
        tilted[asset_class] = base_weight * (1.0 + TILT_FACTOR * signal)

    # Enforce min/max constraints
    for asset_class in tilted:
        tilted[asset_class] = np.clip(tilted[asset_class], MIN_CLASS_WEIGHT, MAX_CLASS_WEIGHT)

    # Renormalize to sum to 1.0
    total = sum(tilted.values())
    if total > 0:
        tilted = {k: v / total for k, v in tilted.items()}

    # Conviction level per class
    conviction = {}
    for asset_class in base:
        score = class_scores.get(asset_class, 0.0)
        if abs(score) >= 0.06:
            conviction[asset_class] = "HIGH"
        elif abs(score) >= 0.03:
            conviction[asset_class] = "MEDIUM"
        else:
            conviction[asset_class] = "LOW"

    return {
        "weights": tilted,
        "conviction": conviction,
        "base_weights": base,
    }


def _get_top_picks(signal_df: pd.DataFrame, n_per_class: int = 3) -> dict[str, list[dict]]:
    """Get top N picks per asset class from the latest date's signals."""
    if signal_df.empty:
        return {}

    latest_date = signal_df["Date"].max()
    latest = signal_df[signal_df["Date"] == latest_date].copy()
    latest = latest.sort_values("consensus_score", ascending=False)

    picks = {}
    for asset_class, group in latest.groupby("AssetGroup"):
        top = group.head(n_per_class)
        picks[asset_class] = [
            {
                "ticker": row["Ticker"],
                "score": row["consensus_score"],
                "signal": row["signal"],
                "agreement": row["signal_agreement"],
            }
            for _, row in top.iterrows()
        ]

    return picks


def _get_action_lists(signal_df: pd.DataFrame) -> dict[str, list[dict]]:
    """Group all tickers into BUY, HOLD, SELL lists from the latest date."""
    if signal_df.empty:
        return {"BUY": [], "HOLD": [], "SELL": []}

    latest_date = signal_df["Date"].max()
    latest = signal_df[signal_df["Date"] == latest_date].copy()
    latest = latest.sort_values("consensus_score", ascending=False)

    actions = {"BUY": [], "HOLD": [], "SELL": []}
    for _, row in latest.iterrows():
        actions[row["signal"]].append({
            "ticker": row["Ticker"],
            "asset_class": row["AssetGroup"],
            "score": row["consensus_score"],
            "agreement": row["signal_agreement"],
        })

    return actions


def build_portfolio_strategy(
    all_predictions: list[pd.DataFrame],
    results_df: pd.DataFrame,
    risk_profile: str = "moderate",
) -> dict:
    """Build a complete portfolio strategy from model predictions.

    Returns a dict with: allocation, top_picks, actions, consensus_df, metadata.
    """
    consensus_df = aggregate_model_signals(all_predictions, results_df)
    signal_df = classify_signals(consensus_df)
    allocation = compute_asset_class_allocation(signal_df, risk_profile=risk_profile)
    top_picks = _get_top_picks(signal_df)
    actions = _get_action_lists(signal_df)

    latest_date = consensus_df["Date"].max() if not consensus_df.empty else None

    return {
        "allocation": allocation,
        "top_picks": top_picks,
        "actions": actions,
        "consensus_df": signal_df,
        "metadata": {
            "date": latest_date,
            "n_models": int(consensus_df["n_models"].max()) if not consensus_df.empty else 0,
            "n_tickers": consensus_df["Ticker"].nunique() if not consensus_df.empty else 0,
            "risk_profile": risk_profile,
        },
    }
