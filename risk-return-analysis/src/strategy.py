import numpy as np
import pandas as pd

from src.features import ASSET_GROUP_MAP


def compute_asset_group_scores(
    prediction_df: pd.DataFrame,
    score_col: str = "score",
) -> pd.DataFrame:
    """Aggregate per-ticker scores into asset-group-level scores per date."""
    df = prediction_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["AssetGroup"] = df["Ticker"].map(ASSET_GROUP_MAP)

    group_scores = (
        df.groupby(["Date", "AssetGroup"])[score_col]
        .agg(["mean", "count"])
        .rename(columns={"mean": "group_score", "count": "n_assets"})
        .reset_index()
    )
    return group_scores


def compute_group_volatilities(
    feature_df: pd.DataFrame,
    vol_col: str = "volatility_21d",
) -> pd.DataFrame:
    """Compute average volatility per asset group per date."""
    df = feature_df[["Date", "Ticker", "AssetGroup", vol_col]].copy()
    df["Date"] = pd.to_datetime(df["Date"])

    group_vol = (
        df.groupby(["Date", "AssetGroup"])[vol_col]
        .mean()
        .rename("group_volatility")
        .reset_index()
    )
    return group_vol


def compute_portfolio_weights(
    group_scores: pd.DataFrame,
    group_volatilities: pd.DataFrame,
    method: str = "score_weighted",
    min_weight: float = 0.05,
    max_weight: float = 0.80,
    vol_scaling: bool = True,
) -> pd.DataFrame:
    """Convert group scores and volatilities into portfolio weights per asset group.

    Steps:
    1. Softmax on group_score to get raw weights
    2. If vol_scaling, multiply by 1/group_volatility and renormalize
    3. Clip to [min_weight, max_weight] and renormalize
    """
    merged = group_scores.merge(group_volatilities, on=["Date", "AssetGroup"], how="left")

    rows = []
    for date, day_df in merged.groupby("Date"):
        scores = day_df["group_score"].values
        groups = day_df["AssetGroup"].values
        vols = day_df["group_volatility"].values

        # Step 1: softmax on scores
        shifted = scores - scores.max()
        exp_scores = np.exp(shifted)
        weights = exp_scores / exp_scores.sum()

        # Step 2: inverse-volatility scaling
        if vol_scaling and not np.any(np.isnan(vols)) and np.all(vols > 0):
            inv_vol = 1.0 / vols
            weights = weights * inv_vol
            weights = weights / weights.sum()

        # Step 3: clip and renormalize
        weights = np.clip(weights, min_weight, max_weight)
        weights = weights / weights.sum()

        for group, weight in zip(groups, weights):
            rows.append({"Date": date, "AssetGroup": group, "weight": float(weight)})

    return pd.DataFrame(rows)


def compute_within_group_weights(
    prediction_df: pd.DataFrame,
    group_weights: pd.DataFrame,
    score_col: str = "score",
) -> pd.DataFrame:
    """Distribute each asset group's weight across individual tickers
    proportional to their individual scores."""
    df = prediction_df[["Date", "Ticker", "AssetGroup", score_col]].copy()
    df["Date"] = pd.to_datetime(df["Date"])

    merged = df.merge(group_weights, on=["Date", "AssetGroup"], how="inner")

    rows = []
    for (date, group), sub in merged.groupby(["Date", "AssetGroup"]):
        group_w = sub["weight"].iloc[0]
        scores = sub[score_col].values.copy()

        # Shift scores to be non-negative for proportional allocation
        if scores.min() < 0:
            scores = scores - scores.min() + 1e-8

        total = scores.sum()
        if total <= 0:
            # Equal weight within group
            ticker_weights = np.ones(len(scores)) / len(scores) * group_w
        else:
            ticker_weights = (scores / total) * group_w

        for ticker, tw in zip(sub["Ticker"].values, ticker_weights):
            rows.append({
                "Date": date,
                "Ticker": ticker,
                "AssetGroup": group,
                "group_weight": group_w,
                "ticker_weight": float(tw),
            })

    return pd.DataFrame(rows)


def build_allocation_timeline(
    prediction_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    rebalance_every_n_days: int = 5,
    method: str = "score_weighted",
    vol_scaling: bool = True,
    min_group_weight: float = 0.05,
    max_group_weight: float = 0.80,
) -> pd.DataFrame:
    """Build a complete allocation timeline with group-level and ticker-level
    weights at each rebalance date.

    Returns DataFrame: Date, Ticker, AssetGroup, group_weight, ticker_weight
    """
    df = prediction_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # Filter to rebalance dates
    unique_dates = sorted(df["Date"].unique())
    rebalance_dates = unique_dates[::rebalance_every_n_days]
    df = df[df["Date"].isin(rebalance_dates)].copy()

    group_scores = compute_asset_group_scores(df)
    group_vols = compute_group_volatilities(feature_df)

    # Filter group_vols to rebalance dates
    group_vols = group_vols[group_vols["Date"].isin(rebalance_dates)].copy()

    group_weights = compute_portfolio_weights(
        group_scores,
        group_vols,
        method=method,
        min_weight=min_group_weight,
        max_weight=max_group_weight,
        vol_scaling=vol_scaling,
    )

    allocation = compute_within_group_weights(df, group_weights)
    return allocation.sort_values(["Date", "AssetGroup", "Ticker"]).reset_index(drop=True)
