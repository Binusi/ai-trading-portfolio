import numpy as np
import pandas as pd


TRADING_DAYS = 252



def max_drawdown_from_returns(return_series: pd.Series) -> float:
    equity_curve = (1 + return_series.fillna(0.0)).cumprod()
    rolling_peak = equity_curve.cummax()
    drawdown = equity_curve / rolling_peak - 1
    return float(drawdown.min())



def build_top_k_backtest(
    prediction_df: pd.DataFrame,
    realized_return_col: str,
    score_col: str = "score",
    top_k: int = 2,
    rebalance_every_n_days: int = 5,
    long_only: bool = True,
    positive_score_only: bool = True,
    transaction_cost_bps: float = 10.0,
) -> tuple[pd.DataFrame, dict]:
    df = prediction_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", score_col], ascending=[True, False]).reset_index(drop=True)

    unique_dates = sorted(df["Date"].unique())
    rebalance_dates = unique_dates[::rebalance_every_n_days]

    portfolio_rows = []
    previous_weights = {}
    top_pick_hits = []

    for date in rebalance_dates:
        day_df = df[df["Date"] == date].copy()
        day_df = day_df.sort_values(score_col, ascending=False)

        if positive_score_only:
            day_df = day_df[day_df[score_col] > 0].copy()

        selected = day_df.head(top_k).copy()

        if selected.empty:
            current_weights = {}
            gross_return = 0.0
            top_pick_hit = np.nan
        else:
            if long_only:
                weight = 1.0 / len(selected)
                selected["weight"] = weight
            else:
                raw_scores = selected[score_col].abs()
                selected["weight"] = raw_scores / raw_scores.sum()

            current_weights = dict(zip(selected["Ticker"], selected["weight"]))
            gross_return = float((selected["weight"] * selected[realized_return_col]).sum())
            top_pick_hit = int(selected.iloc[0][realized_return_col] > 0)

        turnover = 0.0
        all_tickers = set(previous_weights) | set(current_weights)
        for ticker in all_tickers:
            old_w = previous_weights.get(ticker, 0.0)
            new_w = current_weights.get(ticker, 0.0)
            turnover += abs(new_w - old_w)

        transaction_cost = turnover * (transaction_cost_bps / 10000.0)
        net_return = gross_return - transaction_cost

        portfolio_rows.append(
            {
                "Date": date,
                "gross_return": gross_return,
                "transaction_cost": transaction_cost,
                "net_return": net_return,
                "turnover": turnover,
                "n_positions": len(current_weights),
                "top_pick_hit": top_pick_hit,
            }
        )

        if not pd.isna(top_pick_hit):
            top_pick_hits.append(top_pick_hit)

        previous_weights = current_weights.copy()

    portfolio_df = pd.DataFrame(portfolio_rows).sort_values("Date").reset_index(drop=True)

    mean_return = portfolio_df["net_return"].mean()
    std_return = portfolio_df["net_return"].std()
    downside_std = portfolio_df.loc[portfolio_df["net_return"] < 0, "net_return"].std()

    sharpe = float(np.sqrt(TRADING_DAYS / rebalance_every_n_days) * mean_return / std_return) if std_return and not np.isnan(std_return) else np.nan
    sortino = float(np.sqrt(TRADING_DAYS / rebalance_every_n_days) * mean_return / downside_std) if downside_std and not np.isnan(downside_std) else np.nan
    max_dd = max_drawdown_from_returns(portfolio_df["net_return"]) if not portfolio_df.empty else np.nan
    avg_turnover = float(portfolio_df["turnover"].mean()) if not portfolio_df.empty else np.nan
    top_pick_hit_rate = float(np.mean(top_pick_hits)) if top_pick_hits else np.nan

    metrics = {
        "backtest_sharpe": sharpe,
        "backtest_sortino": sortino,
        "backtest_max_drawdown": max_dd,
        "backtest_avg_turnover": avg_turnover,
        "top_pick_hit_rate": top_pick_hit_rate,
        "rebalance_count": int(len(portfolio_df)),
        "avg_positions": float(portfolio_df["n_positions"].mean()) if not portfolio_df.empty else np.nan,
        "cumulative_return": float((1 + portfolio_df["net_return"].fillna(0.0)).prod() - 1) if not portfolio_df.empty else np.nan,
    }

    return portfolio_df, metrics
