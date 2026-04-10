import numpy as np
import pandas as pd


def get_trading_days(
    start_date: str,
    end_date: str,
    price_data: pd.DataFrame,
) -> list[pd.Timestamp]:
    """Extract actual trading days from price data within a date range."""
    close = price_data["Close"]
    if isinstance(close, pd.DataFrame):
        dates = close.dropna(how="all").index
    else:
        dates = close.dropna().index

    dates = pd.to_datetime(dates)
    mask = (dates >= pd.Timestamp(start_date)) & (dates <= pd.Timestamp(end_date))
    return sorted(dates[mask].tolist())


def get_rebalance_dates(
    trading_days: list[pd.Timestamp],
    rebalance_every_n_days: int,
) -> list[pd.Timestamp]:
    """Select every Nth trading day as a rebalance date."""
    return trading_days[::rebalance_every_n_days]


def get_prices_for_date(
    price_data: pd.DataFrame,
    date: pd.Timestamp,
    tickers: list[str],
) -> dict[str, float]:
    """Extract closing prices for all tickers on a given date."""
    prices = {}
    for ticker in tickers:
        try:
            val = price_data["Close"][ticker].loc[date]
            if pd.notna(val):
                prices[ticker] = float(val)
        except (KeyError, TypeError):
            continue
    return prices


def format_metrics_table(metrics: dict) -> str:
    """Pretty-print simulation metrics for console output."""
    lines = []
    lines.append("=" * 60)
    lines.append("SIMULATION PERFORMANCE METRICS")
    lines.append("=" * 60)

    fmt = {
        "initial_capital": ("Initial Capital", "${:,.2f}"),
        "final_portfolio_value": ("Final Portfolio Value", "${:,.2f}"),
        "total_return": ("Total Return", "{:.2%}"),
        "annualized_return": ("Annualized Return", "{:.2%}"),
        "sharpe_ratio": ("Sharpe Ratio", "{:.3f}"),
        "sortino_ratio": ("Sortino Ratio", "{:.3f}"),
        "max_drawdown": ("Max Drawdown", "{:.2%}"),
        "max_drawdown_date": ("Max Drawdown Date", "{}"),
        "total_transaction_costs": ("Total Transaction Costs", "${:,.2f}"),
        "total_trades": ("Total Trades", "{:,}"),
        "win_rate": ("Win Rate (% positive days)", "{:.1%}"),
        "best_day_return": ("Best Day Return", "{:.2%}"),
        "worst_day_return": ("Worst Day Return", "{:.2%}"),
        "avg_equity_weight": ("Avg Equity Weight", "{:.1%}"),
        "avg_bond_weight": ("Avg Bond Weight", "{:.1%}"),
        "avg_etf_weight": ("Avg ETF Index Weight", "{:.1%}"),
    }

    for key, (label, template) in fmt.items():
        if key in metrics:
            val = metrics[key]
            if val is None or (isinstance(val, float) and np.isnan(val)):
                lines.append(f"  {label:<35s}  N/A")
            else:
                lines.append(f"  {label:<35s}  {template.format(val)}")

    lines.append("=" * 60)
    return "\n".join(lines)
