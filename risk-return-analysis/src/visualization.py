import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np


COLORS = {
    "equity": "#4C72B0",
    "bond_etf": "#DD8452",
    "equity_index_etf": "#55A868",
}

TICKER_COLORS = {
    "AAPL": "#4C72B0",
    "MSFT": "#DD8452",
    "NVDA": "#C44E52",
    "SPY": "#55A868",
    "TLT": "#8172B3",
}


def _style_axis(ax: plt.Axes) -> None:
    """Apply consistent date formatting to an axis."""
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3)


def plot_asset_class_weights_over_time(
    daily_log: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Stacked area chart of asset class weights over time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    dates = pd.to_datetime(daily_log["Date"])

    groups = ["equity", "bond_etf", "equity_index_etf"]
    labels = ["Equity", "Bonds (TLT)", "Equity Index (SPY)"]
    colors = [COLORS[g] for g in groups]

    data = []
    for g in groups:
        col = f"{g}_weight"
        data.append(daily_log[col].values if col in daily_log.columns else np.zeros(len(daily_log)))

    ax.stackplot(dates, *data, labels=labels, colors=colors, alpha=0.8)
    ax.set_ylabel("Portfolio Weight")
    ax.set_title("Portfolio Allocation by Asset Class Over Time")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", framealpha=0.9)
    _style_axis(ax)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_individual_ticker_weights(
    daily_log: pd.DataFrame,
    tickers: list[str],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Stacked area chart of individual ticker weights over time."""
    fig, ax = plt.subplots(figsize=(14, 6))
    dates = pd.to_datetime(daily_log["Date"])

    data = []
    labels = []
    colors = []
    for ticker in tickers:
        col = f"{ticker}_weight"
        if col in daily_log.columns:
            data.append(daily_log[col].values)
            labels.append(ticker)
            colors.append(TICKER_COLORS.get(ticker, "#999999"))

    ax.stackplot(dates, *data, labels=labels, colors=colors, alpha=0.8)
    ax.set_ylabel("Portfolio Weight")
    ax.set_title("Portfolio Allocation by Individual Asset Over Time")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", framealpha=0.9)
    _style_axis(ax)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_portfolio_value(
    daily_log: pd.DataFrame,
    benchmark_prices: Optional[pd.Series] = None,
    initial_capital: float = 1000.0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Line chart of portfolio value ($) over time, with optional benchmark."""
    fig, ax = plt.subplots(figsize=(14, 6))
    dates = pd.to_datetime(daily_log["Date"])

    ax.plot(dates, daily_log["total_value"], label="ML Portfolio", color="#4C72B0", linewidth=1.5)

    if benchmark_prices is not None and len(benchmark_prices) > 0:
        bench_dates = pd.to_datetime(benchmark_prices.index)
        # Normalize benchmark to same starting capital
        bench_values = benchmark_prices / benchmark_prices.iloc[0] * initial_capital
        ax.plot(bench_dates, bench_values, label="SPY (Buy & Hold)", color="#999999",
                linewidth=1.5, linestyle="--")

    ax.axhline(y=initial_capital, color="red", linestyle=":", alpha=0.5, label="Initial Capital")
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_title("Portfolio Value Over Time")
    ax.legend(loc="upper left", framealpha=0.9)
    _style_axis(ax)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_drawdown(
    daily_log: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Drawdown from peak chart."""
    fig, ax = plt.subplots(figsize=(14, 4))
    dates = pd.to_datetime(daily_log["Date"])

    equity = (1 + daily_log["daily_return"]).cumprod()
    peak = equity.cummax()
    drawdown = (equity / peak - 1) * 100

    ax.fill_between(dates, drawdown, 0, color="#C44E52", alpha=0.5)
    ax.plot(dates, drawdown, color="#C44E52", linewidth=0.8)
    ax.set_ylabel("Drawdown (%)")
    ax.set_title("Portfolio Drawdown from Peak")
    ax.set_ylim(drawdown.min() * 1.1, 1)
    _style_axis(ax)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_daily_pnl(
    daily_log: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Bar chart of daily P&L in dollars."""
    fig, ax = plt.subplots(figsize=(14, 4))
    dates = pd.to_datetime(daily_log["Date"])

    pnl = daily_log["total_value"].diff().fillna(0)
    colors = ["#55A868" if v >= 0 else "#C44E52" for v in pnl]

    ax.bar(dates, pnl, color=colors, width=1.0, alpha=0.7)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Daily P&L ($)")
    ax.set_title("Daily Profit & Loss")
    _style_axis(ax)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_regime_analysis(
    daily_log: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Multi-panel: portfolio value + asset class weights together."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True,
                                    gridspec_kw={"height_ratios": [2, 1]})
    dates = pd.to_datetime(daily_log["Date"])

    # Top: portfolio value
    ax1.plot(dates, daily_log["total_value"], color="#4C72B0", linewidth=1.5)
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.set_title("Portfolio Performance & Asset Allocation Regime Analysis")
    ax1.grid(True, alpha=0.3)

    # Bottom: asset class weights stacked
    groups = ["equity", "bond_etf", "equity_index_etf"]
    labels = ["Equity", "Bonds (TLT)", "Equity Index (SPY)"]
    colors = [COLORS[g] for g in groups]

    data = []
    for g in groups:
        col = f"{g}_weight"
        data.append(daily_log[col].values if col in daily_log.columns else np.zeros(len(daily_log)))

    ax2.stackplot(dates, *data, labels=labels, colors=colors, alpha=0.8)
    ax2.set_ylabel("Weight")
    ax2.set_ylim(0, 1)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8)
    _style_axis(ax2)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def generate_all_plots(
    daily_log: pd.DataFrame,
    trade_log: pd.DataFrame,
    tickers: list[str],
    benchmark_prices: Optional[pd.Series] = None,
    initial_capital: float = 1000.0,
    output_dir: str = "output",
) -> list[str]:
    """Generate all visualizations and save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    saved = []

    path = os.path.join(output_dir, "asset_class_weights.png")
    plot_asset_class_weights_over_time(daily_log, save_path=path)
    plt.close()
    saved.append(path)

    path = os.path.join(output_dir, "ticker_weights.png")
    plot_individual_ticker_weights(daily_log, tickers, save_path=path)
    plt.close()
    saved.append(path)

    path = os.path.join(output_dir, "portfolio_value.png")
    plot_portfolio_value(daily_log, benchmark_prices, initial_capital, save_path=path)
    plt.close()
    saved.append(path)

    path = os.path.join(output_dir, "drawdown.png")
    plot_drawdown(daily_log, save_path=path)
    plt.close()
    saved.append(path)

    path = os.path.join(output_dir, "daily_pnl.png")
    plot_daily_pnl(daily_log, save_path=path)
    plt.close()
    saved.append(path)

    path = os.path.join(output_dir, "regime_analysis.png")
    plot_regime_analysis(daily_log, save_path=path)
    plt.close()
    saved.append(path)

    return saved
