from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from src.features import ASSET_GROUP_MAP
from src.utils import get_trading_days, get_rebalance_dates, get_prices_for_date

TRADING_DAYS_PER_YEAR = 252


@dataclass
class Position:
    """Tracks a single asset position."""
    ticker: str
    shares: float
    avg_cost: float
    current_price: float

    @property
    def market_value(self) -> float:
        return self.shares * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.shares * (self.current_price - self.avg_cost)


@dataclass
class PortfolioState:
    """Snapshot of portfolio at a point in time."""
    date: pd.Timestamp
    cash: float
    positions: dict[str, Position] = field(default_factory=dict)

    @property
    def total_value(self) -> float:
        positions_value = sum(p.market_value for p in self.positions.values())
        return self.cash + positions_value

    @property
    def positions_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def position_weights(self) -> dict[str, float]:
        total = self.total_value
        if total <= 0:
            return {}
        return {
            ticker: pos.market_value / total
            for ticker, pos in self.positions.items()
            if pos.shares > 0
        }


def compute_trades(
    state: PortfolioState,
    target_weights: dict[str, float],
    prices: dict[str, float],
    transaction_cost_bps: float = 10.0,
    min_trade_value: float = 1.0,
) -> tuple[list[dict], float]:
    """Determine buy/sell orders to move from current weights to target weights.

    Returns (list of trade dicts, total transaction cost in dollars).
    """
    total_value = state.total_value
    current_weights = state.position_weights
    trades = []
    total_cost = 0.0

    all_tickers = set(target_weights.keys()) | set(current_weights.keys())

    for ticker in all_tickers:
        if ticker not in prices:
            continue

        current_w = current_weights.get(ticker, 0.0)
        target_w = target_weights.get(ticker, 0.0)
        price = prices[ticker]

        current_value = current_w * total_value
        target_value = target_w * total_value
        trade_value = target_value - current_value

        if abs(trade_value) < min_trade_value:
            continue

        shares = trade_value / price
        cost = abs(trade_value) * (transaction_cost_bps / 10000.0)

        action = "buy" if trade_value > 0 else "sell"
        trades.append({
            "ticker": ticker,
            "action": action,
            "shares": abs(shares),
            "price": price,
            "dollar_value": abs(trade_value),
            "transaction_cost": cost,
        })
        total_cost += cost

    return trades, total_cost


def execute_trades(
    state: PortfolioState,
    trades: list[dict],
    transaction_cost: float,
) -> PortfolioState:
    """Apply trades to portfolio state. Returns new PortfolioState."""
    new_positions = {t: Position(t, p.shares, p.avg_cost, p.current_price)
                     for t, p in state.positions.items()}
    new_cash = state.cash - transaction_cost

    for trade in trades:
        ticker = trade["ticker"]
        shares = trade["shares"]
        price = trade["price"]

        if trade["action"] == "buy":
            if ticker in new_positions:
                pos = new_positions[ticker]
                total_shares = pos.shares + shares
                if total_shares > 0:
                    pos.avg_cost = (pos.avg_cost * pos.shares + price * shares) / total_shares
                pos.shares = total_shares
            else:
                new_positions[ticker] = Position(ticker, shares, price, price)
            new_cash -= shares * price

        elif trade["action"] == "sell":
            if ticker in new_positions:
                pos = new_positions[ticker]
                pos.shares = max(0.0, pos.shares - shares)
                new_cash += shares * price
                if pos.shares <= 1e-10:
                    del new_positions[ticker]

    return PortfolioState(
        date=state.date,
        cash=new_cash,
        positions=new_positions,
    )


def update_prices(
    state: PortfolioState,
    prices: dict[str, float],
    date: pd.Timestamp,
) -> PortfolioState:
    """Mark positions to market with new prices."""
    new_positions = {}
    for ticker, pos in state.positions.items():
        new_price = prices.get(ticker, pos.current_price)
        new_positions[ticker] = Position(ticker, pos.shares, pos.avg_cost, new_price)

    return PortfolioState(date=date, cash=state.cash, positions=new_positions)


def run_simulation(
    allocation_timeline: pd.DataFrame,
    price_data: pd.DataFrame,
    tickers: list[str],
    start_date: str = "2024-01-01",
    end_date: str = "2025-12-31",
    initial_capital: float = 1000.0,
    rebalance_every_n_days: int = 5,
    transaction_cost_bps: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run dollar-based trade simulation.

    Returns (daily_log_df, trade_log_df).
    """
    trading_days = get_trading_days(start_date, end_date, price_data)
    rebalance_dates_set = set(get_rebalance_dates(trading_days, rebalance_every_n_days))

    # Build allocation lookup: date -> {ticker: weight}
    alloc = allocation_timeline.copy()
    alloc["Date"] = pd.to_datetime(alloc["Date"])
    alloc_lookup = {}
    for date, group in alloc.groupby("Date"):
        alloc_lookup[date] = dict(zip(group["Ticker"], group["ticker_weight"]))

    state = PortfolioState(date=pd.Timestamp(start_date), cash=initial_capital)
    daily_rows = []
    trade_rows = []
    prev_value = initial_capital

    for day in trading_days:
        prices = get_prices_for_date(price_data, day, tickers)
        if not prices:
            continue

        state = update_prices(state, prices, day)

        if day in rebalance_dates_set:
            # Find the closest allocation date <= today
            alloc_dates = sorted(alloc_lookup.keys())
            target_weights = {}
            for ad in reversed(alloc_dates):
                if ad <= day:
                    target_weights = alloc_lookup[ad]
                    break

            if target_weights:
                trades, cost = compute_trades(
                    state, target_weights, prices, transaction_cost_bps
                )
                state = execute_trades(state, trades, cost)
                state = update_prices(state, prices, day)

                for trade in trades:
                    trade_rows.append({"Date": day, **trade})

        # Record daily snapshot
        total = state.total_value
        daily_return = (total / prev_value - 1) if prev_value > 0 else 0.0

        row = {
            "Date": day,
            "cash": state.cash,
            "positions_value": state.positions_value,
            "total_value": total,
            "daily_return": daily_return,
        }

        # Record per-ticker weights
        weights = state.position_weights
        for ticker in tickers:
            row[f"{ticker}_weight"] = weights.get(ticker, 0.0)

        # Record per-group weights
        group_weights: dict[str, float] = {}
        for ticker, w in weights.items():
            group = ASSET_GROUP_MAP.get(ticker, "other")
            group_weights[group] = group_weights.get(group, 0.0) + w
        for group in ["equity", "bond_etf", "equity_index_etf"]:
            row[f"{group}_weight"] = group_weights.get(group, 0.0)

        daily_rows.append(row)
        prev_value = total

    daily_log = pd.DataFrame(daily_rows)
    if not daily_log.empty:
        daily_log["cumulative_return"] = (1 + daily_log["daily_return"]).cumprod() - 1

    trade_log = pd.DataFrame(trade_rows)
    return daily_log, trade_log


def compute_simulation_metrics(
    daily_log: pd.DataFrame,
    initial_capital: float = 1000.0,
) -> dict:
    """Compute performance metrics from the simulation daily log."""
    if daily_log.empty:
        return {}

    returns = daily_log["daily_return"]
    final_value = daily_log["total_value"].iloc[-1]
    n_days = len(daily_log)
    years = n_days / TRADING_DAYS_PER_YEAR

    total_return = final_value / initial_capital - 1
    annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    mean_ret = returns.mean()
    std_ret = returns.std()
    downside_std = returns[returns < 0].std()

    sharpe = (mean_ret / std_ret * np.sqrt(TRADING_DAYS_PER_YEAR)) if std_ret > 0 else np.nan
    sortino = (mean_ret / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)) if downside_std > 0 else np.nan

    # Max drawdown
    equity_curve = (1 + returns).cumprod()
    rolling_peak = equity_curve.cummax()
    drawdown = equity_curve / rolling_peak - 1
    max_dd = float(drawdown.min())
    max_dd_idx = drawdown.idxmin()
    max_dd_date = str(daily_log.loc[max_dd_idx, "Date"].date()) if pd.notna(max_dd_idx) else None

    metrics = {
        "initial_capital": initial_capital,
        "final_portfolio_value": final_value,
        "total_return": total_return,
        "annualized_return": annualized_return,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_date": max_dd_date,
        "total_transaction_costs": 0.0,
        "total_trades": 0,
        "win_rate": float((returns > 0).mean()),
        "best_day_return": float(returns.max()),
        "worst_day_return": float(returns.min()),
    }

    # Group weight averages
    for group in ["equity", "bond_etf", "equity_index_etf"]:
        col = f"{group}_weight"
        if col in daily_log.columns:
            key = f"avg_{group.replace('_etf', '').replace('equity_index', 'etf')}_weight"
            if group == "equity":
                metrics["avg_equity_weight"] = float(daily_log[col].mean())
            elif group == "bond_etf":
                metrics["avg_bond_weight"] = float(daily_log[col].mean())
            elif group == "equity_index_etf":
                metrics["avg_etf_weight"] = float(daily_log[col].mean())

    return metrics
