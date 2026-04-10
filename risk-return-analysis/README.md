# AI Trading Portfolio - Risk & Return Analysis

ML-driven portfolio allocation and trade simulation across multiple asset classes.

## Overview

This project uses machine learning to predict asset returns and directions, then uses those predictions to:

1. **Dynamically allocate** across asset classes (equities, bonds, ETFs) over time
2. **Simulate trading** with a $1,000 starting portfolio, tracking daily P&L through end of 2025

### Assets Analyzed

| Ticker | Asset Class | Description |
|--------|------------|-------------|
| AAPL | Equity | Apple Inc. |
| MSFT | Equity | Microsoft Corp. |
| NVDA | Equity | NVIDIA Corp. |
| SPY | Equity Index ETF | S&P 500 ETF |
| TLT | Bond ETF | 20+ Year Treasury Bond ETF |

## How to Run

### Prerequisites

```bash
pip install pandas numpy scikit-learn yfinance matplotlib
```

### Run the full pipeline

```bash
cd risk-return-analysis
python main.py
```

This will:
1. Fetch historical price data (2015-2026) from Yahoo Finance
2. Engineer 54+ technical/statistical features per asset
3. Train 7 ML models across multiple targets (5-day, 10-day returns and directions)
4. Evaluate models on validation (2022-2023) and test (2024+) sets
5. Select the best model by validation Sharpe ratio
6. Build a dynamic portfolio allocation timeline
7. Run a $1,000 trade simulation (2024-01-01 to 2025-12-31)
8. Generate performance charts in `output/`

Runtime: approximately 5-10 minutes depending on network speed and hardware.

## Project Structure

```
risk-return-analysis/
  main.py                 # Entry point - runs full pipeline
  src/
    data_fetch.py         # Yahoo Finance data retrieval
    features.py           # Feature engineering (54+ features)
    model.py              # ML models, training, evaluation
    backtest.py           # Return-based backtesting
    strategy.py           # Portfolio allocation logic
    simulation.py         # Dollar-based trade simulation
    visualization.py      # Chart generation
    utils.py              # Shared helpers
  output/                 # Generated charts (created on run)
```

## Pipeline Stages

### Stage 1: ML Model Training & Evaluation

Trains 7 models (Ridge, ElasticNet, HistGradientBoosting, RandomForest for regression; Logistic, HistGradientBoosting, RandomForest for classification) on multiple targets:

- **Regression targets**: 5-day return, 10-day return, 5-day risk-adjusted return, cross-sectional 5-day return
- **Classification targets**: 5-day direction, 10-day direction, cross-sectional 5-day direction

**Output**: Model comparison table ranked by validation Sharpe ratio.

### Stage 2: Portfolio Allocation

Uses the best model's predictions to determine asset class weights over time:

1. Aggregate prediction scores by asset group (equity, bond, ETF index)
2. Apply softmax to convert scores to raw weights
3. Scale by inverse volatility (lower-volatility groups get more weight)
4. Enforce min/max constraints (5% floor, 80% ceiling per group)
5. Distribute group weights to individual tickers proportional to their scores

**Rebalance frequency** is tied to the prediction horizon (5 or 10 trading days), matching the model's forward-looking window.

### Stage 3: Trade Simulation

Simulates daily portfolio management with:

- **$1,000 initial capital**
- **Fractional shares** (necessary given small capital and high stock prices)
- **Transaction costs**: 10 basis points per trade
- **Rebalancing**: Every N days (matching prediction horizon), positions drift between rebalances
- **Period**: 2024-01-01 to 2025-12-31 (test set - model never trained on this data)

## Interpreting Results

### Console Output

- **Model Comparison Table**: All model/target combinations ranked by validation Sharpe. Higher Sharpe = better risk-adjusted returns.
- **Simulation Metrics**:
  - **Total Return**: Overall gain/loss percentage
  - **Annualized Return**: Return normalized to annual rate
  - **Sharpe Ratio**: Risk-adjusted return (>1 is good, >2 is excellent)
  - **Sortino Ratio**: Like Sharpe but only penalizes downside volatility
  - **Max Drawdown**: Largest peak-to-trough decline (closer to 0% is better)
  - **Win Rate**: Percentage of days with positive returns

### Generated Charts (in `output/`)

| Chart | What It Shows |
|-------|--------------|
| `asset_class_weights.png` | **Primary allocation chart** - stacked area showing equity vs bond vs ETF weights over time |
| `ticker_weights.png` | Individual asset weights over time |
| `portfolio_value.png` | Portfolio dollar value vs SPY buy-and-hold benchmark |
| `drawdown.png` | Peak-to-trough drawdowns over time |
| `daily_pnl.png` | Daily profit/loss in dollars |
| `regime_analysis.png` | Multi-panel combining portfolio value with allocation shifts |

### Key Questions the Charts Answer

- **"When should I hold more bonds vs stocks?"** - Look at `asset_class_weights.png`. Periods where bond allocation increases indicate the model detected higher equity risk.
- **"Did the ML portfolio beat the market?"** - Compare the blue line (ML portfolio) to the dashed gray line (SPY) in `portfolio_value.png`.
- **"How bad can losses get?"** - The `drawdown.png` chart shows the worst-case decline from any peak.
- **"Do allocation shifts correlate with performance?"** - The `regime_analysis.png` multi-panel lets you visually link allocation changes to portfolio value changes.

## Extending the Project

### Adding new assets

1. Add the ticker to `model_tickers` in `main.py`
2. Add the asset group mapping in `src/features.py` (`ASSET_GROUP_MAP`)

### Changing allocation method

The `build_allocation_timeline()` function in `src/strategy.py` accepts a `method` parameter and `vol_scaling` flag. You can modify `compute_portfolio_weights()` to implement alternative allocation strategies (e.g., mean-variance, risk parity).

### Adjusting simulation parameters

In `main.py`, modify the `run_simulation()` call:
- `initial_capital`: Starting dollar amount
- `transaction_cost_bps`: Trading cost in basis points (10 = 0.1%)
- `start_date` / `end_date`: Simulation date range
