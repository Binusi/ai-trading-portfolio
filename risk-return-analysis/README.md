# risk-return-analysis

Python research pipeline that trains ML models, simulates three risk-profile
portfolios with an optional AI tilt, and exports the results as JSON for the
Expo app in `../app/`.

## How it fits together

```
data fetch (yfinance)
   ↓
feature engineering (54+ features per asset)
   ↓
ML model training & evaluation  ┐
                                ├── select best model by validation Sharpe
ML predictions on the test set  ┘
   ↓
6 simulations:
  {Conservative, Balanced, Aggressive} × {no AI tilt, AI tilt}
   ↓
JSON export → ../app/assets/data/
```

## How to run

```bash
pip install -r ../requirements.txt
python main.py
```

This is the single entry point. It runs everything end-to-end and takes
roughly 5–10 minutes (most of that is ML training across multiple targets
and model types). At the end you get:

- A printed comparison table for all 6 profile/tilt combos
- Sample rationale text for the Balanced + AI tilt run
- PNG plots in `output/` (asset-class weights, portfolio value, drawdown, etc.)
- JSON files in `../app/assets/data/` that the Expo app consumes

If you only want to verify the strategy module without retraining everything:

```bash
python test_profile_strategy.py    # tilt math, quarterly dates, rationale text
python test_export_app_data.py     # JSON shape and content (uses synthetic data)
```

## Risk profiles

Defined in [`src/profiles.py`](src/profiles.py). Each profile is a fixed
cross-asset target that sums to 1.0. The simulator rebalances back to these
targets at the start of every calendar quarter.

| Asset class           | Conservative | Balanced | Aggressive |
|-----------------------|-------------:|---------:|-----------:|
| US individual stocks  | 20%          | 45%      | 65%        |
| Long-term Treasuries  | 60%          | 30%      | 10%        |
| S&P 500 ETF           | 10%          | 15%      | 15%        |
| Gold                  | 5%           | 5%       | 5%         |
| International stocks  | 5%           | 5%       | 5%         |

The non-equity sleeves are held via single ETF proxies (TLT, SPY, GLD, EFA).
The US-equity sleeve is distributed across a fixed list of individual names
(currently `AAPL`, `MSFT`, `NVDA`).

## AI tilt

The optional tilt is the *only* place the ML model influences the portfolio.
It applies inside the equity sleeve and follows three rules:

1. **Capped**: each name's weight changes by at most ±5% in absolute terms.
2. **Zero-sum**: tilts on different names cancel, so the total equity sleeve
   weight stays exactly at the profile target. Cross-asset risk does not
   change.
3. **Inspectable**: each rebalance produces a human-readable rationale that
   names which tickers were over/underweighted, by how much, and what their
   ML scores were.

Asset-class targets (the actual driver of risk and return) are **never**
adjusted by the model. The honest framing: the profile does the heavy
lifting; the tilt is a small, transparent experiment.

## Pipeline stages

### Stage 1 — ML training (`src/model.py`, `main.py`)

Trains 7 models (Ridge / ElasticNet / HistGradientBoosting / RandomForest
regressors plus Logistic / HistGradientBoosting / RandomForest classifiers)
across multiple targets (5d return, 10d return, 5d risk-adjusted return,
cross-sectional 5d return, plus the directional classifier variants).

Train / val / test splits:
- Train: 2015 → 2021-12-31
- Val:   2022 → 2023-12-31
- Test:  2024 → end of available data

The pipeline picks the best (target, model) combination by **validation**
Sharpe — using the test set's Sharpe to pick would be selection bias.

### Stage 2 — Profile-based simulation (`src/profile_strategy.py`, `src/simulation.py`)

For each `(profile, use_tilt)` pair:
1. Build a quarterly rebalance timeline (`get_quarterly_rebalance_dates`)
2. At each rebalance date, look up the latest available ML scores; compute
   the equity sleeve tilts (`compute_equity_tilts`); produce target ticker
   weights and a rationale string
3. Run the dollar-based simulator (`run_simulation`) with explicit
   quarterly rebalance dates, fractional shares, 10 bps transaction cost
4. Compute metrics (Sharpe, Sortino, max drawdown, win rate, etc.)

### Stage 3 — JSON export (`src/export_app_data.py`)

Writes 7 files to `../app/assets/data/`:

| File                       | Contents |
|----------------------------|----------|
| `summary.json`             | profile metadata, 6-row comparison, SPY benchmark series, ML model info, default view, disclaimer |
| `<profile>_<tilt>.json`    | full daily series + per-rebalance event with rationale, tilts, trades, allocation snapshot |

All values use a $1,000 starting-capital base. The app scales linearly to
the user's selected capital at display time.

## Project structure

```
risk-return-analysis/
  main.py                     entry point — runs the full pipeline
  test_profile_strategy.py    smoke test for tilt math + rationale
  test_export_app_data.py     smoke test for JSON export shape

  src/
    asset_config.py           ticker universe definitions
    backtest.py               return-based backtesting
    data_fetch.py             yfinance wrapper
    export_app_data.py        JSON output for the app
    features.py               54+ technical / statistical features
    model.py                  ML training, model registry, eval
    profile_strategy.py       quarterly allocation + AI tilt + rationale
    profiles.py               Conservative / Balanced / Aggressive
    simulation.py             dollar-based trade simulator
    strategy.py               legacy score-weighted allocation (unused by main flow)
    utils.py                  shared helpers
    visualization.py          matplotlib plots → output/

  output/                     generated PNGs (created on run)
```

## Extending

### Adding equities to the tilt universe

1. Add the ticker to `model_tickers` in [`main.py`](main.py)
2. Add the ticker to `DEFAULT_EQUITY_UNIVERSE` in [`src/profiles.py`](src/profiles.py)
3. Re-run `python main.py`

### Adjusting profile allocations

Edit the `targets` dict for the profile in [`src/profiles.py`](src/profiles.py).
The constructor enforces that targets sum to 1.0. Rerun the pipeline.

### Changing the rebalance cadence

`get_quarterly_rebalance_dates()` in `src/profile_strategy.py` is currently
hard-coded to "first trading day of each calendar quarter". Replace it with
a different selection (monthly, semi-annual, etc.) and the rest of the
pipeline picks it up automatically — `run_simulation` accepts an explicit
`rebalance_dates` list.

### Tuning the tilt cap

Pass `tilt_cap=0.03` (or any value) to `build_profile_allocation_timeline`
in `main.py`. The default is `TILT_CAP = 0.05` (5%).

## Notes on honesty

- The test window (2024–2025) is a single bull market. Don't treat the
  reported Sharpe and returns as forecasts.
- The "best model by validation Sharpe" framing is the right way to do
  model selection, but the validation period is also short and may not
  generalize.
- The AI tilt's contribution is small and roughly neutral on a risk-
  adjusted basis. Treat it as a transparent experiment, not edge.
- A 3-name equity universe is too narrow to be a real product. Expanding
  it (`asset_config.py` already lists `GOOGL`, `META`, `JPM`, `JNJ`, etc.)
  is mechanically easy but would require retraining the models.
