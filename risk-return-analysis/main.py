import argparse

import pandas as pd

from src.backtest import build_top_k_backtest
from src.data_fetch import fetch_price_data
from src.features import build_feature_dataset, get_single_ticker_df
from src.model import (
    CLASSIFIER_TARGET_CONFIGS,
    TARGET_CONFIGS,
    build_model_matrices,
    evaluate_prediction_frame,
    evaluate_predictions_by_ticker,
    get_model_registry,
    prepare_ml_dataset,
    split_train_val_test,
    train_model,
    make_prediction_frame,
    FEATURE_COLS,
)
from src.profile_strategy import (
    build_profile_allocation_timeline,
    get_quarterly_rebalance_dates,
)
from src.profiles import (
    DEFAULT_EQUITY_UNIVERSE,
    INDEX_PROXIES,
    RISK_PROFILES,
    get_profile,
    get_required_tickers,
)
from src.deposits import DepositSchedule
from src.export_app_data import export_app_data
from src.simulation import run_simulation, compute_simulation_metrics
from src.utils import format_metrics_table, get_trading_days
from src.visualization import generate_all_plots


def _parse_cli_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the AI-driven trade simulation across risk profiles.",
    )
    parser.add_argument(
        "--initial-capital", type=float, default=1000.0,
        help="Starting cash balance in dollars (default: 1000).",
    )
    parser.add_argument(
        "--deposit-amount", type=float, default=0.0,
        help="Recurring deposit in dollars; 0 disables deposits (default: 0).",
    )
    parser.add_argument(
        "--deposit-every-months", type=int, default=1,
        choices=[1, 2, 3, 6, 12],
        help="Deposit frequency in months (default: 1).",
    )
    parser.add_argument(
        "--deposit-day", type=str, default="1",
        choices=["1", "15", "eom"],
        help="Day of month to deposit on: 1, 15, or 'eom' (default: 1).",
    )
    return parser.parse_args()


CLI_ARGS = _parse_cli_args()


# ---------------------------- CHOOSE ASSETS

# Tickers for ML training (individual stocks we want to tilt across)
model_tickers = ["AAPL", "MSFT", "SPY", "TLT", "NVDA"]

# Equity sleeve for the profile system (subset of model_tickers)
equity_universe = list(DEFAULT_EQUITY_UNIVERSE)

# Full fetch list: ML training tickers + index proxies needed by profiles
# (GLD, EFA, etc. are held passively as index proxies — no ML on them)
simulation_tickers = get_required_tickers(equity_universe)
tickers = sorted(set(["^GSPC"] + model_tickers + simulation_tickers))

TARGETS_TO_RUN = [
    "target_5d_risk_adj_return",
    "target_5d_return",
    "target_10d_return",
    "target_cross_sectional_5d_return",
]

CLASSIFIER_TARGETS_TO_RUN = [
    "target_5d_direction",
    "target_10d_direction",
    "target_cross_sectional_5d_direction",
]

MODEL_NAMES_TO_RUN = [
    "ridge_regression",
    "elastic_net_regression",
    "hist_gradient_boosting_regression",
    "random_forest_regression",
    "logistic_regression_classifier",
    "hist_gradient_boosting_classifier",
    "random_forest_classifier",
    # "xgboost_regression",
    # "xgboost_classifier",
    # "lightgbm_regression",
    # "lightgbm_classifier",
]

ASSET_GROUP_FILTERS = {
    "all_assets": None,
    "equity_only": "equity",
}

start_date = "2015-01-01"
end_date = "2026-01-01"

# ---------------------------- GET DATA

data = fetch_price_data(tickers, start_date, end_date)

print("\nRAW DATA SHAPE:")
print(data.shape)

print("\nFIRST VALID CLOSE DATE FOR EACH TICKER:")
print(data["Close"].apply(lambda x: x.first_valid_index()))

example_df = get_single_ticker_df(data, "AAPL")
print("\nAAPL SINGLE-TICKER DATA HEAD:")
print(example_df.head())

# ---------------------------- FEATURE ENGINEERING

feature_df = build_feature_dataset(
    price_data=data,
    tickers=model_tickers,
    market_ticker="^GSPC",
    dropna=False,
)

print("\nFEATURE DATA SHAPE BEFORE ML CLEANING:")
print(feature_df.shape)
print("\nFEATURE DATA COLUMNS:")
print(feature_df.columns.tolist())

missing_feature_cols = [col for col in FEATURE_COLS if col not in feature_df.columns]
print("\nMISSING FEATURE_COLS FROM FEATURE DATA:")
print(missing_feature_cols)

results = []
all_test_predictions = {}
registry = get_model_registry()

all_target_configs = {}
all_target_configs.update(TARGET_CONFIGS)
all_target_configs.update(CLASSIFIER_TARGET_CONFIGS)

for target_name in TARGETS_TO_RUN + CLASSIFIER_TARGETS_TO_RUN:
    target_config = all_target_configs[target_name]
    task_type = target_config["task_type"]
    target_col = target_config["target_col"]
    label_col = target_config["label_col"]
    realized_return_col = target_config["backtest_return_col"]
    rebalance_every_n_days = target_config["rebalance_every_n_days"]

    print(f"\n{'=' * 100}")
    print(f"RUNNING TARGET: {target_name}")
    print(f"TASK TYPE: {task_type}")

    ml_df = prepare_ml_dataset(
        feature_df=feature_df,
        target_col=target_col,
        label_col=label_col,
        min_history=252,
    )

    for group_name, group_filter in ASSET_GROUP_FILTERS.items():
        group_df = ml_df.copy()
        if group_filter is not None:
            group_df = group_df[group_df["AssetGroup"] == group_filter].copy()

        if group_df["Ticker"].nunique() < 2:
            print(f"Skipping {group_name} because it has fewer than 2 tickers.")
            continue

        train_df, val_df, test_df = split_train_val_test(
            group_df,
            train_end="2021-12-31",
            val_end="2023-12-31",
        )

        print(f"\nGROUP: {group_name}")
        print("TRAIN / VAL / TEST SHAPES:", train_df.shape, val_df.shape, test_df.shape)

        (
            X_train,
            y_train_reg,
            y_train_cls,
            meta_train,
            X_val,
            y_val_reg,
            y_val_cls,
            meta_val,
            X_test,
            y_test_reg,
            y_test_cls,
            meta_test,
        ) = build_model_matrices(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            target_col=target_col,
            label_col=label_col,
        )

        for model_name in MODEL_NAMES_TO_RUN:
            model_info = registry.get(model_name)
            if model_info is None:
                print(f"Skipping {model_name} because library is not installed.")
                continue

            if model_info["task_type"] != task_type:
                continue

            print(f"Training {model_name} ...")
            y_train = y_train_reg if task_type == "regression" else y_train_cls
            model = train_model(model_name, X_train, y_train)

            val_predictions = make_prediction_frame(
                model=model,
                model_name=model_name,
                task_type=task_type,
                X=X_val,
                y_reg=y_val_reg,
                y_cls=y_val_cls,
                meta=meta_val,
            )
            test_predictions = make_prediction_frame(
                model=model,
                model_name=model_name,
                task_type=task_type,
                X=X_test,
                y_reg=y_test_reg,
                y_cls=y_test_cls,
                meta=meta_test,
            )

            val_ml_metrics = evaluate_prediction_frame(val_predictions)
            test_ml_metrics = evaluate_prediction_frame(test_predictions)

            val_backtest_df, val_backtest_metrics = build_top_k_backtest(
                prediction_df=val_predictions,
                realized_return_col=realized_return_col,
                top_k=2,
                rebalance_every_n_days=rebalance_every_n_days,
                transaction_cost_bps=10.0,
            )
            test_backtest_df, test_backtest_metrics = build_top_k_backtest(
                prediction_df=test_predictions,
                realized_return_col=realized_return_col,
                top_k=2,
                rebalance_every_n_days=rebalance_every_n_days,
                transaction_cost_bps=10.0,
            )

            # Store test predictions for later use in simulation
            all_test_predictions[(target_name, group_name, model_name)] = test_predictions

            result_row = {
                "target_name": target_name,
                "group_name": group_name,
                "model_name": model_name,
                **{f"val_{k}": v for k, v in val_ml_metrics.items()},
                **{f"test_{k}": v for k, v in test_ml_metrics.items()},
                **{f"val_{k}": v for k, v in val_backtest_metrics.items()},
                **{f"test_{k}": v for k, v in test_backtest_metrics.items()},
            }
            results.append(result_row)

            print("VALIDATION BACKTEST METRICS:", val_backtest_metrics)
            print("TEST BACKTEST METRICS:", test_backtest_metrics)
            print("VALIDATION TICKER METRICS:")
            print(evaluate_predictions_by_ticker(val_predictions))
            print("TEST TICKER METRICS:")
            print(evaluate_predictions_by_ticker(test_predictions))


results_df = (
    __import__("pandas").DataFrame(results)
    .sort_values(["val_backtest_sharpe", "val_top_pick_hit_rate"], ascending=[False, False])
    .reset_index(drop=True)
)

print("\n" + "#" * 120)
print("FINAL MODEL COMPARISON TABLE")
print(results_df)

print("\nTOP 10 BY VALIDATION SHARPE")
print(results_df.head(10))


# ============================================================================
# PROFILE-BASED PORTFOLIO SIMULATIONS
# ============================================================================
# We run six simulations: each of {Conservative, Balanced, Aggressive} with
# and without the AI tilt. The asset-class targets are profile-defined (the
# honest, transparent part); the tilt only adjusts individual equity-name
# weights within the equity sleeve, capped at ±5%.

print("\n" + "=" * 120)
print("PROFILE-BASED PORTFOLIO SIMULATIONS")
print("=" * 120)

# Select best ML model by validation Sharpe (preferring the all-assets cohort
# so we get equity-name signals from a model that saw the broader universe).
all_assets_results = results_df[results_df["group_name"] == "all_assets"]
best_row = (all_assets_results if not all_assets_results.empty else results_df).iloc[0]
best_target = best_row["target_name"]
best_group = best_row["group_name"]
best_model = best_row["model_name"]

print(f"\nML model used for AI tilt signal:")
print(f"  Model: {best_model}")
print(f"  Target: {best_target}")
print(f"  Asset group: {best_group}")
print(f"  Validation Sharpe: {best_row.get('val_backtest_sharpe', 'N/A')}")

best_test_predictions = all_test_predictions[(best_target, best_group, best_model)]

SIM_START = "2024-01-01"
SIM_END = "2025-12-31"
INITIAL_CAPITAL = float(CLI_ARGS.initial_capital)
TILT_CAP = 0.05

if CLI_ARGS.deposit_amount > 0:
    deposit_day_arg = (
        "EOM" if CLI_ARGS.deposit_day.lower() == "eom" else int(CLI_ARGS.deposit_day)
    )
    DEPOSIT_SCHEDULE = DepositSchedule(
        amount=float(CLI_ARGS.deposit_amount),
        period_months=int(CLI_ARGS.deposit_every_months),
        day_of_month=deposit_day_arg,
    )
else:
    DEPOSIT_SCHEDULE = None

trading_days = get_trading_days(SIM_START, SIM_END, data)
quarterly_dates = get_quarterly_rebalance_dates(trading_days)
print(f"\nSimulation window: {SIM_START} to {SIM_END}")
print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")
if DEPOSIT_SCHEDULE is not None:
    every = DEPOSIT_SCHEDULE.period_months
    every_label = "month" if every == 1 else f"{every} months"
    print(
        f"Deposits: ${DEPOSIT_SCHEDULE.amount:,.2f} every {every_label} "
        f"on day {DEPOSIT_SCHEDULE.day_of_month}"
    )
print(f"Quarterly rebalance dates: {len(quarterly_dates)}")

profile_results: dict[tuple[str, bool], dict] = {}

for profile_key in ["conservative", "balanced", "aggressive"]:
    profile = get_profile(profile_key)
    for use_tilt in [False, True]:
        label = f"{profile.name} ({'tilt' if use_tilt else 'no tilt'})"
        print(f"\n--- Running {label} ---")

        allocation, rationale_log = build_profile_allocation_timeline(
            profile=profile,
            trading_days=trading_days,
            equity_tickers=equity_universe,
            prediction_df=best_test_predictions if use_tilt else None,
            tilt_cap=TILT_CAP,
            use_tilt=use_tilt,
        )

        daily_log, trade_log = run_simulation(
            allocation_timeline=allocation,
            price_data=data,
            tickers=simulation_tickers,
            start_date=SIM_START,
            end_date=SIM_END,
            initial_capital=INITIAL_CAPITAL,
            transaction_cost_bps=10.0,
            rebalance_dates=quarterly_dates,
            deposit_schedule=DEPOSIT_SCHEDULE,
        )

        metrics = compute_simulation_metrics(daily_log, initial_capital=INITIAL_CAPITAL)
        if not trade_log.empty:
            metrics["total_trades"] = len(trade_log)
            metrics["total_transaction_costs"] = float(trade_log["transaction_cost"].sum())

        profile_results[(profile_key, use_tilt)] = {
            "profile": profile,
            "use_tilt": use_tilt,
            "allocation": allocation,
            "daily_log": daily_log,
            "trade_log": trade_log,
            "rationale_log": rationale_log,
            "metrics": metrics,
        }

        contrib_part = (
            f"Contributions: ${metrics['total_contributions']:.2f}  "
            if DEPOSIT_SCHEDULE is not None
            else ""
        )
        print(f"  Final value: ${metrics['final_portfolio_value']:.2f}  "
              f"{contrib_part}"
              f"Total return: {metrics['total_return']:.2%}  "
              f"Sharpe: {metrics.get('sharpe_ratio', float('nan')):.2f}  "
              f"Max DD: {metrics['max_drawdown']:.2%}  "
              f"Trades: {metrics.get('total_trades', 0)}")

# ----------------------------- Summary comparison

print("\n" + "=" * 120)
print("PROFILE COMPARISON SUMMARY")
print("=" * 120)
deposit_part = ""
if DEPOSIT_SCHEDULE is not None:
    every = DEPOSIT_SCHEDULE.period_months
    every_label = "month" if every == 1 else f"{every} months"
    deposit_part = (
        f", plus ${DEPOSIT_SCHEDULE.amount:.0f} every {every_label} "
        f"on day {DEPOSIT_SCHEDULE.day_of_month}"
    )
print(f"\nAll figures based on ${INITIAL_CAPITAL:.0f} initial capital{deposit_part}, "
      f"{SIM_START} → {SIM_END}, quarterly rebalancing.\n")

header = f"{'Profile':<14s} {'Tilt':<6s} {'Final $':>10s} {'Total':>9s} {'Annual':>9s} {'Sharpe':>7s} {'Max DD':>8s} {'Trades':>7s}"
print(header)
print("-" * len(header))
for (profile_key, use_tilt), r in profile_results.items():
    m = r["metrics"]
    print(
        f"{r['profile'].name:<14s} "
        f"{'AI' if use_tilt else 'none':<6s} "
        f"${m['final_portfolio_value']:>9.2f} "
        f"{m['total_return']:>8.2%} "
        f"{m['annualized_return']:>8.2%} "
        f"{m.get('sharpe_ratio', float('nan')):>7.2f} "
        f"{m['max_drawdown']:>7.2%} "
        f"{m.get('total_trades', 0):>7d}"
    )

# ----------------------------- Sample rationales

print("\n" + "=" * 120)
print("SAMPLE RATIONALES (Balanced profile + AI tilt)")
print("=" * 120)
sample = profile_results[("balanced", True)]["rationale_log"]
for entry in sample[:3]:
    print(f"\n[{entry['date'].date()}] {entry['rationale_text']}")

# ----------------------------- Plots for the default profile

print("\n" + "=" * 120)
print("GENERATING PLOTS (Balanced + AI tilt)")
print("=" * 120)

default_run = profile_results[("balanced", True)]
benchmark_prices = None
try:
    spy_close = data["Close"]["SPY"]
    sim_dates = pd.to_datetime(default_run["daily_log"]["Date"])
    benchmark_prices = spy_close.loc[
        (spy_close.index >= sim_dates.min()) & (spy_close.index <= sim_dates.max())
    ]
except (KeyError, TypeError):
    pass

output_dir = "output"
saved_plots = generate_all_plots(
    daily_log=default_run["daily_log"],
    trade_log=default_run["trade_log"],
    tickers=simulation_tickers,
    benchmark_prices=benchmark_prices,
    initial_capital=INITIAL_CAPITAL,
    output_dir=output_dir,
)
print(f"\nSaved {len(saved_plots)} plots to {output_dir}/:")
for p in saved_plots:
    print(f"  - {p}")

# ----------------------------- Export JSON for the Expo app

print("\n" + "=" * 120)
print("EXPORTING APP DATA")
print("=" * 120)

actual_start = pd.to_datetime(default_run["daily_log"]["Date"]).min()
actual_end = pd.to_datetime(default_run["daily_log"]["Date"]).max()

simulation_config = {
    "start_date": actual_start.strftime("%Y-%m-%d"),
    "end_date": actual_end.strftime("%Y-%m-%d"),
    "initial_capital": INITIAL_CAPITAL,
    "transaction_cost_bps": 10.0,
    "rebalance_cadence": "quarterly",
    "tilt_cap_pct": TILT_CAP * 100.0,
    "equity_universe": equity_universe,
    "simulation_tickers": simulation_tickers,
}
ml_model_info = {
    "model_name": best_model,
    "target": best_target,
    "asset_group": best_group,
    "validation_sharpe": float(best_row.get("val_backtest_sharpe", float("nan")))
        if best_row.get("val_backtest_sharpe") is not None
        else None,
}

app_data_dir = "../app/assets/data"

if DEPOSIT_SCHEDULE is not None:
    # The app expects exports to be the canonical $1,000 lump-sum baseline so
    # it can reconstruct any user-chosen initial + deposit schedule on the
    # client. Writing deposit-mode totals here would break that contract.
    print(
        "\nSkipping JSON export: --deposit-amount is set, but the app expects "
        "the canonical $1,000 lump-sum baseline. Re-run main.py without "
        "--deposit-amount to refresh the exports."
    )
else:
    written_files = export_app_data(
        profile_results=profile_results,
        price_data=data,
        ml_model_info=ml_model_info,
        simulation_config=simulation_config,
        output_dir=app_data_dir,
        default_profile_key="balanced",
        default_use_tilt=False,
    )
    print(f"\nWrote {len(written_files)} JSON files to {app_data_dir}:")
    for p in written_files:
        print(f"  - {p.name}  ({p.stat().st_size / 1024:.1f} KB)")
