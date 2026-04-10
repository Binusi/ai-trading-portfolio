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
from src.strategy import build_allocation_timeline
from src.simulation import run_simulation, compute_simulation_metrics
from src.visualization import generate_all_plots
from src.utils import format_metrics_table


# ---------------------------- CHOOSE ASSETS

tickers = ["AAPL", "MSFT", "SPY", "^GSPC", "TLT", "NVDA", "PATH"]
model_tickers = ["AAPL", "MSFT", "SPY", "TLT", "NVDA"]  # leave PATH out of first serious benchmark

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
# PORTFOLIO ALLOCATION & TRADE SIMULATION
# ============================================================================

print("\n" + "=" * 120)
print("PORTFOLIO ALLOCATION & TRADE SIMULATION")
print("=" * 120)

# Select best model by validation Sharpe, preferring all_assets for meaningful
# cross-asset-class allocation (equity vs bonds vs ETFs)
all_assets_results = results_df[results_df["group_name"] == "all_assets"]
if not all_assets_results.empty:
    best_row = all_assets_results.iloc[0]
else:
    best_row = results_df.iloc[0]
best_target = best_row["target_name"]
best_group = best_row["group_name"]
best_model = best_row["model_name"]

print(f"\nBest model: {best_model}")
print(f"Target: {best_target}")
print(f"Asset group: {best_group}")
print(f"Validation Sharpe: {best_row.get('val_backtest_sharpe', 'N/A')}")

# Get the stored test predictions for the best model
best_key = (best_target, best_group, best_model)
best_test_predictions = all_test_predictions[best_key]

# Get rebalance frequency from target config
rebalance_n = all_target_configs[best_target]["rebalance_every_n_days"]

# Build allocation timeline
print("\nBuilding portfolio allocation timeline...")
allocation = build_allocation_timeline(
    prediction_df=best_test_predictions,
    feature_df=feature_df,
    rebalance_every_n_days=rebalance_n,
)

print(f"Allocation timeline: {len(allocation)} entries across {allocation['Date'].nunique()} rebalance dates")
print("\nSample allocation (first rebalance date):")
first_date = allocation["Date"].min()
print(allocation[allocation["Date"] == first_date][["Ticker", "AssetGroup", "group_weight", "ticker_weight"]])

# Run trade simulation
print("\nRunning trade simulation ($1,000 initial capital, 2024-01-01 to 2025-12-31)...")
daily_log, trade_log = run_simulation(
    allocation_timeline=allocation,
    price_data=data,
    tickers=model_tickers,
    start_date="2024-01-01",
    end_date="2025-12-31",
    initial_capital=1000.0,
    rebalance_every_n_days=rebalance_n,
    transaction_cost_bps=10.0,
)

# Compute and print metrics
sim_metrics = compute_simulation_metrics(daily_log, initial_capital=1000.0)
if not trade_log.empty:
    sim_metrics["total_trades"] = len(trade_log)
    sim_metrics["total_transaction_costs"] = float(trade_log["transaction_cost"].sum())
print("\n" + format_metrics_table(sim_metrics))

# Print trade summary
if not trade_log.empty:
    print(f"\nTotal trades executed: {len(trade_log)}")
    print(f"Total transaction costs: ${trade_log['transaction_cost'].sum():.2f}")
    print("\nTrade log sample (first 20):")
    print(trade_log.head(20).to_string(index=False))

# Generate benchmark for comparison (SPY buy & hold)
benchmark_prices = None
try:
    spy_close = data["Close"]["SPY"]
    sim_dates = pd.to_datetime(daily_log["Date"])
    benchmark_prices = spy_close.loc[
        (spy_close.index >= sim_dates.min()) & (spy_close.index <= sim_dates.max())
    ]
except (KeyError, TypeError):
    pass

# Generate all plots
print("\nGenerating visualizations...")
output_dir = "output"
saved_plots = generate_all_plots(
    daily_log=daily_log,
    trade_log=trade_log,
    tickers=model_tickers,
    benchmark_prices=benchmark_prices,
    initial_capital=1000.0,
    output_dir=output_dir,
)
print(f"Saved {len(saved_plots)} plots to {output_dir}/:")
for p in saved_plots:
    print(f"  - {p}")
