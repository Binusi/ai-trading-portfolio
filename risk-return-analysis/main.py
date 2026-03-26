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
