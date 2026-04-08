import pandas as pd

from src.asset_config import (
    ASSET_UNIVERSE,
    MARKET_TICKER,
    get_all_tickers,
    get_asset_class_filters,
)
from src.backtest import build_top_k_backtest
from src.data_fetch import fetch_price_data
from src.features import build_feature_dataset
from src.model import (
    CLASSIFIER_TARGET_CONFIGS,
    FEATURE_COLS,
    TARGET_CONFIGS,
    build_model_matrices,
    evaluate_prediction_frame,
    evaluate_predictions_by_ticker,
    get_model_registry,
    make_prediction_frame,
    prepare_ml_dataset,
    split_train_val_test,
    train_model,
)
from src.strategy import build_portfolio_strategy
from src.utils import (
    print_asset_class_summary,
    print_data_summary,
    print_header,
    print_model_leaderboard,
    print_portfolio_recommendation,
    print_signal_legend,
    print_subheader,
    print_training_progress,
)


# ============================================================
# CONFIGURATION
# ============================================================

# All tickers from the asset universe + the market reference index
all_tickers = get_all_tickers()
fetch_tickers = list(set(all_tickers + [MARKET_TICKER]))
model_tickers = all_tickers  # use all for modelling

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
    # Linear models
    "ridge_regression",
    "elastic_net_regression",
    "logistic_regression_classifier",
    # Gradient boosting
    "hist_gradient_boosting_regression",
    "hist_gradient_boosting_classifier",
    # Random forest
    "random_forest_regression",
    "random_forest_classifier",
    # Extra trees
    "extra_trees_regression",
    "extra_trees_classifier",
    # SVM (slow on large datasets - uncomment for small asset groups)
    # "svm_regression",
    # "svm_classifier",
    # AdaBoost
    "adaboost_regression",
    "adaboost_classifier",
    # KNN
    "knn_regression",
    "knn_classifier",
    # Stacking ensembles (slower - uses cross-validation internally)
    # "stacking_regression",
    # "stacking_classifier",
    # XGBoost (if installed)
    "xgboost_regression",
    "xgboost_classifier",
    # LightGBM (if installed)
    "lightgbm_regression",
    "lightgbm_classifier",
]

ASSET_GROUP_FILTERS = get_asset_class_filters()

START_DATE = "2015-01-01"
END_DATE = "2026-01-01"


# ============================================================
# 1. FETCH DATA
# ============================================================

data = fetch_price_data(fetch_tickers, START_DATE, END_DATE)
print_data_summary(data, all_tickers)


# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================

print_header("FEATURE ENGINEERING")

feature_df = build_feature_dataset(
    price_data=data,
    tickers=model_tickers,
    market_ticker=MARKET_TICKER,
    dropna=False,
)

print(f"\n  Features:   {len(FEATURE_COLS)}")
print(f"  Rows:       {len(feature_df):,}")
print(f"  Tickers:    {feature_df['Ticker'].nunique()}")

missing = [col for col in FEATURE_COLS if col not in feature_df.columns]
if missing:
    print(f"\n  WARNING - Missing features: {missing}")


# ============================================================
# 3. MODEL TRAINING & EVALUATION
# ============================================================

print_header("MODEL TRAINING & EVALUATION")

results = []
all_test_predictions = []
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

    print_subheader(f"Target: {target_name} ({task_type})")

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
            continue

        train_df, val_df, test_df = split_train_val_test(
            group_df,
            train_end="2021-12-31",
            val_end="2023-12-31",
        )

        print(f"\n  Group: {group_name} | Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

        (
            X_train, y_train_reg, y_train_cls, meta_train,
            X_val, y_val_reg, y_val_cls, meta_val,
            X_test, y_test_reg, y_test_cls, meta_test,
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
                continue

            if model_info["task_type"] != task_type:
                continue

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

            # Tag predictions with target and group for strategy aggregation
            test_predictions["target_name"] = target_name
            test_predictions["group_name"] = group_name
            all_test_predictions.append(test_predictions)

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

            print_training_progress(
                target=target_name,
                group=group_name,
                model=model_name,
                val_metrics=val_backtest_metrics,
                test_metrics=test_backtest_metrics,
            )


# ============================================================
# 4. RESULTS & STRATEGY
# ============================================================

results_df = (
    pd.DataFrame(results)
    .sort_values(["val_backtest_sharpe", "val_top_pick_hit_rate"], ascending=[False, False])
    .reset_index(drop=True)
)

print_model_leaderboard(results_df)
print_asset_class_summary(results_df)

strategy = build_portfolio_strategy(
    all_predictions=all_test_predictions,
    results_df=results_df,
    risk_profile="moderate",
)

print_portfolio_recommendation(strategy)
print_signal_legend()
