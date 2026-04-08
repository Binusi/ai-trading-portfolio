import numpy as np
import pandas as pd

from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.linear_model import ElasticNet, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


FEATURE_COLS = [
    "return_1d",
    "return_5d",
    "return_10d",
    "return_21d",
    "return_63d",
    "ma_10_gap",
    "ma_20_gap",
    "ma_50_gap",
    "ma_200_gap",
    "volatility_21d",
    "volatility_63d",
    "downside_vol_21d",
    "downside_vol_63d",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "intraday_range",
    "open_close_change",
    "distance_from_20d_high",
    "distance_from_20d_low",
    "drawdown_63d",
    "volume_ratio_20",
    "volume_zscore_20",
    "sharpe_like_63d",
    "market_return_1d",
    "market_return_5d",
    "market_return_21d",
    "market_volatility_21d",
    "market_ma_50_gap",
    "market_ma_200_gap",
    "market_drawdown_63d",
    "beta_90d",
    "return_5d_xs_z",
    "return_21d_xs_z",
    "return_63d_xs_z",
    "volatility_21d_xs_z",
    "rsi_14_xs_z",
    "ma_50_gap_xs_z",
    "sharpe_like_63d_xs_z",
    "volume_ratio_20_xs_z",
]


TARGET_CONFIGS = {
    "target_5d_risk_adj_return": {
        "task_type": "regression",
        "target_col": "target_5d_risk_adj_return",
        "label_col": "target_5d_risk_adj_direction",
        "backtest_return_col": "target_5d_return",
        "rebalance_every_n_days": 5,
    },
    "target_5d_return": {
        "task_type": "regression",
        "target_col": "target_5d_return",
        "label_col": "target_5d_direction",
        "backtest_return_col": "target_5d_return",
        "rebalance_every_n_days": 5,
    },
    "target_10d_return": {
        "task_type": "regression",
        "target_col": "target_10d_return",
        "label_col": "target_10d_direction",
        "backtest_return_col": "target_10d_return",
        "rebalance_every_n_days": 10,
    },
    "target_cross_sectional_5d_return": {
        "task_type": "regression",
        "target_col": "target_cross_sectional_5d_return",
        "label_col": "target_cross_sectional_5d_direction",
        "backtest_return_col": "target_5d_return",
        "rebalance_every_n_days": 5,
    },
}


CLASSIFIER_TARGET_CONFIGS = {
    "target_5d_direction": {
        "task_type": "classification",
        "target_col": "target_5d_return",
        "label_col": "target_5d_direction",
        "backtest_return_col": "target_5d_return",
        "rebalance_every_n_days": 5,
    },
    "target_10d_direction": {
        "task_type": "classification",
        "target_col": "target_10d_return",
        "label_col": "target_10d_direction",
        "backtest_return_col": "target_10d_return",
        "rebalance_every_n_days": 10,
    },
    "target_cross_sectional_5d_direction": {
        "task_type": "classification",
        "target_col": "target_cross_sectional_5d_return",
        "label_col": "target_cross_sectional_5d_direction",
        "backtest_return_col": "target_5d_return",
        "rebalance_every_n_days": 5,
    },
}


OPTIONAL_IMPORTS = {
    "xgboost": None,
    "lightgbm": None,
}

try:
    from xgboost import XGBClassifier, XGBRegressor

    OPTIONAL_IMPORTS["xgboost"] = (XGBClassifier, XGBRegressor)
except Exception:
    pass

try:
    from lightgbm import LGBMClassifier, LGBMRegressor

    OPTIONAL_IMPORTS["lightgbm"] = (LGBMClassifier, LGBMRegressor)
except Exception:
    pass


def get_model_registry() -> dict:
    models = {
        "ridge_regression": {
            "task_type": "regression",
            "builder": lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
        },
        "elastic_net_regression": {
            "task_type": "regression",
            "builder": lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", ElasticNet(alpha=0.0005, l1_ratio=0.2, max_iter=20000)),
                ]
            ),
        },
        "hist_gradient_boosting_regression": {
            "task_type": "regression",
            "builder": lambda: HistGradientBoostingRegressor(
                learning_rate=0.03,
                max_depth=4,
                max_iter=250,
                min_samples_leaf=30,
                random_state=42,
            ),
        },
        "random_forest_regression": {
            "task_type": "regression",
            "builder": lambda: RandomForestRegressor(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
            ),
        },
        "logistic_regression_classifier": {
            "task_type": "classification",
            "builder": lambda: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("model", LogisticRegression(max_iter=5000, class_weight="balanced")),
                ]
            ),
        },
        "hist_gradient_boosting_classifier": {
            "task_type": "classification",
            "builder": lambda: HistGradientBoostingClassifier(
                learning_rate=0.03,
                max_depth=4,
                max_iter=250,
                min_samples_leaf=30,
                random_state=42,
            ),
        },
        "random_forest_classifier": {
            "task_type": "classification",
            "builder": lambda: RandomForestClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        },
        "svm_regression": {
            "task_type": "regression",
            "builder": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVR(kernel="rbf", C=1.0)),
            ]),
        },
        "svm_classifier": {
            "task_type": "classification",
            "builder": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, class_weight="balanced")),
            ]),
        },
        "extra_trees_regression": {
            "task_type": "regression",
            "builder": lambda: ExtraTreesRegressor(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1,
            ),
        },
        "extra_trees_classifier": {
            "task_type": "classification",
            "builder": lambda: ExtraTreesClassifier(
                n_estimators=400,
                max_depth=8,
                min_samples_leaf=20,
                class_weight="balanced_subsample",
                random_state=42,
                n_jobs=-1,
            ),
        },
        "adaboost_regression": {
            "task_type": "regression",
            "builder": lambda: AdaBoostRegressor(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42,
            ),
        },
        "adaboost_classifier": {
            "task_type": "classification",
            "builder": lambda: AdaBoostClassifier(
                n_estimators=200,
                learning_rate=0.05,
                random_state=42,
            ),
        },
        "knn_regression": {
            "task_type": "regression",
            "builder": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=20)),
            ]),
        },
        "knn_classifier": {
            "task_type": "classification",
            "builder": lambda: Pipeline([
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=20)),
            ]),
        },
        "stacking_regression": {
            "task_type": "regression",
            "builder": lambda: StackingRegressor(
                estimators=[
                    ("ridge", Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])),
                    ("hgb", HistGradientBoostingRegressor(max_depth=4, max_iter=150, random_state=42)),
                ],
                final_estimator=Ridge(alpha=1.0),
                n_jobs=-1,
            ),
        },
        "stacking_classifier": {
            "task_type": "classification",
            "builder": lambda: StackingClassifier(
                estimators=[
                    ("lr", Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=5000))])),
                    ("hgb", HistGradientBoostingClassifier(max_depth=4, max_iter=150, random_state=42)),
                ],
                final_estimator=LogisticRegression(max_iter=5000),
                n_jobs=-1,
            ),
        },
    }

    if OPTIONAL_IMPORTS["xgboost"] is not None:
        XGBClassifier, XGBRegressor = OPTIONAL_IMPORTS["xgboost"]
        models["xgboost_regression"] = {
            "task_type": "regression",
            "builder": lambda: XGBRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                random_state=42,
            ),
        }
        models["xgboost_classifier"] = {
            "task_type": "classification",
            "builder": lambda: XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
            ),
        }

    if OPTIONAL_IMPORTS["lightgbm"] is not None:
        LGBMClassifier, LGBMRegressor = OPTIONAL_IMPORTS["lightgbm"]
        models["lightgbm_regression"] = {
            "task_type": "regression",
            "builder": lambda: LGBMRegressor(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            ),
        }
        models["lightgbm_classifier"] = {
            "task_type": "classification",
            "builder": lambda: LGBMClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            ),
        }

    return models



def prepare_ml_dataset(
    feature_df: pd.DataFrame,
    target_col: str,
    label_col: str,
    min_history: int = 252,
) -> pd.DataFrame:
    df = feature_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    df["asset_history_days"] = df.groupby("Ticker").cumcount() + 1
    df = df[df["asset_history_days"] >= min_history].copy()

    required_cols = list(
        dict.fromkeys(FEATURE_COLS + [target_col, label_col, "target_5d_return", "target_10d_return"])
    )
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    keep_cols = list(
        dict.fromkeys(
            [
                "Date",
                "Ticker",
                "AssetGroup",
                *FEATURE_COLS,
                target_col,
                label_col,
                "target_5d_return",
                "target_10d_return",
            ]
        )
    )
    return df[keep_cols].copy()



def split_train_val_test(
    df: pd.DataFrame,
    train_end: str = "2021-12-31",
    val_end: str = "2023-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    train_df = df[df["Date"] <= pd.Timestamp(train_end)].copy()
    val_df = df[(df["Date"] > pd.Timestamp(train_end)) & (df["Date"] <= pd.Timestamp(val_end))].copy()
    test_df = df[df["Date"] > pd.Timestamp(val_end)].copy()

    return train_df, val_df, test_df



def _make_one_matrix(split_df: pd.DataFrame, target_col: str, label_col: str):
    X_numeric = split_df[FEATURE_COLS].copy()

    X_ticker = pd.get_dummies(
        split_df["Ticker"],
        prefix="ticker",
        drop_first=True,
        dtype=float,
    )

    X = pd.concat([X_numeric.reset_index(drop=True), X_ticker.reset_index(drop=True)], axis=1)
    y_reg = split_df[target_col]
    if isinstance(y_reg, pd.DataFrame):
        y_reg = y_reg.iloc[:, 0]
    y_reg = y_reg.reset_index(drop=True)

    y_cls = split_df[label_col]
    if isinstance(y_cls, pd.DataFrame):
        y_cls = y_cls.iloc[:, 0]
    y_cls = y_cls.astype(int).reset_index(drop=True)
    meta = split_df[["Date", "Ticker", "AssetGroup", "target_5d_return", "target_10d_return"]].reset_index(drop=True)
    return X, y_reg, y_cls, meta



def build_model_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str,
    label_col: str,
):
    X_train, y_train_reg, y_train_cls, meta_train = _make_one_matrix(train_df, target_col, label_col)
    X_val, y_val_reg, y_val_cls, meta_val = _make_one_matrix(val_df, target_col, label_col)
    X_test, y_test_reg, y_test_cls, meta_test = _make_one_matrix(test_df, target_col, label_col)

    all_columns = X_train.columns.union(X_val.columns).union(X_test.columns)

    X_train = X_train.reindex(columns=all_columns, fill_value=0.0)
    X_val = X_val.reindex(columns=all_columns, fill_value=0.0)
    X_test = X_test.reindex(columns=all_columns, fill_value=0.0)

    return (
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
    )



def train_model(model_name: str, X_train: pd.DataFrame, y_train: pd.Series):
    model_registry = get_model_registry()
    model = model_registry[model_name]["builder"]()
    model.fit(X_train, y_train)
    return model



def make_prediction_frame(
    model,
    model_name: str,
    task_type: str,
    X: pd.DataFrame,
    y_reg: pd.Series,
    y_cls: pd.Series,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    out = meta.copy()
    out["model_name"] = model_name
    out["task_type"] = task_type
    out["actual_target"] = y_reg.values
    out["actual_label"] = y_cls.astype(int).values

    if task_type == "regression":
        prediction = model.predict(X)
        out["prediction"] = prediction
        out["predicted_label"] = (out["prediction"] > 0).astype(int)
        out["score"] = out["prediction"]
    else:
        probabilities = model.predict_proba(X)[:, 1]
        out["prediction"] = probabilities
        out["predicted_label"] = (probabilities >= 0.5).astype(int)
        out["score"] = probabilities - 0.5

    return out



def evaluate_prediction_frame(prediction_df: pd.DataFrame) -> dict:
    task_type = prediction_df["task_type"].iloc[0]

    if task_type == "regression":
        y_true = prediction_df["actual_target"]
        y_pred = prediction_df["prediction"]

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        corr = float(y_true.corr(y_pred)) if len(y_true) > 1 else np.nan
        directional_accuracy = float(((y_true > 0) == (y_pred > 0)).mean())

        return {
            "rmse": rmse,
            "mae": mae,
            "correlation": corr,
            "directional_accuracy": directional_accuracy,
        }

    accuracy = float(accuracy_score(prediction_df["actual_label"], prediction_df["predicted_label"]))
    return {
        "classification_accuracy": accuracy,
        "directional_accuracy": accuracy,
    }



def evaluate_predictions_by_ticker(prediction_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for ticker, group in prediction_df.groupby("Ticker"):
        task_type = group["task_type"].iloc[0]

        row = {
            "Ticker": ticker,
            "rows": len(group),
            "avg_realized_5d_return": float(group["target_5d_return"].mean()),
            "avg_score": float(group["score"].mean()),
        }

        if task_type == "regression":
            y_true = group["actual_target"].reset_index(drop=True)
            y_pred = group["prediction"].reset_index(drop=True)
            row.update(
                {
                    "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                    "mae": float(mean_absolute_error(y_true, y_pred)),
                    "correlation": float(y_true.corr(y_pred)) if len(y_true) > 1 else np.nan,
                    "directional_accuracy": float(((y_true > 0) == (y_pred > 0)).mean()),
                }
            )
        else:
            row.update(
                {
                    "classification_accuracy": float(
                        accuracy_score(group["actual_label"], group["predicted_label"])
                    ),
                    "directional_accuracy": float(
                        accuracy_score(group["actual_label"], group["predicted_label"])
                    ),
                }
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)
