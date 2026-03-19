import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


FEATURE_COLS = [
    "return_1d",
    "return_5d",
    "return_21d",
    "ma_10_gap",
    "ma_20_gap",
    "ma_50_gap",
    "ma_200_gap",
    "volatility_21d",
    "volatility_63d",
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_hist",
    "intraday_range",
    "open_close_change",
    "volume_ratio_20",
    "sharpe_like_63d",
    "market_return_1d",
    "market_return_5d",
    "market_volatility_21d",
    "beta_90d",
]

TARGET_COL = "target_next_return"


def prepare_ml_dataset(feature_df: pd.DataFrame, min_history: int = 252) -> pd.DataFrame:
    """
    Final cleanup before train/validation/test splitting.

    What this does:
    1. Sort by ticker and date
    2. Keep only rows where each ticker has enough own history
    3. Drop rows that still have missing model features or missing target
    """
    df = feature_df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Ticker", "Date"]).reset_index(drop=True)

    # Count how many observations each asset has accumulated so far
    df["asset_history_days"] = df.groupby("Ticker").cumcount() + 1

    # Keep only rows where the asset already has enough history
    df = df[df["asset_history_days"] >= min_history].copy()

    # Drop rows where model inputs or target are missing
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)

    keep_cols = ["Date", "Ticker"] + FEATURE_COLS + [TARGET_COL, "target_next_direction"]
    return df[keep_cols].copy()


def split_train_val_test(
    df: pd.DataFrame,
    train_end: str = "2021-12-31",
    val_end: str = "2023-12-31",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological split:
    - Train: everything up to train_end
    - Validation: after train_end up to val_end
    - Test: after val_end
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    train_df = df[df["Date"] <= pd.Timestamp(train_end)].copy()
    val_df = df[(df["Date"] > pd.Timestamp(train_end)) & (df["Date"] <= pd.Timestamp(val_end))].copy()
    test_df = df[df["Date"] > pd.Timestamp(val_end)].copy()

    return train_df, val_df, test_df


def _make_one_matrix(split_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Build X, y, and metadata for one split.
    We add ticker dummies so one pooled model can still learn ticker-specific behavior.
    """
    X_numeric = split_df[FEATURE_COLS].copy()

    # Add ticker dummy variables
    X_ticker = pd.get_dummies(
        split_df["Ticker"],
        prefix="ticker",
        drop_first=True,
        dtype=float,
    )

    X = pd.concat([X_numeric.reset_index(drop=True), X_ticker.reset_index(drop=True)], axis=1)
    y = split_df[TARGET_COL].reset_index(drop=True)
    meta = split_df[["Date", "Ticker"]].reset_index(drop=True)

    return X, y, meta


def build_model_matrices(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
):
    """
    Build aligned X/y matrices for train, validation, and test.
    """
    X_train, y_train, meta_train = _make_one_matrix(train_df)
    X_val, y_val, meta_val = _make_one_matrix(val_df)
    X_test, y_test, meta_test = _make_one_matrix(test_df)

    # Make sure all splits have the same columns
    all_columns = X_train.columns.union(X_val.columns).union(X_test.columns)

    X_train = X_train.reindex(columns=all_columns, fill_value=0.0)
    X_val = X_val.reindex(columns=all_columns, fill_value=0.0)
    X_test = X_test.reindex(columns=all_columns, fill_value=0.0)

    return (
        X_train,
        y_train,
        meta_train,
        X_val,
        y_val,
        meta_val,
        X_test,
        y_test,
        meta_test,
    )


def train_linear_model(X_train: pd.DataFrame, y_train: pd.Series) -> LinearRegression:
    """
    First baseline model.
    This is NOT the final trading model.
    It is just the first clean baseline to prove the pipeline works.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def make_prediction_frame(
    model: LinearRegression,
    X: pd.DataFrame,
    y: pd.Series,
    meta: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return predictions together with date/ticker info.
    """
    predictions = model.predict(X)

    out = meta.copy()
    out["actual_next_return"] = y.values
    out["predicted_next_return"] = predictions
    out["actual_direction"] = (out["actual_next_return"] > 0).astype(int)
    out["predicted_direction"] = (out["predicted_next_return"] > 0).astype(int)

    return out


def evaluate_regression(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """
    Simple regression evaluation metrics.
    """
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

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

def evaluate_predictions_by_ticker(prediction_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate prediction quality separately for each ticker.
    """
    rows = []

    for ticker, group in prediction_df.groupby("Ticker"):
        y_true = group["actual_next_return"].reset_index(drop=True)
        y_pred = group["predicted_next_return"].reset_index(drop=True)

        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        corr = float(y_true.corr(y_pred)) if len(y_true) > 1 else np.nan
        directional_accuracy = float(((y_true > 0) == (y_pred > 0)).mean())

        rows.append({
            "Ticker": ticker,
            "rows": len(group),
            "rmse": rmse,
            "mae": mae,
            "correlation": corr,
            "directional_accuracy": directional_accuracy,
            "avg_actual_return": float(y_true.mean()),
            "avg_predicted_return": float(y_pred.mean()),
        })

    return pd.DataFrame(rows).sort_values("Ticker").reset_index(drop=True)