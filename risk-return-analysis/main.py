from src.data_fetch import fetch_price_data
from src.features import build_feature_dataset, get_single_ticker_df
from src.model import (
    prepare_ml_dataset,
    split_train_val_test,
    build_model_matrices,
    train_linear_model,
    make_prediction_frame,
    evaluate_regression,
    evaluate_predictions_by_ticker,
)

# ---------------------------- CHOOSE ASSETS

tickers = ["AAPL", "MSFT", "SPY", "^GSPC", "TLT", "NVDA", "PATH"]
model_tickers = ["AAPL", "MSFT", "SPY", "TLT", "NVDA", "PATH"]

start_date = "2015-01-01"
end_date = "2026-01-01"

# ---------------------------- GET DATA

data = fetch_price_data(tickers, start_date, end_date)

print("\nRAW DATA SHAPE:")
print(data.shape)

print("\nFIRST VALID CLOSE DATE FOR EACH TICKER:")
print(data["Close"].apply(lambda x: x.first_valid_index()))

# Example: inspect one ticker after extraction
example_df = get_single_ticker_df(data, "AAPL")

print("\nAAPL SINGLE-TICKER DATA HEAD:")
print(example_df.head())

print("\nAAPL SINGLE-TICKER COLUMNS:")
print(example_df.columns.tolist())

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

print("\nTOP MISSING VALUES BEFORE ML CLEANING:")
print(feature_df.isna().sum().sort_values(ascending=False).head(20))

# ---------------------------- FINAL ML DATASET PREP

ml_df = prepare_ml_dataset(feature_df, min_history=252)

print("\nML-READY DATA SHAPE:")
print(ml_df.shape)

print("\nML-READY DATE RANGE:")
print(ml_df["Date"].min(), "to", ml_df["Date"].max())

print("\nROWS PER TICKER IN ML-READY DATA:")
print(ml_df["Ticker"].value_counts().sort_index())

# ---------------------------- TRAIN / VALIDATION / TEST SPLIT

train_df, val_df, test_df = split_train_val_test(
    ml_df,
    train_end="2021-12-31",
    val_end="2023-12-31",
)

print("\nTRAIN SHAPE:", train_df.shape)
print("VALIDATION SHAPE:", val_df.shape)
print("TEST SHAPE:", test_df.shape)

print("\nTRAIN DATE RANGE:", train_df["Date"].min(), "to", train_df["Date"].max())
print("VALIDATION DATE RANGE:", val_df["Date"].min(), "to", val_df["Date"].max())
print("TEST DATE RANGE:", test_df["Date"].min(), "to", test_df["Date"].max())

# ---------------------------- BUILD MATRICES

(
    X_train,
    y_train,
    meta_train,
    X_val,
    y_val,
    meta_val,
    X_test,
    y_test,
    meta_test,
) = build_model_matrices(train_df, val_df, test_df)

print("\nX_train SHAPE:", X_train.shape)
print("X_val SHAPE:", X_val.shape)
print("X_test SHAPE:", X_test.shape)

# ---------------------------- FIRST BASELINE MODEL

model = train_linear_model(X_train, y_train)

val_predictions = make_prediction_frame(model, X_val, y_val, meta_val)
test_predictions = make_prediction_frame(model, X_test, y_test, meta_test)

val_metrics = evaluate_regression(
    val_predictions["actual_next_return"],
    val_predictions["predicted_next_return"],
)

test_metrics = evaluate_regression(
    test_predictions["actual_next_return"],
    test_predictions["predicted_next_return"],
)

print("\nVALIDATION METRICS:")
for key, value in val_metrics.items():
    print(f"{key}: {value}")

print("\nTEST METRICS:")
for key, value in test_metrics.items():
    print(f"{key}: {value}")

print("\nVALIDATION PREDICTIONS HEAD:")
print(val_predictions.head())

print("\nTEST PREDICTIONS HEAD:")
print(test_predictions.head())

# ---------------------------- PER-TICKER EVALUATION

val_ticker_metrics = evaluate_predictions_by_ticker(val_predictions)
test_ticker_metrics = evaluate_predictions_by_ticker(test_predictions)

print("\nVALIDATION METRICS BY TICKER:")
print(val_ticker_metrics)

print("\nTEST METRICS BY TICKER:")
print(test_ticker_metrics)