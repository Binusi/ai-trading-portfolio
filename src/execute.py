from data_fetch import fetch_price_data
from features import build_feature_dataset, get_single_ticker_df

# ---------------------------- GET DATA

tickers = ["AAPL", "MSFT", "SPY", "^GSPC", "TLT", "NVDA", "PATH"]
start_date = "2015-01-01"
end_date = "2026-01-01"

data = fetch_price_data(tickers, start_date, end_date)

print("\nRAW DATA SHAPE:")
print(data.shape)

print("\nFIRST VALID CLOSE DATE FOR EACH TICKER:")
print(data["Close"].apply(lambda x: x.first_valid_index()))

print("\nRAW DATA HEAD:")
print(data.head())

# Example: inspect one ticker after extraction
aapl_df = get_single_ticker_df(data, "AAPL")

print("\nAAPL SINGLE-TICKER DATA HEAD:")
print(aapl_df.head())

print("\nAAPL SINGLE-TICKER COLUMNS:")
print(aapl_df.columns.tolist())

# ---------------------------- GENERATE FEATURES

model_df = build_feature_dataset(
    price_data=data,
    tickers=["AAPL", "MSFT", "SPY", "TLT", "NVDA", "PATH"],
    market_ticker="^GSPC"
)

print("\nMODEL DATA HEAD:")
print(model_df.head())

print("\nMODEL DATA COLUMNS:")
print(model_df.columns.tolist())

print("\nMODEL DATA SHAPE:")
print(model_df.shape)

print("\nMODEL DATA INFO:")
print(model_df.info())

print("\nMISSING VALUES PER COLUMN:")
print(model_df.isna().sum().sort_values(ascending=False).head(20))