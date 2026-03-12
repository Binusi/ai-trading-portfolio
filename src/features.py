import numpy as np
import pandas as pd
from typing import Optional

def get_single_ticker_df(price_data: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Extract one ticker from yfinance multi-index output into a normal dataframe.

    Expected input columns like:
    ('Close', 'AAPL'), ('Open', 'AAPL'), ...

    Returns dataframe with columns like:
    Open, High, Low, Close, Volume
    """
    if not isinstance(price_data.columns, pd.MultiIndex):
        raise ValueError("Expected a MultiIndex dataframe from yfinance download.")

    df = price_data.xs(ticker, axis=1, level="Ticker").copy()
    df = df.sort_index()

    # Drop rows where Close is missing (e.g. before IPO / listing date)
    df = df[df["Close"].notna()].copy()

    return df


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Compute RSI using simple rolling averages.
    """
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def add_features(
    df: pd.DataFrame,
    market_returns: Optional[pd.Series] = None,
    ticker: Optional[str] = None,
    market_ticker: str = "^GSPC",
) -> pd.DataFrame:
    """
    Add technical / statistical features to a single-ticker dataframe.

    Expected columns:
    Open, High, Low, Close, Volume
    """
    df = df.copy()
    close = df["Close"]

    # -----------------------------
    # Basic returns
    # -----------------------------
    df["return_1d"] = close.pct_change()
    df["return_5d"] = close.pct_change(5)
    df["return_21d"] = close.pct_change(21)

    # -----------------------------
    # Moving averages / trend
    # -----------------------------
    df["ma_10"] = close.rolling(10).mean()
    df["ma_20"] = close.rolling(20).mean()
    df["ma_50"] = close.rolling(50).mean()
    df["ma_200"] = close.rolling(200).mean()

    # Gaps are usually more useful than raw MA values
    df["ma_10_gap"] = close / df["ma_10"] - 1
    df["ma_20_gap"] = close / df["ma_20"] - 1
    df["ma_50_gap"] = close / df["ma_50"] - 1
    df["ma_200_gap"] = close / df["ma_200"] - 1

    # -----------------------------
    # Volatility
    # -----------------------------
    df["volatility_21d"] = df["return_1d"].rolling(21).std()
    df["volatility_63d"] = df["return_1d"].rolling(63).std()

    # -----------------------------
    # RSI
    # -----------------------------
    df["rsi_14"] = compute_rsi(close, window=14)

    # -----------------------------
    # EMA / MACD
    # -----------------------------
    df["ema_12"] = close.ewm(span=12, adjust=False).mean()
    df["ema_26"] = close.ewm(span=26, adjust=False).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # -----------------------------
    # Price range features
    # -----------------------------
    df["intraday_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["open_close_change"] = (df["Close"] - df["Open"]) / df["Open"]

    # -----------------------------
    # Volume features
    # -----------------------------
    df["volume_ma_20"] = df["Volume"].rolling(20).mean()
    df["volume_ratio_20"] = df["Volume"] / df["volume_ma_20"]

    # -----------------------------
    # Rolling Sharpe-like feature
    # (daily mean / daily std over last 63 days)
    # -----------------------------
    rolling_mean_63 = df["return_1d"].rolling(63).mean()
    rolling_std_63 = df["return_1d"].rolling(63).std()
    df["sharpe_like_63d"] = rolling_mean_63 / rolling_std_63

    # -----------------------------
    # Market / macro context
    # -----------------------------
    if market_returns is not None:
        aligned_market_returns = market_returns.reindex(df.index)

        df["market_return_1d"] = aligned_market_returns
        df["market_return_5d"] = aligned_market_returns.rolling(5).sum()
        df["market_volatility_21d"] = aligned_market_returns.rolling(21).std()

        # Rolling beta against market
        if ticker != market_ticker:
            cov = df["return_1d"].rolling(90).cov(aligned_market_returns)
            var = aligned_market_returns.rolling(90).var()
            df["beta_90d"] = cov / var
        else:
            df["beta_90d"] = np.nan

    # -----------------------------
    # Targets
    # IMPORTANT:
    # target at date t = return from t to t+1
    # -----------------------------
    df["target_next_return"] = close.pct_change().shift(-1)
    df["target_next_direction"] = (df["target_next_return"] > 0).astype("Int64")

    return df


def build_feature_dataset(
    price_data: pd.DataFrame,
    tickers: list[str],
    market_ticker: str = "^GSPC",
    dropna: bool = True,
) -> pd.DataFrame:
    """
    Build one combined long-form ML dataset for all requested tickers.
    """
    market_df = get_single_ticker_df(price_data, market_ticker)
    market_returns = market_df["Close"].pct_change()

    all_frames = []

    for ticker in tickers:
        ticker_df = get_single_ticker_df(price_data, ticker)

        feature_df = add_features(
            ticker_df,
            market_returns=market_returns,
            ticker=ticker,
            market_ticker=market_ticker,
        )

        feature_df = feature_df.copy()
        feature_df["Ticker"] = ticker
        feature_df = feature_df.reset_index()  # keeps Date as a column

        all_frames.append(feature_df)

    combined = pd.concat(all_frames, ignore_index=True)

    if dropna:
        # Drop rows that do not yet have enough history for rolling features
        required_cols = [
            "return_1d",
            "ma_10_gap",
            "ma_50_gap",
            "volatility_21d",
            "rsi_14",
            "target_next_return",
        ]
        combined = combined.dropna(subset=required_cols).reset_index(drop=True)

    return combined