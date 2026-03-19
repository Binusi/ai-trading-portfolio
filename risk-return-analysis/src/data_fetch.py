import yfinance as yf


def fetch_price_data(tickers, start, end, interval="1d"):
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    return data