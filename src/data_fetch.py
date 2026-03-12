import yfinance as yf

def fetch_price_data(tickers, start, end, interval="1d"):
    data = yf.download(tickers, start=start, end=end, interval=interval)
    return data
