import yfinance as yf

tickers = ["AAPL", "MSFT", "SPY", "^GSPC", "TLT"] # list of stock, ETF, index tickers
data = yf.download(tickers, start="2015-01-01", end="2026-01-01", interval="1d")