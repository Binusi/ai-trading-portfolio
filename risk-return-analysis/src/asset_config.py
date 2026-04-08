from __future__ import annotations


ASSET_UNIVERSE = {
    "us_equity": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "JNJ"],
    "equity_index_etf": ["SPY", "QQQ", "IWM"],
    "bond_etf": ["TLT", "IEF", "AGG", "HYG"],
    "commodity_etf": ["GLD", "SLV", "USO"],
    "reit_etf": ["VNQ", "IYR"],
    "international_etf": ["EFA", "EEM"],
    "crypto_etf": ["IBIT"],
}

MARKET_TICKER = "^GSPC"


def get_all_tickers() -> list[str]:
    tickers = []
    for class_tickers in ASSET_UNIVERSE.values():
        tickers.extend(class_tickers)
    return tickers


def get_asset_group_map() -> dict[str, str]:
    mapping = {}
    for asset_class, tickers in ASSET_UNIVERSE.items():
        for ticker in tickers:
            mapping[ticker] = asset_class
    return mapping


def get_tickers_for_class(class_name: str) -> list[str]:
    return ASSET_UNIVERSE.get(class_name, [])


def get_asset_class_filters() -> dict[str, str | None]:
    filters = {"all_assets": None}
    for asset_class in ASSET_UNIVERSE:
        filters[asset_class] = asset_class
    return filters
