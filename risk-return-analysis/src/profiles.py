from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RiskProfile:
    key: str
    name: str
    description: str
    targets: dict[str, float]

    def __post_init__(self) -> None:
        total = sum(self.targets.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Profile {self.key!r} target weights sum to {total:.6f}, expected 1.0"
            )


RISK_PROFILES: dict[str, RiskProfile] = {
    "conservative": RiskProfile(
        key="conservative",
        name="Conservative",
        description=(
            "Capital preservation focus. A heavy bond allocation cushions equity "
            "drawdowns at the cost of lower expected long-term returns. Suitable "
            "if you're nervous about volatility or have a short time horizon."
        ),
        targets={
            "us_equity": 0.20,
            "bond_etf": 0.60,
            "equity_index_etf": 0.10,
            "commodity_etf": 0.05,
            "international_etf": 0.05,
        },
    ),
    "balanced": RiskProfile(
        key="balanced",
        name="Balanced",
        description=(
            "Mix of growth and defense. Moderate equity exposure with meaningful "
            "bond ballast. A reasonable default for medium-term goals."
        ),
        targets={
            "us_equity": 0.45,
            "bond_etf": 0.30,
            "equity_index_etf": 0.15,
            "commodity_etf": 0.05,
            "international_etf": 0.05,
        },
    ),
    "aggressive": RiskProfile(
        key="aggressive",
        name="Aggressive",
        description=(
            "Growth focus. Heavy equity weighting maximises long-term expected "
            "return but expect periodic 20%+ drawdowns. Suitable if you have a "
            "long time horizon and won't panic-sell during downturns."
        ),
        targets={
            "us_equity": 0.65,
            "bond_etf": 0.10,
            "equity_index_etf": 0.15,
            "commodity_etf": 0.05,
            "international_etf": 0.05,
        },
    ),
}


# For each non-equity asset class, the single ticker we hold as the index proxy.
# The us_equity sleeve is distributed across individual names (with optional AI tilt).
INDEX_PROXIES: dict[str, str] = {
    "bond_etf": "TLT",
    "equity_index_etf": "SPY",
    "commodity_etf": "GLD",
    "international_etf": "EFA",
}


# Default individual-stock universe used for the equity sleeve and AI tilt.
# These should be tickers the ML model has been trained on.
DEFAULT_EQUITY_UNIVERSE: list[str] = ["AAPL", "MSFT", "NVDA"]


FRIENDLY_CLASS_NAMES: dict[str, str] = {
    "us_equity": "US stocks",
    "bond_etf": "long-term Treasuries (TLT)",
    "equity_index_etf": "S&P 500 (SPY)",
    "commodity_etf": "gold (GLD)",
    "international_etf": "international stocks (EFA)",
}


def get_profile(key: str) -> RiskProfile:
    if key not in RISK_PROFILES:
        raise ValueError(
            f"Unknown profile {key!r}. Choose from {sorted(RISK_PROFILES)}."
        )
    return RISK_PROFILES[key]


def get_required_tickers(equity_universe: list[str]) -> list[str]:
    """All tickers needed to run a profile-based simulation."""
    seen: list[str] = []
    for t in list(equity_universe) + list(INDEX_PROXIES.values()):
        if t not in seen:
            seen.append(t)
    return seen
