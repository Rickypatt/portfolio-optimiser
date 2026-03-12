"""
config.py
---------
Central configuration for the Portfolio Optimizer.
All parameters that might change between runs live here —
never hardcode values in module files.
"""

from datetime import datetime, timedelta

# ── Universe ──────────────────────────────────────────────────────────────────
# Default asset universe: diversified across sectors & geographies.
# Tickers follow Yahoo Finance conventions.
DEFAULT_TICKERS: list[str] = [
    "AAPL",   # Technology
    "MSFT",   # Technology
    "JPM",    # Financials
    "JNJ",    # Healthcare
    "XOM",    # Energy
    "GLD",    # Gold ETF  (alternative / hedge)
    "TLT",    # 20yr Treasury ETF (bonds)
    "SPY",    # S&P 500 benchmark
]

# ── Time Window ───────────────────────────────────────────────────────────────
END_DATE:   str = datetime.today().strftime("%Y-%m-%d")
START_DATE: str = (datetime.today() - timedelta(days=5 * 365)).strftime("%Y-%m-%d")  # 5-year window

# ── Returns ───────────────────────────────────────────────────────────────────
RETURN_TYPE: str = "log"          # "log" | "simple"
TRADING_DAYS: int = 252           # annualisation factor

# ── Risk-Free Rate ────────────────────────────────────────────────────────────
# Approximate US 3-month T-bill yield (annualised). Update periodically.
RISK_FREE_RATE: float = 0.053     # 5.3% as of early 2024

# ── Sector Map ────────────────────────────────────────────────────────────────
# Maps ticker → GICS sector. Used for sector concentration constraints.
# Extend this as you add more tickers to your universe.
SECTOR_MAP: dict[str, str] = {
    "AAPL":  "Technology",
    "MSFT":  "Technology",
    "NVDA":  "Technology",
    "GOOGL": "Technology",
    "META":  "Technology",
    "AMZN":  "Consumer Discretionary",
    "TSLA":  "Consumer Discretionary",
    "JPM":   "Financials",
    "GS":    "Financials",
    "BAC":   "Financials",
    "V":     "Financials",
    "BRK-B": "Financials",
    "JNJ":   "Healthcare",
    "UNH":   "Healthcare",
    "PFE":   "Healthcare",
    "MRK":   "Healthcare",
    "XOM":   "Energy",
    "CVX":   "Energy",
    "CL=F":  "Energy",
    "GLD":   "Commodities",
    "GC=F":  "Commodities",
    "TLT":   "Fixed Income",
    "AGG":   "Fixed Income",
    "IEF":   "Fixed Income",
    "SPY":   "Equity Index",
    "QQQ":   "Equity Index",
    "VTI":   "Equity Index",
    "BTC-USD": "Crypto",
    "ETH-USD": "Crypto",
}

SECTOR_COLORS: dict[str, str] = {
    "Technology":             "#00d4ff",
    "Financials":             "#ffd700",
    "Healthcare":             "#00ff9d",
    "Energy":                 "#ff9500",
    "Consumer Discretionary": "#a78bfa",
    "Commodities":            "#f87171",
    "Fixed Income":           "#94a3b8",
    "Equity Index":           "#34d399",
    "Crypto":                 "#fb923c",
    "Other":                  "#64748b",
}

# ── Constraint Defaults ───────────────────────────────────────────────────────
DEFAULT_CONSTRAINTS: dict = {
    "min_weight":        0.0,    # minimum weight per asset (0 = can be excluded)
    "max_weight":        1.0,    # maximum weight per asset
    "max_sector":        1.0,    # maximum total exposure to any single sector
    "target_vol":        None,   # target annualised portfolio volatility (None = unconstrained)
    "max_turnover":      1.0,    # max portfolio turnover per rebalance (1.0 = unconstrained)
    "allow_short":       False,
}

DATA_DIR:    str = "outputs/"
PRICE_FILE:  str = "prices_raw.csv"
RETURN_FILE: str = "log_returns.csv"