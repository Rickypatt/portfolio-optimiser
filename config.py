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

# ── Data ──────────────────────────────────────────────────────────────────────
DATA_DIR:    str = "outputs/"
PRICE_FILE:  str = "prices_raw.csv"
RETURN_FILE: str = "log_returns.csv"