"""
data/market_data.py
--------------------
Handles all data ingestion and preprocessing for the portfolio optimizer.

Responsibilities:
  - Fetch adjusted closing prices from Yahoo Finance
  - Compute log returns and simple returns
  - Compute the annualised covariance matrix and correlation matrix
  - Run data quality checks (missing data, survivorship warnings)

Design note: Every public function returns a plain pandas object so downstream
modules stay decoupled from the data source. Swapping yfinance for Bloomberg
or a database should only require changes here.
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
try:
    import yfinance as yf
    _YFINANCE_AVAILABLE = True
except ImportError:
    _YFINANCE_AVAILABLE = False

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── Logger ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Data Fetching ─────────────────────────────────────────────────────────────

def fetch_prices(
    tickers:    list[str],
    start_date: str,
    end_date:   str,
    save:       bool = True,
) -> pd.DataFrame:
    """
    Download adjusted closing prices from Yahoo Finance.

    Parameters
    ----------
    tickers    : List of Yahoo Finance ticker symbols.
    start_date : ISO date string, e.g. "2019-01-01".
    end_date   : ISO date string, e.g. "2024-01-01".
    save       : If True, write raw prices to outputs/.

    Returns
    -------
    pd.DataFrame  — shape (T, N), columns = tickers, index = Date.
    """
    if not _YFINANCE_AVAILABLE:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    log.info("Fetching prices for %d tickers: %s", len(tickers), tickers)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw: pd.DataFrame = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,   # use adjusted prices (splits + dividends)
            progress=False,
        )

    # yfinance returns MultiIndex columns when >1 ticker
    if isinstance(raw.columns, pd.MultiIndex):
        # Flatten: newer yfinance uses ("Close", "AAPL") structure
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        elif "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"].copy()
        else:
            prices = raw.xs("Close", axis=1, level=0).copy()
    else:
        if "Close" in raw.columns:
            prices = raw[["Close"]].copy()
            prices.columns = tickers
        else:
            prices = raw.copy()

    # Drop any all-NaN columns
    prices = prices.dropna(axis=1, how="all")

    prices.index = pd.to_datetime(prices.index)
    prices.index.name = "Date"

    # ── Quality Checks ────────────────────────────────────────────────────────
    _quality_check(prices, tickers)

    if save:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        out_path = os.path.join(config.DATA_DIR, config.PRICE_FILE)
        prices.to_csv(out_path)
        log.info("Raw prices saved → %s", out_path)

    log.info(
        "Price matrix: %d trading days × %d assets  [%s → %s]",
        len(prices), len(prices.columns),
        prices.index[0].date(), prices.index[-1].date(),
    )
    return prices


# ── Returns ───────────────────────────────────────────────────────────────────

def compute_returns(
    prices: pd.DataFrame,
    method: str = config.RETURN_TYPE,
    save:   bool = True,
) -> pd.DataFrame:
    """
    Compute daily asset returns.

    Log returns  : r_t = ln(P_t / P_{t-1})
    Simple returns: r_t = (P_t / P_{t-1}) - 1

    Log returns are preferred in quant finance because they are:
      - Time-additive (weekly log return = sum of daily log returns)
      - More normally distributed for small intervals
      - Symmetric around zero for equal up/down moves

    Parameters
    ----------
    prices : Price DataFrame from fetch_prices().
    method : "log" (default) or "simple".
    save   : If True, write returns to outputs/.

    Returns
    -------
    pd.DataFrame — shape (T-1, N), daily returns.
    """
    if method == "log":
        returns = np.log(prices / prices.shift(1)).dropna()
        log.info("Computed log returns: shape %s", returns.shape)
    elif method == "simple":
        returns = prices.pct_change().dropna()
        log.info("Computed simple returns: shape %s", returns.shape)
    else:
        raise ValueError(f"Unknown return method '{method}'. Use 'log' or 'simple'.")

    if save:
        os.makedirs(config.DATA_DIR, exist_ok=True)
        out_path = os.path.join(config.DATA_DIR, config.RETURN_FILE)
        returns.to_csv(out_path)
        log.info("Returns saved → %s", out_path)

    return returns


# ── Covariance & Correlation ──────────────────────────────────────────────────

def compute_covariance_matrix(
    returns:      pd.DataFrame,
    annualise:    bool = True,
    trading_days: int  = config.TRADING_DAYS,
) -> pd.DataFrame:
    """
    Compute the sample covariance matrix from daily returns.

    Annualisation scales daily covariance by the number of trading days,
    giving units of (annual return)^2 — consistent with annualised volatility.

    Σ_annual = Σ_daily × T

    Parameters
    ----------
    returns      : Daily returns DataFrame.
    annualise    : Scale to annual units (default True).
    trading_days : Number of trading days per year (default 252).

    Returns
    -------
    pd.DataFrame — symmetric (N × N) covariance matrix.
    """
    cov = returns.cov()
    if annualise:
        cov = cov * trading_days
        log.info("Annualised covariance matrix computed (T=%d)", trading_days)
    return cov


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson correlation matrix from daily returns.

    Unlike covariance, correlation is scale-free (values in [-1, 1]),
    making it easier to interpret cross-asset relationships.

    Returns
    -------
    pd.DataFrame — symmetric (N × N) correlation matrix.
    """
    corr = returns.corr()
    log.info("Correlation matrix computed")
    return corr


# ── Summary Statistics ────────────────────────────────────────────────────────

def compute_summary_stats(
    returns:      pd.DataFrame,
    trading_days: int = config.TRADING_DAYS,
) -> pd.DataFrame:
    """
    Compute annualised summary statistics for each asset.

    Columns returned:
      - Ann. Return   : Geometric mean annual return
      - Ann. Volatility: Annualised standard deviation
      - Sharpe Ratio  : (Return - Rf) / Volatility  (uses config.RISK_FREE_RATE)
      - Skewness      : Asymmetry of the return distribution
      - Kurtosis      : Tail heaviness (excess kurtosis; normal = 0)
      - Max Drawdown  : Largest peak-to-trough decline

    Returns
    -------
    pd.DataFrame — shape (N, 7), one row per asset.
    """
    stats = pd.DataFrame(index=returns.columns)

    stats["Ann. Return (%)"]     = (returns.mean() * trading_days * 100).round(2)
    stats["Ann. Volatility (%)"] = (returns.std() * np.sqrt(trading_days) * 100).round(2)
    stats["Sharpe Ratio"]        = (
        (returns.mean() * trading_days - config.RISK_FREE_RATE)
        / (returns.std() * np.sqrt(trading_days))
    ).round(3)
    stats["Skewness"]  = returns.skew().round(3)
    stats["Kurtosis"]  = returns.kurt().round(3)   # excess kurtosis

    # Max drawdown per asset
    stats["Max Drawdown (%)"] = returns.apply(_max_drawdown).round(2)

    log.info("Summary statistics computed for %d assets", len(stats))
    return stats


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _quality_check(prices: pd.DataFrame, requested_tickers: list[str]) -> None:
    """Warn on missing tickers and flag assets with excessive NaN rates."""
    missing = [t for t in requested_tickers if t not in prices.columns]
    if missing:
        log.warning("Tickers not found in response: %s", missing)

    nan_rates = prices.isna().mean()
    problematic = nan_rates[nan_rates > 0.02]   # >2% NaN is suspicious
    if not problematic.empty:
        log.warning(
            "High NaN rate detected (>2%%) — consider removing:\n%s",
            problematic.to_string()
        )


def _max_drawdown(return_series: pd.Series) -> float:
    """
    Compute maximum drawdown from a return series.

    Max drawdown = max(peak - trough) / peak, expressed as a percentage loss.
    """
    cumulative = (1 + return_series).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min() * 100   # as a percentage