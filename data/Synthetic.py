"""
data/synthetic.py
------------------
Generates realistic synthetic price data for offline demos and testing.
Uses a correlated Geometric Brownian Motion (GBM) model — the same
stochastic process underlying the Black-Scholes framework.

This is NOT used in production; it exists only so the pipeline can be
demonstrated without a live internet connection.
"""

import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ── GBM Parameters (loosely calibrated to historical data) ───────────────────
_PARAMS = {
    #           mu (ann.)   sigma (ann.)
    "AAPL":  (  0.28,        0.28 ),
    "MSFT":  (  0.25,        0.25 ),
    "JPM":   (  0.14,        0.24 ),
    "JNJ":   (  0.07,        0.17 ),
    "XOM":   (  0.12,        0.28 ),
    "GLD":   (  0.06,        0.15 ),
    "TLT":   ( -0.04,        0.14 ),
    "SPY":   (  0.12,        0.18 ),
}

# Approximate correlation structure
_CORR = np.array([
    # AAPL  MSFT   JPM   JNJ   XOM   GLD   TLT   SPY
    [ 1.00, 0.82,  0.45, 0.30, 0.25, 0.05, -0.15, 0.75],  # AAPL
    [ 0.82, 1.00,  0.42, 0.28, 0.22, 0.03, -0.12, 0.72],  # MSFT
    [ 0.45, 0.42,  1.00, 0.38, 0.35, 0.08, -0.20, 0.68],  # JPM
    [ 0.30, 0.28,  0.38, 1.00, 0.20, 0.12, -0.05, 0.55],  # JNJ
    [ 0.25, 0.22,  0.35, 0.20, 1.00, 0.18, -0.08, 0.52],  # XOM
    [ 0.05, 0.03,  0.08, 0.12, 0.18, 1.00,  0.25, 0.10],  # GLD
    [-0.15,-0.12, -0.20,-0.05,-0.08, 0.25,  1.00,-0.25],  # TLT
    [ 0.75, 0.72,  0.68, 0.55, 0.52, 0.10, -0.25, 1.00],  # SPY
])


def generate_prices(
    tickers:      list[str] = config.DEFAULT_TICKERS,
    start_date:   str       = config.START_DATE,
    end_date:     str       = config.END_DATE,
    seed:         int       = 42,
) -> pd.DataFrame:
    """
    Simulate correlated GBM price paths.

    GBM discretisation (Euler-Maruyama):
        ln(S_{t+1}/S_t) = (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_t

    where Z_t ~ N(0,1) and correlations are introduced via Cholesky decomposition.
    """
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start=start_date, end=end_date)   # business days only
    T     = len(dates)
    N     = len(tickers)
    dt    = 1 / config.TRADING_DAYS

    # Subset correlation matrix to requested tickers
    all_tickers = list(_PARAMS.keys())
    idx = [all_tickers.index(t) for t in tickers]
    corr_sub = _CORR[np.ix_(idx, idx)]

    # Cholesky decomposition to generate correlated normals
    L = np.linalg.cholesky(corr_sub)

    prices = np.zeros((T, N))
    prices[0] = 100.0    # all assets start at 100

    mus    = np.array([_PARAMS[t][0] for t in tickers])
    sigmas = np.array([_PARAMS[t][1] for t in tickers])

    for t in range(1, T):
        Z = rng.standard_normal(N)
        Z_corr = L @ Z                                         # correlate shocks
        drift   = (mus - 0.5 * sigmas**2) * dt
        diffuse = sigmas * np.sqrt(dt) * Z_corr
        prices[t] = prices[t-1] * np.exp(drift + diffuse)

    return pd.DataFrame(prices, index=dates, columns=tickers)