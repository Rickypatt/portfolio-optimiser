"""
backtest/rolling.py
--------------------
Walk-Forward Rolling Backtest for the Portfolio Optimizer.

The core problem with standard Markowitz optimization is LOOK-AHEAD BIAS:
when we compute the "optimal" portfolio using 5 years of data, we're using
information that wasn't available at the start of that period. A real investor
could never have known future returns or covariances.

The walk-forward backtest solves this by simulating what would have happened
if you had rebalanced periodically using ONLY information available at that
point in time:

    ┌─────────────────────────────────────────────────────────────────┐
    │  ESTIMATION WINDOW (lookback)  │  HOLDING PERIOD (rebal. freq) │
    │  Use this data to optimize     │  Hold portfolio, measure P&L   │
    └─────────────────────────────────────────────────────────────────┘
    Then roll forward by one holding period and repeat.

This produces a realistic out-of-sample performance track record.

Key concepts:
    - Estimation window : how far back to look when computing μ and Σ
    - Rebalancing frequency : how often to re-run the optimizer
    - Out-of-sample period : the actual "live" period being tested

Strategies backtested:
    1. Max Sharpe      : tangency portfolio, rebalanced periodically
    2. Min Variance    : minimum risk portfolio
    3. Equal Weight    : naive 1/N benchmark (no optimization)
    4. Buy & Hold SPY  : passive S&P 500 benchmark (if available)
"""

import logging
import numpy as np
import pandas as pd
from typing import Literal

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from data.MarketData import compute_returns, compute_covariance_matrix
from optimisation.Markowitz import (
    find_minimum_variance_constrained,
    find_maximum_sharpe_constrained,
)
from risk.Analytics import compute_drawdown_series, max_drawdown

log = logging.getLogger(__name__)

RebalFreq = Literal["monthly", "quarterly", "semi-annual", "annual"]

REBAL_DAYS: dict[str, int] = {
    "monthly":     21,
    "quarterly":   63,
    "semi-annual": 126,
    "annual":      252,
}


# ── Core backtest engine ──────────────────────────────────────────────────────

def run_backtest(
    prices:         pd.DataFrame,
    lookback_days:  int        = 252,
    rebal_freq:     RebalFreq  = "quarterly",
    initial_value:  float      = 1_000_000,
    allow_short:    bool       = False,
    max_weight:     float      = 1.0,
    min_weight:     float      = 0.0,
    max_sector:     float      = 1.0,
    target_vol:     float | None = None,
    max_turnover:   float      = 1.0,
) -> dict[str, pd.DataFrame]:
    """
    Run a walk-forward backtest for all strategies simultaneously.

    Parameters
    ----------
    prices        : Adjusted closing prices (T × N), any assets.
    lookback_days : Size of the estimation window in trading days.
    rebal_freq    : How often to reoptimize the portfolio.
    initial_value : Starting portfolio value in $.
    allow_short   : Whether to allow short positions.
    max_weight    : Maximum weight per asset (e.g. 0.4 = 40% cap).

    Returns
    -------
    dict mapping strategy name → DataFrame with columns:
        date, portfolio_value, daily_return, weights_*
    """
    rebal_period = REBAL_DAYS[rebal_freq]
    tickers      = list(prices.columns)
    N            = len(tickers)

    # We need at least lookback_days of history before we can start
    start_idx = lookback_days
    if start_idx >= len(prices):
        raise ValueError(
            f"Not enough data: need {lookback_days} days lookback "
            f"but only have {len(prices)} rows."
        )

    log.info(
        "Backtest: %d assets | lookback=%d days | rebal=%s | "
        "%d out-of-sample days",
        N, lookback_days, rebal_freq,
        len(prices) - start_idx,
    )

    # ── Initialise strategy trackers ─────────────────────────────────────────
    strategies = {
        "Max Sharpe":   {"weights": np.ones(N) / N, "value": initial_value},
        "Min Variance": {"weights": np.ones(N) / N, "value": initial_value},
        "Equal Weight": {"weights": np.ones(N) / N, "value": initial_value},
    }

    # Storage for results
    records: dict[str, list] = {s: [] for s in strategies}
    weight_records: dict[str, list] = {s: [] for s in strategies}

    # ── Walk-forward loop ─────────────────────────────────────────────────────
    for t in range(start_idx, len(prices)):
        date       = prices.index[t]
        prev_date  = prices.index[t - 1]

        # Daily returns for each asset (t-1 → t)
        daily_asset_returns = (
            prices.iloc[t] / prices.iloc[t - 1] - 1
        ).values

        # ── Rebalance check ───────────────────────────────────────────────────
        steps_since_start = t - start_idx
        should_rebalance  = (steps_since_start % rebal_period == 0)

        if should_rebalance:
            # Estimation window: lookback_days of history ending yesterday
            window       = prices.iloc[t - lookback_days: t]
            win_returns  = compute_returns(window, save=False)
            cov_matrix   = compute_covariance_matrix(win_returns).values
            mean_returns = win_returns.mean().values * config.TRADING_DAYS

            try:
                max_sr_result = find_maximum_sharpe_constrained(
                    mean_returns, cov_matrix, tickers,
                    risk_free    = config.RISK_FREE_RATE,
                    allow_short  = allow_short,
                    min_weight   = min_weight,
                    max_weight   = max_weight,
                    max_sector   = max_sector,
                    target_vol   = target_vol,
                    max_turnover = max_turnover,
                    current_weights = strategies["Max Sharpe"]["weights"],
                )
                strategies["Max Sharpe"]["weights"] = max_sr_result["weights"]
            except Exception as e:
                log.warning("Max Sharpe rebalance failed at %s: %s", date, e)

            try:
                mvp_result = find_minimum_variance_constrained(
                    mean_returns, cov_matrix, tickers,
                    allow_short  = allow_short,
                    min_weight   = min_weight,
                    max_weight   = max_weight,
                    max_sector   = max_sector,
                    target_vol   = target_vol,
                    max_turnover = max_turnover,
                    current_weights = strategies["Min Variance"]["weights"],
                )
                strategies["Min Variance"]["weights"] = mvp_result["weights"]
            except Exception as e:
                log.warning("Min Variance rebalance failed at %s: %s", date, e)

            # Equal weight never changes
            strategies["Equal Weight"]["weights"] = np.ones(N) / N

        # ── Update portfolio values ───────────────────────────────────────────
        for name, strat in strategies.items():
            w           = strat["weights"]
            port_return = float(w @ daily_asset_returns)
            new_value   = strat["value"] * (1 + port_return)
            strat["value"] = new_value

            records[name].append({
                "date":            date,
                "portfolio_value": new_value,
                "daily_return":    port_return,
            })
            weight_records[name].append(
                {"date": date, **{f"w_{t}": w[i] for i, t in enumerate(tickers)}}
            )

    # ── Package results ───────────────────────────────────────────────────────
    results = {}
    for name in strategies:
        df = pd.DataFrame(records[name]).set_index("date")
        df.index = pd.to_datetime(df.index)
        results[name] = df

    log.info("Backtest complete. Strategies: %s", list(results.keys()))
    return results, pd.DataFrame(weight_records["Max Sharpe"]).set_index("date")


# ── Performance analytics ─────────────────────────────────────────────────────

def compute_backtest_metrics(
    results:      dict[str, pd.DataFrame],
    risk_free:    float = config.RISK_FREE_RATE,
    trading_days: int   = config.TRADING_DAYS,
) -> pd.DataFrame:
    """
    Compute a comprehensive performance table for all backtested strategies.

    Metrics:
        - Total Return        : overall P&L over the backtest period
        - CAGR                : Compound Annual Growth Rate
        - Annualised Volatility
        - Sharpe Ratio        : (CAGR - Rf) / Ann. Vol
        - Sortino Ratio       : like Sharpe but only penalises downside vol
        - Max Drawdown        : worst peak-to-trough loss
        - Calmar Ratio        : CAGR / |Max Drawdown| — reward per drawdown risk
        - Win Rate            : % of days with positive return
        - Worst Day           : single worst daily return
        - Best Day            : single best daily return
    """
    rows = []
    for name, df in results.items():
        r = df["daily_return"]
        n_years = len(r) / trading_days

        total_return = (df["portfolio_value"].iloc[-1] /
                        df["portfolio_value"].iloc[0] - 1) * 100
        cagr         = ((1 + total_return / 100) ** (1 / n_years) - 1) * 100
        ann_vol      = r.std() * np.sqrt(trading_days) * 100
        sharpe       = (cagr / 100 - risk_free) / (ann_vol / 100) if ann_vol > 0 else 0

        # Sortino: downside deviation only
        downside = r[r < 0].std() * np.sqrt(trading_days) * 100
        sortino  = (cagr / 100 - risk_free) / (downside / 100) if downside > 0 else 0

        mdd     = max_drawdown(r)
        calmar  = (cagr / 100) / abs(mdd / 100) if mdd != 0 else 0
        win_rate = (r > 0).mean() * 100

        rows.append({
            "Strategy":         name,
            "Total Return (%)": round(total_return, 2),
            "CAGR (%)":         round(cagr, 2),
            "Ann. Vol (%)":     round(ann_vol, 2),
            "Sharpe":           round(sharpe, 3),
            "Sortino":          round(sortino, 3),
            "Max Drawdown (%)": round(mdd, 2),
            "Calmar":           round(calmar, 3),
            "Win Rate (%)":     round(win_rate, 1),
            "Best Day (%)":     round(r.max() * 100, 2),
            "Worst Day (%)":    round(r.min() * 100, 2),
        })

    return pd.DataFrame(rows).set_index("Strategy")


def compute_rolling_metrics(
    results:      dict[str, pd.DataFrame],
    window:       int = 63,
    trading_days: int = config.TRADING_DAYS,
) -> dict[str, pd.DataFrame]:
    """
    Compute rolling Sharpe and rolling volatility for each strategy.
    Window default = 63 days (~1 quarter).
    """
    rolling = {}
    for name, df in results.items():
        r = df["daily_return"]
        roll_vol   = r.rolling(window).std() * np.sqrt(trading_days) * 100
        roll_ret   = r.rolling(window).mean() * trading_days
        roll_sharpe = (roll_ret - config.RISK_FREE_RATE) / (r.rolling(window).std() * np.sqrt(trading_days))
        rolling[name] = pd.DataFrame({
            "rolling_vol":    roll_vol,
            "rolling_sharpe": roll_sharpe,
        })
    return rolling