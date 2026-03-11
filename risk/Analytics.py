"""
risk/analytics.py
------------------
Implements industry-standard portfolio risk metrics.

Metrics covered:

  1. VALUE AT RISK (VaR)
     "What is the maximum loss we would expect to suffer over a given
     time horizon, at a given confidence level?"

     Three methodologies:
       a) Historical Simulation  — uses the empirical return distribution, no
                                   distributional assumptions. Simple and robust.
       b) Parametric (Gaussian)  — assumes returns are normally distributed.
                                   Fast but underestimates tail risk.
       c) Monte Carlo            — simulates thousands of future return paths
                                   using the estimated covariance structure.
                                   Most flexible, handles non-linearities.

  2. CONDITIONAL VALUE AT RISK (CVaR) — also called Expected Shortfall (ES)
     "Given that we ARE in the worst x% of outcomes, what is the expected loss?"

     CVaR is the mean of all losses beyond the VaR threshold. It is a
     coherent risk measure (unlike VaR) and is preferred by regulators
     (Basel III/IV) and sophisticated risk desks because it captures
     tail risk more completely.

  3. MONTE CARLO SIMULATION
     Simulates N future portfolio paths using correlated Geometric Brownian
     Motion. Gives a full distribution of possible outcomes, not just a
     single point estimate.

  4. MAX DRAWDOWN
     The largest peak-to-trough decline in portfolio value over the period.
     Critical for understanding the worst historical loss an investor would
     have experienced holding the portfolio.

  5. ROLLING RISK METRICS
     Time-varying VaR and volatility — shows how risk evolves through
     different market regimes (crises, recoveries, quiet periods).
"""

import logging
import numpy as np
import pandas as pd
from scipy import stats

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

log = logging.getLogger(__name__)


# ── Value at Risk ─────────────────────────────────────────────────────────────

def var_historical(
    portfolio_returns: pd.Series,
    confidence:        float = 0.95,
    horizon_days:      int   = 1,
) -> float:
    """
    Historical Simulation VaR.

    Takes the empirical return distribution and reads off the loss at
    the (1 - confidence) quantile. No distributional assumptions.

    Parameters
    ----------
    portfolio_returns : Daily portfolio return series.
    confidence        : Confidence level (e.g. 0.95 = 95% VaR).
    horizon_days      : Holding period in days. Scales by sqrt(T) — valid
                        only under i.i.d. assumption.

    Returns
    -------
    float : VaR as a positive number (i.e. loss magnitude).
    """
    alpha   = 1 - confidence
    daily_var = float(np.percentile(portfolio_returns, alpha * 100))
    scaled_var = daily_var * np.sqrt(horizon_days)   # square-root-of-time scaling
    return -scaled_var   # return as positive loss


def var_parametric(
    portfolio_returns: pd.Series,
    confidence:        float = 0.95,
    horizon_days:      int   = 1,
) -> float:
    """
    Parametric (Gaussian) VaR.

    Assumes returns are normally distributed: R ~ N(μ, σ²).
    VaR = -(μ * T - z_α * σ * √T)

    Where z_α is the inverse CDF of the standard normal at level α.

    This is the fastest method but systematically underestimates risk
    because real return distributions have fat tails (excess kurtosis > 0).
    """
    mu    = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    z     = stats.norm.ppf(1 - confidence)   # negative z for left tail

    daily_var = mu + z * sigma
    scaled_var = daily_var * np.sqrt(horizon_days)
    return -scaled_var


def var_monte_carlo(
    weights:      np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix:   np.ndarray,
    confidence:   float = 0.95,
    horizon_days: int   = 1,
    n_simulations: int  = 10_000,
    seed:         int   = 42,
) -> float:
    """
    Monte Carlo VaR.

    Simulates n_simulations portfolio returns by drawing from a
    multivariate normal distribution calibrated to the historical
    mean and covariance. The VaR is then read from the simulated
    empirical distribution.

    The multivariate normal draw uses Cholesky decomposition to
    generate correlated asset returns:
        R = μ*dt + L * Z * √dt
    where L is the Cholesky factor of Σ and Z ~ N(0, I).
    """
    rng = np.random.default_rng(seed)

    dt      = horizon_days / config.TRADING_DAYS
    mu_day  = mean_returns * dt
    cov_day = cov_matrix   * dt

    # Draw correlated multivariate normal returns
    sim_asset_returns = rng.multivariate_normal(mu_day, cov_day, size=n_simulations)

    # Portfolio return = weighted sum of asset returns
    sim_port_returns = sim_asset_returns @ weights

    alpha = 1 - confidence
    return -float(np.percentile(sim_port_returns, alpha * 100))


# ── Conditional VaR (Expected Shortfall) ─────────────────────────────────────

def cvar_historical(
    portfolio_returns: pd.Series,
    confidence:        float = 0.95,
    horizon_days:      int   = 1,
) -> float:
    """
    Historical CVaR (Expected Shortfall).

    CVaR = E[R | R ≤ VaR_α]  — the mean of all returns in the tail.

    CVaR is always ≥ VaR for the same confidence level. The gap between
    CVaR and VaR tells you how bad the tail is beyond the VaR threshold.
    A large gap indicates a fat-tailed, dangerous distribution.
    """
    alpha    = 1 - confidence
    var      = np.percentile(portfolio_returns, alpha * 100)
    tail     = portfolio_returns[portfolio_returns <= var]
    daily_cvar = float(tail.mean())
    return -daily_cvar * np.sqrt(horizon_days)


def cvar_parametric(
    portfolio_returns: pd.Series,
    confidence:        float = 0.95,
    horizon_days:      int   = 1,
) -> float:
    """
    Parametric CVaR (Gaussian assumption).

    CVaR = -(μ - σ * φ(z_α) / α)

    Where φ is the standard normal PDF and α = 1 - confidence.
    This is the analytical closed-form solution under normality.
    """
    mu    = portfolio_returns.mean()
    sigma = portfolio_returns.std()
    alpha = 1 - confidence
    z     = stats.norm.ppf(alpha)

    # Expected value of the truncated normal in the left tail
    daily_cvar = mu - sigma * stats.norm.pdf(z) / alpha
    return -daily_cvar * np.sqrt(horizon_days)


# ── Monte Carlo Path Simulation ───────────────────────────────────────────────

def simulate_portfolio_paths(
    weights:        np.ndarray,
    mean_returns:   np.ndarray,
    cov_matrix:     np.ndarray,
    horizon_days:   int   = 252,
    n_simulations:  int   = 1_000,
    initial_value:  float = 1_000_000.0,
    seed:           int   = 42,
) -> np.ndarray:
    """
    Simulate future portfolio value paths using correlated GBM.

    Each path represents one possible evolution of a portfolio worth
    `initial_value` over `horizon_days` trading days.

    Returns
    -------
    np.ndarray — shape (horizon_days + 1, n_simulations)
                 Row 0 = initial_value for all paths.
    """
    rng = np.random.default_rng(seed)

    N  = len(weights)
    dt = 1 / config.TRADING_DAYS

    # Daily drift and diffusion parameters
    mu_day  = mean_returns * dt
    cov_day = cov_matrix   * dt
    L       = np.linalg.cholesky(cov_day)   # Cholesky factor

    # Initialise path matrix
    paths = np.zeros((horizon_days + 1, n_simulations))
    paths[0] = initial_value

    for t in range(1, horizon_days + 1):
        Z = rng.standard_normal((N, n_simulations))
        correlated_shocks = L @ Z                        # shape (N, n_simulations)

        # Log-return for each asset on each path
        log_returns = (mu_day - 0.5 * np.diag(cov_day))[:, None] + correlated_shocks

        # Portfolio return = weighted sum
        port_log_return = weights @ log_returns          # shape (n_simulations,)

        paths[t] = paths[t-1] * np.exp(port_log_return)

    log.info(
        "Monte Carlo: %d paths × %d days | "
        "Final value P5=%.0f  Median=%.0f  P95=%.0f",
        n_simulations, horizon_days,
        np.percentile(paths[-1], 5),
        np.median(paths[-1]),
        np.percentile(paths[-1], 95),
    )
    return paths


# ── Drawdown Analysis ─────────────────────────────────────────────────────────

def compute_drawdown_series(portfolio_returns: pd.Series) -> pd.Series:
    """
    Compute the full drawdown time series.

    Drawdown_t = (Cumulative_t - Peak_t) / Peak_t

    Returns a series of values ≤ 0, where 0 means no drawdown (at a peak).
    """
    cumulative  = (1 + portfolio_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    return drawdown


def max_drawdown(portfolio_returns: pd.Series) -> float:
    """Maximum drawdown as a positive percentage loss."""
    return float(compute_drawdown_series(portfolio_returns).min() * 100)


# ── Rolling Risk ──────────────────────────────────────────────────────────────

def rolling_var(
    portfolio_returns: pd.Series,
    window:     int   = 63,    # ~1 quarter
    confidence: float = 0.95,
) -> pd.Series:
    """
    Rolling Historical VaR — shows how tail risk evolves over time.

    Spikes correspond to periods of market stress (crashes, crises).
    """
    return portfolio_returns.rolling(window).apply(
        lambda x: var_historical(pd.Series(x), confidence),
        raw=False,
    )


def rolling_volatility(
    portfolio_returns: pd.Series,
    window:       int = 21,    # ~1 month
    trading_days: int = config.TRADING_DAYS,
) -> pd.Series:
    """Annualised rolling volatility."""
    return portfolio_returns.rolling(window).std() * np.sqrt(trading_days)


# ── Full Risk Report ──────────────────────────────────────────────────────────

def compute_risk_report(
    portfolio_returns: pd.Series,
    weights:           np.ndarray,
    mean_returns:      np.ndarray,
    cov_matrix:        np.ndarray,
    label:             str   = "Portfolio",
    confidence:        float = 0.95,
) -> pd.DataFrame:
    """
    Generate a comprehensive risk report for a portfolio.

    Mirrors what a risk desk would produce for a daily P&L report.
    """
    report = {
        "Portfolio":            label,
        "Ann. Return (%)":      round(portfolio_returns.mean() * config.TRADING_DAYS * 100, 2),
        "Ann. Volatility (%)":  round(portfolio_returns.std() * np.sqrt(config.TRADING_DAYS) * 100, 2),
        "Sharpe Ratio":         round(
            (portfolio_returns.mean() * config.TRADING_DAYS - config.RISK_FREE_RATE)
            / (portfolio_returns.std() * np.sqrt(config.TRADING_DAYS)), 3
        ),
        "Skewness":             round(float(stats.skew(portfolio_returns)), 3),
        "Excess Kurtosis":      round(float(stats.kurtosis(portfolio_returns)), 3),
        f"VaR {int(confidence*100)}% Historical (%)":   round(var_historical(portfolio_returns, confidence) * 100, 3),
        f"VaR {int(confidence*100)}% Parametric (%)":   round(var_parametric(portfolio_returns, confidence) * 100, 3),
        f"VaR {int(confidence*100)}% Monte Carlo (%)":  round(var_monte_carlo(weights, mean_returns, cov_matrix, confidence) * 100, 3),
        f"CVaR {int(confidence*100)}% Historical (%)":  round(cvar_historical(portfolio_returns, confidence) * 100, 3),
        f"CVaR {int(confidence*100)}% Parametric (%)":  round(cvar_parametric(portfolio_returns, confidence) * 100, 3),
        "Max Drawdown (%)":     round(max_drawdown(portfolio_returns), 2),
    }

    return pd.DataFrame([report]).set_index("Portfolio").T