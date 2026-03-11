"""
optimization/markowitz.py
--------------------------
Implements Mean-Variance Optimization (MVO) as formulated by Harry Markowitz (1952).

Core idea:
    For a given level of expected return, find the portfolio weights that
    MINIMISE portfolio variance. Doing this across all feasible return levels
    traces out the EFFICIENT FRONTIER — the set of portfolios that offer the
    best possible return for each unit of risk taken.

Mathematical formulation:
    Variables : w ∈ R^N  (portfolio weights)
    Minimise  : w^T Σ w                      (portfolio variance)
    Subject to: w^T μ = μ_target             (target return constraint)
                1^T w = 1                    (weights sum to 1)
                w_i ≥ 0  ∀i                  (no short-selling, long-only)

    Where:
        Σ = annualised covariance matrix (N×N)
        μ = vector of annualised expected returns (N×1)

This is a convex Quadratic Program (QP) — guaranteed to find the global optimum.
We solve it using CVXPY, a domain-specific language for convex optimisation.

Key portfolios computed:
    - Minimum Variance Portfolio (MVP) : lowest risk regardless of return
    - Maximum Sharpe Ratio Portfolio   : best risk-adjusted return
    - Efficient Frontier               : full set of optimal portfolios
"""

import logging
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

log = logging.getLogger(__name__)


# ── Portfolio Metrics (building blocks) ──────────────────────────────────────

def portfolio_return(weights: np.ndarray, mean_returns: np.ndarray) -> float:
    """
    Compute annualised expected portfolio return.

    E[R_p] = w^T μ
    """
    return float(weights @ mean_returns)


def portfolio_volatility(weights: np.ndarray, cov_matrix: np.ndarray) -> float:
    """
    Compute annualised portfolio volatility (standard deviation).

    σ_p = sqrt(w^T Σ w)

    Note: we take the square root of variance to get volatility,
    which is expressed in the same units as returns (%).
    """
    variance = weights @ cov_matrix @ weights
    return float(np.sqrt(variance))


def sharpe_ratio(
    weights:      np.ndarray,
    mean_returns: np.ndarray,
    cov_matrix:   np.ndarray,
    risk_free:    float = config.RISK_FREE_RATE,
) -> float:
    """
    Compute the Sharpe Ratio for a given portfolio.

    SR = (E[R_p] - R_f) / σ_p

    The Sharpe Ratio measures excess return per unit of risk.
    Higher is better. A ratio above 1.0 is generally considered good.
    """
    ret = portfolio_return(weights, mean_returns)
    vol = portfolio_volatility(weights, cov_matrix)
    return (ret - risk_free) / vol if vol > 0 else 0.0


# ── Core Optimisation ─────────────────────────────────────────────────────────

def minimise_variance(
    mean_returns: np.ndarray,
    cov_matrix:   np.ndarray,
    target_return: float,
    allow_short:  bool = False,
) -> dict:
    """
    Solve the Markowitz QP for a single target return level.

    Returns a dict with:
        weights    : optimal weight vector
        return     : achieved portfolio return
        volatility : achieved portfolio volatility
        sharpe     : Sharpe ratio
        success    : whether the solver converged
    """
    N = len(mean_returns)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},                   # weights sum to 1
        {"type": "eq", "fun": lambda w: w @ mean_returns - target_return}, # hit target return
    ]

    bounds = None if allow_short else [(0.0, 1.0)] * N   # long-only by default

    # Initial guess: equal-weight portfolio
    w0 = np.ones(N) / N

    result = minimize(
        fun     = lambda w: portfolio_volatility(w, cov_matrix),
        x0      = w0,
        method  = "SLSQP",     # Sequential Least Squares Programming — standard for QPs
        bounds  = bounds,
        constraints = constraints,
        options = {"ftol": 1e-10, "maxiter": 1000},
    )

    if result.success:
        w = result.x
        w = np.clip(w, 0, 1)           # numerical clean-up
        w /= w.sum()                    # re-normalise after clip
        return {
            "weights":    w,
            "return":     portfolio_return(w, mean_returns),
            "volatility": portfolio_volatility(w, cov_matrix),
            "sharpe":     sharpe_ratio(w, mean_returns, cov_matrix),
            "success":    True,
        }

    log.warning("Optimiser did not converge for target return %.4f", target_return)
    return {"weights": w0, "return": np.nan, "volatility": np.nan, "sharpe": np.nan, "success": False}


def compute_efficient_frontier(
    mean_returns: np.ndarray,
    cov_matrix:   np.ndarray,
    n_points:     int  = 200,
    allow_short:  bool = False,
) -> pd.DataFrame:
    """
    Trace the full Efficient Frontier by solving the QP across a range
    of target returns — from the Minimum Variance Portfolio return up to
    the maximum single-asset return.

    Parameters
    ----------
    mean_returns : Annualised expected returns vector (N,).
    cov_matrix   : Annualised covariance matrix (N×N).
    n_points     : Number of points on the frontier (more = smoother curve).
    allow_short  : If False, enforce long-only constraints.

    Returns
    -------
    pd.DataFrame with columns: return, volatility, sharpe, weights_*
    """
    # Return range: from MVP return to max possible return
    min_ret = mean_returns.min()
    max_ret = mean_returns.max()
    target_returns = np.linspace(min_ret, max_ret, n_points)

    results = []
    for target in target_returns:
        res = minimise_variance(mean_returns, cov_matrix, target, allow_short)
        if res["success"]:
            results.append(res)

    frontier = pd.DataFrame({
        "return":     [r["return"]     for r in results],
        "volatility": [r["volatility"] for r in results],
        "sharpe":     [r["sharpe"]     for r in results],
    })

    log.info("Efficient Frontier computed: %d valid points", len(frontier))
    return frontier


# ── Special Portfolios ────────────────────────────────────────────────────────

def find_minimum_variance_portfolio(
    mean_returns: np.ndarray,
    cov_matrix:   np.ndarray,
    allow_short:  bool = False,
) -> dict:
    """
    Find the Global Minimum Variance Portfolio (MVP).

    This is the leftmost point on the Efficient Frontier — the portfolio
    with the lowest possible risk, regardless of expected return.
    Investors who are extremely risk-averse should hold this portfolio.

    Formulation (no return constraint):
        Minimise  : w^T Σ w
        Subject to: 1^T w = 1,  w_i ≥ 0
    """
    N = len(mean_returns)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = None if allow_short else [(0.0, 1.0)] * N
    w0          = np.ones(N) / N

    result = minimize(
        fun         = lambda w: portfolio_volatility(w, cov_matrix),
        x0          = w0,
        method      = "SLSQP",
        bounds      = bounds,
        constraints = constraints,
        options     = {"ftol": 1e-10, "maxiter": 1000},
    )

    w = result.x
    w = np.clip(w, 0, 1)
    w /= w.sum()

    log.info("Minimum Variance Portfolio — Vol: %.2f%%  Return: %.2f%%",
             portfolio_volatility(w, cov_matrix) * 100,
             portfolio_return(w, mean_returns) * 100)

    return {
        "weights":    w,
        "return":     portfolio_return(w, mean_returns),
        "volatility": portfolio_volatility(w, cov_matrix),
        "sharpe":     sharpe_ratio(w, mean_returns, cov_matrix),
        "label":      "Min Variance",
    }


def find_maximum_sharpe_portfolio(
    mean_returns: np.ndarray,
    cov_matrix:   np.ndarray,
    risk_free:    float = config.RISK_FREE_RATE,
    allow_short:  bool  = False,
) -> dict:
    """
    Find the Maximum Sharpe Ratio Portfolio (tangency portfolio).

    This is the portfolio on the Efficient Frontier that maximises
    risk-adjusted return. It corresponds to the point where the
    Capital Market Line (CML) is tangent to the Efficient Frontier.

    In the CAPM framework, this IS the market portfolio — every rational
    investor should hold this combination of risky assets, combined with
    the risk-free asset according to their personal risk tolerance.

    We MAXIMISE Sharpe by MINIMISING its negative.
    """
    N = len(mean_returns)

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds      = None if allow_short else [(0.0, 1.0)] * N
    w0          = np.ones(N) / N

    result = minimize(
        fun         = lambda w: -sharpe_ratio(w, mean_returns, cov_matrix, risk_free),
        x0          = w0,
        method      = "SLSQP",
        bounds      = bounds,
        constraints = constraints,
        options     = {"ftol": 1e-10, "maxiter": 1000},
    )

    w = result.x
    w = np.clip(w, 0, 1)
    w /= w.sum()

    log.info("Maximum Sharpe Portfolio     — Vol: %.2f%%  Return: %.2f%%  Sharpe: %.3f",
             portfolio_volatility(w, cov_matrix) * 100,
             portfolio_return(w, mean_returns) * 100,
             sharpe_ratio(w, mean_returns, cov_matrix, risk_free))

    return {
        "weights":    w,
        "return":     portfolio_return(w, mean_returns),
        "volatility": portfolio_volatility(w, cov_matrix),
        "sharpe":     sharpe_ratio(w, mean_returns, cov_matrix, risk_free),
        "label":      "Max Sharpe",
    }


# ── Results Formatting ────────────────────────────────────────────────────────

def format_portfolio(
    portfolio:   dict,
    tickers:     list[str],
    label:       str = "",
) -> pd.DataFrame:
    """
    Format a portfolio result into a clean summary DataFrame.
    """
    weights_df = pd.DataFrame({
        "Asset":       tickers,
        "Weight (%)":  np.round(portfolio["weights"] * 100, 2),
    }).sort_values("Weight (%)", ascending=False).reset_index(drop=True)

    print(f"\n── {label or portfolio.get('label', 'Portfolio')} ────────────────────────────────────")
    print(f"  Expected Return : {portfolio['return']*100:.2f}%")
    print(f"  Volatility      : {portfolio['volatility']*100:.2f}%")
    print(f"  Sharpe Ratio    : {portfolio['sharpe']:.3f}")
    print(f"\n  Weights:")
    print(weights_df[weights_df["Weight (%)"] > 0.01].to_string(index=False))

    return weights_df