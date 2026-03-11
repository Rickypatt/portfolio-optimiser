"""
phase3_risk.py
--------------
Phase 3 entry point: Portfolio Risk Analytics.

Takes the two key portfolios from Phase 2 (Min Variance, Max Sharpe)
and runs a full risk analysis on each, including:
  - VaR and CVaR (three methodologies each)
  - Monte Carlo forward simulation
  - Drawdown analysis
  - Rolling risk over time

Run with:
    python phase3_risk.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from scipy import stats

import config
from data.MarketData import compute_returns, compute_covariance_matrix
from optimisation.Markowitz import (
    find_minimum_variance_portfolio,
    find_maximum_sharpe_portfolio,
)
from risk.Analytics import (
    simulate_portfolio_paths,
    compute_drawdown_series,
    rolling_var,
    rolling_volatility,
    compute_risk_report,
    var_historical, var_parametric,
    cvar_historical,
)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#0f1117",
    "axes.edgecolor":   "#444",
    "axes.labelcolor":  "#ccc",
    "text.color":       "#eee",
    "xtick.color":      "#aaa",
    "ytick.color":      "#aaa",
    "grid.color":       "#2a2a2a",
    "grid.linestyle":   "--",
    "font.family":      "monospace",
})

GREEN  = "#00ff9d"
GOLD   = "#ffd700"
RED    = "#ff4d6d"
BLUE   = "#00d4ff"
PURPLE = "#a78bfa"


def load_phase2_results():
    """Reload data and recompute Phase 2 portfolios."""
    price_path = os.path.join(config.DATA_DIR, config.PRICE_FILE)
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)

    tickers       = [t for t in prices.columns if t != "SPY"]
    prices_assets = prices[tickers]

    returns      = compute_returns(prices_assets, save=False)
    cov_matrix   = compute_covariance_matrix(returns).values
    mean_returns = returns.mean().values * config.TRADING_DAYS

    mvp    = find_minimum_variance_portfolio(mean_returns, cov_matrix)
    max_sr = find_maximum_sharpe_portfolio(mean_returns, cov_matrix)

    return returns, mean_returns, cov_matrix, tickers, mvp, max_sr


def portfolio_return_series(weights, returns_df):
    """Compute historical daily portfolio return series."""
    return returns_df @ weights


def run_phase3():
    print("\n" + "═"*65)
    print("  PORTFOLIO OPTIMIZER  ·  Phase 3: Risk Analytics")
    print("═"*65 + "\n")

    # ── 1. Load data & Phase 2 portfolios ─────────────────────────────────────
    returns, mean_returns, cov_matrix, tickers, mvp, max_sr = load_phase2_results()

    # Equal-weight benchmark
    N  = len(tickers)
    ew = np.ones(N) / N

    portfolios = {
        "Min Variance": (mvp["weights"],  GREEN),
        "Max Sharpe":   (max_sr["weights"], GOLD),
        "Equal Weight": (ew,               RED),
    }

    # ── 2. Historical portfolio return series ─────────────────────────────────
    port_returns = {
        name: portfolio_return_series(w, returns)
        for name, (w, _) in portfolios.items()
    }

    # ── 3. Risk reports ───────────────────────────────────────────────────────
    print("\n── Risk Reports ────────────────────────────────────────────────")
    risk_reports = {}
    for name, (weights, _) in portfolios.items():
        report = compute_risk_report(
            portfolio_returns = port_returns[name],
            weights           = weights,
            mean_returns      = mean_returns,
            cov_matrix        = cov_matrix,
            label             = name,
            confidence        = 0.95,
        )
        risk_reports[name] = report
        print(f"\n  {name}:")
        print(report.to_string())

    # ── 4. Combined risk table ────────────────────────────────────────────────
    combined = pd.concat([risk_reports[n] for n in portfolios], axis=1)
    combined.columns = list(portfolios.keys())
    print("\n── Full Comparison ─────────────────────────────────────────────")
    print(combined.to_string())

    # ── 5. Monte Carlo simulation ─────────────────────────────────────────────
    print("\n  Running Monte Carlo simulations (1-year horizon)...")
    mc_paths = {
        name: simulate_portfolio_paths(
            weights       = weights,
            mean_returns  = mean_returns,
            cov_matrix    = cov_matrix,
            horizon_days  = 252,
            n_simulations = 2_000,
            initial_value = 1_000_000,
            seed          = 42,
        )
        for name, (weights, _) in portfolios.items()
    }

    # ── 6. Charts ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 16), facecolor="#0f1117")
    fig.suptitle(
        "Phase 3  ·  Portfolio Risk Analytics",
        fontsize=16, color="#eee", fontfamily="monospace", y=0.99
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # --- Plot 1: Monte Carlo paths — Max Sharpe (top, full width) ------------
    ax1 = fig.add_subplot(gs[0, :])
    paths = mc_paths["Max Sharpe"]
    t_axis = np.arange(paths.shape[0])

    # Fan chart: percentile bands
    p5, p25, p50, p75, p95 = (
        np.percentile(paths, q, axis=1) for q in [5, 25, 50, 75, 95]
    )
    ax1.fill_between(t_axis, p5,  p95, alpha=0.15, color=GOLD, label="5–95th pct")
    ax1.fill_between(t_axis, p25, p75, alpha=0.30, color=GOLD, label="25–75th pct")
    ax1.plot(t_axis, p50, color=GOLD, linewidth=2.0, label="Median path")

    # Sample individual paths (faint)
    for i in range(0, 200, 4):
        ax1.plot(t_axis, paths[:, i], color=GOLD, alpha=0.04, linewidth=0.5)

    ax1.axhline(1_000_000, color="#555", linewidth=0.8, linestyle="--")
    ax1.set_title("Monte Carlo Simulation — Max Sharpe Portfolio (1-Year, 2,000 paths)",
                  color="#ccc", fontsize=11)
    ax1.set_xlabel("Trading Days")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"${x/1e6:.2f}M")
    )
    ax1.legend(fontsize=9, framealpha=0.2, labelcolor="#ddd")
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Return distributions with VaR/CVaR markers -----------------
    ax2 = fig.add_subplot(gs[1, :2])
    for name, (_, color) in portfolios.items():
        r = port_returns[name]
        ax2.hist(r * 100, bins=80, alpha=0.35, color=color,
                 density=True, histtype="stepfilled", label=name)

    # VaR and CVaR lines for Max Sharpe
    r_ms = port_returns["Max Sharpe"]
    var_h  = var_historical(r_ms)
    cvar_h = cvar_historical(r_ms)
    ax2.axvline(-var_h  * 100, color=GOLD,   linewidth=1.5, linestyle="--",
                label=f"Max Sharpe VaR 95% ({-var_h*100:.2f}%)")
    ax2.axvline(-cvar_h * 100, color=RED,    linewidth=1.5, linestyle=":",
                label=f"Max Sharpe CVaR 95% ({-cvar_h*100:.2f}%)")

    # Fit normal for overlay
    mu_ms, sigma_ms = r_ms.mean(), r_ms.std()
    x_range = np.linspace(r_ms.min() * 100, r_ms.max() * 100, 300)
    ax2.plot(x_range,
             stats.norm.pdf(x_range, mu_ms * 100, sigma_ms * 100),
             color=GOLD, linewidth=1.2, linestyle="-", alpha=0.6,
             label="Normal fit (Max Sharpe)")

    ax2.set_title("Daily Return Distributions with VaR / CVaR", color="#ccc", fontsize=11)
    ax2.set_xlabel("Daily Return (%)")
    ax2.set_ylabel("Density")
    ax2.legend(fontsize=7.5, framealpha=0.2, labelcolor="#ddd")
    ax2.grid(True, alpha=0.3)

    # --- Plot 3: Drawdown series ---------------------------------------------
    ax3 = fig.add_subplot(gs[1, 2])
    for name, (_, color) in portfolios.items():
        dd = compute_drawdown_series(port_returns[name]) * 100
        ax3.fill_between(dd.index, dd, 0, alpha=0.3, color=color)
        ax3.plot(dd.index, dd, color=color, linewidth=0.8, label=name)

    ax3.set_title("Historical Drawdown (%)", color="#ccc", fontsize=11)
    ax3.set_ylabel("Drawdown (%)")
    ax3.legend(fontsize=8, framealpha=0.2, labelcolor="#ddd")
    ax3.grid(True, alpha=0.3)

    # --- Plot 4: Rolling VaR -------------------------------------------------
    ax4 = fig.add_subplot(gs[2, :2])
    for name, (_, color) in portfolios.items():
        rv = rolling_var(port_returns[name], window=63) * 100
        ax4.plot(rv.index, rv, color=color, linewidth=1.2,
                 alpha=0.85, label=name)

    ax4.set_title("Rolling 63-Day VaR 95% (%)", color="#ccc", fontsize=11)
    ax4.set_ylabel("VaR (%)")
    ax4.legend(fontsize=9, framealpha=0.2, labelcolor="#ddd")
    ax4.grid(True, alpha=0.3)

    # --- Plot 5: MC final value distribution ---------------------------------
    ax5 = fig.add_subplot(gs[2, 2])
    for name, (_, color) in portfolios.items():
        final_vals = mc_paths[name][-1] / 1e6
        ax5.hist(final_vals, bins=60, alpha=0.4, color=color,
                 density=True, histtype="stepfilled", label=name)
        ax5.axvline(np.percentile(final_vals, 5), color=color,
                    linewidth=1.5, linestyle="--", alpha=0.8)

    ax5.axvline(1.0, color="#fff", linewidth=1, linestyle=":", alpha=0.5,
                label="Initial value")
    ax5.set_title("MC Final Value Distribution (1Y)\nDashed = P5 (worst 5%)",
                  color="#ccc", fontsize=10)
    ax5.set_xlabel("Portfolio Value ($M)")
    ax5.set_ylabel("Density")
    ax5.legend(fontsize=8, framealpha=0.2, labelcolor="#ddd")
    ax5.grid(True, alpha=0.3)

    out_path = os.path.join(config.DATA_DIR, "phase3_risk_analytics.png")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print(f"\n── Chart saved → {out_path}")
    print("\n✓ Phase 3 complete.\n")


if __name__ == "__main__":
    run_phase3()