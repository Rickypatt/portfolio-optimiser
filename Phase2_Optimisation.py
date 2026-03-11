"""
phase2_optimisation.py
-----------------------
Phase 2 entry point: Markowitz Mean-Variance Optimisation.

Builds on Phase 1 by taking the returns and covariance matrix
and computing:
  1. The full Efficient Frontier
  2. The Global Minimum Variance Portfolio
  3. The Maximum Sharpe Ratio Portfolio (tangency portfolio)
  4. A comparison against the naive equal-weight benchmark

Run with:
    python phase2_optimisation.py
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

import config
from data.MarketData import compute_returns, compute_covariance_matrix, compute_summary_stats
from optimisation.Markowitz import (
    compute_efficient_frontier,
    find_minimum_variance_portfolio,
    find_maximum_sharpe_portfolio,
    portfolio_return,
    portfolio_volatility,
    sharpe_ratio,
    format_portfolio,
)

# ── Style (consistent with Phase 1) ──────────────────────────────────────────
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

ACCENT   = "#00d4ff"
GOLD     = "#ffd700"
GREEN    = "#00ff9d"
RED      = "#ff4d6d"
FRONTIER = "#a78bfa"


def load_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """Load prices, compute returns and covariance matrix."""
    price_path = os.path.join(config.DATA_DIR, config.PRICE_FILE)

    if os.path.exists(price_path):
        prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
        print("  Loaded prices from cache (outputs/prices_raw.csv)")
    else:
        print("  Price cache not found — regenerating from Phase 1...")
        try:
            from data.market_data import fetch_prices
            prices = fetch_prices(config.DEFAULT_TICKERS, config.START_DATE, config.END_DATE)
        except ImportError:
            from data.Synthetic import generate_prices
            prices = generate_prices()

    # Exclude SPY — keep it as a benchmark, not an optimisation asset
    tickers = [t for t in prices.columns if t != "SPY"]
    prices_assets = prices[tickers]

    returns    = compute_returns(prices_assets, save=False)
    cov_matrix = compute_covariance_matrix(returns).values
    mean_returns = returns.mean().values * config.TRADING_DAYS  # annualise

    return returns, mean_returns, cov_matrix, tickers


def run_phase2():
    print("\n" + "═"*65)
    print("  PORTFOLIO OPTIMIZER  ·  Phase 2: Markowitz Optimisation")
    print("═"*65 + "\n")

    # ── 1. Load data ──────────────────────────────────────────────────────────
    returns, mean_returns, cov_matrix, tickers = load_data()

    # ── 2. Compute Efficient Frontier ─────────────────────────────────────────
    print("\n  Computing Efficient Frontier...")
    frontier = compute_efficient_frontier(mean_returns, cov_matrix, n_points=300)

    # ── 3. Key portfolios ─────────────────────────────────────────────────────
    mvp      = find_minimum_variance_portfolio(mean_returns, cov_matrix)
    max_sr   = find_maximum_sharpe_portfolio(mean_returns, cov_matrix)

    # Equal-weight benchmark
    N  = len(tickers)
    ew = np.ones(N) / N
    eq_weight = {
        "weights":    ew,
        "return":     portfolio_return(ew, mean_returns),
        "volatility": portfolio_volatility(ew, cov_matrix),
        "sharpe":     sharpe_ratio(ew, mean_returns, cov_matrix),
        "label":      "Equal Weight",
    }

    # ── 4. Print results ──────────────────────────────────────────────────────
    format_portfolio(mvp,      tickers, "Global Minimum Variance Portfolio")
    format_portfolio(max_sr,   tickers, "Maximum Sharpe Ratio Portfolio (Tangency)")
    format_portfolio(eq_weight, tickers, "Equal-Weight Benchmark")

    # ── 5. Comparison table ───────────────────────────────────────────────────
    print("\n── Portfolio Comparison ────────────────────────────────────────")
    comparison = pd.DataFrame([
        {"Portfolio": "Min Variance",   "Return (%)": round(mvp["return"]*100, 2),      "Vol (%)": round(mvp["volatility"]*100, 2),      "Sharpe": round(mvp["sharpe"], 3)},
        {"Portfolio": "Max Sharpe",     "Return (%)": round(max_sr["return"]*100, 2),    "Vol (%)": round(max_sr["volatility"]*100, 2),    "Sharpe": round(max_sr["sharpe"], 3)},
        {"Portfolio": "Equal Weight",   "Return (%)": round(eq_weight["return"]*100, 2), "Vol (%)": round(eq_weight["volatility"]*100, 2), "Sharpe": round(eq_weight["sharpe"], 3)},
    ])
    print(comparison.to_string(index=False))

    # ── 6. Charts ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 12), facecolor="#0f1117")
    fig.suptitle(
        "Phase 2  ·  Markowitz Mean-Variance Optimisation",
        fontsize=16, color="#eee", fontfamily="monospace", y=0.98
    )

    # --- Plot 1: Efficient Frontier (main) -----------------------------------
    ax1 = fig.add_subplot(2, 2, (1, 2))

    # Colour frontier by Sharpe ratio
    sc = ax1.scatter(
        frontier["volatility"] * 100,
        frontier["return"] * 100,
        c      = frontier["sharpe"],
        cmap   = "plasma",
        s      = 8,
        alpha  = 0.8,
        zorder = 3,
    )
    cbar = plt.colorbar(sc, ax=ax1, pad=0.01)
    cbar.set_label("Sharpe Ratio", color="#aaa", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#aaa")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaa")

    # Individual assets
    asset_colors = plt.cm.cool(np.linspace(0.1, 0.9, N))
    for i, ticker in enumerate(tickers):
        ax1.scatter(
            np.sqrt(cov_matrix[i, i]) * 100,
            mean_returns[i] * 100,
            color  = asset_colors[i],
            s      = 80,
            zorder = 5,
            edgecolors = "#222",
            linewidths = 0.8,
        )
        ax1.annotate(
            ticker,
            (np.sqrt(cov_matrix[i, i]) * 100, mean_returns[i] * 100),
            textcoords = "offset points",
            xytext     = (8, 4),
            fontsize   = 9,
            color      = "#ddd",
        )

    # Key portfolios
    for portfolio, color, marker, size in [
        (mvp,      GREEN, "*", 400),
        (max_sr,   GOLD,  "*", 400),
        (eq_weight, RED,  "D", 100),
    ]:
        ax1.scatter(
            portfolio["volatility"] * 100,
            portfolio["return"] * 100,
            color      = color,
            marker     = marker,
            s          = size,
            zorder     = 6,
            edgecolors = "#000",
            linewidths = 1,
        )

    # Capital Market Line (CML)
    # The CML connects the risk-free rate to the tangency (Max Sharpe) portfolio
    # and represents all possible combinations of the risk-free asset + market portfolio
    rf_vol = 0.0
    rf_ret = config.RISK_FREE_RATE * 100
    tang_vol = max_sr["volatility"] * 100
    tang_ret = max_sr["return"] * 100
    cml_x = np.linspace(0, tang_vol * 1.3, 100)
    slope = (tang_ret - rf_ret) / tang_vol
    cml_y = rf_ret + slope * cml_x
    ax1.plot(cml_x, cml_y, color=GOLD, linewidth=1.2,
             linestyle="--", alpha=0.7, label="Capital Market Line", zorder=2)
    ax1.scatter(rf_vol, rf_ret, color=GOLD, s=60, zorder=6, marker="o",
                label=f"Risk-Free Rate ({config.RISK_FREE_RATE*100:.1f}%)")

    # Legend
    legend_items = [
        mpatches.Patch(color=GREEN, label=f"Min Variance  (SR={mvp['sharpe']:.2f})"),
        mpatches.Patch(color=GOLD,  label=f"Max Sharpe    (SR={max_sr['sharpe']:.2f})"),
        mpatches.Patch(color=RED,   label=f"Equal Weight  (SR={eq_weight['sharpe']:.2f})"),
    ]
    ax1.legend(handles=legend_items, fontsize=9, framealpha=0.2,
               labelcolor="#ddd", loc="lower right")

    ax1.set_xlabel("Annualised Volatility (%)", fontsize=10)
    ax1.set_ylabel("Annualised Return (%)",     fontsize=10)
    ax1.set_title("Efficient Frontier with Capital Market Line", color="#ccc", fontsize=11)
    ax1.axhline(0, color="#555", linewidth=0.6, linestyle=":")
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Weight allocation — Min Variance ----------------------------
    ax2 = fig.add_subplot(2, 2, 3)
    _plot_weights(ax2, mvp["weights"], tickers, "Min Variance — Weight Allocation", GREEN)

    # --- Plot 3: Weight allocation — Max Sharpe ------------------------------
    ax3 = fig.add_subplot(2, 2, 4)
    _plot_weights(ax3, max_sr["weights"], tickers, "Max Sharpe — Weight Allocation", GOLD)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(config.DATA_DIR, "phase2_efficient_frontier.png")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print(f"\n── Chart saved → {out_path}")
    print("\n✓ Phase 2 complete.\n")

    return frontier, mvp, max_sr


def _plot_weights(ax, weights, tickers, title, color):
    """Horizontal bar chart of portfolio weights."""
    sorted_idx = np.argsort(weights)[::-1]
    sorted_w   = weights[sorted_idx]
    sorted_t   = [tickers[i] for i in sorted_idx]
    bars = ax.barh(sorted_t, sorted_w * 100, color=color, alpha=0.75, edgecolor="#222")

    for bar, w in zip(bars, sorted_w):
        if w > 0.005:
            ax.text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{w*100:.1f}%", va="center", fontsize=8, color="#ccc"
            )

    ax.set_xlabel("Weight (%)")
    ax.set_title(title, color="#ccc", fontsize=10)
    ax.set_xlim(0, max(weights) * 100 * 1.2)
    ax.grid(True, alpha=0.3, axis="x")
    ax.invert_yaxis()


if __name__ == "__main__":
    run_phase2()