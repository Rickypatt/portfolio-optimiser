"""
phase1_data.py
--------------
Phase 1 entry point: fetch market data, compute returns and risk matrices,
and print a clean summary to the console.

Run with:
    python phase1_data.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import numpy as np
import pandas as pd

import config
from data.MarketData import (
    fetch_prices,
    compute_returns,
    compute_covariance_matrix,
    compute_correlation_matrix,
    compute_summary_stats,
)

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#0f1117",
    "axes.edgecolor":    "#444",
    "axes.labelcolor":   "#ccc",
    "text.color":        "#eee",
    "xtick.color":       "#aaa",
    "ytick.color":       "#aaa",
    "grid.color":        "#2a2a2a",
    "grid.linestyle":    "--",
    "font.family":       "monospace",
})

ACCENT   = "#00d4ff"
POSITIVE = "#00ff9d"
NEGATIVE = "#ff4d6d"


def run_phase1():
    print("\n" + "═"*65)
    print("  PORTFOLIO OPTIMIZER  ·  Phase 1: Data Foundation")
    print("═"*65 + "\n")

    # ── 1. Fetch prices ───────────────────────────────────────────────────────
    try:
        prices = fetch_prices(
            tickers    = config.DEFAULT_TICKERS,
            start_date = config.START_DATE,
            end_date   = config.END_DATE,
            save       = True,
        )
    except ImportError:
        print("  [demo mode] yfinance not found — using synthetic GBM prices.\n")
        from data.Synthetic import generate_prices
        prices = generate_prices()
        os.makedirs(config.DATA_DIR, exist_ok=True)
        prices.to_csv(os.path.join(config.DATA_DIR, config.PRICE_FILE))

    # ── 2. Compute returns ────────────────────────────────────────────────────
    returns = compute_returns(prices, method=config.RETURN_TYPE, save=True)

    # ── 3. Covariance & Correlation ───────────────────────────────────────────
    cov  = compute_covariance_matrix(returns)
    corr = compute_correlation_matrix(returns)

    # ── 4. Summary statistics ─────────────────────────────────────────────────
    stats = compute_summary_stats(returns)

    print("\n── Asset Summary Statistics ────────────────────────────────────")
    print(stats.to_string())
    print()

    # ── 5. Plots ──────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 13), facecolor="#0f1117")
    fig.suptitle(
        "Phase 1  ·  Market Data Overview",
        fontsize=16, color="#eee", fontfamily="monospace", y=0.98
    )

    # Remove SPY from optimization assets for cleaner chart (keep as benchmark)
    assets = [t for t in returns.columns if t != "SPY"]

    # --- Plot 1: Normalised cumulative returns --------------------------------
    ax1 = fig.add_subplot(3, 2, (1, 2))
    cumulative = (1 + returns).cumprod()
    spy_cum    = cumulative["SPY"]
    colors     = plt.cm.cool(np.linspace(0.1, 0.9, len(assets)))

    for ticker, color in zip(assets, colors):
        ax1.plot(cumulative.index, cumulative[ticker],
                 label=ticker, linewidth=1.4, color=color, alpha=0.85)

    ax1.plot(cumulative.index, spy_cum,
             label="SPY (benchmark)", linewidth=2,
             color="#ffd700", linestyle="--", alpha=0.9)

    ax1.set_title("Cumulative Returns (base = 1.0)", color="#ccc", fontsize=11)
    ax1.legend(ncol=4, fontsize=8, framealpha=0.15, labelcolor="#ddd")
    ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax1.grid(True, alpha=0.4)

    # --- Plot 2: Annualised return vs volatility (risk/return scatter) --------
    ax2 = fig.add_subplot(3, 2, 3)
    ann_ret = stats["Ann. Return (%)"]
    ann_vol = stats["Ann. Volatility (%)"]

    scatter_colors = [POSITIVE if r > 0 else NEGATIVE for r in ann_ret]
    ax2.scatter(ann_vol, ann_ret, c=scatter_colors, s=90, zorder=5, edgecolors="#555", linewidths=0.5)

    for ticker in stats.index:
        ax2.annotate(
            ticker,
            (ann_vol[ticker], ann_ret[ticker]),
            textcoords="offset points", xytext=(6, 4),
            fontsize=8, color="#ddd"
        )

    ax2.axhline(0, color="#555", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("Ann. Volatility (%)")
    ax2.set_ylabel("Ann. Return (%)")
    ax2.set_title("Risk / Return per Asset", color="#ccc", fontsize=11)
    ax2.grid(True, alpha=0.4)

    # --- Plot 3: Correlation heatmap -----------------------------------------
    ax3 = fig.add_subplot(3, 2, 4)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    sns.heatmap(
        corr,
        ax         = ax3,
        mask       = mask,
        cmap       = "coolwarm",
        center     = 0,
        vmin       = -1, vmax = 1,
        annot      = True,
        fmt        = ".2f",
        annot_kws  = {"size": 8},
        linewidths = 0.5,
        linecolor  = "#222",
        cbar_kws   = {"shrink": 0.7},
    )
    ax3.set_title("Asset Correlation Matrix", color="#ccc", fontsize=11)
    ax3.tick_params(axis="x", labelrotation=45, labelsize=8)
    ax3.tick_params(axis="y", labelrotation=0,  labelsize=8)

    # --- Plot 4: Rolling 30-day volatility ------------------------------------
    ax4 = fig.add_subplot(3, 2, 5)
    rolling_vol = returns[assets].rolling(30).std() * np.sqrt(config.TRADING_DAYS) * 100

    for ticker, color in zip(assets, colors):
        ax4.plot(rolling_vol.index, rolling_vol[ticker],
                 label=ticker, linewidth=1.2, color=color, alpha=0.8)

    ax4.set_title("Rolling 30-Day Volatility (Ann. %)", color="#ccc", fontsize=11)
    ax4.set_ylabel("Volatility (%)")
    ax4.legend(ncol=4, fontsize=7, framealpha=0.15, labelcolor="#ddd")
    ax4.grid(True, alpha=0.4)

    # --- Plot 5: Return distribution (histogram) for each asset --------------
    ax5 = fig.add_subplot(3, 2, 6)
    for ticker, color in zip(assets, colors):
        ax5.hist(
            returns[ticker] * 100,
            bins=60, alpha=0.4,
            color=color, label=ticker,
            density=True, histtype="stepfilled"
        )

    ax5.axvline(0, color="#fff", linewidth=0.8, linestyle="--")
    ax5.set_title("Daily Return Distribution (%)", color="#ccc", fontsize=11)
    ax5.set_xlabel("Daily Return (%)")
    ax5.set_ylabel("Density")
    ax5.legend(ncol=2, fontsize=7, framealpha=0.15, labelcolor="#ddd")
    ax5.grid(True, alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(config.DATA_DIR, "phase1_overview.png")
    os.makedirs(config.DATA_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
    print(f"\n── Chart saved → {out_path}")

    print("\n── Covariance Matrix (annualised) ──────────────────────────────")
    print((cov * 100).round(4).to_string())
    print("\n✓ Phase 1 complete. Outputs saved to outputs/\n")

    return prices, returns, cov, corr, stats


if __name__ == "__main__":
    run_phase1()