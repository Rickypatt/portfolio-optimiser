<div align="center">

# 📈 Portfolio Optimiser

### Institutional-grade portfolio optimisation with walk-forward backtesting

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://your-app-name.streamlit.app)
[![License](https://img.shields.io/badge/License-MIT-00d4ff?style=flat)](LICENSE)
[![yfinance](https://img.shields.io/badge/Data-yfinance-4CAF50?style=flat)](https://pypi.org/project/yfinance/)

**[→ Open Live Dashboard](https://optimising-portfolio.streamlit.app/)**

*No install required — runs entirely in the browser*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Methodology](#methodology)
- [Features](#features)
- [Project Structure](#project-structure)
- [Running Locally](#running-locally)
- [Tech Stack](#tech-stack)
- [Limitations](#limitations)

---

## Overview

Most portfolio optimisers compute efficient frontier weights on historical data and stop there. That approach has a fundamental flaw: it uses the full historical record to find "optimal" weights — information that would never have been available in real time.

This project goes further on three fronts:

**1 · Eliminates look-ahead bias** via walk-forward backtesting. The optimiser is retrained at each rebalancing date using only data available up to that point. Out-of-sample performance is recorded, not in-sample fit.

**2 · Enforces real-world constraints** — position limits, sector concentration caps, volatility targets, and turnover limits. These are the constraints that actually appear in institutional investment mandates.

**3 · Benchmarks honestly** — every strategy is measured against S&P 500 buy-and-hold on the same capital base over the same period, using the same data source.

---

## Live Demo

Open the dashboard and paste in any Yahoo Finance tickers to analyse your own portfolio:

**[→ Launch Portfolio Optimiser](https://optimising-portfolio.streamlit.app/)**

The app fetches live market data from yfinance, runs the full optimisation and backtesting pipeline, and renders results interactively. No account, no install, no setup.

---

## Methodology

### Optimisation — Markowitz Mean-Variance

Solves the constrained quadratic programme at each rebalancing date:

```
minimise    w' Σ w
subject to  Σ w_i = 1                               (fully invested)
            w_i ∈ [w_min, w_max]                    (position limits)
            Σ_{i ∈ sector_k} w_i ≤ sector_cap  ∀k  (sector concentration)
            σ_p ≤ σ_target                          (volatility ceiling, optional)
            Σ |w_i - w_prev| ≤ turnover_limit       (rebalancing cost proxy, optional)
```

Solved via **SLSQP** (Sequential Least Squares Programming) from `scipy.optimize`.  
The tangency portfolio (maximum Sharpe ratio) is found by minimising the negative Sharpe subject to the same constraints.

---

### Backtesting — Walk-Forward

```
──────────────────────────────────────────────────────────────────
  t=0         t=252       t=315       t=378       t=441  ...
   │                │           │           │           │
   │◄─ lookback ───►│◄─ hold ──►│◄─ hold ──►│◄─ hold ──►│
                    ▲           ▲           ▲
               reoptimise  reoptimise  reoptimise
──────────────────────────────────────────────────────────────────
```

At each rebalancing date, the optimiser is trained **only on the estimation window ending that day**. Weights are held for one period, P&L is recorded, then the window rolls forward. This is the standard methodology used in quantitative strategy research to avoid look-ahead bias.

Configurable parameters:
- **Estimation window** — 3 months / 6 months / 1 year / 2 years
- **Rebalancing frequency** — monthly / quarterly / semi-annual / annual

---

### Risk Analytics

| Metric | Method |
|---|---|
| VaR | Historical (empirical quantile), Parametric (Gaussian), Monte Carlo |
| CVaR / Expected Shortfall | Mean loss beyond the VaR threshold |
| Drawdown | Peak-to-trough series and maximum drawdown |
| Sharpe Ratio | Rolling 63-day window |
| Sortino Ratio | Downside deviation only |
| Calmar Ratio | CAGR / Maximum Drawdown |
| Volatility | Rolling 21-day annualised |

Monte Carlo uses correlated GBM paths via Cholesky decomposition of the empirical covariance matrix.

---

## Features

### Efficient Frontier
- 4,000 random portfolios coloured by Sharpe ratio
- Constrained optimal portfolios — Max Sharpe and Min Variance
- Capital Market Line from the risk-free rate through the tangency portfolio
- Individual asset positions plotted in risk/return space

### Portfolio Constraints
| Constraint | Description |
|---|---|
| Min / Max weight | Per-asset allocation bounds — prevents concentration |
| Sector cap | Maximum total exposure to any GICS sector |
| Volatility target | Hard ceiling on annualised portfolio volatility |
| Turnover limit | Bounds weight changes per rebalance — implicit transaction cost model |

### Walk-Forward Backtest
- Out-of-sample equity curves for all strategies vs S&P 500 benchmark
- Drawdown analysis and rolling Sharpe across the full backtest period
- Weight allocation chart showing how the optimiser shifted positions over time
- Full performance table: Total Return, CAGR, Sharpe, Sortino, Max Drawdown, Calmar, Win Rate

### Risk Analytics
- VaR and CVaR comparison across three methodologies
- Monte Carlo simulation with configurable horizon and paths
- Return distribution with VaR/CVaR overlaid

### Asset Universe
- Correlation heatmap
- Summary statistics table
- Normalised price history (base = 100)

---

## Project Structure

```
portfolio-optimiser/
│
├── config.py                   # Single source of truth — tickers, dates, parameters
│
├── data/
│   ├── market_data.py          # yfinance download, returns, covariance
│   └── synthetic.py            # Correlated GBM fallback for offline / demo mode
│
├── optimization/
│   └── markowitz.py            # Efficient frontier, constrained Max Sharpe / Min Variance
│
├── risk/
│   └── analytics.py            # VaR, CVaR, drawdown, Monte Carlo simulation
│
├── backtest/
│   └── rolling.py              # Walk-forward engine, performance metrics
│
├── dashboard/
│   └── app.py                  # Streamlit UI — pure presentation layer
│
├── .streamlit/
│   └── config.toml             # Theme and server config for Streamlit Cloud
│
├── phase1_data.py              # Standalone script: data pipeline
├── phase2_optimisation.py      # Standalone script: optimisation
├── phase3_risk.py              # Standalone script: risk analytics
└── requirements.txt
```

Each `phase*.py` script runs independently and saves outputs to `outputs/`.  
The dashboard imports from the same modules — no duplication.

---

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/your-username/portfolio-optimiser.git
cd portfolio-optimiser
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the dashboard**
```bash
streamlit run dashboard/app.py
```

Opens at `http://localhost:8501`. Enter any Yahoo Finance tickers in the sidebar.

**4. (Optional) Run individual phase scripts**
```bash
python phase1_data.py            # fetch and cache price data
python phase2_optimisation.py    # compute efficient frontier
python phase3_risk.py            # run full risk report
```

---

## Tech Stack

| Layer | Library | Purpose |
|---|---|---|
| Data | `yfinance`, `pandas` | Price download and manipulation |
| Optimisation | `scipy.optimize` | SLSQP constrained QP solver |
| Statistics | `numpy`, `scipy.stats` | Covariance, distributions, simulation |
| Visualisation | `plotly` | All interactive charts |
| Dashboard | `streamlit` | Web interface |

---

## Limitations

**Distributional assumptions** — Returns are modelled as log-normal (Gaussian innovations). Fat tails are not explicitly modelled. CVaR from historical simulation partially compensates, but extreme events will be underestimated.

**Transaction costs** — Not explicitly modelled. The turnover constraint acts as a proxy by limiting how much the portfolio can change per rebalance.

**Survivorship bias** — Backtesting on a fixed ticker universe does not account for companies that were delisted or went bankrupt during the period. Historical performance will be overstated relative to a strategy implemented in real time.

**Liquidity** — The optimiser assumes all positions can be entered and exited at closing prices with no market impact.

---

<div align="center">

Built by **Riccardo Pattono**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/riccardopattono/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat&logo=github&logoColor=white)](https://github.com/Rickypatt/)

</div>