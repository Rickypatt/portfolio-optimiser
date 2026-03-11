"""
dashboard/app.py
-----------------
Phase 4: Interactive Portfolio Optimizer Dashboard.

Built with Streamlit + Plotly. All computations delegate to the
same modules from Phases 1–3 — the dashboard is a pure UI layer.

Run with:
    streamlit run dashboard/app.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import config
from data.MarketData import (
    compute_returns, compute_covariance_matrix,
    compute_correlation_matrix, compute_summary_stats,
)
from optimisation.Markowitz import (
    compute_efficient_frontier,
    find_minimum_variance_portfolio,
    find_maximum_sharpe_portfolio,
    portfolio_return, portfolio_volatility, sharpe_ratio,
)
from risk.Analytics import (
    simulate_portfolio_paths, compute_drawdown_series,
    var_historical, var_parametric, var_monte_carlo,
    cvar_historical, cvar_parametric, rolling_volatility,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Portfolio Optimizer",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080b12;
    color: #e2e8f0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1117 0%, #0a0f1a 100%);
    border-right: 1px solid #1e2d3d;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #00d4ff;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e2d3d;
    padding-bottom: 0.4rem;
    margin-top: 1.5rem;
}

/* Main header */
.main-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.02em;
    line-height: 1.1;
}
.main-subtitle {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: #00d4ff;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* KPI cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.kpi-card {
    background: linear-gradient(135deg, #0d1420 0%, #111827 100%);
    border: 1px solid #1e2d3d;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, #00d4ff);
}
.kpi-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'Space Mono', monospace;
    font-size: 1.6rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
}
.kpi-sub {
    font-size: 0.72rem;
    color: #94a3b8;
    margin-top: 0.3rem;
}

/* Section headers */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #00d4ff;
    border-left: 3px solid #00d4ff;
    padding-left: 0.8rem;
    margin: 2rem 0 1rem 0;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: transparent;
    border-bottom: 1px solid #1e2d3d;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    background: transparent;
    border: none;
    padding: 0.6rem 1.2rem;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
    background: transparent !important;
}

/* Metric delta colours */
.positive { color: #00ff9d; }
.negative { color: #ff4d6d; }

/* Dataframe */
[data-testid="stDataFrame"] {
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
}

/* Slider labels */
.stSlider label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
}

/* Multiselect */
.stMultiSelect label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
}

/* Selectbox */
.stSelectbox label {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #94a3b8;
}

/* Divider */
hr {
    border-color: #1e2d3d;
    margin: 1.5rem 0;
}

/* Plot backgrounds match dark theme */
.js-plotly-plot .plotly .main-svg {
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly theme ──────────────────────────────────────────────────────────────
PLOTLY_THEME = dict(
    paper_bgcolor = "rgba(0,0,0,0)",
    plot_bgcolor  = "rgba(0,0,0,0)",
    font          = dict(family="Space Mono, monospace", color="#94a3b8", size=11),
    xaxis         = dict(gridcolor="#1e2d3d", linecolor="#1e2d3d", zerolinecolor="#1e2d3d"),
    yaxis         = dict(gridcolor="#1e2d3d", linecolor="#1e2d3d", zerolinecolor="#1e2d3d"),
    margin        = dict(l=50, r=20, t=40, b=50),
)

COLORS = {
    "frontier": "#a78bfa",
    "mvp":      "#00ff9d",
    "max_sr":   "#ffd700",
    "eq_w":     "#ff4d6d",
    "accent":   "#00d4ff",
    "assets":   px.colors.qualitative.Pastel,
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (cached)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_prices(tickers: tuple[str, ...]) -> pd.DataFrame:
    price_path = os.path.join(config.DATA_DIR, config.PRICE_FILE)
    if os.path.exists(price_path):
        prices = pd.read_csv(price_path, index_col=0, parse_dates=True)
        available = [t for t in tickers if t in prices.columns]
        return prices[available]

    # fallback: synthetic
    from data.Synthetic import generate_prices
    return generate_prices(list(tickers))


@st.cache_data(show_spinner=False)
def compute_all(tickers: tuple[str, ...], rf_rate: float):
    prices       = load_prices(tickers)
    returns      = compute_returns(prices, save=False)
    cov_matrix   = compute_covariance_matrix(returns).values
    mean_returns = returns.mean().values * config.TRADING_DAYS
    corr_matrix  = compute_correlation_matrix(returns)
    stats        = compute_summary_stats(returns)

    mvp    = find_minimum_variance_portfolio(mean_returns, cov_matrix)
    max_sr = find_maximum_sharpe_portfolio(mean_returns, cov_matrix, risk_free=rf_rate)
    frontier = compute_efficient_frontier(mean_returns, cov_matrix, n_points=250)

    return returns, mean_returns, cov_matrix, corr_matrix, stats, mvp, max_sr, frontier


# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown('<div class="main-title">PORT<br>FOLIO<br>OPT.</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Quant Analytics Engine</div>', unsafe_allow_html=True)

    st.markdown("---")

    # ── Universe ──────────────────────────────────────────────────────────────
    st.markdown("### Universe")
    ALL_TICKERS = ["AAPL", "MSFT", "JPM", "JNJ", "XOM", "GLD", "TLT",
                   "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "UNH", "V"]

    selected_tickers = st.multiselect(
        "Select assets",
        options   = ALL_TICKERS,
        default   = config.DEFAULT_TICKERS[:-1],   # exclude SPY
        help      = "Choose 3–10 assets for the optimisation universe",
    )

    if len(selected_tickers) < 3:
        st.error("Select at least 3 assets.")
        st.stop()

    # ── Parameters ────────────────────────────────────────────────────────────
    st.markdown("### Parameters")

    rf_rate = st.slider(
        "Risk-Free Rate (%)",
        min_value = 0.0, max_value = 8.0,
        value     = config.RISK_FREE_RATE * 100,
        step      = 0.1,
        format    = "%.1f%%",
    ) / 100

    confidence = st.slider(
        "VaR Confidence Level (%)",
        min_value = 90, max_value = 99,
        value     = 95, step = 1,
        format    = "%d%%",
    ) / 100

    mc_horizon = st.slider(
        "Monte Carlo Horizon (days)",
        min_value = 63, max_value = 756,
        value     = 252, step = 63,
    )

    mc_sims = st.select_slider(
        "Monte Carlo Simulations",
        options = [500, 1000, 2000, 5000],
        value   = 1000,
    )

    st.markdown("### Constraints")
    allow_short = st.toggle("Allow Short Selling", value=False)

    st.markdown("---")
    st.markdown(
        '<div style="font-family:Space Mono,monospace;font-size:0.6rem;'
        'color:#334155;text-align:center;line-height:1.8;">'
        'DATA · yfinance / synthetic GBM<br>'
        'OPTIMISER · SLSQP (scipy)<br>'
        'RISK · Historical · Parametric · MC<br>'
        '</div>',
        unsafe_allow_html=True
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN — compute
# ═══════════════════════════════════════════════════════════════════════════════

tickers_key = tuple(sorted(selected_tickers))

with st.spinner("Running optimisation engine..."):
    (returns, mean_returns, cov_matrix, corr_matrix,
     stats, mvp, max_sr, frontier) = compute_all(tickers_key, rf_rate)

tickers = list(returns.columns)
N       = len(tickers)

# Portfolio return series
ew_weights    = np.ones(N) / N
mvp_returns   = returns @ mvp["weights"]
maxsr_returns = returns @ max_sr["weights"]
ew_returns    = returns @ ew_weights


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER KPI BAR
# ═══════════════════════════════════════════════════════════════════════════════

col_title, col_space = st.columns([3, 1])
with col_title:
    st.markdown('<div class="main-title">Portfolio Optimizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Mean-Variance · Risk Analytics · Monte Carlo</div>',
                unsafe_allow_html=True)

st.markdown("---")

# KPI row
k1, k2, k3, k4, k5 = st.columns(5)

def kpi(col, label, value, sub="", accent="#00d4ff"):
    col.markdown(
        f'<div class="kpi-card" style="--accent:{accent};">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

kpi(k1, "Max Sharpe Return",
    f"{max_sr['return']*100:.1f}%",
    f"SR = {max_sr['sharpe']:.2f}", "#ffd700")
kpi(k2, "Max Sharpe Vol",
    f"{max_sr['volatility']*100:.1f}%",
    "Annualised σ", "#ffd700")
kpi(k3, "Min Var Volatility",
    f"{mvp['volatility']*100:.1f}%",
    f"SR = {mvp['sharpe']:.2f}", "#00ff9d")
kpi(k4, "VaR 95% (Max SR)",
    f"{var_historical(maxsr_returns, confidence)*100:.2f}%",
    f"1-day · {int(confidence*100)}% conf", "#ff4d6d")
kpi(k5, "Assets in Universe",
    f"{N}",
    f"{len(returns)} trading days", "#00d4ff")

st.markdown("<br>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "⬡  Efficient Frontier",
    "◈  Risk Analytics",
    "◎  Monte Carlo",
    "⊞  Asset Universe",
])


# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — Efficient Frontier
# ───────────────────────────────────────────────────────────────────────────────
with tab1:

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<div class="section-header">Efficient Frontier · Capital Market Line</div>',
                    unsafe_allow_html=True)

        fig = go.Figure()

        # Frontier coloured by Sharpe
        fig.add_trace(go.Scatter(
            x    = frontier["volatility"] * 100,
            y    = frontier["return"] * 100,
            mode = "markers",
            marker = dict(
                color     = frontier["sharpe"],
                colorscale= "Plasma",
                size      = 5,
                colorbar  = dict(
                    title      = dict(text="Sharpe", font=dict(color="#94a3b8", size=10)),
                    tickfont   = dict(color="#94a3b8", size=9),
                    thickness  = 12,
                ),
                showscale = True,
            ),
            name       = "Efficient Frontier",
            hovertemplate = (
                "<b>Efficient Frontier</b><br>"
                "Vol: %{x:.2f}%<br>Return: %{y:.2f}%<br>"
                "Sharpe: %{marker.color:.3f}<extra></extra>"
            ),
        ))

        # Capital Market Line
        tang_vol = max_sr["volatility"] * 100
        tang_ret = max_sr["return"] * 100
        rf_ret   = rf_rate * 100
        slope    = (tang_ret - rf_ret) / tang_vol
        cml_x    = np.linspace(0, tang_vol * 1.4, 100)
        cml_y    = rf_ret + slope * cml_x
        fig.add_trace(go.Scatter(
            x    = cml_x, y = cml_y,
            mode = "lines",
            line = dict(color=COLORS["max_sr"], width=1.5, dash="dash"),
            name = "Capital Market Line",
            hoverinfo = "skip",
        ))

        # Risk-free rate
        fig.add_trace(go.Scatter(
            x=[0], y=[rf_ret],
            mode   = "markers",
            marker = dict(color=COLORS["max_sr"], size=10, symbol="circle"),
            name   = f"Risk-Free ({rf_rate*100:.1f}%)",
            hovertemplate = f"Risk-Free Rate: {rf_ret:.1f}%<extra></extra>",
        ))

        # Individual assets
        for i, ticker in enumerate(tickers):
            vol_i = np.sqrt(cov_matrix[i, i]) * 100
            ret_i = mean_returns[i] * 100
            color_i = COLORS["assets"][i % len(COLORS["assets"])]
            fig.add_trace(go.Scatter(
                x    = [vol_i], y = [ret_i],
                mode = "markers+text",
                marker = dict(color=color_i, size=10,
                              line=dict(color="#0d1117", width=1.5)),
                text      = [ticker],
                textposition = "top right",
                textfont  = dict(size=10, color=color_i),
                name      = ticker,
                hovertemplate = (
                    f"<b>{ticker}</b><br>Vol: {vol_i:.1f}%<br>"
                    f"Return: {ret_i:.1f}%<extra></extra>"
                ),
            ))

        # Key portfolios
        for p, color, symbol, label in [
            (mvp,    COLORS["mvp"],    "star",    "Min Variance"),
            (max_sr, COLORS["max_sr"], "star",    "Max Sharpe"),
        ]:
            fig.add_trace(go.Scatter(
                x    = [p["volatility"] * 100],
                y    = [p["return"] * 100],
                mode = "markers+text",
                marker = dict(color=color, size=18, symbol=symbol,
                              line=dict(color="#000", width=1)),
                text = [label],
                textposition = "top left",
                textfont = dict(size=10, color=color),
                name = label,
                hovertemplate = (
                    f"<b>{label}</b><br>"
                    f"Vol: {p['volatility']*100:.2f}%<br>"
                    f"Return: {p['return']*100:.2f}%<br>"
                    f"Sharpe: {p['sharpe']:.3f}<extra></extra>"
                ),
            ))

        fig.update_layout(
            **PLOTLY_THEME,
            height  = 500,
            legend  = dict(font=dict(size=9, color="#94a3b8"),
                           bgcolor="rgba(0,0,0,0)"),
            xaxis_title = "Annualised Volatility (%)",
            yaxis_title = "Annualised Return (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header">Portfolio Weights</div>',
                    unsafe_allow_html=True)

        port_choice = st.radio(
            "View weights for",
            ["Max Sharpe", "Min Variance"],
            horizontal=True,
            label_visibility="collapsed",
        )
        w_data = max_sr if port_choice == "Max Sharpe" else mvp
        color  = COLORS["max_sr"] if port_choice == "Max Sharpe" else COLORS["mvp"]

        weights_df = pd.DataFrame({
            "Asset":  tickers,
            "Weight": w_data["weights"] * 100,
        }).sort_values("Weight", ascending=True)

        fig_w = go.Figure(go.Bar(
            x           = weights_df["Weight"],
            y           = weights_df["Asset"],
            orientation = "h",
            marker      = dict(
                color   = weights_df["Weight"],
                colorscale = [[0, "#1e2d3d"], [1, color]],
                line    = dict(color="rgba(0,0,0,0)"),
            ),
            text        = [f"{w:.1f}%" for w in weights_df["Weight"]],
            textposition= "outside",
            textfont    = dict(size=10, color="#94a3b8"),
            hovertemplate = "%{y}: %{x:.2f}%<extra></extra>",
        ))
        fig_w.update_layout(
            **PLOTLY_THEME,
            height      = 320,
            showlegend  = False,
            xaxis_title = "Weight (%)",
            xaxis_range = [0, weights_df["Weight"].max() * 1.25],
        )
        st.plotly_chart(fig_w, use_container_width=True)

        # Stats table for selected portfolio
        st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
        metrics_df = pd.DataFrame({
            "Metric": ["Return", "Volatility", "Sharpe", "VaR 95%", "CVaR 95%"],
            "Value":  [
                f"{w_data['return']*100:.2f}%",
                f"{w_data['volatility']*100:.2f}%",
                f"{w_data['sharpe']:.3f}",
                f"{var_historical(mvp_returns if port_choice=='Min Variance' else maxsr_returns, confidence)*100:.3f}%",
                f"{cvar_historical(mvp_returns if port_choice=='Min Variance' else maxsr_returns, confidence)*100:.3f}%",
            ]
        })
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)

    # Cumulative returns comparison
    st.markdown('<div class="section-header">Cumulative Performance · All Portfolios vs Benchmark</div>',
                unsafe_allow_html=True)

    cum_mvp   = (1 + mvp_returns).cumprod()
    cum_maxsr = (1 + maxsr_returns).cumprod()
    cum_ew    = (1 + ew_returns).cumprod()

    fig_cum = go.Figure()
    for series, name, color in [
        (cum_maxsr, "Max Sharpe",   COLORS["max_sr"]),
        (cum_mvp,   "Min Variance", COLORS["mvp"]),
        (cum_ew,    "Equal Weight", COLORS["eq_w"]),
    ]:
        fig_cum.add_trace(go.Scatter(
            x=series.index, y=series.values,
            mode="lines", name=name,
            line=dict(color=color, width=2),
            hovertemplate=f"<b>{name}</b><br>%{{x|%b %Y}}: %{{y:.3f}}x<extra></extra>",
        ))
    fig_cum.update_layout(**PLOTLY_THEME, height=280,
                          yaxis_title="Growth of $1",
                          legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"))
    st.plotly_chart(fig_cum, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — Risk Analytics
# ───────────────────────────────────────────────────────────────────────────────
with tab2:

    st.markdown('<div class="section-header">Value at Risk · Three Methodologies</div>',
                unsafe_allow_html=True)

    # VaR comparison table
    port_series = {
        "Max Sharpe":   maxsr_returns,
        "Min Variance": mvp_returns,
        "Equal Weight": ew_returns,
    }
    port_weights = {
        "Max Sharpe":   max_sr["weights"],
        "Min Variance": mvp["weights"],
        "Equal Weight": ew_weights,
    }

    var_rows = []
    for name, r in port_series.items():
        w = port_weights[name]
        var_rows.append({
            "Portfolio":      name,
            "Ann. Return (%)":   f"{r.mean()*config.TRADING_DAYS*100:.2f}%",
            "Ann. Vol (%)":   f"{r.std()*np.sqrt(config.TRADING_DAYS)*100:.2f}%",
            "Sharpe":         f"{(r.mean()*config.TRADING_DAYS - rf_rate)/(r.std()*np.sqrt(config.TRADING_DAYS)):.3f}",
            "VaR Hist (%)":   f"{var_historical(r, confidence)*100:.3f}%",
            "VaR Param (%)":  f"{var_parametric(r, confidence)*100:.3f}%",
            "VaR MC (%)":     f"{var_monte_carlo(w, mean_returns, cov_matrix, confidence)*100:.3f}%",
            "CVaR Hist (%)":  f"{cvar_historical(r, confidence)*100:.3f}%",
            "CVaR Param (%)": f"{cvar_parametric(r, confidence)*100:.3f}%",
            "Max DD (%)":     f"{compute_drawdown_series(r).min()*100:.2f}%",
        })

    st.dataframe(pd.DataFrame(var_rows).set_index("Portfolio"),
                 use_container_width=True)

    col_dd, col_rvol = st.columns(2)

    with col_dd:
        st.markdown('<div class="section-header">Drawdown Analysis</div>',
                    unsafe_allow_html=True)
        fig_dd = go.Figure()
        for name, r, color in [
            ("Max Sharpe",   maxsr_returns, COLORS["max_sr"]),
            ("Min Variance", mvp_returns,   COLORS["mvp"]),
            ("Equal Weight", ew_returns,    COLORS["eq_w"]),
        ]:
            dd = compute_drawdown_series(r) * 100
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd.values,
                fill="tozeroy", fillcolor=color.replace(")", ",0.1)").replace("rgb", "rgba"),
                mode="lines", name=name,
                line=dict(color=color, width=1.5),
                hovertemplate=f"<b>{name}</b><br>%{{x|%b %Y}}: %{{y:.2f}}%<extra></extra>",
            ))
        fig_dd.update_layout(**PLOTLY_THEME, height=300,
                             yaxis_title="Drawdown (%)",
                             legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_dd, use_container_width=True)

    with col_rvol:
        st.markdown('<div class="section-header">Rolling 21-Day Volatility (Ann.)</div>',
                    unsafe_allow_html=True)
        fig_rv = go.Figure()
        for name, r, color in [
            ("Max Sharpe",   maxsr_returns, COLORS["max_sr"]),
            ("Min Variance", mvp_returns,   COLORS["mvp"]),
            ("Equal Weight", ew_returns,    COLORS["eq_w"]),
        ]:
            rv = rolling_volatility(r, window=21) * 100
            fig_rv.add_trace(go.Scatter(
                x=rv.index, y=rv.values,
                mode="lines", name=name,
                line=dict(color=color, width=1.5),
                hovertemplate=f"<b>{name}</b><br>%{{x|%b %Y}}: %{{y:.2f}}%<extra></extra>",
            ))
        fig_rv.update_layout(**PLOTLY_THEME, height=300,
                             yaxis_title="Volatility (%)",
                             legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_rv, use_container_width=True)

    # Return distribution with VaR overlay
    st.markdown('<div class="section-header">Return Distribution · VaR & CVaR Overlay</div>',
                unsafe_allow_html=True)

    selected_port = st.selectbox("Select portfolio", list(port_series.keys()))
    r_sel = port_series[selected_port]
    w_sel = port_weights[selected_port]

    var_h  = var_historical(r_sel, confidence)
    var_p  = var_parametric(r_sel, confidence)
    var_mc = var_monte_carlo(w_sel, mean_returns, cov_matrix, confidence)
    cvar_h = cvar_historical(r_sel, confidence)

    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x      = r_sel * 100,
        nbinsx = 80,
        name   = "Daily Returns",
        marker = dict(color=COLORS["accent"], opacity=0.4,
                      line=dict(color=COLORS["accent"], width=0.3)),
        histnorm = "probability density",
    ))
    for val, color, label in [
        (-var_h,  COLORS["max_sr"], f"VaR Hist {confidence*100:.0f}%:  {var_h*100:.3f}%"),
        (-var_p,  COLORS["mvp"],    f"VaR Param {confidence*100:.0f}%: {var_p*100:.3f}%"),
        (-var_mc, COLORS["accent"], f"VaR MC {confidence*100:.0f}%:    {var_mc*100:.3f}%"),
        (-cvar_h, COLORS["eq_w"],   f"CVaR Hist {confidence*100:.0f}%: {cvar_h*100:.3f}%"),
    ]:
        fig_dist.add_vline(x=val*100, line=dict(color=color, width=1.5, dash="dash"),
                           annotation_text=label,
                           annotation_font=dict(size=9, color=color),
                           annotation_position="top left")

    fig_dist.update_layout(**PLOTLY_THEME, height=320,
                           xaxis_title="Daily Return (%)",
                           yaxis_title="Density",
                           showlegend=False)
    st.plotly_chart(fig_dist, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — Monte Carlo
# ───────────────────────────────────────────────────────────────────────────────
with tab3:

    st.markdown('<div class="section-header">Monte Carlo Forward Simulation</div>',
                unsafe_allow_html=True)

    mc_port = st.radio(
        "Portfolio", ["Max Sharpe", "Min Variance", "Equal Weight"],
        horizontal=True, label_visibility="collapsed",
    )
    mc_w = {"Max Sharpe": max_sr["weights"],
             "Min Variance": mvp["weights"],
             "Equal Weight": ew_weights}[mc_port]
    mc_color = {"Max Sharpe": COLORS["max_sr"],
                "Min Variance": COLORS["mvp"],
                "Equal Weight": COLORS["eq_w"]}[mc_port]

    with st.spinner(f"Simulating {mc_sims:,} paths..."):
        paths = simulate_portfolio_paths(
            weights       = mc_w,
            mean_returns  = mean_returns,
            cov_matrix    = cov_matrix,
            horizon_days  = mc_horizon,
            n_simulations = mc_sims,
            initial_value = 1_000_000,
            seed          = 42,
        )

    t_axis = np.arange(paths.shape[0])
    p5, p10, p25, p50, p75, p90, p95 = (
        np.percentile(paths, q, axis=1) for q in [5, 10, 25, 50, 75, 90, 95]
    )

    fig_mc = go.Figure()

    # Colour map for rgba bands
    MC_RGBA = {
        COLORS["max_sr"]: "255,215,0",
        COLORS["mvp"]:    "0,255,157",
        COLORS["eq_w"]:   "255,77,109",
        COLORS["accent"]: "0,212,255",
    }
    band_rgb = MC_RGBA.get(mc_color, "0,212,255")

    # Percentile bands
    for lo, hi, alpha in [(p5, p95, 0.08), (p10, p90, 0.12), (p25, p75, 0.2)]:
        fig_mc.add_trace(go.Scatter(
            x=np.concatenate([t_axis, t_axis[::-1]]),
            y=np.concatenate([hi / 1e6, lo[::-1] / 1e6]),
            fill="toself",
            fillcolor=f"rgba({band_rgb},{alpha})",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Sample paths
    step = max(1, mc_sims // 80)
    for i in range(0, min(mc_sims, 80) * step, step):
        fig_mc.add_trace(go.Scatter(
            x=t_axis, y=paths[:, i] / 1e6,
            mode="lines", line=dict(color=mc_color, width=0.4),
            opacity=0.08, showlegend=False, hoverinfo="skip",
        ))

    # Median
    fig_mc.add_trace(go.Scatter(
        x=t_axis, y=p50 / 1e6,
        mode="lines", name="Median",
        line=dict(color=mc_color, width=2.5),
        hovertemplate="Day %{x}<br>Median: $%{y:.3f}M<extra></extra>",
    ))
    fig_mc.add_hline(y=1.0, line=dict(color="#334155", width=1, dash="dot"),
                     annotation_text="Initial $1M",
                     annotation_font=dict(size=9, color="#64748b"))

    fig_mc.update_layout(
        **PLOTLY_THEME, height=420,
        xaxis_title="Trading Days",
        yaxis_title="Portfolio Value ($M)",
        legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
        title=dict(
            text=f"{mc_port} · {mc_sims:,} Paths · {mc_horizon}-Day Horizon",
            font=dict(size=11, color="#94a3b8"),
        ),
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    # Final value distribution + stats
    col_hist, col_stats = st.columns([2, 1])
    final_vals = paths[-1] / 1e6

    with col_hist:
        st.markdown('<div class="section-header">Final Value Distribution</div>',
                    unsafe_allow_html=True)
        fig_fv = go.Figure()
        fig_fv.add_trace(go.Histogram(
            x=final_vals, nbinsx=60,
            marker=dict(color=mc_color, opacity=0.5,
                        line=dict(color=mc_color, width=0.3)),
            histnorm="probability density", name="Final Value",
        ))
        for pct, color, label in [
            (5,  COLORS["eq_w"],   "P5  (worst 5%)"),
            (50, mc_color,          "Median"),
            (95, COLORS["mvp"],    "P95 (best 5%)"),
        ]:
            val = np.percentile(final_vals, pct)
            fig_fv.add_vline(
                x=val, line=dict(color=color, width=1.5, dash="dash"),
                annotation_text=f"{label}: ${val:.2f}M",
                annotation_font=dict(size=9, color=color),
                annotation_position="top right" if pct >= 50 else "top left",
            )
        fig_fv.add_vline(x=1.0, line=dict(color="#475569", width=1, dash="dot"))
        fig_fv.update_layout(**PLOTLY_THEME, height=280,
                             xaxis_title="Portfolio Value ($M)",
                             yaxis_title="Density", showlegend=False)
        st.plotly_chart(fig_fv, use_container_width=True)

    with col_stats:
        st.markdown('<div class="section-header">Simulation Stats</div>',
                    unsafe_allow_html=True)

        prob_loss  = (final_vals < 1.0).mean() * 100
        prob_gain  = (final_vals > 1.0).mean() * 100
        expected   = np.mean(final_vals)

        sim_stats = pd.DataFrame({
            "Metric": [
                "Expected Final Value",
                "Median Final Value",
                "P5 (Worst 5%)",
                "P95 (Best 5%)",
                "Prob. of Loss",
                "Prob. of Gain",
                "Expected Return",
            ],
            "Value": [
                f"${expected:.3f}M",
                f"${np.median(final_vals):.3f}M",
                f"${np.percentile(final_vals, 5):.3f}M",
                f"${np.percentile(final_vals, 95):.3f}M",
                f"{prob_loss:.1f}%",
                f"{prob_gain:.1f}%",
                f"{(expected - 1.0) * 100:.1f}%",
            ]
        })
        st.dataframe(sim_stats, hide_index=True, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 4 — Asset Universe
# ───────────────────────────────────────────────────────────────────────────────
with tab4:

    col_stats_tab, col_corr = st.columns([1, 1])

    with col_stats_tab:
        st.markdown('<div class="section-header">Asset Summary Statistics</div>',
                    unsafe_allow_html=True)
        st.dataframe(stats.style.background_gradient(
            cmap="Blues", subset=["Ann. Return (%)"]
        ).background_gradient(
            cmap="Reds_r", subset=["Ann. Volatility (%)"]
        ), use_container_width=True)

    with col_corr:
        st.markdown('<div class="section-header">Correlation Matrix</div>',
                    unsafe_allow_html=True)
        corr_vals = corr_matrix.values
        fig_corr  = go.Figure(go.Heatmap(
            z          = corr_vals,
            x          = tickers,
            y          = tickers,
            colorscale = "RdBu_r",
            zmin=-1, zmax=1,
            text       = np.round(corr_vals, 2),
            texttemplate = "%{text}",
            textfont   = dict(size=10),
            hovertemplate = "%{y} / %{x}: %{z:.3f}<extra></extra>",
            colorbar   = dict(
                title    = dict(text="ρ", font=dict(color="#94a3b8", size=10)),
                tickfont = dict(color="#94a3b8", size=9),
                thickness= 12,
            ),
        ))
        fig_corr.update_layout(**PLOTLY_THEME, height=380)
        st.plotly_chart(fig_corr, use_container_width=True)

    # Individual price + returns charts
    st.markdown('<div class="section-header">Normalised Price History</div>',
                unsafe_allow_html=True)

    price_path = os.path.join(config.DATA_DIR, config.PRICE_FILE)
    if os.path.exists(price_path):
        prices_raw = pd.read_csv(price_path, index_col=0, parse_dates=True)
        prices_raw = prices_raw[[t for t in tickers if t in prices_raw.columns]]
        normalised = prices_raw / prices_raw.iloc[0] * 100

        fig_px = go.Figure()
        for i, ticker in enumerate(normalised.columns):
            color_i = COLORS["assets"][i % len(COLORS["assets"])]
            fig_px.add_trace(go.Scatter(
                x=normalised.index, y=normalised[ticker],
                mode="lines", name=ticker,
                line=dict(color=color_i, width=1.5),
                hovertemplate=f"<b>{ticker}</b><br>%{{x|%b %Y}}: %{{y:.1f}}<extra></extra>",
            ))
        fig_px.add_hline(y=100, line=dict(color="#334155", width=1, dash="dot"))
        fig_px.update_layout(**PLOTLY_THEME, height=320,
                             yaxis_title="Normalised Price (base=100)",
                             legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"))
        st.plotly_chart(fig_px, use_container_width=True)