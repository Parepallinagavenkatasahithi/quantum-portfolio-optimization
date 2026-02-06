
import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO

from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

# Qiskit
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler

# Classical optimizer
from cvxpy import Variable, quad_form, Minimize, Problem

#Config
st.set_page_config(page_title="Quantum Portfolio Optimizer", layout="wide")

MAX_ASSETS_UI = 50
MAX_QAOA_ASSETS = 10
DEFAULT_BUDGET = 100000

#Helper Functions
@st.cache_data(show_spinner=False)
def fetch_data(tickers, start, end):
    raw = yf.download(tickers, start=start, end=end)
    if raw.empty:
        raise ValueError("No data returned")

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Adj Close"] if "Adj Close" in raw.columns.get_level_values(0) else raw["Close"]
    else:
        prices = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]

    return prices.ffill().dropna(axis=1, how="all")

def classical_optimizer(mu, cov, budget=1):
    n = len(mu)
    w = Variable(n)
    risk = quad_form(w, cov.values)
    constraints = [sum(w) == 1, w >= 0]
    prob = Problem(Minimize(risk), constraints)
    prob.solve()
    weights = np.array(w.value).flatten() * budget
    ret = float(mu.values @ (weights / budget))
    risk_val = float((weights / budget) @ cov.values @ (weights / budget))
    return weights, ret, risk_val

def select_qaoa_assets(mu, cov, returns_df, max_assets=10):
    vol = np.sqrt(np.diag(cov.values))
    sharpe = mu.values / np.where(vol == 0, 1e-6, vol)
    selected = pd.Series(sharpe, index=mu.index).sort_values(ascending=False).head(max_assets).index
    return mu.loc[selected], cov.loc[selected, selected], returns_df[selected]

@st.cache_resource(show_spinner=False)
def solve_qaoa(mu, cov, returns_df, reps=1, budget=1, risk_aversion=0.5):
    """
    Solves the selection problem: Maximize [λ * Returns - (1 - λ) * Risk]
    """
    qp = QuadraticProgram()
    n = len(mu)
    for i in range(n):
        qp.binary_var(name=f"x_{i}")

    # Objective: Maximize (risk_aversion * returns) - ((1 - risk_aversion) * risk)
    # Risk is the quadratic term: x^T * Cov * x
    linear = {f"x_{i}": float(mu.iloc[i] * risk_aversion) for i in range(n)}
    
    # We multiply by -1 because we want to penalize risk in a MAXIMIZATION problem
    risk_penalty = -(1 - risk_aversion)
    quadratic = {
        (f"x_{i}", f"x_{j}"): float(cov.iloc[i, j] * risk_penalty) 
        for i in range(n) for j in range(i, n)
    }
    
    qp.maximize(linear=linear, quadratic=quadratic)

    # Optimization setup
    qaoa = QAOA(
        sampler=StatevectorSampler(),
        optimizer=COBYLA(maxiter=100),
        reps=reps
    )

    solver = MinimumEigenOptimizer(qaoa)
    result = solver.solve(qp)

    bits = np.array([int(result.variables_dict[f"x_{i}"]) for i in range(n)])
    
    if bits.sum() == 0:
        # Fallback: if no assets selected, pick the one with highest Sharpe
        idx = np.argmax(mu.values / np.sqrt(np.diag(cov.values)))
        bits[idx] = 1

    selected = np.where(bits == 1)[0]
    assets = mu.index[selected]
    
    # Inverse-volatility weighting for the selected subset
    vol = returns_df[assets].std().replace(0, 1e-6)
    inv_vol = 1 / vol
    weights = inv_vol / inv_vol.sum() * budget

    full_weights = np.zeros(len(mu))
    for i, idx in enumerate(selected):
        full_weights[idx] = weights.iloc[i]

    ret = float(mu.values @ (full_weights / budget))
    risk = float((full_weights / budget) @ cov.values @ (full_weights / budget))
    
    return full_weights, ret, risk
def herfindahl_index(weights):
    w = weights / weights.sum()
    return float((w ** 2).sum())

def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def annualized_sharpe(daily):
    return (daily.mean() * 252) / (daily.std() * np.sqrt(252)) if daily.std() > 0 else np.nan

def monte_carlo(weights, returns_df, budget, sims=5000, horizon=252):
    mean = returns_df.mean().values
    cov = returns_df.cov().values
    sims_ret = np.random.multivariate_normal(mean, cov, size=(horizon, sims))
    port_ret = sims_ret @ (weights / budget)
    return np.cumprod(1 + port_ret, axis=0)

def quantum_sensitivity_analysis(
    mu, cov, returns_df,
    reps=1,
    budget=1,
    risk_aversion=0.5,
    noise_level=0.05,
    runs=30
):
    """
    Measures how stable QAOA selections are under small perturbations.
    """
    n = len(mu)
    selection_counts = np.zeros(n)

    for _ in range(runs):
        noisy_mu = mu * (1 + np.random.normal(0, noise_level, size=len(mu)))
        weights, _, _ = solve_qaoa(
            noisy_mu,
            cov,
            returns_df,
            reps=reps,
            budget=budget,
            risk_aversion=risk_aversion
        )
        selection_counts += (weights > 0).astype(int)

    stability = selection_counts / runs
    return pd.DataFrame({
        "Asset": mu.index,
        "Quantum Selection Probability": stability
    }).sort_values("Quantum Selection Probability", ascending=False)

#Sidebar
sample_tickers = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","NFLX","PYPL","ADBE",
    "INTC","IBM","ORCL","CSCO","AMD","QCOM","TXN","CRM","UBER","SHOP",
    "BA","NKE","DIS","SBUX","MCD","KO","PEP","WMT","JPM","GS","MS",
    "V","MA","SPY","QQQ","XOM","CVX","AMAT","MU","LRCX","PLTR"
][:MAX_ASSETS_UI]

st.sidebar.subheader("🔎 Asset Selection")
assets = st.sidebar.multiselect("Choose assets", sample_tickers, ["AAPL","MSFT","GOOGL"])
budget = st.sidebar.number_input("💰 Budget ($)", DEFAULT_BUDGET, step=1000)
start = st.sidebar.date_input("📅 Start", pd.to_datetime("2023-01-01"))
end = st.sidebar.date_input("📅 End", pd.to_datetime("today"))
reps = st.sidebar.slider("⚛️ QAOA depth (p)", 1, 3, 1)

# Main 
st.title("💹 Quantum Portfolio Optimizer")

if len(assets) < 2:
    st.error("Select at least two assets")
    st.stop()

# Fetch prices
try:
    prices = fetch_data(assets, start, end)
except:
    st.error("Failed to fetch market data")
    st.stop()

returns_df = prices.pct_change().dropna()
mu = returns_df.mean()
cov = returns_df.cov()

# Optimization
c_weights, c_ret, c_risk = classical_optimizer(mu, cov, budget)
mu_q, cov_q, returns_q = select_qaoa_assets(mu, cov, returns_df, MAX_QAOA_ASSETS)
q_sub, q_ret, q_risk = solve_qaoa(mu_q, cov_q, returns_q, reps, budget)

q_weights = np.zeros(len(mu))
if q_sub is not None:
    for i, a in enumerate(mu_q.index):
        q_weights[mu.index.get_loc(a)] = q_sub[i]
else:
    q_weights = c_weights.copy()
    q_ret, q_risk = c_ret, c_risk

blend_ratio = 0.5
b_weights = blend_ratio * c_weights + (1 - blend_ratio) * q_weights

# Tabs
tabs = st.tabs([
    "🏛️ Classical Portfolio",
    "⚛️ QAOA Portfolio",
    "⚖️ Comparison",
    "🔀 Blended Portfolio",
    "📜 Historical Data",
    "🎲 Monte Carlo Simulation",
    "📡 Beta-Market Tracker",
    "📈 Efficient Frontier",
    "📊 Risk Metrics",
    "⚡ Stress Test",
    "🔥 Correlation Heatmap",
    "🔍 AI Summary",
    "🧬 Quantum Stability",
    "⬇️ Download"
])

# Classical Portfolio 
with tabs[0]:
    st.subheader("🏛️ Classical Portfolio Optimization")

    st.caption(
        "This portfolio is constructed using classical mean-variance optimization, "
        "focusing on minimizing overall portfolio risk."
    )

    #Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📈 Expected Return", f"{c_ret:.4f}")
    col2.metric("📉 Risk (Variance)", f"{c_risk:.6f}")
    col3.metric("⚖️ Volatility", f"{np.sqrt(c_risk):.4f}")
    col4.metric("🧮 Assets Allocated", int(np.count_nonzero(c_weights)))

    st.divider()

    #  Portfolio Data
    classical_df = pd.DataFrame({
        "Asset": mu.index,
        "Weight ($)": c_weights.round(2)
    })

    allocated_df = classical_df[classical_df["Weight ($)"] > 0]

    # Allocation Table
    st.subheader("🎯 Asset Allocation")
    st.dataframe(
        allocated_df.sort_values("Weight ($)", ascending=False),
        use_container_width=True
    )

    st.divider()

    # Donut Chart 
    st.subheader("🍩 Capital Allocation")

    classic_palette = [
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#ff7f0e",  # orange
        "#d62728",  # red
        "#f1c40f",  # yellow
        "#9467bd",
        "#7f7f7f"
    ]

    fig_classical = px.pie(
        allocated_df,
        values="Weight ($)",
        names="Asset",
        hole=0.45,
        color_discrete_sequence=classic_palette,
        title="Classical Portfolio Allocation"
    )

    fig_classical.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color="white", width=2))
    )

    fig_classical.update_layout(
        title_font_size=18,
        template="plotly_white"
    )

    st.plotly_chart(fig_classical, use_container_width=True)

    st.divider()

    #Explanation
    st.markdown("""
    ### 🧠 How the Classical portfolio is built
    - Uses **mean–variance optimization**
    - Objective is to **minimize risk**
    - All assets can receive capital
    - Portfolio is fully invested and diversified
    """)

    st.markdown(
        f"**Diversification (Herfindahl Index):** `{herfindahl_index(c_weights):.4f}`"
    )

# QAOA Portfolio
with tabs[1]:
    st.subheader("⚛️ Quantum Portfolio (QAOA)")

    st.caption(
        "QAOA selects a subset of assets using a quantum-inspired optimization "
        "and allocates capital using inverse-volatility weighting."
    )

    # Key metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Return", f"{q_ret:.4f}")
    col2.metric("Risk (Variance)", f"{q_risk:.6f}")
    col3.metric("Volatility", f"{np.sqrt(q_risk):.4f}")

    st.divider()

    # Portfolio allocation table
    qaoa_df = pd.DataFrame({
        "Asset": mu.index,
        "Weight ($)": q_weights.round(2)
    })

    selected_assets = qaoa_df[qaoa_df["Weight ($)"] > 0]
    unselected_assets = qaoa_df[qaoa_df["Weight ($)"] == 0]

    st.subheader("📊 Selected Assets")
    st.dataframe(
        selected_assets.sort_values("Weight ($)", ascending=False),
        use_container_width=True
    )

    # Allocation chart (only selected assets)
    st.subheader("🥧 Allocation Distribution")
    fig_qaoa = px.pie(
        selected_assets,
        values="Weight ($)",
        names="Asset",
        hole=0.4,
        title="QAOA Portfolio Allocation"
    )
    st.plotly_chart(fig_qaoa, use_container_width=True)

    st.divider()

    # Simple explanation
    st.markdown("""
    ### 🧠 How QAOA works (simplified)
    - Each asset is represented as a **binary decision** (select / not select)
    - QAOA searches for the best combination balancing:
        - 📈 Expected return  
        - 📉 Risk (covariance)
    - Capital is then allocated **only to selected assets**
    """)

    # Optional transparency section
    with st.expander("🔍 Show unselected assets"):
        st.dataframe(unselected_assets, use_container_width=True)

  # Comparison
with tabs[2]:
    sim = cosine_similarity(c_weights, q_weights) * 100
    df_comp = pd.DataFrame({
        "Metric": ["Return", "Variance", "Volatility", "Herfindahl"],
        "Classical": [c_ret, c_risk, np.sqrt(c_risk), herfindahl_index(c_weights)],
        "QAOA": [q_ret, q_risk, np.sqrt(q_risk), herfindahl_index(q_weights)]
    })
    st.dataframe(df_comp)
    st.success(f"Allocation similarity: {sim:.1f}%")
    
    # Add Bar Chart 
    comp_melted = df_comp.melt(id_vars="Metric", var_name="Portfolio", value_name="Value")
    fig_comp = px.bar(
        comp_melted,
        x="Metric",
        y="Value",
        color="Portfolio",
        barmode="group",
        text_auto=".4f",
        title="📊 Classical vs QAOA Portfolio Metrics"
    )
    st.plotly_chart(fig_comp, use_container_width=True)

# Blended Portfolio 
with tabs[3]:
    st.subheader("🔀 Blended Portfolio")

    st.caption(
        "The blended portfolio combines Classical and QAOA allocations, "
        "balancing stability and quantum-driven selection."
    )

    #  Blend Slider 
    ratio = st.slider(
        "Blend Ratio (Classical ↔ QAOA)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="1.0 = Fully Classical, 0.0 = Fully QAOA"
    )

    b_weights = ratio * c_weights + (1 - ratio) * q_weights

    #  Metrics
    b_ret = float(mu.values @ (b_weights / budget))
    b_risk = float((b_weights / budget) @ cov.values @ (b_weights / budget))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📈 Expected Return", f"{b_ret:.4f}")
    col2.metric("📉 Risk (Variance)", f"{b_risk:.6f}")
    col3.metric("⚖️ Volatility", f"{np.sqrt(b_risk):.4f}")
    col4.metric("🧮 Assets Allocated", int(np.count_nonzero(b_weights)))

    st.divider()

    #Portfolio Data
    blend_df = pd.DataFrame({
        "Asset": mu.index,
        "Classical ($)": c_weights.round(2),
        "QAOA ($)": q_weights.round(2),
        "Blended ($)": b_weights.round(2)
    })

    allocated_df = blend_df[blend_df["Blended ($)"] > 0]

    # Allocation Table
    st.subheader("🎯 Blended Asset Allocation")
    st.dataframe(
        allocated_df.sort_values("Blended ($)", ascending=False),
        use_container_width=True
    )

    st.divider()

    #Donut Chart (Consistent Colors) 
    st.subheader("🍩 Capital Allocation")

    classic_palette = [
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#ff7f0e",  # orange
        "#d62728",  # red
        "#f1c40f",  # yellow
        "#9467bd",
        "#7f7f7f"
    ]

    fig_blend = px.pie(
        allocated_df,
        values="Blended ($)",
        names="Asset",
        hole=0.45,
        color_discrete_sequence=classic_palette,
        title="Blended Portfolio Allocation"
    )

    fig_blend.update_traces(
        textposition="inside",
        textinfo="percent+label",
        marker=dict(line=dict(color="white", width=2))
    )

    fig_blend.update_layout(
        title_font_size=18,
        template="plotly_white"
    )

    st.plotly_chart(fig_blend, use_container_width=True)

    st.divider()

    # Explanation 
    st.markdown("""
    ### 🧠 How the Blended portfolio works
    - Combines **Classical** and **QAOA** portfolios
    - Slider controls the contribution of each method
    - Helps balance:
        - 📉 Stability (Classical)
        - ⚛️ Innovation (QAOA)
    """)

    st.markdown(
        f"**Diversification (Herfindahl Index):** `{herfindahl_index(b_weights):.4f}`"
    )

# Historical Data
with tabs[4]:
    st.subheader("📜 Historical Data")
    st.plotly_chart(px.line(prices, title="Historical Adjusted Close Prices"), use_container_width=True)
    st.dataframe(prices.tail(10))
    c_daily = returns_df.dot(c_weights / budget)
    q_daily = returns_df.dot(q_weights / budget)
    perf_df = pd.DataFrame({"Classical": (1 + c_daily).cumprod(), "QAOA": (1 + q_daily).cumprod()})
    st.plotly_chart(px.line(perf_df, title="Cumulative Growth (1.0 = start)"), use_container_width=True)

# Monte Carlo Simulation 
with tabs[5]:
    st.subheader("🎲 Monte Carlo Simulation")

    st.caption(
        "Projection of future portfolio value based on historical return behavior."
    )

    # Run simulation
    sims = 3000
    horizon = 252  # 1 trading year
    mc_paths = monte_carlo(b_weights, returns_df, budget, sims=sims, horizon=horizon)
    mc_df = pd.DataFrame(mc_paths)

    # Key statistics
    mean_path = mc_df.mean(axis=1)
    p25 = mc_df.quantile(0.25, axis=1)
    p75 = mc_df.quantile(0.75, axis=1)

    # Clean Plot 
    fig = go.Figure()

    # Confidence band (middle 50%)
    fig.add_trace(go.Scatter(
        y=p75,
        line=dict(width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        y=p25,
        fill="tonexty",
        fillcolor="rgba(31,119,180,0.18)",
        line=dict(width=0),
        name="Typical Range (25–75%)"
    ))

    # Expected path
    fig.add_trace(go.Scatter(
        y=mean_path,
        mode="lines",
        line=dict(color="#1f77b4", width=3),
        name="Expected Path"
    ))

    fig.update_layout(
        title="Monte Carlo Projection (1 Year)",
        xaxis_title="Time (Trading Days)",
        yaxis_title="Portfolio Growth Factor",
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Interpretation
    st.markdown("""
    ### 🧠 How to read this chart
    - **Blue line** → expected portfolio growth  
    - **Shaded band** → typical outcomes (middle 50%)  
    - Narrow band = more stable portfolio  
    """)

    #  Final Value Summary 
    final_vals = mc_df.iloc[-1]

    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Final Value", f"{final_vals.mean():.2f}")
    col2.metric("25% Downside", f"{final_vals.quantile(0.25):.2f}")
    col3.metric("75% Upside", f"{final_vals.quantile(0.75):.2f}")

#  Beta-Market Tracker 
with tabs[6]:
    for asset in assets:
        try:
            live_data = yf.Ticker(asset).history(period="1d", interval="1m")
            if not live_data.empty:
                fig_live = go.Figure(go.Candlestick(
                    x=live_data.index,
                    open=live_data["Open"], high=live_data["High"],
                    low=live_data["Low"], close=live_data["Close"]
                ))
                fig_live.update_layout(title=f"{asset} Live Candlestick", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig_live, use_container_width=True)
        except Exception as e:
            st.error(f"{asset}: {e}")

#  Efficient Frontier
with tabs[7]:
    st.subheader("📈 Efficient Frontier")

    st.caption(
        "Each point represents a possible portfolio. "
        "Better portfolios lie toward the top-left "
        "(higher return, lower risk)."
    )

    # Generate random portfolios
    n = 4000
    weights = np.random.dirichlet(np.ones(len(mu)), n)
    returns = weights @ mu.values
    risks = np.sqrt(np.array([w @ cov.values @ w for w in weights]))

    ef_df = pd.DataFrame({
        "Risk (Volatility)": risks,
        "Expected Return": returns
    })

    # Key portfolios
    c_vol = np.sqrt(c_risk)
    q_vol = np.sqrt(q_risk)
    b_vol = np.sqrt((b_weights / budget) @ cov.values @ (b_weights / budget))

    key_ports = pd.DataFrame({
        "Portfolio": ["Classical", "QAOA", "Blended"],
        "Risk (Volatility)": [c_vol, q_vol, b_vol],
        "Expected Return": [
            c_ret,
            q_ret,
            (returns_df @ (b_weights / budget)).mean()
        ]
    })

    # Efficient Frontier plot
    fig = px.scatter(
        ef_df,
        x="Risk (Volatility)",
        y="Expected Return",
        color="Expected Return",
        color_continuous_scale="Blues",
        opacity=0.45,
        title="Risk vs Expected Return",
        labels={
            "Risk (Volatility)": "Risk (Standard Deviation)",
            "Expected Return": "Expected Return"
        }
    )

    # Highlight Classical / QAOA / Blended
    fig.add_trace(go.Scatter(
        x=key_ports["Risk (Volatility)"],
        y=key_ports["Expected Return"],
        mode="markers+text",
        marker=dict(
            size=14,
            symbol="diamond",
            color="black"
        ),
        text=key_ports["Portfolio"],
        textposition="top center",
        name="Optimized Portfolios"
    ))

    fig.update_layout(
        template="plotly_white",
        coloraxis_colorbar=dict(
            title="Expected Return",
            thickness=14
        ),
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Simple explanation
    st.markdown("""
    ### 🧠 How to read this chart
    - **Left** → lower risk  
    - **Up** → higher return  
    - **Top-left region** → most efficient portfolios  
    - Highlighted points show optimized strategies
    """)

    # Summary table
    summary_df = key_ports.copy()
    summary_df["Sharpe Ratio"] = (
        summary_df["Expected Return"] / summary_df["Risk (Volatility)"]
    ).round(2)

    st.subheader("📊 Portfolio Summary")
    st.dataframe(summary_df.round(4))

#  Risk Metrics
with tabs[8]:
    st.subheader("📊 Portfolio Risk Metrics")

    def annualized_return(daily_returns):
        return float((1 + daily_returns.mean())**252 - 1)

    def annualized_vol(daily_returns):
        return float(daily_returns.std() * np.sqrt(252))

    portfolios = {
        "Classical": c_weights,
        "QAOA": q_weights,
        "Blended": b_weights
    }

    metrics_data = {
        "Portfolio": [],
        "Expected Return (%)": [],
        "Volatility (%)": [],
        "Sharpe Ratio": []
    }

    for name, weights in portfolios.items():
        daily_port = returns_df @ (weights / budget)
        metrics_data["Portfolio"].append(name)
        metrics_data["Expected Return (%)"].append(round(annualized_return(daily_port) * 100, 2))
        metrics_data["Volatility (%)"].append(round(annualized_vol(daily_port) * 100, 2))
        metrics_data["Sharpe Ratio"].append(round(annualized_sharpe(daily_port), 2))

    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df)

    # Optional: visual bar chart
    st.subheader("Visual Comparison")
    fig_risk = px.bar(
        metrics_df.melt(id_vars="Portfolio", var_name="Metric", value_name="Value"),
        x="Metric", y="Value", color="Portfolio", barmode="group",
        title="Portfolio Risk Metrics Comparison"
    )
    st.plotly_chart(fig_risk, use_container_width=True)

#  Stress Test 
with tabs[9]:
    st.subheader("⚡ Stress Test Scenarios")
    # Define stress scenarios
    crash_returns = c_ret * 0.7, q_ret * 0.75
    bull_returns = c_ret * 1.2, q_ret * 1.25

    stress_df = pd.DataFrame({
        "Scenario": ["Market Crash", "Bull Run"],
        "Classical": [crash_returns[0], bull_returns[0]],
        "Quantum (QAOA)": [crash_returns[1], bull_returns[1]]
    })

    # Bar chart
    st.plotly_chart(
        px.bar(
            stress_df,
            x="Scenario",
            y=["Classical", "Quantum (QAOA)"],
            barmode="group",
            text_auto=".2f",
            title="Portfolio Performance Under Stress Scenarios"
        ),
        use_container_width=True
    )

    # Table
    st.subheader("📋 Stress Test Table")
    st.dataframe(stress_df.style.format({
        "Classical": "{:.4f}",
        "Quantum (QAOA)": "{:.4f}"
    }))

# Correlation Heatmap 
with tabs[10]:
    corr_matrix = returns_df.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                         color_continuous_scale="RdBu_r", labels=dict(color="Correlation"),
                         x=corr_matrix.index, y=corr_matrix.columns)
    st.plotly_chart(fig_corr, use_container_width=True)

# AI Summary 
with tabs[11]:
    st.subheader("🔍 AI Portfolio Summary")

    # Metrics
    sim_val = cosine_similarity(c_weights, q_weights)
    herf_c = herfindahl_index(c_weights)
    herf_q = herfindahl_index(q_weights)
    sharpe_c = annualized_sharpe(returns_df @ (c_weights / budget))
    sharpe_q = annualized_sharpe(returns_df @ (q_weights / budget))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📈 Classical Return", f"{c_ret:.4f}")
    col2.metric("📉 Classical Risk", f"{c_risk:.6f}")
    col3.metric("⚖️ Herfindahl (Classical)", f"{herf_c:.4f}")
    col4.metric("💹 Sharpe Ratio", f"{sharpe_c:.2f}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📈 QAOA Return", f"{q_ret:.4f}")
    col2.metric("📉 QAOA Risk", f"{q_risk:.6f}")
    col3.metric("⚖️ Herfindahl (QAOA)", f"{herf_q:.4f}")
    col4.metric("💹 Sharpe Ratio", f"{sharpe_q:.2f}")

    st.divider()

    # Radar Chart for Metrics Comparison
    metrics_df = pd.DataFrame({
        "Metric": ["Return", "Risk", "Herfindahl", "Sharpe"],
        "Classical": [c_ret, c_risk, herf_c, sharpe_c],
        "QAOA": [q_ret, q_risk, herf_q, sharpe_q]
    })

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=metrics_df["Classical"],
        theta=metrics_df["Metric"],
        fill='toself',
        name='Classical'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=metrics_df["QAOA"],
        theta=metrics_df["Metric"],
        fill='toself',
        name='QAOA'
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        showlegend=True,
        title="📊 Portfolio Metrics Comparison"
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    st.divider()

    # Top and Bottom Assets
    sorted_mu = mu.sort_values(ascending=False)
    top_assets = sorted_mu.index[:3].tolist()
    bottom_assets = sorted_mu.index[-3:].tolist()

    st.markdown("### 🚀 Key Observations")
    st.markdown(f"- **Top Performing Assets:** {', '.join(top_assets)}")
    st.markdown(f"- **Underperforming Assets:** {', '.join(bottom_assets)}")
    st.markdown(f"- **Allocation Similarity (Classical vs QAOA):** {sim_val*100:.1f}%")
    
    # Optional: Visual gauge for allocation similarity
    st.markdown("#### 🔄 Allocation Similarity")
    st.progress(int(sim_val*100))
    
with tabs[12]:
    st.subheader("🧬 Quantum Stability & Sensitivity Analysis")

    st.caption(
        "This analysis measures how stable the QAOA portfolio is when "
        "market expectations are slightly perturbed."
    )

    noise = st.slider(
        "Market Noise Level (μ perturbation)",
        0.0, 0.15, 0.05, 0.01,
        help="Higher values simulate more uncertain markets"
    )

    runs = st.slider(
        "Quantum Re-runs",
        10, 60, 30, 5,
        help="More runs = stronger stability signal"
    )

    stability_df = quantum_sensitivity_analysis(
        mu_q,
        cov_q,
        returns_q,
        reps=reps,
        budget=budget,
        noise_level=noise,
        runs=runs
    )

    st.subheader("📊 Asset Selection Stability")
    st.dataframe(
        stability_df.style.format({
            "Quantum Selection Probability": "{:.2%}"
        }),
        use_container_width=True
    )

    fig = px.bar(
        stability_df,
        x="Asset",
        y="Quantum Selection Probability",
        title="Quantum Asset Stability Under Market Noise",
        color="Quantum Selection Probability",
        color_continuous_scale="Viridis"
    )

    fig.update_layout(
        yaxis_title="Selection Probability",
        xaxis_title="Asset",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    most_stable = stability_df.iloc[0]["Asset"]
    least_stable = stability_df.iloc[-1]["Asset"]

    st.markdown("### 🧠 Insights")
    st.markdown(f"- 🟢 **Most Quantum-Stable Asset:** `{most_stable}`")
    st.markdown(f"- 🔴 **Most Sensitive Asset:** `{least_stable}`")
    st.markdown(
        "- Assets with high stability are **robust to market uncertainty**, "
        "making them ideal long-term quantum selections."
    )

# Download
with tabs[13]:
    latest_prices = prices.iloc[-1]
    export_df = pd.DataFrame({
        "Asset": mu.index,
        "Current Price ($)": latest_prices.values.round(2),
        "Classical Weight ($)": c_weights.round(2),
        "QAOA Weight ($)": q_weights.round(2),
        "Classical Shares": (c_weights / latest_prices).round(2).values,
        "QAOA Shares": (q_weights / latest_prices).round(2).values
    })
    st.dataframe(export_df)
    
    # Excel export
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        export_df.to_excel(writer, index=False, sheet_name="Portfolio")
    st.download_button("📥 Download Excel", excel_buffer.getvalue(), "portfolio.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    
    # PDF export
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [Paragraph("Portfolio Optimization Report", styles["Title"]), Spacer(1, 12)]
    pdf_data = [export_df.columns.tolist()] + export_df.astype(str).values.tolist()
    table = Table(pdf_data, hAlign='LEFT')
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.dodgerblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    elements.append(table)
    doc.build(elements)
    st.download_button("📄 Download PDF", pdf_buffer.getvalue(), "portfolio_report.pdf", mime="application/pdf")
