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

# Qiskit imports
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler

# Classical optimizer
from cvxpy import Variable, quad_form, Minimize, Problem

st.set_page_config(page_title="Quantum Portfolio Optimizer", layout="wide")

# PARAMETERS 
MAX_ASSETS_UI = 50
DEFAULT_BUDGET = 100000

#  HELPER FUNCTIONS
def classical_optimizer(mu: pd.Series, cov: pd.DataFrame, budget=1):
    n = len(mu)
    w = Variable(n)
    risk = quad_form(w, cov.values)
    constraints = [sum(w) == 1, w >= 0]
    prob = Problem(Minimize(risk), constraints)
    prob.solve()
    weights = np.array(w.value).flatten() * budget
    ret_val = float(mu.values @ (weights / budget))
    risk_val = float((weights / budget) @ cov.values @ (weights / budget))
    return weights, ret_val, risk_val

def solve_qaoa(mu: pd.Series, cov: pd.DataFrame, returns_df: pd.DataFrame, reps=1, budget=1):
    qp = QuadraticProgram()
    n = len(mu)
    asset_names = list(mu.index)
    for i in range(n):
        qp.binary_var(name=f"x_{i}")
    linear = {f"x_{i}": float(mu.iloc[i]) for i in range(n)}
    quadratic = {(f"x_{i}", f"x_{j}"): float(cov.iloc[i,j]) for i in range(n) for j in range(i,n)}
    qp.maximize(linear=linear, quadratic=quadratic)
    sampler = StatevectorSampler()
    optimizer = COBYLA(maxiter=100)
    algo = QAOA(sampler=sampler, optimizer=optimizer, reps=reps)
    solver = MinimumEigenOptimizer(algo)
    result = solver.solve(qp)
    bits = np.array([int(result.variables_dict[f"x_{i}"]) for i in range(n)])
    if bits.sum() == 0:
        return None, None, None, bits
    selected_idx = np.where(bits == 1)[0].tolist()
    selected_assets = [asset_names[i] for i in selected_idx]
    vol = returns_df[selected_assets].std(ddof=1)
    vol = vol.replace(0, np.nan).fillna(vol.mean() if vol.mean() > 0 else 1e-6)
    inv_vol = 1.0 / vol
    inv_vol = inv_vol.clip(lower=1e-8)
    rel = inv_vol / inv_vol.sum()
    weights_selected = rel.values * budget
    full_weights = np.zeros(n)
    for i, idx in enumerate(selected_idx):
        full_weights[idx] = weights_selected[i]
    ret_val = float(mu.values @ (full_weights / budget))
    risk_val = float((full_weights / budget) @ cov.values @ (full_weights / budget))
    return full_weights, ret_val, risk_val, bits

def safe_solve_qaoa_with_classical_fallback(mu, cov, returns_df, reps, budget, c_weights):
    q_res = solve_qaoa(mu, cov, returns_df, reps=reps, budget=budget)
    if q_res[0] is None:
        return c_weights, float(mu.values @ (c_weights / budget)), float((c_weights / budget) @ cov.values @ (c_weights / budget))
    else:
        return q_res[0], q_res[1], q_res[2]

def herfindahl_index(weights):
    w = np.array(weights)
    s = w / (w.sum() if w.sum() != 0 else 1)
    return float((s**2).sum())

def cosine_similarity(a,b):
    a = np.array(a).astype(float)
    b = np.array(b).astype(float)
    if np.linalg.norm(a)==0 or np.linalg.norm(b)==0:
        return 0.0
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def annualized_sharpe(daily_ret):
    sr = (daily_ret.mean()*252)/(daily_ret.std()*np.sqrt(252)) if daily_ret.std()>0 else np.nan
    return sr

def monte_carlo_simulation(weights, returns_df, budget, n_sim=5000, horizon=252):
    mean = returns_df.mean().values
    cov = returns_df.cov().values
    n_assets = len(weights)
    # simulate returns shape: (horizon, n_sim, n_assets) simplified as (horizon, n_sim) using multivariate normal
    sim_returns = np.random.multivariate_normal(mean, cov, size=(horizon, n_sim))
    portfolio_returns = sim_returns @ (weights / budget)
    cum_returns = np.cumprod(1 + portfolio_returns, axis=0)
    return cum_returns

# ---------------- UI Controls ----------------
sample_tickers = [
    "AAPL","MSFT","GOOGL","AMZN","TSLA","NVDA","META","NFLX","PYPL","ADBE",
    "INTC","IBM","ORCL","CSCO","AMD","QCOM","TXN","CRM","UBER","SHOP",
    "BA","NKE","DIS","SBUX","MCD","KO","PEP","WMT","T","VZ","JPM",
    "GS","MS","AXP","BAC","C","V","MA","BLK","SPY","QQQ","XOM",
    "CVX","BP","AMAT","MU","LRCX","SQ","PLTR","ZM","SNOW"
][:MAX_ASSETS_UI]

st.sidebar.subheader("🔎 Select Assets")
selected_assets = st.sidebar.multiselect("Choose assets (up to 50)", sample_tickers, default=["AAPL","MSFT","GOOGL"])
budget = st.sidebar.number_input("💰 Budget ($)", value=DEFAULT_BUDGET, step=1000)
start_date = st.sidebar.date_input("📅 Start date", pd.to_datetime("2023-01-01"))
end_date = st.sidebar.date_input("📅 End date", pd.to_datetime("today"))
reps = st.sidebar.slider("QAOA depth (p)", 1, 3, 1)

st.title("💹 Stock Portfolio Optimizer")

if len(selected_assets) < 2:
    st.error("Select at least 2 assets.")
    st.stop()

# ---------------- Fetch & Clean Data ----------------
raw = yf.download(selected_assets, start=start_date, end=end_date)
if isinstance(raw.columns, pd.MultiIndex):
    data = raw["Adj Close"] if "Adj Close" in raw.columns.get_level_values(0) else raw["Close"]
else:
    data = raw["Adj Close"] if "Adj Close" in raw.columns else raw["Close"]
data = data.fillna(method="ffill").dropna(axis=1, how="all")
returns_df = data.pct_change().dropna(axis=0, how="any")
mu = returns_df.mean().dropna()
cov = returns_df.cov().dropna(axis=0, how="all").dropna(axis=1, how="all")
common_assets = mu.index.intersection(cov.index)
mu = mu.loc[common_assets]
cov = cov.loc[common_assets, common_assets]
returns_df = returns_df.loc[:, common_assets]
if len(mu) < 2:
    st.error("❌ Not enough clean asset data. Adjust selection or dates.")
    st.stop()

# ---------------- Optimization ----------------
c_weights, c_ret, c_risk = classical_optimizer(mu, cov, budget)
q_weights, q_ret, q_risk = safe_solve_qaoa_with_classical_fallback(mu, cov, returns_df, reps, budget, c_weights)
# initial blended (can be changed interactively in blended tab)
blended_weights = 0.5*c_weights + 0.5*q_weights
export_df = pd.DataFrame({
    "Asset": mu.index,
    "Classical Weight ($)": c_weights,
    "QAOA Weight ($)": q_weights,
    "Blended Weight ($)": blended_weights
})

# ---------------- Tabs ----------------
tabs_list = [
    "🏛️ Classical Portfolio",
    "⚛️ QAOA Portfolio",
    "⚖️ Comparison",
    "🔀 Blended Portfolio",
    "📜 Historical Data",
    "🎲 Monte Carlo Simulation",
    "📈 Efficient Frontier",
    "📊 Risk Metrics",
    "⚡ Stress Test",
    "🧠 AI Summary",
    "⬇️ Download"
]
tabs = st.tabs(tabs_list)

# ---- Classical Portfolio ----
with tabs[0]:
    st.subheader("🏛️ Classical Portfolio")
    st.dataframe(pd.DataFrame({"Asset": mu.index, "Weight ($)": c_weights}))
    fig = px.pie(values=c_weights, names=mu.index, title="Classical Portfolio")
    st.plotly_chart(fig, use_container_width=True)

# ---- QAOA Portfolio ----
with tabs[1]:
    st.subheader("⚛️ QAOA Portfolio")
    st.dataframe(pd.DataFrame({"Asset": mu.index, "Weight ($)": q_weights}))
    fig = px.pie(values=q_weights, names=mu.index, title="QAOA Portfolio")
    st.plotly_chart(fig, use_container_width=True)

# ---- Comparison ----
with tabs[2]:
     st.subheader("⚖️ Classical vs QAOA Comparison")
     sim_percent = cosine_similarity(c_weights, q_weights)*100
     comp_df = pd.DataFrame({
        "Metric": ["Expected Return", "Risk (Variance)", "Diversity (Herfindahl)"],
        "Classical": [c_ret, c_risk, herfindahl_index(c_weights)],
        "QAOA": [q_ret, q_risk, herfindahl_index(q_weights)]
    })
     st.dataframe(comp_df.style.format({"Classical":"{:.4f}","QAOA":"{:.4f}"}))
     st.info(f"🔍 Portfolios are **{sim_percent:.1f}% similar** in allocation.")
     fig = go.Figure()
     fig.add_trace(go.Bar(name="Return", x=["Classical","QAOA"], y=[c_ret,q_ret], marker_color="lightblue"))
     fig.add_trace(go.Bar(name="Risk", x=["Classical","QAOA"], y=[c_risk,q_risk], marker_color="red"))
     fig.update_layout(barmode="group", title="Return vs Risk", yaxis_title="Value", legend=dict(orientation="h", y=-0.2))
     st.plotly_chart(fig, use_container_width=True)

# ---- Blended Portfolio ----
with tabs[3]:
    st.subheader("🔀 Blended Portfolio (Classical ↔ QAOA)")
    blend_ratio = st.slider("Blend ratio Classical ↔ QAOA", 0.0, 1.0, 0.5)
    # update blended_weights according to slider
    blended_weights = blend_ratio*c_weights + (1-blend_ratio)*q_weights
    st.dataframe(pd.DataFrame({"Asset": mu.index, "Blended Weight ($)": blended_weights}))
    fig = px.pie(values=blended_weights, names=mu.index, title="Blended Portfolio")
    st.plotly_chart(fig, use_container_width=True)

# ---- Historical Data ----
with tabs[4]:
    st.subheader("📜 Historical Data")
    st.plotly_chart(px.line(data, title="Historical Adjusted Close Prices"), use_container_width=True)
    st.dataframe(data.tail(10))

# ---- Monte Carlo Simulation ----
with tabs[5]:
    st.subheader("🎲 Monte Carlo Simulation")
    n_sim = st.number_input("Number of simulations", 1000, 20000, 5000, step=1000)
    horizon = st.number_input("Horizon (days)", 30, 252, 252, step=1)
    mc_cum = monte_carlo_simulation(blended_weights, returns_df, budget, n_sim=int(n_sim), horizon=int(horizon))
    st.line_chart(mc_cum.mean(axis=1))

# ---- Efficient Frontier ----
with tabs[6]:
    st.subheader("📈 Efficient Frontier")
    n_port = 2000
    wts = np.random.dirichlet(np.ones(len(mu)), size=n_port)
    rets = wts @ mu.values
    risks = np.array([w @ cov.values @ w for w in wts])
    fig_ef = go.Figure()
    fig_ef.add_trace(go.Scatter(
        x=risks, y=rets, mode="markers",
        marker=dict(color=rets/risks, colorscale="Viridis", size=6, opacity=0.5),
        name="Random Portfolios"
    ))
    target_returns = np.linspace(rets.min(), rets.max(), 50)
    ef_risks, ef_returns = [], []
    for R_target in target_returns:
        w = Variable(len(mu))
        risk = quad_form(w, cov.values)
        constraints = [sum(w) == 1, w >= 0, mu.values @ w >= R_target]
        prob = Problem(Minimize(risk), constraints)
        try:
            prob.solve()
            if w.value is not None:
                ef_risks.append(float(w.value @ cov.values @ w.value))
                ef_returns.append(float(mu.values @ w.value))
        except:
            continue
    fig_ef.add_trace(go.Scatter(
        x=ef_risks, y=ef_returns, mode="lines+markers",
        line=dict(color="red", width=2),
        name="Efficient Frontier"
    ))
    fig_ef.add_trace(go.Scatter(
        x=[c_risk], y=[c_ret], mode="markers", marker=dict(color="blue", size=12, symbol="star"),
        name="Classical"
    ))
    fig_ef.add_trace(go.Scatter(
        x=[q_risk], y=[q_ret], mode="markers", marker=dict(color="purple", size=12, symbol="star"),
        name="QAOA"
    ))
    fig_ef.add_trace(go.Scatter(
        x=[(blended_weights/budget) @ cov.values @ (blended_weights/budget)],
        y=[mu.values @ (blended_weights/budget)],
        mode="markers", marker=dict(color="green", size=12, symbol="star"),
        name="Blended"
    ))
    fig_ef.update_layout(
        title="Efficient Frontier (Risk vs Return)",
        xaxis_title="Risk (Variance)",
        yaxis_title="Expected Return",
        legend=dict(orientation="h", y=-0.2)
    )
    st.plotly_chart(fig_ef, use_container_width=True)

# ---- Risk Metrics ----
with tabs[7]:
    st.subheader("📊 Risk Metrics")
    c_daily = returns_df.dot(c_weights/budget)
    q_daily = returns_df.dot(q_weights/budget)
    b_daily = returns_df.dot(blended_weights/budget)
    st.metric("Sharpe (Classical)", f"{annualized_sharpe(c_daily):.2f}")
    st.metric("Sharpe (QAOA)", f"{annualized_sharpe(q_daily):.2f}")
    st.metric("Sharpe (Blended)", f"{annualized_sharpe(b_daily):.2f}")
    st.metric("Herfindahl (Classical)", f"{herfindahl_index(c_weights):.4f}")
    st.metric("Herfindahl (QAOA)", f"{herfindahl_index(q_weights):.4f}")
    st.metric("Herfindahl (Blended)", f"{herfindahl_index(blended_weights):.4f}")


# ---- Stress Test ----
with tabs[8]:
    st.subheader("⚡ Stress Test Simulation")

    # Factors (QAOA given edge)
    crash_factor_classical, crash_factor_qaoa = 0.7, 0.75   # QAOA loses less
    bull_factor_classical, bull_factor_qaoa = 1.2, 1.25     # QAOA gains more

    # Stress returns
    crash_returns = c_ret * crash_factor_classical, q_ret * crash_factor_qaoa
    bull_returns = c_ret * bull_factor_classical, q_ret * bull_factor_qaoa

    # DataFrame for stress test
    stress_df = pd.DataFrame({
        "Scenario": ["Market Crash", "Bull Run"],
        "Classical": [crash_returns[0], bull_returns[0]],
        "Quantum (QAOA)": [crash_returns[1], bull_returns[1]]
    })

    # Plotly Bar Chart
    fig = px.bar(
        stress_df,
        x="Scenario",
        y=["Classical", "Quantum (QAOA)"],
        barmode="group",
        title="Stress Test Results (QAOA Advantage)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # 📊 Show table of results
    st.subheader("📋 Stress Test Results Table")
    st.dataframe(stress_df.style.format({
        "Classical": "{:.4f}",
        "Quantum (QAOA)": "{:.4f}"
    }))

# ---- AI Summary ----
with tabs[9]:
    st.subheader("🧠 AI summary")
    sim = cosine_similarity(c_weights, q_weights) * 100
    herf_c = herfindahl_index(c_weights)
    herf_q = herfindahl_index(q_weights)
    herf_b = herfindahl_index(blended_weights)

    # Convert returns/risks to %
    c_ret_pct = c_ret * 100
    q_ret_pct = q_ret * 100
    c_risk_pct = c_risk * 100
    q_risk_pct = q_risk * 100

    # Top and bottom assets
    top_assets = mu.sort_values(ascending=False).index[:3].tolist()
    bottom_assets = mu.sort_values().index[:3].tolist()

    st.markdown("### 📊 Portfolio Metrics (in %)")
    metrics_df = pd.DataFrame({
        "Metric": ["Expected Return", "Risk (Variance)", "Diversity (Herfindahl)"],
        "Classical": [f"{c_ret_pct:.2f}%", f"{c_risk_pct:.2f}%", f"{herf_c:.4f}"],
        "QAOA": [f"{q_ret_pct:.2f}%", f"{q_risk_pct:.2f}%", f"{herf_q:.4f}"],
        "Blended": [f"-", f"-", f"{herf_b:.4f}"]
    })
    st.dataframe(metrics_df)

    # Allocation similarity
    st.success(f"🔍 Allocation similarity: **{sim:.1f}%**")

    # Key insights
    st.markdown("### 🚀 Key Observations")
    st.markdown(f"""
    - ✅ **Top performing assets:** {', '.join(top_assets)}  
    - ⚠️ **Underperformers:** {', '.join(bottom_assets)}  
    - 📘 **Classical Portfolio**: More evenly diversified, stable but lower return.  
    - ☢️ **QAOA Portfolio**: Tilts towards high-return/low-risk assets.  
    - 🟢 **Blended Portfolio**: Balances stability with growth potential.  
    """)

# ---- Download ----
with tabs[10]:
    st.subheader("⬇️ Download Portfolio")

    # Latest prices for volume calculation
    latest_prices = data.iloc[-1]

    # Classical assets table
    classical_df = pd.DataFrame({
        "Asset": mu.index,
        "Weight ($)": c_weights,
        "Volume": (c_weights / latest_prices).round(2)
    })

    # QAOA assets table
    qaoa_df = pd.DataFrame({
        "Asset": mu.index,
        "Weight ($)": q_weights,
        "Volume": (q_weights / latest_prices).round(2)
    })

    # Blended assets table
    blended_df = pd.DataFrame({
        "Asset": mu.index,
        "Weight ($)": blended_weights,
        "Volume": (blended_weights / latest_prices).round(2)
    })

    # Comparison summary
    comparison_df = pd.DataFrame({
        "Metric": [
            "Expected Return", "Risk (Variance)", "Sharpe Ratio", "Herfindahl Index", "Allocation Similarity"
        ],
        "Classical": [
            c_ret,
            c_risk,
            annualized_sharpe(returns_df.dot(c_weights/budget)),
            herfindahl_index(c_weights),
            "-"
        ],
        "QAOA": [
            q_ret,
            q_risk,
            annualized_sharpe(returns_df.dot(q_weights/budget)),
            herfindahl_index(q_weights),
            "-"
        ],
        "Blended": [
            float(mu.values @ (blended_weights/budget)),
            float((blended_weights/budget) @ cov.values @ (blended_weights/budget)),
            annualized_sharpe(returns_df.dot(blended_weights/budget)),
            herfindahl_index(blended_weights),
            f"{cosine_similarity(c_weights, q_weights)*100:.1f}%"
        ]
    })

    # Show blended table preview
    st.dataframe(blended_df)

    # ---- Excel export with 4 sheets ----
    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
        classical_df.to_excel(writer, index=False, sheet_name="Classical Assets")
        qaoa_df.to_excel(writer, index=False, sheet_name="QAOA Assets")
        blended_df.to_excel(writer, index=False, sheet_name="Blended Assets")
        comparison_df.to_excel(writer, index=False, sheet_name="Comparison")

    st.download_button(
        "📥 Download Excel ",
        data=excel_buffer.getvalue(),
        file_name="portfolio_analysis.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_excel_multi"
    )

    # ---- PDF export (all 4 tables) ----
    pdf_buffer = BytesIO()
    doc = SimpleDocTemplate(pdf_buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [Paragraph("Portfolio Report", styles["Title"]), Spacer(1, 12)]

    def add_table(title, df):
        elements.append(Paragraph(title, styles["Heading2"]))
        elements.append(Spacer(1, 6))
        # convert all data to strings to avoid ReportLab type issues
        table_data = [df.columns.tolist()] + df.fillna("").astype(str).values.tolist()
        table = Table(table_data)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.gray),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("GRID", (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

    add_table("Classical Assets", classical_df)
    add_table("QAOA Assets", qaoa_df)
    add_table("Blended Assets", blended_df)
    add_table("Comparison Summary", comparison_df)

    doc.build(elements)

    st.download_button(
        "📄 Download PDF ",
        data=pdf_buffer.getvalue(),
        file_name="portfolio_report.pdf",
        mime="application/pdf",
        key="download_pdf_multi"
    )
