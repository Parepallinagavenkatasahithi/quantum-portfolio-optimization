⚛️ Quantum Portfolio Optimizer

Hybrid Classical–Quantum Portfolio Optimization using QAOA

📌 Project Overview

The Quantum Portfolio Optimizer is a hybrid financial optimization system that combines classical mean–variance optimization with a quantum-inspired algorithm (QAOA) to construct risk-aware investment portfolios.

The system demonstrates how quantum optimization techniques can be applied to real-world financial problems while remaining executable on classical simulators today.

🚀 Key Features

📈 Classical portfolio optimization using Markowitz Mean–Variance Model

⚛️ Quantum-inspired portfolio selection using QAOA

🔀 Blended portfolio combining classical and quantum strategies

📊 Risk metrics (Volatility, Sharpe Ratio, Diversification)

🎲 Monte Carlo simulation for future portfolio projection

📉 Efficient Frontier visualization

🔥 Correlation heatmaps and stress testing

🖥️ Interactive Streamlit dashboard

📥 Export results as Excel and PDF

🧠 System Evolution

Prototype Stage

Static visualization using portfolio.html

Used to validate portfolio behavior and outputs

Final System

Full Streamlit-based interactive application

Hybrid Classical + QAOA optimization

Advanced analytics and visualizations

⚙️ Technologies Used

Programming Language: Python

Frontend / Dashboard: Streamlit

Quantum Computing: Qiskit (QAOA, Optimization Module)

Optimization: CVXPY

Data Handling: NumPy, Pandas

Visualization: Plotly

Market Data: Yahoo Finance API (yFinance)

📂 Project Structure
├── quantum_portfolio_optimizer_final.py   # Main Streamlit application
├── portfolio.html                         # Early static visualization (prototype)
├── README.md                              # Project documentation

📊 Data Source

Source: Yahoo Finance (via yFinance)

Frequency: Daily adjusted close prices

Default Time Range:

Start Date: January 1, 2023

End Date: Current date (latest available trading day)

Date range is user-configurable via the UI

🔢 Algorithms Used
1. Classical Optimization

Markowitz Mean–Variance Optimization

Objective: Minimize portfolio risk

Solver: CVXPY

2. Quantum Optimization

Quantum Approximate Optimization Algorithm (QAOA)

Portfolio modeled as a QUBO problem

Assets represented as binary decision variables

Executed on a statevector simulator

3. Supporting Techniques

Sharpe Ratio–based asset selection

Inverse volatility weighting

Monte Carlo simulations

Risk and diversification metrics

📈 Evaluation Metrics

Expected Return

Portfolio Variance & Volatility

Sharpe Ratio

Herfindahl Index (Diversification)

Cosine Similarity (Classical vs QAOA)

▶️ How to Run the Project
1️⃣ Install Dependencies
pip install streamlit numpy pandas yfinance plotly qiskit qiskit-optimization cvxpy

2️⃣ Run the Application
streamlit run quantum_portfolio_optimizer_final.py

🧪 Example Use Cases

Retail and institutional portfolio optimization

Quantum finance research and experimentation

Risk management and stress testing

FinTech product prototyping

Educational demonstrations of quantum algorithms

🔮 Future Enhancements

Integration with real quantum hardware

Support for transaction costs and constraints

Multi-objective optimization

Real-time market data streaming

Larger asset universe with advanced heuristics

🏆 Hackathon Context

Event: Amaravati Quantum Valley Hackathon 2025

Theme: Quantum Optimization – Portfolio Optimization

Approach: Hybrid Classical + Quantum-Inspired System

📜 References

Includes foundational works by Markowitz (1952), Farhi et al. (QAOA), and recent research on quantum portfolio optimization using QAOA and quantum annealing.

✅ Summary

This project demonstrates a practical pathway for applying quantum-inspired optimization techniques to financial decision-making, bridging academic research and real-world applications.
