# RL Portfolio — MLOps Paper Trading System

Real-time NSE portfolio optimisation using PPO (Reinforcement Learning).
**No broker API keys required** — pure paper trading via yfinance.

---

## Project Structure

```
rl_mlops_trading/
├── config.py            # All settings (no .env needed)
├── data_pipeline.py     # yfinance fetch + feature engineering
├── rl_env.py            # Gymnasium PortfolioEnv
├── train.py             # PPO training + MLflow + Optuna tuning
├── execution.py         # Paper broker (realistic slippage + commission)
├── mlops_monitor.py     # Drift detection + Prometheus + alerts
├── dashboard.py         # Real-time Dash dashboard
├── main.py              # Paper trading loop (orchestrator)
├── requirements.txt
├── data/
│   └── rl_dataset.csv   # Pre-built training dataset (included)
├── models/              # Saved model artefacts (after training)
└── logs/                # Trade logs, alerts, metrics
```

---

## Quick Start (VS Code)

### 1. Install
```bash
cd rl_mlops_trading
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train.py                # default (80k steps, ~5 min)
python train.py --tune         # + Optuna search (better accuracy)
```

### 3. Launch dashboard
```bash
python dashboard.py
# Open http://localhost:8050
```

### 4. Start paper trading loop
```bash
python main.py                 # continuous loop (every 5 min)
python main.py --once          # single cycle then exit
python main.py --no-trade      # inference only, no orders
```

### 5. MLflow experiment UI
```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

---

## MLOps Stack

| Tool | Purpose |
|---|---|
| **MLflow** | Experiment tracking, metric logging, model registry |
| **Optuna** | Hyperparameter tuning (10-trial Bayesian search) |
| **Prometheus** | Real-time metrics on :8000 (portfolio, drift, Sharpe) |
| **JS-Divergence** | Per-feature data drift detection vs training baseline |
| **Dash + Plotly** | Live dashboard — candlestick, equity curve, weights, drift |
| **SB3 EvalCallback** | Best-model checkpoint during training |

## Paper Broker Features
- Simulated BUY / SELL with 0.05% slippage
- 0.1% commission per trade (NSE realistic)
- Integer share quantities only
- Balance + holdings tracked across session
- Rebalance threshold: only trades if weight shifts >1%
