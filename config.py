"""
config.py — Central configuration for RL MLOps Trading System
NO API KEYS REQUIRED — pure paper trading via yfinance
"""
import os

# ── Symbols ──────────────────────────────────────────────────────
SYMBOLS  = ['RELIANCE', 'TCS', 'HDFCBANK', 'SBIN', 'ITC']
N_ASSETS = len(SYMBOLS)

# ── RL Environment ───────────────────────────────────────────────
WINDOW_SIZE        = 10
INITIAL_CAPITAL    = 100_000.0      # ₹1 lakh paper capital
TRANSACTION_COST   = 0.001          # 0.1% per trade (NSE realistic)
MAX_DRAWDOWN_LIMIT = 0.25
REWARD_SCALE       = 100.0

# ── Feature columns ──────────────────────────────────────────────
FEAT_COLS  = ['RSI', 'MACD_D', 'P_EMA9', 'P_EMA21',
              'ATR_PCT', 'BB_PCT', 'VOL_R', 'RET1', 'RET5', 'VOL20']
N_FEATURES = len(FEAT_COLS)

# ── PPO Hyperparameters ──────────────────────────────────────────
PPO_CONFIG = {
    "learning_rate"   : 3e-4,
    "n_steps"         : 512,
    "batch_size"      : 128,
    "n_epochs"        : 5,
    "gamma"           : 0.99,
    "gae_lambda"      : 0.95,
    "clip_range"      : 0.2,
    "ent_coef"        : 0.01,
    "total_timesteps" : 80_000,
}

# ── Trading (paper only — no broker API needed) ───────────────────
TRADE_MODE      = "paper"           # always paper; no live broker
TRADE_THRESHOLD = 0.20
SLEEP_SECONDS   = 300               # seconds between cycles

# ── Paths ────────────────────────────────────────────────────────
MODEL_DIR   = "models"
LOG_DIR     = "logs"
DATA_DIR    = "data"
MODEL_PATH  = os.path.join(MODEL_DIR, "best_model")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
VEC_PATH = os.path.join(MODEL_DIR, "vec_normalize.pkl")

# ── MLflow ───────────────────────────────────────────────────────
MLFLOW_URI        = "./mlruns"
MLFLOW_EXPERIMENT = "RL_MLOPS_TRADING"

# ── Prometheus ───────────────────────────────────────────────────
PROMETHEUS_PORT = 8000

# ── Dashboard ────────────────────────────────────────────────────
DASHBOARD_PORT  = 8050

# ── Drift detection ──────────────────────────────────────────────
DRIFT_THRESHOLD = 0.15              # Jensen-Shannon divergence

# ── Risk limits ──────────────────────────────────────────────────
MAX_POSITION_PCT = 0.40             # max 40% in one stock
MIN_POSITION_PCT = 0.00             # long-only
MAX_DAILY_LOSS   = 0.03             # halt if daily loss > 3%
