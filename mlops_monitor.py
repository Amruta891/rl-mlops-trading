"""
mlops_monitor.py — Data drift detection, model performance monitoring,
                   Prometheus metrics export, alert logging
"""
import warnings; warnings.filterwarnings("ignore")
import os, json, logging
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from prometheus_client import (
    Gauge, Counter, Histogram, start_http_server, REGISTRY
)
import mlflow

import config

logger = logging.getLogger(__name__)

# ── Prometheus Metrics ───────────────────────────────────────────
_metrics_started = False

def start_prometheus(port=None):
    global _metrics_started
    if not _metrics_started:
        port = port or config.PROMETHEUS_PORT
        try:
            start_http_server(port)
            logger.info(f"Prometheus metrics served on :{port}")
            _metrics_started = True
        except OSError:
            logger.warning(f"Prometheus port {port} already in use — skipping")

# Gauges
PORTFOLIO_VALUE  = Gauge("portfolio_value",         "Current portfolio value (₹)")
PORTFOLIO_RETURN = Gauge("portfolio_return_pct",    "Cumulative return %")
DAILY_PNL        = Gauge("daily_pnl_pct",           "Today's PnL %")
SHARPE_LIVE      = Gauge("live_sharpe",             "Rolling 30-day Sharpe ratio")
DRAWDOWN         = Gauge("current_drawdown_pct",    "Current drawdown %")
DRIFT_SCORE      = Gauge("data_drift_score",        "Jensen-Shannon divergence vs training")
MODEL_VERSION    = Gauge("model_version",           "Current model version tag")

WEIGHT_GAUGES = {
    sym: Gauge(f"weight_{sym.lower()}", f"Portfolio weight for {sym}")
    for sym in config.SYMBOLS
}

PRICE_GAUGES = {
    sym: Gauge(f"price_{sym.lower()}", f"Latest close price for {sym}")
    for sym in config.SYMBOLS
}

TRADE_COUNTER = Counter("trades_total", "Total trades executed", ["symbol", "side"])
TRADE_ERRORS  = Counter("trade_errors_total", "Total trade errors")
CYCLE_TIME    = Histogram("cycle_duration_seconds", "Time per trading cycle")


# ── Drift Detection ──────────────────────────────────────────────
class DriftDetector:
    """
    Compares live feature distribution vs training baseline
    using Jensen-Shannon divergence per feature.
    """
    def __init__(self, train_df: pd.DataFrame):
        self.baseline = {}
        for col in config.FEAT_COLS:
            vals = train_df[col].dropna().values
            hist, edges = np.histogram(vals, bins=50, density=True)
            self.baseline[col] = (hist + 1e-9, edges)
        logger.info("DriftDetector initialised on training baseline")

    def score(self, live_df: pd.DataFrame) -> dict:
        scores = {}
        for col in config.FEAT_COLS:
            if col not in live_df.columns:
                continue
            vals = live_df[col].dropna().values
            if len(vals) < 5:
                continue
            ref_hist, edges = self.baseline[col]
            live_hist, _    = np.histogram(vals, bins=edges, density=True)
            live_hist = live_hist + 1e-9
            # Normalise
            p = ref_hist  / ref_hist.sum()
            q = live_hist / live_hist.sum()
            scores[col] = float(jensenshannon(p, q))
        return scores

    def is_drifted(self, live_df: pd.DataFrame) -> tuple:
        scores    = self.score(live_df)
        mean_drift= np.mean(list(scores.values())) if scores else 0.0
        flagged   = {k: v for k, v in scores.items()
                     if v > config.DRIFT_THRESHOLD}
        drifted   = mean_drift > config.DRIFT_THRESHOLD
        return drifted, mean_drift, scores, flagged


# ── Performance Tracker ──────────────────────────────────────────
class PerformanceTracker:
    """
    Tracks rolling portfolio metrics and logs to MLflow + Prometheus.
    """
    def __init__(self):
        self.history   = []      # list of {"time", "balance", "weights", "prices"}
        self.peak_bal  = config.INITIAL_CAPITAL
        self.start_bal = config.INITIAL_CAPITAL

    def update(self, balance, weights, prices, daily_pnl_pct=0.0):
        self.peak_bal = max(self.peak_bal, balance)
        dd = (self.peak_bal - balance) / (self.peak_bal + 1e-9) * 100
        cum_ret = (balance / self.start_bal - 1) * 100

        entry = {
            "time"        : datetime.now().isoformat(),
            "balance"     : round(balance, 2),
            "cum_return"  : round(cum_ret, 3),
            "daily_pnl"   : round(daily_pnl_pct, 3),
            "drawdown"    : round(dd, 3),
            "weights"     : {s: round(float(w), 4)
                             for s, w in zip(config.SYMBOLS, weights)},
            "prices"      : {s: round(float(p), 2)
                             for s, p in zip(config.SYMBOLS, prices)},
        }
        self.history.append(entry)

        # Prometheus
        PORTFOLIO_VALUE.set(balance)
        PORTFOLIO_RETURN.set(cum_ret)
        DAILY_PNL.set(daily_pnl_pct)
        DRAWDOWN.set(dd)
        for sym, w in zip(config.SYMBOLS, weights):
            WEIGHT_GAUGES[sym].set(float(w))
        for sym, p in zip(config.SYMBOLS, prices):
            PRICE_GAUGES[sym].set(float(p))

        # Rolling Sharpe (last 30 cycles)
        if len(self.history) >= 5:
            bals  = [h["balance"] for h in self.history[-30:]]
            rets  = np.diff(bals) / (np.array(bals[:-1]) + 1e-9)
            sharpe= float(np.mean(rets) / (np.std(rets) + 1e-9)) * np.sqrt(252)
            SHARPE_LIVE.set(sharpe)

        return entry

    def rolling_sharpe(self, n=30):
        if len(self.history) < 3:
            return 0.0
        bals = [h["balance"] for h in self.history[-n:]]
        rets = np.diff(bals) / (np.array(bals[:-1]) + 1e-9)
        return float(np.mean(rets) / (np.std(rets) + 1e-9)) * np.sqrt(252)

    def to_dataframe(self):
        return pd.DataFrame(self.history)

    def save(self, path="logs/performance.jsonl"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            for entry in self.history[-1:]:
                f.write(json.dumps(entry) + "\n")


# ── MLflow Cycle Logger ──────────────────────────────────────────
def log_cycle_to_mlflow(cycle: int, weights, prices, balance,
                         drift_score: float, trades: list, run_id=None):
    """Log one trading cycle's data to the active MLflow run."""
    try:
        metrics = {"cycle_balance": balance, "cycle_drift": drift_score}
        for sym, w in zip(config.SYMBOLS, weights):
            metrics[f"w_{sym}"] = float(w)
        for sym, p in zip(config.SYMBOLS, prices):
            metrics[f"p_{sym}"] = float(p)
        mlflow.log_metrics(metrics, step=cycle)

        if trades:
            trades_df = pd.DataFrame(trades)
            trades_df["cycle"] = cycle
            trades_path = f"logs/trades_cycle_{cycle}.csv"
            trades_df.to_csv(trades_path, index=False)
            mlflow.log_artifact(trades_path, artifact_path="trades")
    except Exception as e:
        logger.warning(f"MLflow log failed: {e}")


# ── Alert System ─────────────────────────────────────────────────
class AlertManager:
    """Simple rule-based alerting — logs to file and console."""

    def __init__(self, log_path="logs/alerts.log"):
        os.makedirs("logs", exist_ok=True)
        self.log_path = log_path

    def _write(self, level, msg):
        line = f"[{datetime.now().isoformat()}] [{level}] {msg}"
        logger.warning(line)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def check_all(self, balance, drawdown_pct, drift_score,
                  daily_pnl_pct, weights):
        alerts = []

        if drawdown_pct > config.MAX_DRAWDOWN_LIMIT * 100:
            msg = f"DRAWDOWN BREACH: {drawdown_pct:.1f}% > limit {config.MAX_DRAWDOWN_LIMIT*100:.0f}%"
            self._write("CRITICAL", msg); alerts.append(msg)

        if daily_pnl_pct < -config.MAX_DAILY_LOSS * 100:
            msg = f"DAILY LOSS LIMIT: {daily_pnl_pct:.2f}%"
            self._write("CRITICAL", msg); alerts.append(msg)

        if drift_score > config.DRIFT_THRESHOLD:
            msg = f"DATA DRIFT: JS={drift_score:.3f} > threshold {config.DRIFT_THRESHOLD}"
            self._write("WARNING", msg); alerts.append(msg)

        if balance < config.INITIAL_CAPITAL * 0.70:
            msg = f"CAPITAL EROSION: ₹{balance:,.0f} (< 70% of start)"
            self._write("CRITICAL", msg); alerts.append(msg)

        max_w = max(weights) if len(weights) else 0
        if max_w > config.MAX_POSITION_PCT:
            sym = config.SYMBOLS[int(np.argmax(weights))]
            msg = f"CONCENTRATION: {sym}={max_w*100:.1f}% > {config.MAX_POSITION_PCT*100:.0f}%"
            self._write("WARNING", msg); alerts.append(msg)

        return alerts
