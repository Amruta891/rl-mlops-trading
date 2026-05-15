"""
main.py — Paper Trading Orchestrator (no broker API required)
Uses yfinance for live prices + RL model for decisions.

Usage:
    python main.py              # continuous paper loop
    python main.py --once       # single cycle and exit
    python main.py --no-trade   # inference only, no orders placed
"""
import warnings; warnings.filterwarnings("ignore")
import os, time, argparse, logging, joblib
import numpy as np
import mlflow

import config
from data_pipeline  import run_pipeline, build_obs, latest_prices
from execution      import PaperBroker, execute_rebalance, save_trades
from mlops_monitor  import (
    DriftDetector, PerformanceTracker, AlertManager,
    start_prometheus, log_cycle_to_mlflow, CYCLE_TIME, DRIFT_SCORE,
)

os.makedirs(config.LOG_DIR,   exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)

logging.basicConfig(
    level    = logging.INFO,
    format   = "%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(config.LOG_DIR, "main.log"), mode="a"),
    ],
)
logger = logging.getLogger("main")


# ── Load model artefacts ─────────────────────────────────────────
def load_artefacts():
    from stable_baselines3 import PPO
    zip_path = config.MODEL_PATH + ".zip"
    if not os.path.exists(zip_path):
        raise FileNotFoundError(
            f"Trained model not found at {zip_path}\n"
            "Please run:  python train.py"
        )
    model  = PPO.load(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    logger.info("Model + scaler loaded ✅")
    return model, scaler


# ── Single cycle ─────────────────────────────────────────────────
def trading_cycle(model, scaler, broker, drift_det, tracker,
                  alert_mgr, prev_weights, cycle, no_trade=False):

    t0 = time.time()
    logger.info(f"── Cycle {cycle} ──────────────────────────")

    # 1. Fetch live data + engineer features
    df = run_pipeline(period="60d")

    # 2. Build RL observation
    obs = build_obs(df, scaler)

    # 3. RL inference → portfolio weights
    action, _ = model.predict(obs, deterministic=True)
    action     = np.clip(action, 1e-8, 1.0)
    weights    = action / action.sum()
    logger.info("Weights: " + str(
        {s: round(float(w), 3) for s, w in zip(config.SYMBOLS, weights)}
    ))

    # 4. Data drift check
    drifted, drift_score, _, flagged = drift_det.is_drifted(df)
    DRIFT_SCORE.set(drift_score)
    if drifted:
        logger.warning(f"Drift detected (JS={drift_score:.3f}): {flagged}")

    # 5. Latest prices
    prices_dict = latest_prices(df)
    prices_arr  = np.array([prices_dict.get(s, 0.0) for s in config.SYMBOLS])

    # 6. Execute rebalance on paper broker
    trades = []
    if not no_trade:
        capital = broker.portfolio_value(prices_dict)
        trades, weights = execute_rebalance(
            broker, weights, prices_arr, capital, prev_weights
        )
        if trades:
            save_trades(trades)
            logger.info(f"Executed {len(trades)} trade(s)")
    else:
        logger.info("no-trade mode — skipping execution")

    # 7. Track performance
    balance   = broker.portfolio_value(prices_dict)
    daily_pnl = 0.0
    if tracker.history:
        daily_pnl = (balance / (tracker.history[-1]["balance"] + 1e-9) - 1) * 100

    entry = tracker.update(balance, weights, prices_arr, daily_pnl)

    # 8. Alerts
    alerts = alert_mgr.check_all(
        balance, entry["drawdown"], drift_score, daily_pnl, weights
    )
    for a in alerts:
        logger.critical(f"ALERT: {a}")

    # 9. MLflow
    log_cycle_to_mlflow(cycle, weights, prices_arr, balance, drift_score, trades)
    tracker.save()

    elapsed = time.time() - t0
    CYCLE_TIME.observe(elapsed)
    logger.info(
        f"Cycle {cycle} done in {elapsed:.1f}s | "
        f"Balance ₹{balance:,.0f} | Drift {drift_score:.3f}"
    )
    return weights


# ── Main ─────────────────────────────────────────────────────────
def main(run_once=False, no_trade=False):
    start_prometheus()

    model, scaler = load_artefacts()

    # Drift baseline
    csv_path = os.path.join(config.DATA_DIR, "rl_dataset.csv")
    if os.path.exists(csv_path):
        import pandas as pd
        drift_det = DriftDetector(pd.read_csv(csv_path))
    else:
        drift_det = DriftDetector(run_pipeline(period="500d"))

    broker    = PaperBroker(capital=config.INITIAL_CAPITAL)
    tracker   = PerformanceTracker()
    alert_mgr = AlertManager()

    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    prev_weights = np.full(config.N_ASSETS, 1.0 / config.N_ASSETS)
    cycle        = 0

    with mlflow.start_run(run_name="paper_trading"):
        mlflow.log_params({
            "trade_mode"  : "paper",
            "symbols"     : str(config.SYMBOLS),
            "capital"     : config.INITIAL_CAPITAL,
            "tx_cost"     : config.TRANSACTION_COST,
        })

        if run_once:
            prev_weights = trading_cycle(
                model, scaler, broker, drift_det, tracker,
                alert_mgr, prev_weights, cycle, no_trade,
            )
        else:
            logger.info(f"Paper trading loop started (sleep={config.SLEEP_SECONDS}s)")
            while True:
                try:
                    prev_weights = trading_cycle(
                        model, scaler, broker, drift_det, tracker,
                        alert_mgr, prev_weights, cycle, no_trade,
                    )
                    cycle += 1
                except KeyboardInterrupt:
                    logger.info("Stopped by user (Ctrl+C)")
                    break
                except Exception as e:
                    logger.error(f"Cycle {cycle} error: {e}", exc_info=True)

                logger.info(f"Sleeping {config.SLEEP_SECONDS}s …")
                time.sleep(config.SLEEP_SECONDS)

    logger.info(f"Session ended. Final: {broker.summary()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RL Paper Trading Loop")
    p.add_argument("--once",     action="store_true", help="Run one cycle then exit")
    p.add_argument("--no-trade", action="store_true", help="Inference only, no orders")
    args = p.parse_args()
    main(run_once=args.once, no_trade=args.no_trade)
