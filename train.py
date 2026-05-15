"""
train.py — PPO training with MLflow tracking + Optuna hyperparameter tuning
Run: python train.py [--tune]
"""
import warnings; warnings.filterwarnings("ignore")
import os, argparse, joblib, logging
import numpy as np
import pandas as pd
import mlflow
import mlflow.pytorch
import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from sklearn.preprocessing import RobustScaler
import torch.nn as nn

import config
from rl_env import PortfolioEnv
from data_pipeline import run_pipeline

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Callbacks ────────────────────────────────────────────────────
class MLflowCallback(BaseCallback):
    def __init__(self, log_freq=2000):
        super().__init__(); self.log_freq = log_freq

    def _on_step(self):
        if self.n_calls % self.log_freq == 0:
            buf = self.model.ep_info_buffer
            if buf:
                mean_r = np.mean([e["r"] for e in buf])
                mean_l = np.mean([e["l"] for e in buf])
                mlflow.log_metrics({
                    "mean_ep_reward": float(mean_r),
                    "mean_ep_length": float(mean_l),
                    "timestep"      : int(self.num_timesteps),
                }, step=self.num_timesteps)
        return True


# ── Load & Prepare Data ──────────────────────────────────────────
def load_data():
    csv_path = os.path.join(config.DATA_DIR, "rl_dataset.csv")
    if os.path.exists(csv_path):
        logger.info(f"Loading cached dataset from {csv_path}")
        df = pd.read_csv(csv_path)
        df["Date"] = pd.to_datetime(df["Date"])
    else:
        logger.info("Fetching live data …")
        df = run_pipeline(period="500d")
        os.makedirs(config.DATA_DIR, exist_ok=True)
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved dataset → {csv_path}")

    df = df.drop_duplicates(subset=["Date", "Symbol"], keep="first")
    dc = df.groupby("Date")["Symbol"].nunique()
    df = df[df["Date"].isin(dc[dc == config.N_ASSETS].index)].reset_index(drop=True)
    df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    return df


# ── Build envs ───────────────────────────────────────────────────
def make_envs(df_train, df_val, scaler):
    def _make(data):
        def _fn():
            d = data.copy()
            d[config.FEAT_COLS] = np.clip(scaler.transform(d[config.FEAT_COLS]), -5, 5)
            return Monitor(PortfolioEnv(d))
        return _fn

    train_vec = DummyVecEnv([_make(df_train)])
    train_vec = VecNormalize(train_vec, norm_obs=True, norm_reward=True, clip_obs=10.0)

    val_vec   = DummyVecEnv([_make(df_val)])
    val_vec   = VecNormalize(val_vec, norm_obs=False, norm_reward=False)

    return train_vec, val_vec


# ── Single training run ──────────────────────────────────────────
def train_model(df_train, df_val, scaler, hparams=None, run_name="ppo_train"):
    hp = {**config.PPO_CONFIG, **(hparams or {})}

    train_vec, val_vec = make_envs(df_train, df_val, scaler)

    model = PPO(
        "MlpPolicy", train_vec,
        learning_rate = hp["learning_rate"],
        n_steps       = hp["n_steps"],
        batch_size    = hp["batch_size"],
        n_epochs      = hp["n_epochs"],
        gamma         = hp["gamma"],
        gae_lambda    = hp["gae_lambda"],
        clip_range    = hp["clip_range"],
        ent_coef      = hp["ent_coef"],
        policy_kwargs = dict(
            net_arch      = dict(pi=[256, 128], vf=[256, 128]),
            activation_fn = nn.Tanh,
        ),
        verbose = 0,
    )

    eval_cb = EvalCallback(
        val_vec,
        best_model_save_path = config.MODEL_DIR,
        log_path             = config.LOG_DIR,
        eval_freq            = 2000,
        n_eval_episodes      = 1,
        deterministic        = True,
        verbose              = 0,
    )

    with mlflow.start_run(run_name=run_name, nested=True):
        mlflow.log_params(hp)
        model.learn(
            total_timesteps = hp["total_timesteps"],
            callback        = [MLflowCallback(), eval_cb],
            progress_bar    = True,
        )
        # Evaluate on val
        env = PortfolioEnv(
            df_val.assign(**{
                c: np.clip(scaler.transform(df_val[config.FEAT_COLS]), -5, 5)[:, i]
                for i, c in enumerate(config.FEAT_COLS)
            })
        )
        obs, _ = env.reset()
        done   = False
        mu, sg = train_vec.obs_rms.mean, np.sqrt(train_vec.obs_rms.var + 1e-8)
        while not done:
            obs_n       = np.clip((obs - mu) / sg, -10, 10)
            act, _      = model.predict(obs_n, deterministic=True)
            obs, _, done, _, _ = env.step(act)

        s = env.stats()
        mlflow.log_metrics(s)
        logger.info(f"[{run_name}] Sharpe={s['sharpe']:.3f} | "
                    f"Return={s['total_return_pct']:.1f}% | "
                    f"MaxDD={s['max_drawdown_pct']:.1f}%")

    return model, train_vec, s["sharpe"]


# ── Optuna objective ─────────────────────────────────────────────
def optuna_objective(trial, df_train, df_val, scaler):
    hp = {
        "learning_rate"   : trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "n_steps"         : trial.suggest_categorical("n_steps", [256, 512, 1024]),
        "batch_size"      : trial.suggest_categorical("batch", [64, 128, 256]),
        "n_epochs"        : trial.suggest_int("epochs", 3, 10),
        "gamma"           : trial.suggest_float("gamma", 0.95, 0.999),
        "gae_lambda"      : trial.suggest_float("gae", 0.9, 0.99),
        "clip_range"      : trial.suggest_float("clip", 0.1, 0.3),
        "ent_coef"        : trial.suggest_float("ent", 1e-4, 0.05, log=True),
        "total_timesteps" : 30_000,   # short for tuning
    }
    _, _, sharpe = train_model(df_train, df_val, scaler,
                               hparams=hp, run_name=f"optuna_{trial.number}")
    return sharpe


# ── Main ─────────────────────────────────────────────────────────
def main(tune=False):
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR,   exist_ok=True)

    mlflow.set_tracking_uri(config.MLFLOW_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    # Data
    df     = load_data()
    dates  = sorted(df["Date"].unique())
    split  = int(len(dates) * 0.75)
    tr_d   = dates[:split]
    val_d  = dates[split:]
    df_tr  = df[df["Date"].isin(tr_d)].reset_index(drop=True)
    df_val = df[df["Date"].isin(val_d)].reset_index(drop=True)

    # Fit scaler on train only
    scaler = RobustScaler()
    scaler.fit(df_tr[config.FEAT_COLS])
    joblib.dump(scaler, config.SCALER_PATH)
    logger.info(f"Scaler saved → {config.SCALER_PATH}")

    with mlflow.start_run(run_name="main_training"):
        mlflow.log_param("train_days",  len(tr_d))
        mlflow.log_param("val_days",    len(val_d))
        mlflow.log_param("n_assets",    config.N_ASSETS)
        mlflow.log_param("symbols",     str(config.SYMBOLS))
        mlflow.log_param("tune_mode",   tune)

        if tune:
            logger.info("Running Optuna hyperparameter search …")
            study = optuna.create_study(direction="maximize",
                                        study_name="ppo_portfolio")
            study.optimize(
                lambda t: optuna_objective(t, df_tr, df_val, scaler),
                n_trials=10, show_progress_bar=True,
            )
            best_hp = study.best_params
            logger.info(f"Best hyperparams: {best_hp}")
            mlflow.log_params({f"best_{k}": v for k, v in best_hp.items()})

            hp_full = {**config.PPO_CONFIG, **best_hp,
                       "total_timesteps": config.PPO_CONFIG["total_timesteps"]}
            model, vec, sharpe = train_model(
                df_tr, df_val, scaler, hparams=hp_full, run_name="best_tuned")
        else:
            model, vec, sharpe = train_model(
                df_tr, df_val, scaler, run_name="default")

        # Save artefacts
        model.save(config.MODEL_PATH)
        vec.save(config.VEC_PATH)
        mlflow.log_artifacts(config.MODEL_DIR, artifact_path="model")
        mlflow.log_metric("final_sharpe", sharpe)
        logger.info(f"Model saved → {config.MODEL_PATH}.zip")
        logger.info(f"VecNorm saved → {config.VEC_PATH}")

    logger.info("Training complete ✅")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter search before final training")
    args = parser.parse_args()
    main(tune=args.tune)
