# ================================================================
#  RL PORTFOLIO OPTIMIZATION — NSE 5 STOCKS
#  Uses pre-built rl_dataset.csv  |  Fast · Bug-Free · Colab Ready
# ================================================================

# ── CELL 1: Install (run once, restart runtime after) ───────────
# !pip install stable-baselines3[extra] gymnasium shimmy --quiet

# ── CELL 2: Imports ─────────────────────────────────────────────
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn as nn

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from sklearn.preprocessing import RobustScaler

print("✅ Imports OK")

# ── CELL 3: Config ──────────────────────────────────────────────
CFG = {
    "SYMBOLS"        : ['HDFC','ICICIBANK','ITC','RELIANCE','SBIN'],
    "FEAT_COLS"      : ['RSI','MACD_D','P_EMA9','P_EMA21',
                        'ATR_PCT','BB_PCT','VOL_R','RET1','RET5','VOL20'],
    "TRAIN_RATIO"    : 0.75,
    "WINDOW"         : 10,
    "INITIAL_BAL"    : 1_000_000,
    "TX_COST"        : 0.001,
    "MAX_DD"         : 0.30,
    "R_SCALE"        : 100.0,
    # PPO
    "TIMESTEPS"      : 80_000,
    "LR"             : 3e-4,
    "N_STEPS"        : 512,
    "BATCH"          : 128,
    "EPOCHS"         : 5,
    "GAMMA"          : 0.99,
}
FEAT_COLS = CFG["FEAT_COLS"]
N_FEAT    = len(FEAT_COLS)
N_SYM     = len(CFG["SYMBOLS"])

# ── CELL 4: Load Pre-built Dataset ──────────────────────────────
# Upload rl_dataset.csv to Colab, then run:
# from google.colab import files
# files.upload()   # select rl_dataset.csv

df = pd.read_csv("rl_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Date','Symbol']).reset_index(drop=True)

# Safety dedup
df = df.drop_duplicates(subset=['Date','Symbol'], keep='first')

# Keep only dates where all 5 symbols are present
dc = df.groupby('Date')['Symbol'].nunique()
df = df[df['Date'].isin(dc[dc == N_SYM].index)].reset_index(drop=True)

dates = sorted(df['Date'].unique())
print(f"✅ Dataset: {len(dates):,} trading days | "
      f"{N_SYM} stocks | "
      f"{dates[0].date()} → {dates[-1].date()}")
print(f"   Symbols : {CFG['SYMBOLS']}")
print(f"   Features: {FEAT_COLS}")

# ── CELL 5: Train / Test Split ──────────────────────────────────
split    = int(len(dates) * CFG["TRAIN_RATIO"])
tr_dates = dates[:split]
te_dates = dates[split:]

df_train = df[df['Date'].isin(tr_dates)].reset_index(drop=True)
df_test  = df[df['Date'].isin(te_dates)].reset_index(drop=True)

print(f"\n✅ Train: {len(tr_dates):,} days  "
      f"({tr_dates[0].date()} → {tr_dates[-1].date()})")
print(f"   Test : {len(te_dates):,} days  "
      f"({te_dates[0].date()} → {te_dates[-1].date()})")

# ── CELL 6: Normalise (train-fit only, no leakage) ──────────────
scaler = RobustScaler()
scaler.fit(df_train[FEAT_COLS])

df_train = df_train.copy()
df_test  = df_test.copy()
df_train[FEAT_COLS] = np.clip(scaler.transform(df_train[FEAT_COLS]), -5, 5)
df_test[FEAT_COLS]  = np.clip(scaler.transform(df_test[FEAT_COLS]),  -5, 5)
print(f"\n✅ Normalised {N_FEAT} features (RobustScaler, train-fit only)")

# ── CELL 7: RL Environment ──────────────────────────────────────
class PortfolioEnv(gym.Env):
    """
    Observation : window x n_stocks x n_features (flattened) + current weights
    Action      : softmax-normalised weights [long-only]
    Reward      : scaled net return - drawdown penalty - HHI concentration penalty
    """
    metadata = {"render_modes": []}

    def __init__(self, df, window=10):
        super().__init__()
        self.window  = window
        self.symbols = CFG["SYMBOLS"]
        self.n       = N_SYM

        self.dates = sorted(df['Date'].unique())
        self.T     = len(self.dates)

        sym_idx = {s: i for i, s in enumerate(self.symbols)}
        self._F = np.zeros((self.T, self.n, N_FEAT), dtype=np.float32)
        self._C = np.zeros((self.T, self.n),          dtype=np.float32)

        for t, date in enumerate(self.dates):
            day = df[df['Date'] == date]
            for _, row in day.iterrows():
                i = sym_idx.get(row['Symbol'])
                if i is not None:
                    self._F[t, i] = [row[f] for f in FEAT_COLS]
                    self._C[t, i] = float(row['Close'])

        obs_dim = window * self.n * N_FEAT + self.n
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space      = spaces.Box(0.0, 1.0, (self.n,),  np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t        = self.window
        self.bal      = float(CFG["INITIAL_BAL"])
        self.peak     = self.bal
        self.w        = np.full(self.n, 1.0 / self.n, np.float32)
        self.bal_hist = [self.bal]
        self.w_hist   = []
        return self._obs(), {}

    def _obs(self):
        feat = self._F[self.t - self.window : self.t]
        return np.concatenate([feat.flatten(), self.w]).astype(np.float32)

    def step(self, action):
        action = np.clip(action, 1e-8, 1.0)
        w      = action / (action.sum() + 1e-8)

        c0  = self._C[self.t]
        c1  = self._C[self.t + 1]
        ret = np.where(c0 > 0, (c1 - c0) / c0, 0.0)
        ret = np.clip(ret, -0.15, 0.15)

        port_r = float(np.dot(w, ret))
        cost   = float(np.sum(np.abs(w - self.w))) * CFG["TX_COST"]
        net_r  = port_r - cost

        self.bal  *= (1.0 + net_r)
        self.peak  = max(self.peak, self.bal)
        dd         = (self.peak - self.bal) / (self.peak + 1e-9)

        reward  = net_r * CFG["R_SCALE"]
        reward -= dd * 5.0
        reward -= (float(np.sum(w ** 2)) - 1.0 / self.n) * 2.0

        self.w = w
        self.bal_hist.append(self.bal)
        self.w_hist.append(w.copy())
        self.t += 1

        done = (self.t >= self.T - 2) or (dd > CFG["MAX_DD"])
        if done and dd > CFG["MAX_DD"]:
            reward -= 50.0

        return self._obs(), float(reward), done, False, {}

    def stats(self):
        b   = np.array(self.bal_hist)
        r   = np.diff(b) / (b[:-1] + 1e-9)
        n   = max(len(r), 1)
        tr  = b[-1] / b[0] - 1
        ar  = (1 + tr) ** (252 / n) - 1
        vol = r.std() * np.sqrt(252)
        sh  = ar / (vol + 1e-9)
        rm  = np.maximum.accumulate(b)
        mdd = ((rm - b) / (rm + 1e-9)).max()
        return {
            "Final Balance Rs" : f"{b[-1]:,.0f}",
            "Total Return %"   : f"{tr*100:.2f}",
            "Ann Return %"     : f"{ar*100:.2f}",
            "Ann Vol %"        : f"{vol*100:.2f}",
            "Sharpe Ratio"     : f"{sh:.3f}",
            "Max Drawdown %"   : f"{mdd*100:.2f}",
            "Calmar Ratio"     : f"{ar/(mdd+1e-9):.3f}",
            "Win Rate %"       : f"{(r>0).mean()*100:.2f}",
        }

# ── CELL 8: Train ────────────────────────────────────────────────
class LogCallback(BaseCallback):
    def __init__(self, freq=20_000):
        super().__init__(); self.freq = freq
    def _on_step(self):
        if self.n_calls % self.freq == 0:
            buf = self.model.ep_info_buffer
            mr  = np.mean([e['r'] for e in buf]) if buf else 0.0
            print(f"  step {self.n_calls:>7,} | mean_ep_reward {mr:+.2f}")
        return True

print("\n" + "="*55)
print("  TRAINING PPO AGENT")
print("="*55)

def make_env():
    return Monitor(PortfolioEnv(df_train, window=CFG["WINDOW"]))

vec = DummyVecEnv([make_env])
vec = VecNormalize(vec, norm_obs=True, norm_reward=True, clip_obs=10.0)

model = PPO(
    "MlpPolicy", vec,
    learning_rate = CFG["LR"],
    n_steps       = CFG["N_STEPS"],
    batch_size    = CFG["BATCH"],
    n_epochs      = CFG["EPOCHS"],
    gamma         = CFG["GAMMA"],
    ent_coef      = 0.01,
    policy_kwargs = dict(
        net_arch      = dict(pi=[256, 128], vf=[256, 128]),
        activation_fn = nn.Tanh,
    ),
    verbose = 0,
)

model.learn(CFG["TIMESTEPS"], callback=LogCallback(), progress_bar=True)
print("✅ Training complete!")

# ── CELL 9: Backtest Helper ──────────────────────────────────────
def run_backtest(df, label=""):
    env    = PortfolioEnv(df, window=CFG["WINDOW"])
    obs, _ = env.reset()
    done   = False
    mu     = vec.obs_rms.mean
    sigma  = np.sqrt(vec.obs_rms.var + 1e-8)
    while not done:
        obs_n      = np.clip((obs - mu) / sigma, -10, 10)
        act, _     = model.predict(obs_n, deterministic=True)
        obs, _, done, _, _ = env.step(act)
    if label:
        print(f"\n📊 {label}")
        print("-" * 40)
        for k, v in env.stats().items():
            print(f"  {k:<18}: {v:>14}")
        print("-" * 40)
    return env

# ── CELL 10: Benchmarks ──────────────────────────────────────────
def bench_equal_weight(df):
    df_d = df.drop_duplicates(subset=['Date','Symbol'], keep='last')
    pv   = df_d.pivot(index='Date', columns='Symbol', values='Close').sort_index()
    r    = pv.pct_change().fillna(0)
    bal  = [CFG["INITIAL_BAL"]]
    for i in range(len(r) - 1):
        bal.append(bal[-1] * (1.0 + r.iloc[i + 1].mean()))
    return np.array(bal)

def bench_buy_hold(df):
    df_d = df.drop_duplicates(subset=['Date','Symbol'], keep='last')
    pv   = df_d.pivot(index='Date', columns='Symbol', values='Close').sort_index()
    norm = pv / pv.iloc[0]
    return (CFG["INITIAL_BAL"] * norm.mean(axis=1)).values

def bench_stats(bal, name):
    r   = np.diff(bal) / (bal[:-1] + 1e-9)
    tr  = bal[-1] / bal[0] - 1
    ar  = (1 + tr) ** (252 / max(len(r), 1)) - 1
    sh  = ar / (r.std() * np.sqrt(252) + 1e-9)
    rm  = np.maximum.accumulate(bal)
    mdd = ((rm - bal) / (rm + 1e-9)).max()
    print(f"  [{name:<22}]  Return {tr*100:+6.1f}%  "
          f"Sharpe {sh:+5.2f}  MaxDD {mdd*100:.1f}%")

# ── CELL 11: Run Backtest & Compare ─────────────────────────────
print("\n" + "="*55)
print("  BACKTEST — TEST SET")
print("="*55)

test_env = run_backtest(df_test, "TEST — RL Agent (PPO)")
eq_bal   = bench_equal_weight(df_test)
bh_bal   = bench_buy_hold(df_test)
rl_bal   = np.array(test_env.bal_hist)

print("\n📊 BENCHMARK COMPARISON (Test Period):")
bench_stats(rl_bal, "RL Agent (PPO)")
bench_stats(eq_bal, "Equal Weight")
bench_stats(bh_bal, "Buy & Hold")

# ── CELL 12: Charts ──────────────────────────────────────────────
fig = plt.figure(figsize=(18, 12))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.33)
td  = range(len(rl_bal))

# 1. Equity curves
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(td, rl_bal / CFG["INITIAL_BAL"],
         label="RL Agent (PPO)", color="#2196F3", lw=2.2)
ax1.plot(range(len(eq_bal)), eq_bal / CFG["INITIAL_BAL"],
         label="Equal Weight",   color="#FF9800", ls="--", lw=1.6)
ax1.plot(range(len(bh_bal)), bh_bal / CFG["INITIAL_BAL"],
         label="Buy & Hold",     color="#4CAF50", ls=":",  lw=1.6)
ax1.axhline(1.0, color="gray", lw=0.8)
ax1.set_title("Portfolio Equity Curves — Test Period", fontsize=14, fontweight="bold")
ax1.set_ylabel("Normalised Value  (1.0 = Rs 10L)")
ax1.set_xlabel("Trading Days")
ax1.legend(fontsize=11); ax1.grid(alpha=0.3)

# 2. Drawdown
ax2 = fig.add_subplot(gs[1, 0])
rm  = np.maximum.accumulate(rl_bal)
dd  = (rm - rl_bal) / (rm + 1e-9) * 100
ax2.fill_between(td, -dd, 0, color="#F44336", alpha=0.65)
ax2.set_title("Drawdown (%)", fontsize=12, fontweight="bold")
ax2.set_xlabel("Trading Days"); ax2.set_ylabel("DD (%)")
ax2.grid(alpha=0.3)

# 3. Daily return distribution
ax3 = fig.add_subplot(gs[1, 1])
dr  = np.diff(rl_bal) / (rl_bal[:-1] + 1e-9) * 100
ax3.hist(dr, bins=60, color="#9C27B0", alpha=0.75, edgecolor="white")
ax3.axvline(np.mean(dr), color="red",   ls="--", lw=1.5,
            label=f"Mean {np.mean(dr):.3f}%")
ax3.axvline(0,           color="black", lw=0.8)
ax3.set_title("Daily Return Distribution", fontsize=12, fontweight="bold")
ax3.set_xlabel("Daily Return (%)"); ax3.set_ylabel("Frequency")
ax3.legend(); ax3.grid(alpha=0.3)

# 4. Final portfolio weights
ax4 = fig.add_subplot(gs[2, 0])
if test_env.w_hist:
    fw     = test_env.w_hist[-1]
    colors = plt.cm.tab10(np.linspace(0, 1, test_env.n))
    ax4.bar(test_env.symbols, fw * 100, color=colors, edgecolor="white")
    ax4.axhline(100.0 / test_env.n, color="red", ls="--", lw=1.4,
                label=f"Equal ({100/test_env.n:.0f}%)")
    ax4.set_title("Final Portfolio Weights (%)", fontsize=12, fontweight="bold")
    ax4.set_ylabel("Weight (%)")
    ax4.set_xticks(range(test_env.n))
    ax4.set_xticklabels(test_env.symbols, rotation=30, ha='right', fontsize=10)
    ax4.legend(); ax4.grid(alpha=0.3, axis='y')

# 5. Rolling 60-day Sharpe
ax5 = fig.add_subplot(gs[2, 1])
W = 60
if len(dr) > W:
    rs = [
        np.mean(dr[max(0, i-W):i]) /
        (np.std(dr[max(0, i-W):i]) + 1e-9) * np.sqrt(252)
        for i in range(W, len(dr))
    ]
    ax5.plot(range(W, len(dr)), rs, color="#00BCD4", lw=1.5)
    ax5.axhline(0,   color="gray",  ls="--", lw=0.8)
    ax5.axhline(1.0, color="green", ls=":",  lw=1.2, label="Sharpe = 1")
    ax5.set_title(f"Rolling {W}-Day Sharpe (Ann.)", fontsize=12, fontweight="bold")
    ax5.set_xlabel("Trading Days"); ax5.set_ylabel("Sharpe")
    ax5.legend(); ax5.grid(alpha=0.3)

plt.suptitle("RL Portfolio Optimisation — NSE 5 Stocks | PPO Agent",
             fontsize=15, fontweight="bold", y=1.01)
plt.savefig("rl_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Chart saved → rl_results.png")

# ── CELL 13: Save Model ──────────────────────────────────────────
model.save("ppo_portfolio")
vec.save("vec_norm.pkl")
print("✅ Model  → ppo_portfolio.zip")
print("✅ Scaler → vec_norm.pkl")
print("\n🎉 All done!")
