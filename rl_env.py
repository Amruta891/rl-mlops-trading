"""
rl_env.py — Portfolio RL Gymnasium Environment
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import config


class PortfolioEnv(gym.Env):
    """
    Multi-asset continuous portfolio allocation environment.

    Observation : flattened (window × n_assets × n_features) + current weights
    Action      : softmax-normalised weights  [long-only, sums to 1]
    Reward      : scaled net return − drawdown penalty − HHI concentration penalty
    """
    metadata = {"render_modes": []}

    def __init__(self, df, window=None):
        super().__init__()
        self.window   = window or config.WINDOW_SIZE
        self.symbols  = config.SYMBOLS
        self.n        = config.N_ASSETS
        self.feat_cols= config.FEAT_COLS
        self.n_feat   = config.N_FEATURES

        self.dates = sorted(df["Date"].unique())
        self.T     = len(self.dates)

        # Pre-build numpy caches for speed
        sym_idx  = {s: i for i, s in enumerate(self.symbols)}
        self._F  = np.zeros((self.T, self.n, self.n_feat), dtype=np.float32)
        self._C  = np.zeros((self.T, self.n),               dtype=np.float32)

        for t, date in enumerate(self.dates):
            day = df[df["Date"] == date]
            for _, row in day.iterrows():
                i = sym_idx.get(row["Symbol"])
                if i is not None:
                    self._F[t, i] = [row[f] for f in self.feat_cols]
                    self._C[t, i] = float(row["Close"])

        obs_dim = self.window * self.n * self.n_feat + self.n
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
        self.action_space      = spaces.Box(0.0, 1.0, (self.n,), np.float32)
        self.reset()

    # ──────────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t        = self.window
        self.bal      = float(config.INITIAL_CAPITAL)
        self.peak     = self.bal
        self.w        = np.full(self.n, 1.0 / self.n, np.float32)
        self.bal_hist = [self.bal]
        self.w_hist   = []
        self.ret_hist = []
        return self._obs(), {}

    def _obs(self):
        feat = self._F[self.t - self.window : self.t]
        return np.concatenate([feat.flatten(), self.w]).astype(np.float32)

    # ──────────────────────────────────────────────────────────────
    def step(self, action):
        action = np.clip(action, 1e-8, 1.0)
        w      = action / (action.sum() + 1e-8)

        c0  = self._C[self.t]
        c1  = self._C[self.t + 1]
        ret = np.where(c0 > 0, (c1 - c0) / c0, 0.0)
        ret = np.clip(ret, -0.15, 0.15)

        port_r = float(np.dot(w, ret))
        cost   = float(np.sum(np.abs(w - self.w))) * config.TRANSACTION_COST
        net_r  = port_r - cost

        self.bal  *= (1.0 + net_r)
        self.peak  = max(self.peak, self.bal)
        dd         = (self.peak - self.bal) / (self.peak + 1e-9)

        reward  = net_r * config.REWARD_SCALE
        reward -= dd * 5.0
        reward -= (float(np.sum(w ** 2)) - 1.0 / self.n) * 2.0   # anti-HHI

        self.w = w
        self.bal_hist.append(self.bal)
        self.w_hist.append(w.copy())
        self.ret_hist.append(net_r)
        self.t += 1

        done = (self.t >= self.T - 2) or (dd > config.MAX_DRAWDOWN_LIMIT)
        if done and dd > config.MAX_DRAWDOWN_LIMIT:
            reward -= 50.0

        return self._obs(), float(reward), done, False, {}

    # ──────────────────────────────────────────────────────────────
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
            "final_balance"   : round(float(b[-1]), 2),
            "total_return_pct": round(tr * 100, 3),
            "ann_return_pct"  : round(ar * 100, 3),
            "ann_vol_pct"     : round(vol * 100, 3),
            "sharpe"          : round(float(sh), 4),
            "max_drawdown_pct": round(mdd * 100, 3),
            "calmar"          : round(float(ar / (mdd + 1e-9)), 4),
            "win_rate_pct"    : round(float((r > 0).mean() * 100), 2),
        }
