"""
data_pipeline.py — Real-time & historical data fetching + feature engineering
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
import logging

import config

logger = logging.getLogger(__name__)


# ── Fetch live OHLCV from yfinance ───────────────────────────────
def fetch_live(symbols=None, period="60d", interval="1d"):
    """
    Download recent OHLCV for all symbols.
    Returns a clean, deduplicated DataFrame sorted by [Date, Symbol].
    """
    symbols = symbols or config.SYMBOLS
    parts = []

    for sym in symbols:
        ticker = sym + ".NS"
        try:
            raw = yf.download(ticker, period=period, interval=interval,
                              progress=False, auto_adjust=True)
            if raw.empty:
                logger.warning(f"No data returned for {ticker}")
                continue

            raw = raw.reset_index()
            # yfinance column names vary — normalise
            raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
            raw = raw.rename(columns={"Datetime": "Date", "index": "Date"})
            raw["Symbol"] = sym

            keep = ["Date", "Symbol", "Open", "High", "Low", "Close", "Volume"]
            raw = raw[[c for c in keep if c in raw.columns]]
            parts.append(raw)

        except Exception as e:
            logger.error(f"Fetch error {ticker}: {e}")

    if not parts:
        raise RuntimeError("No data fetched for any symbol.")

    df = pd.concat(parts, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Volume"] = df["Volume"].fillna(0)
    df = df.dropna(subset=["Close", "High", "Low", "Open"])
    df = df[df["Close"] > 0]
    df = df.sort_values(["Symbol", "Date", "Volume"], ascending=[True, True, False])
    df = df.drop_duplicates(subset=["Date", "Symbol"], keep="first")
    df = df.sort_values(["Date", "Symbol"]).reset_index(drop=True)

    logger.info(f"Fetched {len(df)} rows | {df['Symbol'].nunique()} symbols | "
                f"{df['Date'].min().date()} → {df['Date'].max().date()}")
    return df


# ── Feature Engineering ──────────────────────────────────────────
def add_features(df):
    """
    Compute all 10 technical features per symbol.
    Returns DataFrame with FEAT_COLS added; NaNs dropped.
    """
    parts = []
    for sym in sorted(df["Symbol"].unique()):
        t = df[df["Symbol"] == sym].copy().reset_index(drop=True)
        if len(t) < 30:
            logger.warning(f"Too few rows for {sym} ({len(t)}), skipping")
            continue

        c, h, l, v = t["Close"], t["High"], t["Low"], t["Volume"]

        t["RSI"]    = RSIIndicator(c, window=14).rsi()
        t["MACD_D"] = MACD(c).macd_diff()
        e9          = EMAIndicator(c, window=9).ema_indicator()
        e21         = EMAIndicator(c, window=21).ema_indicator()
        t["P_EMA9"] = (c / (e9  + 1e-9) - 1)
        t["P_EMA21"]= (c / (e21 + 1e-9) - 1)
        t["ATR_PCT"]= AverageTrueRange(h, l, c, window=14).average_true_range() / (c + 1e-9)
        t["BB_PCT"] = BollingerBands(c, window=20).bollinger_pband()
        vsma        = v.rolling(20).mean()
        t["VOL_R"]  = v / (vsma + 1e-9)
        t["RET1"]   = c.pct_change(1)
        t["RET5"]   = c.pct_change(5)
        t["VOL20"]  = t["RET1"].rolling(20).std()

        t = t.dropna()
        if len(t) >= 15:
            parts.append(t)

    if not parts:
        raise RuntimeError("Feature engineering produced no valid data.")

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["Date", "Symbol"]).reset_index(drop=True)
    return out


# ── Align so all symbols present on every date ───────────────────
def align(df, n_sym=None):
    n_sym = n_sym or df["Symbol"].nunique()
    dc    = df.groupby("Date")["Symbol"].nunique()
    valid = dc[dc == n_sym].index
    df    = df[df["Date"].isin(valid)].reset_index(drop=True)
    logger.info(f"Aligned: {df['Date'].nunique()} dates × {n_sym} symbols")
    return df


# ── Build latest observation vector for inference ────────────────
def build_obs(df, scaler, window=None):
    """
    Returns numpy float32 array ready for model.predict().
    Shape: (window * n_assets * n_features + n_assets,)
    """
    window   = window or config.WINDOW_SIZE
    n_assets = config.N_ASSETS
    n_feat   = config.N_FEATURES
    feat_cols= config.FEAT_COLS

    dates  = sorted(df["Date"].unique())
    if len(dates) < window:
        raise ValueError(f"Need ≥{window} dates, got {len(dates)}")

    last_w = dates[-window:]
    sym_idx = {s: i for i, s in enumerate(config.SYMBOLS)}

    F = np.zeros((window, n_assets, n_feat), dtype=np.float32)
    for ti, date in enumerate(last_w):
        day = df[df["Date"] == date]
        for _, row in day.iterrows():
            i = sym_idx.get(row["Symbol"])
            if i is not None:
                raw = np.array([row[f] for f in feat_cols], dtype=np.float32)
                F[ti, i] = raw

    # Scale using fitted scaler
    flat = F.reshape(-1, n_feat)
    flat = np.clip(scaler.transform(flat), -5, 5).astype(np.float32)
    F    = flat.reshape(window, n_assets, n_feat)

    equal_w = np.full(n_assets, 1.0 / n_assets, dtype=np.float32)
    obs     = np.concatenate([F.flatten(), equal_w])
    return obs


# ── Latest close prices for display ─────────────────────────────
def latest_prices(df):
    last_date = df["Date"].max()
    snap      = df[df["Date"] == last_date].set_index("Symbol")["Close"]
    return snap.reindex(config.SYMBOLS).to_dict()


# ── Full pipeline: fetch → features → align ──────────────────────
def run_pipeline(period="60d"):
    raw  = fetch_live(period=period)
    feat = add_features(raw)
    aln  = align(feat, n_sym=len(config.SYMBOLS))
    return aln
