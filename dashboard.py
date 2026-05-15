"""
dashboard.py — Real-time trading dashboard
Run: python dashboard.py

Features:
  • Live stock prices (yfinance, auto-refreshes every 30s)
  • RL model portfolio weights (from latest inference)
  • Equity curve & drawdown
  • Per-stock OHLCV candlestick chart
  • Data drift scores
  • Trade log
  • MLOps metrics panel
"""
import warnings; warnings.filterwarnings("ignore")
import os, json, threading, time, logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table, Input, Output, State
import dash_bootstrap_components as dbc

import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Shared state (thread-safe via lock) ─────────────────────────
import threading
_lock   = threading.Lock()
_state  = {
    "prices"       : {s: 0.0  for s in config.SYMBOLS},
    "weights"      : {s: 1/config.N_ASSETS for s in config.SYMBOLS},
    "ohlcv"        : {s: pd.DataFrame() for s in config.SYMBOLS},
    "balance_hist" : [],
    "drift_scores" : {},
    "trades"       : [],
    "alerts"       : [],
    "last_update"  : "—",
    "model_loaded" : False,
    "cycle"        : 0,
}

# ── Try loading model ─────────────────────────────────────────────
_model  = None
_scaler = None
_vec    = None

def _try_load_model():
    global _model, _scaler, _vec
    try:
        import joblib
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
        from rl_env import PortfolioEnv

        if os.path.exists(config.MODEL_PATH + ".zip"):
            _model  = PPO.load(config.MODEL_PATH)
            _scaler = joblib.load(config.SCALER_PATH)
            logger.info("Model loaded ✅")
            with _lock:
                _state["model_loaded"] = True
        else:
            logger.warning("No trained model found — run train.py first")
    except Exception as e:
        logger.warning(f"Model load skipped: {e}")

_try_load_model()


# ── Background data fetch thread ─────────────────────────────────
def _fetch_ohlcv(sym, period="5d", interval="5m"):
    try:
        raw = yf.download(sym + ".NS", period=period,
                          interval=interval, progress=False, auto_adjust=True)
        if raw.empty:
            return pd.DataFrame()
        raw = raw.reset_index()
        raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
        raw = raw.rename(columns={"Datetime": "Date"})
        return raw
    except Exception as e:
        logger.warning(f"OHLCV fetch {sym}: {e}")
        return pd.DataFrame()


def _fetch_loop():
    """Runs in background — updates _state every 30 seconds."""
    while True:
        try:
            new_prices = {}
            new_ohlcv  = {}
            for sym in config.SYMBOLS:
                df = _fetch_ohlcv(sym)
                if not df.empty and "Close" in df.columns:
                    new_prices[sym] = float(df["Close"].iloc[-1])
                    new_ohlcv[sym]  = df
                else:
                    new_prices[sym] = _state["prices"].get(sym, 0.0)
                    new_ohlcv[sym]  = _state["ohlcv"].get(sym, pd.DataFrame())

            # RL inference
            weights = {s: 1/config.N_ASSETS for s in config.SYMBOLS}
            if _model is not None and _scaler is not None:
                try:
                    from data_pipeline import run_pipeline, build_obs
                    df_feat = run_pipeline(period="60d")
                    obs = build_obs(df_feat, _scaler)
                    action, _ = _model.predict(obs, deterministic=True)
                    action = np.clip(action, 1e-8, 1.0)
                    action = action / action.sum()
                    weights = {s: float(action[i]) for i, s in enumerate(config.SYMBOLS)}
                except Exception as e:
                    logger.warning(f"Inference skipped: {e}")

            # Simulated balance (paper mode)
            prev_bal = _state["balance_hist"][-1]["balance"] \
                       if _state["balance_hist"] else config.INITIAL_CAPITAL
            sim_ret   = np.random.normal(0.0002, 0.005)   # placeholder
            new_bal   = prev_bal * (1 + sim_ret)

            with _lock:
                _state["prices"]       = new_prices
                _state["ohlcv"]        = new_ohlcv
                _state["weights"]      = weights
                _state["last_update"]  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                _state["cycle"]       += 1
                _state["balance_hist"].append({
                    "time"   : _state["last_update"],
                    "balance": round(new_bal, 2),
                })

        except Exception as e:
            logger.error(f"Fetch loop error: {e}")

        time.sleep(30)


_bg = threading.Thread(target=_fetch_loop, daemon=True)
_bg.start()


# ── Dash App ─────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="RL Portfolio | MLOps Dashboard",
    suppress_callback_exceptions=True,
)

# Colour palette
C = {
    "bg"    : "#0d1117",
    "card"  : "#161b22",
    "border": "#30363d",
    "green" : "#39d353",
    "red"   : "#f85149",
    "blue"  : "#58a6ff",
    "purple": "#a855f7",
    "yellow": "#e3b341",
    "text"  : "#e6edf3",
    "muted" : "#8b949e",
}

CARD_STYLE = {
    "background"  : C["card"],
    "border"      : f"1px solid {C['border']}",
    "borderRadius": "8px",
    "padding"     : "16px",
    "marginBottom": "12px",
}


def _badge(label, value, color=C["blue"], suffix=""):
    return html.Div([
        html.P(label, style={"color": C["muted"], "fontSize": "11px",
                              "marginBottom": "2px", "textTransform": "uppercase"}),
        html.H4(f"{value}{suffix}",
                style={"color": color, "margin": 0, "fontWeight": "700"}),
    ], style={"textAlign": "center"})


# ── Layout ───────────────────────────────────────────────────────
app.layout = dbc.Container(fluid=True, style={"background": C["bg"],
                                               "minHeight": "100vh",
                                               "padding": "20px"}, children=[

    # Header
    dbc.Row([
        dbc.Col(html.Div([
            html.H2("🤖 RL Portfolio — Real-Time MLOps Dashboard",
                    style={"color": C["text"], "margin": 0}),
            html.Small(id="last-update",
                       style={"color": C["muted"]}),
        ]), width=9),
        dbc.Col(html.Div([
            html.Span(id="mode-badge"),
            html.Span(id="model-badge", style={"marginLeft": "8px"}),
        ], style={"textAlign": "right", "paddingTop": "8px"}), width=3),
    ], style={"marginBottom": "20px"}),

    # KPI Row
    dbc.Row(id="kpi-row", style={"marginBottom": "16px"}),

    # Price Tickers
    dbc.Row([dbc.Col(
        html.Div(id="price-tickers", style={**CARD_STYLE, "display": "flex",
                                             "gap": "24px", "flexWrap": "wrap"}),
        width=12
    )], style={"marginBottom": "8px"}),

    # Charts row 1
    dbc.Row([
        dbc.Col(dcc.Graph(id="equity-chart",   config={"displayModeBar": False}), width=8),
        dbc.Col(dcc.Graph(id="weights-chart",  config={"displayModeBar": False}), width=4),
    ], style={"marginBottom": "8px"}),

    # Charts row 2
    dbc.Row([
        dbc.Col([
            dcc.Dropdown(
                id="sym-select",
                options=[{"label": s, "value": s} for s in config.SYMBOLS],
                value=config.SYMBOLS[0],
                clearable=False,
                style={"background": C["card"], "color": "#000",
                       "marginBottom": "4px", "width": "180px"},
            ),
            dcc.Graph(id="candle-chart", config={"displayModeBar": False}),
        ], width=8),
        dbc.Col(dcc.Graph(id="drift-chart",    config={"displayModeBar": False}), width=4),
    ], style={"marginBottom": "8px"}),

    # Drawdown
    dbc.Row([
        dbc.Col(dcc.Graph(id="dd-chart",       config={"displayModeBar": False}), width=12),
    ], style={"marginBottom": "8px"}),

    # Trade log
    dbc.Row([dbc.Col(
        html.Div([
            html.H6("📋 Trade Log", style={"color": C["text"]}),
            html.Div(id="trade-table"),
        ], style=CARD_STYLE),
        width=12
    )]),

    # Auto-refresh
    dcc.Interval(id="interval", interval=30_000, n_intervals=0),
])


# ── Callbacks ─────────────────────────────────────────────────────
@app.callback(
    Output("last-update", "children"),
    Output("mode-badge",  "children"),
    Output("model-badge", "children"),
    Output("kpi-row",     "children"),
    Output("price-tickers","children"),
    Output("equity-chart", "figure"),
    Output("weights-chart","figure"),
    Output("drift-chart",  "figure"),
    Output("dd-chart",     "figure"),
    Output("trade-table",  "children"),
    Input("interval", "n_intervals"),
)
def refresh(_):
    with _lock:
        prices   = dict(_state["prices"])
        weights  = dict(_state["weights"])
        bal_hist = list(_state["balance_hist"])
        drift    = dict(_state["drift_scores"])
        updated  = _state["last_update"]
        mode     = config.TRADE_MODE
        loaded   = _state["model_loaded"]

    # ── Badges
    mode_badge = dbc.Badge(
        f"{'🟢 PAPER' if mode=='paper' else '🔴 LIVE'}",
        color="success" if mode == "paper" else "danger", pill=True)
    model_badge= dbc.Badge(
        "✅ Model" if loaded else "⚠️ No Model",
        color="primary" if loaded else "warning", pill=True)

    # ── KPI
    balance = bal_hist[-1]["balance"] if bal_hist else config.INITIAL_CAPITAL
    ret_pct = (balance / config.INITIAL_CAPITAL - 1) * 100
    if len(bal_hist) >= 2:
        bals = [h["balance"] for h in bal_hist]
        rets = np.diff(bals) / (np.array(bals[:-1]) + 1e-9)
        sh   = float(np.mean(rets) / (np.std(rets) + 1e-9)) * np.sqrt(252) \
               if len(rets) >= 2 else 0.0
        pk   = max(bals)
        dd   = (pk - balance) / (pk + 1e-9) * 100
    else:
        sh, dd = 0.0, 0.0

    kpi_row = [
        dbc.Col(_badge("Portfolio Value",
                        f"₹{balance:,.0f}", C["green"]), width=3,
                style=CARD_STYLE),
        dbc.Col(_badge("Cum. Return",
                        f"{ret_pct:+.2f}", C["green"] if ret_pct>=0 else C["red"], "%"),
                width=2, style=CARD_STYLE),
        dbc.Col(_badge("Live Sharpe",
                        f"{sh:.2f}", C["blue"]), width=2, style=CARD_STYLE),
        dbc.Col(_badge("Drawdown",
                        f"{dd:.2f}", C["yellow"], "%"), width=2, style=CARD_STYLE),
        dbc.Col(_badge("Cycles", str(_state["cycle"]), C["muted"]),
                width=1, style=CARD_STYLE),
        dbc.Col(_badge("Last Update", updated, C["muted"]),
                width=2, style=CARD_STYLE),
    ]

    # ── Price Tickers
    tickers = []
    for sym in config.SYMBOLS:
        p = prices.get(sym, 0)
        tickers.append(html.Div([
            html.Span(sym, style={"color": C["text"], "fontWeight": "700",
                                   "fontSize": "14px"}),
            html.Br(),
            html.Span(f"₹{p:,.2f}", style={"color": C["green"],
                                             "fontSize": "18px", "fontWeight": "600"}),
        ], style={"textAlign": "center", "minWidth": "90px"}))

    # ── Equity curve
    eq_fig = go.Figure()
    if bal_hist:
        times = [h["time"]    for h in bal_hist]
        bals  = [h["balance"] for h in bal_hist]
        eq_fig.add_trace(go.Scatter(
            x=times, y=bals, mode="lines",
            line=dict(color=C["green"], width=2),
            fill="tozeroy", fillcolor="rgba(57,211,83,0.08)",
            name="Portfolio Value",
        ))
        eq_fig.add_hline(y=config.INITIAL_CAPITAL,
                         line_dash="dash", line_color=C["muted"],
                         annotation_text="Start")
    eq_fig.update_layout(
        title="📈 Portfolio Equity Curve",
        paper_bgcolor=C["card"], plot_bgcolor=C["card"],
        font_color=C["text"], margin=dict(l=40, r=20, t=40, b=30),
        yaxis_title="Value (₹)", xaxis_title="",
        showlegend=False,
    )

    # ── Weights donut
    syms_list = list(weights.keys())
    wts_list  = [weights[s] for s in syms_list]
    w_fig = go.Figure(go.Pie(
        labels=syms_list, values=wts_list, hole=0.5,
        marker_colors=px.colors.qualitative.Plotly,
        textinfo="label+percent",
        hovertemplate="%{label}: %{value:.2%}<extra></extra>",
    ))
    w_fig.update_layout(
        title="🏦 Portfolio Weights",
        paper_bgcolor=C["card"], font_color=C["text"],
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=False,
    )

    # ── Drift
    dr_fig = go.Figure()
    if drift:
        feat_names  = list(drift.keys())
        drift_vals  = [drift[f] for f in feat_names]
        bar_colors  = [C["red"] if v > config.DRIFT_THRESHOLD else C["green"]
                       for v in drift_vals]
        dr_fig.add_trace(go.Bar(
            x=drift_vals, y=feat_names, orientation="h",
            marker_color=bar_colors,
            hovertemplate="%{y}: %{x:.4f}<extra></extra>",
        ))
        dr_fig.add_vline(x=config.DRIFT_THRESHOLD,
                         line_dash="dash", line_color=C["yellow"],
                         annotation_text="Threshold")
    dr_fig.update_layout(
        title="🔍 Data Drift (JS Divergence)",
        paper_bgcolor=C["card"], plot_bgcolor=C["card"],
        font_color=C["text"], margin=dict(l=80, r=20, t=40, b=30),
        xaxis_title="JS Divergence",
    )

    # ── Drawdown
    dd_fig = go.Figure()
    if len(bal_hist) >= 2:
        bals_arr = np.array([h["balance"] for h in bal_hist])
        peak     = np.maximum.accumulate(bals_arr)
        dd_arr   = (peak - bals_arr) / (peak + 1e-9) * 100
        times    = [h["time"] for h in bal_hist]
        dd_fig.add_trace(go.Scatter(
            x=times, y=-dd_arr, mode="lines",
            fill="tozeroy", fillcolor="rgba(248,81,73,0.15)",
            line=dict(color=C["red"], width=1.5), name="Drawdown",
        ))
    dd_fig.update_layout(
        title="📉 Drawdown (%)",
        paper_bgcolor=C["card"], plot_bgcolor=C["card"],
        font_color=C["text"], margin=dict(l=40, r=20, t=40, b=30),
        yaxis_title="DD (%)", xaxis_title="",
    )

    # ── Trade table
    trade_path = "logs/trades.csv"
    if os.path.exists(trade_path):
        try:
            tdf = pd.read_csv(trade_path).tail(20)
            table = dash_table.DataTable(
                data=tdf.to_dict("records"),
                columns=[{"name": c, "id": c} for c in tdf.columns],
                style_table={"overflowX": "auto"},
                style_cell={"background": C["card"], "color": C["text"],
                             "border": f"1px solid {C['border']}",
                             "fontSize": "12px", "padding": "6px"},
                style_header={"background": C["bg"], "fontWeight": "700",
                               "color": C["muted"]},
                page_size=10,
            )
        except Exception:
            table = html.P("No trades yet.", style={"color": C["muted"]})
    else:
        table = html.P("No trades logged yet.", style={"color": C["muted"]})

    return (
        f"Last update: {updated}",
        mode_badge, model_badge,
        kpi_row, tickers,
        eq_fig, w_fig,
        dr_fig, dd_fig,
        table,
    )


@app.callback(
    Output("candle-chart", "figure"),
    Input("sym-select", "value"),
    Input("interval", "n_intervals"),
)
def update_candle(sym, _):
    with _lock:
        df = _state["ohlcv"].get(sym, pd.DataFrame()).copy()

    fig = go.Figure()
    if df.empty or "Close" not in df.columns:
        fig.update_layout(
            title=f"{sym} — No data",
            paper_bgcolor=C["card"], plot_bgcolor=C["card"],
            font_color=C["text"],
        )
        return fig

    date_col = "Date" if "Date" in df.columns else df.columns[0]
    fig.add_trace(go.Candlestick(
        x=df[date_col],
        open=df["Open"], high=df["High"],
        low=df["Low"],   close=df["Close"],
        increasing_line_color=C["green"],
        decreasing_line_color=C["red"],
        name=sym,
    ))
    # Volume as bar
    if "Volume" in df.columns:
        fig.add_trace(go.Bar(
            x=df[date_col], y=df["Volume"],
            marker_color="rgba(88,166,255,0.25)",
            name="Volume", yaxis="y2",
        ))
    fig.update_layout(
        title=f"📊 {sym} — 5-Day Candlestick (5min)",
        paper_bgcolor=C["card"], plot_bgcolor=C["card"],
        font_color=C["text"], margin=dict(l=40, r=40, t=40, b=30),
        xaxis_rangeslider_visible=False,
        yaxis=dict(title="Price (₹)"),
        yaxis2=dict(title="Volume", overlaying="y", side="right",
                    showgrid=False),
        legend=dict(orientation="h"),
    )
    return fig


# ── Entry ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = config.DASHBOARD_PORT
    logger.info(f"Dashboard starting on http://localhost:{port}")
    app.run(debug=False, port=port, host="0.0.0.0")
