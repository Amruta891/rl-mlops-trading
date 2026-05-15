"""
execution.py — Paper Trading Engine (no broker API keys required)
Simulates realistic NSE execution with slippage, commission, and
integer share quantities using live yfinance prices.
"""
import os, logging
from datetime import datetime
import numpy as np
import pandas as pd

import config
from mlops_monitor import TRADE_COUNTER, TRADE_ERRORS

logger = logging.getLogger(__name__)


# ── Risk Engine ──────────────────────────────────────────────────
def apply_risk(weights: np.ndarray) -> np.ndarray:
    """Clip each position to [MIN, MAX] then re-normalise."""
    weights = np.clip(weights, config.MIN_POSITION_PCT, config.MAX_POSITION_PCT)
    total   = weights.sum()
    if total < 1e-8:
        return np.full(len(weights), 1.0 / len(weights))
    return weights / total


def compute_quantities(weights, prices, capital):
    """Convert weights + prices + capital → integer share counts."""
    return {
        sym: max(0, int((w * capital) / p)) if p > 0 else 0
        for sym, w, p in zip(config.SYMBOLS, weights, prices)
    }


# ── Paper Broker ─────────────────────────────────────────────────
class PaperBroker:
    """
    Simulates order execution locally — no API keys, no internet calls.

    Features:
      • Slippage  : ±0.05% of fill price (market impact)
      • Commission: 0.1% per trade (NSE realistic)
      • Integer   : only whole shares are traded
      • Balance   : tracked across the session
      • Holdings  : per-symbol share count tracked
    """

    def __init__(self, capital: float = None):
        self.capital  = capital or config.INITIAL_CAPITAL
        self.balance  = self.capital
        self.holdings = {s: 0 for s in config.SYMBOLS}
        self.orders   = []
        logger.info(f"PaperBroker ready — starting capital ₹{self.balance:,.0f}")

    # ------------------------------------------------------------------
    def place_order(self, symbol: str, side: str,
                qty: int, price: float):
        if qty <= 0 or price <= 0:
            return None

        # Slippage: buy at slightly higher, sell at slightly lower
        slip       = price * 0.0005 * (1 if side == "BUY" else -1)
        fill_price = max(0.01, price + slip)
        commission = fill_price * qty * config.TRANSACTION_COST

        if side == "BUY":
            total_cost = fill_price * qty + commission
            if total_cost > self.balance:
                qty = int(self.balance / (fill_price * (1 + config.TRANSACTION_COST)))
            if qty <= 0:
                logger.debug(f"Skip BUY {symbol}: insufficient balance")
                return None
            self.balance            -= fill_price * qty + commission
            self.holdings[symbol]   += qty

        elif side == "SELL":
            qty = min(qty, self.holdings.get(symbol, 0))
            if qty <= 0:
                logger.debug(f"Skip SELL {symbol}: no holdings")
                return None
            self.balance            += fill_price * qty - commission
            self.holdings[symbol]   -= qty

        else:
            return None

        order = {
            "order_id"   : f"PAPER-{len(self.orders)+1:06d}",
            "symbol"     : symbol,
            "side"       : side,
            "qty"        : qty,
            "fill_price" : round(fill_price, 2),
            "commission" : round(commission, 2),
            "net_amount" : round(fill_price * qty + commission
                                 if side == "BUY"
                                 else fill_price * qty - commission, 2),
            "timestamp"  : datetime.now().isoformat(),
            "status"     : "FILLED",
        }
        self.orders.append(order)
        TRADE_COUNTER.labels(symbol=symbol, side=side).inc()

        logger.info(
            f"[PAPER] {side:4s} {qty:>5}×{symbol:<10} "
            f"@ ₹{fill_price:>9.2f}  comm ₹{commission:.2f}  "
            f"bal ₹{self.balance:,.0f}"
        )
        return order

    # ------------------------------------------------------------------
    def portfolio_value(self, prices: dict) -> float:
        """Mark-to-market value = cash + holdings × latest price."""
        holdings_val = sum(
            self.holdings.get(s, 0) * prices.get(s, 0)
            for s in config.SYMBOLS
        )
        return self.balance + holdings_val

    def summary(self) -> dict:
        return {
            "cash_balance": round(self.balance, 2),
            "total_orders": len(self.orders),
            "holdings"    : dict(self.holdings),
        }


# ── Rebalance Helper ─────────────────────────────────────────────
def execute_rebalance(broker: PaperBroker,
                      weights: np.ndarray,
                      prices:  np.ndarray,
                      capital: float,
                      prev_weights = None) -> tuple:
    """
    Compute weight deltas → place BUY/SELL orders to rebalance.
    Skips positions where weight change < 1% (avoids churn).
    Returns (list[trade_records], new_weights).
    """
    weights      = apply_risk(weights)
    target_qtys  = compute_quantities(weights, prices, capital)
    trades       = []
    prev_weights = prev_weights if prev_weights is not None \
                   else np.zeros(config.N_ASSETS)

    for i, sym in enumerate(config.SYMBOLS):
        delta = float(weights[i]) - float(prev_weights[i])
        if abs(delta) < 0.01:          # <1% weight shift → skip
            continue

        side  = "BUY" if delta > 0 else "SELL"
        qty   = target_qtys[sym]
        price = float(prices[i])

        result = broker.place_order(sym, side, qty, price)
        if result:
            result["target_weight"] = round(float(weights[i]), 4)
            result["price_ref"]     = round(price, 2)
            trades.append(result)

    return trades, weights


# ── Trade log persistence ─────────────────────────────────────────
def save_trades(trades: list, path: str = "logs/trades.csv"):
    if not trades:
        return
    os.makedirs("logs", exist_ok=True)
    df     = pd.DataFrame(trades)
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", index=False, header=header)
