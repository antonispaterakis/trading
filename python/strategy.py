"""Strategy logic: trend-filtered mean reversion (symmetric long/short).

The strategy is intentionally simple and transparent. Empirically (see the
walk-forward results in the README) a *simple* rule generalises better than a
heavily-parameterised one. Signals are causal: bar i uses only indicators up to i.

generate_signals returns, for every bar:
  entry[i]      : +1 open long, -1 open short, 0 nothing
  exit[i]       : True -> close any open position at this bar's close
  stop_dist[i]  : ATR-based stop distance to use if a position opens at bar i
"""
from __future__ import annotations
from typing import Dict, List

from indicators import ema, rsi, atr

DEFAULTS: Dict = dict(
    ema_len=200,      # trend filter: longs only above it, shorts only below
    rsi_len=14,
    oversold=30,      # long when RSI crosses up through this
    overbought=70,    # short when RSI crosses down through this
    exit_level=50,    # take profit when RSI reverts back through the midline
    atr_len=14,
    stop_atr=2.5,     # hard stop = stop_atr * ATR
    allow_long=True,
    allow_short=True,
)

PARAM_GRID = {
    "ema_len": [100, 150, 200],
    "rsi_len": [7, 14],
    "oversold": [25, 30, 35],
    "stop_atr": [2.0, 2.5, 3.0],
}


def generate_signals(data: Dict, p: Dict) -> Dict[str, List]:
    close, high, low = data["close"], data["high"], data["low"]
    n = len(close)
    e = ema(close, p["ema_len"])
    r = rsi(close, p["rsi_len"])
    a = atr(high, low, close, p.get("atr_len", 14))

    entry = [0] * n
    exit_ = [False] * n
    stop_dist: List = [None] * n
    ex = p["exit_level"]

    for i in range(1, n):
        if e[i] is None or r[i] is None or r[i - 1] is None or a[i] is None:
            continue
        price = close[i]
        # mean-reversion take profit: RSI crosses the midline in either direction
        if (r[i - 1] < ex <= r[i]) or (r[i - 1] > ex >= r[i]):
            exit_[i] = True
        up = price > e[i]
        dn = price < e[i]
        long_c = p["allow_long"] and up and (r[i - 1] < p["oversold"] <= r[i])
        short_c = p["allow_short"] and dn and (r[i - 1] > p["overbought"] >= r[i])
        if long_c:
            entry[i] = 1
        elif short_c:
            entry[i] = -1
        stop_dist[i] = a[i] * p["stop_atr"]

    return {"entry": entry, "exit": exit_, "stop_dist": stop_dist}
