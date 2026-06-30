"""Donchian Channel Breakout Strategy with Trailing Stop.

A trend-following strategy that buys when price breaks out of the Donchian
Channel (n-bar high) and exits only via a trailing stop.
"""

from __future__ import annotations
from typing import Dict, List

from indicators import donchian_channel, atr

DEFAULTS: Dict = dict(
    breakout_len=40,
    atr_len=14,
    stop_atr=2.0,  # Initial hard stop
    trail_atr=3.0,  # Trailing stop distance
    allow_long=True,
    allow_short=True,
)

PARAM_GRID = {
    "breakout_len": [20, 40, 60],
    "stop_atr": [1.5, 2.0],
    "trail_atr": [2.0, 3.0, 4.0],
}


def generate_signals(data: Dict, p: Dict) -> Dict[str, List]:
    close, high, low = data["close"], data["high"], data["low"]
    n = len(close)

    dc_upper, dc_lower = donchian_channel(high, low, p["breakout_len"])
    a = atr(high, low, close, p.get("atr_len", 14))

    entry = [0] * n
    exit_ = [False] * n
    stop_dist: List = [None] * n
    trail_dist: List = [None] * n

    for i in range(1, n):
        if dc_upper[i - 1] is None or dc_lower[i - 1] is None or a[i] is None:
            continue

        # Breakout occurs if current high/low breaks the PREVIOUS bar's channel.
        # This is because the current bar's high/low is included in the current channel
        # which would mean price == channel upper always if it's making a new high.
        long_c = p["allow_long"] and high[i] > dc_upper[i - 1]
        short_c = p["allow_short"] and low[i] < dc_lower[i - 1]

        if long_c:
            entry[i] = 1
            stop_dist[i] = a[i] * p["stop_atr"]
            trail_dist[i] = a[i] * p["trail_atr"]
        elif short_c:
            entry[i] = -1
            stop_dist[i] = a[i] * p["stop_atr"]
            trail_dist[i] = a[i] * p["trail_atr"]

    return {
        "entry": entry,
        "exit": exit_,
        "stop_dist": stop_dist,
        "trail_dist": trail_dist,
    }
