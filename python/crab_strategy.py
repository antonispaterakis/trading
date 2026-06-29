"""Crab Market Day Trader Strategy (Scalping).

Focuses on ranging (sideways) markets and extracts small, high-probability
profits by bouncing between Bollinger Bands when the market is not trending.
"""

from __future__ import annotations
from typing import Dict, List

from indicators import bollinger_bands, macd, sma, adx, atr

DEFAULTS: Dict = dict(
    bb_len=20,
    bb_mult=2.0,
    macd_fast=12,
    macd_slow=26,
    macd_sig=9,
    vma_len=20,
    adx_len=14,
    adx_threshold=25,  # Above 25 = trending (dangerous breakout). Below = ranging (crab).
    atr_len=14,
    stop_atr=1.0,  # Tight stop loss for quick exits
    tp_atr=1.5,  # Tight take profit for scalping
    allow_long=True,
    allow_short=True,
)

PARAM_GRID = {
    "bb_len": [14, 20],
    "bb_mult": [2.0],
    "adx_threshold": [25, 30],  # Below this means crab market
    "stop_atr": [0.8, 1.0, 1.2],
    "tp_atr": [1.0, 1.5],
}


def generate_signals(data: Dict, p: Dict) -> Dict[str, List]:
    close, high, low, volume = (
        data["close"],
        data["high"],
        data["low"],
        data.get("volume", []),
    )
    n = len(close)

    # Calculate indicators
    a = atr(high, low, close, p.get("atr_len", 14))
    adx_line = adx(high, low, close, p.get("adx_len", 14))
    bb_basis, bb_upper, bb_lower = bollinger_bands(
        close, p.get("bb_len", 20), p.get("bb_mult", 2.0)
    )
    macd_line, sig_line, hist = macd(
        close, p.get("macd_fast", 12), p.get("macd_slow", 26), p.get("macd_sig", 9)
    )

    if volume:
        vma = sma(volume, p.get("vma_len", 20))
    else:
        vma = [0] * n  # Fallback if no volume data
        volume = [1] * n

    entry = [0] * n
    exit_ = [False] * n
    stop_dist: List = [None] * n
    tp_dist: List = [None] * n

    for i in range(1, n):
        if (
            a[i] is None
            or adx_line[i] is None
            or bb_upper[i] is None
            or hist[i] is None
            or hist[i - 1] is None
            or vma[i] is None
        ):
            continue

        # 1. Crab Market Check: ADX must be below threshold (no strong trend)
        # If it's above, we PAUSE all trading.
        if adx_line[i] > p["adx_threshold"]:
            continue

        # 2. Wallet/Liquidity Check: Volume must be above moving average
        good_volume = volume[i] > vma[i]

        # 3. Reversion Triggers
        # Long: Price hits lower band, MACD momentum is shifting up
        hit_lower = low[i] <= bb_lower[i] and close[i] > bb_lower[i]
        macd_up = hist[i] > hist[i - 1]

        # Short: Price hits upper band, MACD momentum is shifting down
        hit_upper = high[i] >= bb_upper[i] and close[i] < bb_upper[i]
        macd_down = hist[i] < hist[i - 1]

        long_c = p["allow_long"] and good_volume and hit_lower and macd_up
        short_c = p["allow_short"] and good_volume and hit_upper and macd_down

        if long_c:
            entry[i] = 1
            stop_dist[i] = a[i] * p["stop_atr"]
            tp_dist[i] = a[i] * p["tp_atr"]
        elif short_c:
            entry[i] = -1
            stop_dist[i] = a[i] * p["stop_atr"]
            tp_dist[i] = a[i] * p["tp_atr"]

    return {"entry": entry, "exit": exit_, "stop_dist": stop_dist, "tp_dist": tp_dist}
