"""Hybrid AI strategy — regime-switching master controller.

The AI predicts the market regime (Crab vs Trend) and delegates trade
execution to the appropriate mathematical strategy.  The AI never places
trades directly; it only flips the switch.

Key rules:
  - If AI confidence < ``min_confidence`` → sit out (no trade).
  - If a position is open and the regime flips, the *current* strategy's
    exit rules finish the trade.  The new regime governs the next entry.
"""

from __future__ import annotations
from typing import Dict, List, Optional

from indicators import ema, rsi, atr, adx, bollinger_bands, macd, sma

# Sub-strategy defaults (imported from their modules)
from strategy import DEFAULTS as TREND_DEFAULTS
from crab_strategy import DEFAULTS as CRAB_DEFAULTS

DEFAULTS: Dict = dict(
    # Feature params
    rsi_len=14,
    adx_len=14,
    atr_len=14,
    macd_fast=12,
    macd_slow=26,
    macd_sig=9,
    bb_len=20,
    bb_mult=2.0,
    vma_len=20,
    # AI params
    min_confidence=0.60,
    horizon=20,
    # Sub-strategy params are pulled from their respective DEFAULTS
)

# No parameter grid — the AI *is* the optimizer.
PARAM_GRID = {}


def generate_signals(
    data: Dict, p: Dict, regime_predictions: Optional[List] = None
) -> Dict[str, List]:
    """Generate signals by delegating to sub-strategies based on regime.

    ``regime_predictions`` is a list of (prediction, confidence) tuples from
    ai_model.predict_regime().  If None, falls back to all-Crab (for testing).
    """
    close, high, low = data["close"], data["high"], data["low"]
    volume = data.get("volume", [])
    n = len(close)

    min_conf = p.get("min_confidence", 0.60)

    # -- Pre-compute all indicators needed by both sub-strategies --

    # Trend strategy indicators
    trend_p = dict(TREND_DEFAULTS)
    trend_ema = ema(close, trend_p["ema_len"])
    trend_rsi = rsi(close, trend_p["rsi_len"])
    trend_atr = atr(high, low, close, trend_p.get("atr_len", 14))

    # Crab strategy indicators
    crab_p = dict(CRAB_DEFAULTS)
    crab_atr = atr(high, low, close, crab_p.get("atr_len", 14))
    crab_adx = adx(high, low, close, crab_p.get("adx_len", 14))
    bb_basis, bb_upper, bb_lower = bollinger_bands(
        close, crab_p.get("bb_len", 20), crab_p.get("bb_mult", 2.0)
    )
    macd_line, sig_line, hist = macd(
        close,
        crab_p.get("macd_fast", 12),
        crab_p.get("macd_slow", 26),
        crab_p.get("macd_sig", 9),
    )

    if volume:
        vma = sma(volume, crab_p.get("vma_len", 20))
    else:
        vma = [0] * n
        volume = [1] * n

    # -- Output arrays --
    entry = [0] * n
    exit_ = [False] * n
    stop_dist: List = [None] * n
    tp_dist: List = [None] * n

    for i in range(1, n):
        # Determine regime for this bar
        if regime_predictions is not None and i < len(regime_predictions):
            pred, conf = regime_predictions[i]
        else:
            pred, conf = 0, 0.5  # default: uncertain

        # Sit out if AI is uncertain
        if conf < min_conf:
            continue

        if pred == 1:
            # === TREND REGIME ===
            if (
                trend_ema[i] is None
                or trend_rsi[i] is None
                or trend_rsi[i - 1] is None
                or trend_atr[i] is None
            ):
                continue
            price = close[i]

            # Trend exit: RSI crosses the midline
            ex = trend_p["exit_level"]
            if (trend_rsi[i - 1] < ex <= trend_rsi[i]) or (
                trend_rsi[i - 1] > ex >= trend_rsi[i]
            ):
                exit_[i] = True

            # Trend entry
            up = price > trend_ema[i]
            dn = price < trend_ema[i]
            long_c = (
                trend_p["allow_long"]
                and up
                and trend_rsi[i - 1] < trend_p["oversold"] <= trend_rsi[i]
            )
            short_c = (
                trend_p["allow_short"]
                and dn
                and trend_rsi[i - 1] > trend_p["overbought"] >= trend_rsi[i]
            )
            if long_c:
                entry[i] = 1
                stop_dist[i] = trend_atr[i] * trend_p["stop_atr"]
            elif short_c:
                entry[i] = -1
                stop_dist[i] = trend_atr[i] * trend_p["stop_atr"]

        else:
            # === CRAB REGIME ===
            if (
                crab_atr[i] is None
                or crab_adx[i] is None
                or bb_upper[i] is None
                or hist[i] is None
                or hist[i - 1] is None
                or vma[i] is None
            ):
                continue

            # Breakout protection still overrides even if AI says crab
            if crab_adx[i] > crab_p["adx_threshold"]:
                continue

            good_volume = volume[i] > vma[i]
            hit_lower = low[i] <= bb_lower[i] and close[i] > bb_lower[i]
            macd_up = hist[i] > hist[i - 1]
            hit_upper = high[i] >= bb_upper[i] and close[i] < bb_upper[i]
            macd_down = hist[i] < hist[i - 1]

            long_c = crab_p["allow_long"] and good_volume and hit_lower and macd_up
            short_c = crab_p["allow_short"] and good_volume and hit_upper and macd_down

            if long_c:
                entry[i] = 1
                stop_dist[i] = crab_atr[i] * crab_p["stop_atr"]
                tp_dist[i] = crab_atr[i] * crab_p["tp_atr"]
            elif short_c:
                entry[i] = -1
                stop_dist[i] = crab_atr[i] * crab_p["stop_atr"]
                tp_dist[i] = crab_atr[i] * crab_p["tp_atr"]

    return {"entry": entry, "exit": exit_, "stop_dist": stop_dist, "tp_dist": tp_dist}
