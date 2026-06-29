"""Feature engineering for the AI regime classifier.

Every feature is either a bounded oscillator (RSI 0–100, ADX 0–100) or a
normalised ratio so the Random Forest sees comparable scales without
non-stationary raw prices leaking in.

The target label is based on *realized volatility*: did the market actually
move hard (Trending) or stay flat (Crab) over the next ``horizon`` bars?
"""

from __future__ import annotations
import math
import statistics
from typing import Dict, List, Optional, Tuple

from indicators import rsi, adx, atr, macd, bollinger_bands, sma

# The names, in order, of every feature column we produce.
FEATURE_NAMES = [
    "rsi",
    "adx",
    "macd_hist_norm",  # MACD histogram / ATR  (removes price-scale)
    "bb_width_pct",  # (upper - lower) / basis  (% bandwidth)
    "volume_ratio",  # volume / volume_ma
    "atr_pct",  # ATR / close  (normalised volatility)
]


def extract_features(data: Dict, p: Dict) -> Tuple[List[List[float]], List[bool]]:
    """Return (rows, valid) where rows[i] is a feature vector for bar i.

    ``valid[i]`` is True when all indicators have warmed up and the row is
    usable.  Rows where valid is False should be masked out before training.
    """
    close, high, low = data["close"], data["high"], data["low"]
    volume = data.get("volume", [])
    n = len(close)

    # -- compute raw indicators --
    r = rsi(close, p.get("rsi_len", 14))
    a = adx(high, low, close, p.get("adx_len", 14))
    atr_vals = atr(high, low, close, p.get("atr_len", 14))
    macd_line, sig_line, hist = macd(
        close, p.get("macd_fast", 12), p.get("macd_slow", 26), p.get("macd_sig", 9)
    )
    bb_basis, bb_upper, bb_lower = bollinger_bands(
        close, p.get("bb_len", 20), p.get("bb_mult", 2.0)
    )
    if volume:
        vma = sma(volume, p.get("vma_len", 20))
    else:
        vma = [None] * n
        volume = [None] * n

    # -- build feature matrix --
    rows: List[List[float]] = []
    valid: List[bool] = []

    for i in range(n):
        # Check all indicators have warmed up
        if (
            r[i] is None
            or a[i] is None
            or atr_vals[i] is None
            or hist[i] is None
            or bb_basis[i] is None
            or bb_upper[i] is None
            or bb_lower[i] is None
        ):
            rows.append([0.0] * len(FEATURE_NAMES))
            valid.append(False)
            continue

        # RSI (0-100)
        feat_rsi = r[i]

        # ADX (0-100)
        feat_adx = a[i]

        # MACD histogram normalised by ATR (removes price scale)
        feat_macd = hist[i] / atr_vals[i] if atr_vals[i] > 0 else 0.0

        # Bollinger Band width as percentage of basis
        feat_bb = (
            (bb_upper[i] - bb_lower[i]) / bb_basis[i] * 100.0
            if bb_basis[i] > 0
            else 0.0
        )

        # Volume / Volume MA ratio
        if vma[i] is not None and vma[i] > 0 and volume[i] is not None:
            feat_vol = volume[i] / vma[i]
        else:
            feat_vol = 1.0  # neutral when no volume data

        # ATR as percentage of close (normalised volatility)
        feat_atr_pct = atr_vals[i] / close[i] * 100.0 if close[i] > 0 else 0.0

        rows.append([feat_rsi, feat_adx, feat_macd, feat_bb, feat_vol, feat_atr_pct])
        valid.append(True)

    return rows, valid


def compute_labels(data: Dict, horizon: int = 20) -> Tuple[List[int], List[bool]]:
    """Compute the regime label for every bar.

    Label = 1 (Trending) if the realised volatility over the next ``horizon``
    bars exceeds an adaptive threshold.  Label = 0 (Crab) otherwise.

    The threshold adapts: it is the rolling median of historical realised
    volatility so the definition of "trending" evolves with the market.

    Returns (labels, label_valid).  label_valid[i] is False for the last
    ``horizon`` bars (where we can't look ahead) and during the warmup.
    """
    close = data["close"]
    n = len(close)

    labels: List[int] = [0] * n
    label_valid: List[bool] = [False] * n

    # Bar-to-bar log returns
    returns = [0.0] * n
    for i in range(1, n):
        if close[i - 1] > 0:
            returns[i] = math.log(close[i] / close[i - 1])

    # Compute realised volatility for each bar's future window
    realised_vols: List[Optional[float]] = [None] * n
    for i in range(n - horizon):
        window = returns[i + 1 : i + 1 + horizon]
        if len(window) >= 2:
            realised_vols[i] = statistics.pstdev(window)

    # Adaptive threshold: rolling median of past realised volatilities
    # Use a lookback of 200 bars to compute the median
    lookback = 200
    for i in range(n - horizon):
        if realised_vols[i] is None:
            continue

        # Gather past valid realised vols for the threshold
        past_vols = []
        for j in range(max(0, i - lookback), i + 1):
            if realised_vols[j] is not None:
                past_vols.append(realised_vols[j])

        if len(past_vols) < 20:
            continue  # Not enough history for a stable threshold

        threshold = statistics.median(past_vols)
        labels[i] = 1 if realised_vols[i] > threshold else 0
        label_valid[i] = True

    return labels, label_valid
