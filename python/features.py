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
    "rsi_slope",  # RSI(i) - RSI(i-5)
    "atr_pct_slope",  # ATR_PCT(i) - ATR_PCT(i-5)
    "funding_rate",
    "ls_ratio",
    "oi_change_pct",  # (OI(i) - OI(i-5)) / OI(i-5)
    "oi_change_pct_12",  # OI momentum over 12 bars
    "ma_100_dist",  # close / sma(100) - 1
    "ma_200_dist",  # close / sma(200) - 1
]


def extract_features(data: Dict, p: Dict) -> Tuple[List[List[float]], List[bool]]:
    """Return (rows, valid) where rows[i] is a feature vector for bar i.

    ``valid[i]`` is True when all indicators have warmed up and the row is
    usable.  Rows where valid is False should be masked out before training.
    """
    close, high, low = data["close"], data["high"], data["low"]
    volume = data.get("volume", [])

    # Alt data arrays (default to zeros if missing)
    n = len(close)
    funding_rate = data.get("funding_rate", [0.0] * n)
    ls_ratio = data.get("ls_ratio", [1.0] * n)
    oi_value = data.get("oi_value", [0.0] * n)

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

    sma100 = sma(close, 100)
    sma200 = sma(close, 200)

    # -- build feature matrix --
    rows: List[List[float]] = []
    valid: List[bool] = []

    slope_len = 5  # Used for temporal context (e.g. 5 bars ago)
    slope_len_long = 12  # Used for longer temporal context

    for i in range(n):
        # Check all indicators have warmed up
        if (
            i < max(slope_len_long, 200)
            or r[i] is None
            or a[i] is None
            or atr_vals[i] is None
            or hist[i] is None
            or bb_basis[i] is None
            or bb_upper[i] is None
            or bb_lower[i] is None
            or r[i - slope_len] is None
            or atr_vals[i - slope_len] is None
            or sma100[i] is None
            or sma200[i] is None
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

        # Temporal context: Slopes
        feat_rsi_slope = r[i] - r[i - slope_len]

        prev_atr_pct = (
            atr_vals[i - slope_len] / close[i - slope_len] * 100.0
            if close[i - slope_len] > 0
            else 0.0
        )
        feat_atr_pct_slope = feat_atr_pct - prev_atr_pct

        # Alt Data
        feat_funding = funding_rate[i]
        feat_ls = ls_ratio[i]

        # OI Change Pct over 5 bars
        if oi_value[i - slope_len] > 0:
            feat_oi_change = (
                (oi_value[i] - oi_value[i - slope_len])
                / oi_value[i - slope_len]
                * 100.0
            )
        else:
            feat_oi_change = 0.0

        # OI Change Pct over 12 bars (3 hours)
        if oi_value[i - slope_len_long] > 0:
            feat_oi_change_12 = (
                (oi_value[i] - oi_value[i - slope_len_long])
                / oi_value[i - slope_len_long]
                * 100.0
            )
        else:
            feat_oi_change_12 = 0.0

        # Macro distance
        feat_ma_100_dist = (
            (close[i] / sma100[i] - 1.0) * 100.0 if sma100[i] > 0 else 0.0
        )
        feat_ma_200_dist = (
            (close[i] / sma200[i] - 1.0) * 100.0 if sma200[i] > 0 else 0.0
        )

        rows.append(
            [
                feat_rsi,
                feat_adx,
                feat_macd,
                feat_bb,
                feat_vol,
                feat_atr_pct,
                feat_rsi_slope,
                feat_atr_pct_slope,
                feat_funding,
                feat_ls,
                feat_oi_change,
                feat_oi_change_12,
                feat_ma_100_dist,
                feat_ma_200_dist,
            ]
        )
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
