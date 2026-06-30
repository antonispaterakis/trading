"""Causal technical indicators in pure Python (no third-party deps).

Every value at bar i depends only on bars <= i, so computing an indicator over
the full series and then slicing by index introduces no look-ahead bias.
Wilder smoothing (rma) is used for RSI/ATR/ADX to match TradingView.
"""

from __future__ import annotations
from typing import List, Optional

Series = List[Optional[float]]


def ema(src: List[float], length: int) -> Series:
    out: Series = [None] * len(src)
    k = 2.0 / (length + 1)
    prev: Optional[float] = None
    for i, v in enumerate(src):
        prev = v if prev is None else (v - prev) * k + prev
        out[i] = prev
    return out


def rma(src: Series, length: int) -> Series:
    """Wilder's running moving average. None inputs carry the previous value."""
    out: Series = [None] * len(src)
    prev: Optional[float] = None
    for i, v in enumerate(src):
        if v is None:
            out[i] = prev
            continue
        prev = v if prev is None else (prev * (length - 1) + v) / length
        out[i] = prev
    return out


def rsi(src: List[float], length: int) -> Series:
    n = len(src)
    gains: Series = [None] * n
    losses: Series = [None] * n
    for i in range(1, n):
        ch = src[i] - src[i - 1]
        gains[i] = max(ch, 0.0)
        losses[i] = max(-ch, 0.0)
    ag = rma(gains, length)
    al = rma(losses, length)
    out: Series = [None] * n
    for i in range(n):
        if ag[i] is None or al[i] is None:
            continue
        out[i] = 100.0 if al[i] == 0 else 100.0 - 100.0 / (1.0 + ag[i] / al[i])
    return out


def true_range(high: List[float], low: List[float], close: List[float]) -> Series:
    n = len(close)
    tr: Series = [None] * n
    for i in range(n):
        if i == 0:
            tr[i] = high[i] - low[i]
        else:
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
    return tr


def atr(high: List[float], low: List[float], close: List[float], length: int) -> Series:
    return rma(true_range(high, low, close), length)


def adx(
    high: List[float], low: List[float], close: List[float], length: int = 14
) -> Series:
    n = len(close)
    pdm = [0.0] * n
    ndm = [0.0] * n
    tr = [0.0] * n
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        pdm[i] = up if (up > dn and up > 0) else 0.0
        ndm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(
            high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])
        )
    str_ = rma(tr, length)
    spdm = rma(pdm, length)
    sndm = rma(ndm, length)
    dx: Series = [None] * n
    for i in range(n):
        if str_[i] and str_[i] > 0:
            pdi = 100.0 * spdm[i] / str_[i]
            ndi = 100.0 * sndm[i] / str_[i]
            s = pdi + ndi
            dx[i] = 100.0 * abs(pdi - ndi) / s if s > 0 else 0.0
    return rma(dx, length)


def sma(src: List[float], length: int) -> Series:
    n = len(src)
    out: Series = [None] * n
    # Keep a running sum to make it O(N) instead of O(N*length)
    if n < length:
        return out

    current_sum = sum(src[:length])
    out[length - 1] = current_sum / length

    for i in range(length, n):
        current_sum = current_sum + src[i] - src[i - length]
        out[i] = current_sum / length

    return out


def stdev(src: List[float], length: int, mean_series: Series = None) -> Series:
    import math

    n = len(src)
    out: Series = [None] * n
    if mean_series is None:
        mean_series = sma(src, length)

    for i in range(length - 1, n):
        if mean_series[i] is None:
            continue
        m = mean_series[i]
        variance = sum((x - m) ** 2 for x in src[i - length + 1 : i + 1]) / length
        out[i] = math.sqrt(variance)
    return out


def bollinger_bands(
    src: List[float], length: int, mult: float
) -> tuple[Series, Series, Series]:
    basis = sma(src, length)
    dev = stdev(src, length, basis)
    n = len(src)
    upper: Series = [None] * n
    lower: Series = [None] * n
    for i in range(n):
        if basis[i] is not None and dev[i] is not None:
            upper[i] = basis[i] + mult * dev[i]
            lower[i] = basis[i] - mult * dev[i]
    return basis, upper, lower


def macd(
    src: List[float], fast_len: int, slow_len: int, sig_len: int
) -> tuple[Series, Series, Series]:
    fast_ema = ema(src, fast_len)
    slow_ema = ema(src, slow_len)
    n = len(src)

    macd_line: Series = [None] * n
    for i in range(n):
        if fast_ema[i] is not None and slow_ema[i] is not None:
            macd_line[i] = fast_ema[i] - slow_ema[i]

    # Calculate signal line which is EMA of MACD line
    # Since ema() doesn't handle None inputs natively when calculating the first value,
    # we filter out Nones, run EMA, and pad back
    valid_idx = [i for i, v in enumerate(macd_line) if v is not None]
    if not valid_idx:
        return macd_line, [None] * n, [None] * n

    first_valid = valid_idx[0]
    valid_macd = [v for v in macd_line if v is not None]
    sig_line_valid = ema(valid_macd, sig_len)

    sig_line: Series = [None] * first_valid + sig_line_valid
    hist_line: Series = [None] * n

    for i in range(n):
        if macd_line[i] is not None and sig_line[i] is not None:
            hist_line[i] = macd_line[i] - sig_line[i]

    return macd_line, sig_line, hist_line


def donchian_channel(
    high: List[float], low: List[float], length: int
) -> tuple[Series, Series]:
    n = len(high)
    upper: Series = [None] * n
    lower: Series = [None] * n

    if n < length:
        return upper, lower

    for i in range(length - 1, n):
        # The breakout uses the high of the PREVIOUS n bars.
        # But commonly Donchian channel is defined over the last n bars.
        # We will compute it over the last n bars. The breakout strategy
        # can check if price > previous upper channel.
        upper[i] = max(high[i - length + 1 : i + 1])
        lower[i] = min(low[i - length + 1 : i + 1])

    return upper, lower
