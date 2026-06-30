"""Fetch alternative data from Binance Futures and strictly align it to OHLCV.

Alternative data includes:
- Funding rates
- Open Interest (historical)
- Long/Short Ratio
"""

from __future__ import annotations
import json
import os
import time
import bisect
from typing import Dict, List

from data import CACHE_DIR, CACHE_TTL, _get


def fetch_funding_rate(symbol: str = "BTCUSDT", limit: int = 1000) -> Dict:
    """Fetch funding rate history."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{symbol}_funding_{limit}.json")
    if (
        os.path.exists(cache_path)
        and (time.time() - os.path.getmtime(cache_path)) < CACHE_TTL
    ):
        return json.load(open(cache_path))

    collected: Dict[int, float] = {}
    end = int(time.time() * 1000)

    while len(collected) < limit:
        url = f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={symbol}&limit=1000&endTime={end}"
        data = _get(url)
        if not data:
            break
        for item in data:
            collected[item["fundingTime"]] = float(item["fundingRate"])
        end = data[0]["fundingTime"] - 1
        if len(data) < 1000:
            break
        time.sleep(0.25)

    keys = sorted(collected)[-limit:]
    res = {
        "time": [t // 1000 for t in keys],
        "funding_rate": [collected[t] for t in keys],
    }
    json.dump(res, open(cache_path, "w"))
    return res


def fetch_open_interest_hist(
    symbol: str = "BTCUSDT", period: str = "4h", limit: int = 1000
) -> Dict:
    """Fetch historical Open Interest."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{symbol}_oi_{period}_{limit}.json")
    if (
        os.path.exists(cache_path)
        and (time.time() - os.path.getmtime(cache_path)) < CACHE_TTL
    ):
        return json.load(open(cache_path))

    collected: Dict[int, float] = {}
    end = int(time.time() * 1000)

    while len(collected) < limit:
        url = f"https://fapi.binance.com/futures/data/openInterestHist?symbol={symbol}&period={period}&limit=500&endTime={end}"
        data = _get(url)
        if not data:
            break
        for item in data:
            collected[item["timestamp"]] = float(item["sumOpenInterestValue"])
        end = data[0]["timestamp"] - 1
        if len(data) < 500:
            break
        time.sleep(0.25)

    keys = sorted(collected)[-limit:]
    res = {"time": [t // 1000 for t in keys], "oi_value": [collected[t] for t in keys]}
    json.dump(res, open(cache_path, "w"))
    return res


def fetch_long_short_ratio(
    symbol: str = "BTCUSDT", period: str = "4h", limit: int = 1000
) -> Dict:
    """Fetch Top Trader Long/Short Ratio (Accounts)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{symbol}_ls_{period}_{limit}.json")
    if (
        os.path.exists(cache_path)
        and (time.time() - os.path.getmtime(cache_path)) < CACHE_TTL
    ):
        return json.load(open(cache_path))

    collected: Dict[int, float] = {}
    end = int(time.time() * 1000)

    while len(collected) < limit:
        url = f"https://fapi.binance.com/futures/data/globalLongShortAccountRatio?symbol={symbol}&period={period}&limit=500&endTime={end}"
        data = _get(url)
        if not data:
            break
        for item in data:
            collected[item["timestamp"]] = float(item["longShortRatio"])
        end = data[0]["timestamp"] - 1
        if len(data) < 500:
            break
        time.sleep(0.25)

    keys = sorted(collected)[-limit:]
    res = {"time": [t // 1000 for t in keys], "ls_ratio": [collected[t] for t in keys]}
    json.dump(res, open(cache_path, "w"))
    return res


def align_alt_data(klines: Dict, alt_data: Dict, value_key: str) -> List[float]:
    """Strictly align alternative data to klines without lookahead bias.

    For a given kline open time T, we can only use alt data with timestamp <= T.
    Uses bisect to efficiently find the latest available alt data point.
    """
    k_times = klines["time"]
    a_times = alt_data["time"]
    a_vals = alt_data[value_key]

    aligned = [0.0] * len(k_times)

    if not a_times:
        return aligned

    for i, t in enumerate(k_times):
        # Find rightmost index where a_time <= t
        idx = bisect.bisect_right(a_times, t) - 1
        if idx >= 0:
            aligned[i] = a_vals[idx]
        else:
            aligned[i] = a_vals[0] if a_vals else 0.0

    return aligned
