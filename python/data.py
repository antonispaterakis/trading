"""Fetch and cache real OHLCV candles from the Binance public API (no API key).

Data is paginated backwards in 1000-candle pages and cached on disk for one hour
so repeated runs are fast and don't hammer the API.
"""

from __future__ import annotations
import json
import os
import time
import urllib.request

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "data_cache")
CACHE_TTL = 3600  # seconds


def _get(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def fetch_klines(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    bars: int = 4800,
    use_cache: bool = True,
    market_type: str = "futures",
) -> dict:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(
        CACHE_DIR, f"{symbol}_{interval}_{bars}_{market_type}.json"
    )
    if (
        use_cache
        and os.path.exists(cache_path)
        and (time.time() - os.path.getmtime(cache_path)) < CACHE_TTL
    ):
        return json.load(open(cache_path))

    collected: dict[int, list] = {}
    end = int(time.time() * 1000)

    base_url = (
        "https://fapi.binance.com/fapi/v1"
        if market_type == "futures"
        else "https://api.binance.com/api/v3"
    )

    while len(collected) < bars:
        url = (
            f"{base_url}/klines?symbol={symbol}"
            f"&interval={interval}&limit=1000&endTime={end}"
        )
        data = _get(url)
        if not data:
            break
        for k in data:
            collected[int(k[0])] = k
        end = int(data[0][0]) - 1
        if len(data) < 1000:
            break
        time.sleep(0.25)

    keys = sorted(collected)[-bars:]
    ohlcv = {
        "symbol": symbol,
        "interval": interval,
        "time": [collected[t][0] // 1000 for t in keys],
        "open": [float(collected[t][1]) for t in keys],
        "high": [float(collected[t][2]) for t in keys],
        "low": [float(collected[t][3]) for t in keys],
        "close": [float(collected[t][4]) for t in keys],
        "volume": [float(collected[t][5]) for t in keys],
    }
    if use_cache:
        json.dump(ohlcv, open(cache_path, "w"))
    return ohlcv
