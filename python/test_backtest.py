from __future__ import annotations
import math

from backtest import run


def test_backtester():
    # 10 bars, 1 day.
    # Let's say 86400 is exactly 1 day. Let's start from time = 0.
    data = {
        "time": [i * 3600 for i in range(10)],
        "close": [100.0, 105.0, 110.0, 108.0, 115.0, 120.0, 118.0, 112.0, 105.0, 100.0],
        "high": [100.0, 106.0, 112.0, 110.0, 116.0, 122.0, 120.0, 115.0, 108.0, 102.0],
        "low": [98.0, 100.0, 105.0, 106.0, 108.0, 115.0, 112.0, 108.0, 102.0, 98.0],
    }

    # We enter long on bar 2 at close = 110.0
    # Stop is 105.0 (stop_dist = 5.0)
    # Trail is 4.0
    signals = {
        "entry": [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        "exit": [False] * 10,
        "stop_dist": [None, None, 5.0, None, None, None, None, None, None, None],
        "trail_dist": [None, None, 4.0, None, None, None, None, None, None, None],
    }

    params = {
        "commission": 0.0,  # 0 for easier math first
        "slippage": 0.0,
        "risk_pct": 1.0,  # 1% of 10000 = 100. 100 / 5.0 = 20 quantity.
        "equity": 10000.0,
        "conservative_trail": True,
        "max_daily_loss_pct": None,
        "cooldown_bars": 0,
    }

    # Trace:
    # Bar 2: Enter Long at 110.0. Qty = 100 / 5 = 20. Stop = 105.0.
    # Bar 3: Close = 108.0. Conservative: updates stop using bar 2 high (112.0).
    #        New stop = max(105.0, 112.0 - 4.0) = 108.0.
    #        Check stop against Bar 3 low (106.0). 106.0 <= 108.0 -> STOPPED OUT!
    #        Exit price = 108.0. PNL = (108 - 110) * 20 = -40.0.

    res = run(data, signals, params, lo=1)

    trades = res["trades"]
    assert len(trades) == 1
    t = trades[0]
    assert t["entry"] == 110.0
    assert t["exit"] == 108.0
    assert t["pnl"] == -40.0
    assert t["stopped"] is True

    # Now let's test short trade with slippage and commission
    # Bar 4: Enter Short at 115.0
    # Stop dist = 5.0 -> stop at 120.0
    signals = {
        "entry": [0, 0, 0, 0, -1, 0, 0, 0, 0, 0],
        "exit": [False] * 10,
        "stop_dist": [None, None, None, None, 5.0, None, None, None, None, None],
        "trail_dist": [None, None, None, None, 10.0, None, None, None, None, None],
    }
    params["commission"] = 0.001  # 0.1%
    params["slippage"] = 0.001  # 0.1%

    # Trace:
    # Bar 4: Entry price = 115.0. d = -1. slip = 0.001.
    #        entry_eff = 115 * (1 + 0.001 * -1) = 115 * 0.999 = 114.885
    #        Stop dist = 5.0. Qty = (10000 * 0.01) / 5.0 = 20.
    #        Comm paid = 114.885 * 20 * 0.001 = 2.2977
    #        Stop = 115 + 5 = 120.0
    # Bar 5: High is 122.0. This hits the stop of 120.0!
    #        Exit price = 120.0. Stopped = True.
    #        eff_exit = 120 * (1 - 0.001 * -1) = 120 * 1.001 = 120.12
    #        PNL = (120.12 - 114.885) * 20 * -1 - (120.12 * 20 * 0.001)
    #        PNL = (5.235) * -20 - (2.4024) = -104.7 - 2.4024 = -107.1024

    res = run(data, signals, params, lo=1)
    trades = res["trades"]
    assert len(trades) == 1
    t = trades[0]
    assert t["entry"] == 115.0
    assert t["exit"] == 118.0
    assert math.isclose(t["pnl"], -67.02236, rel_tol=1e-5)

    print("Backtester validation passed.")


if __name__ == "__main__":
    test_backtester()
