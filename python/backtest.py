"""Event-driven backtester + honest performance metrics.

Models the things that quietly kill real strategies:
  - commission on both sides of every trade
  - slippage that always works against you
  - one position at a time, sized so a stop-out loses a fixed % of equity
  - intrabar stop fills (checked against the bar's high/low)

run() can be restricted to an index window [lo, hi) so the same causally-computed
signals can be evaluated on train vs test ranges without recomputation.
"""

from __future__ import annotations
import math
import statistics
from typing import Dict, Optional


def run(
    data: Dict, signals: Dict, params: Dict, lo: int = 1, hi: Optional[int] = None
) -> Dict:
    close, high, low = data["close"], data["high"], data["low"]
    time_arr = data.get("time", [0] * len(close))
    n = len(close)
    hi = n if hi is None else hi
    comm = params.get("commission", 0.0005)
    slip = params.get("slippage", 0.0003)
    risk_pct = params.get("risk_pct", 1.0)
    equity0 = params.get("equity", 10000.0)
    conservative_trail = params.get("conservative_trail", True)
    max_daily_loss_pct = params.get("max_daily_loss_pct", None)
    cooldown_bars = params.get("cooldown_bars", 0)

    entry, exit_, stop_dist = signals["entry"], signals["exit"], signals["stop_dist"]
    tp_dist = signals.get("tp_dist", [None] * n)
    trail_dist = signals.get("trail_dist", [None] * n)
    equity = equity0
    pos = None
    trades = []
    curve = [equity0]

    cooldown_until = 0
    current_day_start_equity = equity0
    current_day = time_arr[0] // 86400 if len(time_arr) > 0 else 0

    for i in range(max(1, lo), hi):
        price = close[i]
        bar_day = time_arr[i] // 86400 if len(time_arr) > i else 0

        if bar_day != current_day:
            current_day = bar_day
            current_day_start_equity = equity

        # ---- manage an open position ----
        if pos is not None:
            td = pos.get("trail_dist")
            if td is not None:
                # Use i-1 high/low for conservative trail to avoid intra-bar lookahead
                ref_high = high[i - 1] if conservative_trail and i > 0 else high[i]
                ref_low = low[i - 1] if conservative_trail and i > 0 else low[i]

                if pos["dir"] == 1:
                    pos["stop"] = max(pos["stop"], ref_high - td)
                elif pos["dir"] == -1:
                    pos["stop"] = min(pos["stop"], ref_low + td)

            exitp = None
            stopped = False
            if pos["dir"] == 1 and low[i] <= pos["stop"]:
                exitp, stopped = pos["stop"], True
            elif pos["dir"] == -1 and high[i] >= pos["stop"]:
                exitp, stopped = pos["stop"], True
            elif "tp" in pos and pos["dir"] == 1 and high[i] >= pos["tp"]:
                exitp, stopped = pos["tp"], False  # Win!
            elif "tp" in pos and pos["dir"] == -1 and low[i] <= pos["tp"]:
                exitp, stopped = pos["tp"], False  # Win!
            elif exit_[i]:
                exitp = price

            if exitp is not None:
                eff = exitp * (1 - slip * pos["dir"])
                pnl = (eff - pos["entry_eff"]) * pos["qty"] * pos["dir"] - eff * pos[
                    "qty"
                ] * comm
                equity += pnl
                trades.append(
                    {
                        "dir": pos["dir"],
                        "entry": pos["entry"],
                        "stop": pos["stop"],
                        "i_in": pos["i_in"],
                        "i_out": i,
                        "exit": exitp,
                        "pnl": pnl,
                        "stopped": stopped,
                    }
                )
                if pnl <= 0 and cooldown_bars > 0:
                    cooldown_until = i + cooldown_bars
                pos = None

        daily_loss_hit = False
        if max_daily_loss_pct is not None:
            daily_drop = (
                (current_day_start_equity - equity) / current_day_start_equity * 100
            )
            if daily_drop >= max_daily_loss_pct:
                daily_loss_hit = True

        # ---- open a new position ----
        if (
            pos is None
            and entry[i] != 0
            and stop_dist[i]
            and i >= cooldown_until
            and not daily_loss_hit
        ):
            d = entry[i]
            sd = stop_dist[i]
            qty = (equity * risk_pct / 100.0) / sd
            entry_eff = price * (1 + slip * d)
            equity -= entry_eff * qty * comm  # entry commission
            pos = {
                "dir": d,
                "entry": price,
                "entry_eff": entry_eff,
                "qty": qty,
                "stop": price - sd * d,
                "i_in": i,
            }
            if tp_dist[i]:
                pos["tp"] = price + tp_dist[i] * d
            if trail_dist[i]:
                pos["trail_dist"] = trail_dist[i]
        curve.append(equity)

    return {"trades": trades, "curve": curve, "equity": equity, "equity0": equity0}


def metrics(result: Dict, bars_per_year: float) -> Dict:
    trades = result["trades"]
    curve = result["curve"]
    eq0 = result["equity0"]
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gp = sum(t["pnl"] for t in wins)
    gl = -sum(t["pnl"] for t in losses)
    n_t = len(trades)

    peak = curve[0]
    mdd = 0.0
    for v in curve:
        if v > peak:
            peak = v
        if peak > 0:
            mdd = max(mdd, (peak - v) / peak)

    rets = [
        curve[i] / curve[i - 1] - 1 for i in range(1, len(curve)) if curve[i - 1] > 0
    ]
    if len(rets) > 2 and statistics.pstdev(rets) > 0:
        sharpe = (
            statistics.mean(rets) / statistics.pstdev(rets) * math.sqrt(bars_per_year)
        )
    else:
        sharpe = 0.0

    net = result["equity"] - eq0
    if gl > 0:
        pf = gp / gl
    else:
        pf = float("inf") if gp > 0 else 0.0
    return {
        "trades": n_t,
        "win_rate": 100 * len(wins) / n_t if n_t else 0.0,
        "profit_factor": pf,
        "net": net,
        "return_pct": 100 * net / eq0,
        "max_drawdown": 100 * mdd,
        "sharpe": sharpe,
        "avg_win": gp / len(wins) if wins else 0.0,
        "avg_loss": gl / len(losses) if losses else 0.0,
        "expectancy": net / n_t if n_t else 0.0,
    }
