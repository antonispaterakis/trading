"""Walk-forward optimisation — the honesty engine of this toolkit.

Instead of optimising parameters on all history (which guarantees an
overfit, fantasy result), we repeatedly:
  1. optimise on a TRAIN window,
  2. lock those params and measure on the next, unseen TEST window,
  3. roll forward.

The stitched TEST results are an out-of-sample estimate of what the strategy
would actually have done live. If walk-forward returns are near zero or
negative while the single-shot backtest looks great, the edge was an illusion.
"""
from __future__ import annotations
import itertools
from typing import Dict, List, Optional, Tuple

from backtest import run, metrics
from strategy import generate_signals

# Kept deliberately small. Fewer knobs -> less overfitting.
PARAM_GRID = {
    "ema_len": [100, 150, 200],
    "rsi_len": [7, 14],
    "oversold": [25, 30, 35],
    "stop_atr": [2.0, 2.5, 3.0],
}


def optimize(data: Dict, base: Dict, lo: int, hi: int, bars_per_year: float,
             generate_signals_fn, param_grid: Dict,
             objective: str = "sharpe", min_trades: int = 10) -> Optional[Tuple[float, Dict, Dict]]:
    best: Optional[Tuple[float, Dict, Dict]] = None
    keys = list(param_grid)
    for combo in itertools.product(*[param_grid[k] for k in keys]):
        p = dict(base)
        for k, v in zip(keys, combo):
            p[k] = v
        if "oversold" in p:
            p["overbought"] = 100 - p["oversold"]
        sig = generate_signals_fn(data, p)
        res = run(data, sig, p, lo, hi)
        m = metrics(res, bars_per_year)
        if m["trades"] < min_trades:
            continue
        score = m[objective]
        if best is None or score > best[0]:
            best = (score, dict(p), m)
    return best


def walk_forward(data: Dict, base: Dict, generate_signals_fn, param_grid: Dict,
                 train: int = 1500, test: int = 500,
                 bars_per_year: float = 8760, objective: str = "sharpe") -> List[Dict]:
    n = len(data["close"])
    segments: List[Dict] = []
    start = 0
    while start + train + test <= n:
        tr_lo, tr_hi = start, start + train
        te_lo, te_hi = start + train, start + train + test
        best = optimize(data, base, tr_lo, tr_hi, bars_per_year, generate_signals_fn, param_grid, objective)
        if best is not None:
            _, bp, _ = best
            sig = generate_signals_fn(data, bp)
            res = run(data, sig, bp, te_lo, te_hi)
            m = metrics(res, bars_per_year)
            segments.append({"params": bp, "train_range": (tr_lo, tr_hi),
                             "test_range": (te_lo, te_hi), "metrics": m})
        start += test
    return segments
