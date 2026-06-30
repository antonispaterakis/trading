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

# Kept deliberately small. Fewer knobs -> less overfitting.
PARAM_GRID = {
    "ema_len": [100, 150, 200],
    "rsi_len": [7, 14],
    "oversold": [25, 30, 35],
    "stop_atr": [2.0, 2.5, 3.0],
}


def optimize(
    data: Dict,
    base: Dict,
    lo: int,
    hi: int,
    bars_per_year: float,
    generate_signals_fn,
    param_grid: Dict,
    objective: str = "sharpe",
    min_trades: int = 10,
) -> Optional[Tuple[float, Dict, Dict]]:
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


def walk_forward(
    data: Dict,
    base: Dict,
    generate_signals_fn,
    param_grid: Dict,
    train: int = 1000,
    test: int = 300,
    bars_per_year: float = 8760,
    objective: str = "sharpe",
) -> List[Dict]:
    n = len(data["close"])
    segments: List[Dict] = []
    start = 0
    while start + train + test <= n:
        tr_lo, tr_hi = start, start + train
        te_lo, te_hi = start + train, start + train + test
        best = optimize(
            data,
            base,
            tr_lo,
            tr_hi,
            bars_per_year,
            generate_signals_fn,
            param_grid,
            objective,
        )
        if best is not None:
            _, bp, _ = best
            sig = generate_signals_fn(data, bp)
            res = run(data, sig, bp, te_lo, te_hi)
            m = metrics(res, bars_per_year)
            segments.append(
                {
                    "params": bp,
                    "train_range": (tr_lo, tr_hi),
                    "test_range": (te_lo, te_hi),
                    "metrics": m,
                }
            )
        start += test
    return segments


def walk_forward_hybrid(
    data: Dict,
    base: Dict,
    train: int = 1000,
    test: int = 300,
    bars_per_year: float = 35040,  # 15m
    horizon: int = 8,
    min_confidence: float = 0.60,
) -> List[Dict]:
    """Walk-forward loop for the hybrid AI strategy.

    Instead of optimising static parameters, each segment:
      1. Extracts features + labels on the TRAIN window.
      2. Trains a fresh Random Forest on that data.
      3. Uses the trained model to predict regimes on the TEST window.
      4. Generates hybrid signals and runs the backtest on the test window.
    """
    from features import extract_features, compute_labels, FEATURE_NAMES
    from ai_model import train_model, predict_regime, log_importance
    from hybrid_strategy import generate_signals as hybrid_gen
    from hybrid_strategy import DEFAULTS as HYBRID_DEFAULTS

    n = len(data["close"])
    segments: List[Dict] = []
    start = 0

    # Pre-compute features and labels for the entire dataset
    feat_rows, feat_valid = extract_features(data, base)
    all_labels, label_valid = compute_labels(data, horizon)

    while start + train + test <= n:
        tr_lo, tr_hi = start, start + train
        te_lo, te_hi = start + train, start + train + test

        # Build training mask: feature must be valid AND label must be valid
        # Labels near the end of the train window bleed into test → truncate
        train_mask = []
        for i in range(tr_lo, tr_hi):
            valid = (
                feat_valid[i] and label_valid[i] and i + horizon < tr_hi
            )  # prevent label leakage
            train_mask.append(valid)

        train_features = feat_rows[tr_lo:tr_hi]
        train_labels = all_labels[tr_lo:tr_hi]

        # Train a fresh model on this segment
        model = train_model(train_features, train_labels, train_mask)

        # Predict regimes for the full dataset (we'll only use test range)
        predictions = predict_regime(model, feat_rows)

        # Generate hybrid signals using the AI predictions
        p = dict(HYBRID_DEFAULTS)
        p["min_confidence"] = min_confidence
        sig = hybrid_gen(data, p, regime_predictions=predictions)

        # Run backtest on the test window only
        res = run(data, sig, p, te_lo, te_hi)
        m = metrics(res, bars_per_year)

        # Count regime distribution in test window
        n_crab = sum(
            1
            for i in range(te_lo, te_hi)
            if predictions[i][0] == 0 and predictions[i][1] >= min_confidence
        )
        n_trend = sum(
            1
            for i in range(te_lo, te_hi)
            if predictions[i][0] == 1 and predictions[i][1] >= min_confidence
        )
        n_skip = sum(
            1 for i in range(te_lo, te_hi) if predictions[i][1] < min_confidence
        )

        importance_str = log_importance(model, FEATURE_NAMES)

        segments.append(
            {
                "params": p,
                "train_range": (tr_lo, tr_hi),
                "test_range": (te_lo, te_hi),
                "metrics": m,
                "regime_dist": {"crab": n_crab, "trend": n_trend, "skipped": n_skip},
                "feature_importance": importance_str,
            }
        )
        start += test

    return segments
