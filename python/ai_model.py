"""AI regime classifier — Random Forest wrapper.

Trains a lightweight classifier to predict whether the upcoming market
regime is Trending (1) or Crab/Ranging (0).  Designed to be retrained
on every walk-forward segment so it adapts to evolving markets.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Tuple

try:
    from sklearn.ensemble import RandomForestClassifier
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def _check_sklearn():
    if not HAS_SKLEARN:
        raise ImportError(
            "scikit-learn is required for the hybrid AI strategy.\n"
            "Install it with:  pip install scikit-learn"
        )


def train_model(features: List[List[float]], labels: List[int],
                valid_mask: List[bool],
                max_depth: int = 6,
                n_estimators: int = 200) -> object:
    """Train a Random Forest on the valid subset of features/labels.

    Returns the fitted sklearn model.
    """
    _check_sklearn()

    X = [f for f, v in zip(features, valid_mask) if v]
    y = [l for l, v in zip(labels, valid_mask) if v]

    if len(X) < 30:
        return None  # Not enough training samples

    # Check that both classes are present
    unique = set(y)
    if len(unique) < 2:
        return None  # Can't train a classifier with only one class

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=42,
        n_jobs=1,
    )
    clf.fit(X, y)
    return clf


def predict_regime(model, features: List[List[float]]) -> List[Tuple[int, float]]:
    """Predict regime for every bar.

    Returns a list of (prediction, confidence) tuples.
    prediction: 0 = Crab, 1 = Trend
    confidence: probability of the predicted class (0.5–1.0)
    """
    _check_sklearn()

    if model is None:
        # Fallback: predict Crab with zero confidence (will trigger sit-out)
        return [(0, 0.5)] * len(features)

    probas = model.predict_proba(features)
    results = []
    for p in probas:
        pred = 0 if p[0] >= p[1] else 1
        conf = max(p[0], p[1])
        results.append((pred, conf))
    return results


def log_importance(model, feature_names: List[str]) -> str:
    """Return a formatted string of feature importances, ranked."""
    if model is None:
        return "  (no model trained)"

    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances),
                   key=lambda x: x[1], reverse=True)
    lines = []
    for name, imp in pairs:
        bar = "█" * int(imp * 40)
        lines.append(f"    {name:<20s} {imp:.3f}  {bar}")
    return "\n".join(lines)
