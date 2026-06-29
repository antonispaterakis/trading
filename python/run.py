"""Command-line entry point.

Examples
--------
  python3 run.py                                  # walk-forward on BTCUSDT 1h (hybrid AI)
  python3 run.py --strategy baseline              # walk-forward baseline strategy
  python3 run.py --strategy crab                  # walk-forward crab scalper
  python3 run.py --strategy hybrid                # walk-forward hybrid AI regime manager
  python3 run.py --mode backtest                  # single in-sample backtest
  python3 run.py --symbol ETHUSDT --interval 4h   # different market/timeframe
  python3 run.py --portfolio BTCUSDT,ETHUSDT,SOLUSDT   # diversified walk-forward
"""

from __future__ import annotations
import argparse
import importlib
import statistics

from data import fetch_klines
from backtest import run, metrics
from walkforward import walk_forward, walk_forward_hybrid

BARS_PER_YEAR = {"15m": 35040, "1h": 8760, "2h": 4380, "4h": 2190, "1d": 365}


def _pf(m):
    return "inf" if m["profit_factor"] == float("inf") else f"{m['profit_factor']:.2f}"


def fmt(m):
    return (
        f"trades={m['trades']:>3}  win={m['win_rate']:>4.0f}%  PF={_pf(m):>5}  "
        f"return={m['return_pct']:>+6.1f}%  maxDD={m['max_drawdown']:>4.1f}%  "
        f"Sharpe={m['sharpe']:>5.2f}"
    )


def base_params(risk, defaults):
    p = dict(defaults)
    p.update(risk_pct=risk, commission=0.0005, slippage=0.0003, equity=10000.0)
    return p


def run_one(symbol, interval, bars, risk, mode, strategy_name):
    data = fetch_klines(symbol, interval, bars)
    bpy = BARS_PER_YEAR.get(interval, 8760)
    print(
        f"\n=== {symbol} {interval}  ({len(data['close'])} bars)  [{strategy_name.upper()} STRATEGY] ==="
    )

    # ---- HYBRID AI STRATEGY ----
    if strategy_name == "hybrid":
        from hybrid_strategy import DEFAULTS as HYBRID_DEFAULTS

        p = base_params(risk, HYBRID_DEFAULTS)

        if mode == "backtest":
            # For backtest mode, train on first 80% and test on all
            from features import extract_features, compute_labels
            from ai_model import train_model, predict_regime
            from hybrid_strategy import generate_signals as hybrid_gen

            feat_rows, feat_valid = extract_features(data, p)
            labels, label_valid = compute_labels(data, p.get("horizon", 20))
            horizon = p.get("horizon", 20)
            n = len(data["close"])
            split = int(n * 0.8)

            train_mask = [
                feat_valid[i] and label_valid[i] and i + horizon < split
                for i in range(split)
            ]
            model = train_model(feat_rows[:split], labels[:split], train_mask)
            predictions = predict_regime(model, feat_rows)
            sig = hybrid_gen(data, p, regime_predictions=predictions)
            res = run(data, sig, p)
            print("In-sample backtest (optimistic, do NOT trust this alone):")
            print("  " + fmt(metrics(res, bpy)))
            return None

        # Walk-forward mode
        segs = walk_forward_hybrid(
            data,
            p,
            bars_per_year=bpy,
            horizon=p.get("horizon", 20),
            min_confidence=p.get("min_confidence", 0.60),
        )
        if not segs:
            print("  Not enough data for walk-forward (need > train+test bars).")
            return None

        print("Walk-forward OUT-OF-SAMPLE segments (the honest numbers):")
        comp = 1.0
        for i, s in enumerate(segs):
            m = s["metrics"]
            comp *= 1 + m["return_pct"] / 100
            rd = s.get("regime_dist", {})
            regime_str = (
                f"   [crab={rd.get('crab', 0)} trend={rd.get('trend', 0)} "
                f"skip={rd.get('skipped', 0)}]"
            )
            print(f"  seg{i + 1}: " + fmt(m) + regime_str)

        # Print feature importance from the last segment
        if segs and segs[-1].get("feature_importance"):
            print("\n  AI Feature Importance (last segment):")
            print(segs[-1]["feature_importance"])

        agg = {
            "compounded_return": (comp - 1) * 100,
            "total_trades": sum(s["metrics"]["trades"] for s in segs),
            "avg_win": statistics.mean(s["metrics"]["win_rate"] for s in segs)
            if segs
            else 0.0,
            "avg_dd": statistics.mean(s["metrics"]["max_drawdown"] for s in segs)
            if segs
            else 0.0,
            "avg_sharpe": statistics.mean(s["metrics"]["sharpe"] for s in segs)
            if segs
            else 0.0,
        }
        print("  " + "-" * 60)
        print(
            f"  AGGREGATE OOS: return={agg['compounded_return']:+.1f}%  "
            f"trades={agg['total_trades']}  avgWin={agg['avg_win']:.0f}%  "
            f"avgMaxDD={agg['avg_dd']:.1f}%  avgSharpe={agg['avg_sharpe']:.2f}"
        )
        return agg

    # ---- BASELINE / CRAB STRATEGY ----
    if strategy_name == "baseline":
        mod = importlib.import_module("strategy")
    else:
        mod = importlib.import_module("crab_strategy")

    DEFAULTS = getattr(mod, "DEFAULTS")
    generate_signals = getattr(mod, "generate_signals")
    PARAM_GRID = getattr(mod, "PARAM_GRID")

    p = base_params(risk, DEFAULTS)

    if mode == "backtest":
        sig = generate_signals(data, p)
        res = run(data, sig, p)
        print("In-sample backtest (optimistic, do NOT trust this alone):")
        print("  " + fmt(metrics(res, bpy)))
        return None

    segs = walk_forward(
        data,
        p,
        generate_signals_fn=generate_signals,
        param_grid=PARAM_GRID,
        bars_per_year=bpy,
    )
    if not segs:
        print("  Not enough data for walk-forward (need > train+test bars).")
        return None
    print("Walk-forward OUT-OF-SAMPLE segments (the honest numbers):")
    comp = 1.0
    for i, s in enumerate(segs):
        m = s["metrics"]
        comp *= 1 + m["return_pct"] / 100
        bp = s["params"]
        if strategy_name == "baseline":
            param_str = f"   [ema{bp.get('ema_len')} rsi{bp.get('rsi_len')} os{bp.get('oversold')} stop{bp.get('stop_atr')}]"
        else:
            param_str = f"   [bb{bp.get('bb_len')} adx{bp.get('adx_threshold')} tp{bp.get('tp_atr')} stop{bp.get('stop_atr')}]"

        print(f"  seg{i + 1}: " + fmt(m) + param_str)

    agg = {
        "compounded_return": (comp - 1) * 100,
        "total_trades": sum(s["metrics"]["trades"] for s in segs),
        "avg_win": statistics.mean(s["metrics"]["win_rate"] for s in segs)
        if segs
        else 0.0,
        "avg_dd": statistics.mean(s["metrics"]["max_drawdown"] for s in segs)
        if segs
        else 0.0,
        "avg_sharpe": statistics.mean(s["metrics"]["sharpe"] for s in segs)
        if segs
        else 0.0,
    }
    print("  " + "-" * 60)
    print(
        f"  AGGREGATE OOS: return={agg['compounded_return']:+.1f}%  "
        f"trades={agg['total_trades']}  avgWin={agg['avg_win']:.0f}%  "
        f"avgMaxDD={agg['avg_dd']:.1f}%  avgSharpe={agg['avg_sharpe']:.2f}"
    )
    return agg


def main():
    ap = argparse.ArgumentParser(
        description="Honest, walk-forward-validated trading backtester."
    )
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--bars", type=int, default=4800)
    ap.add_argument(
        "--risk", type=float, default=1.0, help="percent of equity risked per trade"
    )
    ap.add_argument(
        "--mode", choices=["backtest", "walkforward"], default="walkforward"
    )
    ap.add_argument(
        "--strategy", choices=["baseline", "crab", "hybrid"], default="hybrid"
    )
    ap.add_argument(
        "--portfolio", default="", help="comma-separated symbols for a diversified run"
    )
    args = ap.parse_args()

    if args.portfolio:
        symbols = [s.strip() for s in args.portfolio.split(",") if s.strip()]
        aggs = []
        for sym in symbols:
            a = run_one(
                sym, args.interval, args.bars, args.risk, "walkforward", args.strategy
            )
            if a:
                aggs.append(a["compounded_return"])
        if aggs:
            print(
                "\n=== PORTFOLIO (equal-weight average of per-market OOS returns) ==="
            )
            print(
                f"  mean OOS return across {len(aggs)} markets: {statistics.mean(aggs):+.1f}%"
            )
    else:
        run_one(
            args.symbol, args.interval, args.bars, args.risk, args.mode, args.strategy
        )


if __name__ == "__main__":
    main()
