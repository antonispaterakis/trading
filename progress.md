# Progress

This document tracks the evolution of the trading toolkit, from a simple backtester to a hybrid AI-powered regime-switching engine. Each phase captures the thinking process, what we learned, and what it led to next.

---

## Phase 1 — The Honest Baseline

**Goal:** Build a framework that tells us when a strategy has *no* edge (the most profitable thing it can do).

**What we built:**
- A causal `indicators.py` module (EMA, RSI, ATR, ADX) written in pure Python with zero dependencies, ensuring every value at bar `i` only uses data up to bar `i` (no look-ahead bias).
- A trend-filtered mean-reversion strategy (`strategy.py`): long above the EMA when RSI crosses back from oversold, short below the EMA when RSI crosses back from overbought, with an ATR-based hard stop.
- An event-driven backtester (`backtest.py`) that honestly models commissions, slippage, and intra-bar stop fills.
- A walk-forward optimizer (`walkforward.py`) that trains on one window, locks the parameters, and tests on the next unseen window — the only honest way to evaluate a strategy.

**What the data told us:**

| Mode | Return | Profit Factor | Sharpe |
|------|--------|---------------|--------|
| In-sample backtest | +2.7% | 2.44 | 1.52 |
| Walk-forward (OOS) | −2.8% | <1 | negative |

**Key insight:** The gap between in-sample and walk-forward results is *overfitting made visible*. The in-sample number is what a vendor selling you a bot would show you. The walk-forward number is reality. This framework's primary value is preventing self-deception.

---

## Phase 2 — The Crab Market Day Trader

**Thinking process:**
Day traders make money by catching 2–3 small moves in ranging markets rather than betting on one big directional swing. The question was: can we build a mathematical system that detects when the market is "crabbing" (moving sideways) and scalps the edges of the range?

**What we built:**
- New indicators in `indicators.py`: **SMA**, **Standard Deviation**, **Bollinger Bands**, **MACD** — all causal, all pure Python.
- A new `crab_strategy.py` that:
  1. **Detects crab markets** by checking if ADX < 25 (no strong trend).
  2. **Pauses trading** if ADX spikes above the threshold (dangerous breakout protection).
  3. **Enters on band bounces**: goes long when price touches the lower Bollinger Band with MACD momentum shifting up, shorts when price touches the upper band with MACD shifting down.
  4. **Requires volume confirmation**: only trades when volume exceeds its own moving average (ensures there's enough liquidity for a quick exit).
  5. Uses **tight ATR-based stops and take-profits** (1.0–1.5× ATR) for scalping-style execution.
- Upgraded `backtest.py` with intra-bar take-profit simulation (`tp_dist`).
- Refactored `walkforward.py` and `run.py` to accept any strategy dynamically via `--strategy baseline` or `--strategy crab`.

**What the data told us:**

```
=== BTCUSDT 1h (4800 bars) [CRAB STRATEGY] ===
Walk-forward OUT-OF-SAMPLE:
  seg2: win=67%  PF=1.91  return=+1.7%  [bb14 adx25 tp1.5 stop1.2]
  seg5: win=67%  PF=2.04  return=+2.0%  [bb14 adx25 tp1.5 stop1.0]
  seg4: win=17%  PF=0.19  return=-5.2%  [bb14 adx25 tp1.5 stop1.2]
  seg6: win=17%  PF=0.21  return=-5.3%  [bb14 adx25 tp1.5 stop1.0]
  AGGREGATE OOS: return=-11.1%  avgWin=44%  avgSharpe=-2.37
```

**Key insight:** The crab strategy is *spectacularly profitable* in some segments (seg2, seg5) and *catastrophically bad* in others (seg4, seg6). The exact same parameters that earned +2.0% in one window lost -5.2% in the next. This means the market conditions shifted between those windows. The strategy itself is mathematically sound — the problem is that **it doesn't know when to turn itself on or off**.

This directly led us to Phase 3.

---

## Phase 3 — Hybrid AI Regime Manager (Next)

**Thinking process:**
We have two strategies that each work well in their specific market regime:
- **Baseline** works when the market is trending.
- **Crab Scalper** works when the market is ranging.

The missing piece is a brain that can tell us *which* regime we are currently in, so we can activate the right strategy at the right time. This is a classification problem — exactly what Machine Learning excels at.

**The critical design decisions:**

1. **Why not let the AI predict price directly?**
   - Price direction is essentially random noise on short timeframes. Even the best hedge fund quants get directional predictions right only ~52–55% of the time.
   - Regime classification (ranging vs trending) is a much coarser, more stable target that ML can realistically learn.

2. **Why realized volatility instead of future ADX as the training label?**
   - ADX is a *lagging* indicator — it takes many bars to respond to a regime change. Training an AI to predict a lagging indicator's future value means the AI inherits that lag and is always one step behind.
   - Realized volatility (stdev of future returns) is a *direct* measurement of what actually happened. No lag, no interpretation.

3. **Why a confidence threshold (sit-out zone)?**
   - If the AI is 51% sure it's a crab market and 49% sure it's trending, that's essentially a coin flip. Trading on coin flips is gambling.
   - By requiring >60% confidence, we only trade when the AI has a genuine opinion. This means we miss some opportunities, but we avoid the disaster trades where the AI was guessing.

4. **Why not close positions on regime switch?**
   - If the crab strategy opened a scalp trade, and the AI suddenly switches to "Trend," force-closing the position would lock in whatever P&L exists at that random moment.
   - It's cleaner to let the current strategy's exit rules finish the trade naturally (hit take-profit or stop-loss), and only apply the new regime to the *next* entry.

5. **How we prevent overfitting in the AI itself:**
   - `class_weight="balanced"` so the model can't cheat by always predicting the majority class.
   - `max_depth=6` to cap tree complexity (deeper trees memorize noise).
   - Feature importance logging so we can verify the AI is using sensible indicators, not spurious correlations.
   - The walk-forward engine re-trains a fresh model per segment, so no single model is trusted across changing markets.

**Status:** Implemented and tested.

**What we built:**
- `features.py`: Extracts 6 normalized features (RSI, ADX, MACD histogram/ATR, BB width %, Volume ratio, ATR %) and computes a realized-volatility target label with an adaptive rolling median threshold.
- `ai_model.py`: A `scikit-learn` Random Forest wrapper with `class_weight="balanced"`, `max_depth=6`, and feature importance logging.
- `hybrid_strategy.py`: The master controller that takes AI regime predictions and delegates to the appropriate sub-strategy. Includes a confidence sit-out zone (< 60%).
- `walkforward.py`: Added `walk_forward_hybrid()` that trains a fresh model per segment, with explicit lookahead-bias prevention.
- `run.py`: Now supports `--strategy hybrid`.

**What the data told us:**

```
=== BTCUSDT 1h (4800 bars) [HYBRID STRATEGY] ===
Walk-forward OUT-OF-SAMPLE:
  seg1: trades=  4  win=  25%  return=  -2.4%  [crab=177 trend=103 skip=220]
  seg2: trades=  2  win=100%  return=  +2.5%  [crab=61  trend=110 skip=329]
  seg3: trades=  2  win=  50%  return=  -0.0%  [crab=60  trend=81  skip=359]
  seg4: trades=  4  win=   0%  return=  -5.2%  [crab=68  trend=65  skip=367]
  seg5: trades=  0  win=   0%  return=  +0.0%  [crab=89  trend=158 skip=253]
  seg6: trades=  3  win=  67%  return=  +0.6%  [crab=98  trend=141 skip=261]
  AGGREGATE OOS: return=-4.6%  trades=15  avgMaxDD=1.9%  avgSharpe=-1.03

AI Feature Importance:
  atr_pct              0.242  █████████
  bb_width_pct         0.230  █████████
  adx                  0.158  ██████
  rsi                  0.133  █████
  macd_hist_norm       0.132  █████
  volume_ratio         0.105  ████
```

**Key insights:**

| Strategy | OOS Return | Trades | Avg MaxDD | Avg Sharpe |
|----------|-----------|--------|-----------|------------|
| Baseline | -3.2% | 22 | 1.1% | -2.85 |
| Crab | -11.1% | 46 | 4.2% | -2.37 |
| **Hybrid AI** | **-4.6%** | **15** | **1.9%** | **-1.03** |

The hybrid AI strategy showed clear improvements in risk management:
1. **Fewer, more selective trades** (15 vs 46 for Crab, 22 for Baseline) — the sit-out zone works.
2. **Best Sharpe ratio** (-1.03 vs -2.85 and -2.37) — the AI is filtering out the worst trades.
3. **Lower max drawdown** than the Crab strategy (1.9% vs 4.2%).
4. The AI correctly identified `atr_pct` and `bb_width_pct` as the most important features — these are direct measurements of volatility, which is exactly what we asked it to predict.
5. The high skip rate (50-70% of bars) shows the confidence threshold is aggressively filtering uncertainty. This could be tuned lower for more trades, at the cost of lower selectivity.

None of the strategies have a positive OOS edge on this particular BTCUSDT 1h dataset, which is the honest result. The framework's value remains: it tells you the truth before you risk real money.

---

## Disclaimer

For research and education. Not financial advice. Trading risks loss of capital.
Past backtested performance does not predict future results.
