# Development Progress & Architecture Deep Dives

This document acts as a development log tracking the evolution of the trading engine, specifically addressing critical architectural flaws and how they were resolved.

## Phase 1: Exposing and Eliminating Lookahead Bias

### The Flaw
The original iteration of the bot produced wildly optimistic, 45-degree upward equity curves. In algorithmic trading, if a generic strategy (using basic RSI, MACD, and ADX) prints perfect returns, it is almost a 100% guarantee of data leakage.

Upon red-teaming the code, a massive intra-bar lookahead bias was discovered in the `backtest.py` logic. The trailing stop was using the *current* bar's high and low to move the stop-loss to safety *before* checking if the stop was actually hit. Mathematically, the bot was peeking into the future of the bar, moving its stop to the absolute optimal price, and avoiding drawdowns that would have destroyed it in real live trading.

### The Resolution
1. **Pessimistic Fill Model:** The trailing stop logic was re-engineered to use the *previous* bar's extremes (`high[i-1]`, `low[i-1]`), absolutely guaranteeing the bot can never use future knowledge within the same bar.
2. **Backtester Math Validation:** A hand-crafted 10-bar synthetic dataset was written in `test_backtest.py` to assert PnL math, commission logic, and the new trailing stop execution are perfectly accurate.
3. **Portfolio-Level Risk Controls:** Added `max_daily_loss_pct` and `cooldown_bars` to halt trading after severe drawdowns.

## Phase 2: Shorter Timeframes & Alpha Discovery

With the lookahead bias eliminated, the "honest" baseline engine revealed that traditional indicators (RSI, MACD) possessed a flat to slightly negative expectancy. To hunt for real inefficiencies, the engine was transitioned to shorter timeframes (15-minute execution) and Alternative Data features were engineered.

### High-Frequency Alt Data Infrastructure
We integrated the Binance Futures API to fetch:
- **Funding Rates:** To track leverage costs.
- **Open Interest (OI):** To track momentum and liquidity build-up.
- **Long/Short Ratio:** To track retail sentiment imbalances.

> **Crucial Rule:** To prevent data leakage, all alternative data is strictly joined using an `as_of` temporal join (`align_alt_data`). We only use alternative data whose timestamp is strictly less than or equal to the current kline open time.

### Feature Engineering
The AI Random Forest was updated to consume:
- `oi_change_pct_12`: Open Interest momentum over a 3-hour intra-day horizon.
- `ma_100_dist` & `ma_200_dist`: Macro moving average distances to anchor the 15m noise.

### Empirical Results
When running the updated hybrid model over a 30-day (2,800 bar) out-of-sample period on the 15m chart, the Random Forest revealed:

```text
    atr_pct_slope        0.160  ██████
    ls_ratio             0.153  ██████
    atr_pct              0.106  ████
    ma_100_dist          0.095  ███
    rsi                  0.093  ███
```

The **Long/Short Ratio** (`ls_ratio`) was identified as the #2 most important feature for predicting volatility/regime changes on the 15m timeframe, far surpassing traditional indicators. This confirms the hypothesis that traditional oscillators are losing their edge to structural data (leverage imbalances) in the modern crypto market. 

We now possess a mathematically honest, AI-driven backtesting environment actively mining alternative data for edge.
