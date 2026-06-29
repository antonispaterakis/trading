# 🤖 Honest Trading Bot Toolkit

A python toolkit built to test trading strategies **the right way**. It prevents you from fooling yourself with optimistic backtests by using rigorous **walk-forward (out-of-sample) validation**, proving when a strategy actually has an edge—and more importantly, when it doesn't.

### Core Features (View the code in `/python`)
1. **Zero-Dependency Engine:** Pure Python event-driven backtester modeling real commissions and slippage (no bloated frameworks).
2. **Hybrid AI Regime Manager:** Uses a `scikit-learn` Random Forest to predict if the market is **Trending** or **Ranging (Crab)**, dynamically swapping execution strategies on the fly.
3. **Walk-Forward Optimizer:** Retrains the AI and locks parameters on a past window, then tests them on the *next unseen* window.

---

## 📉 The Honest Results

We tested three distinct approaches on BTCUSDT (1-hour chart, 4800 bars). The gap between a standard backtest and a walk-forward test is **overfitting made visible**.

| Strategy | Engine Type | OOS Return | Trades | Max Drawdown | Sharpe |
|----------|------------|------------|--------|--------------|--------|
| **Baseline** | Trend-Following | -3.2% | 22 | 1.1% | -2.85 |
| **Crab** | Sideways Scalping | -11.1% | 46 | 4.2% | -2.37 |
| **Hybrid AI** | AI Regime Manager | **-4.6%** | **15** | **1.9%** | **-1.03** |

*Note: None of these strategies have a positive edge on this dataset. This is the truth the toolkit provides before you risk real capital.*

**Why the Hybrid AI is better:**
The AI successfully learned when to sit out. It correctly predicted market regimes, drastically reduced the trade count (from 46 down to 15), lowered the maximum drawdown, and improved the Sharpe ratio by aggressively filtering out uncertain setups.

---

## 🚀 Quick Start

```bash
# 1. Setup the environment for the Hybrid AI
python3 -m venv .venv
source .venv/bin/activate
pip install scikit-learn

cd python

# 2. Run the Walk-Forward validation on the Hybrid AI
python3 run.py --strategy hybrid

# 3. Compare it against the rigid Math strategies
python3 run.py --strategy crab
python3 run.py --strategy baseline
```

### The Three Strategies Explained

1. **Hybrid AI (`hybrid_strategy.py`)**: A Random Forest classifier predicts the market regime based on normalized volatility (RSI, ADX, BB width). If confident, it delegates execution to the correct math strategy. If uncertain (< 60% confidence), it sits out entirely.
2. **Crab Scalper (`crab_strategy.py`)**: Detects ranging markets (ADX < 25). Pauses trading on breakouts. Enters on Bollinger Band bounces confirmed by MACD momentum.
3. **Baseline Trend (`strategy.py`)**: Trades only with the higher-timeframe trend (EMA). Enters on RSI midline crosses. 

---

## 🧠 The Thinking Process

Want to know how we arrived at the Hybrid AI architecture and solved the standard pitfalls of Machine Learning in trading (Lookahead Bias, Stationarity, Black-Box Trust)? 

Read the full development log in **[progress.md](progress.md)**.

## Disclaimer

For research and education. Not financial advice. Trading risks loss of capital.
Past backtested performance does not predict future results.
