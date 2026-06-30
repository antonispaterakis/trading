# HonestEdge: AI Crypto Backtester

A highly rigorous, "honest" algorithmic trading engine designed to eliminate lookahead bias and mine Alternative Data for structural market inefficiencies.

## Results Up Front
Traditional generic indicators (RSI, ADX, MACD) are losing their edge in modern crypto markets. When stripped of lookahead bias and subjected to strict Walk-Forward Out-Of-Sample testing, basic indicator strategies regress to flat or negative expectancy.

However, our AI Random Forest regime classifier has uncovered that **structural leverage imbalances are highly predictive**. 

**Top Predictive Features (15m Timeframe):**
| Rank | Feature | Importance | Description |
|---|---|---|---|
| 1 | `atr_pct_slope` | 0.160 | Volatility acceleration |
| 2 | `ls_ratio` | 0.153 | Top Trader Long/Short sentiment ratio |
| 3 | `atr_pct` | 0.106 | Normalized realized volatility |
| 4 | `ma_100_dist` | 0.095 | Macro distance from 100-period MA |
| 5 | `rsi` | 0.093 | Standard momentum |

*The Long/Short Ratio dominates traditional oscillators, proving that knowing the market's leverage position is more valuable than lagging price data.*

## Features

- **Pessimistic Fill Model:** Trailing stops and fills are strictly calculated using `i-1` metrics, entirely eradicating intra-bar lookahead bias.
- **Walk-Forward Validation:** Evaluates strategies across rolling out-of-sample segments to prevent curve-fitting and over-optimization.
- **Alt Data Integration:** Strictly causal `as_of` temporal joins for Binance Futures data (Open Interest, Long/Short Ratio, Funding Rates).
- **Hybrid AI Controller:** Uses a Random Forest classifier to predict structural market regimes (Trending vs Crabbing) and delegates execution to specialized sub-strategies.

## Documentation
- Read the [Development Progress & Deep Dives](progress.md) to understand the engineering journey, how we hunted down lookahead bias, and our Phase 2 pivot into high-frequency Alternative Data.

## Quickstart

```bash
# Set up environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run an honest walk-forward test with the AI hybrid strategy on the 15m chart
python python/run.py --strategy hybrid --bars 2800 --interval 15m
```
