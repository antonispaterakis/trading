# Honest trading bot toolkit

A small, dependency-free toolkit for **researching and validating** systematic
trading strategies the right way — with walk-forward (out-of-sample) testing
that stops you fooling yourself, and a matching TradingView strategy for live
forward-testing.

> **Read this first.** There is no "close to perfect" bot. Markets are close to
> efficient; any simple edge gets competed away. This toolkit's value is not a
> magic signal — it is the **discipline that stops you fooling yourself**:
> realistic costs, risk-based sizing, and walk-forward (out-of-sample) testing.
> Used honestly, it tells you when a strategy has *no* edge — which is the single
> most profitable thing it can do for you.

## What's inside

```
trading/
├── python/
│   ├── indicators.py      EMA, RSI, ATR, ADX, SMA, Bollinger Bands, MACD
│   ├── data.py            fetch + cache real OHLCV from Binance (no API key)
│   ├── strategy.py        baseline: trend-filtered mean reversion
│   ├── crab_strategy.py   crab market day-trader scalper
│   ├── backtest.py        event-driven engine + honest metrics
│   ├── walkforward.py     walk-forward optimisation (the core idea)
│   └── run.py             command-line entry point
├── pinescript/
│   └── mean_reversion.pine   no-repaint TradingView strategy
├── progress.md            development log + thinking process
└── README.md
```

No `pip install` needed — it runs on a stock Python 3.

## Quick start

```bash
cd python

# Walk-forward validation with the baseline strategy:
python3 run.py --strategy baseline

# Walk-forward validation with the crab market scalper:
python3 run.py --strategy crab

# Optimistic single-shot backtest (for contrast — do NOT trust it alone):
python3 run.py --strategy baseline --mode backtest

# Other markets / timeframes:
python3 run.py --symbol ETHUSDT --interval 4h

# Diversified run across uncorrelated markets:
python3 run.py --portfolio BTCUSDT,ETHUSDT,SOLUSDT
```

## Strategies

### Baseline — Trend-Filtered Mean Reversion

- Trade only with the higher-trend (longs above the EMA, shorts below).
- Enter when RSI crosses back out of an extreme (oversold / overbought).
- Take profit when RSI reverts to the midline; hard ATR stop otherwise.
- Size every trade so a stop-out loses a fixed `--risk` % of equity.

### Crab Market Day Trader

- Detect ranging (sideways) markets using ADX < threshold.
- **Pause all trading** if ADX spikes (breakout protection).
- Enter on Bollinger Band bounces confirmed by MACD momentum and Volume.
- Use tight ATR-based stop-loss and take-profit for scalping-style execution.

## How to read the output

Run walk-forward mode and check the **aggregate OOS (out-of-sample) return**.
That number is the closest estimate of what the strategy would have done live.
If it isn't clearly positive *after costs*, the strategy has no edge — do not
trade it.

## The honest path forward

1. **Keep judging on out-of-sample data**, never on the backtest peak.
2. **Diversify across uncorrelated markets** — the one real "free lunch". Use `--portfolio`.
3. **Reduce cost drag** — fewer/longer trades; costs flipped several configs from green to red.
4. **Expect Sharpe ~0.5–1 and survivable drawdowns**, not a money printer.
5. **Forward-test on TradingView** with `pinescript/mean_reversion.pine` (paper account) for weeks/months before risking a cent.

## Disclaimer

For research and education. Not financial advice. Trading risks loss of capital.
Past backtested performance does not predict future results.
