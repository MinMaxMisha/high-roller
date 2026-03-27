"""
run.py — Entry point. Run a single backtest or a parameter sweep.

USAGE:
  python run.py                    # single backtest with default params
  python run.py --optimize         # sweep fast/slow EMA periods
  python run.py --symbol ETHUSDT   # test on ETH instead

AVOID OVERFITTING:
  When sweeping parameters, NEVER pick the best in-sample result and call
  it done. Use walk-forward validation:
    1. Train on bars 0–700 to find best params
    2. Test on bars 700–1000 (out-of-sample)
  If out-of-sample is much worse, your params are overfit.
"""

import argparse
import itertools

from backtester import Backtester


def run_single(symbol="BTCUSDT", interval="4h", limit=1000,
               fast=12, slow=26, capital=10_000.0):
    bt = Backtester(
        symbol          = symbol,
        interval        = interval,
        limit           = limit,
        initial_capital = capital,
        fast_period     = fast,
        slow_period     = slow,
    )
    return bt.run()


def run_optimization(symbol="BTCUSDT", interval="4h", limit=1000, capital=10_000.0):
    """
    Simple grid search over EMA periods.
    WARNING: This is IN-SAMPLE optimization — use walk-forward for real research.
    """
    fast_periods = [8, 12, 21]
    slow_periods = [26, 50, 100]

    results = []
    for fast, slow in itertools.product(fast_periods, slow_periods):
        if fast >= slow:
            continue
        print(f"\n{'─'*40}")
        print(f"Testing fast={fast}, slow={slow}")
        bt = Backtester(
            symbol=symbol, interval=interval, limit=limit,
            initial_capital=capital, fast_period=fast, slow_period=slow,
        )
        metrics = bt.run()
        results.append({"fast": fast, "slow": slow, **metrics})

    print("\n" + "═" * 60)
    print("  OPTIMIZATION RESULTS (sorted by Sharpe)")
    print("═" * 60)
    results.sort(key=lambda x: x.get("sharpe_ratio", -999), reverse=True)
    for r in results:
        print(f"  fast={r['fast']:3d}  slow={r['slow']:3d} | "
              f"Sharpe={r.get('sharpe_ratio', 0):6.3f} | "
              f"Return={r.get('total_return_pct', 0):7.2f}% | "
              f"MaxDD={r.get('max_drawdown_pct', 0):6.2f}% | "
              f"Trades={r.get('n_trades', 0):3d}")
    print("═" * 60)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crypto Momentum Backtester")
    parser.add_argument("--symbol",   default="BTCUSDT",  help="Trading pair (e.g. ETHUSDT)")
    parser.add_argument("--interval", default="4h",       help="Bar size: 1h, 4h, 1d")
    parser.add_argument("--limit",    default=1000, type=int, help="Number of bars")
    parser.add_argument("--capital",  default=10000, type=float, help="Initial capital USD")
    parser.add_argument("--fast",     default=12,   type=int, help="Fast EMA period")
    parser.add_argument("--slow",     default=26,   type=int, help="Slow EMA period")
    parser.add_argument("--optimize", action="store_true", help="Run parameter sweep")
    args = parser.parse_args()

    if args.optimize:
        run_optimization(args.symbol, args.interval, args.limit, args.capital)
    else:
        run_single(args.symbol, args.interval, args.limit,
                   args.fast, args.slow, args.capital)
