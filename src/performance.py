"""
performance.py — Strategy performance analytics.

METRICS EXPLAINED (know these for interviews):

  Sharpe Ratio:
    (mean_return - risk_free_rate) / std_return * sqrt(annualization_factor)
    > 1.0 is acceptable, > 2.0 is good, > 3.0 is exceptional.
    Annualization: daily=252, hourly=252*24=6048, 4h=252*6=1512

  Maximum Drawdown:
    Largest peak-to-trough equity decline.
    Calmar Ratio = annual_return / max_drawdown (> 1.0 is decent)

  Win Rate:
    % of trades that were profitable. High win rate with bad risk/reward
    can still lose money overall.

  Profit Factor:
    gross_profit / gross_loss. > 1.5 is solid.

  Avg Trade:
    Average P&L per trade. Tells you if the strategy is viable after
    adding more realistic costs.
"""

import numpy as np
import pandas as pd
from typing import List

from portfolio import EquitySnapshot


ANNUALIZATION = {
    "1m":  252 * 24 * 60,
    "5m":  252 * 24 * 12,
    "15m": 252 * 24 * 4,
    "30m": 252 * 24 * 2,
    "1h":  252 * 24,
    "4h":  252 * 6,
    "1d":  252,
}


def compute_performance(
    equity_curve: List[EquitySnapshot],
    trade_log:    List[dict],
    interval:     str = "1h",
    risk_free:    float = 0.05,   # annual risk-free rate (5%)
    initial_capital: float = 10_000.0,
) -> dict:
    """
    Compute full performance report from equity curve and trade log.
    """
    if not equity_curve:
        return {}

    # ── Equity curve as Series ──────────────────────────────────────────────
    timestamps = [s.timestamp for s in equity_curve]
    equity     = pd.Series([s.total_equity for s in equity_curve], index=timestamps)
    drawdowns  = pd.Series([s.drawdown_pct for s in equity_curve], index=timestamps)

    # ── Returns ─────────────────────────────────────────────────────────────
    returns = equity.pct_change().dropna()
    ann_factor = ANNUALIZATION.get(interval, 252)
    rf_per_bar = risk_free / ann_factor

    # ── Sharpe Ratio ────────────────────────────────────────────────────────
    excess_returns = returns - rf_per_bar
    sharpe = (
        (excess_returns.mean() / excess_returns.std()) * np.sqrt(ann_factor)
        if excess_returns.std() > 0 else 0.0
    )

    # ── Max Drawdown ────────────────────────────────────────────────────────
    max_drawdown = drawdowns.max()

    # ── Total & Annualized Return ────────────────────────────────────────────
    total_return = (equity.iloc[-1] - initial_capital) / initial_capital
    n_bars = len(equity)
    years  = n_bars / ann_factor
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

    # ── Calmar Ratio ────────────────────────────────────────────────────────
    calmar = annual_return / max_drawdown if max_drawdown > 0 else np.inf

    # ── Trade-level stats ────────────────────────────────────────────────────
    sells = [t for t in trade_log if t["action"] == "SELL"]
    buys  = [t for t in trade_log if t["action"] == "BUY"]

    if sells:
        pnls       = [t["realized_pnl"] for t in sells]
        wins       = [p for p in pnls if p > 0]
        losses     = [p for p in pnls if p <= 0]
        win_rate   = len(wins) / len(pnls)
        avg_win    = np.mean(wins)   if wins   else 0.0
        avg_loss   = np.mean(losses) if losses else 0.0
        profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else np.inf
        avg_trade  = np.mean(pnls)
    else:
        win_rate = avg_win = avg_loss = profit_factor = avg_trade = 0.0

    total_commission = sum(t["commission"] for t in trade_log)

    return {
        "total_return_pct":   total_return * 100,
        "annual_return_pct":  annual_return * 100,
        "sharpe_ratio":       sharpe,
        "max_drawdown_pct":   max_drawdown * 100,
        "calmar_ratio":       calmar,
        "n_trades":           len(buys),
        "win_rate_pct":       win_rate * 100,
        "avg_win":            avg_win,
        "avg_loss":           avg_loss,
        "profit_factor":      profit_factor,
        "avg_trade_pnl":      avg_trade,
        "total_commission":   total_commission,
        "final_equity":       equity.iloc[-1],
        "initial_capital":    initial_capital,
    }


def print_report(metrics: dict):
    """Pretty-print the performance report."""
    print("\n" + "═" * 50)
    print("  BACKTEST PERFORMANCE REPORT")
    print("═" * 50)
    print(f"  Initial Capital:    ${metrics['initial_capital']:>12,.2f}")
    print(f"  Final Equity:       ${metrics['final_equity']:>12,.2f}")
    print(f"  Total Return:       {metrics['total_return_pct']:>11.2f}%")
    print(f"  Annual Return:      {metrics['annual_return_pct']:>11.2f}%")
    print("─" * 50)
    print(f"  Sharpe Ratio:       {metrics['sharpe_ratio']:>12.3f}")
    print(f"  Max Drawdown:       {metrics['max_drawdown_pct']:>11.2f}%")
    print(f"  Calmar Ratio:       {metrics['calmar_ratio']:>12.3f}")
    print("─" * 50)
    print(f"  # Trades:           {metrics['n_trades']:>12d}")
    print(f"  Win Rate:           {metrics['win_rate_pct']:>11.2f}%")
    print(f"  Avg Win:            ${metrics['avg_win']:>12,.2f}")
    print(f"  Avg Loss:           ${metrics['avg_loss']:>12,.2f}")
    print(f"  Profit Factor:      {metrics['profit_factor']:>12.3f}")
    print(f"  Avg Trade P&L:      ${metrics['avg_trade_pnl']:>12,.2f}")
    print(f"  Total Commission:   ${metrics['total_commission']:>12,.2f}")
    print("═" * 50 + "\n")
