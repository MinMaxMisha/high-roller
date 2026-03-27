"""
portfolio.py — Portfolio management: position sizing, P&L tracking, risk management.

RESPONSIBILITIES:
  1. Listen for SignalEvents → decide HOW MUCH to trade → emit OrderEvents
  2. Listen for FillEvents → update positions and cash
  3. Track equity curve, realized/unrealized P&L

POSITION SIZING (fixed fractional):
  We risk a fixed fraction of current equity per trade (default 95% — near-full
  deployment for a single-asset strategy). In a multi-asset system you'd size
  each position as e.g. 2% risk / ATR to equalize volatility exposure.

KEY CONCEPTS:
  - cash:     liquid USD not deployed in positions
  - equity:   cash + mark-to-market value of open positions
  - drawdown: peak equity - current equity (as %)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue
from typing import Dict, List, Optional

from events import SignalEvent, SignalDirection, OrderEvent, OrderType, FillEvent, MarketEvent


@dataclass
class Position:
    symbol:       str
    quantity:     float = 0.0
    avg_price:    float = 0.0
    realized_pnl: float = 0.0
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        if self.quantity == 0:
            return 0.0
        return (self.current_price - self.avg_price) * self.quantity

    @property
    def total_pnl(self) -> float:
        return self.realized_pnl + self.unrealized_pnl


@dataclass
class EquitySnapshot:
    timestamp:   datetime
    cash:        float
    holdings:    float   # mark-to-market value of positions
    total_equity: float
    drawdown_pct: float


class Portfolio:
    """
    Manages cash, positions, and risk for a single-asset strategy.
    """

    def __init__(
        self,
        event_queue:     Queue,
        initial_capital: float = 10_000.0,
        position_pct:    float = 0.95,    # fraction of equity to deploy per trade
        max_drawdown_pct: float = 0.20,   # halt trading if drawdown exceeds this
    ):
        self.event_queue      = event_queue
        self.initial_capital  = initial_capital
        self.position_pct     = position_pct
        self.max_drawdown_pct = max_drawdown_pct

        self.cash: float = initial_capital
        self.positions: Dict[str, Position] = {}
        self.equity_curve: List[EquitySnapshot] = []

        self._peak_equity  = initial_capital
        self._trading_halted = False

        # Trade log for analysis
        self.trade_log: List[dict] = []

    @property
    def holdings_value(self) -> float:
        return sum(p.market_value for p in self.positions.values())

    @property
    def total_equity(self) -> float:
        return self.cash + self.holdings_value

    @property
    def current_drawdown_pct(self) -> float:
        if self._peak_equity == 0:
            return 0.0
        return (self._peak_equity - self.total_equity) / self._peak_equity

    def on_market(self, event: MarketEvent):
        """Update mark-to-market prices and record equity snapshot."""
        if event.symbol in self.positions:
            self.positions[event.symbol].current_price = event.close

        equity = self.total_equity
        if equity > self._peak_equity:
            self._peak_equity = equity

        self.equity_curve.append(EquitySnapshot(
            timestamp    = event.timestamp,
            cash         = self.cash,
            holdings     = self.holdings_value,
            total_equity = equity,
            drawdown_pct = self.current_drawdown_pct,
        ))

        # Risk check: halt if max drawdown exceeded
        if self.current_drawdown_pct >= self.max_drawdown_pct and not self._trading_halted:
            self._trading_halted = True
            print(f"  [Portfolio] ⚠️  MAX DRAWDOWN REACHED ({self.current_drawdown_pct:.1%}) — trading halted")

    def on_signal(self, event: SignalEvent):
        """Convert a signal into a sized order."""
        if self._trading_halted:
            return

        if event.symbol not in self.positions:
            self.positions[event.symbol] = Position(symbol=event.symbol)

        pos = self.positions[event.symbol]

        if event.direction == SignalDirection.LONG and pos.quantity == 0:
            # Size: deploy position_pct of current equity
            capital_to_deploy = self.total_equity * self.position_pct * event.strength
            # We don't know exact fill price yet — use approximate (last close)
            # The execution handler will apply slippage
            approx_price = pos.current_price if pos.current_price > 0 else 1.0
            quantity = capital_to_deploy / approx_price

            order = OrderEvent(
                timestamp  = event.timestamp,
                symbol     = event.symbol,
                order_type = OrderType.MARKET,
                quantity   = quantity,
                direction  = SignalDirection.LONG,
            )
            self.event_queue.put(order)

        elif event.direction == SignalDirection.EXIT and pos.quantity > 0:
            order = OrderEvent(
                timestamp  = event.timestamp,
                symbol     = event.symbol,
                order_type = OrderType.MARKET,
                quantity   = -pos.quantity,   # negative = sell all
                direction  = SignalDirection.EXIT,
            )
            self.event_queue.put(order)

    def on_fill(self, event: FillEvent):
        """Update cash and positions after a fill."""
        if event.symbol not in self.positions:
            self.positions[event.symbol] = Position(symbol=event.symbol)

        pos = self.positions[event.symbol]
        cost = event.fill_price * event.quantity + event.commission

        if event.direction == SignalDirection.LONG:
            # Update average entry price
            total_qty   = pos.quantity + event.quantity
            pos.avg_price = (
                (pos.avg_price * pos.quantity + event.fill_price * event.quantity) / total_qty
                if total_qty > 0 else event.fill_price
            )
            pos.quantity += event.quantity
            self.cash    -= cost

            self.trade_log.append({
                "timestamp": event.timestamp,
                "action": "BUY",
                "symbol": event.symbol,
                "quantity": event.quantity,
                "price": event.fill_price,
                "commission": event.commission,
                "cash_after": self.cash,
            })

        elif event.direction == SignalDirection.EXIT:
            # Realized P&L = (fill_price - avg_price) * quantity
            qty_sold = abs(event.quantity)
            realized = (event.fill_price - pos.avg_price) * qty_sold - event.commission
            pos.realized_pnl += realized
            pos.quantity     -= qty_sold
            self.cash        += event.fill_price * qty_sold - event.commission

            if pos.quantity < 1e-9:   # fully closed, reset
                pos.quantity  = 0.0
                pos.avg_price = 0.0

            self.trade_log.append({
                "timestamp":   event.timestamp,
                "action":      "SELL",
                "symbol":      event.symbol,
                "quantity":    qty_sold,
                "price":       event.fill_price,
                "commission":  event.commission,
                "realized_pnl": realized,
                "cash_after":  self.cash,
            })

    def summary(self) -> dict:
        """Return a snapshot of current portfolio state."""
        return {
            "cash":         self.cash,
            "holdings":     self.holdings_value,
            "total_equity": self.total_equity,
            "drawdown_pct": self.current_drawdown_pct,
            "n_trades":     len([t for t in self.trade_log if t["action"] == "BUY"]),
        }
