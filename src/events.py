"""
events.py — Core event types for the event-driven backtester.

In an event-driven system, components communicate only through events.
This decouples strategy logic from execution logic, mirroring real production systems.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class EventType(Enum):
    MARKET = "MARKET"   # New price data arrived
    SIGNAL = "SIGNAL"   # Strategy generated a trade signal
    ORDER  = "ORDER"    # Portfolio decided to place an order
    FILL   = "FILL"     # Order was executed (simulated)


class SignalDirection(Enum):
    LONG  = "LONG"
    SHORT = "SHORT"
    EXIT  = "EXIT"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT  = "LIMIT"


@dataclass
class MarketEvent:
    """Fired whenever new OHLCV bar data is available."""
    type: EventType = field(default=EventType.MARKET, init=False)
    timestamp: datetime = None
    symbol: str = ""
    open:   float = 0.0
    high:   float = 0.0
    low:    float = 0.0
    close:  float = 0.0
    volume: float = 0.0


@dataclass
class SignalEvent:
    """Fired by the Strategy when it detects a trade opportunity."""
    type: EventType = field(default=EventType.SIGNAL, init=False)
    timestamp: datetime = None
    symbol: str = ""
    direction: SignalDirection = SignalDirection.LONG
    strength: float = 1.0   # 0.0–1.0, used for position sizing


@dataclass
class OrderEvent:
    """Fired by the Portfolio after deciding how to act on a signal."""
    type: EventType = field(default=EventType.ORDER, init=False)
    timestamp: datetime = None
    symbol: str = ""
    order_type: OrderType = OrderType.MARKET
    quantity: float = 0.0          # positive = buy, negative = sell
    direction: SignalDirection = SignalDirection.LONG


@dataclass
class FillEvent:
    """Fired by the ExecutionHandler after simulating order execution."""
    type: EventType = field(default=EventType.FILL, init=False)
    timestamp: datetime = None
    symbol: str = ""
    quantity: float = 0.0          # actual filled quantity
    fill_price: float = 0.0        # price after slippage
    commission: float = 0.0        # trading fees
    direction: SignalDirection = SignalDirection.LONG
