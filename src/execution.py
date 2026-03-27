"""
execution.py — Simulated execution handler with slippage and commissions.

WHY THIS IS CRITICAL:
  A backtester without realistic transaction costs will wildly overstate
  performance. A strategy that looks like 40% annual return before costs
  might be 5% after. Interviewers WILL ask about this.

SLIPPAGE MODEL:
  We use a simple fixed-percentage slippage on market orders.
  More realistic models include:
    - Volume-based (larger orders = more slippage): slippage ∝ order_size / avg_volume
    - Bid-ask spread model
    - Market impact model (Almgren-Chriss)

COMMISSION MODEL:
  Binance charges ~0.1% per trade (0.075% with BNB discount).
  We use 0.1% as a conservative estimate.
"""

from queue import Queue

from events import OrderEvent, FillEvent, SignalDirection


class SimulatedExecutionHandler:
    """
    Converts OrderEvents into FillEvents with realistic slippage and fees.

    Args:
        event_queue:      The shared event queue
        slippage_pct:     One-way slippage as a fraction (default 0.05% = 0.0005)
        commission_pct:   Commission as a fraction of trade value (default 0.1% = 0.001)
        last_prices:      Dict mapping symbol → last known price (updated by backtester)
    """

    def __init__(
        self,
        event_queue:    Queue,
        last_prices:    dict,
        slippage_pct:   float = 0.0005,   # 0.05% slippage
        commission_pct: float = 0.001,    # 0.10% Binance fee
    ):
        self.event_queue    = event_queue
        self.last_prices    = last_prices
        self.slippage_pct   = slippage_pct
        self.commission_pct = commission_pct

    def on_order(self, event: OrderEvent):
        """
        Simulate execution of a market order.

        For buys:  fill at close * (1 + slippage)   — we pay a little more
        For sells: fill at close * (1 - slippage)   — we receive a little less
        """
        symbol = event.symbol
        base_price = self.last_prices.get(symbol)

        if base_price is None or base_price <= 0:
            print(f"  [Execution] WARNING: No price for {symbol}, order skipped")
            return

        qty = event.quantity

        if event.direction in (SignalDirection.LONG,):
            # Buying — slippage works against us (price is higher)
            fill_price = base_price * (1 + self.slippage_pct)
        else:
            # Selling / exiting — slippage works against us (price is lower)
            fill_price = base_price * (1 - self.slippage_pct)
            qty = -abs(qty)   # ensure negative for portfolio logic

        trade_value = abs(qty) * fill_price
        commission  = trade_value * self.commission_pct

        fill = FillEvent(
            timestamp  = event.timestamp,
            symbol     = symbol,
            quantity   = abs(qty),
            fill_price = fill_price,
            commission = commission,
            direction  = event.direction,
        )

        self.event_queue.put(fill)
