"""
strategy.py — Momentum strategy: Dual EMA crossover with volume confirmation.

STRATEGY LOGIC:
  - Compute a fast EMA (default 12 bars) and slow EMA (default 26 bars)
  - LONG signal:  fast EMA crosses ABOVE slow EMA + volume above average
  - EXIT signal:  fast EMA crosses BELOW slow EMA
  - No short selling (crypto longs only for simplicity; easy to extend)

WHY EMA CROSSOVER FOR MOMENTUM:
  EMAs are trend-following — they capture sustained directional moves.
  The volume filter reduces false signals in low-conviction moves.
  This is a classic, well-understood strategy: ideal for learning
  because its behavior is interpretable and debuggable.

EXTENDING THIS:
  - Add RSI filter (avoid buying overbought assets)
  - Add ATR-based position sizing (risk-parity)
  - Add multi-asset signals (trade BTC and ETH simultaneously)
"""

from queue import Queue
from typing import Optional

import numpy as np
import pandas as pd

from events import MarketEvent, SignalEvent, SignalDirection
from data import HistoricalDataHandler


class MomentumStrategy:
    """
    Dual EMA crossover momentum strategy with volume confirmation.

    Listens for MarketEvents, computes indicators on available history,
    and emits SignalEvents when entry/exit conditions are met.
    """

    def __init__(
        self,
        event_queue:     Queue,
        data_handler:    HistoricalDataHandler,
        symbol:          str,
        fast_period:     int   = 12,
        slow_period:     int   = 26,
        volume_ma_period: int  = 20,
        volume_multiplier: float = 1.2,   # volume must be X times its MA to confirm
    ):
        self.event_queue        = event_queue
        self.data_handler       = data_handler
        self.symbol             = symbol.upper()
        self.fast_period        = fast_period
        self.slow_period        = slow_period
        self.volume_ma_period   = volume_ma_period
        self.volume_multiplier  = volume_multiplier

        # Track current position state to avoid duplicate signals
        self._in_position = False

        # Need at least slow_period + 1 bars to compute a crossover
        self._min_bars = slow_period + 1

    def _compute_ema(self, series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    def on_market(self, event: MarketEvent):
        """
        Called for every new MarketEvent. Computes indicators and
        fires a SignalEvent if entry/exit conditions are met.
        """
        if event.symbol != self.symbol:
            return

        # Fetch history up to (not including) the current bar
        n_bars = max(self.slow_period * 3, self.volume_ma_period * 2)
        history = self.data_handler.get_history(n_bars)

        if len(history) < self._min_bars:
            return  # Not enough data yet

        closes  = history["close"]
        volumes = history["volume"]

        fast_ema = self._compute_ema(closes, self.fast_period)
        slow_ema = self._compute_ema(closes, self.slow_period)
        vol_ma   = volumes.rolling(self.volume_ma_period).mean()

        # Current and previous bar values (crossover detection)
        fast_now,  fast_prev  = fast_ema.iloc[-1],  fast_ema.iloc[-2]
        slow_now,  slow_prev  = slow_ema.iloc[-1],  slow_ema.iloc[-2]
        vol_now,   vol_ma_now = volumes.iloc[-1],   vol_ma.iloc[-1]

        # Crossover conditions
        bullish_cross = (fast_prev <= slow_prev) and (fast_now > slow_now)
        bearish_cross = (fast_prev >= slow_prev) and (fast_now < slow_now)

        # Volume confirmation: current volume must exceed its moving average
        volume_confirmed = vol_now > (vol_ma_now * self.volume_multiplier)

        signal = None

        if bullish_cross and volume_confirmed and not self._in_position:
            signal = SignalEvent(
                timestamp = event.timestamp,
                symbol    = self.symbol,
                direction = SignalDirection.LONG,
                strength  = min(1.0, vol_now / vol_ma_now / 2),  # scale by volume strength
            )
            self._in_position = True
            print(f"  [Strategy] LONG signal @ {event.timestamp}  close={event.close:.2f}  "
                  f"fast_ema={fast_now:.2f}  slow_ema={slow_now:.2f}")

        elif bearish_cross and self._in_position:
            signal = SignalEvent(
                timestamp = event.timestamp,
                symbol    = self.symbol,
                direction = SignalDirection.EXIT,
                strength  = 1.0,
            )
            self._in_position = False
            print(f"  [Strategy] EXIT signal @ {event.timestamp}  close={event.close:.2f}")

        if signal:
            self.event_queue.put(signal)
