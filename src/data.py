"""
data.py — Historical data handler.

Fetches free OHLCV data from the Binance public API (no API key required).
Feeds bars one-at-a-time into the event queue, simulating live data arrival.

WHY THIS MATTERS: A common backtesting mistake is feeding all data at once,
which can cause look-ahead bias. Here we iterate bar-by-bar, exactly as
a live system would receive ticks.
"""

import time
import requests
import pandas as pd
from datetime import datetime
from queue import Queue
from typing import Generator

from events import MarketEvent


BINANCE_URL = "https://api.binance.com/api/v3/klines"

INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "4h": "4h", "1d": "1d",
}


def fetch_ohlcv(symbol: str, interval: str = "1h", limit: int = 1000) -> pd.DataFrame:
    """
    Fetch historical OHLCV bars from Binance public API.

    Args:
        symbol:   Trading pair, e.g. 'BTCUSDT' or 'ETHUSDT'
        interval: Bar size — '1m', '5m', '1h', '4h', '1d'
        limit:    Number of bars (max 1000 per request)

    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    """
    if interval not in INTERVAL_MAP:
        raise ValueError(f"Invalid interval '{interval}'. Choose from: {list(INTERVAL_MAP)}")

    params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}

    print(f"[DataHandler] Fetching {limit} x {interval} bars for {symbol}...")
    response = requests.get(BINANCE_URL, params=params, timeout=10)
    response.raise_for_status()
    raw = response.json()

    df = pd.DataFrame(raw, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    print(f"[DataHandler] Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df


class HistoricalDataHandler:
    """
    Streams historical bars into the event queue one-at-a-time.

    This is the key architectural decision: by emitting one MarketEvent
    per bar, every other component (Strategy, Portfolio, Execution) only
    ever 'sees' data up to the current bar — no look-ahead possible.
    """

    def __init__(self, event_queue: Queue, symbol: str, interval: str = "1h", limit: int = 1000):
        self.event_queue = event_queue
        self.symbol      = symbol.upper()
        self.interval    = interval
        self.data        = fetch_ohlcv(symbol, interval, limit)
        self._index      = 0

    def __len__(self):
        return len(self.data)

    @property
    def current_index(self):
        return self._index

    def has_next(self) -> bool:
        return self._index < len(self.data)

    def stream_next(self) -> bool:
        """
        Emit the next bar as a MarketEvent into the queue.
        Returns True if a bar was emitted, False if data is exhausted.
        """
        if not self.has_next():
            return False

        row = self.data.iloc[self._index]
        event = MarketEvent(
            timestamp = self.data.index[self._index].to_pydatetime(),
            symbol    = self.symbol,
            open      = row["open"],
            high      = row["high"],
            low       = row["low"],
            close     = row["close"],
            volume    = row["volume"],
        )
        self.event_queue.put(event)
        self._index += 1
        return True

    def get_history(self, n_bars: int) -> pd.DataFrame:
        """
        Return the last n_bars of data UP TO (not including) the current bar.
        Used by the strategy to compute indicators without look-ahead.
        """
        end = self._index
        start = max(0, end - n_bars)
        return self.data.iloc[start:end]
