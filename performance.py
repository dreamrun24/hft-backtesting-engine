# performance.py -- Performance metrics tracker
# Computes PnL (realized + unrealized), inventory, turnover, Sharpe, drawdown,
# win rate, and maintains time-series snapshots for charting.
# Dependencies: config, order_manager (Fill), numpy, collections

import math
from collections import deque

import numpy as np

import config
from order_manager import Fill


class PerformanceTracker:
    """
    Tracks all trading performance metrics in real-time during a backtest.

    PnL calculation:
      Realized PnL  -- FIFO matching of buy/sell fill pairs
      Unrealized PnL -- net_inventory x (current_mid - avg_entry_price)
      Total PnL      -- Realized + Unrealized
    """

    def __init__(self):
        # Running totals
        self.realized_pnl:    float = 0.0
        self.inventory:       float = 0.0   # net position (positive = long)
        self.turnover:        float = 0.0   # cumulative traded volume
        self.trade_count:     int   = 0     # number of fill events

        # FIFO queues for realized PnL
        self._buy_queue:  deque[tuple[float, float]] = deque()  # (price, size)
        self._sell_queue: deque[tuple[float, float]] = deque()  # (price, size)

        # Round-trip win tracking
        self._round_trips_total:      int = 0
        self._round_trips_profitable: int = 0

        # Weighted-average entry price of the CURRENT open position (WAP cost basis).
        # Updated incrementally on each fill; reset to 0 when position is flat.
        self._avg_entry: float = 0.0

        # Snapshots for time-series charts
        self.timestamps:           list[float] = []
        self.total_pnl_series:     list[float] = []
        self.realized_pnl_series:  list[float] = []
        self.inventory_series:     list[float] = []
        self.spread_series:        list[float] = []
        self.trade_price_series:   list[float] = []
        self.trade_ts_series:      list[float] = []

        # Order-flow / quoted spread tracking (populated by backtest loop)
        self.total_orders_placed:   int         = 0
        self.quoted_spread_series:  list[float] = []

        # For drawdown calculation
        self._peak_pnl: float = 0.0
        self.max_drawdown: float = 0.0

        self._current_mid:  float = 0.0
        self._snapshot_ctr: int   = 0

    # -- Fill ingestion --------------------------------------------------------

    def record_fill(self, fill: Fill) -> None:
        """
        Update all metrics after a fill event.

        Args:
            fill: a Fill object from the OMS
        """
        self.trade_count += 1
        self.turnover    += fill.fill_size

        if fill.side == "buy":
            old_inv = self.inventory
            new_inv = old_inv + fill.fill_size
            if old_inv >= 0:
                # Extending or opening long: WAP update
                if new_inv > 0:
                    self._avg_entry = (self._avg_entry * old_inv + fill.fill_price * fill.fill_size) / new_inv
            else:
                # Partially or fully covering a short position
                if new_inv > 0:
                    # Crossed from short to long — new long entry at fill price
                    self._avg_entry = fill.fill_price
                elif abs(new_inv) < 1e-12:
                    self._avg_entry = 0.0   # flat
                # else: still short, avg_entry unchanged (not needed for long PnL)
            self.inventory = new_inv
            self._buy_queue.append((fill.fill_price, fill.fill_size))
        else:
            old_inv = self.inventory
            new_inv = old_inv - fill.fill_size
            if old_inv <= 0:
                # Extending or opening short: WAP update
                abs_old = abs(old_inv)
                abs_new = abs(new_inv)
                if abs_new > 0:
                    self._avg_entry = (self._avg_entry * abs_old + fill.fill_price * fill.fill_size) / abs_new
            else:
                # Partially or fully covering a long position
                if new_inv < 0:
                    # Crossed from long to short — new short entry at fill price
                    self._avg_entry = fill.fill_price
                elif abs(new_inv) < 1e-12:
                    self._avg_entry = 0.0   # flat
                # else: still long, avg_entry unchanged
            self.inventory = new_inv
            self._sell_queue.append((fill.fill_price, fill.fill_size))

        self._settle_fifo()
        self.trade_price_series.append(fill.fill_price)
        self.trade_ts_series.append(fill.timestamp)

    def _settle_fifo(self) -> None:
        """
        Match buy and sell fills FIFO to compute realized PnL.
        For each matched pair records whether it was profitable.
        """
        while self._buy_queue and self._sell_queue:
            buy_px,  buy_sz  = self._buy_queue[0]
            sell_px, sell_sz = self._sell_queue[0]

            matched = min(buy_sz, sell_sz)
            pnl_this = (sell_px - buy_px) * matched
            self.realized_pnl += pnl_this

            self._round_trips_total += 1
            if pnl_this > 0:
                self._round_trips_profitable += 1

            # Consume from queues
            remaining_buy  = buy_sz  - matched
            remaining_sell = sell_sz - matched
            self._buy_queue.popleft()
            self._sell_queue.popleft()
            if remaining_buy  > 1e-12:
                self._buy_queue.appendleft((buy_px, remaining_buy))
            if remaining_sell > 1e-12:
                self._sell_queue.appendleft((sell_px, remaining_sell))

    # -- Mid-price update ------------------------------------------------------

    def update_mid(self, mid_price: float) -> None:
        """Call on every LOB tick so unrealized PnL stays current."""
        self._current_mid = mid_price

    # -- Snapshot --------------------------------------------------------------

    def snapshot(self, timestamp: float, spread: float) -> None:
        """
        Record a PnL + inventory snapshot (called every SNAPSHOT_INTERVAL ticks).

        Args:
            timestamp: current LOB timestamp (seconds)
            spread:    current best bid-ask spread
        """
        total = self._total_pnl()

        self.timestamps.append(timestamp)
        self.total_pnl_series.append(total)
        self.realized_pnl_series.append(self.realized_pnl)
        self.inventory_series.append(self.inventory)
        self.spread_series.append(spread)

        # Drawdown
        if total > self._peak_pnl:
            self._peak_pnl = total
        dd = self._peak_pnl - total
        if dd > self.max_drawdown:
            self.max_drawdown = dd

        self._snapshot_ctr += 1

    # -- Metrics ---------------------------------------------------------------

    def _unrealized_pnl(self) -> float:
        """Mark-to-market unrealized PnL on open inventory."""
        if abs(self.inventory) < 1e-12 or self._current_mid == 0 or self._avg_entry == 0:
            return 0.0
        if self.inventory > 0:
            return self.inventory * (self._current_mid - self._avg_entry)
        else:
            return abs(self.inventory) * (self._avg_entry - self._current_mid)

    def _total_pnl(self) -> float:
        return self.realized_pnl + self._unrealized_pnl()

    def win_rate(self) -> float:
        """Fraction of round-trip pairs that were profitable."""
        if self._round_trips_total == 0:
            return 0.0
        return self._round_trips_profitable / self._round_trips_total

    def sharpe_ratio(self) -> float:
        """
        Sharpe ratio from PnL-snapshot returns, annualised with sqrt(252).
        Uses pnl_changes = diff(total_pnl_series) as the return series.
        Returns 0.0 if std == 0 or insufficient data.
        """
        if len(self.total_pnl_series) < 2:
            return 0.0
        pnl_changes = np.diff(self.total_pnl_series)
        std = float(np.std(pnl_changes, ddof=1))
        if std == 0 or math.isnan(std):
            return 0.0
        return float(np.mean(pnl_changes)) / std * math.sqrt(252)

    def avg_quoted_spread(self) -> float:
        """Mean spread quoted by the strategy (ask_px - bid_px per requote cycle)."""
        if not self.quoted_spread_series:
            return 0.0
        return float(np.mean(self.quoted_spread_series))

    def fill_rate(self) -> float:
        """
        Fill rate = total fills / total orders placed * 100%.
        Returns 0.0 if no orders were placed.
        """
        if self.total_orders_placed == 0:
            return 0.0
        return self.trade_count / self.total_orders_placed * 100.0

    def get_metrics(self) -> dict:
        """
        Return all performance metrics as a flat dict.

        Keys:
          total_pnl, realized_pnl, unrealized_pnl, inventory,
          turnover, trade_count, win_rate, max_drawdown, sharpe_ratio,
          avg_quoted_spread, fill_rate
        """
        return {
            "total_pnl":          round(self._total_pnl(),        8),
            "realized_pnl":       round(self.realized_pnl,        8),
            "unrealized_pnl":     round(self._unrealized_pnl(),   8),
            "inventory":          round(self.inventory,            6),
            "turnover":           round(self.turnover,             4),
            "trade_count":        self.trade_count,
            "win_rate":           round(self.win_rate(),           4),
            "max_drawdown":       round(self.max_drawdown,         8),
            "sharpe_ratio":       round(self.sharpe_ratio(),       4),
            "avg_quoted_spread":  round(self.avg_quoted_spread(),  10),
            "fill_rate":          round(self.fill_rate(),          4),
        }


# -- Standalone test -----------------------------------------------------------

if __name__ == "__main__":
    import random
    from order_manager import Fill

    print("=== Performance tracker test (20 synthetic fills) ===\n")
    tracker = PerformanceTracker()
    mid = 100.0
    tracker.update_mid(mid)

    fills_raw = []
    for i in range(20):
        side  = "buy" if i % 2 == 0 else "sell"
        px    = mid + random.uniform(-0.5, 0.5)
        sz    = random.uniform(0.5, 2.0)
        ts    = float(i)
        fill  = Fill(timestamp=ts, order_id=f"O{i:03d}", side=side,
                     fill_price=px, fill_size=sz, fill_type="full")
        tracker.record_fill(fill)
        tracker.snapshot(timestamp=ts, spread=0.1)
        fills_raw.append(fill)

    metrics = tracker.get_metrics()
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v}")
