# order_manager.py -- Order Management System (OMS) with partial fills
# Implements exact execution assumption: buy fills when trade_price <= bid_price,
# sell fills when trade_price >= ask_price. Partial fills supported.
# Dependencies: config, dataclasses, uuid

import uuid
from dataclasses import dataclass, field
from typing import Callable

import config


# -- Order data structure ------------------------------------------------------

@dataclass
class Order:
    """Represents a single limit order with lifecycle tracking."""
    order_id:       str
    side:           str    # 'buy' or 'sell'
    price:          float
    size:           float
    filled_size:    float = 0.0
    remaining_size: float = field(init=False)
    status:         str   = "open"   # open | partially_filled | filled | cancelled
    created_at:     float = 0.0
    updated_at:     float = 0.0

    def __post_init__(self):
        self.remaining_size = self.size


@dataclass
class Fill:
    """Records one fill event (full or partial)."""
    timestamp:  float
    order_id:   str
    side:       str
    fill_price: float
    fill_size:  float
    fill_type:  str    # 'full' or 'partial'


# -- OMS -----------------------------------------------------------------------

class OMS:
    """
    Order Management System.

    Maintains a dict of open orders and processes fills against incoming
    trade events using the specified execution assumption.
    """

    def __init__(self, on_fill: Callable[[Fill], None] | None = None):
        """
        Args:
            on_fill: optional callback invoked for every fill event.
        """
        self.orders:    dict[str, Order] = {}
        self.fill_log:  list[Fill]       = []
        self._on_fill   = on_fill
        self._order_seq = 0

    # -- Order lifecycle -------------------------------------------------------

    def place_order(self, side: str, price: float, size: float,
                    timestamp: float = 0.0) -> str:
        """
        Place a new limit order.

        Args:
            side:      'buy' or 'sell'
            price:     limit price
            size:      order size
            timestamp: event timestamp (seconds)

        Returns:
            order_id string
        """
        self._order_seq += 1
        oid = f"{side[0].upper()}{self._order_seq:06d}"
        order = Order(
            order_id=oid, side=side, price=price, size=size,
            created_at=timestamp, updated_at=timestamp,
        )
        self.orders[oid] = order
        return oid

    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel a specific order by ID.

        Returns True if the order existed and was cancellable.
        """
        order = self.orders.get(order_id)
        if order is None:
            return False
        if order.status in ("filled", "cancelled"):
            return False
        order.status = "cancelled"
        del self.orders[order_id]
        return True

    def cancel_all_orders(self) -> int:
        """Cancel all open orders. Returns count cancelled."""
        ids = list(self.orders.keys())
        for oid in ids:
            self.cancel_order(oid)
        return len(ids)

    def cancel_side(self, side: str) -> int:
        """Cancel all open orders on a given side. Returns count cancelled."""
        ids = [oid for oid, o in self.orders.items() if o.side == side]
        for oid in ids:
            self.cancel_order(oid)
        return len(ids)

    # -- Fill processing -------------------------------------------------------

    def process_trade(self, trade_price: float, trade_size: float,
                      timestamp: float = 0.0) -> list[Fill]:
        """
        Check all open orders against an incoming trade event and apply fills.

        Execution assumption (from spec):
          - Buy  order fills if trade_price <= buy_order_price
          - Sell order fills if trade_price >= sell_order_price

        Partial fills:
          - If trade_size < order.remaining_size: partially fill, keep open
          - If trade_size >= order.remaining_size: fully fill, close order

        Args:
            trade_price: price of the incoming market trade
            trade_size:  size of the incoming market trade
            timestamp:   event timestamp

        Returns:
            List of Fill objects generated this tick.
        """
        fills_this_tick: list[Fill] = []
        remaining_trade = trade_size

        # Iterate over a snapshot of order IDs to allow safe removal
        for oid in list(self.orders.keys()):
            if remaining_trade <= 0:
                break

            order = self.orders.get(oid)
            if order is None:
                continue

            # Execution assumption check
            can_fill = (
                (order.side == "buy"  and trade_price <= order.price) or
                (order.side == "sell" and trade_price >= order.price)
            )
            if not can_fill:
                continue

            fill_size = min(remaining_trade, order.remaining_size)
            fill_type = "partial" if fill_size < order.remaining_size else "full"

            # Apply fill
            order.filled_size    += fill_size
            order.remaining_size -= fill_size
            order.updated_at      = timestamp
            remaining_trade      -= fill_size

            if order.remaining_size <= 1e-12:
                order.status = "filled"
                del self.orders[oid]
            else:
                order.status = "partially_filled"

            fill = Fill(
                timestamp=timestamp,
                order_id=oid,
                side=order.side,
                fill_price=order.price,
                fill_size=fill_size,
                fill_type=fill_type,
            )
            self.fill_log.append(fill)
            fills_this_tick.append(fill)

            if self._on_fill:
                self._on_fill(fill)

        return fills_this_tick

    # -- Queries ---------------------------------------------------------------

    def open_orders(self) -> list[Order]:
        """Return list of all currently open orders."""
        return list(self.orders.values())

    def open_count(self, side: str | None = None) -> int:
        """Count open orders, optionally filtered by side."""
        if side is None:
            return len(self.orders)
        return sum(1 for o in self.orders.values() if o.side == side)

    def total_fills(self) -> int:
        """Total number of fill events recorded."""
        return len(self.fill_log)


# -- Standalone test -----------------------------------------------------------

if __name__ == "__main__":
    print("=== OMS unit test ===\n")
    oms = OMS()

    # Place a buy @ 100 and a sell @ 102
    bid_id = oms.place_order("buy",  100.0, 5.0, timestamp=1.0)
    ask_id = oms.place_order("sell", 102.0, 5.0, timestamp=1.0)
    print(f"Placed: {bid_id} buy@100, {ask_id} sell@102")

    # Trade at 99 -> should fill buy (99 <= 100), no fill for sell
    fills = oms.process_trade(trade_price=99.0, trade_size=2.0, timestamp=2.0)
    for f in fills:
        print(f"  FILL [{f.fill_type}] {f.side} {f.fill_size} @ {f.fill_price}")

    # Trade at 103 -> should fill sell (103 >= 102), no fill for remaining buy
    fills = oms.process_trade(trade_price=103.0, trade_size=10.0, timestamp=3.0)
    for f in fills:
        print(f"  FILL [{f.fill_type}] {f.side} {f.fill_size} @ {f.fill_price}")

    print(f"\nTotal fill events: {oms.total_fills()}")
    print(f"Open orders remaining: {oms.open_count()}")
    print("\nFull fill log:")
    for f in oms.fill_log:
        print(f"  t={f.timestamp:.1f}  {f.order_id:10s}  {f.side:4s}  "
              f"size={f.fill_size:.1f}  px={f.fill_price:.1f}  [{f.fill_type}]")
