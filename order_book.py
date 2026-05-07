# order_book.py -- Limit Order Book simulator
# Maintains best bid/ask + up to 5 levels; computes mid, microprice, spread,
# book imbalance, and rolling volatility from mid-price history.
# Dependencies: config, numpy, collections

import math
from collections import deque

import numpy as np

import config


class OrderBook:
    """
    Stateful LOB simulator updated tick-by-tick from LOB snapshot rows.

    Attributes updated on every call to update():
      bid_price, bid_size, ask_price, ask_size   (level-1 best)
      mid_price, spread, microprice, book_imbalance
      sigma  (rolling std of mid_price over last SIGMA_WINDOW ticks)
    """

    def __init__(self, sigma_window: int = config.SIGMA_WINDOW,
                 sigma_default: float = config.SIGMA_DEFAULT,
                 min_spread: float = config.MIN_SPREAD):
        self.sigma_window  = sigma_window
        self.sigma_default = sigma_default
        self.min_spread    = min_spread

        # Level-1 state
        self.bid_price: float | None = None
        self.bid_size:  float | None = None
        self.ask_price: float | None = None
        self.ask_size:  float | None = None

        # Derived
        self.mid_price:      float | None = None
        self.spread:         float | None = None
        self.microprice:     float | None = None
        self.book_imbalance: float | None = None

        # Up to 5 levels stored as lists of (price, size)
        self.bids: list[tuple[float, float]] = []
        self.asks: list[tuple[float, float]] = []

        # Rolling history for sigma
        self._mid_history: deque[float] = deque(maxlen=sigma_window)
        self.sigma: float = sigma_default

        self.tick_count: int = 0

    # -- Core update ----------------------------------------------------------

    def update(self, row: dict) -> None:
        """
        Ingest one LOB snapshot row (as a dict keyed by standardised names).

        Expected keys: bid_price_1, bid_size_1, ask_price_1, ask_size_1,
        and optionally bid_price_N / bid_size_N / ask_price_N / ask_size_N for N=2..5.
        """
        # Level-1
        bp1 = float(row.get("bid_price_1", 0) or 0)
        bs1 = float(row.get("bid_size_1",  0) or 0)
        ap1 = float(row.get("ask_price_1", 0) or 0)
        as1 = float(row.get("ask_size_1",  0) or 0)

        if bp1 <= 0 or ap1 <= 0:
            # Degenerate row -- skip update
            return

        self.bid_price = bp1
        self.bid_size  = bs1
        self.ask_price = ap1
        self.ask_size  = as1

        # Multi-level storage (levels 1-5)
        self.bids = []
        self.asks = []
        for lvl in range(1, 6):
            bp = float(row.get(f"bid_price_{lvl}", 0) or 0)
            bs = float(row.get(f"bid_size_{lvl}",  0) or 0)
            ap = float(row.get(f"ask_price_{lvl}", 0) or 0)
            as_ = float(row.get(f"ask_size_{lvl}", 0) or 0)
            if bp > 0:
                self.bids.append((bp, bs))
            if ap > 0:
                self.asks.append((ap, as_))

        # Derived quantities
        self.mid_price = (bp1 + ap1) / 2.0
        raw_spread = ap1 - bp1
        self.spread = max(abs(raw_spread), self.min_spread)

        # Microprice: weights mid toward thinner side
        total_size = bs1 + as1
        if total_size > 0:
            self.microprice = (bp1 * as1 + ap1 * bs1) / total_size
        else:
            self.microprice = self.mid_price

        # Book imbalance in [-1, 1]
        if total_size > 0:
            self.book_imbalance = (bs1 - as1) / total_size
        else:
            self.book_imbalance = 0.0

        # Rolling volatility
        self._mid_history.append(self.mid_price)
        if len(self._mid_history) >= 2:
            self.sigma = float(np.std(list(self._mid_history), ddof=1))
            if math.isnan(self.sigma) or self.sigma <= 0:
                self.sigma = self.sigma_default
        else:
            self.sigma = self.sigma_default

        self.tick_count += 1

    # -- Convenience accessors -------------------------------------------------

    def is_ready(self) -> bool:
        """True once at least one valid tick has been processed."""
        return self.mid_price is not None

    def state_dict(self) -> dict:
        """Return current state as a plain dict (for logging / testing)."""
        return {
            "tick":            self.tick_count,
            "bid_price":       self.bid_price,
            "ask_price":       self.ask_price,
            "bid_size":        self.bid_size,
            "ask_size":        self.ask_size,
            "mid_price":       self.mid_price,
            "spread":          self.spread,
            "microprice":      self.microprice,
            "book_imbalance":  self.book_imbalance,
            "sigma":           self.sigma,
        }


# -- Standalone test -----------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import save_samples, load_lob

    save_samples()
    lob_df, _ = load_lob(use_sample=True)

    book = OrderBook()
    print("\n5 live book states (mid + microprice):\n")
    for i, row in lob_df.head(5).iterrows():
        book.update(row.to_dict())
        s = book.state_dict()
        print(
            f"  tick={s['tick']:>3}  "
            f"bid={s['bid_price']:.7f}  "
            f"ask={s['ask_price']:.7f}  "
            f"mid={s['mid_price']:.7f}  "
            f"micro={s['microprice']:.7f}  "
            f"spread={s['spread']:.2e}  "
            f"imbalance={s['book_imbalance']:+.3f}"
        )
