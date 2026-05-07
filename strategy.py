# strategy.py -- Avellaneda-Stoikov market making strategies
# Implements AS 2008 (base) and AS 2018 enhanced with microprice +
# volatility-regime filter + momentum filter.
# Dependencies: config, order_book, math, collections

import math
from collections import deque
from typing import Optional

import config
from order_book import OrderBook


class AvellanedaStoikov:
    """
    Avellaneda-Stoikov (2008) market making strategy.

    Computes optimal bid/ask quotes using the reservation price and
    optimal spread formulas, with inventory-based skew controls.

    Formulas:
      sigma           = rolling std of mid prices (last SIGMA_WINDOW ticks)
      time_remaining  = 1 - (t - t_start) / (t_end - t_start)
      reservation_px  = mid - q * gamma * sigma^2 * T
      optimal_spread  = gamma * sigma^2 * T + (2/gamma) * ln(1 + gamma/kappa)
      bid             = reservation_px - spread/2
      ask             = reservation_px + spread/2
    """

    def __init__(self, gamma: float = config.GAMMA,
                 kappa: float = config.KAPPA,
                 order_size: float = config.ORDER_SIZE,
                 max_inventory: float = config.MAX_INVENTORY,
                 min_spread: float = config.MIN_SPREAD):
        self.gamma         = gamma
        self.kappa         = kappa
        self.order_size    = order_size
        self.max_inventory = max_inventory
        self.min_spread    = min_spread

    def _time_remaining(self, t: float, t_start: float, t_end: float) -> float:
        """
        Fraction of the trading session remaining, clamped to [0, 1].
        As this approaches 0 the strategy widens spreads to close inventory.
        """
        denom = t_end - t_start
        if denom <= 0:
            return 1.0
        remaining = 1.0 - (t - t_start) / denom
        return max(0.001, min(1.0, remaining))   # floor avoids div-by-zero

    def _reference_price(self, book: OrderBook) -> float:
        """Mid price used as the unbiased reference for this base model."""
        return book.mid_price

    def compute_quotes(self, book: OrderBook, inventory: float,
                       t: float, t_start: float, t_end: float
                       ) -> tuple[Optional[float], Optional[float]]:
        """
        Compute (bid_price, ask_price) for the current market state.

        Returns (None, None) if the book is not ready.
        One side may be None if inventory cap prevents posting.

        Args:
            book:      current OrderBook state
            inventory: current net position (units)
            t:         current timestamp (seconds)
            t_start:   session start timestamp
            t_end:     session end timestamp

        Returns:
            (bid_price, ask_price) -- either may be None
        """
        if not book.is_ready():
            return None, None

        sigma = max(book.sigma, config.SIGMA_DEFAULT)
        T     = self._time_remaining(t, t_start, t_end)
        q     = inventory
        ref   = self._reference_price(book)

        # Reservation price: skews toward flattening inventory
        reservation_px = ref - q * self.gamma * sigma**2 * T

        # Optimal spread: wider when volatile or near session end
        log_term = math.log(1.0 + self.gamma / self.kappa) if self.kappa > 0 else 0
        optimal_spread = (self.gamma * sigma**2 * T
                          + (2.0 / self.gamma) * log_term)
        optimal_spread = max(optimal_spread, self.min_spread)

        bid_price = reservation_px - optimal_spread / 2.0
        ask_price = reservation_px + optimal_spread / 2.0

        # Inventory skew: extra push when approaching cap
        skew_threshold = self.max_inventory * 0.7
        if inventory > skew_threshold:
            bid_price -= optimal_spread * 0.1
        elif inventory < -skew_threshold:
            ask_price += optimal_spread * 0.1

        # Hard cap: suppress the side that would worsen exposure
        post_bid = True
        post_ask = True
        if inventory >= self.max_inventory:
            post_bid = False   # already too long
        if inventory <= -self.max_inventory:
            post_ask = False   # already too short

        return (bid_price if post_bid else None,
                ask_price if post_ask else None)


class EnhancedAS(AvellanedaStoikov):
    """
    Enhanced strategy: AS 2008 + three improvements from AS 2018 context.

    Improvement 1 -- Microprice reference:
      Uses microprice instead of mid-price as the unbiased reference.
      Microprice accounts for order-book imbalance at the best bid/ask.

    Improvement 2 -- Volatility regime filter:
      When current sigma > VOLATILITY_REGIME_MULT x avg(recent sigma),
      widen the spread by VOLATILITY_REGIME_SPREAD to protect against
      adverse selection during fast markets.

    Improvement 3 -- Momentum filter:
      If the last MOMENTUM_WINDOW trades were all on one side, shift
      the corresponding quote slightly toward the pressure, improving
      fill rate on the more likely leg.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._sigma_history:  deque[float] = deque(
            maxlen=config.VOLATILITY_REGIME_WINDOW)
        self._recent_sides:   deque[str]   = deque(
            maxlen=config.MOMENTUM_WINDOW)

    def _reference_price(self, book: OrderBook) -> float:
        """Use microprice as the unbiased reference."""
        return book.microprice if book.microprice is not None else book.mid_price

    def record_trade_side(self, side: str) -> None:
        """Call on every trade event to track recent order flow direction."""
        self._recent_sides.append(side)

    def compute_quotes(self, book: OrderBook, inventory: float,
                       t: float, t_start: float, t_end: float
                       ) -> tuple[Optional[float], Optional[float]]:
        """
        Compute enhanced quotes with microprice, regime filter, and momentum.
        """
        bid, ask = super().compute_quotes(book, inventory, t, t_start, t_end)

        if bid is None and ask is None:
            return None, None

        sigma = max(book.sigma, config.SIGMA_DEFAULT)
        self._sigma_history.append(sigma)

        # Improvement 2: volatility regime
        if len(self._sigma_history) >= 2:
            sigma_avg = sum(self._sigma_history) / len(self._sigma_history)
            if sigma_avg > 0 and sigma > config.VOLATILITY_REGIME_MULT * sigma_avg:
                half_extra = (sigma * config.VOLATILITY_REGIME_SPREAD) / 2.0
                if bid is not None:
                    bid -= half_extra
                if ask is not None:
                    ask += half_extra

        # Improvement 3: momentum filter
        if len(self._recent_sides) == config.MOMENTUM_WINDOW:
            all_buys  = all(s == "buy"  for s in self._recent_sides)
            all_sells = all(s == "sell" for s in self._recent_sides)
            # Re-derive optimal_spread approximation for adjustment size
            T   = self._time_remaining(t, t_start, t_end)
            sp_approx = max(self.gamma * sigma**2 * T, self.min_spread)
            adj = sp_approx * config.MOMENTUM_ADJUST

            if all_buys and ask is not None:
                ask -= adj   # lean ask in since buyers are aggressive
            if all_sells and bid is not None:
                bid += adj   # lean bid in since sellers are aggressive

        return bid, ask


# -- Standalone test -----------------------------------------------------------

if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from data_loader import save_samples, load_lob

    save_samples()
    lob_df, _ = load_lob(use_sample=True)

    book      = OrderBook()
    strategy  = AvellanedaStoikov()
    t_start   = float(lob_df["timestamp"].iloc[0])
    t_end     = float(lob_df["timestamp"].iloc[-1])
    inventory = 0.0

    print("\n10 quote decisions (must satisfy bid < mid < ask):\n")
    for i, row in lob_df.head(15).iterrows():
        book.update(row.to_dict())
        if not book.is_ready():
            continue
        t   = float(row["timestamp"])
        bid, ask = strategy.compute_quotes(book, inventory, t, t_start, t_end)
        mid = book.mid_price
        valid = (bid is None or bid < mid) and (ask is None or ask > mid)
        flag  = "v" if valid else "x"
        print(
            f"  {flag}  mid={mid:.7f}  "
            f"bid={bid:.7f if bid else 'None':>12}  "
            f"ask={ask:.7f if ask else 'None':>12}  "
            f"spread={book.spread:.2e}  sigma={book.sigma:.2e}"
        )
