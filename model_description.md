# Avellaneda-Stoikov Market Making Strategy -- Model Description

---

## 1. What Is Market Making?

A market maker is a trader who continuously quotes both a **bid** (willingness to buy)
and an **ask** (willingness to sell), earning the spread when both sides fill. Unlike
directional traders who bet on price movements, market makers profit from *providing
liquidity*: they absorb orders from other participants and aim to close each round-trip
with a small gain.

The core challenge is **inventory risk**. If the market drifts directionally before
an open position can be closed, the accumulated unrealized loss can easily exceed the
spread revenue earned on individual trades. A market maker must therefore quote
*asymmetrically* when holding a skewed position -- leaning the ask side lower when
long, or the bid higher when short -- to attract the trades that reduce inventory.

---

## 2. The Avellaneda-Stoikov 2008 Model

Avellaneda and Stoikov (2008) solved the market maker's stochastic control problem
analytically and derived two closed-form expressions:

### Reservation Price

The price at which the maker is indifferent between holding current inventory and
trading one more unit:

```
r = mid_price - q * gamma * sigma^2 * T
```

- `mid_price` = (best_bid + best_ask) / 2
- `q`         = net inventory (positive = long, negative = short)
- `gamma`     = risk-aversion coefficient (0 = risk neutral; higher = tighter control)
- `sigma`     = rolling standard deviation of mid-price changes
- `T`         = time remaining in session as a fraction (1.0 at open, 0.0 at close)

**Intuition:** When long (q > 0), `r` falls below mid. The maker then posts both bid
and ask below mid -- making the ask more aggressive to attract sells and reduce inventory.

### Optimal Spread

The minimum spread required to remain profitable against adverse selection:

```
delta = gamma * sigma^2 * T  +  (2/gamma) * ln(1 + gamma/kappa)
```

- `kappa`        = order-arrival rate (higher liquid markets allow tighter spreads)
- First term     = volatility-and-time penalty (shrinks as session ends)
- Second term    = adverse-selection premium (price of being potentially uninformed)

Final posted quotes:

```
bid = r - delta/2
ask = r + delta/2
```

### Inventory Controls

This implementation adds hard position caps:

| Trigger               | Action                                          |
|-----------------------|-------------------------------------------------|
| abs(inventory) > 70%  | Widen spread by 2x on the accumulating side     |
| abs(inventory) = 100% | Suppress new orders on the inventory-extending side |

---

## 3. Microprice Enhancement (2018)

The standard mid-price treats both sides of the book as equally informative. The
**microprice** weights each side by the *opposing* quantity:

```
microprice = (bid_price * ask_size + ask_price * bid_size) / (bid_size + ask_size)
```

When ask_size is thin (few offers), microprice tilts toward the ask -- signalling
upward pressure. When bid_size is thin, it tilts toward the bid.

The `EnhancedAS` class uses microprice as the reservation-price reference and adds:

1. **Volatility regime filter** -- when the current sigma exceeds `VOLATILITY_REGIME_MULT`
   times the recent average, the spread is widened by `VOLATILITY_REGIME_SPREAD`. This
   reduces fill probability in turbulent conditions where adverse selection is highest.

2. **Momentum skew** -- tracks the last `MOMENTUM_WINDOW` market trades. When more than
   70% are buys, the ask is raised fractionally (and vice versa for sells) to avoid
   being picked off on the wrong side of a short-lived directional burst.

---

## 4. When the Strategy Works Best

The AS model is designed for **stationary, mean-reverting markets** where:

- The mid-price oscillates around a stable level (low drift)
- Both sides of the book are regularly replenished (healthy liquidity)
- No persistent directional information asymmetry exists

In practice this corresponds to:

- Crypto pairs during off-peak consolidation
- Equity futures during low-volatility pre-open periods
- FX spot during off-session overlap hours

The `find_best_period()` function in this backtester identifies the most sideways
window in the dataset by minimising `(max_mid - min_mid) / mean_mid * 100`, allowing
a direct comparison between the strategy's behaviour in its ideal regime (ranging)
versus its typical regime (trending).

---

## 5. Known Limitations

| Limitation | Detail |
|---|---|
| **No trend filter** | The model has no mechanism to detect persistent drift and suspend quoting. The inventory skew partially compensates but cannot prevent accumulation in a fast trend. |
| **Fixed kappa** | Real order-arrival rates fluctuate throughout the day. A fixed kappa means spreads are too wide in liquid periods (missed revenue) and too tight in illiquid ones (excess adverse selection). |
| **No queue position** | This simulation assumes fills occur whenever a market trade crosses the posted price. In reality, earlier-posted orders ahead in the queue fill first; the maker may not fill at all despite a crossing trade. |
| **No transaction costs** | Exchange fees (maker rebate or taker fee) and market impact are not modelled. For thin-margin strategies these can flip a profitable backtest to loss. |
| **Single asset** | The model quotes one instrument. A multi-asset market maker can hedge residual inventory in correlated instruments, dramatically reducing directional risk. |
| **Sigma estimation lag** | Rolling sigma over `SIGMA_WINDOW` ticks reacts slowly to volatility spikes, causing the spread to remain too tight in the first few ticks of a volatile episode. |

---

## 6. What Real HFT Firms Do Differently

This implementation faithfully reproduces the AS 2008 theoretical model plus modest
2018 enhancements. Production HFT market-making systems differ substantially:

| Dimension | This Model | Production HFT |
|---|---|---|
| **Latency** | Tick-by-tick simulation (~ms) | Co-located hardware, FPGA, <1 microsecond |
| **Quote updates** | Every `REFRESH_N` ticks | Every market data event, sub-microsecond |
| **Kappa** | Fixed constant | Real-time MLE on streaming trade arrival process |
| **Sigma** | Rolling std of mid | Realised vol + implied vol surface blending |
| **Reference price** | Microprice | Proprietary fair-value model (multi-level, multi-exchange) |
| **Inventory hedging** | Spread skew only | Delta hedging in futures/ETFs + cross-venue netting |
| **Adverse selection** | None beyond spread | Toxicity metrics (VPIN, order flow imbalance, short-range alpha) |
| **Regime detection** | None (EnhancedAS: simple vol multiplier) | ML classifier on 100+ microstructure features |
| **Risk limits** | Single MAX_INVENTORY cap | Multi-dimensional: per-symbol, per-desk, gross exposure, VaR, CVaR |
