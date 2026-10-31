# HFT Backtesting Engine — Avellaneda-Stoikov Market Making

A complete, self-contained high-frequency-trading backtesting engine implementing the
Avellaneda-Stoikov (2008) market-making strategy and its 2018 extensions.

---

## 1. Project Overview

This engine replays historical Limit Order Book (LOB) and trade-tape data tick-by-tick,
simulates limit-order placement and partial fills, and evaluates the performance of two
market-making strategies:

- **Base** — Avellaneda-Stoikov (2008): reservation price + optimal spread
- **Enhanced** — AS 2008 + microprice reference + volatility regime filter + momentum filter

Three gamma-variation experiments are also run automatically and compared in a single report.

---

## 2. Architecture

```
hft-project/
|-- backtest.py        Entry point: loads data, runs strategies, runs experiments
|-- config.py          All parameters and file paths (single source of truth)
|-- data_loader.py     Auto-detect columns, convert timestamps, save sample files
|-- order_book.py      LOB simulator: mid price, spread, microprice, volatility
|-- order_manager.py   OMS: limit order lifecycle + partial fill processing
|-- performance.py     Metrics: PnL (realized+unrealized), Sharpe, drawdown, win rate
|-- strategy.py        AvellanedaStoikov (base) and EnhancedAS (microprice+regime+momentum)
|-- report.py          Saves report.txt and PNG charts to results/
|-- model_description.md  Plain-English explanation of the AS model
|-- data/
|   |-- lob_sample.csv     First 50,000 LOB rows (committed to repo)
|   |-- trades_sample.csv  First 50,000 trade rows (committed to repo)
|-- results/
    |-- report.txt                 Full 8-section text report
    |-- pnl_over_time.png          Dual-line PnL: trending (red) vs ranging (green)
    |-- inventory_over_time.png    Inventory with long/short fill, both periods
    |-- spread_over_time.png       Bid-ask spread + market avg reference line
    |-- regime_comparison.png      Grouped bar: trending vs ranging key metrics
    |-- experiment_comparison.png  Grouped bar chart: gamma experiments
```

**Data flow:** `backtest.py` calls `data_loader` to get normalised DataFrames, then drives
an event loop that alternates between LOB ticks (fed to `OrderBook`) and trade events
(fed to `OMS.process_trade`). After every `REFRESH_N` LOB ticks the strategy recomputes
quotes and the OMS is refreshed. The `PerformanceTracker` accumulates fills and snapshots.
At completion `report.py` generates all artefacts.

---

## 3. Installation

```bash
# Requires Python 3.10+
pip install -r requirements.txt
```

Required packages: `pandas`, `numpy`, `matplotlib`

---

## 4. How to Run

**Default run (sample data, ~30 seconds):**
```bash
python backtest.py
```

**Full data run (900 MB files, ~20 minutes):**
```bash
python backtest.py --full
```

**What happens:**
1. Sample files are saved to `data/` if absent
2. Data is loaded and column mapping is printed
3. Best ranging window identified via sliding-window price-range minimisation
4. Base strategy run on full sample (trending period)
5. Base strategy run on ranging window -- side-by-side comparison printed
6. Enhanced strategy run on full sample (microprice + regime + momentum)
7. Three gamma-variation experiments (0.01, 0.10, 0.50)
8. All results printed + saved to `results/` (report.txt + 5 PNG charts)

---

## 5. Configuration Guide (config.py)

| Parameter | Default | Description |
|---|---|---|
| `USE_SAMPLE` | `True` | Use 50k-row sample files (True) or full 900MB files (False) |
| `SAMPLE_ROWS` | `50_000` | Rows to extract as sample |
| `CHUNK_SIZE` | `100_000` | Pandas read_csv chunk size for large files |
| `USE_ENHANCED` | `False` | `False` = base AS 2008, `True` = enhanced (microprice + filters) |
| `GAMMA` | `0.1` | Risk-aversion coefficient. Higher = penalises inventory more heavily |
| `KAPPA` | `1.5e9` | Order arrival rate, calibrated for this crypto asset (see note) |
| `ORDER_SIZE` | `1.0` | Size of each posted limit order in base-asset units |
| `REFRESH_N` | `10` | Cancel and requote every N LOB ticks |
| `MAX_INVENTORY` | `10` | Hard inventory cap; stops posting on the accumulating side |
| `SIGMA_WINDOW` | `100` | Rolling window for mid-price volatility estimate |
| `SIGMA_DEFAULT` | `0.001` | Fallback sigma when fewer than SIGMA_WINDOW samples available |
| `MIN_SPREAD` | `1e-8` | Floor spread to prevent negative or zero spread |
| `SNAPSHOT_INTERVAL` | `1_000` | PnL snapshot interval (LOB ticks) |
| `EXPERIMENT_GAMMAS` | `[0.01, 0.10, 0.50]` | Gamma values for the comparison experiments |

**Note on KAPPA:** The AS 2008 paper used `kappa = 1.5` for equity markets at ~$100.
For this crypto asset (price ~$0.011, tick ~1e-7), the order arrival rate in model units
is extremely high. Setting `kappa = 1.5e9` makes the log term negligible, so
`optimal_spread ≈ gamma * sigma^2 * T` — a few ticks, which is the correct economic regime.

---

## 6. Strategy Explanation

### Avellaneda-Stoikov (2008) — Base Model

The market maker posts a bid and ask symmetrically around a **reservation price**:

```
sigma           = rolling_std(mid_prices, window=SIGMA_WINDOW)
T               = 1 - (current_time - session_start) / (session_end - session_start)
reservation_px  = mid_price - q * GAMMA * sigma^2 * T
optimal_spread  = GAMMA * sigma^2 * T + (2/GAMMA) * ln(1 + GAMMA/KAPPA)
bid_price       = reservation_px - optimal_spread / 2
ask_price       = reservation_px + optimal_spread / 2
```

Where `q` is current net inventory. When long (q > 0), the reservation price drops below
mid, automatically skewing the quotes to prefer selling.

**Inventory controls:**
- If `q >= 0.7 * MAX_INVENTORY`: bid is pushed down an extra 10% of spread
- If `q >= MAX_INVENTORY`: stop posting bid entirely
- Symmetric controls on the short side

### Enhanced Strategy — Additional Improvements

1. **Microprice reference** (AS 2018): uses `microprice = (bid*ask_sz + ask*bid_sz)/(bid_sz+ask_sz)`
   as the reference instead of mid. Accounts for order-book imbalance.

2. **Volatility regime filter**: when `sigma > 2 * mean(recent_sigmas)`, the spread is
   widened by 1.5x to protect against adverse selection in fast markets.

3. **Momentum filter**: if the last 10 trades were all buys (aggressive buying pressure),
   the ask is lowered by 5% of spread to capture more fill rate. Similarly for all-sell momentum.

---

## 7. Actual Performance Results

Run on `data/lob_sample.csv` (50,000 LOB rows) and `data/trades_sample.csv` (50,000 trade rows)
covering approximately 47 minutes of a crypto order book (August 2024).

### Base Strategy (gamma=0.10)
```
Total PnL:       -0.00086429
Realized PnL:    -0.00086429
Final Inventory:  0.000000
Total Turnover:   946.0000
Total Trades:     946
Win Rate:         49.47%
Max Drawdown:     0.00086429
Sharpe Ratio:    -5.7433
```

### Enhanced Strategy (gamma=0.10)
```
Total PnL:       -0.00073857
Total Trades:     952
Win Rate:         48.53%
Max Drawdown:     0.00073857
Sharpe Ratio:    -5.9745
```

### Gamma Experiment Comparison
```
gamma=0.01  PnL=-0.000784  Sharpe=-3.19  Drawdown=0.000784  Trades=952
gamma=0.10  PnL=-0.000864  Sharpe=-5.74  Drawdown=0.000864  Trades=946
gamma=0.50  PnL=-0.000761  Sharpe=-6.11  Drawdown=0.000761  Trades=890
```

**Interpretation:** All configurations show negative PnL because the 47-minute sample
contains a significant downward price trend (~60 ticks = 0.6% decline). Market makers
accumulate long inventory by filling seller demand early in the move, then close positions
at lower prices. This is adverse selection — the fundamental risk of market making in
trending markets. The strategy's inventory controls (larger gamma reduces inventory
accumulation) partially mitigate the loss: `gamma=0.50` loses least despite fewer fills.

---

## 8. Improvement Roadmap

The following enhancements would materially improve strategy performance in production:

1. **Dynamic KAPPA calibration**: Estimate kappa from the empirical distribution of
   inter-arrival times in the trades tape. Re-calibrate every N minutes to adapt to
   changing liquidity regimes, rather than using a fixed constant.

2. **Multi-level LOB utilisation**: Currently only level-1 (best bid/ask) is used for
   quoting. Incorporating levels 2-5 allows the strategy to detect depth "cliffs"
   (where liquidity thins sharply) and widen spreads proactively before adverse moves.

3. **Trend-detection filter**: Add a fast-vs-slow moving-average signal. When the filter
   detects a trend (e.g., EMA(10) crosses EMA(50)), temporarily suspend market-making
   and switch to directional risk reduction to avoid accumulating inventory against the move.

4. **Session-aware time horizon**: Rather than using the full dataset as a single session,
   segment by calendar time (e.g., hourly windows). This improves the `T` parameter
   accuracy and allows the strategy to manage end-of-window inventory resets correctly.

5. **Fill-rate feedback loop**: Track actual fill rates vs. theoretical AS fill rates.
   If the actual rate diverges significantly (e.g., due to queue priority), adjust quote
   offsets dynamically to restore the target fill probability — a closed-loop controller
   on top of the open-loop AS model.

6. **Transaction cost modelling**: Add maker/taker fee tiers and rebate structures.
   For venues that charge takers and rebate makers, the optimal spread formula widens
   when our queue position deteriorates, and tightens when we are at the front of queue.

7. **Cross-venue hedging**: In a production setup, outstanding inventory can be hedged
   on a correlated instrument (perpetual futures, another exchange) while the primary
   market-making quotes replenish. This dramatically reduces adverse-selection exposure.
