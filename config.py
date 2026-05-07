# config.py -- Central configuration for HFT backtester
# All parameters live here; no magic numbers in other modules.
# Dependencies: none (imported by all other modules)

import os

# -- Paths --------------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

RAW_LOB_FILE    = os.path.join(BASE_DIR, "lob.csv")
RAW_TRADES_FILE = os.path.join(BASE_DIR, "trades.csv")
SAMPLE_LOB_FILE    = os.path.join(DATA_DIR, "lob_sample.csv")
SAMPLE_TRADES_FILE = os.path.join(DATA_DIR, "trades_sample.csv")

# -- Data loading -------------------------------------------------------------
USE_SAMPLE  = True       # True = sample files (fast dev), False = full data
SAMPLE_ROWS = 50_000     # rows saved as sample
CHUNK_SIZE  = 100_000    # pandas chunksize for full-data streaming

# -- Strategy selection --------------------------------------------------------
USE_ENHANCED = False     # False = AS 2008 base, True = AS 2018 enhanced

# -- Avellaneda-Stoikov parameters --------------------------------------------
GAMMA         = 0.1      # risk aversion (higher -> tighter inventory management)
KAPPA         = 1.5e9    # order arrival rate, calibrated for crypto micro-priced assets.
                         # AS 2008 paper used kappa=1.5 for equity at ~$100; at price ~0.011
                         # the log term (2/gamma)*ln(1+gamma/kappa) must be ~1 tick (1e-7),
                         # requiring kappa >> gamma.  kappa=1.5e9 makes log term ~2e-9 (negligible),
                         # so spread ~ gamma*sigma^2*T, which is ~1 tick for SIGMA_DEFAULT=0.001.
ORDER_SIZE    = 1.0      # size of each posted limit order (in base asset units)
REFRESH_N     = 10       # cancel + requote every N LOB ticks
MAX_INVENTORY = 10       # hard position cap (units); enforced symmetrically
SIGMA_WINDOW  = 100      # rolling window length for mid-price volatility
SIGMA_DEFAULT = 0.001    # fallback sigma when fewer than SIGMA_WINDOW samples
MIN_SPREAD    = 1e-8     # floor spread (prevents negative or zero spread)

# -- Experiment parameters -----------------------------------------------------
EXPERIMENT_GAMMAS = [0.01, 0.10, 0.50]  # gamma values for comparison runs

# -- Market period selection ---------------------------------------------------
USE_BEST_PERIOD    = True    # True = run on best (most-ranging) window; False = full sample
BEST_PERIOD_WINDOW = 5_000   # sliding window size for find_best_period()
BEST_PERIOD_START  = 0       # auto-filled by find_best_period(); override here to fix a window
BEST_PERIOD_END    = 5_000   # auto-filled by find_best_period(); override here to fix a window

# -- Performance tracking ------------------------------------------------------
SNAPSHOT_INTERVAL = 1_000  # record a PnL snapshot every N LOB ticks

# -- Enhanced-strategy parameters ---------------------------------------------
VOLATILITY_REGIME_WINDOW   = 20    # window for avg sigma in regime filter
VOLATILITY_REGIME_MULT     = 2.0   # spread multiplier when sigma > mult x avg_sigma
VOLATILITY_REGIME_SPREAD   = 1.5   # how much to widen spread in high-vol regime
MOMENTUM_WINDOW            = 10    # trades to inspect for one-sided momentum
MOMENTUM_ADJUST            = 0.05  # fraction of spread to shift ask/bid on momentum
