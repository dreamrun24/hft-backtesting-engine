# data_loader.py -- Load, validate, and normalise LOB + trade data
# Auto-detects column names, timestamp units, and bid/ask structure.
# Dependencies: config, pandas, numpy

import os
import re
import numpy as np
import pandas as pd

import config


# -- Timestamp detection -------------------------------------------------------

def _detect_ts_unit(sample_val: float) -> tuple[str, float]:
    """
    Infer the timestamp unit from a representative value and return
    (unit_label, divisor_to_get_seconds).

    Thresholds calibrated to 2024-era Unix timestamps:
      nanoseconds  ~1.72e18  -> divide by 1e9
      microseconds ~1.72e15  -> divide by 1e6
      milliseconds ~1.72e12  -> divide by 1e3
      seconds      ~1.72e9   -> divide by 1e0
    """
    if sample_val > 1e16:
        return "nanoseconds", 1e9
    elif sample_val > 1e13:
        return "microseconds", 1e6
    elif sample_val > 1e10:
        return "milliseconds", 1e3
    else:
        return "seconds", 1.0


def _convert_timestamps(series: pd.Series) -> pd.Series:
    """
    Convert a raw timestamp column to float seconds since Unix epoch.
    Handles numeric (ns/us/ms/s) and ISO-string formats.
    """
    if pd.api.types.is_numeric_dtype(series):
        sample = float(series.dropna().iloc[0])
        label, divisor = _detect_ts_unit(sample)
        return series.astype(float) / divisor
    else:
        # ISO string fallback
        return pd.to_datetime(series, infer_datetime_format=True).astype(np.int64) / 1e9


# -- Column detection helpers --------------------------------------------------

def _find_col(columns: list[str], *keywords: str, exclude: list[str] | None = None) -> str | None:
    """Return first column whose name contains any keyword (case-insensitive)."""
    lower_cols = [c.lower() for c in columns]
    ex = [e.lower() for e in (exclude or [])]
    for kw in keywords:
        for orig, low in zip(columns, lower_cols):
            if kw in low and low not in ex:
                return orig
    return None


def _find_cols_matching(columns: list[str], pattern: str) -> list[str]:
    """Return all columns matching a regex pattern."""
    rx = re.compile(pattern, re.IGNORECASE)
    return [c for c in columns if rx.search(c)]


def _detect_tick_size(price_series: pd.Series, n_rows: int = 1000) -> float:
    """
    Estimate the minimum non-zero price increment from the first n_rows samples.
    Falls back to 1e-8 if indeterminate.
    """
    sample = price_series.dropna().head(n_rows).round(10)
    diffs = sample.diff().dropna().abs()
    nonzero = diffs[diffs > 0]
    if nonzero.empty:
        return 1e-8
    return float(nonzero.min())


# -- LOB loader ----------------------------------------------------------------

def _load_lob_chunk(path: str, nrows: int | None = None) -> pd.DataFrame:
    """
    Load the LOB CSV (chunked for large files) and return a DataFrame
    with standardised columns: timestamp, bid_price, bid_size, ask_price,
    ask_size, and optionally bid_price_N / ask_price_N for levels 2-5.
    """
    chunks = []
    rows_read = 0
    for chunk in pd.read_csv(path, chunksize=config.CHUNK_SIZE, index_col=0):
        chunks.append(chunk)
        rows_read += len(chunk)
        if nrows and rows_read >= nrows:
            break
    df = pd.concat(chunks, ignore_index=True)
    if nrows:
        df = df.head(nrows)
    return df


def detect_lob_columns(df: pd.DataFrame) -> dict:
    """
    Auto-detect LOB column mapping.

    Returns a dict with keys:
      timestamp, bid_price_1, bid_size_1, ask_price_1, ask_size_1,
      and optionally bid_price_N / ask_price_N / bid_size_N / ask_size_N (N=2..5)
    """
    cols = list(df.columns)
    mapping = {}

    # Timestamp
    ts_col = _find_col(cols, "timestamp", "time", "ts", "date")
    if ts_col is None:
        raise ValueError("LOB: cannot find timestamp column in " + str(cols[:10]))
    mapping["timestamp"] = ts_col

    # Structured bracket notation: asks[0].price, bids[0].amount, etc.
    # Sort numerically by level index (not alphabetically, which would give asks[0],asks[1],asks[10]...)
    def _lvl_key(col_name: str) -> int:
        m = re.search(r"\[(\d+)\]", col_name)
        return int(m.group(1)) if m else 999

    bracket_asks_price = sorted(_find_cols_matching(cols, r"asks?\[(\d+)\]\.price"),  key=_lvl_key)
    bracket_bids_price = sorted(_find_cols_matching(cols, r"bids?\[(\d+)\]\.price"),  key=_lvl_key)
    bracket_asks_size  = sorted(_find_cols_matching(cols, r"asks?\[(\d+)\]\.amount"), key=_lvl_key)
    bracket_bids_size  = sorted(_find_cols_matching(cols, r"bids?\[(\d+)\]\.amount"), key=_lvl_key)

    if bracket_asks_price and bracket_bids_price:
        for i, (ap, bp, as_, bs) in enumerate(
            zip(bracket_asks_price, bracket_bids_price,
                bracket_asks_size,  bracket_bids_size), start=1
        ):
            if i > 5:
                break
            mapping[f"ask_price_{i}"] = ap
            mapping[f"ask_size_{i}"]  = as_
            mapping[f"bid_price_{i}"] = bp
            mapping[f"bid_size_{i}"]  = bs
    else:
        # Generic fallback: search for bid/ask price/size columns
        mapping["ask_price_1"] = _find_col(cols, "ask_price", "askprice", "ask")
        mapping["ask_size_1"]  = _find_col(cols, "ask_size", "asksize", "ask_qty", "ask_vol")
        mapping["bid_price_1"] = _find_col(cols, "bid_price", "bidprice", "bid")
        mapping["bid_size_1"]  = _find_col(cols, "bid_size", "bidsize", "bid_qty", "bid_vol")

    # Validate level-1 always exists
    for key in ("ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1"):
        if mapping.get(key) is None:
            raise ValueError(f"LOB: cannot detect column for '{key}'. Columns: {cols[:20]}")

    return mapping


def load_lob(use_sample: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Load LOB data, detect columns, normalise timestamps.

    Returns (df, mapping) where df has standardised column names and
    mapping shows what raw column each standard name came from.
    """
    path = config.SAMPLE_LOB_FILE if use_sample else config.RAW_LOB_FILE
    nrows = None if use_sample else None  # samples already pre-truncated

    print(f"[data_loader] Loading LOB from: {path}")
    df_raw = _load_lob_chunk(path, nrows=nrows)
    mapping = detect_lob_columns(df_raw)

    # Build normalised dataframe
    result = pd.DataFrame()
    result["timestamp"] = _convert_timestamps(df_raw[mapping["timestamp"]])

    n_levels = sum(1 for k in mapping if k.startswith("bid_price_"))
    for lvl in range(1, n_levels + 1):
        result[f"bid_price_{lvl}"] = pd.to_numeric(df_raw[mapping[f"bid_price_{lvl}"]], errors="coerce")
        result[f"bid_size_{lvl}"]  = pd.to_numeric(df_raw[mapping[f"bid_size_{lvl}"]], errors="coerce")
        result[f"ask_price_{lvl}"] = pd.to_numeric(df_raw[mapping[f"ask_price_{lvl}"]], errors="coerce")
        result[f"ask_size_{lvl}"]  = pd.to_numeric(df_raw[mapping[f"ask_size_{lvl}"]], errors="coerce")

    # Validation
    initial_len = len(result)
    result.dropna(subset=["timestamp", "bid_price_1", "ask_price_1"], inplace=True)
    result.drop_duplicates(subset=["timestamp"], keep="first", inplace=True)
    result.sort_values("timestamp", inplace=True)
    result.reset_index(drop=True, inplace=True)

    out_of_order = (df_raw[mapping["timestamp"]].astype(float).diff() < 0).sum()

    print(f"[data_loader] LOB rows loaded: {initial_len} -> {len(result)} after cleanup")
    print(f"[data_loader] LOB out-of-order timestamps found: {out_of_order}")

    return result, mapping


# -- Trades loader -------------------------------------------------------------

def detect_trade_columns(df: pd.DataFrame, lob_mid: float | None = None) -> dict:
    """
    Auto-detect trade column mapping.

    Returns a dict with keys: timestamp, price, size, side.
    If no side column is found, infers side from price vs mid.
    """
    cols = list(df.columns)
    mapping = {}

    mapping["timestamp"] = _find_col(cols, "timestamp", "time", "ts", "date")
    if mapping["timestamp"] is None:
        raise ValueError("Trades: cannot find timestamp column")

    mapping["price"] = _find_col(cols, "price", "px", "trd_price", "trade_price")
    if mapping["price"] is None:
        raise ValueError("Trades: cannot find price column")

    mapping["size"] = _find_col(cols, "amount", "size", "qty", "quantity", "volume", "vol")
    if mapping["size"] is None:
        raise ValueError("Trades: cannot find size column")

    side_col = _find_col(cols, "side", "aggressor", "direction", "taker", "taker_side")
    mapping["side"]     = side_col
    mapping["side_inferred"] = (side_col is None)

    return mapping


def load_trades(use_sample: bool = True, lob_mid: float | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Load trade data, detect columns, normalise timestamps.

    If no side column exists, infers buy/sell from price vs mid.
    Returns (df, mapping).
    """
    path = config.SAMPLE_TRADES_FILE if use_sample else config.RAW_TRADES_FILE
    print(f"[data_loader] Loading trades from: {path}")

    chunks = []
    for chunk in pd.read_csv(path, chunksize=config.CHUNK_SIZE, index_col=0):
        chunks.append(chunk)
    df_raw = pd.concat(chunks, ignore_index=True)

    mapping = detect_trade_columns(df_raw, lob_mid=lob_mid)

    result = pd.DataFrame()
    result["timestamp"] = _convert_timestamps(df_raw[mapping["timestamp"]])
    result["price"]     = pd.to_numeric(df_raw[mapping["price"]], errors="coerce")
    result["size"]      = pd.to_numeric(df_raw[mapping["size"]], errors="coerce")

    if not mapping["side_inferred"] and mapping["side"] is not None:
        raw_side = df_raw[mapping["side"]].str.lower().str.strip()
        result["side"] = raw_side.map(
            lambda s: "buy" if s in ("buy", "b", "1", "long") else "sell"
        )
    else:
        # Infer: trade_price >= running mid -> buy, else sell
        if lob_mid is not None:
            result["side"] = result["price"].apply(
                lambda p: "buy" if p >= lob_mid else "sell"
            )
        else:
            mid_est = result["price"].median()
            result["side"] = result["price"].apply(
                lambda p: "buy" if p >= mid_est else "sell"
            )

    # Validation
    initial_len = len(result)
    result.dropna(subset=["timestamp", "price", "size"], inplace=True)
    result.drop_duplicates(subset=["timestamp", "price", "size"], keep="first", inplace=True)
    result.sort_values("timestamp", inplace=True)
    result.reset_index(drop=True, inplace=True)

    print(f"[data_loader] Trades rows loaded: {initial_len} -> {len(result)} after cleanup")

    return result, mapping


# -- Sample saver --------------------------------------------------------------

def save_samples():
    """
    Extract first SAMPLE_ROWS rows from raw LOB + trade CSVs and save to
    data/lob_sample.csv and data/trades_sample.csv.
    Skips if sample files already exist.
    """
    os.makedirs(config.DATA_DIR, exist_ok=True)

    for raw_path, sample_path, label in [
        (config.RAW_LOB_FILE,    config.SAMPLE_LOB_FILE,    "LOB"),
        (config.RAW_TRADES_FILE, config.SAMPLE_TRADES_FILE, "trades"),
    ]:
        if os.path.exists(sample_path):
            print(f"[data_loader] {label} sample already exists: {sample_path}")
            continue

        print(f"[data_loader] Saving {config.SAMPLE_ROWS:,} rows for {label} sample ...")
        rows_written = 0
        header_written = False
        with open(sample_path, "w", newline="", encoding="utf-8") as fout:
            for chunk in pd.read_csv(raw_path, chunksize=config.CHUNK_SIZE, index_col=0):
                need = config.SAMPLE_ROWS - rows_written
                subset = chunk.head(need)
                subset.to_csv(fout, header=not header_written)
                header_written = True
                rows_written += len(subset)
                if rows_written >= config.SAMPLE_ROWS:
                    break
        print(f"[data_loader] Saved: {sample_path} ({rows_written:,} rows)")


# -- Best-period finder --------------------------------------------------------

def find_best_period(lob_df: "pd.DataFrame",
                     window: int | None = None) -> tuple[int, int, float, float]:
    """
    Scan the LOB DataFrame in overlapping sliding windows and return the
    5000-row window with the LOWEST price range % (most sideways/ranging).

    Price range % = (max_mid - min_mid) / mean_mid * 100

    Args:
        lob_df: normalised LOB DataFrame (must contain bid_price_1, ask_price_1)
        window: window size in rows (defaults to config.BEST_PERIOD_WINDOW)

    Returns:
        (start_row, end_row, best_range_pct, full_range_pct)
    """
    if window is None:
        window = config.BEST_PERIOD_WINDOW

    mid = (lob_df["bid_price_1"] + lob_df["ask_price_1"]) / 2.0
    n   = len(mid)

    # Full-sample range (for comparison)
    full_mean = float(mid.mean())
    full_range_pct = (float(mid.max()) - float(mid.min())) / full_mean * 100 if full_mean > 0 else 0.0

    best_range_pct = float("inf")
    best_start     = 0

    # Slide with 50 % overlap so we don't miss a good window that straddles
    # a chunk boundary.
    step = max(1, window // 2)
    for start in range(0, max(1, n - window + 1), step):
        end       = min(start + window, n)
        w_mid     = mid.iloc[start:end]
        w_mean    = float(w_mid.mean())
        if w_mean <= 0:
            continue
        rng_pct   = (float(w_mid.max()) - float(w_mid.min())) / w_mean * 100
        if rng_pct < best_range_pct:
            best_range_pct = rng_pct
            best_start     = start

    best_end = min(best_start + window, n)

    print("\nBest ranging period found:")
    print(f"  Start row:             {best_start:,}")
    print(f"  End row:               {best_end:,}")
    print(f"  Price range:           {best_range_pct:.4f}%")
    print(f"  vs Full sample range:  {full_range_pct:.4f}%")

    # Persist to config so other modules can read the window without re-scanning
    config.BEST_PERIOD_START = best_start
    config.BEST_PERIOD_END   = best_end

    return best_start, best_end, best_range_pct, full_range_pct


# -- Column mapping report -----------------------------------------------------

def print_column_mapping(lob_mapping: dict, trade_mapping: dict):
    """Print a human-readable COLUMN MAPPING REPORT to stdout."""
    print("\n" + "=" * 60)
    print("  COLUMN MAPPING REPORT")
    print("=" * 60)
    print("LOB columns:")
    for k, v in sorted(lob_mapping.items()):
        print(f"  {k:25s} <- '{v}'")
    print("Trade columns:")
    for k, v in sorted(trade_mapping.items()):
        inferred = " (INFERRED)" if k == "side" and trade_mapping.get("side_inferred") else ""
        print(f"  {k:25s} <- '{v}'{inferred}")
    print("=" * 60 + "\n")


# -- Standalone test -----------------------------------------------------------

if __name__ == "__main__":
    save_samples()
    lob_df, lob_map   = load_lob(use_sample=True)
    trd_df, trd_map   = load_trades(use_sample=True)
    print_column_mapping(lob_map, trd_map)
    print("LOB head:\n", lob_df[["timestamp", "bid_price_1", "ask_price_1",
                                  "bid_size_1", "ask_size_1"]].head())
    print("Trades head:\n", trd_df.head())
