# backtest.py -- Main entry point for the HFT backtesting engine
# Wires data loader -> order book -> OMS -> performance tracker -> strategy.
# Run: python backtest.py
# Dependencies: all project modules + pandas, numpy

import os
import sys
import time
import argparse

import pandas as pd
import numpy as np

import config
from data_loader   import (save_samples, load_lob, load_trades,
                            print_column_mapping, find_best_period)
from order_book    import OrderBook
from order_manager import OMS
from performance   import PerformanceTracker
from strategy      import AvellanedaStoikov, EnhancedAS
from report        import save_report, generate_all_charts


# -- Core backtest runner ------------------------------------------------------

def run_backtest(
    lob_df:       pd.DataFrame,
    trades_df:    pd.DataFrame,
    gamma:        float = config.GAMMA,
    use_enhanced: bool  = config.USE_ENHANCED,
    label:        str   = "backtest",
    verbose:      bool  = True,
) -> PerformanceTracker:
    """
    Execute a single full backtest pass.

    Args:
        lob_df:       normalised LOB DataFrame (any row-slice is accepted)
        trades_df:    normalised trades DataFrame (pre-filtered to same time window)
        gamma:        risk-aversion parameter (overrides config.GAMMA)
        use_enhanced: whether to use the enhanced AS strategy
        label:        name used in progress prints
        verbose:      print progress every 10,000 LOB ticks

    Returns:
        PerformanceTracker with all metrics and snapshot series populated.
    """
    book    = OrderBook()
    oms     = OMS()
    tracker = PerformanceTracker()

    StrategyClass = EnhancedAS if use_enhanced else AvellanedaStoikov
    strategy = StrategyClass(gamma=gamma)

    lob_df    = lob_df.reset_index(drop=True)
    t_start   = float(lob_df["timestamp"].iloc[0])
    t_end     = float(lob_df["timestamp"].iloc[-1])

    # Merge LOB and trade events sorted by timestamp
    lob_events   = [(float(ts), "lob",   idx)
                    for idx, ts in enumerate(lob_df["timestamp"])]
    trade_events = [(float(ts), "trade", idx)
                    for idx, ts in enumerate(trades_df["timestamp"])]
    all_events   = sorted(lob_events + trade_events, key=lambda x: x[0])

    total_events = len(all_events)
    lob_tick     = 0
    last_t       = t_start

    if verbose:
        print(f"\n[{label}] {total_events:,} events  "
              f"({len(lob_df):,} LOB + {len(trades_df):,} trades)  "
              f"gamma={gamma}  enhanced={use_enhanced}")

    start_wall = time.time()

    for ev_i, (ts, ev_type, row_idx) in enumerate(all_events):
        last_t = ts

        if ev_type == "lob":
            row = lob_df.iloc[row_idx].to_dict()
            book.update(row)

            if not book.is_ready():
                continue

            tracker.update_mid(book.mid_price)
            lob_tick += 1

            # Snapshot every SNAPSHOT_INTERVAL LOB ticks
            if lob_tick % config.SNAPSHOT_INTERVAL == 0:
                tracker.snapshot(ts, book.spread)

            # Requote every REFRESH_N LOB ticks
            if lob_tick % config.REFRESH_N == 0:
                oms.cancel_all_orders()
                bid_px, ask_px = strategy.compute_quotes(
                    book, tracker.inventory, ts, t_start, t_end
                )
                if bid_px is not None:
                    oms.place_order("buy",  bid_px, config.ORDER_SIZE, ts)
                    tracker.total_orders_placed += 1
                if ask_px is not None:
                    oms.place_order("sell", ask_px, config.ORDER_SIZE, ts)
                    tracker.total_orders_placed += 1
                # Record quoted spread when both sides are live
                if bid_px is not None and ask_px is not None:
                    tracker.quoted_spread_series.append(ask_px - bid_px)

            # Progress print every 10,000 LOB ticks
            if verbose and lob_tick % 10_000 == 0:
                m       = tracker.get_metrics()
                pct     = ev_i / total_events * 100
                elapsed = time.time() - start_wall
                print(
                    f"  Tick {lob_tick:>6,}/{len(lob_df):,}"
                    f"  ({pct:5.1f}%)"
                    f"  PnL: {m['total_pnl']:+.6f}"
                    f"  Inv: {m['inventory']:+.2f}"
                    f"  Trades: {m['trade_count']:,}"
                    f"  {elapsed:.1f}s"
                )

        else:  # trade event
            trade_row = trades_df.iloc[row_idx]
            trd_price = float(trade_row["price"])
            trd_size  = float(trade_row["size"])
            trd_side  = str(trade_row.get("side", "buy"))

            if use_enhanced and isinstance(strategy, EnhancedAS):
                strategy.record_trade_side(trd_side)

            fills = oms.process_trade(trd_price, trd_size, ts)
            for fill in fills:
                tracker.record_fill(fill)

    # Final snapshot
    if book.is_ready():
        tracker.snapshot(last_t, book.spread or 0.0)

    if verbose:
        print(f"  [DONE]  {time.time()-start_wall:.1f}s  "
              f"LOB ticks: {lob_tick:,}")

    return tracker


# -- Experiment runner ---------------------------------------------------------

def run_experiments(lob_df: pd.DataFrame,
                    trades_df: pd.DataFrame) -> list[dict]:
    """Run 3 gamma experiments, return list of metrics dicts."""
    results = []
    for gamma in config.EXPERIMENT_GAMMAS:
        label = f"gamma={gamma}"
        print(f"\n{'-'*55}")
        print(f"  Experiment: {label}")
        tracker = run_backtest(lob_df, trades_df, gamma=gamma,
                               use_enhanced=False, label=label)
        m = tracker.get_metrics()
        m["label"] = label
        m["gamma"] = gamma
        results.append(m)
    return results


# -- Helpers -------------------------------------------------------------------

def _filter_trades_to_window(trades_df: pd.DataFrame,
                              t_start: float, t_end: float) -> pd.DataFrame:
    """Return trades whose timestamps fall within [t_start, t_end]."""
    mask = (trades_df["timestamp"] >= t_start) & (trades_df["timestamp"] <= t_end)
    return trades_df[mask].reset_index(drop=True)


def _print_comparison(trending_metrics: dict, ranging_metrics: dict):
    """Print TRENDING vs RANGING metrics side by side."""
    print("\n" + "=" * 64)
    print("  TRENDING vs RANGING PERIOD COMPARISON")
    print("=" * 64)
    print(f"  {'METRIC':<22} {'TRENDING PERIOD':>18} {'RANGING PERIOD':>18}")
    print("  " + "-" * 60)
    rows = [
        ("Total PnL",       "total_pnl",       ".8f"),
        ("Sharpe Ratio",    "sharpe_ratio",     ".4f"),
        ("Max Drawdown",    "max_drawdown",     ".8f"),
        ("Total Trades",    "trade_count",      "d"),
        ("Win Rate",        "win_rate",         ".2%"),
        ("Final Inventory", "inventory",        ".4f"),
        ("Fill Rate (%)",   "fill_rate",        ".2f"),
        ("Avg Quoted Spread","avg_quoted_spread",".2e"),
    ]
    for label, key, fmt in rows:
        tv = trending_metrics.get(key, 0)
        rv = ranging_metrics.get(key, 0)
        try:
            ts = format(tv, fmt)
            rs = format(rv, fmt)
        except Exception:
            ts = str(tv)
            rs = str(rv)
        print(f"  {label:<22} {ts:>18} {rs:>18}")
    print("=" * 64)


# -- Main ----------------------------------------------------------------------

def main():
    """
    Full pipeline:
      1. Save sample files if absent
      2. Load LOB + trade data
      3. Find best (most-ranging) 5000-row window
      4. Run base strategy on FULL sample   (trending period)
      5. Run base strategy on RANGING window
      6. Print comparison table
      7. Run enhanced strategy (full sample)
      8. Run 3 gamma experiments (full sample)
      9. Generate report + charts
    """
    parser = argparse.ArgumentParser(description="HFT backtester")
    parser.add_argument("--full", action="store_true",
                        help="Use full 900MB data files instead of sample")
    args = parser.parse_args()
    use_sample = not args.full

    print("\n" + "=" * 60)
    print("  HFT Backtester -- Avellaneda-Stoikov Market Making")
    print("=" * 60)

    # 1. Sample files
    print("\n[1/6] Preparing data ...")
    save_samples()

    # 2. Load data
    print("\n[2/6] Loading data ...")
    try:
        lob_df,    lob_map = load_lob(use_sample=use_sample)
        trades_df, trd_map = load_trades(use_sample=use_sample)
    except Exception as e:
        print(f"ERROR loading data: {e}")
        sys.exit(1)

    print_column_mapping(lob_map, trd_map)

    # Market info for report
    start_price = float(lob_df["bid_price_1"].iloc[0])
    end_price   = float(lob_df["bid_price_1"].iloc[-1])
    avg_mkt_spread = float((lob_df["ask_price_1"] - lob_df["bid_price_1"]).mean())
    session_min    = (float(lob_df["timestamp"].iloc[-1])
                      - float(lob_df["timestamp"].iloc[0])) / 60.0
    price_chg_pct  = (end_price - start_price) / start_price * 100 if start_price > 0 else 0.0

    # 3. Find best ranging period
    print("\n[3/6] Finding best ranging period ...")
    best_start, best_end, range_pct, full_range_pct = find_best_period(lob_df)

    # Slice LOB and matching trades for the ranging window
    lob_ranging    = lob_df.iloc[best_start:best_end].reset_index(drop=True)
    t_rng_start    = float(lob_ranging["timestamp"].iloc[0])
    t_rng_end      = float(lob_ranging["timestamp"].iloc[-1])
    trades_ranging = _filter_trades_to_window(trades_df, t_rng_start, t_rng_end)

    # 4. Full-sample (trending) base strategy run
    print("\n[4/6] Running BASE strategy (full sample = trending period) ...")
    tracker_trending = run_backtest(
        lob_df, trades_df,
        gamma=config.GAMMA, use_enhanced=False,
        label="base_trending",
    )

    # 5. Ranging period base strategy run
    print("\n[5/6] Running BASE strategy (ranging period) ...")
    tracker_ranging = run_backtest(
        lob_ranging, trades_ranging,
        gamma=config.GAMMA, use_enhanced=False,
        label="base_ranging",
    )

    # 6. Print comparison
    _print_comparison(tracker_trending.get_metrics(),
                      tracker_ranging.get_metrics())

    # 7. Enhanced strategy (full sample)
    print("\n[EXTRA] Running ENHANCED strategy (full sample) ...")
    tracker_enh = run_backtest(
        lob_df, trades_df,
        gamma=config.GAMMA, use_enhanced=True,
        label="enhanced",
    )

    # 8. Gamma experiments (full sample)
    print("\n[EXTRA] Running gamma experiments ...")
    experiment_results = run_experiments(lob_df, trades_df)

    # -- Report + Charts -------------------------------------------------------
    market_info = {
        "asset":              "CRYPTO (BTC-like)",
        "start_price":        start_price,
        "end_price":          end_price,
        "price_change_pct":   price_chg_pct,
        "avg_market_spread":  avg_mkt_spread,
        "session_duration_min": session_min,
        "total_lob_rows":     len(lob_df),
        "total_trade_events": len(trades_df),
        "ranging_start_row":  best_start,
        "ranging_end_row":    best_end,
        "ranging_price_pct":  range_pct,
        "full_price_pct":     full_range_pct,
    }

    cfg_snapshot = {
        "GAMMA":         config.GAMMA,
        "KAPPA":         config.KAPPA,
        "ORDER_SIZE":    config.ORDER_SIZE,
        "REFRESH_N":     config.REFRESH_N,
        "MAX_INVENTORY": config.MAX_INVENTORY,
        "SIGMA_WINDOW":  config.SIGMA_WINDOW,
        "SIGMA_DEFAULT": config.SIGMA_DEFAULT,
        "USE_SAMPLE":    use_sample,
        "USE_ENHANCED":  config.USE_ENHANCED,
    }

    report_path = save_report(
        trending_tracker   = tracker_trending,
        ranging_tracker    = tracker_ranging,
        cfg_snapshot       = cfg_snapshot,
        market_info        = market_info,
        experiment_results = experiment_results,
        enhanced_metrics   = tracker_enh.get_metrics(),
    )

    chart_paths = generate_all_charts(
        tracker_trending   = tracker_trending,
        tracker_ranging    = tracker_ranging,
        experiment_results = experiment_results,
        market_avg_spread  = avg_mkt_spread,
    )

    # -- Final console output --------------------------------------------------
    tm = tracker_trending.get_metrics()
    rm = tracker_ranging.get_metrics()
    print("\n" + "=" * 60)
    print("  FINAL RESULTS")
    print("=" * 60)
    print(f"  [OK] Total PnL (Trending):  {tm['total_pnl']:+.8f}")
    print(f"  [OK] Total PnL (Ranging):   {rm['total_pnl']:+.8f}")
    print(f"  [OK] Sharpe (Trending):     {tm['sharpe_ratio']:.4f}")
    print(f"  [OK] Sharpe (Ranging):      {rm['sharpe_ratio']:.4f}")
    print(f"  [OK] Total Trades:          {tm['trade_count']}")
    print(f"  [OK] Win Rate:              {tm['win_rate']:.2%}")
    print(f"  [OK] Fill Rate:             {tm['fill_rate']:.2f}%")
    print(f"  [OK] Report saved:          {report_path}")
    print(f"  [OK] Charts saved:          {config.RESULTS_DIR}")
    print()

    print("  Experiment Summary:")
    for r in experiment_results:
        print(f"    {r['label']:12s}  PnL={r['total_pnl']:+.6f}  "
              f"Sharpe={r['sharpe_ratio']:.3f}")

    print("\n  Base vs Enhanced (full sample):")
    bm, em = tracker_trending.get_metrics(), tracker_enh.get_metrics()
    for k in ("total_pnl", "sharpe_ratio", "trade_count"):
        print(f"    {k:<20}  base={bm[k]}  enhanced={em[k]}")


if __name__ == "__main__":
    main()
