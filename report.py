# report.py -- Generate performance report (report.txt) and PNG charts
# Saves all artefacts to results/ directory.
# Dependencies: config, performance, matplotlib, datetime

import os
import datetime

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

import config
from performance import PerformanceTracker


def _ensure_results_dir() -> str:
    """Create results/ directory if absent and return its path."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    return config.RESULTS_DIR


def _ts_to_dt(ts_seconds: list[float]) -> list[datetime.datetime]:
    """Convert float Unix seconds to datetime objects for matplotlib."""
    return [datetime.datetime.utcfromtimestamp(t) for t in ts_seconds]


# -- Charts --------------------------------------------------------------------

def plot_pnl(tracker_trending: PerformanceTracker,
             tracker_ranging:  PerformanceTracker) -> str:
    """
    Dual-line PnL chart: red for trending period, green for ranging period.
    X-axis is normalised time (0-100%) so both runs overlay cleanly.
    """
    out  = _ensure_results_dir()
    path = os.path.join(out, "pnl_over_time.png")

    fig, ax = plt.subplots(figsize=(12, 5))

    def _pct_axis(ts: list[float]) -> list[float]:
        if len(ts) < 2:
            return [0.0] * len(ts)
        t0, t1 = ts[0], ts[-1]
        span = t1 - t0 if t1 > t0 else 1.0
        return [(t - t0) / span * 100.0 for t in ts]

    if tracker_trending.timestamps:
        x_t = _pct_axis(tracker_trending.timestamps)
        ax.plot(x_t, tracker_trending.total_pnl_series,
                color="crimson", linewidth=1.2, label="Total PnL (Trending)")
        ax.plot(x_t, tracker_trending.realized_pnl_series,
                color="crimson", linewidth=0.8, linestyle="--", alpha=0.5,
                label="Realized (Trending)")

    if tracker_ranging.timestamps:
        x_r = _pct_axis(tracker_ranging.timestamps)
        ax.plot(x_r, tracker_ranging.total_pnl_series,
                color="seagreen", linewidth=1.2, label="Total PnL (Ranging)")
        ax.plot(x_r, tracker_ranging.realized_pnl_series,
                color="seagreen", linewidth=0.8, linestyle="--", alpha=0.5,
                label="Realized (Ranging)")

    ax.axhline(0, color="black", linewidth=0.6, linestyle=":")
    ax.set_title("PnL Over Time -- Trending vs Ranging Period")
    ax.set_xlabel("Session Progress (%)")
    ax.set_ylabel("PnL (base asset units)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_inventory(tracker_trending: PerformanceTracker,
                   tracker_ranging:  PerformanceTracker) -> str:
    """
    Inventory chart with colored fill: positive=green, negative=red.
    Shows both trending and ranging on separate sub-axes.
    """
    out  = _ensure_results_dir()
    path = os.path.join(out, "inventory_over_time.png")

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    for ax, tracker, title, colour in [
        (axes[0], tracker_trending, "Inventory -- Trending Period", "crimson"),
        (axes[1], tracker_ranging,  "Inventory -- Ranging Period",  "seagreen"),
    ]:
        if tracker.timestamps:
            def _pct(ts):
                if len(ts) < 2:
                    return list(range(len(ts)))
                t0, span = ts[0], ts[-1] - ts[0]
                return [(t - t0) / (span or 1) * 100 for t in ts]

            x    = _pct(tracker.timestamps)
            inv  = np.array(tracker.inventory_series)
            zeros = np.zeros_like(inv)

            ax.fill_between(x, inv, zeros,
                            where=(inv >= 0), alpha=0.35, color="seagreen", label="Long")
            ax.fill_between(x, inv, zeros,
                            where=(inv <  0), alpha=0.35, color="crimson",  label="Short")
            ax.plot(x, inv, linewidth=0.8, color=colour)
            ax.axhline(0,                  color="black",  linewidth=0.5, linestyle=":")
            ax.axhline( config.MAX_INVENTORY, color="grey", linewidth=0.5, linestyle="--")
            ax.axhline(-config.MAX_INVENTORY, color="grey", linewidth=0.5, linestyle="--")

        ax.set_title(title)
        ax.set_ylabel("Net Position (units)")
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Session Progress (%)")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_spread(tracker_trending: PerformanceTracker,
                tracker_ranging:  PerformanceTracker,
                market_avg_spread: float = 0.0) -> str:
    """
    Spread-over-time chart with a horizontal dashed line for market avg spread.
    """
    out  = _ensure_results_dir()
    path = os.path.join(out, "spread_over_time.png")

    fig, ax = plt.subplots(figsize=(12, 4))

    def _pct(ts):
        if len(ts) < 2:
            return list(range(len(ts)))
        t0, span = ts[0], ts[-1] - ts[0]
        return [(t - t0) / (span or 1) * 100 for t in ts]

    if tracker_trending.timestamps and tracker_trending.spread_series:
        ax.plot(_pct(tracker_trending.timestamps), tracker_trending.spread_series,
                linewidth=0.7, color="crimson",  alpha=0.7, label="Market Spread (Trending)")

    if tracker_ranging.timestamps and tracker_ranging.spread_series:
        ax.plot(_pct(tracker_ranging.timestamps), tracker_ranging.spread_series,
                linewidth=0.7, color="seagreen", alpha=0.7, label="Market Spread (Ranging)")

    if market_avg_spread > 0:
        ax.axhline(market_avg_spread, color="navy", linewidth=1.0, linestyle="--",
                   label=f"Session Avg Spread ({market_avg_spread:.2e})")

    ax.set_title("Bid-Ask Spread Over Time")
    ax.set_xlabel("Session Progress (%)")
    ax.set_ylabel("Spread (price units)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_regime_comparison(trending_metrics: dict,
                           ranging_metrics:  dict) -> str:
    """
    Grouped bar chart comparing key metrics between trending and ranging periods.
    """
    out  = _ensure_results_dir()
    path = os.path.join(out, "regime_comparison.png")

    metrics = [
        ("Total PnL",    "total_pnl"),
        ("Sharpe Ratio", "sharpe_ratio"),
        ("Max Drawdown", "max_drawdown"),
        ("Win Rate",     "win_rate"),
        ("Fill Rate",    "fill_rate"),
    ]
    labels  = [m[0] for m in metrics]
    keys    = [m[1] for m in metrics]
    t_vals  = [trending_metrics.get(k, 0) for k in keys]
    r_vals  = [ranging_metrics.get(k,  0) for k in keys]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    bars_t = ax.bar(x - width / 2, t_vals, width, label="Trending Period",
                    color="crimson", alpha=0.8)
    bars_r = ax.bar(x + width / 2, r_vals, width, label="Ranging Period",
                    color="seagreen", alpha=0.8)

    ax.set_title("Market Regime Comparison -- Trending vs Ranging")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend()

    # Value labels on top of bars
    for bar in list(bars_t) + list(bars_r):
        h = bar.get_height()
        ax.annotate(f"{h:.3g}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_experiment_comparison(results: list[dict]) -> str:
    """
    Grouped bar chart: PnL + Sharpe + Drawdown for each gamma experiment.
    """
    out  = _ensure_results_dir()
    path = os.path.join(out, "experiment_comparison.png")

    if not results:
        return path

    labels    = [r["label"]        for r in results]
    pnls      = [r["total_pnl"]    for r in results]
    sharpes   = [r["sharpe_ratio"] for r in results]
    drawdowns = [r["max_drawdown"] for r in results]

    x     = np.arange(len(labels))
    width = 0.25

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    colours = ["steelblue", "seagreen", "tomato"]

    for i, (ax, vals, title, ylabel) in enumerate(zip(
        axes,
        [pnls, sharpes, drawdowns],
        ["Total PnL", "Sharpe Ratio", "Max Drawdown"],
        ["PnL",       "Sharpe",       "Drawdown"],
    )):
        bars = ax.bar(x, vals, width=0.5,
                      color=[colours[j % len(colours)] for j in range(len(vals))],
                      alpha=0.85)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.axhline(0, color="black", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.3g}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    fig.suptitle("Experiment Comparison (varying gamma)", fontsize=13)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return path


# -- Text report ---------------------------------------------------------------

def save_report(
    trending_tracker:   PerformanceTracker,
    ranging_tracker:    PerformanceTracker,
    cfg_snapshot:       dict,
    market_info:        dict,
    experiment_results: list[dict] | None = None,
    enhanced_metrics:   dict | None       = None,
) -> str:
    """
    Write results/report.txt with 8 professional sections.

    Sections:
      1. Run Info
      2. Market Conditions
      3. Trending Period Performance
      4. Ranging Period Performance
      5. Market Regime Analysis
      6. Gamma Experiment Table
      7. Base vs Enhanced Strategy
      8. Conclusions

    Returns:
        path to report.txt
    """
    out  = _ensure_results_dir()
    path = os.path.join(out, "report.txt")
    now  = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    tm   = trending_tracker.get_metrics()
    rm   = ranging_tracker.get_metrics()

    W = 72   # line width
    SEP = "=" * W

    def section(title: str) -> list[str]:
        return ["", SEP, f"  {title}", SEP]

    def row(label: str, val) -> str:
        return f"  {label:<30} {val}"

    def metric_row(label: str, t_val, r_val, fmt: str = "") -> str:
        try:
            ts = format(t_val, fmt) if fmt else str(t_val)
            rs = format(r_val, fmt) if fmt else str(r_val)
        except Exception:
            ts, rs = str(t_val), str(r_val)
        return f"  {label:<28} {ts:>18} {rs:>18}"

    lines: list[str] = []

    # ---- Section 1: Run Info -------------------------------------------------
    lines += section("1. RUN INFO")
    lines += [
        row("Generated",       now),
        row("Strategy",        "Avellaneda-Stoikov 2008 (base)"),
        row("Enhanced mode",   cfg_snapshot.get("USE_ENHANCED", False)),
        row("GAMMA",           cfg_snapshot.get("GAMMA")),
        row("KAPPA",           cfg_snapshot.get("KAPPA")),
        row("ORDER_SIZE",      cfg_snapshot.get("ORDER_SIZE")),
        row("REFRESH_N",       cfg_snapshot.get("REFRESH_N")),
        row("MAX_INVENTORY",   cfg_snapshot.get("MAX_INVENTORY")),
        row("SIGMA_WINDOW",    cfg_snapshot.get("SIGMA_WINDOW")),
        row("SIGMA_DEFAULT",   cfg_snapshot.get("SIGMA_DEFAULT")),
        row("Use sample data", cfg_snapshot.get("USE_SAMPLE")),
    ]

    # ---- Section 2: Market Conditions ----------------------------------------
    lines += section("2. MARKET CONDITIONS")
    lines += [
        row("Asset",               market_info.get("asset", "N/A")),
        row("Start price",         f"{market_info.get('start_price', 0):.8f}"),
        row("End price",           f"{market_info.get('end_price', 0):.8f}"),
        row("Price change %",      f"{market_info.get('price_change_pct', 0):+.4f}%"),
        row("Avg market spread",   f"{market_info.get('avg_market_spread', 0):.2e}"),
        row("Session duration",    f"{market_info.get('session_duration_min', 0):.1f} min"),
        row("Total LOB rows",      f"{market_info.get('total_lob_rows', 0):,}"),
        row("Total trade events",  f"{market_info.get('total_trade_events', 0):,}"),
        row("Ranging window rows", f"{market_info.get('ranging_start_row', 0):,} - "
                                   f"{market_info.get('ranging_end_row', 0):,}"),
        row("Ranging price range", f"{market_info.get('ranging_price_pct', 0):.4f}%"),
        row("Full period range",   f"{market_info.get('full_price_pct', 0):.4f}%"),
    ]

    # ---- Section 3: Trending Period Performance --------------------------------
    lines += section("3. TRENDING PERIOD PERFORMANCE  (full sample)")
    lines += [
        row("Total PnL",         f"{tm['total_pnl']:+.8f}"),
        row("Realized PnL",      f"{tm['realized_pnl']:+.8f}"),
        row("Unrealized PnL",    f"{tm['unrealized_pnl']:+.8f}"),
        row("Final Inventory",   f"{tm['inventory']:+.6f}"),
        row("Total Turnover",    f"{tm['turnover']:.4f}"),
        row("Total Trades",      f"{tm['trade_count']}"),
        row("Win Rate",          f"{tm['win_rate']:.2%}"),
        row("Max Drawdown",      f"{tm['max_drawdown']:.8f}"),
        row("Sharpe Ratio",      f"{tm['sharpe_ratio']:.4f}"),
        row("Avg Quoted Spread", f"{tm['avg_quoted_spread']:.2e}"),
        row("Fill Rate",         f"{tm['fill_rate']:.2f}%"),
    ]

    # ---- Section 4: Ranging Period Performance --------------------------------
    lines += section("4. RANGING PERIOD PERFORMANCE  (best sideways window)")
    lines += [
        row("Total PnL",         f"{rm['total_pnl']:+.8f}"),
        row("Realized PnL",      f"{rm['realized_pnl']:+.8f}"),
        row("Unrealized PnL",    f"{rm['unrealized_pnl']:+.8f}"),
        row("Final Inventory",   f"{rm['inventory']:+.6f}"),
        row("Total Turnover",    f"{rm['turnover']:.4f}"),
        row("Total Trades",      f"{rm['trade_count']}"),
        row("Win Rate",          f"{rm['win_rate']:.2%}"),
        row("Max Drawdown",      f"{rm['max_drawdown']:.8f}"),
        row("Sharpe Ratio",      f"{rm['sharpe_ratio']:.4f}"),
        row("Avg Quoted Spread", f"{rm['avg_quoted_spread']:.2e}"),
        row("Fill Rate",         f"{rm['fill_rate']:.2f}%"),
    ]

    # ---- Section 5: Market Regime Analysis ------------------------------------
    lines += section("5. MARKET REGIME ANALYSIS")
    lines += [
        f"  {'METRIC':<28} {'TRENDING':>18} {'RANGING':>18}",
        "  " + "-" * 66,
        metric_row("Total PnL",         tm["total_pnl"],      rm["total_pnl"],      "+.8f"),
        metric_row("Sharpe Ratio",      tm["sharpe_ratio"],   rm["sharpe_ratio"],   ".4f"),
        metric_row("Max Drawdown",      tm["max_drawdown"],   rm["max_drawdown"],   ".8f"),
        metric_row("Win Rate",          tm["win_rate"],       rm["win_rate"],       ".2%"),
        metric_row("Fill Rate (%)",     tm["fill_rate"],      rm["fill_rate"],      ".2f"),
        metric_row("Avg Quoted Spread", tm["avg_quoted_spread"], rm["avg_quoted_spread"], ".2e"),
        metric_row("Total Trades",      tm["trade_count"],    rm["trade_count"]),
        metric_row("Final Inventory",   tm["inventory"],      rm["inventory"],      ".4f"),
        "",
        "  Key insight:",
        "  Adverse selection is the dominant risk in trending markets. When price",
        "  drifts directionally, the market maker accumulates inventory at stale",
        "  prices and unwinds at a loss. The ranging window isolates a sideways",
        "  period where the spread-capture edge can materialise without inventory",
        "  bleed from momentum.",
    ]

    # ---- Section 6: Gamma Experiment Table ------------------------------------
    if experiment_results:
        lines += section("6. GAMMA EXPERIMENT RESULTS")
        lines += [
            f"  {'Label':<16} {'Total PnL':>14} {'Sharpe':>10} "
            f"{'Drawdown':>12} {'Trades':>8} {'Win Rate':>10}",
            "  " + "-" * 72,
        ]
        for r in experiment_results:
            lines.append(
                f"  {r['label']:<16} {r['total_pnl']:>14.8f} "
                f"{r['sharpe_ratio']:>10.4f} {r['max_drawdown']:>12.8f} "
                f"{r['trade_count']:>8} {r.get('win_rate', 0):>10.2%}"
            )
        lines += [
            "",
            "  Interpretation:",
            "  Higher gamma aggressively skews quotes away from inventory",
            "  accumulation but also widens spreads, reducing fill probability.",
            "  In a trending market the optimal gamma trades off fill rate vs",
            "  directional exposure -- higher gamma tends to limit drawdown.",
        ]
    else:
        lines += section("6. GAMMA EXPERIMENT RESULTS")
        lines.append("  No experiment results provided.")

    # ---- Section 7: Base vs Enhanced ------------------------------------------
    if enhanced_metrics:
        em = enhanced_metrics
        lines += section("7. BASE vs ENHANCED STRATEGY  (full sample)")
        lines += [
            f"  {'Metric':<28} {'Base':>18} {'Enhanced':>18}",
            "  " + "-" * 66,
        ]
        for key, label in [
            ("total_pnl",         "Total PnL"),
            ("realized_pnl",      "Realized PnL"),
            ("sharpe_ratio",      "Sharpe Ratio"),
            ("max_drawdown",      "Max Drawdown"),
            ("trade_count",       "Trade Count"),
            ("win_rate",          "Win Rate"),
            ("fill_rate",         "Fill Rate (%)"),
            ("avg_quoted_spread", "Avg Quoted Spread"),
        ]:
            bv = tm.get(key, 0)
            ev = em.get(key, 0)
            lines.append(f"  {label:<28} {str(bv):>18} {str(ev):>18}")
        lines += [
            "",
            "  Enhanced additions: microprice reference price, volatility regime",
            "  spread multiplier, and order-flow momentum quote skew.",
        ]
    else:
        lines += section("7. BASE vs ENHANCED STRATEGY")
        lines.append("  Enhanced strategy metrics not provided.")

    # ---- Section 8: Conclusions -----------------------------------------------
    lines += section("8. CONCLUSIONS")
    pnl_better = rm["total_pnl"] > tm["total_pnl"]
    sharpe_better = rm["sharpe_ratio"] > tm["sharpe_ratio"]
    lines += [
        f"  Ranging period PnL {'BETTER' if pnl_better else 'NOT better'} than "
        f"trending ({rm['total_pnl']:+.6f} vs {tm['total_pnl']:+.6f}).",
        f"  Ranging period Sharpe {'BETTER' if sharpe_better else 'NOT better'} than "
        f"trending ({rm['sharpe_ratio']:.4f} vs {tm['sharpe_ratio']:.4f}).",
        "",
        "  The Avellaneda-Stoikov model is well-suited to mean-reverting markets.",
        "  Its reservation-price mechanism provides inherent inventory dampening,",
        "  but it has no explicit mechanism to detect or avoid trending regimes.",
        "  The regime comparison above quantifies this structural weakness.",
        "",
        "  Recommended next steps: add a rolling trend filter (fast/slow EMA",
        "  divergence) to suspend quoting when directional drift exceeds a",
        "  threshold, and implement dynamic kappa estimation from trade-arrival",
        "  rates rather than a fixed calibration constant.",
        "",
        SEP,
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return path


def generate_all_charts(
    tracker_trending:   PerformanceTracker,
    tracker_ranging:    PerformanceTracker,
    experiment_results: list[dict] | None = None,
    market_avg_spread:  float             = 0.0,
) -> list[str]:
    """
    Generate and save all 5 standard charts. Returns list of saved paths.

    Charts:
      1. pnl_over_time.png          -- dual-line trending vs ranging
      2. inventory_over_time.png    -- colored long/short fill, both periods
      3. spread_over_time.png       -- market spread + avg reference line
      4. regime_comparison.png      -- grouped bar: trending vs ranging metrics
      5. experiment_comparison.png  -- gamma experiment grouped bars
    """
    tm = tracker_trending.get_metrics()
    rm = tracker_ranging.get_metrics()

    paths = [
        plot_pnl(tracker_trending, tracker_ranging),
        plot_inventory(tracker_trending, tracker_ranging),
        plot_spread(tracker_trending, tracker_ranging, market_avg_spread),
        plot_regime_comparison(tm, rm),
    ]
    if experiment_results:
        paths.append(plot_experiment_comparison(experiment_results))
    return paths


# -- Standalone test -----------------------------------------------------------

if __name__ == "__main__":
    import random
    from order_manager import Fill

    print("=== report.py test with synthetic data ===")
    base_ts = 1722470402.0
    mid     = 0.011044

    def _make_tracker(seed: int) -> PerformanceTracker:
        rng = random.Random(seed)
        t   = PerformanceTracker()
        for i in range(200):
            side = "buy" if i % 2 == 0 else "sell"
            px   = mid + rng.uniform(-0.000005, 0.000005)
            fill = Fill(timestamp=base_ts + i * 60,
                        order_id=f"O{i:03d}", side=side,
                        fill_price=px, fill_size=1.0, fill_type="full")
            t.record_fill(fill)
            t.update_mid(mid + rng.uniform(-0.0001, 0.0001))
            if i % 10 == 0:
                t.snapshot(base_ts + i * 60, spread=0.000001)
        return t

    t_trending = _make_tracker(42)
    t_ranging  = _make_tracker(99)

    cfg = {"GAMMA": 0.1, "KAPPA": 1.5e9, "ORDER_SIZE": 1.0,
           "REFRESH_N": 10, "MAX_INVENTORY": 10, "SIGMA_WINDOW": 100,
           "SIGMA_DEFAULT": 0.001, "USE_SAMPLE": True, "USE_ENHANCED": False}
    mkt = {"asset": "CRYPTO (BTC-like)", "start_price": 0.01104, "end_price": 0.01098,
           "price_change_pct": -0.54, "avg_market_spread": 1e-7,
           "session_duration_min": 47.0, "total_lob_rows": 50000,
           "total_trade_events": 946, "ranging_start_row": 10000,
           "ranging_end_row": 15000, "ranging_price_pct": 0.12,
           "full_price_pct": 0.55}
    fake_exp = [
        {"label": "gamma=0.01", "total_pnl": 0.001, "sharpe_ratio": 1.2,
         "max_drawdown": 0.0001, "trade_count": 180, "win_rate": 0.55},
        {"label": "gamma=0.10", "total_pnl": 0.0008, "sharpe_ratio": 0.9,
         "max_drawdown": 0.0002, "trade_count": 150, "win_rate": 0.50},
        {"label": "gamma=0.50", "total_pnl": 0.0003, "sharpe_ratio": 0.4,
         "max_drawdown": 0.0005, "trade_count": 80,  "win_rate": 0.45},
    ]
    fake_enh = t_trending.get_metrics()

    report_path = save_report(t_trending, t_ranging, cfg, mkt, fake_exp, fake_enh)
    chart_paths = generate_all_charts(t_trending, t_ranging, fake_exp, mkt["avg_market_spread"])

    print(f"Report: {report_path}")
    for cp in chart_paths:
        print(f"Chart:  {cp}")
