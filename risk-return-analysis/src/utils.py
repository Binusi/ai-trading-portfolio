import pandas as pd


ASSET_CLASS_LABELS = {
    "us_equity": "US Equities",
    "equity_index_etf": "Index ETFs",
    "bond_etf": "Bond ETFs",
    "commodity_etf": "Commodity ETFs",
    "reit_etf": "REIT ETFs",
    "international_etf": "International ETFs",
    "crypto_etf": "Crypto ETFs",
}

WIDTH = 80


def _label(asset_class: str) -> str:
    return ASSET_CLASS_LABELS.get(asset_class, asset_class)


def print_header(title: str) -> None:
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    print("=" * WIDTH)


def print_subheader(title: str) -> None:
    print()
    print(f"--- {title} ---")


def print_data_summary(data: pd.DataFrame, tickers: list[str]) -> None:
    print_header("DATA SUMMARY")

    first_valid = data["Close"].apply(lambda x: x.first_valid_index())
    print(f"\n  Tickers loaded: {len(tickers)}")
    print(f"  Date range:     {data.index.min().date()} to {data.index.max().date()}")
    print(f"  Trading days:   {len(data)}")

    print_subheader("First Valid Date Per Ticker")
    for ticker in sorted(tickers):
        if ticker in first_valid.index:
            date = first_valid[ticker]
            date_str = date.date() if date is not None else "N/A"
            print(f"  {ticker:<8} {date_str}")


def print_training_progress(target: str, group: str, model: str, val_metrics: dict, test_metrics: dict) -> None:
    val_sharpe = val_metrics.get("backtest_sharpe", float("nan"))
    test_sharpe = test_metrics.get("backtest_sharpe", float("nan"))
    val_hit = val_metrics.get("top_pick_hit_rate", float("nan"))

    sharpe_str = f"{val_sharpe:+.2f}" if not pd.isna(val_sharpe) else "  N/A"
    test_str = f"{test_sharpe:+.2f}" if not pd.isna(test_sharpe) else "  N/A"
    hit_str = f"{val_hit:.0%}" if not pd.isna(val_hit) else " N/A"

    print(f"  {model:<40} Val Sharpe: {sharpe_str}  Test Sharpe: {test_str}  Hit Rate: {hit_str}", flush=True)


def print_model_leaderboard(results_df: pd.DataFrame, top_n: int = 15) -> None:
    print_header("MODEL LEADERBOARD (Top Performers)")

    if results_df.empty:
        print("\n  No results to display.")
        return

    display_cols = {
        "model_name": "Model",
        "target_name": "Target",
        "group_name": "Group",
    }
    optional_cols = {
        "val_backtest_sharpe": "Val Sharpe",
        "test_backtest_sharpe": "Test Sharpe",
        "val_top_pick_hit_rate": "Hit Rate",
        "val_cumulative_return": "Val Return",
        "val_backtest_max_drawdown": "Max DD",
    }

    cols_to_show = list(display_cols.keys())
    rename_map = dict(display_cols)
    for col, label in optional_cols.items():
        if col in results_df.columns:
            cols_to_show.append(col)
            rename_map[col] = label

    top = results_df.head(top_n)[cols_to_show].copy()
    top = top.rename(columns=rename_map)
    top.index = range(1, len(top) + 1)
    top.index.name = "Rank"

    # Format numeric columns
    for col in top.columns:
        if top[col].dtype in ("float64", "float32"):
            if "Rate" in col or "Return" in col:
                top[col] = top[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
            elif "DD" in col:
                top[col] = top[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
            else:
                top[col] = top[col].apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "N/A")

    print()
    print(top.to_string())


def print_asset_class_summary(results_df: pd.DataFrame) -> None:
    print_header("PERFORMANCE BY ASSET CLASS")

    if results_df.empty or "group_name" not in results_df.columns:
        print("\n  No asset class data available.")
        return

    # Filter to per-class groups (exclude "all_assets")
    class_results = results_df[results_df["group_name"] != "all_assets"].copy()
    if class_results.empty:
        print("\n  No per-class results. Only 'all_assets' group was run.")
        return

    summary_cols = []
    if "val_backtest_sharpe" in class_results.columns:
        summary_cols.append(("val_backtest_sharpe", "Avg Val Sharpe"))
    if "val_top_pick_hit_rate" in class_results.columns:
        summary_cols.append(("val_top_pick_hit_rate", "Avg Hit Rate"))

    if not summary_cols:
        print("\n  Insufficient metrics for summary.")
        return

    agg_dict = {col: "mean" for col, _ in summary_cols}
    agg_dict["model_name"] = "count"

    summary = class_results.groupby("group_name").agg(agg_dict).reset_index()
    summary = summary.rename(columns={"group_name": "Asset Class", "model_name": "Models Run"})
    for col, label in summary_cols:
        summary = summary.rename(columns={col: label})

    summary["Asset Class"] = summary["Asset Class"].map(_label)
    summary = summary.sort_values(summary.columns[1], ascending=False)
    summary.index = range(1, len(summary) + 1)

    for col in summary.columns:
        if summary[col].dtype in ("float64", "float32"):
            if "Rate" in col:
                summary[col] = summary[col].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
            else:
                summary[col] = summary[col].apply(lambda x: f"{x:+.3f}" if pd.notna(x) else "N/A")

    print()
    print(summary.to_string())


def print_portfolio_recommendation(strategy: dict) -> None:
    allocation = strategy["allocation"]
    top_picks = strategy["top_picks"]
    actions = strategy["actions"]
    metadata = strategy["metadata"]

    print_header("PORTFOLIO RECOMMENDATION")

    date_str = metadata["date"].date() if metadata["date"] is not None else "N/A"
    print(f"\n  Based on:    {metadata['n_models']} model signals across {metadata['n_tickers']} assets")
    print(f"  As of:       {date_str}")
    print(f"  Risk profile: {metadata['risk_profile'].upper()}")

    # --- Asset Class Allocation ---
    print_subheader("Recommended Asset Class Allocation")

    weights = allocation["weights"]
    base_weights = allocation["base_weights"]
    conviction = allocation["conviction"]

    for asset_class in sorted(weights, key=lambda x: weights[x], reverse=True):
        w = weights[asset_class]
        base_w = base_weights.get(asset_class, 0)
        diff = w - base_w
        conv = conviction.get(asset_class, "LOW")

        bar_len = int(w * 40)
        bar = "#" * bar_len

        if diff > 0.01:
            direction = "OVERWEIGHT"
        elif diff < -0.01:
            direction = "UNDERWEIGHT"
        else:
            direction = "NEUTRAL"

        label = _label(asset_class)
        print(f"  {label:<22} {w:5.1%}  [{bar:<40}]  {direction:<12} (conviction: {conv})")

    total = sum(weights.values())
    print(f"\n  Total: {total:.1%}")

    # --- Top Picks ---
    print_subheader("Top Picks by Asset Class")

    for asset_class in sorted(top_picks, key=lambda x: weights.get(x, 0), reverse=True):
        picks = top_picks[asset_class]
        label = _label(asset_class)
        print(f"\n  {label}:")
        for pick in picks:
            score_str = f"{pick['score']:+.4f}"
            agree_str = f"{pick['agreement']:.0%}"
            print(f"    {pick['signal']:<5} {pick['ticker']:<8} score: {score_str}  agreement: {agree_str}")

    # --- Action Summary ---
    print_subheader("Action Summary")

    for signal in ["BUY", "HOLD", "SELL"]:
        items = actions.get(signal, [])
        if not items:
            print(f"\n  {signal}: (none)")
            continue

        tickers = [f"{item['ticker']}" for item in items]
        print(f"\n  {signal}: {', '.join(tickers)}")
        for item in items:
            print(f"    {item['ticker']:<8} ({_label(item['asset_class']):<22})  score: {item['score']:+.4f}  agreement: {item['agreement']:.0%}")


def print_signal_legend() -> None:
    print_header("HOW TO READ THESE RESULTS")
    print("""
  SIGNALS:
    BUY   = The ML models collectively predict this asset will outperform.
            Consider increasing your position.
    HOLD  = Mixed or weak signals. Keep your current position.
    SELL  = The models predict underperformance. Consider reducing exposure.

  METRICS:
    Sharpe Ratio  = Risk-adjusted return. Higher is better. Above 1.0 is good,
                    above 2.0 is excellent.
    Hit Rate      = How often the model's top pick actually went up. Above 55%
                    is meaningful.
    Agreement     = What percentage of models agree on the direction (0-100%).
                    Higher agreement = stronger conviction.
    Consensus Score = Weighted average of all model predictions. Positive means
                      models expect the asset to go up, negative means down.

  ASSET ALLOCATION:
    OVERWEIGHT  = The models suggest allocating MORE than the baseline to this
                  asset class.
    UNDERWEIGHT = The models suggest allocating LESS.
    NEUTRAL     = Close to the baseline allocation.

  NOTE: These are model-generated signals, not financial advice. Always do
  your own research and consider your personal risk tolerance.
""")
