import pandas as pd
from config_new import RESULTS_DIR
from forward_test_engine import ForwardTestEngine

PER_TICKER_COLS = ["ticker", "n_signals", "IC", "RankIC",
                   "hit_rate", "Sharpe_annual", "MDD"]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("MODEL: DUAL FUSION")
    print("=" * 60)

    engine = ForwardTestEngine(use_dual_fusion=True)
    results = engine.run()
    
    per_ticker_metrics = results.get("per_ticker_metrics")
    if per_ticker_metrics is not None and not per_ticker_metrics.empty:
        per_ticker_df = per_ticker_metrics[PER_TICKER_COLS]
        per_ticker_df.to_csv(
            RESULTS_DIR / "dual_fusion_per_ticker.csv", index=False
        )
        print(f"Saved per-ticker metrics for {len(per_ticker_df)} tickers")
    else:
        print("WARNING: No per-ticker metrics generated. Creating empty file.")
        # Create empty DataFrame with correct columns for reference
        empty_df = pd.DataFrame(columns=PER_TICKER_COLS)
        empty_df.to_csv(
            RESULTS_DIR / "dual_fusion_per_ticker.csv", index=False
        )

    # Save portfolio returns
    portfolio_returns = results.get("portfolio_returns")
    if portfolio_returns is not None and len(portfolio_returns) > 0:
        portfolio_returns.to_csv(
            RESULTS_DIR / "dual_fusion_returns.csv"
        )
        print(f"Saved portfolio returns: {len(portfolio_returns)} days")

    # Save summary metrics
    summary = pd.DataFrame([{
        "Model": results["model"],
        "Sharpe": results["sharpe_ratio"],
        "MaxDD": results["max_drawdown"],
        "Mean IC": results["mean_IC"],
        "Std IC": results["std_IC"],
        "IC IR": results["IC_IR"],
        "Mean RankIC": results["mean_RankIC"],
        "Hit Rate": results["hit_rate"],
        "Turnover": results["turnover"],
        "N Dates": results["n_signal_dates"],
    }])
    summary.to_csv(RESULTS_DIR / "dual_fusion_summary.csv", index=False)
    print(f"Saved summary metrics")

    print(f"\nSaved results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()