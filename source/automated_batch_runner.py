
"""
Automated batch runner for short interest prediction pipeline.

This script runs the pipeline for multiple tickers with different configurations
and aggregates the out-of-sample results for comparison.
"""

import os
import sys
import json
import pandas as pd
from datetime import datetime
from typing import Dict, List
import importlib.util

# Import the pipeline functions
spec = importlib.util.spec_from_file_location("pipeline", "full_pipeline.py")
pipeline_module = importlib.util.module_from_spec(spec)
sys.modules["pipeline"] = pipeline_module
spec.loader.exec_module(pipeline_module)

# Access pipeline functions
run_pipeline = pipeline_module.run_pipeline
get_finra_short_data = pipeline_module.get_finra_short_data
get_stock_data = pipeline_module.get_stock_data
create_daily_master_df = pipeline_module.create_daily_master_df
create_target_variants = pipeline_module.create_target_variants
run_pipeline_for_target = pipeline_module.run_pipeline_for_target

# Configuration
TICKERS_CONFIG = [
    {"ticker": "ISPR", "index": "RUA"},
    {"ticker": "QUBT", "index": "RUA"},
    {"ticker": "BHF", "index": "RUI"},
    {"ticker": "CART", "index": "RUI"},
    {"ticker": "AMPG", "index": "RUA"},
    {"ticker": "ORMP", "index": "RUA"},
    {"ticker": "ATLX", "index": "RUA"},
    {"ticker": "PLTR", "index": "SPX"},
    {"ticker": "MHK", "index": "SPX"},
    {"ticker": "KMI", "index": "SPX"},
]

# Output directory for batch results
BATCH_OUTPUT_DIR = "batch_results"
os.makedirs(BATCH_OUTPUT_DIR, exist_ok=True)

# Results aggregator
class ResultsAggregator:
    def __init__(self):
        self.results = []

    def add_result(self, ticker: str, index: str, stock_mode: str, 
                   image_mode: bool, pruning_mode: bool, target_type: str,
                   model_type: str, metrics: Dict):
        """Add a single test result."""
        self.results.append({
            "ticker": ticker,
            "index": index,
            "stock_mode": stock_mode,  # "single" or "multiple"
            "image_mode": "image" if image_mode else "table",
            "pruning_mode": "pruning" if pruning_mode else "no_pruning",
            "target_type": target_type,  # "raw", "pct_change", "log_change"
            "model_type": model_type,  # "xgb" or "ridge"
            "train_rmse": metrics.get("train_rmse"),
            "train_r2": metrics.get("train_r2"),
            "test_rmse": metrics.get("test_rmse"),
            "test_r2": metrics.get("test_r2"),
            "timestamp": datetime.now().isoformat()
        })

    def save_results(self, filename: str = "aggregated_results.csv"):
        """Save all results to CSV."""
        df = pd.DataFrame(self.results)
        filepath = os.path.join(BATCH_OUTPUT_DIR, filename)
        df.to_csv(filepath, index=False)
        print(f"\n✓ Results saved to: {filepath}")
        return df

    def create_comparison_table(self):
        """Create comparison tables for different test dimensions."""
        if not self.results:
            print("No results to compare")
            return

        df = pd.DataFrame(self.results)

        # 1. Compare stock modes (single vs multiple)
        print("\n" + "="*80)
        print("COMPARISON 1: Single vs Multiple Stocks in Feature Matrix")
        print("="*80)
        stock_comparison = df.groupby(['ticker', 'stock_mode', 'model_type'])[['test_r2', 'test_rmse']].mean()
        print(stock_comparison)
        stock_comparison.to_csv(os.path.join(BATCH_OUTPUT_DIR, "comparison_stock_mode.csv"))

        # 2. Compare image modes (table vs image)
        print("\n" + "="*80)
        print("COMPARISON 2: Table vs Image Mode")
        print("="*80)
        image_comparison = df.groupby(['ticker', 'image_mode', 'model_type'])[['test_r2', 'test_rmse']].mean()
        print(image_comparison)
        image_comparison.to_csv(os.path.join(BATCH_OUTPUT_DIR, "comparison_image_mode.csv"))

        # 3. Compare pruning modes
        print("\n" + "="*80)
        print("COMPARISON 3: Pruning vs No Pruning")
        print("="*80)
        pruning_comparison = df.groupby(['ticker', 'pruning_mode', 'model_type'])[['test_r2', 'test_rmse']].mean()
        print(pruning_comparison)
        pruning_comparison.to_csv(os.path.join(BATCH_OUTPUT_DIR, "comparison_pruning_mode.csv"))

        # 4. Compare target types
        print("\n" + "="*80)
        print("COMPARISON 4: Target Transformations (Raw vs Pct Change vs Log Change)")
        print("="*80)
        target_comparison = df.groupby(['ticker', 'target_type', 'model_type'])[['test_r2', 'test_rmse']].mean()
        print(target_comparison)
        target_comparison.to_csv(os.path.join(BATCH_OUTPUT_DIR, "comparison_target_type.csv"))

        # 5. Overall best configurations per ticker
        print("\n" + "="*80)
        print("BEST CONFIGURATIONS PER TICKER (by test R²)")
        print("="*80)
        best_configs = df.loc[df.groupby('ticker')['test_r2'].idxmax()]
        print(best_configs[['ticker', 'stock_mode', 'image_mode', 'pruning_mode', 
                           'target_type', 'model_type', 'test_r2', 'test_rmse']])
        best_configs.to_csv(os.path.join(BATCH_OUTPUT_DIR, "best_configurations.csv"), index=False)

def run_single_test(ticker: str, index: str, stock_list: List[str], 
                    use_images: bool, use_pruning: bool, 
                    aggregator: ResultsAggregator,
                    n_trials: int = 100, n_passes: int = 3):
    """Run a single test configuration."""

    test_name = f"{ticker}_{'multi' if len(stock_list) > 1 else 'single'}_{'image' if use_images else 'table'}_{'pruning' if use_pruning else 'nopruning'}"

    print("\n" + "="*80)
    print(f"RUNNING TEST: {test_name}")
    print("="*80)
    print(f"Ticker: {ticker}")
    print(f"Stock list: {stock_list}")
    print(f"Use images: {use_images}")
    print(f"Use pruning: {use_pruning}")
    print("="*80 + "\n")

    try:
        # Load data
        print(f"Loading short interest data for {ticker}...")
        short_df = get_finra_short_data(ticker)
        print(f"✓ Found {len(short_df)} biweekly measurements")

        print(f"\nLoading daily stock data for {stock_list}...")
        stock_dict = {}
        for stock in stock_list:
            try:
                stock_dict[stock] = get_stock_data(stock, use_alpha_vantage=False)
                print(f"  ✓ Loaded {stock}")
            except Exception as e:
                print(f"  ⚠ Skipping {stock}: {e}")

        if not stock_dict:
            raise RuntimeError("No stock data loaded")

        daily_master = create_daily_master_df(stock_dict)
        print(f"✓ Master DataFrame: {daily_master.shape}")

        # Create target variants
        target_variants = create_target_variants(short_df)

        # Run for each target type
        for target_type in ['raw', 'pct_change', 'log_change']:
            target_label = {'raw': 'Raw', 'pct_change': 'Pct Change', 'log_change': 'Log Change'}[target_type]

            print(f"\n{'='*60}")
            print(f"Processing {target_label} target for {ticker}")
            print(f"{'='*60}")

            try:
                results = run_pipeline_for_target(
                    target_type=target_type,
                    target_label=target_label,
                    daily_master=daily_master,
                    short_df_variant=target_variants[target_type],
                    stock_list=stock_list,
                    short_stock=ticker,
                    use_llm=True,
                    use_images=use_images,
                    n_trials=n_trials,
                    n_passes=n_passes,
                    max_leakage_retries=3,
                    use_pruning=use_pruning
                )

                # Store results for both models
                stock_mode = "multiple" if len(stock_list) > 1 else "single"

                aggregator.add_result(
                    ticker=ticker,
                    index=index,
                    stock_mode=stock_mode,
                    image_mode=use_images,
                    pruning_mode=use_pruning,
                    target_type=target_type,
                    model_type="xgb",
                    metrics=results['xgb']['metrics']
                )

                aggregator.add_result(
                    ticker=ticker,
                    index=index,
                    stock_mode=stock_mode,
                    image_mode=use_images,
                    pruning_mode=use_pruning,
                    target_type=target_type,
                    model_type="ridge",
                    metrics=results['ridge']['metrics']
                )

                print(f"✓ Completed {target_label} for {ticker}")

            except Exception as e:
                print(f"⚠ ERROR processing {target_label} for {ticker}: {e}")
                continue

        print(f"\n✓ Test {test_name} completed successfully")

    except Exception as e:
        print(f"\n✗ Test {test_name} failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main execution function."""

    print("="*80)
    print("BATCH PIPELINE RUNNER - SHORT INTEREST PREDICTION")
    print("="*80)
    print(f"Total tickers: {len(TICKERS_CONFIG)}")
    print(f"Tests per ticker: 6 (2 stock modes × 3 configurations)")
    print(f"Total tests: {len(TICKERS_CONFIG) * 6}")
    print("="*80 + "\n")

    aggregator = ResultsAggregator()

    # Run all tests
    for ticker_info in TICKERS_CONFIG:
        ticker = ticker_info["ticker"]
        index = ticker_info["index"]

        # Test 1: Single stock, table, no pruning
        run_single_test(
            ticker=ticker,
            index=index,
            stock_list=[index],
            use_images=False,
            use_pruning=False,
            aggregator=aggregator,
            n_trials=50,  # Reduced for faster testing
            n_passes=2     # Reduced for faster testing
        )

        # Test 2: Multiple stocks, table, no pruning
        run_single_test(
            ticker=ticker,
            index=index,
            stock_list=[index, ticker],
            use_images=False,
            use_pruning=False,
            aggregator=aggregator,
            n_trials=50,
            n_passes=2
        )

        # Test 3: Single stock, image, no pruning
        run_single_test(
            ticker=ticker,
            index=index,
            stock_list=[index],
            use_images=True,
            use_pruning=False,
            aggregator=aggregator,
            n_trials=50,
            n_passes=2
        )

        # Test 4: Multiple stocks, image, no pruning
        run_single_test(
            ticker=ticker,
            index=index,
            stock_list=[index, ticker],
            use_images=True,
            use_pruning=False,
            aggregator=aggregator,
            n_trials=50,
            n_passes=2
        )

        # Test 5: Single stock, table, with pruning
        run_single_test(
            ticker=ticker,
            index=index,
            stock_list=[index],
            use_images=False,
            use_pruning=True,
            aggregator=aggregator,
            n_trials=50,
            n_passes=2
        )

        # Test 6: Multiple stocks, table, with pruning
        run_single_test(
            ticker=ticker,
            index=index,
            stock_list=[index, ticker],
            use_images=False,
            use_pruning=True,
            aggregator=aggregator,
            n_trials=50,
            n_passes=2
        )

        # Save intermediate results after each ticker
        aggregator.save_results(f"intermediate_results_{ticker}.csv")

    # Final results aggregation
    print("\n" + "="*80)
    print("GENERATING FINAL COMPARISON TABLES")
    print("="*80)

    final_df = aggregator.save_results("final_aggregated_results.csv")
    aggregator.create_comparison_table()

    print("\n" + "="*80)
    print("BATCH PROCESSING COMPLETE")
    print("="*80)
    print(f"Results directory: {BATCH_OUTPUT_DIR}")
    print(f"Total tests completed: {len(aggregator.results)}")

if __name__ == "__main__":
    main()
