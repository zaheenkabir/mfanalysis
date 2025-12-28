
import sys
import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")

# Mock Streamlit to avoid errors if imported
import streamlit
def mock_cache(*args, **kwargs):
    if args and callable(args[0]): return args[0]
    return lambda func: func
sys.modules["streamlit"] = type("MockStreamlit", (object,), {"cache_data": mock_cache})()

# Fix path to allow imports
sys.path.append(os.getcwd())

from backend.analytics import get_nav_history, download_benchmark
from screener_app.utils import calculate_period_metrics, get_benchmark_for_category

def debug_metrics():
    # Setup - Axis Short Duration Fund - Direct Plan - Growth
    fund_code = "120835" 
    category = "Debt Scheme - Short Duration Fund"
    
    print(f"--- Debugging Fund: {fund_code} ({category}) ---")
    
    # 1. Fetch NAV
    print("1. Fetching NAV...")
    nav_series = get_nav_history(fund_code)
    print(f"   NAV Length: {len(nav_series)}")
    if not nav_series.empty:
        print(f"   NAV Range: {nav_series.index.min()} to {nav_series.index.max()}")
        print(f"   NAV Index Type: {type(nav_series.index)}")
    
    # 2. Identify Benchmark
    print("\n2. Identifying Benchmark...")
    bench_name = get_benchmark_for_category(category)
    print(f"   Mapped Benchmark: {bench_name}")
    
    # 3. Fetch Benchmark Data
    print("\n3. Fetching Benchmark Data...")
    bench_series = download_benchmark(bench_name, period="max")
    print(f"   Benchmark Length: {len(bench_series)}")
    if not bench_series.empty:
        print(f"   Benchmark Range: {bench_series.index.min()} to {bench_series.index.max()}")
        print(f"   Benchmark Index Type: {type(bench_series.index)}")
        print(f"   Benchmark TZ: {bench_series.index.tz}")
    
    # 4. Calculate Metrics
    print("\n4. Calculating Metrics...")
    
    # Import specific functions
    from screener_app.utils import calculate_alpha, calculate_tracking_error, calculate_information_ratio
    
    alpha = calculate_alpha(nav_series, bench_series, category=category)
    te = calculate_tracking_error(nav_series, bench_series, category=category)
    ir = calculate_information_ratio(nav_series, bench_series, category=category)
    
    print("\n--- Results ---")
    print(f"Alpha: {alpha}")
    print(f"Tracking Error: {te}")
    print(f"Information Ratio: {ir}")

if __name__ == "__main__":
    debug_metrics()
