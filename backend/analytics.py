import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

@st.cache_data
def download_benchmark(ticker="^NSEI", period="max"):
    """
    Downloads benchmark data from Yahoo Finance.
    Returns a Series of closing prices.
    """
    try:
        data = yf.download(ticker, period=period, progress=False)
        if data.empty:
            return pd.Series(dtype=float)
        # yfinance returns MultiIndex columns sometimes, flatten or select 'Close'
        if isinstance(data.columns, pd.MultiIndex):
             # Try to find 'Close'
             if 'Close' in data.columns:
                 series = data['Close']
                 if isinstance(series, pd.DataFrame): # If ticker level exists
                     series = series.iloc[:, 0]
             else:
                 series = data.iloc[:, 0] # Fallback
        else:
             series = data['Close']
        
        series = series.tz_localize(None) # Remove timezone for alignment
        return series
    except Exception as e:
        print(f"Error fetching benchmark: {e}")
        return pd.Series(dtype=float)

def calculate_correlation_matrix(portfolio_df):
    """
    Calculates the correlation matrix of the portfolio.
    Uses pairwise complete observations (verified standard for financial data).
    Returns a DataFrame of correlations (-1 to 1).
    """
    if portfolio_df.empty:
        return pd.DataFrame()
        
    # Calculate daily returns
    # fill_method=None is required for future pandas compatibility
    daily_returns = portfolio_df.pct_change(fill_method=None)
    
    # Calculate correlation (automatically handles NaNs via pairwise deletion)
    correlation_matrix = daily_returns.corr()
    
    return correlation_matrix

def calculate_cagr(series, periods_per_year=252):
    """
    Calculates the Compound Annual Growth Rate (CAGR).
    Assumes numerical series of NAVs.
    """
    if len(series) < 2:
        return 0.0
    
    start_val = series.iloc[0]
    end_val = series.iloc[-1]
    
    if start_val == 0:
        return 0.0
        
    years = (series.index[-1] - series.index[0]).days / 365.25
    if years <= 0:
        return 0.0
        
    return (end_val / start_val) ** (1 / years) - 1

def calculate_volatility(daily_returns, periods_per_year=252):
    """
    Calculates annualized volatility (standard deviation of returns).
    """
    return daily_returns.std() * np.sqrt(periods_per_year)

def calculate_sharpe_ratio(daily_returns, risk_free_rate=0.06, periods_per_year=252):
    """
    Calculates the Sharpe Ratio. 
    Assumes risk_free_rate is annualized (default 6%).
    """
    mean_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()
    
    if std_daily_return == 0:
        return 0.0
        
    # De-annualize risk free rate for daily calculation approximation
    daily_rf = (1 + risk_free_rate) ** (1/periods_per_year) - 1
    
    # Or simpler approximation: risk_free_rate / periods_per_year
    # Using simple approximation for consistency with standard finance libraries often
    sharpe = (mean_daily_return - (risk_free_rate / periods_per_year)) / std_daily_return
    return sharpe * np.sqrt(periods_per_year)

def calculate_max_drawdown_series(series):
    """
    Returns the drawdown series (percentage drop from peak).
    """
    rolling_max = series.cummax()
    drawdown = (series - rolling_max) / rolling_max
    return drawdown

def calculate_max_drawdown(series):
    """
    Calculates Maximum Drawdown (scalar).
    """
    return calculate_max_drawdown_series(series).min()

def calculate_rolling_returns(series, window_years=1):
    """
    Calculates rolling returns for a specific window in years.
    Returns a Series.
    """
    # shift by approx trading days in a year
    window_days = int(window_years * 252)
    if len(series) < window_days:
        return pd.Series(dtype=float)
        
    return series.pct_change(periods=window_days)

def calculate_rolling_returns_stats(fund_series, benchmark_series, window_years=1):
    """
    Calculates comprehensive rolling returns statistics for fund vs benchmark.
    Returns dict with fund and benchmark rolling returns, CAGR, and comparison metrics.
    """
    if fund_series.empty or benchmark_series.empty:
        return None
    
    # Calculate rolling returns
    fund_rolling = calculate_rolling_returns(fund_series, window_years)
    bench_rolling = calculate_rolling_returns(benchmark_series, window_years)
    
    # Remove NaN values
    fund_rolling = fund_rolling.dropna()
    bench_rolling = bench_rolling.dropna()
    
    if fund_rolling.empty or bench_rolling.empty:
        return None
    
    # Calculate statistics
    fund_mean = fund_rolling.mean() * 100
    bench_mean = bench_rolling.mean() * 100
    fund_median = fund_rolling.median() * 100
    bench_median = bench_rolling.median() * 100
    fund_std = fund_rolling.std() * 100
    bench_std = bench_rolling.std() * 100
    
    # Outperformance percentage
    aligned = pd.concat([fund_rolling, bench_rolling], axis=1).dropna()
    if not aligned.empty:
        aligned.columns = ['Fund', 'Benchmark']
        outperformance_pct = (aligned['Fund'] > aligned['Benchmark']).sum() / len(aligned) * 100
    else:
        outperformance_pct = 0
    
    return {
        'fund_rolling': fund_rolling,
        'bench_rolling': bench_rolling,
        'fund_mean': fund_mean,
        'bench_mean': bench_mean,
        'fund_median': fund_median,
        'bench_median': bench_median,
        'fund_std': fund_std,
        'bench_std': bench_std,
        'outperformance_pct': outperformance_pct
    }


def filter_by_period(series, period_str):
    """
    Filters the series by period string: 1Y, 3Y, 5Y, 10Y, Max.
    """
    if period_str == "Max":
        return series
        
    today = series.index[-1]
    start_date = today
    
    if period_str == "1Y":
        start_date = today - pd.DateOffset(years=1)
    elif period_str == "3Y":
        start_date = today - pd.DateOffset(years=3)
    elif period_str == "5Y":
        start_date = today - pd.DateOffset(years=5)
    elif period_str == "10Y":
        start_date = today - pd.DateOffset(years=10)
        
    return series[series.index >= start_date]

def calculate_lumpsum_returns(series, amount=10000, tenure_years=None):
    """
    Calculates lumpsum investment value.
    tenure_years: If specified, limits calculation to this many years from the end
    """
    if series.empty:
        return 0, 0, 0
    
    # Limit series to tenure if specified
    if tenure_years and tenure_years > 0:
        end_date = series.index[-1]
        start_date = end_date - pd.DateOffset(years=tenure_years)
        series = series[series.index >= start_date]
    
    if series.empty:
        return 0, 0, 0
    
    start_nav = series.iloc[0]
    end_nav = series.iloc[-1]
    
    units = amount / start_nav
    current_value = units * end_nav
    abs_return = ((current_value - amount) / amount) * 100
    cagr = calculate_cagr(series) * 100
    
    return current_value, abs_return, cagr

def calculate_sip_returns(series, monthly_amount=2000, tenure_years=None, return_breakdown=False):
    """
    Calculates SIP returns by iterating monthly.
    tenure_years: If specified, limits calculation to this many years from the end
    return_breakdown: If True, returns DataFrame with monthly breakdown
    Returns (Invested Amount, Current Value, Abs Return %, XIRR) or DataFrame if return_breakdown=True
    """
    if series.empty:
        if return_breakdown:
            return pd.DataFrame()
        return 0, 0, 0, 0
    
    # Limit series to tenure if specified
    if tenure_years and tenure_years > 0:
        end_date = series.index[-1]
        start_date = end_date - pd.DateOffset(years=tenure_years)
        series = series[series.index >= start_date]
        
    # Resample to monthly start
    monthly_data = series.resample('MS').first()
    
    if monthly_data.empty:
        if return_breakdown:
            return pd.DataFrame()
        return 0, 0, 0, 0
         
    total_units = 0
    invested_amount = 0
    cash_flows = []
    breakdown_data = []
    
    for date, nav in monthly_data.items():
        if pd.isna(nav): continue
        units = monthly_amount / nav
        total_units += units
        invested_amount += monthly_amount
        cash_flows.append((date, -monthly_amount))
        
        # Track for visualization
        current_val = total_units * nav
        breakdown_data.append({
            'Date': date,
            'Invested': invested_amount,
            'Value': current_val
        })
        
    current_value = total_units * series.iloc[-1]
    cash_flows.append((series.index[-1], current_value))
    
    # Update last value with final NAV
    if breakdown_data:
        breakdown_data[-1]['Value'] = current_value
    
    if return_breakdown:
        return pd.DataFrame(breakdown_data)
    
    abs_return = ((current_value - invested_amount) / invested_amount) * 100 if invested_amount > 0 else 0
    
    try:
        from scipy import optimize
        
        def xnpv(rate, cashflows):
            return sum([cf / (1 + rate) ** ((t - cashflows[0][0]).days / 365.0) for t, cf in cashflows])
            
        def xirr(cashflows):
            try:
                if not cashflows: return 0.0
                return optimize.newton(lambda r: xnpv(r, cashflows), 0.1)
            except:
                return 0.0
                
        cal_xirr = xirr(cash_flows) * 100
    except ImportError:
        cal_xirr = 0.0
        
    return invested_amount, current_value, abs_return, cal_xirr

def calculate_step_up_sip_returns(series, initial_amount=2000, step_up_percent=10, tenure_years=None, return_breakdown=False):
    """
    Calculates Step-up SIP returns.
    Increases monthly amount by step_up_percent every 12 months.
    tenure_years: If specified, limits calculation to this many years from the end
    return_breakdown: If True, returns DataFrame with monthly breakdown
    Returns (Invested Amount, Current Value, Abs Return %, XIRR) or DataFrame if return_breakdown=True
    """
    if series.empty:
        if return_breakdown:
            return pd.DataFrame()
        return 0, 0, 0, 0
    
    # Limit series to tenure if specified
    if tenure_years and tenure_years > 0:
        end_date = series.index[-1]
        start_date = end_date - pd.DateOffset(years=tenure_years)
        series = series[series.index >= start_date]
        
    monthly_data = series.resample('MS').first()
    
    if monthly_data.empty:
        if return_breakdown:
            return pd.DataFrame()
        return 0, 0, 0, 0
         
    total_units = 0
    invested_amount = 0
    cash_flows = []
    breakdown_data = []
    
    current_sip_amount = initial_amount
    month_count = 0
    
    for date, nav in monthly_data.items():
        if pd.isna(nav): continue
        
        # Increase SIP amount every 12 months
        if month_count > 0 and month_count % 12 == 0:
            current_sip_amount = current_sip_amount * (1 + step_up_percent/100)
            
        units = current_sip_amount / nav
        total_units += units
        invested_amount += current_sip_amount
        cash_flows.append((date, -current_sip_amount))
        
        # Track for visualization
        current_val = total_units * nav
        breakdown_data.append({
            'Date': date,
            'Invested': invested_amount,
            'Value': current_val
        })
        
        month_count += 1
        
    current_value = total_units * series.iloc[-1]
    cash_flows.append((series.index[-1], current_value))
    
    # Update last value with final NAV
    if breakdown_data:
        breakdown_data[-1]['Value'] = current_value
    
    if return_breakdown:
        return pd.DataFrame(breakdown_data)
    
    abs_return = ((current_value - invested_amount) / invested_amount) * 100 if invested_amount > 0 else 0
    
    try:
        from scipy import optimize
        def xnpv(rate, cashflows):
            return sum([cf / (1 + rate) ** ((t - cashflows[0][0]).days / 365.0) for t, cf in cashflows])
        def xirr(cashflows):
            try:
                if not cashflows: return 0.0
                return optimize.newton(lambda r: xnpv(r, cashflows), 0.1)
            except:
                return 0.0
        cal_xirr = xirr(cash_flows) * 100
    except ImportError:
        cal_xirr = 0.0
        
    return invested_amount, current_value, abs_return, cal_xirr

def create_weighted_portfolio(fund_series_dict, weights):
    """
    Creates a weighted portfolio from multiple fund NAV series.
    
    Args:
        fund_series_dict: Dict of {fund_name: nav_series}
        weights: Dict of {fund_name: weight_percentage} (should sum to 100)
    
    Returns:
        Portfolio NAV series (weighted combination)
    """
    if not fund_series_dict or not weights:
        return pd.Series()
    
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    if total_weight == 0:
        return pd.Series()
    
    normalized_weights = {k: v/total_weight for k, v in weights.items()}
    
    # Find common date range
    all_series = list(fund_series_dict.values())
    common_start = max([s.index[0] for s in all_series])
    common_end = min([s.index[-1] for s in all_series])
    
    # Align all series to common dates
    aligned_series = {}
    for name, series in fund_series_dict.items():
        aligned = series[(series.index >= common_start) & (series.index <= common_end)]
        aligned_series[name] = aligned
    
    # Create DataFrame from aligned series
    portfolio_df = pd.DataFrame(aligned_series)
    
    # Calculate weighted portfolio NAV
    # Start with base value of 100 for each fund, then track weighted performance
    portfolio_nav = pd.Series(0.0, index=portfolio_df.index)
    
    for fund_name, weight in normalized_weights.items():
        if fund_name in portfolio_df.columns:
            # Normalize each fund to start at 100, then apply weight
            fund_normalized = (portfolio_df[fund_name] / portfolio_df[fund_name].iloc[0]) * 100
            portfolio_nav += fund_normalized * weight
    
    return portfolio_nav

def calculate_beta_alpha(fund_returns, benchmark_returns, risk_free_rate=0.06):
    """
    Calculates Beta and Alpha (Jensen's Alpha).
    Expects aligned daily returns series.
    """
    # Align data
    aligned = pd.concat([fund_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty or len(aligned) < 30:
        return 0.0, 0.0, 0.0

    aligned.columns = ['Fund', 'Benchmark']
    
    # Covariance for Beta
    covariance = np.cov(aligned['Fund'], aligned['Benchmark'])[0][1]
    variance = np.var(aligned['Benchmark'])
    
    if variance == 0:
        return 0.0, 0.0, 0.0
        
    beta = covariance / variance
    
    daily_rf = (1 + risk_free_rate) ** (1/252) - 1
    
    # Alpha calculation (annualized)
    # R_p = alpha + beta * (R_m - R_f) + R_f
    # alpha = R_p - (R_f + beta * (R_m - R_f))
    
    mean_fund_return = aligned['Fund'].mean() * 252
    mean_bench_return = aligned['Benchmark'].mean() * 252
    
    # Simple Annualized Alpha
    alpha = mean_fund_return - (risk_free_rate + beta * (mean_bench_return - risk_free_rate))
    
    # R-Squared
    correlation = aligned['Fund'].corr(aligned['Benchmark'])
    r_squared = correlation ** 2
    
    return beta, alpha, r_squared

def calculate_capture_ratios(fund_returns, benchmark_returns):
    """
    Calculates Upside and Downside Capture Ratios.
    """
    aligned = pd.concat([fund_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return 0.0, 0.0

    aligned.columns = ['Fund', 'Benchmark']
    
    # Upside
    upside_mask = aligned['Benchmark'] > 0
    if upside_mask.any():
        upside_fund = (1 + aligned.loc[upside_mask, 'Fund']).prod() 
        upside_bench = (1 + aligned.loc[upside_mask, 'Benchmark']).prod()
        # Annualized geometric return during upside periods
        # Simpler approach: Sum of returns or Avg returns
        # Standard approach: (Fund CAGR during Up / Bench CAGR during Up) * 100
        
        # Taking geometric mean return for simplicity in code
        up_period_count = upside_mask.sum()
        up_fund_geo = (upside_fund ** (1/up_period_count)) - 1 if up_period_count > 0 else 0
        up_bench_geo = (upside_bench ** (1/up_period_count)) - 1 if up_period_count > 0 else 0
        
        upside_capture = (up_fund_geo / up_bench_geo * 100) if up_bench_geo != 0 else 0
    else:
        upside_capture = 0.0
        
    # Downside
    downside_mask = aligned['Benchmark'] < 0
    if downside_mask.any():
        downside_fund = (1 + aligned.loc[downside_mask, 'Fund']).prod()
        downside_bench = (1 + aligned.loc[downside_mask, 'Benchmark']).prod()
        
        down_period_count = downside_mask.sum()

        # fixing math
        down_fund_geo = (downside_fund ** (1/down_period_count)) - 1 if down_period_count > 0 else 0
        down_bench_geo = (downside_bench ** (1/down_period_count)) - 1 if down_period_count > 0 else 0

        downside_capture = (down_fund_geo / down_bench_geo * 100) if down_bench_geo != 0 else 0
    else:
        downside_capture = 0.0
        
    return upside_capture, downside_capture


def get_fund_metrics(nav_series, benchmark_series=None):
    """
    Returns a dictionary of all key metrics for a given NAV series.
    Expects nav_series index to be DatetimeIndex.
    """
    if nav_series.empty:
        return {}
        
    nav_series = nav_series.sort_index()
    daily_returns = nav_series.pct_change().dropna()
    
    metrics = {
        "CAGR": calculate_cagr(nav_series),
        "Volatility": calculate_volatility(daily_returns),
        "Sharpe Ratio": calculate_sharpe_ratio(daily_returns),
        "Max Drawdown": calculate_max_drawdown(nav_series)
    }
    
    if benchmark_series is not None and not benchmark_series.empty:
        benchmark_returns = benchmark_series.pct_change().dropna()
        # Align dates
        beta, alpha, r2 = calculate_beta_alpha(daily_returns, benchmark_returns)
        up_cap, down_cap = calculate_capture_ratios(daily_returns, benchmark_returns)
        
        metrics["Alpha"] = alpha
        metrics["Beta"] = beta
        metrics["R-Squared"] = r2
        metrics["Upside Capture"] = up_cap
        metrics["Downside Capture"] = down_cap
        
    return metrics

def run_monte_carlo_simulation(historical_nav, n_simulations=1000, time_horizon_years=5, initial_investment=None):
    """
    Runs a Monte Carlo simulation using Geometric Brownian Motion (GBM).
    
    Args:
        historical_nav (pd.Series): Historical NAV data.
        n_simulations (int): Number of paths to simulate.
        time_horizon_years (int): Number of years to project.
        initial_investment (float): Optional investment amount to scale projections.
        
    Returns:
        dict: Contains 'simulation_df' (paths), 'summary_stats' (percentiles), and 'projected_dates'.
    """
    if historical_nav.empty:
        return None

    # Calculate daily returns
    daily_returns = historical_nav.pct_change().dropna()
    
    # Calculate drift and volatility
    # Drift = mu - 0.5 * sigma^2
    mu = daily_returns.mean()
    sigma = daily_returns.std()
    drift = mu - 0.5 * sigma**2
    
    # Simulation parameters
    days = int(time_horizon_years * 252)
    last_price = historical_nav.iloc[-1]
    
    # Generate random shocks: standard normal distribution
    # Shape: (days, n_simulations)
    daily_shocks = np.random.normal(0, 1, (days, n_simulations))
    
    # Calculate price paths
    # price_t = price_{t-1} * exp(drift + sigma * shock)
    # cumsum(drift + sigma * shock) is faster
    
    daily_drift = drift
    daily_vol = sigma * daily_shocks
    
    # Cumulative returns
    cumulative_returns = np.cumsum(daily_drift + daily_vol, axis=0)
    
    # Project prices
    projected_prices = last_price * np.exp(cumulative_returns)
    
    # Add initial price at day 0
    projected_prices = np.vstack([np.full((1, n_simulations), last_price), projected_prices])
    
    # Scale by initial investment if provided
    # If initial_investment is given, we treat 'projected_prices' as the value of that investment
    # relative to the starting NAV.
    # Actually, simpler: Calculate the multiplier (growth factor) and apply to investment amount.
    
    if initial_investment is not None:
        growth_factors = projected_prices / last_price
        projected_values = initial_investment * growth_factors
        # Use values for stats, but keep prices for CAGR calculation reference?
        # Actually CAGR calculation depends on relative growth, so growth factors are enough.
        # Let's switch projected_prices to mean PROJECTED VALUES
        projected_prices = projected_values
        last_price = initial_investment # Update start baseline
    
    # Create date index
    start_date = historical_nav.index[-1]
    projected_dates = pd.date_range(start=start_date, periods=days+1, freq='B') # Business days
    
    # Ensure dates match length (sometimes date_range might overflow slightly or underflow depending on calendar)
    if len(projected_dates) > len(projected_prices):
        projected_dates = projected_dates[:len(projected_prices)]
    elif len(projected_dates) < len(projected_prices):
        # Fallback to simple addition if freq='B' issues arise
        projected_dates = [start_date + pd.Timedelta(days=i) for i in range(len(projected_prices))]

    # Calculate percentiles for confidence intervals
    # Axis 1 = across simulations
    p5 = np.percentile(projected_prices, 5, axis=1)
    p50 = np.percentile(projected_prices, 50, axis=1)
    p95 = np.percentile(projected_prices, 95, axis=1)
    
    # Calculate Expected Value (Mean path)
    mean_path = np.mean(projected_prices, axis=1)
    
    # Determine "Optimistic" vs "Pessimistic" returns (CAGR)
    # End value / Start value
    start_val = last_price
    
    def get_cagr(end_val, years):
        return ((end_val / start_val) ** (1/years) - 1) * 100
        
    stats = {
        'current_price': last_price,
        'expected_price': mean_path[-1],
        'optimistic_price': p95[-1],
        'pessimistic_price': p5[-1],
        'expected_cagr': get_cagr(mean_path[-1], time_horizon_years),
        'optimistic_cagr': get_cagr(p95[-1], time_horizon_years),
        'pessimistic_cagr': get_cagr(p5[-1], time_horizon_years),
        'volatility': sigma * np.sqrt(252) * 100
    }
    
    return {
        'dates': projected_dates,
        'p5': p5,
        'p50': p50,
        'p95': p95,
        'mean': mean_path,
        'paths': projected_prices[:, :20], # Return first 20 paths for visualization "spaghetti"
        'stats': stats
    }

# --- PORTFOLIO INTELLIGENCE & OPTIMIZATION ---

def calculate_correlation_matrix(portfolio_df):
    """
    Calculates the correlation matrix of the portfolio's daily returns.
    Input: DataFrame of aligned NAVs (prices) or Returns.
    """
    if portfolio_df.empty:
        return pd.DataFrame()
        
    # Use fill_method=None to avoid text-book warning and let corr handle NaNs pairwise
    daily_returns = portfolio_df.pct_change(fill_method=None)
    correlation_matrix = daily_returns.corr()
    return correlation_matrix

def simulate_efficient_frontier(portfolio_df, num_portfolios=2000, risk_free_rate=0.06):
    """
    Simulates random portfolio weights to generate an Efficient Frontier.
    Returns a DataFrame with columns: [Return, Volatility, Sharpe, Weights]
    """
    if portfolio_df.empty:
        return pd.DataFrame()
        
    daily_returns = portfolio_df.pct_change(fill_method=None).dropna()
    mean_daily_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    
    results = []
    
    for _ in range(num_portfolios):
        weights = np.random.random(len(portfolio_df.columns))
        weights /= np.sum(weights)
        
        # Portfolio Return (Annualized)
        port_return = np.sum(mean_daily_returns * weights) * 252
        
        # Portfolio Volatility (Annualized)
        port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        
        # Sharpe Ratio
        sharpe_ratio = (port_return - risk_free_rate) / port_volatility if port_volatility > 0 else 0
        
        results.append({
            'Return': port_return,
            'Volatility': port_volatility,
            'Sharpe': sharpe_ratio,
            'Weights': weights  # Store weights for reference
        })
        
    return pd.DataFrame(results)

def optimize_portfolio_weights(portfolio_df, risk_free_rate=0.06, objective='max_sharpe'):
    """
    Optimizes portfolio weights using scipy.optimize.
    objective: 'max_sharpe' or 'min_volatility'
    Returns: dictionary {fund_name: optimal_weight}
    """
    if portfolio_df.empty:
        return {}
        
    try:
        from scipy.optimize import minimize
    except ImportError:
        return {}
        
    daily_returns = portfolio_df.pct_change(fill_method=None).dropna()
    mean_daily_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    num_assets = len(portfolio_df.columns)
    
    # Helper functions for optimization
    def get_ret_vol_sr(weights):
        weights = np.array(weights)
        ret = np.sum(mean_daily_returns * weights) * 252
        vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sr = (ret - risk_free_rate) / vol
        return np.array([ret, vol, sr])

    def neg_sharpe(weights):
        return -get_ret_vol_sr(weights)[2]

    def volatility(weights):
        return get_ret_vol_sr(weights)[1]

    # Constraints: Sum of weights = 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    # Bounds: 5% <= weight <= 50% for each asset (ensures diversification and no zero allocations)
    bounds = tuple((0.05, 0.50) for _ in range(num_assets))
    
    # Initial Guess: Equal weights
    init_guess = num_assets * [1. / num_assets,]
    
    if objective == 'max_sharpe':
        result = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    elif objective == 'min_volatility':
        result = minimize(volatility, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)
    else:
        return {}
        
    if result.success:
        optimal_weights = result.x
        return dict(zip(portfolio_df.columns, optimal_weights))
    else:
        return {}

# --- REPORTING & INSIGHTS ---

def generate_portfolio_summary(metrics):
    """
    Generates a natural language summary of the portfolio based on metrics.
    metrics: dict containing 'cagr', 'volatility', 'sharpe', 'beta'.
    """
    summary = []
    
    # Risk/Return Profile
    if metrics.get('sharpe', 0) > 2.0:
        summary.append("🌟 **Excellent Risk-Adjusted Returns**: Your portfolio generates exceptional returns for every unit of risk taken.")
    elif metrics.get('sharpe', 0) > 1.0:
        summary.append("✅ **Solid Performance**: The portfolio has a healthy risk-return balance.")
    else:
        summary.append("⚠️ **Risk Warning**: Returns are currently low relative to the volatility. Consider optimizing.")

    # Volatility Check
    if metrics.get('volatility', 0) > 0.20:
        summary.append("🔥 **High Volatility**: This portfolio is aggressive. Expect significant swings in value.")
    elif metrics.get('volatility', 0) < 0.10:
        summary.append("🛡️ **Defensive Stance**: This portfolio is stable and likely preserving capital well.")
        
    # Beta Check (Market Sensitivity)
    beta = metrics.get('beta', 1.0)
    if beta > 1.2:
        summary.append("📈 **Aggressive Growth**: The portfolio moves significantly more than the benchmark.")
    elif beta < 0.8:
        summary.append("📉 **Low Correlation**: The portfolio is less sensitive to market crashes.")
        
    return " ".join(summary)

def calculate_required_sip(target_amount, years, expected_return_annual):
    """
    Calculates the monthly SIP required to reach a target amount.
    """
    if years <= 0 or expected_return_annual <= 0:
        return 0
    
    months = years * 12
    r_monthly = expected_return_annual / 12
    
    # Future Value Formula: FV = P * ((1+r)^n - 1) / r * (1+r)
    # So P = FV / (((1+r)^n - 1) / r * (1+r))
    
    factor = ((1 + r_monthly)**months - 1) / r_monthly * (1 + r_monthly)
    required_sip = target_amount / factor
    return required_sip

# --- SCENARIO ANALYSIS (STRESS TESTING) ---

def simulate_market_scenario(portfolio_series, benchmark_series, scenario_config):
    """
    Simulates portfolio performance during a specific market event.
    
    Args:
        portfolio_series: Portfolio NAV series
        benchmark_series: Benchmark series
        scenario_config: Dict with 'name', 'start_date', 'end_date'
    
    Returns:
        Dict with scenario metrics
    """
    import pandas as pd
    
    start_date = pd.to_datetime(scenario_config['start_date'])
    end_date = pd.to_datetime(scenario_config['end_date'])
    
    # Filter to scenario period
    portfolio_scenario = portfolio_series[(portfolio_series.index >= start_date) & (portfolio_series.index <= end_date)]
    benchmark_scenario = benchmark_series[(benchmark_series.index >= start_date) & (benchmark_series.index <= end_date)]
    
    if portfolio_scenario.empty or benchmark_scenario.empty:
        return {
            'scenario_name': scenario_config['name'],
            'portfolio_return': 0,
            'benchmark_return': 0,
            'portfolio_max_drawdown': 0,
            'benchmark_max_drawdown': 0,
            'days_to_recover': None,
            'data_available': False
        }
    
    # Calculate returns during scenario
    portfolio_return = (portfolio_scenario.iloc[-1] / portfolio_scenario.iloc[0] - 1) * 100
    benchmark_return = (benchmark_scenario.iloc[-1] / benchmark_scenario.iloc[0] - 1) * 100
    
    # Calculate max drawdown during scenario
    portfolio_dd = calculate_max_drawdown(portfolio_scenario) * 100
    benchmark_dd = calculate_max_drawdown(benchmark_scenario) * 100
    
    # Calculate recovery time (days to break even after max drawdown)
    portfolio_cummax = portfolio_scenario.cummax()
    portfolio_drawdown_series = (portfolio_scenario - portfolio_cummax) / portfolio_cummax
    
    # Find the date of maximum drawdown
    max_dd_date = portfolio_drawdown_series.idxmin()
    
    # Find recovery (first date after max DD where portfolio >= previous peak)
    recovery_series = portfolio_series[portfolio_series.index > max_dd_date]
    peak_value = portfolio_cummax.loc[max_dd_date]
    
    recovery_dates = recovery_series[recovery_series >= peak_value]
    days_to_recover = None
    if not recovery_dates.empty:
        recovery_date = recovery_dates.index[0]
        days_to_recover = (recovery_date - max_dd_date).days
    
    return {
        'scenario_name': scenario_config['name'],
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'portfolio_return': portfolio_return,
        'benchmark_return': benchmark_return,
        'portfolio_max_drawdown': portfolio_dd,
        'benchmark_max_drawdown': benchmark_dd,
        'days_to_recover': days_to_recover,
        'data_available': True,
        'outperformed_benchmark': portfolio_return > benchmark_return
    }

def get_predefined_scenarios():
    """
    Returns a list of predefined market crash scenarios.
    """
    return [
        {
            'name': 'Russia-Ukraine War (2022)',
            'start_date': '2022-02-20',
            'end_date': '2022-03-15',
            'description': 'Russia invasion of Ukraine caused ~15% market correction'
        },
        {
            'name': 'US Banking Crisis (2023)',
            'start_date': '2023-03-01',
            'end_date': '2023-03-31',
            'description': 'Silicon Valley Bank collapse triggered banking sector panic'
        },
        {
            'name': 'COVID-19 Crash (2020)',
            'start_date': '2020-02-01',
            'end_date': '2020-03-31',
            'description': 'Global pandemic caused ~35% market crash'
        },
        {
            'name': 'IL&FS Crisis (2018)',
            'start_date': '2018-09-01',
            'end_date': '2018-10-31',
            'description': 'IL&FS default triggered NBFC liquidity crisis in India'
        },
        {
            'name': 'Demonetization (2016)',
            'start_date': '2016-11-01',
            'end_date': '2016-12-31',
            'description': 'India demonetization impact on markets'
        },
        {
            'name': 'Taper Tantrum (2013)',
            'start_date': '2013-05-01',
            'end_date': '2013-08-31',
            'description': 'Fed taper announcement caused ~10% correction'
        },
        {
            'name': '2008 Financial Crisis',
            'start_date': '2008-09-01',
            'end_date': '2009-03-31',
            'description': 'Subprime mortgage crisis led to ~50% crash'
        }
    ]
