import pandas as pd
import numpy as np
import yfinance as yf
import streamlit as st

@st.cache_data(ttl=3600)  # Add TTL to avoid caching failures indefinitely
def download_benchmark(ticker="NIFTY 50", period="max"):
    """
    Downloads benchmark data using nsepython for Indian indices,
    or yfinance for ETFs/Global indices.
    ticker: Can be NSE name (e.g., 'NIFTY 50') or YF symbol (e.g., '^NSEI', 'GOLDBEES.NS')
    """
    from nsepython import index_history
    from datetime import date, datetime
    import yfinance as yf
    
    # Map friendly names / standard NSE names to actual Yahoo Finance tickers for fallback
    ticker_map_yf = {
        "NIFTY 50": "^NSEI",
        "NIFTY MIDCAP 50": "^NSEMDCP50",
        "NIFTY SMALLCAP 50": "^CNXSC",  # Fallback to Smallcap 100 if 50 is missing on YF
        "NIFTY SMALLCAP 100": "^CNXSC",
        "NIFTY BANK": "^NSEBANK",
        "GOLD": "GOLDBEES.NS",
        "SILVER": "SILVERBEES.NS"
    }
    
    # Internal mapping for NSE names (for nsepython)
    ticker_map_nse = {
        "^NSEI": "NIFTY 50",
        "^NSEMDCP50": "NIFTY MIDCAP 50",
        "^CNXSC": "NIFTY SMALLCAP 100",
        "^NSEBANK": "NIFTY BANK"
    }
    
    symbol = ticker_map_nse.get(ticker, ticker)
    
    # Use yfinance for Bees or known YF tickers
    if symbol.endswith(".NS") or symbol.startswith("^") or symbol.endswith("=F"):
        try:
            # Map period to yf format
            yf_period_map = {
                "max": "max",
                "10Y": "10y",
                "5Y": "5y",
                "3Y": "3y",
                "1Y": "1y"
            }
            yf_period = yf_period_map.get(period, "max") # Use max as default for full history
            
            data = yf.download(symbol, period=yf_period, interval="1d", progress=False)
            if data is None or data.empty:
                return pd.Series(dtype=float)
            
            # Use 'Close' column.
            if isinstance(data.columns, pd.MultiIndex):
                series = data['Close'][symbol]
            else:
                series = data['Close']
                
            series = pd.to_numeric(series, errors='coerce').dropna()
            
            # CRITICAL: Strip timezone to align with Mutual Fund data (which is naive)
            if series.index.tz is not None:
                series.index = series.index.tz_localize(None)
                
            return series
        except Exception as e:
            print(f"Error fetching benchmark {symbol} via yfinance: {e}")
            return pd.Series(dtype=float)

    # Default to nsepython for standard NSE indices
    try:
        # Determine date range
        end_date = date.today().strftime("%d-%b-%Y")
        
        # Calculate start date based on period
        years_back = 15
        if period == "10Y": years_back = 10
        elif period == "5Y": years_back = 5
        elif period == "3Y": years_back = 3
        elif period == "1Y": years_back = 1
        
        start_date = (date.today() - pd.DateOffset(years=years_back)).strftime("%d-%b-%Y")

        # Standardize symbol for nsepython (all caps)
        nse_symbol = symbol.upper()
        data = index_history(nse_symbol, start_date, end_date)
        
        if data is None or data.empty:
            # Fallback to yfinance if nsepython fails for this symbol
            yf_symbol = ticker_map_yf.get(nse_symbol, nse_symbol + ".NS")
            return download_benchmark(yf_symbol, period)
            
        # Ensure we have the right columns
        if 'HistoricalDate' not in data.columns and 'Date' in data.columns:
            data['HistoricalDate'] = data['Date']
            
        data['Date'] = pd.to_datetime(data['HistoricalDate'], format='mixed') # use mixed format for safety
        data = data.set_index('Date').sort_index()
        
        # Check for multiple possible 'close' column names
        close_col = 'CLOSE' if 'CLOSE' in data.columns else ('Close' if 'Close' in data.columns else None)
        if close_col:
            series = pd.to_numeric(data[close_col], errors='coerce')
            return series.dropna()
        else:
            # Try to find any column with NAV-like values
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if not numeric_cols.empty:
                return data[numeric_cols[0]].dropna()
            
        return pd.Series(dtype=float)
    except Exception as e:
        print(f"Error fetching benchmark {symbol} via nsepython: {e}")
        # Final fallback to yfinance
        yf_symbol = ticker_map_yf.get(symbol.upper(), symbol.upper() + ".NS")
        if yf_symbol != symbol:
            return download_benchmark(yf_symbol, period)
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
    excess_returns = daily_returns - (risk_free_rate / periods_per_year)
    if daily_returns.std() == 0:
        return 0.0
    return (excess_returns.mean() / daily_returns.std()) * np.sqrt(periods_per_year)

def generate_gold_insights(portfolio_nav, gold_nav):
    """
    Generates specific insights about Gold's role as a hedge for the portfolio.
    """
    if portfolio_nav.empty or gold_nav.empty:
        return []
    
    # Align data
    common = pd.concat([portfolio_nav, gold_nav], axis=1).dropna()
    if common.empty:
        return []
    
    common.columns = ['Portfolio', 'Gold']
    returns = common.pct_change().dropna()
    
    correlation = returns['Portfolio'].corr(returns['Gold'])
    
    insights = []
    
    # Correlation insight
    if correlation < 0.2:
        insights.append({
            "type": "positive",
            "icon": "🛡️",
            "title": "Strong Hedge",
            "text": f"Gold has a very low correlation ({correlation:.2f}) with your portfolio, providing excellent diversification during equity corrections."
        })
    elif correlation < 0.6:
        insights.append({
            "type": "neutral",
            "icon": "⚖️",
            "title": "Moderate Hedge",
            "text": f"Gold shows moderate correlation ({correlation:.2f}), offering some protection during market stress."
        })
    else:
        insights.append({
            "type": "warning",
            "icon": "⚠️",
            "title": "Limited Protection",
            "text": f"Gold is moving somewhat in sync with your portfolio (corr: {correlation:.2f}), reducing its effectiveness as a disaster hedge."
        })
        
    # Crisis performance (Beta during down days)
    down_days = returns[returns['Portfolio'] < -0.01]
    if not down_days.empty:
        gold_on_down_days = down_days['Gold'].mean() * 100
        if gold_on_down_days > 0:
            insights.append({
                "type": "positive",
                "icon": "✨",
                "title": "Crash Buffer",
                "text": f"Historically, when your portfolio dropped >1%, Gold rose by {gold_on_down_days:.2f}% on average, buffering your losses."
            })
            
    return insights

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

def calculate_lumpsum_returns(series, amount=10000, tenure_years=None, mode='historical'):
    """
    Calculates lumpsum investment value.
    """
    if series.empty:
        return 0, 0, 0, 0
    
    actual_tenure_years = tenure_years
    if tenure_years:
        start_date = series.index[-1] - pd.DateOffset(years=tenure_years)
        if start_date < series.index[0]:
            actual_tenure_years = (series.index[-1] - series.index[0]).days / 365.25
            start_date = series.index[0]
        series = series[series.index >= start_date]
    else:
        actual_tenure_years = (series.index[-1] - series.index[0]).days / 365.25
    
    if series.empty:
        return 0, 0, 0, 0
    
    cagr_val = calculate_cagr(series)
    
    if mode == 'theoretical':
        n = tenure_years if tenure_years else actual_tenure_years
        current_value = amount * ((1 + cagr_val) ** n)
        abs_return = ((current_value - amount) / amount) * 100
        return current_value, abs_return, cagr_val * 100, actual_tenure_years

    start_nav = series.iloc[0]
    end_nav = series.iloc[-1]
    units = amount / start_nav
    current_value = units * end_nav
    abs_return = ((current_value - amount) / amount) * 100
    
    return current_value, abs_return, cagr_val * 100, actual_tenure_years

def calculate_xirr(cash_flows):
    """
    Calculates XIRR using scipy.optimize.newton with multiple guesses for robustness.
    Cashflows: list of (date, amount) tuples.
    """
    if not cash_flows:
        return 0.0
        
    try:
        from scipy import optimize
        
        def xnpv(rate, cashflows):
            # Avoid division by zero or complex numbers if rate <= -1
            if rate <= -1:
                return float('inf')
            t0 = cashflows[0][0]
            return sum([cf / ((1 + rate) ** ((t - t0).days / 365.0)) for t, cf in cashflows])
            
        # Try multiple guesses to ensure convergence
        for guess in [0.1, -0.1, 0.2, -0.2, 0.5, -0.5]:
            try:
                return optimize.newton(lambda r: xnpv(r, cash_flows), guess, maxiter=50)
            except (RuntimeError, OverflowError):
                continue
                
        return 0.0
    except ImportError:
        return 0.0


def calculate_sip_returns(series, monthly_amount=2000, tenure_years=None, return_breakdown=False, mode='historical'):
    """
    Calculates SIP returns.
    mode: 'historical' (backtest) or 'theoretical' (formula based on CAGR/requested tenure)
    """
    if series.empty:
        if return_breakdown: return pd.DataFrame()
        return 0, 0, 0, 0, 0
    
    actual_tenure_years = tenure_years
    if tenure_years:
        start_date = series.index[-1] - pd.DateOffset(years=tenure_years)
        if start_date < series.index[0]:
            actual_tenure_years = (series.index[-1] - series.index[0]).days / 365.25
            start_date = series.index[0]
        series = series[series.index >= start_date]
    else:
        actual_tenure_years = (series.index[-1] - series.index[0]).days / 365.25

    if mode == 'theoretical':
        # Formula: FV = P * [((1+r)^n - 1) / r] * (1+r)
        cagr = calculate_cagr(series)
        r = ((1 + cagr) ** (1/12)) - 1
        n = int(tenure_years * 12) if tenure_years else int(actual_tenure_years * 12)
        
        invested_amount = monthly_amount * n
        current_value = monthly_amount * (((1+r)**n - 1) / r) * (1+r) if r > 0 else invested_amount
        abs_return = ((current_value - invested_amount) / invested_amount) * 100 if invested_amount > 0 else 0
        
        if return_breakdown:
            # Generate a smooth curve for theoretical breakdown
            dates = pd.date_range(end=series.index[-1], periods=n, freq='MS')
            breakdown = []
            for i in range(1, n + 1):
                val = monthly_amount * (((1+r)**i - 1) / r) * (1+r) if r > 0 else monthly_amount * i
                breakdown.append({'Date': dates[i-1], 'Invested': monthly_amount * i, 'Value': val})
            return pd.DataFrame(breakdown)

        return invested_amount, current_value, abs_return, cagr * 100, actual_tenure_years

    # Historical (Unit-based backtest)
    monthly_data = series.resample('MS').first()
    if monthly_data.empty:
        if return_breakdown: return pd.DataFrame()
        return 0, 0, 0, 0, 0
         
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
        
        current_val = total_units * nav
        breakdown_data.append({'Date': date, 'Invested': invested_amount, 'Value': current_val})
        
    current_value = total_units * series.iloc[-1]
    last_date = series.index[-1]
    cash_flows.append((last_date, current_value))
    
    if breakdown_data:
        breakdown_data[-1]['Value'] = current_value
        breakdown_data[-1]['Date'] = last_date
    
    if return_breakdown: return pd.DataFrame(breakdown_data)
    
    abs_return = ((current_value - invested_amount) / invested_amount) * 100 if invested_amount > 0 else 0
    cal_xirr = calculate_xirr(cash_flows) * 100
    return invested_amount, current_value, abs_return, cal_xirr, actual_tenure_years


def calculate_step_up_sip_returns(series, initial_amount=2000, step_up_percent=10, tenure_years=None, return_breakdown=False, mode='historical'):
    """
    Calculates Step-up SIP returns.
    """
    if series.empty:
        if return_breakdown: return pd.DataFrame()
        return 0, 0, 0, 0, 0
    
    actual_tenure_years = tenure_years
    if tenure_years:
        start_date = series.index[-1] - pd.DateOffset(years=tenure_years)
        if start_date < series.index[0]:
            actual_tenure_years = (series.index[-1] - series.index[0]).days / 365.25
            start_date = series.index[0]
        series = series[series.index >= start_date]
    else:
        actual_tenure_years = (series.index[-1] - series.index[0]).days / 365.25

    if mode == 'theoretical':
        cagr = calculate_cagr(series)
        r = ((1 + cagr) ** (1/12)) - 1
        n = int(tenure_years * 12) if tenure_years else int(actual_tenure_years * 12)
        
        total_invested = 0
        total_value = 0
        current_sip = initial_amount
        breakdown = []
        
        for i in range(1, n + 1):
            if i > 1 and (i - 1) % 12 == 0:
                current_sip *= (1 + step_up_percent/100)
            
            total_invested += current_sip
            # Each installment grows for (n - i + 1) months
            months_left = n - i + 1
            installment_fv = current_sip * ((1 + r) ** months_left)
            total_value += installment_fv
            
            if return_breakdown:
                breakdown.append({'Date': series.index[-1] - pd.DateOffset(months=n-i), 'Invested': total_invested, 'Value': total_value})
        
        if return_breakdown: return pd.DataFrame(breakdown)
        
        abs_ret = ((total_value - total_invested) / total_invested) * 100 if total_invested > 0 else 0
        return total_invested, total_value, abs_ret, cagr * 100, actual_tenure_years

    # Historical
    monthly_data = series.resample('MS').first()
    if monthly_data.empty:
        if return_breakdown: return pd.DataFrame()
        return 0, 0, 0, 0, 0
         
    total_units = 0
    invested_amount = 0
    cash_flows = []
    breakdown_data = []
    current_sip_amount = initial_amount
    month_count = 0
    
    for date, nav in monthly_data.items():
        if pd.isna(nav): continue
        if month_count > 0 and month_count % 12 == 0:
            current_sip_amount = current_sip_amount * (1 + step_up_percent/100)
            
        units = current_sip_amount / nav
        total_units += units
        invested_amount += current_sip_amount
        cash_flows.append((date, -current_sip_amount))
        
        current_val = total_units * nav
        breakdown_data.append({'Date': date, 'Invested': invested_amount, 'Value': current_val})
        month_count += 1
        
    current_value = total_units * series.iloc[-1]
    last_date = series.index[-1]
    cash_flows.append((last_date, current_value))
    
    if breakdown_data:
        breakdown_data[-1]['Value'] = current_value
        breakdown_data[-1]['Date'] = last_date
    
    if return_breakdown: return pd.DataFrame(breakdown_data)
    
    abs_return = ((current_value - invested_amount) / invested_amount) * 100 if invested_amount > 0 else 0
    cal_xirr = calculate_xirr(cash_flows) * 100
    return invested_amount, current_value, abs_return, cal_xirr, actual_tenure_years

# Portfolio creation function - updated
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
    Standardized to handle edge cases and zero returns.
    """
    aligned = pd.concat([fund_returns, benchmark_returns], axis=1).dropna()
    if aligned.empty:
        return 0.0, 0.0

    aligned.columns = ['Fund', 'Benchmark']
    
    # Upside
    upside_mask = aligned['Benchmark'] > 0
    if upside_mask.any():
        up_fund = aligned.loc[upside_mask, 'Fund']
        up_bench = aligned.loc[upside_mask, 'Benchmark']
        
        # Calculate geometric average returns
        # Using CAGR-style math: (Prod(1+r))^(1/n) - 1
        up_fund_geo = (1 + up_fund).prod()**(1/len(up_fund)) - 1
        up_bench_geo = (1 + up_bench).prod()**(1/len(up_bench)) - 1
        
        upside_capture = (up_fund_geo / up_bench_geo * 100) if up_bench_geo != 0 else 0
    else:
        upside_capture = 0.0
        
    # Downside
    downside_mask = aligned['Benchmark'] < 0
    if downside_mask.any():
        down_fund = aligned.loc[downside_mask, 'Fund']
        down_bench = aligned.loc[downside_mask, 'Benchmark']
        
        down_fund_geo = (1 + down_fund).prod()**(1/len(down_fund)) - 1
        down_bench_geo = (1 + down_bench).prod()**(1/len(down_bench)) - 1

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
        'stats': stats,
        'end_distribution': projected_prices[-1, :] # All end values for probability analysis
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

def get_fund_investment_insights(metrics):
    """
    Generates human-readable investment insights based on fund metrics.
    """
    insights = []
    
    # 1. ALPHA: Skill/Value Add
    alpha = metrics.get('Alpha', 0)
    if alpha > 0.05:
        # High Alpha
        insights.append({
            "type": "positive",
            "icon": "🚀",
            "title": "Wealth Creator",
            "text": f"This fund is significantly outperforming its benchmark, generating an impressive alpha of {alpha*100:.2f}%."
        })
    elif alpha > 0:
        insights.append({
            "type": "positive",
            "icon": "📈",
            "title": "Consistent Alpha",
            "text": f"The fund manager has successfully beaten the market by {alpha*100:.2f}% annually."
        })
    else:
        insights.append({
            "type": "neutral",
            "icon": "⚖️",
            "title": "Benchmark Linked",
            "text": "The fund closely follows its benchmark. It's safe but not significantly outperforming the market index."
        })

    # 2. SHARPE ratio: Efficiency
    sharpe = metrics.get('Sharpe Ratio', 0)
    if sharpe > 1.5:
        insights.append({
            "type": "positive",
            "icon": "🎯",
            "title": "Excellent Efficiency",
            "text": "With a Sharpe ratio of {sharpe:.2f}, this fund delivers high returns with relatively lower risk."
        })
    elif sharpe > 1.0:
        insights.append({
            "type": "neutral",
            "icon": "✅",
            "title": "Good Risk-Return",
            "text": "The fund provides a healthy balance between returns and the risk taken."
        })
    else:
        insights.append({
            "type": "warning",
            "icon": "⚠️",
            "title": "Efficiency Watch",
            "text": "The risk-adjusted returns are lower than ideal. The volatility might be high for the returns generated."
        })

    # 3. VOLATILITY/BETA: Risk Profile
    beta = metrics.get('Beta', 1.0)
    if beta > 1.2:
        insights.append({
            "type": "warning",
            "icon": "🎢",
            "title": "Aggressive Stance",
            "text": f"Expect higher swings. This fund is {((beta-1)*100):.0f}% more sensitive to market movements."
        })
    elif beta < 0.8:
        insights.append({
            "type": "positive",
            "icon": "🛡️",
            "title": "Defensive Quality",
            "text": f"This fund is relatively stable, showing only {(beta*100):.0f}% of the market's volatility."
        })
    
    # 4. CAPTURE RATIOS: Bull/Bear performance
    up_cap = metrics.get('Upside Capture', 0)
    down_cap = metrics.get('Downside Capture', 0)
    
    if up_cap > 100:
        insights.append({
            "type": "positive",
            "icon": "🔋",
            "title": "Bull Market Star",
            "text": f"The fund captures {up_cap:.0f}% of market upside, excelling during bullish phases."
        })
    
    if down_cap < 90 and down_cap > 0:
        insights.append({
            "type": "positive",
            "icon": "🛑",
            "title": "Downside Protection",
            "text": f"Excellent at capital preservation! It only dropped {down_cap:.0f}% when the market fell."
        })
        
    return insights

def calculate_required_sip(target_amount, years, expected_return_annual):
    """
    Calculates the monthly SIP required to reach a target amount.
    Simplistic model (constant SIP).
    """
    if expected_return_annual <= 0 or years <= 0:
        return target_amount / (years * 12)
        
    monthly_rate = (1 + expected_return_annual) ** (1/12) - 1
    months = years * 12
    
    # Formula: FV = P * [((1+r)^n - 1) / r] * (1+r)
    # P = FV / [((1+r)^n - 1) / r * (1+r)]
    req_sip = target_amount / (((1 + monthly_rate)**months - 1) / monthly_rate * (1 + monthly_rate))
    return max(0, req_sip)

def calculate_required_sip_advanced(target_amount, years, expected_return_annual, initial_lump=0, step_up_percent=10):
    """
    Calculates the STARTING monthly SIP required to reach a target amount,
    considering an initial lumpsum and an annual percentage step-up in SIP.
    """
    months = years * 12
    monthly_rate = (1 + expected_return_annual) ** (1/12) - 1
    
    # Future Value of Lumpsum: FV_lump = initial_lump * (1+r)^n
    fv_lump = initial_lump * ((1 + monthly_rate) ** months)
    remaining_target = max(0, target_amount - fv_lump)
    
    if remaining_target == 0:
        return 0.0
        
    # We solve for initial_sip 'S' such that the sum of SIPs with step-ups equals remaining_target.
    # This is a GP-like sum where every 12 months the contribution increases.
    
    def get_fv(initial_sip):
        total_fv = 0
        current_sip = initial_sip
        for month in range(1, months + 1):
            if month > 1 and (month - 1) % 12 == 0:
                current_sip *= (1 + step_up_percent / 100)
            
            remaining_months = months - month + 1
            total_fv += current_sip * ((1 + monthly_rate) ** remaining_months)
        return total_fv

    # Binary search for initial_sip
    low = 0
    high = remaining_target
    for _ in range(50):
        mid = (low + high) / 2
        if get_fv(mid) < remaining_target:
            low = mid
        else:
            high = mid
            
    return low

def calculate_goal_success_probability(portfolio_nav, target_amount, years, initial_lump, sip_amount, step_up_percent=0, n_simulations=500):
    """
    Runs Monte Carlo simulation with SIP/Step-up to find the % of paths that hit the target.
    """
    if portfolio_nav is None or portfolio_nav.empty:
        return 0.5
        
    returns = portfolio_nav.pct_change().dropna()
    mu = returns.mean()
    sigma = returns.std()
    
    if sigma == 0:
        return 0.0
        
    months = years * 12
    dt_m = 1/12
    mu_m = mu * 252
    sigma_m = sigma * np.sqrt(252)
    
    success_count = 0
    
    for _ in range(n_simulations):
        current_val = initial_lump
        current_sip = sip_amount
        
        for m in range(1, months + 1):
            if m > 1 and (m - 1) % 12 == 0:
                current_sip *= (1 + step_up_percent / 100)
            
            current_val += current_sip
            z = np.random.standard_normal()
            growth = np.exp((mu_m - 0.5 * sigma_m**2) * dt_m + sigma_m * np.sqrt(dt_m) * z)
            current_val *= growth
            
        if current_val >= target_amount:
            success_count += 1
            
    return success_count / n_simulations

# --- SCENARIO ANALYSIS (STRESS TESTING) ---

def simulate_market_scenario(portfolio_series, benchmark_series, scenario_config):
    """
    Simulates portfolio performance during a specific market event.
    Returns benchmark performance even if portfolio data is unavailable for context.
    """
    import pandas as pd
    
    start_date = pd.to_datetime(scenario_config['start_date'])
    end_date = pd.to_datetime(scenario_config['end_date'])
    
    # Filter to scenario period
    portfolio_scenario = portfolio_series[(portfolio_series.index >= start_date) & (portfolio_series.index <= end_date)]
    benchmark_scenario = benchmark_series[(benchmark_series.index >= start_date) & (benchmark_series.index <= end_date)]
    
    # Basic response structure
    result = {
        'scenario_name': scenario_config['name'],
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'portfolio_return': 0,
        'benchmark_return': 0,
        'portfolio_max_drawdown': 0,
        'benchmark_max_drawdown': 0,
        'days_to_recover': None,
        'data_available': False,
        'bench_data_available': False,
        'outperformed_benchmark': False
    }

    if not benchmark_scenario.empty:
        result['benchmark_return'] = (benchmark_scenario.iloc[-1] / benchmark_scenario.iloc[0] - 1) * 100
        result['benchmark_max_drawdown'] = calculate_max_drawdown(benchmark_scenario) * 100
        result['bench_data_available'] = True

    if not portfolio_scenario.empty:
        result['portfolio_return'] = (portfolio_scenario.iloc[-1] / portfolio_scenario.iloc[0] - 1) * 100
        result['portfolio_max_drawdown'] = calculate_max_drawdown(portfolio_scenario) * 100
        result['data_available'] = True
        
        # Calculate recovery time
        portfolio_cummax = portfolio_scenario.cummax()
        portfolio_drawdown_series = (portfolio_scenario - portfolio_cummax) / portfolio_cummax
        max_dd_date = portfolio_drawdown_series.idxmin()
        recovery_series = portfolio_series[portfolio_series.index > max_dd_date]
        peak_value = portfolio_cummax.loc[max_dd_date]
        
        recovery_dates = recovery_series[recovery_series >= peak_value]
        if not recovery_dates.empty:
            result['days_to_recover'] = (recovery_dates.index[0] - max_dd_date).days
            
        result['outperformed_benchmark'] = result['portfolio_return'] > result['benchmark_return']
    
    return result

def get_predefined_scenarios():
    """
    Returns a list of predefined market crash scenarios with a focus on Indian context.
    """
    return [
        {
            'name': 'FII Sell-off / China Stimulus (2024)',
            'start_date': '2024-09-27',
            'end_date': '2024-11-15',
            'description': 'Heavy FII selling and rotation to China caused ~10% correction in Indian indices.'
        },
        {
            'name': 'Election Results Volatility (2024)',
            'start_date': '2024-06-03',
            'end_date': '2024-06-10',
            'description': 'Exit poll surge followed by a massive ~6% crash on result day (June 4).'
        },
        {
            'name': 'Adani-Hindenburg Crisis (2023)',
            'start_date': '2023-01-24',
            'end_date': '2023-02-28',
            'description': 'Hindenburg report on Adani Group caused significant volatility in Indian markets.'
        },
        {
            'name': 'Russia-Ukraine War (2022)',
            'start_date': '2022-02-20',
            'end_date': '2022-03-15',
            'description': 'Russia invasion of Ukraine caused global energy price shocks and ~15% market correction.'
        },
        {
            'name': 'COVID-19 Crash (2020)',
            'start_date': '2020-02-20',
            'end_date': '2020-03-24', # Peak crash date
            'description': 'Global pandemic caused a massive ~35-40% crash in just 30 days.'
        },
        {
            'name': 'IL&FS Crisis (2018)',
            'start_date': '2018-09-01',
            'end_date': '2018-10-31',
            'description': 'IL&FS default triggered a systemic liquidity crisis for NBFCs in India.'
        },
        {
            'name': 'Demonetization Impact (2016)',
            'start_date': '2016-11-08',
            'end_date': '2016-12-30',
            'description': 'Sudden demonetization announcement caused short-term economic and market disruption.'
        },
        {
            'name': 'Taper Tantrum (2013)',
            'start_date': '2013-05-22',
            'end_date': '2013-09-15',
            'description': 'Fed taper hints led to sharp rupee fall and market correction in emerging economies.'
        },
        {
            'name': '2008 Global Financial Crisis',
            'start_date': '2008-01-01',
            'end_date': '2009-03-09',
            'description': 'Lehman Brothers collapse and subprime crisis led to ~60% crash in Indian markets.'
        },
        {
            'name': 'Dot-com Bubble Burst (2000)',
            'start_date': '2000-02-11',
            'end_date': '2001-09-21',
            'description': 'Tech bubble pop and 9/11 attacks caused multi-year bear market.'
        }
    ]

def get_all_stress_tests(portfolio_series, benchmark_series):
    """
    Runs all predefined stress tests for a given portfolio/benchmark combo.
    """
    scenarios = get_predefined_scenarios()
    results = []
    for sc in scenarios:
        res = simulate_market_scenario(portfolio_series, benchmark_series, sc)
        if res.get('bench_data_available'):
            results.append(res)
    return results
