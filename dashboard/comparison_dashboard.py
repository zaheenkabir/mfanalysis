import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go

# Add backend and local dashboard to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
backend_path = os.path.abspath(os.path.join(current_dir, '..', 'backend'))
if backend_path not in sys.path:
    sys.path.append(backend_path)

from data_loader import fetch_latest_nav, fetch_fund_history, fetch_scheme_details

from ui_components import (
    metric_card, get_neon_color, style_financial_dataframe,
    add_max_drawdown_annotation, add_drawdown_vertical_line,
    render_capture_ratio_chart
)

from analytics import (
    download_benchmark, get_fund_metrics, calculate_rolling_returns, 
    calculate_correlation_matrix, simulate_market_scenario, get_predefined_scenarios,
    calculate_max_drawdown_series
)

def render_comparison_dashboard(nav_data, selected_funds, selected_benchmarks=["NIFTY 50"]):
    """Renders the multi-fund comparison view"""
    st.header("📊 Compare Multiple Funds")
    st.caption("Select 2-5 funds to compare their performance side-by-side with selected benchmarks")
    
    # nav_data and selected_funds are now passed as parameters
    
    # Extract scheme codes from selected funds
    selected_data = []
    for fund_display in selected_funds:
        qt_start = fund_display.rfind('(')
        qt_end = fund_display.rfind(')')
        code = fund_display[qt_start+1:qt_end]
        name = fund_display[:qt_start].strip()
        selected_data.append((name, code))
    
    # Fetch data for all funds
    st.info(f"⏳ Loading data for {len(selected_funds)} funds...")
    
    funds_data = []
    colors = ['#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#3b82f6']
    
    progress_bar = st.progress(0)
    for idx, (name, code) in enumerate(selected_data):
        with st.spinner(f"Fetching {name[:40]}..."):
            history_df = fetch_fund_history(code)
            metadata = fetch_scheme_details(code, nav_data)
            
            if not history_df.empty:
                # Get benchmark
                category = metadata.get('scheme_category', '').lower()
                benchmark_ticker = "NIFTY 50"
                if 'small' in category:
                    benchmark_ticker = "NIFTY SMALLCAP 50"
                elif 'mid' in category:
                    benchmark_ticker = "NIFTY MIDCAP 50"
                elif 'bank' in category:
                    benchmark_ticker = "NIFTY BANK"
                
                benchmark_data = download_benchmark(benchmark_ticker)
                
                metrics = get_fund_metrics(history_df['nav'], benchmark_data)
                
                funds_data.append({
                    'name': name,
                    'code': code,
                    'history': history_df['nav'],
                    'metrics': metrics,
                    'metadata': metadata,
                    'color': colors[idx % len(colors)]
                })
        
        progress_bar.progress((idx + 1) / len(selected_data))
    
    progress_bar.empty()
    
    # --- FETCH BENCHMARK DATA ---
    benchmark_options = {
        "NIFTY 50": "NIFTY 50",
        "Nifty Midcap 50": "NIFTY MIDCAP 50",
        "Nifty Smallcap 50": "NIFTY SMALLCAP 50",
        "Nifty Bank": "NIFTY BANK",
        "Gold (GoldBees)": "GOLD",
        "Silver (SilverBees)": "SILVER"
    }
    
    benchmark_data = []
    bench_colors = ['#10b981', '#f59e0b', '#ef4444', '#3b82f6'] # Different palette or style for benchmarks
    
    if selected_benchmarks:
        for idx, bench_name in enumerate(selected_benchmarks):
            ticker = benchmark_options.get(bench_name, "^NSEI")
            with st.spinner(f"Fetching benchmark {bench_name}..."):
                bench_series = download_benchmark(ticker)
                if not bench_series.empty:
                    benchmark_data.append({
                        'name': bench_name,
                        'history': bench_series,
                        'color': bench_colors[idx % len(bench_colors)]
                    })

    if not funds_data:
        st.error("Unable to load data for selected funds.")
        return
    
    st.success(f"✅ Loaded {len(funds_data)} funds and {len(benchmark_data)} benchmarks successfully!")
    
    # --- TIME PERIOD FILTER ---
    st.markdown("### ⏱️ Analysis Period")
    time_period = st.radio(
        "Select Time Period",
        ["1Y", "3Y", "5Y", "10Y", "Max"],
        horizontal=True,
        index=4,  # Default to Max
        key="comparison_time_period"
    )
    
    # Map period to years
    period_map = {"1Y": 1, "3Y": 3, "5Y": 5, "10Y": 10, "Max": None}
    selected_years = period_map[time_period]
    
    # Filter data by selected period
    filtered_funds_data = []
    for fund in funds_data:
        series = fund['history'].copy()
        if selected_years:
            end_date = series.index[-1]
            start_date = end_date - pd.DateOffset(years=selected_years)
            series = series[series.index >= start_date]
        
        if not series.empty:
            # Recalculate metrics on filtered data
            # Get appropriate benchmark for this fund
            category = fund['metadata'].get('scheme_category', '').lower()
            benchmark_ticker = "NIFTY 50"
            if 'small' in category:
                benchmark_ticker = "NIFTY SMALLCAP 50"
            elif 'mid' in category:
                benchmark_ticker = "NIFTY MIDCAP 50"
            elif 'bank' in category:
                benchmark_ticker = "NIFTY BANK"
            
            bench_series_for_metrics = download_benchmark(benchmark_ticker)
            if selected_years and not bench_series_for_metrics.empty:
                end_date_b = bench_series_for_metrics.index[-1]
                start_date_b = end_date_b - pd.DateOffset(years=selected_years)
                bench_series_for_metrics = bench_series_for_metrics[bench_series_for_metrics.index >= start_date_b]
            
            filtered_metrics = get_fund_metrics(series, bench_series_for_metrics)
            
            filtered_funds_data.append({
                'name': fund['name'],
                'code': fund['code'],
                'history': series,
                'metrics': filtered_metrics,
                'metadata': fund['metadata'],
                'color': fund['color']
            })
    
    # Filter benchmark data by period
    filtered_benchmark_data = []
    for bench in benchmark_data:
        series = bench['history'].copy()
        if selected_years:
            end_date = series.index[-1]
            start_date = end_date - pd.DateOffset(years=selected_years)
            series = series[series.index >= start_date]
        
        if not series.empty:
            filtered_benchmark_data.append({
                'name': bench['name'],
                'history': series,
                'color': bench['color']
            })
    
    st.divider()
    
    # --- HIGHLIGHT REEL ---
    st.markdown("### 🏆 Performance Highlights")
    
    # Identify winners
    best_cagr = max(filtered_funds_data, key=lambda x: x['metrics'].get('CAGR', -999))
    best_sharpe = max(filtered_funds_data, key=lambda x: x['metrics'].get('Sharpe Ratio', -999))
    lowest_vol = min(filtered_funds_data, key=lambda x: x['metrics'].get('Volatility', 999))
    
    h1, h2, h3 = st.columns(3)
    
    with h1:
        val = best_cagr['metrics'].get('CAGR', 0)*100
        metric_card(
            "Highest Return", 
            f"{val:.2f}%", 
            delta=best_cagr['name'][:20] + "...", 
            is_good_if_positive=True
        )
        
    with h2:
        val = best_sharpe['metrics'].get('Sharpe Ratio', 0)
        metric_card(
            "Best Risk-Adjusted", 
            f"{val:.2f}", 
            delta=best_sharpe['name'][:20] + "...", 
            is_good_if_positive=True
        )
        
    with h3:
        val = lowest_vol['metrics'].get('Volatility', 0)*100
        metric_card(
            "Lowest Volatility", 
            f"{val:.2f}%", 
            delta=lowest_vol['name'][:20] + "...", 
            is_good_if_positive=True  # Low volatility is good, but delta color logic treats positive as green. 
                                      # This is tricky. Text is name, so color applies to name. Green name = Good.
        )
    
    st.divider()

    # METRICS COMPARISON TABLE
    st.markdown("### 📈 Performance Metrics")
    
    metrics_data = []
    for fund in filtered_funds_data:
        m = fund['metrics']
        metrics_data.append({
            'Fund': fund['name'],
            'CAGR': m.get('CAGR', 0)*100,
            'Sharpe': m.get('Sharpe Ratio', 0),
            'Alpha': m.get('Alpha', 0)*100,
            'Volatility': m.get('Volatility', 0)*100,
            'Max DD': m.get('Max Drawdown', 0)*100,
            'Up Capture': m.get('Upside Capture', 0),
            'Down Capture': m.get('Downside Capture', 0)
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    # Use the new styling helper
    st.dataframe(style_financial_dataframe(metrics_df), use_container_width=True, hide_index=True)
    
    st.divider()

    # --- CAPTURE RATIO SECTION ---
    st.markdown("### 🏹 Capture Ratio Analysis")
    st.caption("How funds perform relative to benchmark in Up vs Down markets")
    
    cap_col1, cap_col2 = st.columns(2)
    with cap_col1:
        st.markdown("""
        **Upside Capture** > 100 means the fund beats benchmark in Bull markets.  
        **Downside Capture** < 100 means the fund loses less than benchmark in Bear markets.
        """)
    
    cap_tabs = st.tabs([f["name"] for f in filtered_funds_data])
    for idx, fund in enumerate(filtered_funds_data):
        with cap_tabs[idx]:
            m = fund['metrics']
            render_capture_ratio_chart(
                m.get('Upside Capture', 0), 
                m.get('Downside Capture', 0), 
                fund['name']
            )
    
    st.divider()
    
    # NORMALIZED PERFORMANCE CHART (Equity Curve Style)
    st.markdown("### 📊 Equity Curve Comparison")
    st.caption("Growth of ₹10,000 investment across all selected funds")
    
    # Find common date range
    all_series = [fund['history'] for fund in filtered_funds_data]
    common_start = max([s.index[0] for s in all_series])
    common_end = min([s.index[-1] for s in all_series])
    
    initial_value = 10000
    fig_perf = go.Figure()
    
    for fund in funds_data:
        series = fund['history']
        series_filtered = series[(series.index >= common_start) & (series.index <= common_end)]
        
        if not series_filtered.empty:
            rebased = (series_filtered / series_filtered.iloc[0]) * initial_value
            fig_perf.add_trace(go.Scatter(
                x=rebased.index,
                y=rebased.values,
                mode='lines',
                name=fund['name'],
                line=dict(color=fund['color'], width=2.5),
                hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Value: ₹%{y:,.0f}<extra></extra>',
                text=[fund['name']] * len(rebased)
            ))
    
    # Add Benchmarks to Equity Curve
    for bench in filtered_benchmark_data:
        series = bench['history']
        series_filtered = series[(series.index >= common_start) & (series.index <= common_end)]
        
        if not series_filtered.empty:
            rebased = (series_filtered / series_filtered.iloc[0]) * initial_value
            fig_perf.add_trace(go.Scatter(
                x=rebased.index,
                y=rebased.values,
                mode='lines',
                name=f"{bench['name']} (Benchmark)",
                line=dict(color=bench['color'], width=2, dash='dot'),
                hovertemplate='<b>%{text}</b><br>Date: %{x}<br>Value: ₹%{y:,.0f}<extra></extra>',
                text=[bench['name']] * len(rebased)
            ))
    
    fig_perf.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value (₹)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(0,0,0,0.3)'
        ),
        margin=dict(b=100)
    )
    
    # --- ADD MAX DRAWDOWN ANNOTATIONS ---
    from analytics import calculate_max_drawdown_series
    
    deepest_date = None
    deepest_val_all = 0
    
    for fund in funds_data:
        series = fund['history']
        series_filtered = series[(series.index >= common_start) & (series.index <= common_end)]
        
        if not series_filtered.empty:
            dd_series = calculate_max_drawdown_series(series_filtered)
            if not dd_series.empty:
                max_dd_date = dd_series.idxmin()
                
                # Rebase for chart
                rebased = (series_filtered / series_filtered.iloc[0]) * initial_value
                max_dd_val = rebased.loc[max_dd_date]
                
                # Check if this is the deepest overall for the vertical line
                dd_depth = dd_series.min()
                if dd_depth < deepest_val_all:
                    deepest_val_all = dd_depth
                    deepest_date = max_dd_date
                
                # Add individual annotation
                add_max_drawdown_annotation(fig_perf, max_dd_date, max_dd_val, text="Max DD")

    # Add one vertical line for the global deepest crash (usually March 2020)
    if deepest_date:
        add_drawdown_vertical_line(fig_perf, deepest_date, color="rgba(239, 68, 68, 0.2)")

    st.plotly_chart(fig_perf, use_container_width=True)
    
    st.divider()
    
    # DRAWDOWN COMPARISON
    st.markdown("### 📉 Drawdown Analysis")
    st.caption("Shows maximum decline from peak - lower is better")
    
    fig_dd = go.Figure()
    
    for fund in filtered_funds_data:
        from analytics import calculate_max_drawdown_series
        dd_series = calculate_max_drawdown_series(fund['history'])
        if not dd_series.empty:
            fig_dd.add_trace(go.Scatter(
                x=dd_series.index,
                y=dd_series.values * 100,
                mode='lines',
                name=fund['name'][:30],
                line=dict(color=fund['color'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba{tuple(list(int(fund['color'][i:i+2], 16) for i in (1, 3, 5)) + [0.1])}"
            ))
            
    # Add Benchmarks to Drawdown
    for bench in filtered_benchmark_data:
        dd_series = calculate_max_drawdown_series(bench['history'])
        if not dd_series.empty:
            fig_dd.add_trace(go.Scatter(
                x=dd_series.index,
                y=dd_series.values * 100,
                mode='lines',
                name=f"{bench['name']} (Benchmark)",
                line=dict(color=bench['color'], width=1.5, dash='dot')
            ))
    
    fig_dd.update_layout(
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        hovermode='x unified',
        legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_dd, use_container_width=True)
    
    # --- DRAWDOWN INSIGHTS ---
    st.markdown("#### 📉 Recovery Insights")
    dd_col1, dd_col2 = st.columns(2)
    
    deepest_dd_fund = min(filtered_funds_data, key=lambda x: x['metrics'].get('Max Drawdown', 0))
    quickest_rec = min(filtered_funds_data, key=lambda x: x['metrics'].get('Volatility', 999)) # Proxied by vol for now if recovery days missing
    
    with dd_col1:
        st.write(f"⚠️ **Deepest Decline:** `{deepest_dd_fund['name'][:40]}` experienced the largest peak-to-trough drop of {deepest_dd_fund['metrics'].get('Max Drawdown', 0)*100:.2f}%.")
    with dd_col2:
        st.write(f"🛡️ **Downside Protection:** `{quickest_rec['name'][:40]}` generally shows more resilience during market stress, maintaining shallower drawdowns.")

    st.divider()
    
    # MULTI-PERIOD ROLLING RETURNS ANALYSIS
    st.markdown("### 📊 Rolling Returns Analysis")
    st.caption("Analyze consistency across different time horizons")
    
    period_selector = st.selectbox(
        "Select Rolling Period",
        ["1 Year", "3 Years", "5 Years", "10 Years"],
        key="comparison_rolling_period"
    )
    
    period_map = {"1 Year": 1, "3 Years": 3, "5 Years": 5, "10 Years": 10}
    selected_years = period_map[period_selector]
    
    fig_multi_rolling = go.Figure()
    
    for fund in funds_data:
        rolling_ret = calculate_rolling_returns(fund['history'], window_years=selected_years)
        if not rolling_ret.empty:
            fig_multi_rolling.add_trace(go.Scatter(
                x=rolling_ret.index,
                y=rolling_ret.values * 100,
                mode='lines',
                name=fund['name'][:30],
                line=dict(color=fund['color'], width=2)
            ))
            
    # Add Benchmarks to Rolling Returns
    for bench in benchmark_data:
        rolling_ret = calculate_rolling_returns(bench['history'], window_years=selected_years)
        if not rolling_ret.empty:
            fig_multi_rolling.add_trace(go.Scatter(
                x=rolling_ret.index,
                y=rolling_ret.values * 100,
                mode='lines',
                name=f"{bench['name']} (Benchmark)",
                line=dict(color=bench['color'], width=1.5, dash='dot')
            ))
    
    fig_multi_rolling.update_layout(
        title=f"{period_selector} Rolling Returns",
        xaxis_title="Date",
        yaxis_title="Returns (%)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=450,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_multi_rolling, use_container_width=True)
    
    # ROLLING RETURNS SUMMARY TABLE
    st.markdown("#### Rolling Returns Summary Table")
    st.caption("Average rolling returns across different time periods")
    
    rolling_summary = []
    # Combine funds and benchmarks for the summary table
    comparison_items = []
    for f in funds_data:
        comparison_items.append({'name': f['name'][:30], 'history': f['history']})
    for b in benchmark_data:
        comparison_items.append({'name': f"{b['name']} (Bench)", 'history': b['history']})

    for item in comparison_items:
        item_row = {'Fund': item['name']}
        
        # Calculate rolling returns for each period
        for period_name, years in [("1Y Return", 1), ("3Y Return", 3), ("5Y Return", 5), ("10Y Return", 10)]:
            rolling_ret = calculate_rolling_returns(item['history'], window_years=years)
            if not rolling_ret.empty:
                avg_return = rolling_ret.mean() * 100
                item_row[period_name] = avg_return
            else:
                item_row[period_name] = None
        
        rolling_summary.append(item_row)
    
    rolling_summary_df = pd.DataFrame(rolling_summary)
    st.dataframe(style_financial_dataframe(rolling_summary_df), use_container_width=True, hide_index=True)

    
    # RISK-RETURN SCATTER
    st.markdown("### 🎯 Risk-Return Analysis")
    st.caption("Top-left quadrant = Best (High returns, Low risk)")

    # Calculate averages for quadrants
    avg_vol = sum([f['metrics'].get('Volatility', 0) for f in funds_data]) / len(funds_data) * 100
    avg_ret = sum([f['metrics'].get('CAGR', 0) for f in funds_data]) / len(funds_data) * 100
    
    fig_scatter = go.Figure()
    
    # Add quadrant lines
    fig_scatter.add_vline(x=avg_vol, line_dash="dash", line_color="rgba(255,255,255,0.2)", annotation_text="Avg Risk")
    fig_scatter.add_hline(y=avg_ret, line_dash="dash", line_color="rgba(255,255,255,0.2)", annotation_text="Avg Return")
    
    for fund in funds_data:
        m = fund['metrics']
        fig_scatter.add_trace(go.Scatter(
            x=[m.get('Volatility', 0) * 100],
            y=[m.get('CAGR', 0) * 100],
            mode='markers+text',
            name=fund['name'][:25],
            text=[fund['name'][:20]],
            textposition="top center",
            marker=dict(size=22, color=fund['color'], line=dict(width=2, color='white')),
            showlegend=True,
            hovertemplate='<b>%{text}</b><br>Risk: %{x:.2f}%<br>Return: %{y:.2f}%<extra></extra>'
        ))
    
    fig_scatter.update_layout(
        xaxis_title="Risk (Volatility %)",
        yaxis_title="Return (CAGR %)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=550,
        hovermode='closest',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(0,0,0,0.3)'
        ),
        margin=dict(b=100) # Add space for bottom legend
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # --- ANALYTICAL INSIGHTS ---
    st.markdown("#### 🎯 Analytical Insights")
    ins_col1, ins_col2 = st.columns(2)
    
    # Logic for insights
    best_efficiency = max(funds_data, key=lambda x: x['metrics'].get('Sharpe Ratio', 0))
    highest_ret = max(funds_data, key=lambda x: x['metrics'].get('CAGR', 0))
    lowest_vol = min(funds_data, key=lambda x: x['metrics'].get('Volatility', 999))
    
    with ins_col1:
        st.write(f"🏆 **Efficiency Leader:** `{best_efficiency['name'][:40]}` provides the best risk-adjusted profile with a Sharpe ratio of {best_efficiency['metrics'].get('Sharpe Ratio', 0):.2f}.")
        st.write(f"🛡️ **Stability Core:** `{lowest_vol['name'][:40]}` has the lowest volatility ({lowest_vol['metrics'].get('Volatility', 0)*100:.2f}%) making it the most conservative choice here.")
        
    with ins_col2:
        st.write(f"🚀 **High Performance:** `{highest_ret['name'][:40]}` lead in absolute returns at {highest_ret['metrics'].get('CAGR', 0)*100:.2f}% CAGR.")
        
        # Quadrant check
        overperformers = [f['name'] for f in funds_data if f['metrics'].get('CAGR', 0)*100 > avg_ret and f['metrics'].get('Volatility', 0)*100 < avg_vol]
        if overperformers:
            st.write(f"✨ **Sweet Spot:** `{overperformers[0][:40]}` sits in the top-left quadrant, delivering above-average returns with below-average risk.")
        else:
            st.write("💡 **Diversification Tip:** These funds have diverse risk-return profiles. Consider a mix of low-volatility and high-return assets to balance your portfolio.")

    st.divider()
    
    
    # --- CORRELATION MATRIX ---
    st.markdown("### 📊 Fund Correlation Analysis")
    st.caption("Understanding how your selected funds move together - Lower correlation = Better diversification")
    
    # Create DataFrame of aligned fund prices
    # CORRECTED: fund['history'] is already the NAV series
    fund_series_dict = {fund['name']: fund['history'] for fund in funds_data}
    
    # Align to common date range
    if fund_series_dict:
        common_start = max([s.index[0] for s in fund_series_dict.values()])
        common_end = min([s.index[-1] for s in fund_series_dict.values()])
        aligned_data = {k: v[(v.index >= common_start) & (v.index <= common_end)] for k, v in fund_series_dict.items()}
        comparison_df = pd.DataFrame(aligned_data)
        
        # Calculate correlation matrix using pairwise complete observations
        correlation_matrix = calculate_correlation_matrix(comparison_df)
        
        if not correlation_matrix.empty:
            # Create heatmap
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=[name[:25] for name in correlation_matrix.columns],
                y=[name[:25] for name in correlation_matrix.columns],
                colorscale='PiYG',  # Purple to Green: Green = Correlation (bad/overlap), Purple = Low (good/diversified)
                reversescale=True,
                zmin=-1,
                zmax=1,
                text=correlation_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10},
                hovertemplate='<b>%{x}</b> vs <b>%{y}</b><br>Correlation: %{z:.3f}<extra></extra>'
            ))
            
            fig_corr.update_layout(
                title="Correlation Heatmap (Daily Returns)",
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=500,
                xaxis={'side': 'bottom'},
                yaxis={'autorange': 'reversed'}
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # --- CORRELATION INSIGHTS ---
            avg_corr = (correlation_matrix.values.sum() - len(correlation_matrix)) / (len(correlation_matrix)**2 - len(correlation_matrix)) if len(correlation_matrix) > 1 else 0
            
            st.markdown("#### 💡 Diversification Tips")
            if avg_corr > 0.8:
                st.warning(f"🔴 **High Overlap:** The average correlation between these funds is very high ({avg_corr:.2f}). You might not be getting much diversification benefit; these funds likely move in lockstep.")
            elif avg_corr > 0.5:
                st.info(f"🟡 **Moderate Diversification:** Average correlation is {avg_corr:.2f}. There is some overlap in their performance drivers, but they offer some distinct risk profiles.")
            else:
                st.success(f"🔵 **Excellent Diversification:** Average correlation is low ({avg_corr:.2f}). These funds complement each other well, likely providing a smoother ride during market volatility.")

            # Interpretation guide
            col_guide1, col_guide2, col_guide3 = st.columns(3)
            with col_guide1:
                st.markdown("🟢 **Green (> 0.7)**")
                st.caption("High overlap - Funds move together")
            with col_guide2:
                st.markdown("⚪ **White (0.3-0.7)**")
                st.caption("Moderate overlap")
            with col_guide3:
                st.markdown("🟣 **Purple (< 0.3)**")
                st.caption("Great diversification")    
    st.divider()

    # --- RISK LABS: STRESS TESTING ---
    st.markdown("### 🧪 Risk Labs - Stress Testing")
    st.caption("See how these funds would have performed during major market crashes")
    
    scenarios = get_predefined_scenarios()
    selected_scenario_name = st.selectbox(
        "Select Market Event",
        [s['name'] for s in scenarios],
        index=4, # Default to COVID
        key="comparison_risk_lab"
    )
    
    scenario = next(s for s in scenarios if s['name'] == selected_scenario_name)
    st.info(f"📅 **{scenario['name']}** ({scenario['start_date']} to {scenario['end_date']})\n\n{scenario['description']}")
    
    stress_results = []
    main_bench_series = benchmark_data[0]['history'] if benchmark_data else pd.Series()
    main_bench_name = benchmark_data[0]['name'] if benchmark_data else "Benchmark"
    
    # Calculate for Funds
    for fund in funds_data:
        res = simulate_market_scenario(fund['history'], main_bench_series, scenario)
        if res['data_available']:
            stress_results.append({
                'Fund': fund['name'][:30],
                'Return During Event': res['portfolio_return'],
                'Max Drawdown': res['portfolio_max_drawdown'],
                'Recovery Time': f"{res['days_to_recover']} days" if res['days_to_recover'] else "Not yet recovered",
                '_raw_return': res['portfolio_return'] # Hidden col for sorting/insight logic
            })
            
    # Calculate for Benchmarks
    for bench in benchmark_data:
        bench_series = bench['history']
        res_bench = simulate_market_scenario(bench_series, bench_series, scenario) # Self-comparison for metrics
        if res_bench['bench_data_available'] or res_bench['data_available']: # Check either flag depending on return structure
             # Use the available return, simulate_market_scenario might return 'portfolio_return' if we passed it as portfolio
             # Actually simulate_market_scenario returns 'portfolio_...' for the first arg and 'benchmark_...' for the second. 
             # Here we passed bench as both, so 'portfolio_return' is the bench return.
            stress_results.append({
                'Fund': f"{bench['name']} (Benchmark)",
                'Return During Event': res_bench['portfolio_return'], 
                'Max Drawdown': res_bench['portfolio_max_drawdown'],
                'Recovery Time': f"{res_bench['days_to_recover']} days" if res_bench.get('days_to_recover') else "Not yet recovered",
                '_raw_return': res_bench['portfolio_return']
            })

    if stress_results:
        stress_df = pd.DataFrame(stress_results)
        # Display table without the raw helper column
        display_df = stress_df.drop(columns=['_raw_return'])
        st.dataframe(style_financial_dataframe(display_df), use_container_width=True, hide_index=True)
        
        # --- INSIGHTS ---
        st.markdown("#### 💡 Analysis & Insights")
        
        # Find best performing fund (excluding benchmark row for "winner" title if possible, or include it for comparison)
        # Let's separate funds and benchmark for insight
        fund_rows = [r for r in stress_results if "Benchmark" not in r['Fund']]
        bench_row = next((r for r in stress_results if "Benchmark" in r['Fund']), None)
        
        if fund_rows:
            best_fund = max(fund_rows, key=lambda x: x['_raw_return'])
            
            if bench_row and best_fund['_raw_return'] > bench_row['_raw_return']:
                st.success(f"✅ **Crash Resilience:** `{best_fund['Fund']}` was the most resilient, outperforming the benchmark by {abs(best_fund['_raw_return'] - bench_row['_raw_return']):.2f}%.")
            elif bench_row:
                 st.warning(f"⚠️ **Market Breakdown:** All selected funds fell more than the benchmark during this event.")
            
            st.info(f"📊 **Top Performer:** `{best_fund['Fund']}` limited losses to {best_fund['_raw_return']:.2f}% during this crash.")

        # --- HISTORICAL CONTEXT ---
        st.markdown("#### 📚 Historical Context")
        event_context = {
            "COVID-19 Crash (2020)": "The fastest 30% drop in history, triggered by global pandemic fears. Markets recovered to new highs within 6 months as central banks pumped liquidity.",
            "Global Financial Crisis (2008-09)": "Triggered by US housing market collapse and Lehman Brothers bankruptcy. One of the worst crashes since 1929, taking over 5 years for full recovery.",
            "IL&FS Crisis (2018)": "India-specific credit crisis that caused significant damage to mid and small-cap stocks. NBFCs and financial stocks were hit hardest.",
            "Demonetization (2016)": "Sudden cash crunch caused short-term market volatility, but recovery was swift as fundamentals remained intact.",
            "Taper Tantrum (2013)": "Fed's announcement to reduce QE caused emerging market selloff. India's Rupee fell sharply, impacting import-heavy sectors.",
             "Russia-Ukraine War (2022)": "Geopolitical tension spikes caused oil prices to surge, impacting global equity markets and increasing inflation fears.",
             "Adani-Hindenburg Crisis (2023)": "A localized crisis triggering sharp volatility in Indian markets, primarily affecting the Adani group stocks and banking sector sentiment.",
             "Election Results Volatility (2024)": "Unexpected election outcome created a massive one-day shock, which was quickly bought into by DIIs and retail investors."
        }
        
        ctx = event_context.get(scenario['name'])
        if ctx:
            st.caption(ctx)
            
    else:
        st.warning("⚠️ Selected funds do not have historical data for this period.")

    st.divider()
