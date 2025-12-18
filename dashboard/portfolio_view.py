import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add backend to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

# Add dashboard to path for module imports
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import fetch_latest_nav, fetch_fund_history, fetch_scheme_details
from analytics import (
    download_benchmark, 
    get_fund_metrics, 
    calculate_rolling_returns,
    calculate_max_drawdown_series,
    create_weighted_portfolio,
    run_monte_carlo_simulation,
    calculate_correlation_matrix,
    simulate_market_scenario,
    get_predefined_scenarios
)
from ui_components import metric_card, get_neon_color

def render_portfolio_view():
    """Renders the portfolio builder and analysis view"""
    st.header("🎯 Build & Analyze Portfolio")
    st.caption("Create a custom portfolio with weighted funds and analyze against multiple benchmarks")
    
    # Load fund universe
    with st.spinner("Loading Fund Universe..."):
        nav_data = fetch_latest_nav()
    
    if nav_data.empty:
        st.error("Failed to load fund data.")
        return
    
    # Sidebar for fund selection
    with st.sidebar:
        st.header("Portfolio Construction")
        
        # Create fund list
        nav_data['Display'] = nav_data['Scheme Name'] + " (" + nav_data['Scheme Code'].astype(str) + ")"
        funds_list = nav_data['Display'].tolist()
        
        # Multi-select for funds
        selected_funds = st.multiselect(
            "Select Funds for Portfolio",
            funds_list,
            key="portfolio_fund_selector",
            help="Select 2-10 funds"
        )
        
        if len(selected_funds) < 2:
            st.info("👆 Please select at least 2 funds")
            return
        elif len(selected_funds) > 10:
            st.warning("⚠️ Maximum 10 funds allowed")
            return
        
        st.divider()
        
        # Weightage inputs
        st.subheader("Assign Weightages")
        st.caption("Total must equal 100%")
        
        weights = {}
        for fund_display in selected_funds:
            qt_start = fund_display.rfind('(')
            name = fund_display[:qt_start].strip()
            short_name = name[:30] + "..." if len(name) > 30 else name
            
            weight = st.number_input(
                short_name,
                min_value=0.0,
                max_value=100.0,
                value=round(100.0/len(selected_funds), 2),
                step=0.1,
                key=f"weight_{fund_display}"
            )
            weights[fund_display] = weight
        
        total_weight = sum(weights.values())
        
        # Display total weight with color coding
        if abs(total_weight - 100.0) < 0.01:
            st.success(f"✅ Total: {total_weight:.2f}%")
        else:
            st.error(f"❌ Total: {total_weight:.2f}% (Must be 100%)")
            return
        
        st.divider()
        
        # Benchmark selection
        st.subheader("Select Benchmarks")
        
        benchmark_options = {
            "NIFTY 50": "^NSEI",
            "NIFTY MIDCAP 50": "^NSEMDCP50",
            "NIFTY SMALLCAP 100": "^CNXSC",
            "NIFTY BANK": "^NSEBANK"
        }
        
        selected_benchmarks = st.multiselect(
            "Choose Benchmarks",
            list(benchmark_options.keys()),
            default=["NIFTY 50"],
            key="portfolio_benchmarks"
        )
        
        if not selected_benchmarks:
            st.warning("Please select at least one benchmark")
            return
    
    # Extract scheme codes and fetch data
    st.info(f"⏳ Building portfolio with {len(selected_funds)} funds...")
    
    fund_data = {}
    progress_bar = st.progress(0)
    
    for idx, fund_display in enumerate(selected_funds):
        qt_start = fund_display.rfind('(')
        qt_end = fund_display.rfind(')')
        code = fund_display[qt_start+1:qt_end]
        name = fund_display[:qt_start].strip()
        
        with st.spinner(f"Fetching {name[:40]}..."):
            history_df = fetch_fund_history(code)
            
            if not history_df.empty:
                fund_data[name] = {
                    'series': history_df['nav'],
                    'weight': weights[fund_display]
                }
        
        progress_bar.progress((idx + 1) / len(selected_funds))
    
    progress_bar.empty()
    
    if len(fund_data) < 2:
        st.error("Unable to load sufficient fund data.")
        return
    
    # Create weighted portfolio
    fund_series_dict = {name: data['series'] for name, data in fund_data.items()}
    weight_dict = {name: data['weight'] for name, data in fund_data.items()}
    
    portfolio_nav = create_weighted_portfolio(fund_series_dict, weight_dict)
    
    if portfolio_nav.empty:
        st.error("Unable to create portfolio. Please check fund data.")
        return
    
    st.success(f"✅ Portfolio created successfully!")
    
    # Calculate portfolio overall return
    portfolio_start = portfolio_nav.iloc[0]
    portfolio_end = portfolio_nav.iloc[-1]
    portfolio_overall_return = ((portfolio_end - portfolio_start) / portfolio_start) * 100
    
    # Fetch benchmark data early (needed for fund metrics calculation)
    benchmark_options = {
        "NIFTY 50": "^NSEI",
        "NIFTY MIDCAP 50": "^NSEMDCP50",
        "NIFTY SMALLCAP 100": "^CNXSC",
        "NIFTY BANK": "^NSEBANK"
    }
    
    benchmark_data = {}
    for bench_name in selected_benchmarks:
        ticker = benchmark_options[bench_name]
        bench_series = download_benchmark(ticker)
        if not bench_series.empty:
            benchmark_data[bench_name] = bench_series
    
    # FUND DETAILS TABLE WITH METRICS
    st.markdown("### 📋 Fund Details & Metrics")
    
    fund_details = []
    for name, data in fund_data.items():
        # Fetch metadata and calculate metrics for each fund
        for fund_display in selected_funds:
            if name in fund_display:
                qt_start = fund_display.rfind('(')
                qt_end = fund_display.rfind(')')
                code = fund_display[qt_start+1:qt_end]
                metadata = fetch_scheme_details(code)
                
                # Calculate metrics for this fund
                # Use first benchmark for individual fund metrics
                first_benchmark = list(benchmark_data.values())[0] if benchmark_data else pd.Series()
                fund_metrics = get_fund_metrics(data['series'], first_benchmark) if not first_benchmark.empty else {}
                
                fund_details.append({
                    'Fund Name': name[:35],
                    'Weight': f"{data['weight']:.1f}%",
                    'CAGR': f"{fund_metrics.get('CAGR', 0)*100:.2f}%",
                    'Sharpe': f"{fund_metrics.get('Sharpe Ratio', 0):.2f}",
                    'Alpha': f"{fund_metrics.get('Alpha', 0)*100:.2f}%",
                    'Beta': f"{fund_metrics.get('Beta', 0):.2f}",
                    'Volatility': f"{fund_metrics.get('Volatility', 0)*100:.2f}%",
                    'Max DD': f"{fund_metrics.get('Max Drawdown', 0)*100:.2f}%"
                })
                break
    
    details_df = pd.DataFrame(fund_details)
    st.dataframe(details_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # PORTFOLIO COMPOSITION - ENHANCED
    st.markdown("### 📊 Portfolio Composition")
    
    # Calculate portfolio CAGR
    from analytics import calculate_cagr
    portfolio_cagr = calculate_cagr(portfolio_nav) * 100
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Composition table
        comp_data = []
        for name, data in fund_data.items():
            comp_data.append({
                'Fund': name[:35],
                'Weight': f"{data['weight']:.2f}%"
            })
        comp_df = pd.DataFrame(comp_data)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        
        st.write("") # Spacer
        
        # Summary stats
        col_m1, col_m2 = st.columns(2, gap="medium")
        with col_m1:
            metric_card("Number of Funds", len(fund_data), None, "#A0A0A0")
            metric_card("Overall Return", f"{portfolio_overall_return:.2f}%", "Total Growth", get_neon_color(portfolio_overall_return, 'performance'))
        with col_m2:
            metric_card("Yearly Return (CAGR)", f"{portfolio_cagr:.2f}%", "Annualized", get_neon_color(portfolio_cagr, 'performance'))
            # Calculate time period
            years = (portfolio_nav.index[-1] - portfolio_nav.index[0]).days / 365.25
            metric_card("Time Period", f"{years:.1f} years", "Duration", "#A0A0A0")
    
    with col2:
        # Enhanced pie chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=[name[:25] for name in fund_data.keys()],
            values=[data['weight'] for data in fund_data.values()],
            hole=0.4,
            textinfo='label+percent',
            textposition='auto'
        )])
        fig_pie.update_layout(
            title="Allocation Breakdown",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # --- AI SUMMARY ---
        from analytics import generate_portfolio_summary
        
        # Calculate quick metrics for summary
        p_ret_series = portfolio_nav.pct_change(fill_method=None).dropna()
        p_mean = p_ret_series.mean() * 252
        p_std = p_ret_series.std() * np.sqrt(252)
        p_sharpe = (p_mean - 0.06) / p_std if p_std > 0 else 0
        
        summary_metrics = {
            'cagr': calculate_cagr(portfolio_nav),
            'volatility': p_std,
            'sharpe': p_sharpe,
            'beta': 1.0 # Placeholder until full benchmark analysis
        }
        
        ai_insight = generate_portfolio_summary(summary_metrics)
        
        st.info(f"💡 **AI Insight**: {ai_insight}")
    
    st.divider()
    
    # PERFORMANCE COMPARISON CHART (keep chart, remove table)
    st.markdown("### 📈 Performance Comparison")
    st.caption("Portfolio vs selected benchmarks (rebased to 100)")
    
    fig_perf = go.Figure()
    
    # Add portfolio
    portfolio_rebased = (portfolio_nav / portfolio_nav.iloc[0]) * 100
    fig_perf.add_trace(go.Scatter(
        x=portfolio_rebased.index,
        y=portfolio_rebased.values,
        mode='lines',
        name='Portfolio',
        line=dict(color='#8b5cf6', width=3)
    ))
    
    # Add benchmarks
    bench_colors = ['#10b981', '#f59e0b', '#ef4444', '#3b82f6']
    for idx, (bench_name, bench_series) in enumerate(benchmark_data.items()):
        # Align to portfolio dates
        aligned = bench_series[(bench_series.index >= portfolio_nav.index[0]) & 
                               (bench_series.index <= portfolio_nav.index[-1])]
        if not aligned.empty:
            rebased = (aligned / aligned.iloc[0]) * 100
            fig_perf.add_trace(go.Scatter(
                x=rebased.index,
                y=rebased.values,
                mode='lines',
                name=bench_name,
                line=dict(color=bench_colors[idx % len(bench_colors)], width=2, dash='dot')
            ))
    
    fig_perf.update_layout(
        xaxis_title="Date",
        yaxis_title="Value (₹)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        hovermode='x unified',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_perf, use_container_width=True)
    
    st.divider()
    
    # PORTFOLIO VS BENCHMARK COMPARISON
    st.markdown("### 📊 Portfolio vs Benchmark Comparison")
    
    # Tenure selector
    tenure_selector = st.selectbox(
        "Select Time Period for Comparison",
        ["1 Year", "3 Years", "5 Years", "10 Years", "Max"],
        index=4,  # Default to Max
        key="portfolio_comparison_tenure"
    )
    
    # Filter data based on tenure
    tenure_map = {"1 Year": 1, "3 Years": 3, "5 Years": 5, "10 Years": 10, "Max": None}
    selected_tenure = tenure_map[tenure_selector]
    
    # Filter portfolio NAV
    filtered_portfolio_nav = portfolio_nav.copy()
    if selected_tenure:
        end_date = filtered_portfolio_nav.index[-1]
        start_date = end_date - pd.DateOffset(years=selected_tenure)
        filtered_portfolio_nav = filtered_portfolio_nav[filtered_portfolio_nav.index >= start_date]
    
    # Filter benchmark data
    filtered_benchmark_data = {}
    for bench_name, bench_series in benchmark_data.items():
        filtered_bench = bench_series.copy()
        if selected_tenure:
            end_date = filtered_bench.index[-1]
            start_date = end_date - pd.DateOffset(years=selected_tenure)
            filtered_bench = filtered_bench[filtered_bench.index >= start_date]
        filtered_benchmark_data[bench_name] = filtered_bench
    
    # Comparison table
    st.markdown("#### Metrics Comparison Table")
    st.caption(f"Metrics calculated over {tenure_selector}")
    
    comparison_data = []
    
    # Portfolio row
    # Use first benchmark for portfolio metrics
    first_bench_name = list(filtered_benchmark_data.keys())[0] if filtered_benchmark_data else None
    first_bench_series = filtered_benchmark_data[first_bench_name] if first_bench_name else pd.Series()
    
    portfolio_metrics = get_fund_metrics(filtered_portfolio_nav, first_bench_series) if not first_bench_series.empty else {}
    
    comparison_data.append({
        'Asset': 'Portfolio',
        'CAGR': f"{portfolio_metrics.get('CAGR', 0)*100:.2f}%",
        'Sharpe': f"{portfolio_metrics.get('Sharpe Ratio', 0):.2f}",
        'Alpha': f"{portfolio_metrics.get('Alpha', 0)*100:.2f}%",
        'Beta': f"{portfolio_metrics.get('Beta', 0):.2f}",
        'Volatility': f"{portfolio_metrics.get('Volatility', 0)*100:.2f}%",
        'Max DD': f"{portfolio_metrics.get('Max Drawdown', 0)*100:.2f}%"
    })
    
    # Benchmark rows
    for bench_name, bench_series in filtered_benchmark_data.items():
        # Calculate benchmark metrics against itself (for consistency)
        bench_metrics = get_fund_metrics(bench_series, bench_series)
        
        comparison_data.append({
            'Asset': bench_name,
            'CAGR': f"{bench_metrics.get('CAGR', 0)*100:.2f}%",
            'Sharpe': f"{bench_metrics.get('Sharpe Ratio', 0):.2f}",
            'Alpha': "0.00%",  # Benchmark alpha vs itself is 0
            'Beta': "1.00",     # Benchmark beta vs itself is 1
            'Volatility': f"{bench_metrics.get('Volatility', 0)*100:.2f}%",
            'Max DD': f"{bench_metrics.get('Max Drawdown', 0)*100:.2f}%"
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    # Comparison charts
    st.markdown("#### Metrics Comparison Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CAGR comparison
        assets = [row['Asset'] for row in comparison_data]
        cagr_values = [float(row['CAGR'].rstrip('%')) for row in comparison_data]
        colors_list = ['#8b5cf6'] + bench_colors[:len(benchmark_data)]
        
        fig_cagr = go.Figure(data=[
            go.Bar(
                x=assets,
                y=cagr_values,
                marker_color=colors_list,
                text=[f"{v:.2f}%" for v in cagr_values],
                textposition='auto'
            )
        ])
        fig_cagr.update_layout(
            title="CAGR Comparison",
            yaxis_title="CAGR (%)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig_cagr, use_container_width=True)
    
    with col2:
        # Sharpe Ratio comparison
        sharpe_values = [float(row['Sharpe']) for row in comparison_data]
        
        fig_sharpe = go.Figure(data=[
            go.Bar(
                x=assets,
                y=sharpe_values,
                marker_color=colors_list,
                text=[f"{v:.2f}" for v in sharpe_values],
                textposition='auto'
            )
        ])
        fig_sharpe.update_layout(
            title="Sharpe Ratio Comparison",
            yaxis_title="Sharpe Ratio",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig_sharpe, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Volatility comparison
        vol_values = [float(row['Volatility'].rstrip('%')) for row in comparison_data]
        
        fig_vol = go.Figure(data=[
            go.Bar(
                x=assets,
                y=vol_values,
                marker_color=colors_list,
                text=[f"{v:.2f}%" for v in vol_values],
                textposition='auto'
            )
        ])
        fig_vol.update_layout(
            title="Volatility Comparison",
            yaxis_title="Volatility (%)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col4:
        # Max Drawdown comparison
        dd_values = [abs(float(row['Max DD'].rstrip('%'))) for row in comparison_data]
        
        fig_dd_comp = go.Figure(data=[
            go.Bar(
                x=assets,
                y=dd_values,
                marker_color=colors_list,
                text=[f"{v:.2f}%" for v in dd_values],
                textposition='auto'
            )
        ])
        fig_dd_comp.update_layout(
            title="Max Drawdown Comparison",
            yaxis_title="Max Drawdown (%)",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig_dd_comp, use_container_width=True)
    
    st.divider()
    
    # DRAWDOWN ANALYSIS
    st.markdown("### 📉 Drawdown Analysis")
    
    fig_dd = go.Figure()
    
    # Portfolio drawdown
    portfolio_dd = calculate_max_drawdown_series(portfolio_nav)
    if not portfolio_dd.empty:
        fig_dd.add_trace(go.Scatter(
            x=portfolio_dd.index,
            y=portfolio_dd.values * 100,
            mode='lines',
            name='Portfolio',
            line=dict(color='#8b5cf6', width=3),
            fill='tozeroy',
            fillcolor='rgba(139, 92, 246, 0.2)'
        ))
    
    # Benchmark drawdowns
    for idx, (bench_name, bench_series) in enumerate(benchmark_data.items()):
        bench_dd = calculate_max_drawdown_series(bench_series)
        if not bench_dd.empty:
            # Align to portfolio dates
            aligned_dd = bench_dd[(bench_dd.index >= portfolio_nav.index[0]) & 
                                  (bench_dd.index <= portfolio_nav.index[-1])]
            if not aligned_dd.empty:
                fig_dd.add_trace(go.Scatter(
                    x=aligned_dd.index,
                    y=aligned_dd.values * 100,
                    mode='lines',
                    name=bench_name,
                    line=dict(color=bench_colors[idx % len(bench_colors)], width=2, dash='dot')
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
    
    st.divider()
    
    # ROLLING RETURNS ANALYSIS
    st.markdown("### 📊 Rolling Returns Analysis")
    
    period_selector = st.selectbox(
        "Select Rolling Period",
        ["1 Year", "3 Years", "5 Years"],
        key="portfolio_rolling_period"
    )
    
    period_map = {"1 Year": 1, "3 Years": 3, "5 Years": 5}
    selected_years = period_map[period_selector]
    
    # Rolling returns chart
    fig_rolling = go.Figure()
    
    # Portfolio rolling returns
    portfolio_rolling = calculate_rolling_returns(portfolio_nav, window_years=selected_years)
    if not portfolio_rolling.empty:
        fig_rolling.add_trace(go.Scatter(
            x=portfolio_rolling.index,
            y=portfolio_rolling.values * 100,
            mode='lines',
            name='Portfolio',
            line=dict(color='#8b5cf6', width=3)
        ))
    
    # Benchmark rolling returns
    for idx, (bench_name, bench_series) in enumerate(benchmark_data.items()):
        bench_rolling = calculate_rolling_returns(bench_series, window_years=selected_years)
        if not bench_rolling.empty:
            fig_rolling.add_trace(go.Scatter(
                x=bench_rolling.index,
                y=bench_rolling.values * 100,
                mode='lines',
                name=bench_name,
                line=dict(color=bench_colors[idx % len(bench_colors)], width=2, dash='dot')
            ))
    
    fig_rolling.update_layout(
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
    st.plotly_chart(fig_rolling, use_container_width=True)
    
    # Rolling returns statistics table
    st.markdown("#### Rolling Returns Statistics")
    st.caption(f"Statistics for {period_selector} rolling returns")
    
    rolling_stats = []
    
    # Portfolio stats
    if not portfolio_rolling.empty:
        rolling_stats.append({
            'Asset': 'Portfolio',
            'Average': f"{portfolio_rolling.mean() * 100:.2f}%",
            'Median': f"{portfolio_rolling.median() * 100:.2f}%",
            'Std Dev': f"{portfolio_rolling.std() * 100:.2f}%",
            'Min': f"{portfolio_rolling.min() * 100:.2f}%",
            'Max': f"{portfolio_rolling.max() * 100:.2f}%"
        })
    
    # Benchmark stats
    for bench_name, bench_series in benchmark_data.items():
        bench_rolling = calculate_rolling_returns(bench_series, window_years=selected_years)
        if not bench_rolling.empty:
            rolling_stats.append({
                'Asset': bench_name,
                'Average': f"{bench_rolling.mean() * 100:.2f}%",
                'Median': f"{bench_rolling.median() * 100:.2f}%",
                'Std Dev': f"{bench_rolling.std() * 100:.2f}%",
                'Min': f"{bench_rolling.min() * 100:.2f}%",
                'Max': f"{bench_rolling.max() * 100:.2f}%"
            })
    
    rolling_stats_df = pd.DataFrame(rolling_stats)
    st.dataframe(rolling_stats_df, use_container_width=True, hide_index=True)
    
    st.divider()

    # MONTE CARLO SIMULATION
    st.markdown("### 🔮 Portfolio Future Projections")
    st.caption("Simulate future portfolio value using Monte Carlo methods")
    
    col_mc1, col_mc2 = st.columns([1, 2])
    
    with col_mc1:
        st.subheader("Simulation Settings")
        mc_inv_amt_port = st.number_input("Current Portfolio Value (₹)", min_value=1000, value=100000, step=1000, key="port_mc_val")
        mc_years_port = st.slider("Projection Horizon (Years)", 1, 20, 5, key="port_mc_years")
        mc_sims_port = st.select_slider("Number of Simulations", options=[100, 500, 1000, 2000, 5000], value=1000, key="port_mc_sims")
        
        if st.button("Run Portfolio Simulation", type="primary", key="btn_port_mc"):
            with st.spinner("Running Portfolio Monte Carlo Simulation..."):
                sim_results = run_monte_carlo_simulation(portfolio_nav, n_simulations=mc_sims_port, time_horizon_years=mc_years_port, initial_investment=mc_inv_amt_port)
                st.session_state['port_mc_results'] = sim_results
    
    with col_mc2:
        if 'port_mc_results' in st.session_state and st.session_state['port_mc_results']:
            res = st.session_state['port_mc_results']
            stats = res['stats']
            
            # Metrics Row (Using metric_card for consistency)
            m1, m2, m3 = st.columns(3)
            with m1:
                metric_card("Expected Value", f"₹{stats['expected_price']:,.0f}", f"{stats['expected_cagr']:.2f}% CAGR", "#3b82f6")
            with m2:
                metric_card("Optimistic (95%)", f"₹{stats['optimistic_price']:,.0f}", f"{stats['optimistic_cagr']:.2f}% CAGR", "#4ADE80")
            with m3:
                metric_card("Pessimistic (5%)", f"₹{stats['pessimistic_price']:,.0f}", f"{stats['pessimistic_cagr']:.2f}% CAGR", "#F87171")
            
            # Chart
            try:
                fig_mc = go.Figure()
                
                # Add a few raw paths (faint)
                if 'paths' in res and res['paths'] is not None:
                    for i in range(min(50, res['paths'].shape[1])):
                        fig_mc.add_trace(go.Scatter(
                            x=res['dates'],
                            y=res['paths'][:, i],
                            mode='lines',
                            line=dict(color='rgba(255, 255, 255, 0.05)', width=1),
                            showlegend=False,
                            hoverinfo='skip'
                        ))
                
                # Percentiles
                fig_mc.add_trace(go.Scatter(
                    x=res['dates'], y=res['p95'], mode='lines', 
                    name='95th Percentile',
                    line=dict(color='#4ADE80', width=2, dash='dash')
                ))
                
                fig_mc.add_trace(go.Scatter(
                    x=res['dates'], y=res['mean'], mode='lines', 
                    name='Expected Value',
                    line=dict(color='#3b82f6', width=3)
                ))
                
                fig_mc.add_trace(go.Scatter(
                    x=res['dates'], y=res['p5'], mode='lines', 
                    name='5th Percentile',
                    line=dict(color='#F87171', width=2, dash='dash')
                ))
                
                fig_mc.update_layout(
                    title=f"Portfolio Value Projection ({mc_years_port} Years)",
                    xaxis_title="Date",
                    yaxis_title="Projected Value (₹)",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=450,
                    hovermode='x unified',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig_mc, use_container_width=True)
            except Exception as e:
                 st.error(f"Error rendering portfolio chart: {str(e)}")
        else:
            st.info("👈 Run simulation to see future portfolio projections.")

    st.divider()
    
    # INVESTMENT CALCULATOR
    st.markdown("### 💰 Portfolio Investment Calculator")
    st.caption("See how your investment would have grown in this portfolio")
    
    calc_tab1, calc_tab2, calc_tab3 = st.tabs(["SIP Investment", "Lumpsum Investment", "Goal Planning"])
    
    # SIP Calculator
    with calc_tab1:
        col_sip1, col_sip2 = st.columns(2)
        with col_sip1:
            sip_amount = st.number_input("Monthly SIP Amount (₹)", min_value=500, value=10000, step=500, key="portfolio_sip")
        with col_sip2:
            sip_tenure = st.number_input("Investment Tenure (Years)", min_value=1, max_value=20, value=5, step=1, key="portfolio_sip_tenure")
        
        if st.button("Calculate SIP Returns", key="calc_portfolio_sip"):
            from analytics import calculate_sip_returns
            
            inv, curr, abs_ret, xirr = calculate_sip_returns(portfolio_nav, sip_amount, sip_tenure)
            
            # Display metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Invested Amount", f"₹{inv:,.0f}")
            c2.metric("Current Value", f"₹{curr:,.0f}")
            c3.metric("Absolute Return", f"{abs_ret:.2f}%")
            c4.metric("XIRR", f"{xirr:.2f}%")
            
            # Visualization
            breakdown_df = calculate_sip_returns(portfolio_nav, sip_amount, sip_tenure, return_breakdown=True)
            if not breakdown_df.empty:
                fig_sip = go.Figure()
                fig_sip.add_trace(go.Scatter(
                    x=breakdown_df['Date'], 
                    y=breakdown_df['Invested'],
                    mode='lines',
                    name='Invested Amount',
                    line=dict(color='#94a3b8', width=2, dash='dot'),
                    fill='tozeroy',
                    fillcolor='rgba(148, 163, 184, 0.1)'
                ))
                fig_sip.add_trace(go.Scatter(
                    x=breakdown_df['Date'], 
                    y=breakdown_df['Value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#8b5cf6', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(139, 92, 246, 0.2)'
                ))
                fig_sip.update_layout(
                    title="SIP Investment Growth in Portfolio",
                    xaxis_title="Date",
                    yaxis_title="Amount (₹)",
                    template="plotly_dark",
                    height=400,
                    hovermode='x unified',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig_sip, use_container_width=True)
                
                st.success(f"💡 With a monthly SIP of ₹{sip_amount:,}, your ₹{inv:,} investment would have grown to ₹{curr:,.0f} in {sip_tenure} years!")
    
    # Lumpsum Calculator
    with calc_tab2:
        col_lump1, col_lump2 = st.columns(2)
        with col_lump1:
            lump_amount = st.number_input("Lumpsum Amount (₹)", min_value=10000, value=100000, step=10000, key="portfolio_lump")
        with col_lump2:
            lump_tenure = st.number_input("Investment Tenure (Years)", min_value=1, max_value=20, value=5, step=1, key="portfolio_lump_tenure")
        
        if st.button("Calculate Lumpsum Returns", key="calc_portfolio_lump"):
            from analytics import calculate_lumpsum_returns
            
            curr, abs_ret, cagr = calculate_lumpsum_returns(portfolio_nav, lump_amount, lump_tenure)
            
            # Display metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Invested Amount", f"₹{lump_amount:,.0f}")
            c2.metric("Current Value", f"₹{curr:,.0f}")
            c3.metric("Absolute Return", f"{abs_ret:.2f}%")
            c4.metric("CAGR", f"{cagr:.2f}%")
            
            # Visualization
            limited_series = portfolio_nav.copy()
            if lump_tenure and lump_tenure > 0:
                end_date = limited_series.index[-1]
                start_date = end_date - pd.DateOffset(years=lump_tenure)
                limited_series = limited_series[limited_series.index >= start_date]
            
            if not limited_series.empty:
                start_nav = limited_series.iloc[0]
                units = lump_amount / start_nav
                value_series = limited_series * units
                
                fig_lump = go.Figure()
                fig_lump.add_trace(go.Scatter(
                    x=value_series.index,
                    y=[lump_amount] * len(value_series),
                    mode='lines',
                    name='Invested Amount',
                    line=dict(color='#94a3b8', width=2, dash='dot')
                ))
                fig_lump.add_trace(go.Scatter(
                    x=value_series.index,
                    y=value_series.values,
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#10b981', width=3),
                    fill='tonexty',
                    fillcolor='rgba(16, 185, 129, 0.2)'
                ))
                fig_lump.update_layout(
                    title="Lumpsum Investment Growth in Portfolio",
                    xaxis_title="Date",
                    yaxis_title="Amount (₹)",
                    template="plotly_dark",
                    height=400,
                    hovermode='x unified',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                st.plotly_chart(fig_lump, use_container_width=True)
                

    # Goal Planning Calculator
    with calc_tab3:
        st.caption("Plan for your future goals (e.g. House, Retirement).")
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            goal_amount = st.number_input("Target Amount (₹)", min_value=100000, value=10000000, step=100000, format="%d", key="goal_amount")
        with col_g2:
            goal_years = st.number_input("Time Horizon (Years)", min_value=1, max_value=30, value=10, step=1, key="goal_years")
            
        if st.button("Calculate Required SIP", key="calc_goal_sip"):
            from analytics import calculate_required_sip, calculate_cagr
            
            # Use Portfolio CAGR as expected return
            expected_return = calculate_cagr(portfolio_nav)
            if expected_return <= 0:
                expected_return = 0.12 # Default to 12% if history is bad/short
                st.warning("⚠️ Portfolio history is short/negative. Using default 12% return for projection.")
                
            req_sip = calculate_required_sip(goal_amount, goal_years, expected_return)
            
            # Display Result in big numbers
            st.metric("Required Monthly SIP", f"₹{req_sip:,.0f}", f"Target: ₹{goal_amount/10000000:.2f} Cr in {goal_years} yrs")
            
            st.info(f"💡 based on this portfolio's historical return of **{expected_return*100:.2f}%**, you need to invest **₹{req_sip:,.0f}/month** to reach your goal.")
            
            # Simple progress bar visualization
            # "Gap" Chart
            per_month_impact = req_sip * 12 * goal_years
            growth_impact = goal_amount - per_month_impact
            
            fig_goal = go.Figure(data=[go.Bar(
                x=['Principal Invested', 'Wealth Gained'],
                y=[per_month_impact, growth_impact],
                marker_color=['#94a3b8', '#10b981']
            )])
            fig_goal.update_layout(
                title="Goal Composition (Principal vs Growth)",
                template="plotly_dark",
                yaxis_title="Amount (₹)",
                showlegend=False,
                height=300
            )
            st.plotly_chart(fig_goal, use_container_width=True)

    st.divider()
    
    
    # Reconstruct DataFrame for advanced analytics
    if fund_series_dict:
        common_start = max([s.index[0] for s in fund_series_dict.values()])
        common_end = min([s.index[-1] for s in fund_series_dict.values()])
        aligned_data = {k: v[(v.index >= common_start) & (v.index <= common_end)] for k, v in fund_series_dict.items()}
        portfolio_df = pd.DataFrame(aligned_data)
        
        # --- CORRELATION MATRIX ---
        st.markdown("#### 📊 Fund Correlation Analysis")
        st.caption("Understanding how your funds move together - Lower correlation = Better diversification")
        
        # Calculate correlation matrix using pairwise complete observations
        correlation_matrix = calculate_correlation_matrix(portfolio_df)
        
        if not correlation_matrix.empty:
            # Create heatmap
            fig_corr = go.Figure(data=go.Heatmap(
                z=correlation_matrix.values,
                x=[name[:25] for name in correlation_matrix.columns],
                y=[name[:25] for name in correlation_matrix.columns],
                colorscale='RdBu_r',  # Red = High correlation (bad), Blue = Low/Negative (good)
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
                height=500,
                xaxis={'side': 'bottom'},
                yaxis={'autorange': 'reversed'}
            )
            
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Interpretation guide
            col_guide1, col_guide2, col_guide3 = st.columns(3)
            with col_guide1:
                st.markdown("🔵 **Low Correlation (< 0.3)**")
                st.caption("Good diversification - Funds move independently")
            with col_guide2:
                st.markdown("🟡 **Medium Correlation (0.3-0.7)**")
                st.caption("Moderate diversification - Some overlap")
            with col_guide3:
                st.markdown("🔴 **High Correlation (> 0.7)**")
                st.caption("Poor diversification - Funds move together")
        
        
        st.divider()
        st.markdown("---")
        
        # --- RISK LABS: SCENARIO ANALYSIS ---
        st.markdown("### 🧪 Risk Labs - Stress Testing")
        st.caption("See how your portfolio would have performed during major market crashes")
        
        # Get predefined scenarios
        scenarios = get_predefined_scenarios()
        
        # Scenario selector
        scenario_names = [s['name'] for s in scenarios]
        selected_scenario_name = st.selectbox(
            "Select Market Event",
            scenario_names,
            key="scenario_selector"
        )
        
        # Find selected scenario
        selected_scenario = next((s for s in scenarios if s['name'] == selected_scenario_name), None)
        
        if selected_scenario:
            st.info(f"📅 **{selected_scenario['name']}** ({selected_scenario['start_date']} to {selected_scenario['end_date']})\n\n{selected_scenario['description']}")
            
            # Get first benchmark for scenario analysis
            if benchmark_data:
                first_benchmark_series = list(benchmark_data.values())[0]
                
                # Run scenario analysis for portfolio
                portfolio_result = simulate_market_scenario(
                    portfolio_nav,
                    first_benchmark_series,
                    selected_scenario
                )
                
                if portfolio_result['data_available']:
                    # Display results
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Portfolio Return",
                            f"{portfolio_result['portfolio_return']:.2f}%",
                            delta=f"{portfolio_result['portfolio_return'] - portfolio_result['benchmark_return']:.2f}% vs Benchmark"
                        )
                    
                    with col2:
                        st.metric(
                            "Max Drawdown",
                            f"{portfolio_result['portfolio_max_drawdown']:.2f}%",
                            delta=f"{portfolio_result['portfolio_max_drawdown'] - portfolio_result['benchmark_max_drawdown']:.2f}% vs Benchmark",
                            delta_color="inverse"
                        )
                    
                    with col3:
                        recovery_text = f"{portfolio_result['days_to_recover']} days" if portfolio_result['days_to_recover'] else "Not yet recovered"
                        st.metric(
                            "Recovery Time",
                            recovery_text
                        )
                    
                    # Comparison table
                    st.markdown("#### 📊 Detailed Comparison")
                    comparison_data = {
                        'Metric': ['Return During Event', 'Maximum Drawdown', 'Recovery Time (Days)'],
                        'Your Portfolio': [
                            f"{portfolio_result['portfolio_return']:.2f}%",
                            f"{portfolio_result['portfolio_max_drawdown']:.2f}%",
                            str(portfolio_result['days_to_recover']) if portfolio_result['days_to_recover'] else "N/A"
                        ],
                        'Benchmark': [
                            f"{portfolio_result['benchmark_return']:.2f}%",
                            f"{portfolio_result['benchmark_max_drawdown']:.2f}%",
                            "-"
                        ]
                    }
                    
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # Interpretation
                    if portfolio_result['outperformed_benchmark']:
                        st.success("✅ **Your portfolio outperformed the benchmark during this crisis!** This suggests good defensive characteristics.")
                    else:
                        st.warning("⚠️ **Your portfolio underperformed the benchmark during this crisis.** Consider adding more defensive assets.")
                    
                else:
                    st.warning(f"⚠️ Insufficient data available for the selected scenario period ({selected_scenario['start_date']} to {selected_scenario['end_date']})")
            else:
                st.warning("⚠️ Please select at least one benchmark to run scenario analysis.")

