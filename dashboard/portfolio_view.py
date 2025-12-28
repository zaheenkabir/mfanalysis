import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add backend and local dashboard to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
backend_path = os.path.abspath(os.path.join(current_dir, '..', 'backend'))
if backend_path not in sys.path:
    sys.path.append(backend_path)

from data_loader import fetch_latest_nav, fetch_fund_history, fetch_scheme_details
from analytics import (
    calculate_cagr, calculate_volatility, calculate_sharpe_ratio,
    calculate_max_drawdown, calculate_max_drawdown_series,
    calculate_beta_alpha, download_benchmark,
    calculate_correlation_matrix, calculate_rolling_returns,
    calculate_sip_returns, calculate_step_up_sip_returns, calculate_lumpsum_returns,
    create_weighted_portfolio, get_fund_metrics,
    simulate_efficient_frontier, run_monte_carlo_simulation,
    filter_by_period,
    generate_gold_insights,
    calculate_required_sip,
    calculate_required_sip_advanced,
    calculate_goal_success_probability,
    get_predefined_scenarios, simulate_market_scenario,
    generate_portfolio_summary
)
from ui_components import (
    add_max_drawdown_annotation, add_drawdown_vertical_line,
    animate_chart, render_smart_insights, render_rolling_smart_insights,
    render_verdict_card, style_financial_dataframe, metric_card
)
from verdict_engine import generate_portfolio_verdict, MarkovAnalyzer

# Portfolio view with complete analytics imports

def render_portfolio_view(nav_data, selected_funds):
    """Renders the portfolio builder and analysis view"""
    st.header("🎯 Build & Analyze Portfolio")
    st.caption("Create a custom portfolio with weighted funds and analyze against multiple benchmarks")
    
    # nav_data and selected_funds are now passed as parameters
    
    # Sidebar for weight allocation only
    with st.sidebar:
        st.header("Portfolio Weights")
        st.caption("Total must equal 100%")
        
        weights = {}
        for fund_display in selected_funds:
            qt_start = fund_display.rfind('(')
            name = fund_display[:qt_start].strip()
            
            weight = st.number_input(
                name,
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
            "NIFTY 50": "NIFTY 50",
            "Nifty Midcap 50": "NIFTY MIDCAP 50",
            "Nifty Smallcap 50": "NIFTY SMALLCAP 50",
            "Nifty Bank": "NIFTY BANK",
            "Gold (GoldBees)": "GOLD",
            "Silver (SilverBees)": "SILVER"
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
    portfolio_cagr = calculate_cagr(portfolio_nav) * 100
    
    # Fetch benchmark data early (needed for fund metrics calculation)
    benchmark_options = {
        "NIFTY 50": "NIFTY 50",
        "Nifty Midcap 50": "NIFTY MIDCAP 50",
        "Nifty Smallcap 50": "NIFTY SMALLCAP 50",
        "Nifty Bank": "NIFTY BANK",
        "Gold (GoldBees)": "GOLD",
        "Silver (SilverBees)": "SILVER"
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
                metadata = fetch_scheme_details(code, nav_data)
                
                # Calculate metrics for this fund
                # Use first selected benchmark for individual fund metrics comparison? 
                # Or just show absolute metrics for funds. Let's use the first one if available.
                first_benchmark = list(benchmark_data.values())[0] if benchmark_data else pd.Series()
                fund_metrics = get_fund_metrics(data['series'], first_benchmark) if not first_benchmark.empty else get_fund_metrics(data['series'])
                
                fund_details.append({
                    'Fund Name': name[:35],
                    'Weight': f"{data['weight']:.1f}%", # Keep weight as string or style it? Let's leave weight as string for now implies no coloring needed.
                    'CAGR': fund_metrics.get('CAGR', 0)*100,
                    'Sharpe': fund_metrics.get('Sharpe Ratio', 0),
                    'Alpha': fund_metrics.get('Alpha', 0)*100,
                    'Beta': fund_metrics.get('Beta', 0),
                    'Volatility': fund_metrics.get('Volatility', 0)*100,
                    'Max Drawdown': fund_metrics.get('Max Drawdown', 0)*100
                })
                break
    
    details_df = pd.DataFrame(fund_details)
    st.dataframe(style_financial_dataframe(details_df), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # PORTFOLIO COMPOSITION - ENHANCED
    st.markdown("### 📊 Portfolio Composition")
    
    # Calculate portfolio CAGR
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
            metric_card("Overall Return", f"{portfolio_overall_return:.2f}%", delta=portfolio_overall_return, is_good_if_positive=True)
        with col_m2:
            metric_card("Yearly Return (CAGR)", f"{portfolio_cagr:.2f}%", delta=portfolio_cagr, is_good_if_positive=True)
            # Calculate time period
            years = (portfolio_nav.index[-1] - portfolio_nav.index[0]).days / 365.25
            metric_card("Time Period", f"{years:.1f} years", None, "#A0A0A0")
    
    with col2:
        # Enhanced donut chart
        fig_pie = go.Figure(data=[go.Pie(
            labels=[name[:25] for name in fund_data.keys()],
            values=[data['weight'] for data in fund_data.values()],
            hole=0.6,
            textinfo='label+percent',
            textposition='outside',
            marker=dict(colors=['#8b5cf6', '#10b981', '#f59e0b', '#ef4444', '#3b82f6', '#06b6d4']),
            pull=[0.05] * len(fund_data)
        )])
        fig_pie.update_layout(
            title="Portfolio Allocation Breakdown",
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            showlegend=False,
            annotations=[dict(text='100%', x=0.5, y=0.5, font_size=20, showarrow=False, font_color="white")]
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # --- AI SUMMARY ---
        
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
        height=600,
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
    
    # Add outperformance shading if possible
    if len(benchmark_data) > 0:
        main_bench_name = list(benchmark_data.keys())[0]
        main_bench = benchmark_data[main_bench_name]
        
        # Align and rebase benchmark
        aligned_b_raw = main_bench[(main_bench.index >= portfolio_nav.index[0]) & (main_bench.index <= portfolio_nav.index[-1])]
        if not aligned_b_raw.empty:
            aligned_b = (aligned_b_raw / aligned_b_raw.iloc[0]) * 100
            
            # Find common index to ensure perfect alignment for fill
            common_idx = portfolio_rebased.index.intersection(aligned_b.index)
            if not common_idx.empty:
                p_vals = portfolio_rebased.loc[common_idx]
                b_vals = aligned_b.loc[common_idx]
                
                # Add fill between Portfolio and Primary Benchmark
                fig_perf.add_trace(go.Scatter(
                    x=common_idx.tolist() + common_idx.tolist()[::-1],
                    y=p_vals.tolist() + b_vals.tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(139, 92, 246, 0.15)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name=f'Diff vs {main_bench_name}'
                ))
    
    # --- ADD MAX DRAWDOWN ANNOTATION ---
    p_dd_series = calculate_max_drawdown_series(portfolio_nav)
    if not p_dd_series.empty:
        max_dd_date = p_dd_series.idxmin()
        # Get value at that date in the rebased series
        try:
            max_dd_val = portfolio_rebased.loc[max_dd_date]
            
            add_max_drawdown_annotation(fig_perf, max_dd_date, max_dd_val)
            add_drawdown_vertical_line(fig_perf, max_dd_date)
        except:
            pass

    fig_perf = animate_chart(fig_perf)
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
        'CAGR': portfolio_metrics.get('CAGR', 0)*100,
        'Sharpe': portfolio_metrics.get('Sharpe Ratio', 0),
        'Alpha': portfolio_metrics.get('Alpha', 0)*100,
        'Beta': portfolio_metrics.get('Beta', 0),
        'Volatility': portfolio_metrics.get('Volatility', 0)*100,
        'Max Drawdown': portfolio_metrics.get('Max Drawdown', 0)*100
    })
    
    # Benchmark rows
    for bench_name, bench_series in filtered_benchmark_data.items():
        # Calculate benchmark metrics against itself (for consistency)
        bench_metrics = get_fund_metrics(bench_series, bench_series)
        
        comparison_data.append({
            'Asset': bench_name,
            'CAGR': bench_metrics.get('CAGR', 0)*100,
            'Sharpe': bench_metrics.get('Sharpe Ratio', 0),
            'Alpha': 0.0,
            'Beta': 1.0,
            'Volatility': bench_metrics.get('Volatility', 0)*100,
            'Max Drawdown': bench_metrics.get('Max Drawdown', 0)*100
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(style_financial_dataframe(comparison_df), use_container_width=True, hide_index=True)
    
    # Comparison charts
    st.markdown("#### Metrics Comparison Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CAGR comparison
        assets = [row['Asset'] for row in comparison_data]
        cagr_values = [float(row['CAGR']) for row in comparison_data]
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
        fig_cagr = animate_chart(fig_cagr)
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
        fig_sharpe = animate_chart(fig_sharpe)
        st.plotly_chart(fig_sharpe, use_container_width=True)
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Volatility comparison
        vol_values = [float(row['Volatility']) for row in comparison_data]
        
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
        fig_vol = animate_chart(fig_vol)
        st.plotly_chart(fig_vol, use_container_width=True)
    
    with col4:
        # Max Drawdown comparison
        dd_values = [abs(float(row['Max Drawdown'])) for row in comparison_data]
        
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
        fig_dd_comp = animate_chart(fig_dd_comp)
        st.plotly_chart(fig_dd_comp, use_container_width=True)
    
    # --- PORTFOLIO HEALTH CHECK ---
    st.markdown("### 🏥 Portfolio Health Scorecard")
    st.caption(f"Health analysis based on the selected period: **{tenure_selector}**")
    
    # Filter data based on selected tenure (Dynamic Analysis)
    # We use the same 'selected_tenure' logic as the comparison table
    
    # 1. Prepare Filtered Series for Metrics
    p_series_health = portfolio_nav.copy()
    b_series_health = first_bench_series.copy()
    
    if selected_tenure:
        end_date = p_series_health.index[-1]
        start_date = end_date - pd.DateOffset(years=selected_tenure)
        p_series_health = p_series_health[p_series_health.index >= start_date]
        b_series_health = b_series_health[b_series_health.index >= start_date]
    
    # 2. Prepare Filtered Dataframe for Correlation
    health_df = pd.DataFrame(fund_series_dict).dropna()
    if selected_tenure:
        end_date = health_df.index[-1]
        start_date = end_date - pd.DateOffset(years=selected_tenure)
        health_df = health_df[health_df.index >= start_date]

    if not health_df.empty and not p_series_health.empty:
        # 1. DIVERSIFICATION SCORE
        health_corr = calculate_correlation_matrix(health_df)
        avg_corr = (health_corr.values.sum() - len(health_corr)) / (len(health_corr)**2 - len(health_corr)) if len(health_corr) > 1 else 1.0
        
        # 2. EFFICIENCY SCORE
        p_metrics_health = get_fund_metrics(p_series_health, b_series_health)
        b_metrics_health = get_fund_metrics(b_series_health, b_series_health)
        
        p_sharpe = p_metrics_health.get('Sharpe Ratio', 0)
        b_sharpe = b_metrics_health.get('Sharpe Ratio', 0)
        
        # 3. AGGRESSION SCORE
        beta = p_metrics_health.get('Beta', 1.0)
        
        col_h1, col_h2, col_h3 = st.columns(3)
        
        # Card 1: Diversification
        with col_h1:
            if avg_corr < 0.5:
                status = "✅ High Diversification"
                msg = "Funds act independently"
                color = "#10b981"
            elif avg_corr < 0.8:
                status = "⚠️ Moderate Overlap"
                msg = "Some funds move together"
                color = "#f59e0b"
            else:
                status = "❌ High Overlap"
                msg = "Funds move in lockstep"
                color = "#ef4444"
                
            st.metric("Correlation Score", f"{avg_corr:.2f}", help="Lower is better (0 to 1)")
            st.markdown(f"**{status}**")
            st.caption(msg)

        # Card 2: Efficiency (Sharpe)
        with col_h2:
            is_efficient = p_sharpe > b_sharpe
            status = "✅ Beating Benchmark" if is_efficient else "⚠️ Trailing Benchmark"
            diff = (p_sharpe - b_sharpe) / b_sharpe * 100 if b_sharpe != 0 else 0
            
            st.metric("Risk-Adjusted Quality", f"{p_sharpe:.2f}", delta=f"{diff:.0f}% vs Index")
            st.markdown(f"**{status}**")
            st.caption("Returns per unit of risk")

        # Card 3: Risk Profile
        with col_h3:
            if beta > 1.1:
                risk_label = "🔥 High Risk"
                advice = "Aggressive Growth"
            elif beta < 0.9:
                risk_label = "🛡️ Conservative"
                advice = "Capital Protection"
            else:
                risk_label = "⚖️ Balanced"
                advice = "Market Aligned"
                
            st.metric("Risk Profile (Beta)", f"{beta:.2f}")
            st.markdown(f"**{risk_label}**")
            st.caption(advice)
            
        st.divider()
        
        # Qualitative Verdict
        st.subheader("💡 Analysis Verdict")
        verdict_points = []
        
        if avg_corr > 0.8:
            verdict_points.append("🔴 **Diversify More:** Your funds are very similar. Consider adding different asset classes (e.g., Gold, Debt) or styles (Value vs Growth) to reduce risk.")
        
        if p_sharpe < b_sharpe:
            verdict_points.append(f"🟠 **Improve Efficiency:** Your portfolio takes more risk for the same return as the {first_bench_name}. Consider replacing high-volatility laggards.")
        else:
            verdict_points.append(f"🟢 **Strong Efficiency:** Your portfolio is generating superior risk-adjusted returns compared to the {first_bench_name}. Good job!")
            
        if p_metrics_health.get('Max Drawdown', 0) < b_metrics_health.get('Max Drawdown', 0):
             verdict_points.append("🛡️ **Good Defense:** Your portfolio falls less than the market during crashes.")
        
        for point in verdict_points:
            st.markdown(point)

    else:
        st.info("Insufficient data for full health check.")

    
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
    fig_dd = animate_chart(fig_dd)
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
    fig_rolling = animate_chart(fig_rolling)
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
    # Collect all rolling returns into a single DataFrame for easier statistics calculation
    rolling_returns_data = {'Portfolio': portfolio_rolling}
    for bench_name, bench_series in benchmark_data.items():
        bench_rolling = calculate_rolling_returns(bench_series, window_years=selected_years)
        if not bench_rolling.empty:
            rolling_returns_data[bench_name] = bench_rolling
    
    # Convert to DataFrame, aligning by index
    rolling_returns = pd.DataFrame(rolling_returns_data).dropna()
    
    stats_data = []
    for col in rolling_returns.columns:
        col_stats = rolling_returns[col].describe()
        stats_data.append({
            'Asset': col,
            'Avg Return': col_stats['mean']*100,
            'Median Return': col_stats['50%']*100,
            'Volatility': col_stats['std']*100,
            'Min Return': col_stats['min']*100,
            'Max Return': col_stats['max']*100
        })
    
    rolling_stats_df = pd.DataFrame(stats_data)
    st.dataframe(style_financial_dataframe(rolling_stats_df), use_container_width=True, hide_index=True)
    
    # Portfolio-specific Smart Insights
    if not portfolio_rolling.empty:
        # Calculate extra metrics needed for insights
        fund_min = portfolio_rolling.min() * 100
        neg_periods_pct = (portfolio_rolling < 0).mean() * 100
        
        target_annual = 0.12
        target_period = (1 + target_annual) ** selected_years - 1
        beating_target_pct = (portfolio_rolling > target_period).mean() * 100
        
        # Determine outperformance against the first benchmark if available
        outperformance_pct = 0
        if len(benchmark_data) > 0:
            first_bench_name = list(benchmark_data.keys())[0]
            first_bench_rolling = calculate_rolling_returns(benchmark_data[first_bench_name], window_years=selected_years)
            if not first_bench_rolling.empty:
                # Align
                aligned = pd.concat([portfolio_rolling, first_bench_rolling], axis=1).dropna()
                if not aligned.empty:
                    outperformance_pct = (aligned.iloc[:, 0] > aligned.iloc[:, 1]).mean() * 100
        
        port_stats_for_insights = {
            'outperformance_pct': outperformance_pct,
            'fund_min': fund_min,
            'negative_periods_pct': neg_periods_pct,
            'beating_target_pct': beating_target_pct
        }
        
        st.write("")
        render_rolling_smart_insights(port_stats_for_insights, selected_years)

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
                metric_card("Expected Value", f"₹{stats['expected_price']:,.0f}", delta=f"{stats['expected_cagr']:.2f}% CAGR")
            with m2:
                metric_card("Optimistic (95%)", f"₹{stats['optimistic_price']:,.0f}", delta=f"{stats['optimistic_cagr']:.2f}% CAGR")
            with m3:
                metric_card("Pessimistic (5%)", f"₹{stats['pessimistic_price']:,.0f}", delta=f"{stats['pessimistic_cagr']:.2f}% CAGR")
            
            # --- PROBABILITY ANALYSIS ---
            st.markdown("#### 🎯 Success Probability")
            
            if 'end_distribution' in res:
                end_vals = res['end_distribution']
                # --- PROBABILITY ANALYSIS ---
                target_cagr_15 = 15.0
                target_mult_15 = (1 + target_cagr_15/100) ** mc_years_port
                target_val_15 = mc_inv_amt_port * target_mult_15
                
                prob_success_15 = (end_vals >= target_val_15).mean() * 100
                prob_expected = (end_vals >= stats['expected_price']).mean() * 100
                prob_no_loss = (end_vals >= mc_inv_amt_port).mean() * 100
                
                p_col1, p_col2, p_col3 = st.columns(3)
                with p_col1:
                    st.write(f"🚀 **Target ({target_cagr_15}% CAGR):** `{prob_success_15:.1f}%` chance")
                    st.progress(prob_success_15/100)
                    if prob_success_15 > 50:
                        st.caption("✅ Good chance of hitting 15% target.")
                    else:
                        st.caption("⚠️ Significant performance needed for 15%.")
                with p_col2:
                    st.write(f"📈 **Expected Return:** `{prob_expected:.1f}%` chance")
                    st.progress(prob_expected/100)
                    st.caption("Probability of beating the Mean projection.")
                with p_col3:
                    st.write(f"🛡️ **Capital Protection:** `{prob_no_loss:.1f}%` chance")
                    st.progress(prob_no_loss/100)
                    if prob_no_loss > 95:
                        st.caption("💎 Strong downside protection.")
                    else:
                        st.caption("📉 Portfolio carries moderate market risk.")
                
                # --- TARGET REACH PREDICTION ---
                st.write("")
                st.markdown("##### 🎯 Target Reach Prediction")
                
                # Calculate years needed to reach a custom target
                # Use unique key for portfolio view
                custom_target_port = st.number_input("Goal Target Amount (₹)", min_value=mc_inv_amt_port, value=int(mc_inv_amt_port * 2), step=10000, key="port_goal_target")
                
                # Use expected CAGR to predict
                exp_cagr = stats['expected_cagr'] / 100
                if exp_cagr > 0:
                    years_needed = np.log(custom_target_port / mc_inv_amt_port) / np.log(1 + exp_cagr)
                    st.write(f"⏱️ At current performance, you may reach ₹{custom_target_port:,.0f} in approximately `{years_needed:.1f} years`.")
                    st.info(f"💡 Adjusting monthly contributions or selecting a higher alpha fund could shorten this to `{years_needed * 0.7:.1f} years`.")
                else:
                    st.warning("⚠️ Projected returns are negative. Goal unreachable at current rates.")
            
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
                fig_mc = animate_chart(fig_mc)
                st.plotly_chart(fig_mc, use_container_width=True)
            except Exception as e:
                 st.error(f"Error rendering portfolio chart: {str(e)}")
            
            # Move Smart Insights to Left Column (col_mc1)
            with col_mc1:
                st.divider()
                render_smart_insights(stats, prob_success_15, (100-prob_no_loss), mc_years_port)
        else:
            st.info("👈 Run simulation to see future portfolio projections.")

    st.divider()
    
    # INVESTMENT CALCULATOR
    st.markdown("### 💰 Portfolio Investment Calculator")
    st.caption("See how your investment would have grown in this portfolio (includes Step-up SIP)")
    
    # Portfolio Investment Calculator with Step-up SIP support
    calc_tab1, calc_tab2, calc_tab3 = st.tabs(["SIP Investment", "Step-up SIP", "Lumpsum Investment"])
    
    # SIP Calculator
    with calc_tab1:
        col_sip1, col_sip2 = st.columns(2)
        with col_sip1:
            sip_amount = st.number_input("Monthly SIP Amount (₹)", min_value=500, value=10000, step=500, key="portfolio_sip")
        with col_sip2:
            sip_tenure = st.number_input("Investment Tenure (Years)", min_value=1, max_value=20, value=5, step=1, key="portfolio_sip_tenure")
        
        mode_key = 'theoretical'
        
        if st.button("Calculate SIP Returns", key="calc_portfolio_sip", type="primary"):
            res = calculate_sip_returns(portfolio_nav, sip_amount, sip_tenure, mode=mode_key)
            inv, curr, abs_ret, xirr, actual_years = res
            
            # Display metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1: metric_card("Invested Amount", f"₹{inv:,.0f}")
            with c2: metric_card("Current Value", f"₹{curr:,.0f}")
            with c3: metric_card("Absolute Return", f"{abs_ret:.2f}%", delta=abs_ret)
            with c4: metric_card("XIRR", f"{xirr:.2f}%", delta=xirr)
            
            # Visualization
            breakdown_df = calculate_sip_returns(portfolio_nav, sip_amount, sip_tenure, return_breakdown=True, mode=mode_key)
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
                fig_sip = animate_chart(fig_sip)
                st.plotly_chart(fig_sip, use_container_width=True)
                
                st.success(f"📈 **Result**: With a monthly SIP of ₹{sip_amount:,}, your ₹{inv:,} investment would grow to ₹{curr:,.0f} in {sip_tenure} years (at {xirr:.2f}% CAGR).")
    
    # Step-up SIP Calculator
    with calc_tab2:
        st.caption("Increase your SIP amount annually to accelerate wealth creation")
        col_step1, col_step2, col_step3 = st.columns(3)
        with col_step1:
            step_sip_amount = st.number_input("Initial Monthly SIP (₹)", min_value=500, value=10000, step=500, key="portfolio_step_sip")
        with col_step2:
            step_up_percent = st.number_input("Annual Step-up (%)", min_value=0, max_value=50, value=10, step=5, key="portfolio_step_up")
        with col_step3:
            step_tenure = st.number_input("Tenure (Years)", min_value=1, max_value=20, value=5, step=1, key="portfolio_step_tenure")
        
        mode_key_step = 'theoretical'
        
        if st.button("Calculate Step-up SIP Returns", key="calc_portfolio_step_sip", type="primary"):
            res = calculate_step_up_sip_returns(portfolio_nav, step_sip_amount, step_up_percent, step_tenure, mode=mode_key_step)
            inv, curr, abs_ret, xirr, actual_years = res
            
            # Display metrics
            c1, c2, c3, c4 = st.columns(4)
            with c1: metric_card("Total Invested", f"₹{inv:,.0f}")
            with c2: metric_card("Current Value", f"₹{curr:,.0f}")
            with c3: metric_card("Absolute Return", f"{abs_ret:.2f}%", delta=abs_ret)
            with c4: metric_card("XIRR", f"{xirr:.2f}%", delta=xirr)
            
            # Visualization
            breakdown_df = calculate_step_up_sip_returns(portfolio_nav, step_sip_amount, step_up_percent, step_tenure, return_breakdown=True, mode=mode_key_step)
            if not breakdown_df.empty:
                fig_step = go.Figure()
                fig_step.add_trace(go.Scatter(
                    x=breakdown_df['Date'], 
                    y=breakdown_df['Invested'],
                    mode='lines',
                    name='Invested Amount',
                    line=dict(color='#94a3b8', width=2, dash='dot'),
                    fill='tozeroy',
                    fillcolor='rgba(148, 163, 184, 0.1)'
                ))
                fig_step.add_trace(go.Scatter(
                    x=breakdown_df['Date'], 
                    y=breakdown_df['Value'],
                    mode='lines',
                    name='Portfolio Value',
                    line=dict(color='#10b981', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(16, 185, 129, 0.2)'
                ))
                fig_step.update_layout(
                    title=f"Step-up SIP Growth ({step_up_percent}% annual increase)",
                    xaxis_title="Date",
                    yaxis_title="Amount (₹)",
                    template="plotly_dark",
                    height=400,
                    hovermode='x unified',
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                )
                fig_step = animate_chart(fig_step)
                st.plotly_chart(fig_step, use_container_width=True)
                
                st.success(f"📈 **Result**: Your ₹{inv:,.0f} total investment would grow to ₹{curr:,.0f} in {step_tenure} years (at {xirr:.2f}% CAGR).")
                
                # Calculate comparison with regular SIP
                comparison_res = calculate_sip_returns(portfolio_nav, step_sip_amount, step_tenure, mode=mode_key_step)
                regular_inv, regular_curr, _, _, _ = comparison_res
                extra_gain = curr - regular_curr
                st.info(f"✨ Step-up SIP generated ₹{extra_gain:,.0f} more than regular SIP with the same starting amount!")
    
    # Lumpsum Calculator
    with calc_tab3:
        col_lump1, col_lump2 = st.columns(2)
        with col_lump1:
            lump_amount = st.number_input("Lumpsum Amount (₹)", min_value=10000, value=100000, step=10000, key="portfolio_lump")
        with col_lump2:
            lump_tenure = st.number_input("Investment Tenure (Years)", min_value=1, max_value=20, value=5, step=1, key="portfolio_lump_tenure")
        
        mode_key_lump = 'theoretical'

        if st.button("Calculate Lumpsum Returns", key="calc_portfolio_lump", type="primary"):
            res = calculate_lumpsum_returns(portfolio_nav, lump_amount, lump_tenure, mode=mode_key_lump)
            curr, abs_ret, cagr, actual_years = res
            
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
                fig_lump = animate_chart(fig_lump)
                st.plotly_chart(fig_lump, use_container_width=True)
    
                

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
                x=[name for name in correlation_matrix.columns],
                y=[name for name in correlation_matrix.columns],
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
            st.markdown(f"> **Event Detail:** {selected_scenario['description']}")
            
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
                    # Display results - Consolidated Row like Single Fund View
                    s_col1, s_col2, s_col3 = st.columns(3)
                    
                    with s_col1:
                        st.metric(
                            "Portfolio Return",
                            f"{portfolio_result['portfolio_return']:.2f}%",
                            delta=f"{portfolio_result['portfolio_return'] - portfolio_result['benchmark_return']:.2f}% vs Benchmark"
                        )
                    
                    with s_col2:
                        st.metric(
                            "Max Drawdown",
                            f"{portfolio_result['portfolio_max_drawdown']:.2f}%",
                            delta=f"{portfolio_result['portfolio_max_drawdown'] - portfolio_result['benchmark_max_drawdown']:.2f}% vs Benchmark",
                            delta_color="inverse"
                        )
                    
                    with s_col3:
                        rec_text = f"{portfolio_result['days_to_recover']} days" if portfolio_result['days_to_recover'] else "Not yet recovered"
                        st.metric("Recovery Time", rec_text)
                    
                    # Detailed Comparison Table
                    st.markdown("#### 📊 Detailed Comparison")
                    comparison_data = {
                        'Metric': ['Return During Event', 'Maximum Drawdown', 'Recovery Time (Days)'],
                        'Your Portfolio': [
                            f"{portfolio_result['portfolio_return']:.2f}%",
                            f"{portfolio_result['portfolio_max_drawdown']:.2f}%",
                            str(portfolio_result['days_to_recover']) if portfolio_result['days_to_recover'] else "N/A"
                        ]
                    }
                    
                    # Add all selected benchmarks to comparison
                    for bench_name, bench_series in benchmark_data.items():
                        res_bench = simulate_market_scenario(bench_series, bench_series, selected_scenario)
                        if res_bench.get('data_available') or res_bench.get('bench_data_available'):
                             # simulate_market_scenario with (bench, bench) returns metrics in 'portfolio_...' keys roughly
                             # checking code: actually if bench is first arg, it returns 'portfolio_return'
                             val_ret = res_bench['portfolio_return']
                             val_dd = res_bench['portfolio_max_drawdown']
                             val_rec = res_bench['days_to_recover']
                             
                             comparison_data[bench_name] = [
                                 f"{val_ret:.2f}%",
                                 f"{val_dd:.2f}%",
                                 f"{val_rec} days" if val_rec else "Not yet recovered"
                             ]
                    st.dataframe(pd.DataFrame(comparison_data), use_container_width=True, hide_index=True)
                    
                    # Interpretation & Insights
                    st.markdown("#### 💡 Analysis & Insights")
                    if portfolio_result['outperformed_benchmark']:
                        st.success(f"✅ **Your portfolio outperformed the benchmark during this crisis!** It fell {abs(portfolio_result['portfolio_return'] - portfolio_result['benchmark_return']):.2f}% less than the index.")
                        st.info(f"📊 **What this means:** During {selected_scenario['name']}, your diversification strategy helped reduce losses. A portfolio that beats the benchmark during downturns typically has better risk-adjusted returns over the long term.")
                    else:
                        st.warning(f"⚠️ **Your portfolio underperformed the benchmark during this crisis.** It fell {abs(portfolio_result['portfolio_return'] - portfolio_result['benchmark_return']):.2f}% more than the index.")
                        st.info(f"📊 **What this means:** Your portfolio may have higher beta (market sensitivity) or concentrated exposure to sectors that were hit harder during {selected_scenario['name']}. Consider reviewing your asset allocation for better downside protection.")
                    
                    # Educational Context about the event
                    st.markdown("#### 📚 Historical Context")
                    event_context = {
                        "COVID-19 Crash (2020)": "The fastest 30% drop in history, triggered by global pandemic fears. Markets recovered to new highs within 6 months as central banks pumped liquidity.",
                        "Global Financial Crisis (2008-09)": "Triggered by US housing market collapse and Lehman Brothers bankruptcy. One of the worst crashes since 1929, taking over 5 years for full recovery.",
                        "IL&FS Crisis (2018)": "India-specific credit crisis that caused significant damage to mid and small-cap stocks. NBFCs and financial stocks were hit hardest.",
                        "Demonetization (2016)": "Sudden cash crunch caused short-term market volatility, but recovery was swift as fundamentals remained intact.",
                        "Taper Tantrum (2013)": "Fed's announcement to reduce QE caused emerging market selloff. India's Rupee fell sharply, impacting import-heavy sectors."
                    }
                    context_text = event_context.get(selected_scenario['name'], f"A significant market event during {selected_scenario['start_date']} to {selected_scenario['end_date']}.")
                    st.caption(f"📖 **{selected_scenario['name']}:** {context_text}")
                
                elif portfolio_result['bench_data_available']:
                    st.warning(f"⚠️ **Portfolio not active during this period.** The portfolio funds were not launched or data is missing for {selected_scenario['name']}.")
                    b_col_s1, b_col_s2 = st.columns(2)
                    with b_col_s1:
                         metric_card("Benchmark Return", f"{portfolio_result['benchmark_return']:.2f}%")
                    with b_col_s2:
                         metric_card("Benchmark Max DD", f"{portfolio_result['benchmark_max_drawdown']:.2f}%")
                else:
                    st.warning(f"⚠️ Insufficient data available for the selected scenario period.")
            else:
                st.warning("⚠️ Please select at least one benchmark to run scenario analysis.")

            # --- FINAL PORTFOLIO VERDICT ---
            st.write("")
            st.divider()
            st.header("🏆 Portfolio Health Certificate")
            st.caption("The Welment Engine evaluates your entire portfolio as a single unit, checking efficiency, risk, and future potential.")
            
            try:
                # Prepare portfolio metrics
                p_metrics = portfolio_metrics.copy()
                # Ensure Alpha/Beta are present
                p_metrics['Alpha'] = portfolio_metrics.get('Alpha', 0)
                p_metrics['Beta'] = portfolio_metrics.get('Beta', 1.0)
                
                # Check for rolling and MC results
                p_rolling = port_stats_for_insights if 'port_stats_for_insights' in locals() else None
                
                p_mc = None
                if 'res' in st.session_state.get('port_mc_results', {}) and 'stats' in st.session_state.get('port_mc_results', {}):
                    # res is end_distribution elsewhere, in portfolio view it's the whole dict
                    port_mc_data = st.session_state['port_mc_results']
                    p_mc = {
                        'prob_success_15': prob_success_15 if 'prob_success_15' in locals() else 0,
                        'prob_capital_protection': prob_no_loss if 'prob_no_loss' in locals() else 50
                    }
                
                # Calculate Markov Reliability for Portfolio
                try:
                    markov_bench = first_benchmark_series if 'first_benchmark_series' in locals() and not first_benchmark_series.empty else download_benchmark("NIFTY 50")
                    p_metrics['markov_reliability'] = MarkovAnalyzer.calculate_reliability_score(portfolio_nav, markov_bench)
                except:
                    p_metrics['markov_reliability'] = 50.0

                p_verdict = generate_portfolio_verdict(p_metrics, p_rolling, p_mc)
                render_verdict_card(p_verdict)
            except Exception as e:
                st.info("💡 Complete all simulations (Rolling & Monte Carlo) above to get your full Portfolio Health Certificate.")
                # st.error(f"Error generating portfolio verdict: {str(e)}")

