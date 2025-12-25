import streamlit as st
# Triggering full reload to catch UI component updates - Step-up SIP added
import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

# ML features availability (Default to False as engine is missing)
ML_AVAILABLE = False

# Add dashboard to path for module imports (keeping original if it's still needed)
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import fetch_latest_nav, fetch_fund_history, fetch_scheme_details
from analytics import (
    get_fund_metrics, calculate_cagr, calculate_volatility,
    calculate_sharpe_ratio, calculate_max_drawdown, calculate_max_drawdown_series,
    calculate_beta_alpha, calculate_rolling_returns,
    calculate_lumpsum_returns, calculate_sip_returns,
    calculate_step_up_sip_returns, filter_by_period,
    run_monte_carlo_simulation, calculate_rolling_returns_stats,
    calculate_capture_ratios, download_benchmark,
    get_fund_investment_insights,
    get_predefined_scenarios, simulate_market_scenario
)

# Import comparison view
from comparison_dashboard import render_comparison_dashboard

# Import portfolio view
from portfolio_view import render_portfolio_view

# Page Config
st.set_page_config(
    page_title="Welment Capital",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

from ui_components import (
    apply_custom_css, metric_card, get_neon_color, style_financial_dataframe,
    add_max_drawdown_annotation, add_drawdown_vertical_line,
    render_welcome_card
)

# Apply new SkillVista-inspired theme
apply_custom_css()

# Dashboard Components
def main():
    st.title("Welment Capital 💰")
    st.markdown("### Advanced Mutual Fund Analytics Dashboard")
    
    # MODE SELECTOR
    view_mode = st.radio(
        "📊 View Mode",
        ["Single Fund Analysis", "Compare Multiple Funds", "Build & Analyze Portfolio"],
        horizontal=True,
        key="view_mode_selector"
    )
    
    st.divider()
    
    # Load fund data (needed for sidebar in all views)
    with st.spinner("Loading Fund Universe..."):
        nav_data = fetch_latest_nav()
    
    if nav_data.empty:
        st.error("Failed to load fund data. Please check your internet connection.")
        return
    
    # Create fund list for sidebar (available to all views)
    nav_data['Display'] = nav_data['Scheme Name'] + " (" + nav_data['Scheme Code'].astype(str) + ")"
    funds_list = nav_data['Display'].tolist()

    # CREATE SIDEBAR FOR ALL VIEWS (before routing)
    st.sidebar.header("Fund Selection")
    
    # Show appropriate selector based on view mode
    if view_mode == "Single Fund Analysis":
        selected_option = st.sidebar.selectbox(
            "Search Fund", 
            funds_list,
            index=None,
            placeholder="Type to search...",
            key="fund_selector"
        )
    elif view_mode == "Compare Multiple Funds":
        selected_funds = st.sidebar.multiselect(
            "Choose 2-5 funds to compare",
            funds_list,
            key="multi_fund_selector",
            help="Select between 2 and 5 funds"
        )
    elif view_mode == "Build & Analyze Portfolio":
        selected_funds = st.sidebar.multiselect(
            "Select funds for portfolio",
            funds_list,
            key="portfolio_fund_selector",
            help="Select 2-10 funds"
        )
    
    st.sidebar.divider()
    
    # Route to appropriate view
    if view_mode == "Compare Multiple Funds":
        if len(selected_funds) < 2:
            render_welcome_card(len(nav_data))
            return
        elif len(selected_funds) > 5:
            st.warning("⚠️ Please select maximum 5 funds")
            return
        
        # Benchmark selection for comparison
        st.sidebar.subheader("Select Benchmarks")
        benchmark_options = {
            "NIFTY 50": "NIFTY 50",
            "Nifty Midcap 50": "NIFTY MIDCAP 50",
            "Nifty Smallcap 50": "NIFTY SMALLCAP 50",
            "Nifty Bank": "NIFTY BANK",
            "Gold (GoldBees)": "GOLD",
            "Silver (SilverBees)": "SILVER"
        }
        
        selected_benchmarks = st.sidebar.multiselect(
            "Choose Benchmarks",
            list(benchmark_options.keys()),
            default=["NIFTY 50"],
            key="comparison_benchmarks"
        )
        
        render_comparison_dashboard(nav_data=nav_data, selected_funds=selected_funds, selected_benchmarks=selected_benchmarks)
        return
    elif view_mode == "Build & Analyze Portfolio":
        if len(selected_funds) < 2:
            render_welcome_card(len(nav_data))
            return
        elif len(selected_funds) > 10:
            st.warning("⚠️ Please select maximum 10 funds")
            return
        render_portfolio_view(nav_data, selected_funds)
        return

    # SINGLE FUND VIEW

    if selected_option:
        # Extract scheme code
        qt_start = selected_option.rfind('(')
        qt_end = selected_option.rfind(')')
        scheme_code = selected_option[qt_start+1 : qt_end]
        scheme_name = selected_option[:qt_start].strip()


        # --- DATA FETCHING & BENCHMARK SELECTION ---
        with st.spinner("Fetching data & analytics..."):
            history_df_full = fetch_fund_history(scheme_code) # Fetch full history first
            metadata = fetch_scheme_details(scheme_code, nav_data) # Pass nav_data for metadata
            
            # SIDEBAR: Benchmark Selection for Single Fund
            st.sidebar.divider()
            st.sidebar.subheader("Benchmark Settings")
            
            benchmark_options = {
                "NIFTY 50": "NIFTY 50",
                "NIFTY Midcap 50": "NIFTY MIDCAP 50",
                "NIFTY Smallcap 50": "NIFTY SMALLCAP 50",
                "NIFTY Bank": "NIFTY BANK",
                "Gold (GoldBees)": "GOLD",
                "Silver (SilverBees)": "SILVER"
            }
            
            # Determine Default Benchmark based on Category
            category = metadata.get('scheme_category', '').lower()
            default_bench = "NIFTY 50"
            if 'small' in category:
                default_bench = "NIFTY Smallcap 50"
            elif 'mid' in category:
                default_bench = "NIFTY Midcap 50"
            elif 'bank' in category:
                default_bench = "NIFTY Bank"
                
            selected_bench_name = st.sidebar.selectbox(
                "Compare with Benchmark",
                list(benchmark_options.keys()),
                index=list(benchmark_options.keys()).index(default_bench),
                key="sf_benchmark_selector"
            )
            
            benchmark_name = selected_bench_name
            benchmark_ticker = benchmark_options[benchmark_name]
            
            benchmark_data_full = download_benchmark(benchmark_ticker)
            
            if benchmark_data_full.empty:
                st.sidebar.error(f"❌ Could not load data for {benchmark_name}. Please try another benchmark.")
        
        # --- HEADER SECTION ---
        st.header(f"{scheme_name}")
        st.caption(f"Scheme Code: {scheme_code} | Benchmark: {benchmark_name}")
        
        # TIME FRAME FILTER
        time_period = st.radio("Select Time Period", ["1Y", "3Y", "5Y", "10Y", "Max"], horizontal=True, index=4)
        
        # Filter Data
        history_df = history_df_full.copy()
        if not history_df.empty:
            filtered_nav = filter_by_period(history_df['nav'], time_period)
            history_df = history_df.loc[filtered_nav.index]
            history_df['nav'] = filtered_nav # Ensure correct series is set
            benchmark_data = filter_by_period(benchmark_data_full, time_period)
        else:
            benchmark_data = pd.Series()

        # Metadata Section (Improved)
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
             val = metadata.get('scheme_category', 'N/A')
             if val != 'N/A': st.info(f"**Category:** {val}")
        with m_col2:
             val = metadata.get('fund_house', 'N/A')
             if val != 'N/A': st.info(f"**Fund House:** {val}")
        with m_col3:
             # Using API metadata or last NAV from history
             nav_val = metadata.get('nav', 'N/A')
             if nav_val == 'N/A' and not history_df.empty:
                 nav_val = f"₹{history_df['nav'].iloc[-1]:.4f}"
             st.success(f"**Current NAV:** {nav_val}")


        if not history_df.empty:
            
            # Calculate metrics
            metrics = get_fund_metrics(history_df['nav'], benchmark_data)

            # --- SINGLE PAGE LAYOUT (No Tabs) ---
            
            if True: # --- HEADER SECTION (Was Overview) ---
                # --- HEADER SECTION ---
                col_h1, col_h2, col_h3 = st.columns([2, 1, 1])
                with col_h1:
                    st.markdown(f"""
                    <div style='margin-bottom: 20px;'>
                        <h2 style='color: white; margin: 0; font-size: 2rem;'>{scheme_name}</h2>
                        <div style='display: flex; gap: 12px; margin-top: 8px;'>
                            {f"<span style='background: #3b82f622; border: 1px solid #3b82f644; padding: 4px 12px; border-radius: 16px; color: #3b82f6; font-size: 0.8rem;'>{metadata.get('scheme_category')}</span>" if metadata.get('scheme_category') and metadata.get('scheme_category') != 'N/A' else ""}
                            {f"<span style='background: #10b98122; border: 1px solid #10b98144; padding: 4px 12px; border-radius: 16px; color: #10b981; font-size: 0.8rem;'>{metadata.get('fund_house')}</span>" if metadata.get('fund_house') and metadata.get('fund_house') != 'N/A' else ""}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_h3:
                     st.markdown(f"""
                    <div style='text-align: right;'>
                        <p style='color: #888; margin: 0; font-size: 0.8rem;'>Current NAV</p>
                        <h3 style='color: white; margin: 0; font-size: 1.8rem;'>₹{history_df['nav'].iloc[-1]:.2f}</h3>
                        <p style='color: #888; margin: 0; font-size: 0.8rem;'>{history_df.index[-1].strftime('%d %b %Y')}</p>
                    </div>
                    """, unsafe_allow_html=True)

                st.divider()
                
                # --- PERFORMANCE METRICS ---
                st.markdown(f"<h4 style='color: white; margin-bottom: 20px;'>Performance Overview ({time_period})</h4>", unsafe_allow_html=True)
                
                # Row 1: Key Performance
                c1, c2, c3, c4 = st.columns(4)
                
                with c1:
                    cagr = metrics.get('CAGR', 0)*100
                    metric_card("CAGR", f"{cagr:.2f}%", delta=cagr, is_good_if_positive=True)
                
                with c2:
                    alpha = metrics.get('Alpha', 0)*100
                    metric_card("Alpha", f"{alpha:.2f}%", delta=alpha, is_good_if_positive=True)
                
                with c3:
                    sharpe = metrics.get('Sharpe Ratio', 0)
                    metric_card("Sharpe Ratio", f"{sharpe:.2f}", delta=sharpe, is_good_if_positive=True)
                    
                with c4:
                    dd = metrics.get('Max Drawdown', 0)*100
                    # Drawdown is negative. Less negative (closer to 0) is better? 
                    # Usually we treat deeper drawdown (more negative) as "Bad".
                    metric_card("Max Drawdown", f"{abs(dd):.2f}%", delta=dd, is_good_if_positive=True) 

                st.write("") # Spacer

                # Row 2: Secondary Stats
                c5, c6, c7, c8 = st.columns(4)
                
                with c5:
                    vol = metrics.get('Volatility', 0)*100
                    # Lower volatility is generally "good" for risk
                    metric_card("Volatility", f"{vol:.2f}%", delta=vol, is_good_if_positive=False)
                
                with c6:
                    beta = metrics.get('Beta', 0)
                    metric_card("Beta", f"{beta:.2f}", delta=beta, is_good_if_positive=False)
                    
                with c7:
                    # Sortino Calculation
                    aligned_df = pd.concat([history_df['nav'], benchmark_data], axis=1).dropna()
                    fund_returns = aligned_df.iloc[:, 0].pct_change().dropna()
                    downside_returns = fund_returns[fund_returns < 0]
                    downside_std = downside_returns.std() * (252 ** 0.5) if len(downside_returns) > 0 else 0
                    sortino = (metrics.get('CAGR', 0) / downside_std) if downside_std > 0 else 0
                    
                    metric_card("Sortino", f"{sortino:.2f}", delta=sortino, is_good_if_positive=True)
                    
                with c8:
                    win_rate = (fund_returns > aligned_df.iloc[:, 1].pct_change().dropna()).mean() * 100 if not aligned_df.empty else 0
                    metric_card("Win Rate", f"{win_rate:.0f}%", delta=win_rate-50, is_good_if_positive=True)

                st.divider()

                # --- INVESTMENT INSIGHTS SECTION ---
                st.markdown("<h4 style='color: white; margin-bottom: 20px;'>💡 Smart Insights for Investors</h4>", unsafe_allow_html=True)
                
                insights = get_fund_investment_insights(metrics)
                ins_cols = st.columns(len(insights) if insights else 1)
                
                for i, insight in enumerate(insights):
                    with ins_cols[i]:
                        bg_color = "rgba(16, 185, 129, 0.1)" if insight['type'] == 'positive' else "rgba(245, 158, 11, 0.1)" if insight['type'] == 'warning' else "rgba(255, 255, 255, 0.05)"
                        border_color = "rgba(16, 185, 129, 0.3)" if insight['type'] == 'positive' else "rgba(245, 158, 11, 0.3)" if insight['type'] == 'warning' else "rgba(255, 255, 255, 0.1)"
                        text_color = "#10b981" if insight['type'] == 'positive' else "#f59e0b" if insight['type'] == 'warning' else "#eee"
                        
                        st.markdown(f"""
                        <div style='background: {bg_color}; border: 1px solid {border_color}; padding: 15px; border-radius: 12px; height: 100%;'>
                            <div style='display: flex; align-items: center; gap: 10px; margin-bottom: 10px;'>
                                <span style='font-size: 1.5rem;'>{insight['icon']}</span>
                                <h5 style='color: {text_color}; margin: 0;'>{insight['title']}</h5>
                            </div>
                            <p style='color: #bbb; font-size: 0.9rem; margin: 0;'>{insight['text']}</p>
                        </div>
                        """, unsafe_allow_html=True)

                st.divider()
                
                # --- EQUITY CURVE COMPARISON ---
                st.markdown("### 📊 Equity Curve: Fund vs Benchmark")
                st.caption(f"Normalized growth comparison with {benchmark_name}")
                
                # Create equity curve (Growth of ₹10,000)
                initial_investment = 10000
                
                # Normalize both series to start at 10000
                if not history_df.empty and not benchmark_data.empty:
                    # Align dates for consistent comparison
                    
                    # Ensure series are not containing zeros which break division or look bad
                    f_series = history_df['nav'][history_df['nav'] > 0]
                    b_series = benchmark_data[benchmark_data > 0]
                    
                    # Ensure indices are naive to match
                    if f_series.index.tz is not None: f_series.index = f_series.index.tz_localize(None)
                    if b_series.index.tz is not None: b_series.index = b_series.index.tz_localize(None)
                    
                    common_data = pd.concat([f_series, b_series], axis=1).dropna()
                    if not common_data.empty:
                        fund_equity = (common_data.iloc[:, 0] / common_data.iloc[:, 0].iloc[0]) * initial_investment
                        bench_equity = (common_data.iloc[:, 1] / common_data.iloc[:, 1].iloc[0]) * initial_investment
                    else:
                        fund_equity = pd.Series()
                        bench_equity = pd.Series()
                else:
                    fund_equity = pd.Series()
                    bench_equity = pd.Series()

                if not fund_equity.empty:
                    fig_equity = go.Figure()
                    
                    fig_equity.add_trace(go.Scatter(
                        x=fund_equity.index,
                        y=fund_equity.values,
                        name=f'{scheme_name}',
                        line=dict(color='#10b981', width=2.5),
                        hovertemplate='<b>Fund</b><br>Date: %{x}<br>Value: ₹%{y:,.0f}<extra></extra>'
                    ))
                    
                    fig_equity.add_trace(go.Scatter(
                        x=bench_equity.index,
                        y=bench_equity.values,
                        name=benchmark_name,
                        line=dict(color='#3b82f6', width=2, dash='dash'),
                        hovertemplate='<b>Benchmark</b><br>Date: %{x}<br>Value: ₹%{y:,.0f}<extra></extra>'
                    ))
                    
                    fig_equity.update_layout(
                        title=f"Growth of ₹{initial_investment:,} Investment",
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
                    
                    # --- ADD BACKGROUND MARKINGS ---
                    # Add horizontal bands for better perspective on performance
                    final_val = fund_equity.iloc[-1]
                    start_val = fund_equity.iloc[0]
                    
                    # Add a 12% CAGR reference line (common target)
                    years_elapsed = (fund_equity.index[-1] - fund_equity.index[0]).days / 365.25
                    target_12_cagr = start_val * (1.12 ** years_elapsed)
                    
                    fig_equity.add_hline(y=target_12_cagr, line_dash="dot", line_color="rgba(255, 255, 255, 0.2)", 
                                        annotation_text="12% CAGR Target", annotation_position="bottom right")
                    
                    # Add background gradient color based on performance vs benchmark
                    # Since we can't do easy gradient, we add a faint highlight for areas of outperformance
                    if not bench_equity.empty:
                        common_idx = fund_equity.index.intersection(bench_equity.index)
                        f_vals = fund_equity.loc[common_idx]
                        b_vals = bench_equity.loc[common_idx]
                        
                        # Add a simple fill trace behind the lines
                        fig_equity.add_trace(go.Scatter(
                            x=common_idx.tolist() + common_idx.tolist()[::-1],
                            y=f_vals.tolist() + b_vals.tolist()[::-1],
                            fill='toself',
                            fillcolor='rgba(16, 185, 129, 0.05)',
                            line=dict(color='rgba(255,255,255,0)'),
                            hoverinfo="skip",
                            showlegend=False,
                            name='Alpha Region'
                        ))
                    
                    # Highlight Max Drawdown Point
                    if not fund_equity.empty:
                        # Find the deepest trough
                        dd_series = calculate_max_drawdown_series(common_data.iloc[:, 0])
                        if not dd_series.empty:
                            max_dd_date = dd_series.idxmin()
                            max_dd_val = fund_equity.loc[max_dd_date]
                            
                            add_max_drawdown_annotation(fig_equity, max_dd_date, max_dd_val)
                            add_drawdown_vertical_line(fig_equity, max_dd_date)

                    st.plotly_chart(fig_equity, use_container_width=True)
                    
                    # Show final values
                    col_eq1, col_eq2, col_eq3 = st.columns(3)
                    with col_eq1:
                        final_fund = fund_equity.iloc[-1]
                        metric_card("Fund Final Value", f"₹{final_fund:,.0f}")
                    with col_eq2:
                        final_bench = bench_equity.iloc[-1]
                        metric_card("Benchmark Final Value", f"₹{final_bench:,.0f}")
                    with col_eq3:
                        outperformance = ((final_fund - final_bench) / final_bench) * 100
                        metric_card("Outperformance", f"{outperformance:+.2f}%", delta=outperformance)
                else:
                    st.warning("⚠️ Insufficient overlapping data for Fund and Benchmark in the selected period.")
                
                st.divider()
                
                # ... [Charts code skipped] ...
                
            st.divider()
            # --- ROLLING RETURNS SECTION ---
            if True: # Was Rolling Tab
                st.markdown("### 📈 Rolling Returns Analysis")
                st.caption("Analyze fund performance consistency across different time periods")
                
                # Period selector
                selected_period = st.selectbox(
                    "Select Rolling Period",
                    ["1 Year", "3 Years", "5 Years", "10 Years"],
                    key="rolling_period_selector"
                )
                
                period_map = {"1 Year": 1, "3 Years": 3, "5 Years": 5, "10 Years": 10}
                window_years = period_map[selected_period]
                
                # Calculate rolling returns stats
                rolling_stats = calculate_rolling_returns_stats(
                    history_df_full['nav'], 
                    benchmark_data_full, 
                    window_years
                )
                
                if rolling_stats:
                    # Display statistics
                    st.markdown(f"#### {selected_period} Rolling Returns Statistics")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        metric_card("Fund Avg Return", f"{rolling_stats['fund_mean']:.2f}%", delta=rolling_stats['fund_mean'])
                        st.caption(f"Benchmark: {rolling_stats['bench_mean']:.2f}%")
                    with col2:
                        metric_card("Fund Median", f"{rolling_stats['fund_median']:.2f}%", delta=rolling_stats['fund_median'])
                        st.caption(f"Benchmark: {rolling_stats['bench_median']:.2f}%")
                    with col3:
                        metric_card("Volatility", f"{rolling_stats['fund_std']:.2f}%", delta=rolling_stats['fund_std'], is_good_if_positive=False)
                        st.caption(f"Benchmark: {rolling_stats['bench_std']:.2f}%")
                    with col4:
                        metric_card("Outperformance", f"{rolling_stats['outperformance_pct']:.1f}%", delta=rolling_stats['outperformance_pct']-50)
                        st.caption("% of periods beating benchmark")
                    
                    st.divider()
                    
                    # Rolling returns chart
                    fund_rolling = rolling_stats['fund_rolling']
                    bench_rolling = rolling_stats['bench_rolling']
                    
                    fig_rolling = go.Figure()
                    fig_rolling.add_trace(go.Scatter(
                        x=fund_rolling.index,
                        y=fund_rolling.values * 100,
                        mode='lines',
                        name=scheme_name,
                        line=dict(color='#10b981', width=2.5),
                        hovertemplate='Fund: %{y:.2f}%<extra></extra>'
                    ))
                    fig_rolling.add_trace(go.Scatter(
                        x=bench_rolling.index,
                        y=bench_rolling.values * 100,
                        mode='lines',
                        name=benchmark_name,
                        line=dict(color='#3b82f6', width=2, dash='dash'),
                        hovertemplate='Benchmark: %{y:.2f}%<extra></extra>'
                    ))
                    
                    fig_rolling.update_layout(
                        title=f"{selected_period} Rolling Returns Comparison",
                        xaxis_title="Date",
                        yaxis_title="Rolling Returns (%)",
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=450,
                        hovermode='x unified',
                        legend=dict(
                            yanchor="top", 
                            y=0.99, 
                            xanchor="left", 
                            x=0.01,
                            bgcolor='rgba(0,0,0,0.5)'
                        )
                    )
                    st.plotly_chart(fig_rolling, use_container_width=True)
                    
                    # Distribution comparison
                    st.markdown("#### Returns Distribution")
                    
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(
                        x=fund_rolling.values * 100,
                        name=scheme_name,
                        opacity=0.7,
                        marker=dict(color='#10b981', line=dict(color='white', width=0.5)),
                        nbinsx=30,
                        hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
                    ))
                    fig_dist.add_trace(go.Histogram(
                        x=bench_rolling.values * 100,
                        name=benchmark_name,
                        opacity=0.5,
                        marker=dict(color='#3b82f6', line=dict(color='white', width=0.5)),
                        nbinsx=30,
                        hovertemplate='Range: %{x}<br>Count: %{y}<extra></extra>'
                    ))
                    
                    fig_dist.update_layout(
                        title=f"{selected_period} Rolling Returns Distribution",
                        xaxis_title="Returns (%)",
                        yaxis_title="Frequency",
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=350,
                        barmode='overlay',
                        legend=dict(
                            yanchor="top", 
                            y=0.99, 
                            xanchor="right", 
                            x=0.99,
                            bgcolor='rgba(0,0,0,0.5)'
                        )
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                else:
                    st.warning(f"Insufficient data for {selected_period} rolling returns analysis. Need at least {window_years} years of historical data.")

            st.divider()
            # --- CALCULATOR SECTION ---
            if True: # Was Calculator Tab
                st.markdown("### 💰 Returns Calculator")
                st.caption(f"Based on full historical performance of {scheme_name}.")
                
                cal_tab1, cal_tab2, cal_tab3 = st.tabs(["SIP", "Lumpsum", "Step-up SIP"])
                
                # --- SIP CALCULATOR ---
                with cal_tab1:
                    col_sip1, col_sip2 = st.columns(2)
                    with col_sip1:
                        monthly_amt = st.number_input("Monthly SIP Amount (₹)", min_value=100, value=5000, step=100, key="sip_amt")
                    with col_sip2:
                        sip_tenure = st.number_input("Investment Tenure (Years)", min_value=1, max_value=30, value=5, step=1, key="sip_tenure")
                    
                    mode_key = 'theoretical'

                    if st.button("Calculate SIP", key="btn_sip", type="primary"):
                        res = calculate_sip_returns(history_df_full['nav'], monthly_amt, sip_tenure, mode=mode_key)
                        inv, curr, abs_ret, xirr, actual_years = res
                        
                        # Metrics
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Invested Amount", f"₹{inv:,.0f}")
                        c2.metric("Current Value", f"₹{curr:,.0f}")
                        c3.metric("Absolute Return", f"{abs_ret:.2f}%")
                        c4.metric("XIRR", f"{xirr:.2f}%")
                        
                        # Visualization
                        breakdown_df = calculate_sip_returns(history_df_full['nav'], monthly_amt, sip_tenure, return_breakdown=True, mode=mode_key)
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
                                name='Current Value',
                                line=dict(color='#8b5cf6', width=3),
                                fill='tozeroy',
                                fillcolor='rgba(139, 92, 246, 0.2)'
                            ))
                            fig_sip.update_layout(
                                title="SIP Investment Growth",
                                xaxis_title="Date",
                                yaxis_title="Amount (₹)",
                                template="plotly_dark",
                                height=400,
                                hovermode='x unified',
                                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                            )
                            st.plotly_chart(fig_sip, use_container_width=True)
                            
                            duration_str = f"{actual_years:.1f} years"
                            if mode_key == 'theoretical':
                                st.success(f"📈 **Result**: At current CAGR, your ₹{inv:,.0f} SIP would grow to ₹{curr:,.0f} in {sip_tenure} years.")
                            else:
                                if actual_years < sip_tenure:
                                    st.warning(f"⚠️ **Data Limit**: Only {duration_str} of historical data available. Backtest reflects this period.")
                                st.success(f"💡 **Backtest Result**: SIP would have grown to ₹{curr:,.0f} in {duration_str}!")

                # --- LUMPSUM CALCULATOR ---
                with cal_tab2:
                    col_lump1, col_lump2 = st.columns(2)
                    with col_lump1:
                        lump_amt = st.number_input("Lumpsum Amount (₹)", min_value=1000, value=100000, step=1000, key="lump_amt")
                    with col_lump2:
                        lump_tenure = st.number_input("Investment Tenure (Years)", min_value=1, max_value=30, value=5, step=1, key="lump_tenure")
                    
                    mode_key_l = 'theoretical'
                    if st.button("Calculate Lumpsum", key="btn_lump", type="primary"):
                        res = calculate_lumpsum_returns(history_df_full['nav'], lump_amt, lump_tenure, mode=mode_key_l)
                        curr, abs_ret, cagr, actual_years = res
                        
                        st.success(f"📈 **Result**: Your ₹{lump_amt:,.0f} investment would grow to ₹{curr:,.0f} in {lump_tenure} years (at {cagr:.2f}% CAGR).")
                        
                        # Metrics - Now showing 4 metrics including invested amount
                        c1, c2, c3, c4 = st.columns(4)
                        with c1: metric_card("Invested Amount", f"₹{lump_amt:,.0f}")
                        with c2: metric_card("Current Value", f"₹{curr:,.0f}", delta=curr-lump_amt, is_good_if_positive=True)
                        with c3: metric_card("Absolute Return", f"{abs_ret:.2f}%", delta=abs_ret, is_good_if_positive=True)
                        with c4: metric_card("CAGR", f"{cagr:.2f}%", delta=cagr, is_good_if_positive=True)
                        
                        # Visualization - show growth over time
                        # Get limited series
                        limited_series = history_df_full['nav'].copy()
                        if lump_tenure and lump_tenure > 0:
                            end_date = limited_series.index[-1]
                            start_date = end_date - pd.DateOffset(years=lump_tenure)
                            limited_series = limited_series[limited_series.index >= start_date]
                        
                        if not limited_series.empty:
                            start_nav = limited_series.iloc[0]
                            units = lump_amt / start_nav
                            value_series = limited_series * units
                            
                            fig_lump = go.Figure()
                            fig_lump.add_trace(go.Scatter(
                                x=value_series.index,
                                y=[lump_amt] * len(value_series),
                                mode='lines',
                                name='Invested Amount',
                                line=dict(color='#94a3b8', width=2, dash='dot')
                            ))
                            fig_lump.add_trace(go.Scatter(
                                x=value_series.index,
                                y=value_series.values,
                                mode='lines',
                                name='Current Value',
                                line=dict(color='#10b981', width=3),
                                fill='tonexty',
                                fillcolor='rgba(16, 185, 129, 0.2)'
                            ))
                            fig_lump.update_layout(
                                title="Lumpsum Investment Growth",
                                xaxis_title="Date",
                                yaxis_title="Amount (₹)",
                                template="plotly_dark",
                                height=400,
                                hovermode='x unified',
                                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                            )
                            st.plotly_chart(fig_lump, use_container_width=True)

                # --- STEP-UP SIP CALCULATOR ---
                with cal_tab3:
                    col_su1, col_su2, col_su3 = st.columns(3)
                    with col_su1:
                        initial_sip = st.number_input("Initial Monthly Amount (₹)", min_value=100, value=5000, step=100, key="step_sip_amt")
                    with col_su2:
                        step_up_pct = st.number_input("Annual Step-up %", min_value=0, max_value=100, value=10, step=1, key="step_up_pct")
                    with col_su3:
                        stepup_tenure = st.number_input("Investment Tenure (Years)", min_value=1, max_value=30, value=5, step=1, key="stepup_tenure")
                    
                    mode_key_su = 'theoretical'

                    if st.button("Calculate Step-up SIP", key="btn_stepup", type="primary"):
                        res = calculate_step_up_sip_returns(history_df_full['nav'], initial_sip, step_up_pct, stepup_tenure, mode=mode_key_su)
                        inv, curr, abs_ret, xirr, actual_years = res
                        
                        # Metrics
                        c1, c2, c3, c4 = st.columns(4)
                        with c1: metric_card("Invested Amount", f"₹{inv:,.0f}")
                        with c2: metric_card("Current Value", f"₹{curr:,.0f}", delta=curr-inv, is_good_if_positive=True)
                        with c3: metric_card("Absolute Return", f"{abs_ret:.2f}%", delta=abs_ret, is_good_if_positive=True)
                        with c4: metric_card("XIRR", f"{xirr:.2f}%", delta=xirr, is_good_if_positive=True)
                        
                        # Visualization
                        breakdown_df = calculate_step_up_sip_returns(history_df_full['nav'], initial_sip, step_up_pct, stepup_tenure, return_breakdown=True, mode=mode_key_su)
                        if not breakdown_df.empty:
                            fig_stepup = go.Figure()
                            fig_stepup.add_trace(go.Scatter(
                                x=breakdown_df['Date'], 
                                y=breakdown_df['Invested'],
                                mode='lines',
                                name='Invested Amount',
                                line=dict(color='#94a3b8', width=2, dash='dot'),
                                fill='tozeroy',
                                fillcolor='rgba(148, 163, 184, 0.1)'
                            ))
                            fig_stepup.add_trace(go.Scatter(
                                x=breakdown_df['Date'], 
                                y=breakdown_df['Value'],
                                mode='lines',
                                name='Current Value',
                                line=dict(color='#f59e0b', width=3),
                                fill='tozeroy',
                                fillcolor='rgba(245, 158, 11, 0.2)'
                            ))
                            fig_stepup.update_layout(
                                title="Step-up SIP Investment Growth",
                                xaxis_title="Date",
                                yaxis_title="Amount (₹)",
                                template="plotly_dark",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                height=400,
                                hovermode='x unified',
                                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                            )
                            st.plotly_chart(fig_stepup, use_container_width=True)
                            st.success(f"📈 **Result**: Your starting ₹{initial_sip:,.0f} SIP with {step_up_pct}% annual step-up would grow to ₹{curr:,.0f} in {stepup_tenure} years (at {xirr:.2f}% CAGR).")

            # (Removed placeholder block) 

            st.divider()
            # --- FUTURE PROJECTIONS SECTION ---
            if True: # Was Future Tab 
                st.header("Monte Carlo Simulation")
                st.caption("Projecting future potential price paths using Geometric Brownian Motion (GBM).")
                
                col_mc1, col_mc2 = st.columns([1, 2])
                
                with col_mc1:
                    st.subheader("Simulation Parameters")
                    mc_inv_amt = st.number_input("Investment Amount (₹)", min_value=1000, value=100000, step=1000)
                    mc_years = st.slider("Projection Horizon (Years)", 1, 20, 5)
                    mc_sims = st.select_slider("Number of Simulations", options=[100, 500, 1000, 2000, 5000], value=1000)
                    
                    if st.button("Run Simulation", type="primary"):
                        with st.spinner("Running Monte Carlo Simulation..."):
                            sim_results = run_monte_carlo_simulation(history_df_full['nav'], n_simulations=mc_sims, time_horizon_years=mc_years, initial_investment=mc_inv_amt)
                            st.session_state['mc_results'] = sim_results
                
                with col_mc2:
                    if 'mc_results' in st.session_state and st.session_state['mc_results']:
                        res = st.session_state['mc_results']
                        stats = res['stats']
                        
                        # Metrics Row
                        m1, m2, m3 = st.columns(3)
                        with m1: metric_card("Expected Value", f"₹{stats['expected_price']:,.0f}", delta=f"{stats['expected_cagr']:.2f}% CAGR", is_good_if_positive=True)
                        with m2: metric_card("Optimistic (95%)", f"₹{stats['optimistic_price']:,.0f}", delta=f"{stats['optimistic_cagr']:.2f}% CAGR", is_good_if_positive=True)
                        with m3: metric_card("Pessimistic (5%)", f"₹{stats['pessimistic_price']:,.0f}", delta=f"{stats['pessimistic_cagr']:.2f}% CAGR", is_good_if_positive=True)
                        
                        # Probability Analysis
                        if 'end_distribution' in res:
                            end_vals = res['end_distribution']
                            target_cagr = 12.0
                            target_multiplier = (1 + target_cagr/100) ** mc_years
                            target_value = mc_inv_amt * target_multiplier
                            
                            prob_success = (end_vals >= target_value).mean() * 100
                            prob_regret = (end_vals < mc_inv_amt).mean() * 100
                            
                            p_col1, p_col2 = st.columns(2)
                            with p_col1:
                                st.write(f"🚀 **Target ({target_cagr}% CAGR):** `{prob_success:.1f}%` chance")
                                st.progress(prob_success/100)
                                if prob_success > 60:
                                    st.caption("✅ Strong probability of meeting 12% growth target.")
                                else:
                                    st.caption("⚠️ High growth required to meet target returns.")
                            with p_col2:
                                st.write(f"🛡️ **Capital Protection:** `{100-prob_regret:.1f}%` chance")
                                st.progress((100-prob_regret)/100)
                                if (100-prob_regret) > 90:
                                    st.caption("💎 Very low risk of capital loss in simulate period.")
                                else:
                                    st.caption("📉 Moderate risk of principal erosion detected.")
                            
                            # --- TARGET REACH PREDICTION ---
                            st.write("")
                            st.markdown("##### 🎯 Target Reach Prediction")
                            
                            # Calculate years needed to reach a custom target
                            custom_target = st.number_input("Goal Target Amount (₹)", min_value=mc_inv_amt, value=int(mc_inv_amt * 2), step=10000)
                            
                            # Use expected CAGR to predict
                            exp_cagr = stats['expected_cagr'] / 100
                            if exp_cagr > 0:
                                years_needed = np.log(custom_target / mc_inv_amt) / np.log(1 + exp_cagr)
                                st.write(f"⏱️ At current performance, you may reach ₹{custom_target:,.0f} in approximately `{years_needed:.1f} years`.")
                                st.info(f"💡 Adjusting monthly contributions or selecting a higher alpha fund could shorten this to `{years_needed * 0.7:.1f} years`.")
                            else:
                                st.warning("⚠️ Projected returns are negative. Goal unreachable at current rates.")

                        # Chart
                        try:
                            fig_mc = go.Figure()
                            
                            # Add a few raw paths (faint)
                            # Ensure paths are valid
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
                                name='95th Percentile (Optimistic)',
                                line=dict(color='#4ADE80', width=2, dash='dash')
                            ))
                            
                            fig_mc.add_trace(go.Scatter(
                                x=res['dates'], y=res['mean'], mode='lines', 
                                name='Expected Value (Mean)',
                                line=dict(color='#3b82f6', width=3)
                            ))
                            
                            fig_mc.add_trace(go.Scatter(
                                x=res['dates'], y=res['p5'], mode='lines', 
                                name='5th Percentile (Pessimistic)',
                                line=dict(color='#F87171', width=2, dash='dash')
                            ))
                            
                            fig_mc.update_layout(
                                title=f"Monte Carlo Projection ({mc_years} Years)",
                                xaxis_title="Date",
                                yaxis_title="Projected Value (₹)",
                                template="plotly_dark",
                                paper_bgcolor='rgba(0,0,0,0)',
                                plot_bgcolor='rgba(0,0,0,0)',
                                height=500,
                                hovermode='x unified',
                                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                            )
                            st.plotly_chart(fig_mc, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error rendering chart: {str(e)}")
                    else:
                        st.info("👈 Adjust parameters and click 'Run Simulation' to see projections.")
                            

                # --- RISK LABS: STRESS TESTING ---
                st.markdown("---")
                st.markdown("### 🧪 Risk Labs - Stress Testing")
                st.caption("See how this fund would have performed during major market crashes")
                
                # Get predefined scenarios
                scenarios = get_predefined_scenarios()
                
                # Scenario selector
                scenario_names = [s['name'] for s in scenarios]
                selected_scenario_name_sf = st.selectbox(
                    "Select Market Event",
                    scenario_names,
                    key="scenario_selector_sf"
                )
                
                # Find selected scenario
                selected_scenario_sf = next((s for s in scenarios if s['name'] == selected_scenario_name_sf_sf), None) if 'selected_scenario_name_sf_sf' in locals() else next((s for s in scenarios if s['name'] == selected_scenario_name_sf), None)
                
                if selected_scenario_sf:
                    st.info(f"📅 **{selected_scenario_sf['name']}** ({selected_scenario_sf['start_date']} to {selected_scenario_sf['end_date']})\n\n{selected_scenario_sf['description']}")
                    
                    # Run simulation for the single fund
                    if 'benchmark_data' in locals() and not benchmark_data.empty:
                        # We use the filtered benchmark series that matches the fund's timeline
                        # But for stress tests we might need the full series if the event is old
                        first_bench_series = benchmark_data_full
                        
                        fund_stress_result = simulate_market_scenario(
                            history_df_full['nav'],
                            first_bench_series,
                            selected_scenario_sf
                        )
                        
                        if fund_stress_result['data_available']:
                            # Display results
                            s_col1, s_col2, s_col3 = st.columns(3)
                            
                            with s_col1:
                                st.metric(
                                    "Fund Return",
                                    f"{fund_stress_result['portfolio_return']:.2f}%",
                                    delta=f"{fund_stress_result['portfolio_return'] - fund_stress_result['benchmark_return']:.2f}% vs Benchmark"
                                )
                            
                            with s_col2:
                                st.metric(
                                    "Max Drawdown",
                                    f"{fund_stress_result['portfolio_max_drawdown']:.2f}%",
                                    delta=f"{fund_stress_result['portfolio_max_drawdown'] - fund_stress_result['benchmark_max_drawdown']:.2f}% vs Benchmark",
                                    delta_color="inverse"
                                )
                            
                            with s_col3:
                                rec_text = f"{fund_stress_result['days_to_recover']} days" if fund_stress_result['days_to_recover'] else "Not yet recovered"
                                st.metric("Recovery Time", rec_text)
                                
                            # Detailed Comparison Table
                            st.markdown("#### 📊 Detailed Comparison")
                            sf_comparison_data = {
                                'Metric': ['Return During Event', 'Maximum Drawdown', 'Recovery Time (Days)'],
                                'This Fund': [
                                    f"{fund_stress_result['portfolio_return']:.2f}%",
                                    f"{fund_stress_result['portfolio_max_drawdown']:.2f}%",
                                    str(fund_stress_result['days_to_recover']) if fund_stress_result['days_to_recover'] else "N/A"
                                ],
                                'Benchmark Index': [
                                    f"{fund_stress_result['benchmark_return']:.2f}%",
                                    f"{fund_stress_result['benchmark_max_drawdown']:.2f}%",
                                    f"{fund_stress_result['benchmark_days_to_recover']} days" if fund_stress_result.get('benchmark_days_to_recover') else "Not yet recovered"
                                ]
                            }
                            st.dataframe(pd.DataFrame(sf_comparison_data), use_container_width=True, hide_index=True)

                            # Analysis & Insights
                            st.markdown("#### 💡 Analysis & Insights")
                            if fund_stress_result['outperformed_benchmark']:
                                st.success(f"✅ **This fund was resilient!** It fell {abs(fund_stress_result['portfolio_return'] - fund_stress_result['benchmark_return']):.2f}% less than the benchmark during {selected_scenario_sf['name']}.")
                                st.info(f"📊 **What this means:** The fund manager likely held high-quality stocks or cash during the crash, offering better downside protection than the index.")
                            else:
                                st.warning(f"⚠️ **This fund was sensitive.** It fell {abs(fund_stress_result['portfolio_return'] - fund_stress_result['benchmark_return']):.2f}% more than the benchmark during {selected_scenario_sf['name']}.")
                                st.info(f"📊 **What this means:** This fund has higher beta (market sensitivity). While it may fall more in crashes, high-beta funds often rally harder during bull markets. Check if the upside capture clarifies this trade-off.")
                            
                            # Historical Context
                            st.markdown("#### 📚 Historical Context")
                            event_context = {
                                "COVID-19 Crash (2020)": "The fastest 30% drop in history, triggered by global pandemic fears. Markets recovered to new highs within 6 months as central banks pumped liquidity.",
                                "Global Financial Crisis (2008-09)": "Triggered by US housing market collapse and Lehman Brothers bankruptcy. One of the worst crashes since 1929, taking over 5 years for full recovery.",
                                "IL&FS Crisis (2018)": "India-specific credit crisis that caused significant damage to mid and small-cap stocks. NBFCs and financial stocks were hit hardest.",
                                "Demonetization (2016)": "Sudden cash crunch caused short-term market volatility, but recovery was swift as fundamentals remained intact.",
                                "Taper Tantrum (2013)": "Fed's announcement to reduce QE caused emerging market selloff. India's Rupee fell sharply, impacting import-heavy sectors."
                            }
                            context_text = event_context.get(selected_scenario_sf['name'], f"A significant market event during {selected_scenario_sf['start_date']}.")
                            st.caption(f"📖 **{selected_scenario_sf['name']}:** {context_text}")
                        
                        elif fund_stress_result['bench_data_available']:
                            st.warning(f"⚠️ **Fund not active during this period.** The fund was not launched or data is missing for {selected_scenario_sf['name']}.")
                            
                            b_col_s1, b_col_s2 = st.columns(2)
                            with b_col_s1:
                                st.metric("Benchmark Return", f"{fund_stress_result['benchmark_return']:.2f}%")
                            with b_col_s2:
                                st.metric("Benchmark Max Drawdown", f"{fund_stress_result['benchmark_max_drawdown']:.2f}%")
                            
                            st.caption(f"Historical Context: The market fell significantly during this period. Newer funds often haven't faced a true bear market like this one.")
                        else:
                            st.warning(f"⚠️ Insufficient data available for the selected scenario period.")
                    else:
                        st.warning("⚠️ No benchmark data available for stress testing comparison.")
                
                # --- ADVANCED ML FEATURES (Keep hidden if not available) ---
                if ML_AVAILABLE:
                    pass


        else:
            st.warning("Historical data not available for this fund.")
    else:
        # Dashboard Overview / Hero section
        render_welcome_card(len(nav_data))


if __name__ == "__main__":
    main()
