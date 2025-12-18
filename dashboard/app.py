import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go

# Add backend to path so we can import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

# Add dashboard to path for module imports
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import fetch_latest_nav, fetch_fund_history, fetch_scheme_details
from analytics import (
    get_fund_metrics, 
    download_benchmark, 
    calculate_max_drawdown_series,
    calculate_rolling_returns,
    calculate_rolling_returns_stats,
    filter_by_period,
    calculate_sip_returns,
    calculate_lumpsum_returns,
    calculate_step_up_sip_returns,
    run_monte_carlo_simulation
)

# Import comparison view
from comparison_view import render_comparison_view

# Import portfolio view
from portfolio_view import render_portfolio_view

# Page Config
st.set_page_config(
    page_title="Antigravity Capital",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glassmorphism and premium look
from ui_components import apply_custom_css, get_neon_color, metric_card

# Apply Global Theme
apply_custom_css()

def main():
    st.title("Antigravity Capital")
    st.markdown("### Advanced Mutual Fund Analytics Dashboard")
    
    # MODE SELECTOR
    view_mode = st.radio(
        "📊 View Mode",
        ["Single Fund Analysis", "Compare Multiple Funds", "Build Portfolio"],
        horizontal=True,
        key="view_mode_selector"
    )
    
    st.divider()
    
    if view_mode == "Compare Multiple Funds":
        render_comparison_view()
        return
    elif view_mode == "Build Portfolio":
        render_portfolio_view()
        return

    # SINGLE FUND VIEW (existing code)
    with st.spinner("Loading Fund Universe..."):
        nav_data = fetch_latest_nav()
    
    if nav_data.empty:
        st.error("Failed to load fund data. Please check your internet connection.")
        return

    # Sidebar for selection
    st.sidebar.header("Fund Selection")
    
    # Create valid list for dropdown (Scheme Name + Code)
    nav_data['Display'] = nav_data['Scheme Name'] + " (" + nav_data['Scheme Code'].astype(str) + ")"
    funds_list = nav_data['Display'].tolist()
    
    selected_option = st.sidebar.selectbox("Search Fund", funds_list, index=None, placeholder="Type to search...")

    if selected_option:
        # Extract scheme code
        qt_start = selected_option.rfind('(')
        qt_end = selected_option.rfind(')')
        scheme_code = selected_option[qt_start+1 : qt_end]
        scheme_name = selected_option[:qt_start].strip()


        # --- DATA FETCHING & BENCHMARK SELECTION ---
        with st.spinner("Fetching data & analytics..."):
            history_df_full = fetch_fund_history(scheme_code) # Fetch full history first
            metadata = fetch_scheme_details(scheme_code)
            
            # Determine Benchmark based on Category
            category = metadata.get('scheme_category', '').lower()
            benchmark_ticker = "^NSEI" # Default Nifty 50
            benchmark_name = "NIFTY 50"
            
            if 'small' in category:
                benchmark_ticker = "^CNXSC" # Nifty Smallcap 100
                benchmark_name = "NIFTY SMALLCAP 100"
            elif 'mid' in category:
                benchmark_ticker = "^NSEMDCP50" # Nifty Midcap 50
                benchmark_name = "NIFTY MIDCAP 50"
            elif 'bank' in category:
                benchmark_ticker = "^NSEBANK" # Nifty Bank
                benchmark_name = "NIFTY BANK"
            elif 'gold' in category:
                 # Gold funds need commodity ticker
                 pass
            
            benchmark_data_full = download_benchmark(benchmark_ticker)
            
            # Fallback if specific benchmark data is insufficient
            if len(benchmark_data_full) < 100 and benchmark_ticker != "^NSEI":
                benchmark_ticker = "^NSEI"
                benchmark_name = "NIFTY 50 (Category Index unavailable)"
                benchmark_data_full = download_benchmark(benchmark_ticker)
        
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

        col1, col2, col3 = st.columns(3)
        with col1:
             st.info(f"**Category:** {metadata.get('scheme_category', 'N/A')}")
        with col2:
             st.info(f"**Fund House:** {metadata.get('fund_house', 'N/A')}")
        with col3:
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
                            <span style='background: #2D2D2D; padding: 4px 12px; border-radius: 16px; color: #BBB; font-size: 0.8rem;'>{metadata.get('scheme_category', 'N/A')}</span>
                            <span style='background: #2D2D2D; padding: 4px 12px; border-radius: 16px; color: #BBB; font-size: 0.8rem;'>{metadata.get('fund_house', 'N/A')}</span>
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
                    metric_card("CAGR", f"{cagr:.2f}%", "Annual Growth", get_neon_color(cagr, 'performance'))
                
                with c2:
                    alpha = metrics.get('Alpha', 0)*100
                    metric_card("Alpha", f"{alpha:.2f}%", "vs Benchmark", get_neon_color(alpha, 'alpha'))
                
                with c3:
                    sharpe = metrics.get('Sharpe Ratio', 0)
                    metric_card("Sharpe Ratio", f"{sharpe:.2f}", "Risk-Adjusted", get_neon_color(sharpe, 'sharpe'))
                    
                with c4:
                    dd = metrics.get('Max Drawdown', 0)*100
                    metric_card("Max Drawdown", f"{abs(dd):.2f}%", "Max Loss", get_neon_color(dd, 'drawdown'))

                st.write("") # Spacer

                # Row 2: Secondary Stats
                c5, c6, c7, c8 = st.columns(4)
                
                with c5:
                    vol = metrics.get('Volatility', 0)*100
                    metric_card("Volatility", f"{vol:.2f}%", "Risk Level", get_neon_color(vol, 'risk'))
                
                with c6:
                    beta = metrics.get('Beta', 0)
                    metric_card("Beta", f"{beta:.2f}", "Market Sensitivity", get_neon_color(beta, 'beta'))
                    
                with c7:
                    # Sortino Calculation
                    aligned_df = pd.concat([history_df['nav'], benchmark_data], axis=1).dropna()
                    fund_returns = aligned_df.iloc[:, 0].pct_change().dropna()
                    downside_returns = fund_returns[fund_returns < 0]
                    downside_std = downside_returns.std() * (252 ** 0.5) if len(downside_returns) > 0 else 0
                    sortino = (metrics.get('CAGR', 0) / downside_std) if downside_std > 0 else 0
                    
                    metric_card("Sortino", f"{sortino:.2f}", "Downside Risk", get_neon_color(sortino, 'sharpe'))
                    
                with c8:
                    win_rate = (fund_returns > aligned_df.iloc[:, 1].pct_change().dropna()).mean() * 100 if not aligned_df.empty else 0
                    metric_card("Win Rate", f"{win_rate:.0f}%", "vs Benchmark", get_neon_color(win_rate-50, 'alpha'))

                st.divider()
                
                # --- CHARTS ---
                col_chart_1, col_chart_2 = st.columns([2, 1])
                
                if not aligned_df.empty:
                    aligned_df.columns = ['Fund', 'Benchmark']
                    rebased = aligned_df / aligned_df.iloc[0] * 100
                    
                    # Growth Chart (Main)
                    with col_chart_1:
                        st.markdown("<div class='metric-label' style='margin-bottom: 10px;'>Growth Analysis</div>", unsafe_allow_html=True)
                        fig_growth = go.Figure()
                        
                        # Add Gradient Fill
                        fig_growth.add_trace(go.Scatter(
                            x=rebased.index, y=rebased['Fund'], mode='lines', 
                            name=scheme_name, 
                            line=dict(color='#8B5CF6', width=3),
                            fill='tozeroy',
                            fillcolor='rgba(139, 92, 246, 0.1)' # Transparent Purple
                        ))
                        fig_growth.add_trace(go.Scatter(
                            x=rebased.index, y=rebased['Benchmark'], mode='lines', 
                            name=benchmark_name, 
                            line=dict(color='#6B7280', width=2, dash='dot')
                        ))
                        
                        fig_growth.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            xaxis=dict(showgrid=False, gridcolor='#333', color='#6B7280'),
                            yaxis=dict(showgrid=True, gridcolor='#333', color='#6B7280'),
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=350,
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                        )
                        st.plotly_chart(fig_growth, use_container_width=True)

                    # Allocation & Drawdown (Side)
                    with col_chart_2:
                         # Allocation - Minimal Bar
                        st.markdown("<div class='metric-label' style='margin-bottom: 10px;'>Allocation</div>", unsafe_allow_html=True)
                        
                        category_lower = metadata.get('scheme_category', '').lower()
                        if 'equity' in category_lower or 'stock' in category_lower:
                            alloc_data = {'Equity': 95, 'Debt': 3, 'Cash': 2}
                        elif 'debt' in category_lower or 'bond' in category_lower:
                            alloc_data = {'Debt': 90, 'Equity': 5, 'Cash': 5}
                        else:
                            alloc_data = {'Equity': 70, 'Debt': 25, 'Cash': 5}
                            
                        fig_alloc = go.Figure(go.Bar(
                            x=list(alloc_data.values()),
                            y=list(alloc_data.keys()),
                            orientation='h',
                            text=[f"{v}%" for v in alloc_data.values()],
                            textposition='auto',
                            marker=dict(color=['#8B5CF6', '#10B981', '#F59E0B'], line=dict(width=0))
                        ))
                        fig_alloc.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=120,
                            xaxis=dict(showgrid=False, showticklabels=False),
                            yaxis=dict(showgrid=False, color='#A0A0A0')
                        )
                        st.plotly_chart(fig_alloc, use_container_width=True)
                        
                        st.write("")
                        
                        # Drawdown Mini
                        st.markdown("<div class='metric-label' style='margin-bottom: 10px;'>Drawdown Risk</div>", unsafe_allow_html=True)
                        fund_dd = calculate_max_drawdown_series(aligned_df['Fund'])*100
                        
                        fig_dd = go.Figure()
                        fig_dd.add_trace(go.Scatter(
                            x=fund_dd.index, y=fund_dd, 
                            fill='tozeroy',
                            fillcolor='rgba(239, 68, 68, 0.2)',
                            line=dict(color='#EF4444', width=1.5)
                        ))
                        fig_dd.update_layout(
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=0, r=0, t=0, b=0),
                            height=150,
                            xaxis=dict(showgrid=False, showticklabels=False),
                            yaxis=dict(showgrid=False, showticklabels=True, color='#6B7280')
                        )
                        st.plotly_chart(fig_dd, use_container_width=True)
                else:
                    st.write("Insufficient data overlap for comparison chart.")
                    st.line_chart(history_df['nav'])

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
                        st.metric("Fund Avg Return", f"{rolling_stats['fund_mean']:.2f}%")
                        st.caption(f"Benchmark: {rolling_stats['bench_mean']:.2f}%")
                    with col2:
                        st.metric("Fund Median", f"{rolling_stats['fund_median']:.2f}%")
                        st.caption(f"Benchmark: {rolling_stats['bench_median']:.2f}%")
                    with col3:
                        st.metric("Volatility (Std Dev)", f"{rolling_stats['fund_std']:.2f}%")
                        st.caption(f"Benchmark: {rolling_stats['bench_std']:.2f}%")
                    with col4:
                        st.metric("Outperformance", f"{rolling_stats['outperformance_pct']:.1f}%")
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
                        line=dict(color='#8b5cf6', width=2)
                    ))
                    fig_rolling.add_trace(go.Scatter(
                        x=bench_rolling.index,
                        y=bench_rolling.values * 100,
                        mode='lines',
                        name=benchmark_name,
                        line=dict(color='#9ca3af', width=2, dash='dot')
                    ))
                    
                    fig_rolling.update_layout(
                        title=f"{selected_period} Rolling Returns Comparison",
                        xaxis_title="Date",
                        yaxis_title="Rolling Returns (%)",
                        template="plotly_dark",
                        height=450,
                        hovermode='x unified',
                        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
                    )
                    st.plotly_chart(fig_rolling, use_container_width=True)
                    
                    # Distribution comparison
                    st.markdown("#### Returns Distribution")
                    
                    fig_dist = go.Figure()
                    fig_dist.add_trace(go.Histogram(
                        x=fund_rolling.values * 100,
                        name=scheme_name,
                        opacity=0.7,
                        marker_color='#8b5cf6',
                        nbinsx=30
                    ))
                    fig_dist.add_trace(go.Histogram(
                        x=bench_rolling.values * 100,
                        name=benchmark_name,
                        opacity=0.5,
                        marker_color='#9ca3af',
                        nbinsx=30
                    ))
                    
                    fig_dist.update_layout(
                        title=f"{selected_period} Rolling Returns Distribution",
                        xaxis_title="Returns (%)",
                        yaxis_title="Frequency",
                        template="plotly_dark",
                        height=350,
                        barmode='overlay',
                        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                    )
                    st.plotly_chart(fig_dist, use_container_width=True)
                    
                else:
                    st.warning(f"Insufficient data for {selected_period} rolling returns analysis. Need at least {window_years} years of historical data.")

            st.divider()
            # --- CALCULATOR SECTION ---
            if True: # Was Calculator Tab
                st.markdown("### 💰 Returns Calculator")
                st.caption(f"Based on historical performance of {scheme_name} over the last {time_period} (or relevant duration).")
                
                cal_tab1, cal_tab2, cal_tab3 = st.tabs(["SIP", "Lumpsum", "Step-up SIP"])
                
                # --- SIP CALCULATOR ---
                with cal_tab1:
                    col_sip1, col_sip2 = st.columns(2)
                    with col_sip1:
                        monthly_amt = st.number_input("Monthly SIP Amount (₹)", min_value=100, value=5000, step=100, key="sip_amt")
                    with col_sip2:
                        sip_tenure = st.number_input("Investment Tenure (Years)", min_value=1, max_value=30, value=5, step=1, key="sip_tenure")
                    
                    if st.button("Calculate SIP", key="btn_sip"):
                        inv, curr, abs_ret, xirr = calculate_sip_returns(history_df['nav'], monthly_amt, sip_tenure)
                        
                        # Metrics
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Invested Amount", f"₹{inv:,.0f}")
                        c2.metric("Current Value", f"₹{curr:,.0f}")
                        c3.metric("Absolute Return", f"{abs_ret:.2f}%")
                        c4.metric("XIRR", f"{xirr:.2f}%")
                        
                        # Visualization
                        breakdown_df = calculate_sip_returns(history_df['nav'], monthly_amt, sip_tenure, return_breakdown=True)
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

                # --- LUMPSUM CALCULATOR ---
                with cal_tab2:
                    col_lump1, col_lump2 = st.columns(2)
                    with col_lump1:
                        lump_amt = st.number_input("Lumpsum Amount (₹)", min_value=1000, value=100000, step=1000, key="lump_amt")
                    with col_lump2:
                        lump_tenure = st.number_input("Investment Tenure (Years)", min_value=1, max_value=30, value=5, step=1, key="lump_tenure")
                    
                    if st.button("Calculate Lumpsum", key="btn_lump"):
                        curr, abs_ret, cagr = calculate_lumpsum_returns(history_df['nav'], lump_amt, lump_tenure)
                        
                        # Metrics - Now showing 4 metrics including invested amount
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Invested Amount", f"₹{lump_amt:,.0f}")
                        c2.metric("Current Value", f"₹{curr:,.0f}")
                        c3.metric("Absolute Return", f"{abs_ret:.2f}%")
                        c4.metric("CAGR", f"{cagr:.2f}%")
                        
                        # Visualization - show growth over time
                        # Get limited series
                        limited_series = history_df['nav'].copy()
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
                    
                    if st.button("Calculate Step-up SIP", key="btn_stepup"):
                        inv, curr, abs_ret, xirr = calculate_step_up_sip_returns(history_df['nav'], initial_sip, step_up_pct, stepup_tenure)
                        
                        # Metrics
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Invested Amount", f"₹{inv:,.0f}")
                        c2.metric("Current Value", f"₹{curr:,.0f}")
                        c3.metric("Absolute Return", f"{abs_ret:.2f}%")
                        c4.metric("XIRR", f"{xirr:.2f}%")
                        
                        # Visualization
                        breakdown_df = calculate_step_up_sip_returns(history_df['nav'], initial_sip, step_up_pct, stepup_tenure, return_breakdown=True)
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
                            sim_results = run_monte_carlo_simulation(history_df['nav'], n_simulations=mc_sims, time_horizon_years=mc_years, initial_investment=mc_inv_amt)
                            st.session_state['mc_results'] = sim_results
                
                with col_mc2:
                    if 'mc_results' in st.session_state and st.session_state['mc_results']:
                        res = st.session_state['mc_results']
                        stats = res['stats']
                        
                        # Metrics Row
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Expected Value", f"₹{stats['expected_price']:,.0f}", f"{stats['expected_cagr']:.2f}% CAGR")
                        m2.metric("Optimistic (95%)", f"₹{stats['optimistic_price']:,.0f}", f"{stats['optimistic_cagr']:.2f}% CAGR")
                        m3.metric("Pessimistic (5%)", f"₹{stats['pessimistic_price']:,.0f}", f"{stats['pessimistic_cagr']:.2f}% CAGR")
                        
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
                            
                        st.info("💡 Step-up SIP increases your monthly contribution automatically, helping you invest more as your income grows!")

        else:
            st.warning("Historical data not available for this fund.")
    else:
        st.info("👈 Select a fund from the sidebar to view details.")
        
        # Dashboard Overview / Hero section
        st.markdown("---")
        st.markdown("#### Market Overview")
        st.write(f"Tracking **{len(nav_data)}** active schemes.")

if __name__ == "__main__":
    main()
