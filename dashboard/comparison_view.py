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
from analytics import download_benchmark, get_fund_metrics, calculate_rolling_returns, calculate_correlation_matrix, calculate_correlation_matrix

def render_comparison_view():
    """Renders the multi-fund comparison view"""
    st.header("📊 Compare Multiple Funds")
    st.caption("Select 2-5 funds to compare their performance side-by-side")
    
    # Load fund universe
    with st.spinner("Loading Fund Universe..."):
        nav_data = fetch_latest_nav()
    
    if nav_data.empty:
        st.error("Failed to load fund data.")
        return
    
    # Sidebar for fund selection
    with st.sidebar:
        st.header("Select Funds")
        
        # Create fund list
        nav_data['Display'] = nav_data['Scheme Name'] + " (" + nav_data['Scheme Code'].astype(str) + ")"
        funds_list = nav_data['Display'].tolist()
        
        # Multi-select
        selected_funds = st.multiselect(
            "Choose 2-5 funds to compare",
            funds_list,
            key="multi_fund_selector",
            help="Select between 2 and 5 funds"
        )
        
        if len(selected_funds) < 2:
            st.info("👆 Please select at least 2 funds")
            return
        elif len(selected_funds) > 5:
            st.warning("⚠️ Maximum 5 funds allowed")
            return
    
    # Extract scheme codes
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
            metadata = fetch_scheme_details(code)
            
            if not history_df.empty:
                # Get benchmark
                category = metadata.get('scheme_category', '').lower()
                benchmark_ticker = "^NSEI"
                if 'small' in category:
                    benchmark_ticker = "^CNXSC"
                elif 'mid' in category:
                    benchmark_ticker = "^NSEMDCP50"
                elif 'bank' in category:
                    benchmark_ticker = "^NSEBANK"
                
                benchmark_data = download_benchmark(benchmark_ticker)
                if len(benchmark_data) < 100 and benchmark_ticker != "^NSEI":
                    benchmark_data = download_benchmark("^NSEI")
                
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
    
    if not funds_data:
        st.error("Unable to load data for selected funds.")
        return
    
    st.success(f"✅ Loaded {len(funds_data)} funds successfully!")
    
    # METRICS COMPARISON TABLE
    st.markdown("### 📈 Performance Metrics")
    
    metrics_data = []
    for fund in funds_data:
        m = fund['metrics']
        metrics_data.append({
            'Fund': fund['name'][:35] + '...' if len(fund['name']) > 35 else fund['name'],
            'CAGR': f"{m.get('CAGR', 0)*100:.2f}%",
            'Sharpe': f"{m.get('Sharpe Ratio', 0):.2f}",
            'Alpha': f"{m.get('Alpha', 0)*100:.2f}%",
            'Beta': f"{m.get('Beta', 0):.2f}",
            'Volatility': f"{m.get('Volatility', 0)*100:.2f}%",
            'Max DD': f"{m.get('Max Drawdown', 0)*100:.2f}%"
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # NORMALIZED PERFORMANCE CHART
    st.markdown("### 📊 Normalized Performance (Rebased to 100)")
    st.caption("All funds start at ₹100 for easy comparison")
    
    # Find common date range
    all_series = [fund['history'] for fund in funds_data]
    common_start = max([s.index[0] for s in all_series])
    common_end = min([s.index[-1] for s in all_series])
    
    fig_perf = go.Figure()
    
    for fund in funds_data:
        series = fund['history']
        series_filtered = series[(series.index >= common_start) & (series.index <= common_end)]
        
        if not series_filtered.empty:
            rebased = (series_filtered / series_filtered.iloc[0]) * 100
            fig_perf.add_trace(go.Scatter(
                x=rebased.index,
                y=rebased.values,
                mode='lines',
                name=fund['name'][:30],
                line=dict(color=fund['color'], width=2.5)
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
    
    # DRAWDOWN COMPARISON
    st.markdown("### 📉 Drawdown Analysis")
    st.caption("Shows maximum decline from peak - lower is better")
    
    fig_dd = go.Figure()
    
    for fund in funds_data:
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
    for fund in funds_data:
        fund_row = {'Fund': fund['name'][:30]}
        
        # Calculate rolling returns for each period
        for period_name, years in [("1Y", 1), ("3Y", 3), ("5Y", 5), ("10Y", 10)]:
            rolling_ret = calculate_rolling_returns(fund['history'], window_years=years)
            if not rolling_ret.empty:
                avg_return = rolling_ret.mean() * 100
                fund_row[period_name] = f"{avg_return:.2f}%"
            else:
                fund_row[period_name] = "N/A"
        
        rolling_summary.append(fund_row)
    
    rolling_summary_df = pd.DataFrame(rolling_summary)
    st.dataframe(rolling_summary_df, use_container_width=True, hide_index=True)

    

    
    # RISK-RETURN SCATTER
    st.markdown("### 🎯 Risk-Return Analysis")
    st.caption("Top-left quadrant = Best (High returns, Low risk)")
    
    fig_scatter = go.Figure()
    
    for fund in funds_data:
        m = fund['metrics']
        fig_scatter.add_trace(go.Scatter(
            x=[m.get('Volatility', 0) * 100],
            y=[m.get('CAGR', 0) * 100],
            mode='markers+text',
            name=fund['name'][:25],
            text=[fund['name'][:20]],
            textposition="top center",
            marker=dict(size=20, color=fund['color'], line=dict(width=2, color='white')),
            showlegend=True
        ))
    
    fig_scatter.update_layout(
        xaxis_title="Risk (Volatility %)",
        yaxis_title="Return (CAGR %)",
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=500,
        hovermode='closest',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.divider()
    
    # DETAILED STATISTICS TABLE
    st.markdown("### 📋 Detailed Statistics")
    
    detailed_stats = []
    for fund in funds_data:
        m = fund['metrics']
        detailed_stats.append({
            'Fund': fund['name'][:30],
            'CAGR': f"{m.get('CAGR', 0)*100:.2f}%",
            'Volatility': f"{m.get('Volatility', 0)*100:.2f}%",
            'Sharpe': f"{m.get('Sharpe Ratio', 0):.2f}",
            'Alpha': f"{m.get('Alpha', 0)*100:.2f}%",
            'Beta': f"{m.get('Beta', 0):.2f}",
            'R²': f"{m.get('R-Squared', 0):.2f}",
            'Max DD': f"{m.get('Max Drawdown', 0)*100:.2f}%",
            'Category': fund['metadata'].get('scheme_category', 'N/A')[:20]
        })
    
    detailed_df = pd.DataFrame(detailed_stats)
    st.dataframe(detailed_df, use_container_width=True, hide_index=True)
    
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
