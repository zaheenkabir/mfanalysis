import streamlit as st

def apply_custom_css():
    """Applies the global custom CSS for the modern dark theme."""
    st.markdown("""
    <style>
        /* Global Theme Overrides */
        .stApp {
            background-color: #0E1117;
        }
        
        /* Card Styling */
        .metric-card {
            background-color: #1E1E1E;
            border: 1px solid #333;
            border-radius: 12px;
            padding: 20px;
            text-align: left;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s;
            margin-bottom: 15px;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            border-color: #444;
        }
        
        /* Typography */
        .metric-label {
            color: #A0A0A0;
            font-size: 0.85rem;
            font-weight: 500;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .metric-value {
            color: #FFFFFF;
            font-size: 1.8rem;
            font-weight: 700;
            margin: 0;
            line-height: 1.2;
        }
        .metric-sub {
            font-size: 0.8rem;
            margin-top: 4px;
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0E1117;
        }
        ::-webkit-scrollbar-thumb {
            background: #333;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #555;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px;
            color: #A0A0A0;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: rgba(139, 92, 246, 0.1);
            color: #8b5cf6;
            border-bottom: 2px solid #8b5cf6;
        }
    </style>
    """, unsafe_allow_html=True)

def get_neon_color(val, metric_type='performance'):
    """Returns a neon color hex code based on the metric value and type."""
    # Performance: High is Green (Neon)
    if metric_type == 'performance':
        if val > 12: return '#4ADE80'  # Neon Green
        elif val > 0: return '#FBBF24' # Neon Orange
        return '#F87171' # Neon Red
    # Risk: Low is Green
    elif metric_type == 'risk':
        if val < 10: return '#4ADE80'
        elif val < 20: return '#FBBF24'
        return '#F87171'
    # Beta/Relative
    elif metric_type == 'beta':
        if val < 0.9: return '#4ADE80'
        elif val <= 1.1: return '#FBBF24'
        return '#F87171'
    # Alpha
    elif metric_type == 'alpha':
        return '#4ADE80' if val > 0 else '#F87171'
    # Sharpe
    elif metric_type == 'sharpe':
        if val > 1: return '#4ADE80'
        elif val > 0.5: return '#FBBF24'
        return '#F87171'
    elif metric_type == 'drawdown':
        val = abs(val)
        if val < 10: return '#4ADE80'
        elif val < 20: return '#FBBF24'
        return '#F87171'
    return '#60A5FA' # Neon Blue

def metric_card(label, value, sub_value=None, sub_color=None):
    """Renders a custom HTML metric card."""
    if sub_value:
        sub_html = f"<div class='metric-sub' style='color: {sub_color};'>{sub_value}</div>"
    else:
        sub_html = ""
    
    st.markdown(f"""
    <div class='metric-card'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value'>{value}</div>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)
