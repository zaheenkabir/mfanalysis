import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
# Renamed to solve persistent import issues
def render_main_navigation(current_val):
    """
    Renders a premium navigation menu using a styled radio button.
    Includes a separate link for the standalone Goal Planner.
    """
    options = ["Single Fund Analysis", "Compare Multiple Funds", "Build Portfolio"]
    
    col_nav, col_ext = st.columns([4, 1])
    
    with col_nav:
        selected = st.radio(
            "Navigation Menu",
            options,
            index=options.index(current_val) if current_val in options else 0,
            horizontal=True,
            label_visibility="collapsed"
        )
    
    with col_ext:
        # Styled link for external portal
        st.markdown(f"""
            <a href="http://localhost:8509" target="_blank" style="text-decoration: none;">
                <div style="
                    background: rgba(16, 185, 129, 0.1);
                    color: #10b981;
                    padding: 8px 16px;
                    border-radius: 30px;
                    border: 1px solid rgba(16, 185, 129, 0.3);
                    text-align: center;
                    font-weight: 600;
                    font-size: 0.9rem;
                    transition: all 0.3s ease;
                    cursor: pointer;
                " onmouseover="this.style.background='rgba(16, 185, 129, 0.2)';" onmouseout="this.style.background='rgba(16, 185, 129, 0.1)';">
                    🎯 GoalPlanner ↗
                </div>
            </a>
        """, unsafe_allow_html=True)
        
    return selected
def get_neon_color(value, is_good_if_positive=True):
    """
    Returns neon green or neon red based on the value's sign and context.
    """
    if value == 0:
        return "#e0e0e0" # Neutral grey
    
    is_good = (value > 0) if is_good_if_positive else (value < 0)
    return "#00ff7f" if is_good else "#ff4b4b" # Neon SpringGreen vs Neon Red
def apply_custom_css():
    """
    Injects the SkillVista-inspired dark theme CSS.
    """
    st.markdown("""
        <style>
        /* GLOBAL THEME */
        .stApp {
            background-color: #0b0e14;
            background-image: 
                radial-gradient(circle at 0% 0%, rgba(59, 130, 246, 0.05) 0%, transparent 50%),
                radial-gradient(circle at 100% 100%, rgba(139, 92, 246, 0.05) 0%, transparent 50%);
            color: #e2e8f0;
            font-family: 'Outfit', sans-serif;
        }
        
        /* Glassmorphism Effect */
        .glass-card {
            background: rgba(17, 25, 40, 0.75);
            backdrop-filter: blur(12px) saturate(180%);
            -webkit-backdrop-filter: blur(12px) saturate(180%);
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.125);
            padding: 24px;
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }
        /* CARD CONTAINERS */
        .custom-metric-box {
            background: rgba(23, 28, 36, 0.6);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 18px 22px;
            border-radius: 16px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            margin-bottom: 0.8rem;
            position: relative;
            overflow: hidden;
        }
        
        .custom-metric-box:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 30px rgba(0, 0, 0, 0.4);
            border-color: rgba(139, 92, 246, 0.4);
            background: rgba(23, 28, 36, 0.8);
        }
        .custom-metric-box::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, transparent 100%);
            pointer-events: none;
        }
        /* SIDEBAR */
        section[data-testid="stSidebar"] {
            background-color: #0d1117;
            border-right: 1px solid rgba(255, 255, 255, 0.05);
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
            gap: 12px;
            background-color: transparent;
            padding: 8px 0;
            margin-bottom: 1.5rem;
        }
        .stTabs [data-baseweb="tab"] {
            height: 44px;
            white-space: pre-wrap;
            background-color: rgba(255, 255, 255, 0.03);
            border-radius: 8px;
            color: #A0A0A0;
            font-weight: 600;
            padding: 0 20px;
            margin: 0 4px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
            color: #fff !important;
            border-bottom: none !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        }
        /* GLOBAL SPACING */
        .block-container {
            padding: 2rem 3rem !important;
            max-width: 1400px !important;
        }
        /* Headers spacing */
        h1, h2, h3 {
            margin-top: 1.5rem !important;
            margin-bottom: 1rem !important;
        }
        /* Dividers */
        hr {
            margin: 2rem 0 !important;
            border-color: rgba(255, 255, 255, 0.1) !important;
        }
        /* Metric cards row spacing */
        div[data-testid="stHorizontalBlock"] {
            gap: 1rem !important;
            margin-bottom: 1rem !important;
        }
        /* Chart containers */
        .stPlotlyChart {
            margin: 1.5rem 0 !important;
        }
        /* Expander spacing */
        .streamlit-expanderHeader {
            margin-top: 1rem !important;
        }
        /* Input fields */
        .stNumberInput, .stSelectbox, .stMultiSelect {
            margin-bottom: 1rem !important;
        }
        /* Button containers */
        .stButton {
            margin-top: 0.5rem !important;
            margin-bottom: 1rem !important;
        }
        /* Caption text */
        .stCaption {
            margin-bottom: 1rem !important;
        }
        /* Info/Success/Warning boxes */
        .stAlert {
            margin: 1rem 0 !important;
        }
        /* DataFrame containers */
        .stDataFrame {
            margin: 1rem 0 !important;
        }
        /* Column containers */
        div[data-testid="column"] {
            padding: 0 0.5rem !important;
        }
        /* NAVIGATION MENU OVERRIDE */
        div[data-testid="stHorizontalBlock"] > div:has(div[data-testid="stRadio"]) {
            background: rgba(17, 25, 40, 0.5);
            border-radius: 50px;
            padding: 4px 12px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 1.5rem;
        }
        div[data-testid="stRadio"] > label {
            display: none;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            flex-direction: row !important;
            gap: 12px !important;
            justify-content: center !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label {
            background: transparent !important;
            border: none !important;
            padding: 8px 16px !important;
            border-radius: 30px !important;
            color: #94a3b8 !important;
            transition: all 0.3s ease !important;
            cursor: pointer !important;
            margin: 0 !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            color: #fff !important;
            background: rgba(255, 255, 255, 0.05) !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
            display: none !important; /* Hide the radio circle */
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
            color: #fff !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        }
        /* PRIMARY BUTTONS - Gradient Style */
        .stButton > button[kind="primary"],
        .stButton > button[data-baseweb="button"][kind="primary"],
        button[kind="primary"] {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
            border: none !important;
            color: white !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.5rem !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        }
        .stButton > button[kind="primary"]:hover,
        button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4) !important;
        }
        /* ALL BUTTONS Fallback */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
            border: none !important;
            color: white !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        }
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(139, 92, 246, 0.4) !important;
        }
        /* TABS - Active Tab Gradient */
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
            color: #fff !important;
            border-bottom: none !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
        }
        /* Download button */
        .stDownloadButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
            border: none !important;
            color: white !important;
            border-radius: 8px !important;
        }
        /* Select Slider active elements */
        .stSlider > div > div > div[role="slider"] {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        }
        div[data-baseweb="slider"] div[role="slider"] {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        }
        /* MULTISELECT CHIPS/TAGS */
        div[data-baseweb="tag"] {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
            border: none !important;
            color: white !important;
        }
        div[data-baseweb="tag"] span {
            color: white !important;
        }
        span[data-baseweb="tag"] {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
            border: none !important;
        }
        /* Multiselect selected items */
        .stMultiSelect [data-baseweb="tag"] {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
            border-radius: 6px !important;
        }
        /* Select box hover and focused states */
        div[data-baseweb="select"] div[data-baseweb="input"] {
            border-color: #8b5cf6 !important;
        }
        div[data-baseweb="popover"] li:hover,
        div[data-baseweb="menu"] li:hover {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        }
        /* Progress bar */
        .stProgress > div > div > div > div {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        }
        /* Checkbox checked */
        .stCheckbox input:checked + div {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
            border-color: #8b5cf6 !important;
        }
        /* Radio button selected */
        .stRadio input:checked + div > div:first-child {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        }
        /* Spinner/loading */
        .stSpinner > div > div {
            border-top-color: #8b5cf6 !important;
        }
        /* Toggle switch */
        div[data-baseweb="checkbox"] div[data-baseweb="checkbox"] span {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
        }
        /* Selectbox dropdown item hover */
        [data-baseweb="menu"] [role="option"]:hover {
            background: rgba(139, 92, 246, 0.2) !important;
        }
        /* Number input focus */
        .stNumberInput input:focus {
            border-color: #8b5cf6 !important;
            box-shadow: 0 0 0 1px #8b5cf6 !important;
        }
        /* Text input focus */
        .stTextInput input:focus {
            border-color: #8b5cf6 !important;
            box-shadow: 0 0 0 1px #8b5cf6 !important;
        }
        /* MOBILE RESPONSIVENESS */
        @media only screen and (max-width: 768px) {
            /* Reduce padding on mobile */
            .block-container {
                padding: 1rem 1rem !important;
            }
            
            /* Smaller headers */
            h1 { font-size: 1.8rem !important; }
            h2 { font-size: 1.5rem !important; }
            h3 { font-size: 1.2rem !important; }
            
            /* Stack tabs for better touch targets */
            .stTabs [data-baseweb="tab-list"] {
                flex-wrap: wrap;
                gap: 8px;
            }
            
            .stTabs [data-baseweb="tab"] {
                width: 100%; /* Full width tabs on mobile */
                justify-content: center;
                margin: 0;
            }
            
            /* Adjust charts */
            .stPlotlyChart {
                height: 350px !important;
            }
            
            /* Metric cards spacing */
            div[data-testid="stHorizontalBlock"] {
                gap: 0.5rem !important;
            }
            
            /* Hide non-essential elements if needed */
            .mobile-hide {
                display: none;
            }
        }
    </style>
    """, unsafe_allow_html=True)
def style_financial_dataframe(df):
    """
    Applies custom neon styling to a pandas DataFrame.
    Returns a Styler object.
    Expects numeric columns to be raw floats.
    """
    def color_negative_red(val):
        if not isinstance(val, (int, float)):
            return ''
        color = '#4ADE80' if val > 0 else '#F87171' if val < 0 else '#e0e0e0'
        return f'color: {color}'
    # Identify percentage columns vs scalar columns
    # Heuristic: CAGR, Alpha, Volatility, Max DD, Returns are %, Sharpe, Beta are scalars
    pct_cols = [c for c in df.columns if any(x in c for x in ['CAGR', 'Alpha', 'Volatility', 'Max DD', 'Return', 'Drawdown'])]
    float_cols = [c for c in df.columns if any(x in c for x in ['Sharpe', 'Beta', 'Sortino', 'R²'])]
    
    styler = df.style.applymap(color_negative_red, subset=pct_cols + float_cols)
    
    if pct_cols:
        styler = styler.format("{:+.2f}%", subset=pct_cols)
    if float_cols:
        styler = styler.format("{:.2f}", subset=float_cols)
        
    return styler
def metric_card(label, value, delta=None, is_good_if_positive=True, suffix="", prefix=""):
    """
    Renders a stylized metric card using custom HTML/CSS logic.
    Supports Neon Green/Red indicators based on delta.
    """
    if delta is not None:
        try:
            # Handle string percentages or direct float
            if isinstance(delta, str):
                delta_clean = delta.replace('%', '').replace('+', '')
                delta_val = float(delta_clean)
            else:
                delta_val = float(delta)
        except:
            delta_val = 0
            
        color = get_neon_color(delta_val, is_good_if_positive)
        arrow = "↑" if delta_val > 0 else "↓"
        if delta_val == 0: arrow = ""
        
        # Format delta string if not already formatted
        delta_str = f"{delta}" if isinstance(delta, str) else f"{delta_val:+.2f}%"
        
        # HTML with NO INDENTATION to prevent code block rendering
        html = f"""
<div class="custom-metric-box" style="border-left: 4px solid {color};">
<div style="font-size: 0.85rem; color: #8b949e; margin-bottom: 4px;">{label}</div>
<div style="font-size: 1.8rem; font-weight: 700; color: #fff;">
{prefix}{value}{suffix}
</div>
<div style="font-size: 0.9rem; font-weight: 600; color: {color}; margin-top: 4px;">
{arrow} {delta_str}
</div>
</div>
""".strip()
        st.markdown(html, unsafe_allow_html=True)
    else:
        # Simple display without delta
        html = f"""
<div class="custom-metric-box" style="border-left: 4px solid #555;">
<div style="font-size: 0.85rem; color: #8b949e; margin-bottom: 4px;">{label}</div>
<div style="font-size: 1.8rem; font-weight: 700; color: #fff;">
{prefix}{value}{suffix}
</div>
</div>
""".strip()
        st.markdown(html, unsafe_allow_html=True)


def add_max_drawdown_annotation(fig, x_date, y_val, text="Max Drawdown Point"):
    """
    Adds a consistent Max Drawdown annotation to a Plotly figure.
    """
    fig.add_annotation(
        x=x_date, y=y_val,
        text=text,
        showarrow=True, arrowhead=2,
        ax=0, ay=40,
        bgcolor="rgba(239, 68, 68, 0.8)",
        font=dict(color="white", size=10),
        bordercolor="rgba(239, 68, 68, 1)",
        borderwidth=1,
        borderpad=4
    )


def add_drawdown_vertical_line(fig, x_date, color="rgba(239, 68, 68, 0.3)"):
    """
    Adds a vertical dashed line to highlight a drawdown event.
    """
    fig.add_vline(x=x_date, line_dash="dash", line_color=color)
def render_capture_ratio_chart(upside_capture, downside_capture, fund_name="Fund"):
    """
    Renders a bar chart for Upside/Downside Capture Ratios.
    """
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    # Target benchmark baseline
    fig.add_hline(y=100, line_dash="dash", line_color="rgba(255,255,255,0.3)", 
                  annotation_text="Benchmark (100%)", annotation_position="top right")
    
    fig.add_trace(go.Bar(
        x=['Upside Capture', 'Downside Capture'],
        y=[upside_capture, downside_capture],
        marker_color=['#10b981', '#ef4444'],
        text=[f"{upside_capture:.1f}%", f"{downside_capture:.1f}%"],
        textposition='auto',
        width=0.4
    ))
    
    fig.update_layout(
        title=f"Capture Ratios: {fund_name} vs Benchmark",
        yaxis_title="Ratio (%)",
        yaxis_range=[0, max(120, upside_capture + 20, downside_capture + 20)],
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=300,
        margin=dict(t=50, b=20, l=40, r=40)
    )
    
    # Add animation
    fig = animate_chart(fig, animation_mode="growth")
    
    st.plotly_chart(fig, use_container_width=True)
def animate_chart(fig, animation_mode="auto", num_frames=30):
    """
    Adds a Play/Pause button and an animation to a Plotly Figure.
    
    Args:
        fig (go.Figure): The plotly figure to animate.
        animation_mode (str): 'auto', 'time_series' (reveal x-axis), 'growth' (grow y-axis).
        num_frames (int): Number of animation frames.
        
    Returns:
        go.Figure: The animated figure.
    """
    # Check if figure has data
    if not fig.data:
        return fig
        
    # Determine mode if auto
    if animation_mode == "auto":
        # Heuristic: If scatter/line with > 10 points -> time_series
        # If bar -> growth
        trace_type = fig.data[0].type
        if trace_type in ['scatter', 'scattergl']:
            # Check if there are many points
            if len(fig.data[0].x) > 10:
                animation_mode = "time_series"
            else:
                animation_mode = "none"
        elif trace_type == 'bar':
            animation_mode = "growth"
        else:
            animation_mode = "none"
            
    if animation_mode == "none":
        return fig
        
    # --- TIME SERIES ANIMATION (Reveal X) ---
    if animation_mode == "time_series":
        # Find global x-range
        min_x, max_x = None, None
        is_datetime = False
        
        # Collect all X data to find range
        all_x_list = []
        for trace in fig.data:
            if trace.x is not None:
                # Handle tuple/list/ndarray -> Series
                all_x_list.extend(pd.Series(trace.x).tolist())
        
        if not all_x_list: 
            return fig
            
        # Detect type from sample
        try:
            sample = all_x_list[0]
            if isinstance(sample, (str, pd.Timestamp, datetime)):
                # If it's a date string or timestamp
                pd.to_datetime(sample) # check validity
                is_datetime = True
        except:
            is_datetime = False
            
        all_x_series = pd.Series(all_x_list)
        if is_datetime:
             all_x_series = pd.to_datetime(all_x_series)
        
        all_x_series = all_x_series.sort_values()
        min_x = all_x_series.iloc[0]
        max_x = all_x_series.iloc[-1]
        
        # Generate cut points
        if is_datetime:
            # Linear interpolation on timestamps
            min_ts = min_x.value
            max_ts = max_x.value
            cut_points_ts = np.linspace(min_ts, max_ts, num_frames)
            cut_points = pd.to_datetime(cut_points_ts)
        else:
            cut_points = np.linspace(min_x, max_x, num_frames)
             
        # Create Frames
        frames = []
        for i, cut_val in enumerate(cut_points):
            frame_data = []
            for trace in fig.data:
                # Get data
                x_data = pd.Series(trace.x)
                y_data = pd.Series(trace.y)
                
                if is_datetime:
                    x_data = pd.to_datetime(x_data)
                    mask = x_data <= cut_val
                else:
                    mask = x_data <= cut_val
                
                # Filter (avoid empty arrays if start is late)
                filt_x = x_data[mask]
                filt_y = y_data[mask]
                
                frame_data.append(dict(
                    type=trace.type,
                    x=filt_x,
                    y=filt_y
                ))
            
            frames.append(dict(data=frame_data, name=f"fr{i}"))
            
        fig.frames = frames
        
        # Add Play Button
        # We place it nicely in the top right or modebar?
        # Layout update
        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                direction="left",
                y=1.13,
                x=1.0,
                xanchor="right",
                yanchor="top",
                pad={"r": 10, "t": 10},
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.2)",
                font=dict(color="white"),
                buttons=[dict(
                    label="▶ Play",
                    method="animate",
                    args=[None, dict(frame=dict(duration=60, redraw=False), 
                                     fromcurrent=True, 
                                     transition=dict(duration=0))]
                ),
                dict(
                    label="⏸ Pause",
                    method="animate",
                    args=[[None], dict(frame=dict(duration=0, redraw=False), 
                                       mode="immediate", 
                                       transition=dict(duration=0))]
                )]
            )]
        )
    # --- BAR GROWTH ANIMATION ---
    elif animation_mode == "growth":
        # Bar chart grows from 0 to Y
        frames = []
        for i in range(1, num_frames + 1):
            frac = i / num_frames
            frame_data = []
            for trace in fig.data:
                if trace.y is not None:
                     y_orig = np.array(trace.y, dtype=float) # ensuring float
                     y_new = y_orig * frac
                     frame_data.append(dict(y=y_new))
            frames.append(dict(data=frame_data, name=f"fr{i}"))
            
        fig.frames = frames
        
        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                y=1.13,
                x=1.0,
                xanchor="right",
                yanchor="top",
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="rgba(255,255,255,0.2)",
                font=dict(color="white"),
                buttons=[dict(
                    label="▶ Animate",
                    method="animate",
                    args=[None, dict(frame=dict(duration=50, redraw=True), 
                                     fromcurrent=True, 
                                     transition=dict(duration=0))]
                )]
            )]
        )
    return fig
def render_welcome_card(num_funds=14000):
    """
    Renders the premium 'Welcome to Welment Capital' hero section.
    Used as an empty state for various dashboard views.
    Responsive design that works on mobile and desktop.
    """
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Use a centered column layout that works on all widths
    # On mobile, columns collapse naturally if we rely on CSS/HTML flex
    # But for Streamlit, we'll use a single container with max-width style
    
    html_content = (
        f"<div style='display: flex; justify-content: center; align-items: center; margin: 0 auto; max-width: 900px;'>"
        f"    <div style='text-align: center; padding: 40px; background: rgba(255,255,255,0.03); border-radius: 20px; border: 1px solid rgba(255,255,255,0.05); width: 100%;'>"
        f"        <h2 style='color: white; margin-bottom: 10px; font-size: clamp(1.5rem, 5vw, 2.2rem);'>Welcome to Welment Capital</h2>"
        f"        <p style='color: #888; font-size: clamp(0.9rem, 3vw, 1.1rem);'>Advanced Mutual Fund Analytics & Portfolio Optimization</p>"
        f"        <div style='margin-top: 30px; padding: 15px; background: rgba(59, 130, 246, 0.1); border-radius: 12px; border: 1px dashed rgba(59, 130, 246, 0.3); display: inline-block;'>"
        f"            <p style='color: #3b82f6; margin: 0; font-size: 0.95rem;'>👈 <b>Get Started:</b> Select { 'a fund' if num_funds > 1 else 'funds' } from the sidebar</p>"
        f"        </div>"
        f"        <div style='margin-top: 40px; display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 20px; justify-content: center;'>"
        f"            <div style='text-align: center;'>"
        f"                <h4 style='color: white; margin: 0; font-size: 1.4rem;'>{num_funds}+</h4>"
        f"                <p style='color: #666; font-size: 0.8rem;'>Schemes Tracked</p>"
        f"            </div>"
        f"            <div style='text-align: center;'>"
        f"                <h4 style='color: white; margin: 0; font-size: 1.4rem;'>Real-time</h4>"
        f"                <p style='color: #666; font-size: 0.8rem;'>NAV Updates</p>"
        f"            </div>"
        f"            <div style='text-align: center;'>"
        f"                <h4 style='color: white; margin: 0; font-size: 1.4rem;'>Advanced</h4>"
        f"                <p style='color: #666; font-size: 0.8rem;'>Risk Metrics</p>"
        f"            </div>"
        f"        </div>"
        f"    </div>"
        f"</div>"
    )
    st.markdown(html_content, unsafe_allow_html=True)
def render_smart_insights(stats, prob_success, prob_loss, years):
    """
    Renders smart textual insights based on Monte Carlo simulation results.
    """
    import streamlit as st
    
    st.markdown("#### 🧠 Smart Insights & Interpretation")
    
    # 1. Volatility / Spread Analysis
    optimistic_cagr = stats.get('optimistic_cagr', 0)
    pessimistic_cagr = stats.get('pessimistic_cagr', 0)
    expected_cagr = stats.get('expected_cagr', 0)
    
    spread = optimistic_cagr - pessimistic_cagr
    
    st.markdown("**1. Volatility Assessment**")
    if spread > 25:
        st.warning(f"⚠️ **High Uncertainty**: The gap between optimistic ({optimistic_cagr:.1f}%) and pessimistic ({pessimistic_cagr:.1f}%) outcomes is massive ({spread:.1f}%). Expect significant ups and downs along the way.")
    elif spread > 15:
        st.info(f"ℹ️ **Moderate Volatility**: There is a noticeable difference between best and worst case scenarios. Ensure you have the holding capacity for {years} years.")
    else:
        st.success(f"✅ **High Predictability**: The narrow spread ({spread:.1f}%) suggests this investment tracks its mean relatively closely, offering a smoother ride.")

    st.markdown("**2. Downside vs Inflation**")
    inflation_rate = 6.0
    real_pessimistic = pessimistic_cagr - inflation_rate
    
    if pessimistic_cagr < 0:
        st.error(f"🛑 **Capital Erosion Risk**: In the worst 5% of scenarios, you could lose money ({pessimistic_cagr:.1f}% CAGR). Not suitable for critical short-term goals.")
    elif pessimistic_cagr < inflation_rate:
        st.warning(f"⚠️ **Inflation Risk**: In a worst-case scenario, returns ({pessimistic_cagr:.1f}%) may not beat inflation ({inflation_rate}%), resulting in loss of purchasing power.")
    else:
        st.success(f"🛡️ **Inflation Beating**: Even in the worst 5% of market conditions, this investment is projected to beat inflation ({pessimistic_cagr:.1f}% vs {inflation_rate}%), maintaining your purchasing power.")

    # 3. Overall Verdict
    st.markdown("**3. The Verdict**")
    if prob_success > 75 and pessimistic_cagr > 0:
        st.success("🌟 **Prime Candidate**: This investment shows a high probability of meeting targets with a solid safety net against capital loss. Excellent for core portfolio allocation.")
    elif prob_success < 50:
        st.error("📉 **Underperformer Potential**: Less than 50% chance of meeting your 12% target. You might need to increase your SIP amount or extend your time horizon to compensate for lower expected growth.")
    elif expected_cagr > 15 and spread > 20:
        st.info("🚀 **High Risk / High Reward**: Offers great growth potential but requires 'Iron Hands' to sit through volatility. Best for long-term wealth creation, not for money needed soon.")
    else:
        st.info("⚖️ **Balanced Choice**: Offers a reasonable trade-off between risk and reward. Monitor performance annually.")


def render_rolling_smart_insights(rolling_stats, window_years):
    """
    Renders smart textual insights based on Rolling Return statistics.
    """
    import streamlit as st
    
    st.markdown("#### 🧠 Consistency & Risk Insights")
    
    # 1. Performance Consistency
    outperform = rolling_stats.get('outperformance_pct', 0)
    st.markdown("**1. Outperformance Consistency**")
    if outperform > 80:
        st.success(f"🏆 **Elite Consistency**: This fund beat the benchmark in {outperform:.1f}% of all rolling periods. It shows strong management alpha.")
    elif outperform > 60:
        st.info(f"📈 **Reliable Performer**: Beating the benchmark {outperform:.1f}% of the time indicates a healthy edge over the market.")
    elif outperform > 40:
        st.warning(f"⚖️ **Average Consistency**: At {outperform:.1f}% outperformance, the fund is neck-and-neck with the benchmark. Benefits might be limited.")
    else:
        st.error(f"📉 **Laggard Alert**: The fund beat the benchmark only {outperform:.1f}% of the time. Re-evaluate if this fits your strategy.")

    # 2. Capital Protection (Rolling)
    neg_periods = rolling_stats.get('negative_periods_pct', 0)
    fund_min = rolling_stats.get('fund_min', 0)
    st.markdown("**2. Downside Protection**")
    if neg_periods == 0:
        st.success(f"🛡️ **Fortress Returns**: Zero negative rolling periods over {window_years}Y windows. Highly stable for your time horizon.")
    elif neg_periods < 5:
        st.info(f"✅ **Strong Protection**: Only {neg_periods:.1f}% of periods saw negative returns. Worst case was {fund_min:.1f}%.")
    elif neg_periods < 15:
        st.warning(f"⚠️ **Moderate Risk**: {neg_periods:.1f}% of investors saw negative returns during this window size. Be prepared for occasional dips.")
    else:
        st.error(f"🛑 **High Loss Probability**: {neg_periods:.1f}% of rolling windows ended in losses. High patience required.")

    # 3. Target Achievement (12% CAGR)
    beating_target = rolling_stats.get('beating_target_pct', 0)
    st.markdown("**3. Wealth Creation Potential**")
    if beating_target > 70:
        st.success(f"🚀 **Wealth Multiplier**: Met the 12% growth target in {beating_target:.1f}% of periods. Strong for long-term goals.")
    elif beating_target > 40:
        st.info(f"📊 **Decent Potential**: Met the 12% target in {beating_target:.1f}% of periods. Likely to deliver mid-tier growth.")
    else:
        st.warning(f"🐢 **Slow Grower**: Only met the 12% target in {beating_target:.1f}% of periods. May require higher SIP amounts to reach goals.")

    # 4. Final Verdict
    st.markdown("**4. The Verdict**")
    if outperform > 70 and neg_periods < 5:
        st.success("🌟 **High Conviction**: Combines superior consistency with excellent downside protection. A core portfolio candidate.")
    elif outperform < 40 or neg_periods > 20:
        st.error("📉 **High Caution**: Inconsistent performance or high risk of loss. Consider as a satellite holding or look for alternatives.")
    elif fund_min > 5:
        st.success("💎 **Premium Stability**: Even the 'worst' time to invest yielded >5% returns. Excellent for conservative growth.")
    else:
        st.info("⚖️ **Balanced Mid-Cap/Large-Cap**: Shows typical market behavior with cycles of performance. Good for diversified portfolios.")







