import pandas as pd
import requests
from io import StringIO
import streamlit as st

AMFI_NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt"

def adjust_nav_for_splits(series):
    """
    Detects and adjusts for unadjusted NAV splits or face value changes.
    Uses daily returns to identify massive jumps (>70% drop or >300% rise) 
    that look like splits, and reconstructs the series to be continuous.
    """
    if len(series) < 2:
        return series
    
    # Calculate daily returns
    returns = series.pct_change()
    
    # Heuristic: Mutual funds/ETFs don't drop 70%+ in a day unless it's a split.
    # Similarly, they don't rise 300%+ unless it's a reverse split or merger.
    is_split = (returns < -0.7) | (returns > 3.0)
    
    if not is_split.any():
        return series
    
    # Create a copy of returns and replace split jumps with 0% (neutral bridge)
    clean_returns = returns.copy()
    clean_returns[is_split] = 0.0
    
    # Reconstruct price series: re-cumulative growth starting from the original first value
    reconstructed = (1 + clean_returns.fillna(0)).cumprod() * series.iloc[0]
    
    # Ensure it's the same type/name
    reconstructed.name = series.name
    return reconstructed

@st.cache_data
def fetch_latest_nav():
    """
    Fetches the latest NAV data from AMFI.
    Returns a pandas DataFrame with Scheme metadata.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(AMFI_NAV_URL, headers=headers, timeout=15)
        response.encoding = 'utf-8' # Ensure correct decoding of smart quotes
        response.raise_for_status()
        
        lines = response.text.splitlines()
        processed_data = []
        current_category = "Other"
        current_fund_house = "Unknown"
        
        # Headers we expect in the AMFI file
        # Scheme Code;ISIN Div Payout/ ISIN Growth;ISIN Div Reinvestment;Scheme Name;Net Asset Value;Date
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            # Detect Categories
            # liberal match: any non-data line containing 'Schemes'
            if ";" not in line and "Schemes" in line:
                # Clean up category names
                cat = line
                if "(" in cat and ")" in cat:
                    # Extract content inside parentheses
                    start = cat.find("(") + 1
                    end = cat.find(")")
                    cat = cat[start:end].strip()
                elif " - " in cat:
                    # Remove "Schemes - " or similar prefixes
                    cat = cat.split(" - ", 1)[-1].strip()
                elif "(" in cat:
                     # Handle cases like "Schemes(Equity..." without closing paren on same line
                     cat = cat.split("(", 1)[-1].strip()
                
                # Further cleanup: remove "Open Ended " if it persisted
                cat = cat.replace("Open Ended Schemes", "").replace("Open Ended", "").strip()
                # Fix encoding/smart characters: replace smart quote with normal, smart dash with hyphen
                cat = cat.replace("â€™", "'").replace("’", "'").replace("â€“", "-").replace("–", "-")
                
                current_category = cat
                continue
            
            # Detect Fund Houses
            # These are usually after category headers and before data rows
            if ";" not in line and ("Mutual Fund" in line or "Asset Management" in line):
                current_fund_house = line
                continue
            
            # Detect Data rows
            parts = line.split(";")
            if len(parts) >= 6 and parts[0].isdigit():
                processed_data.append({
                    "Scheme Code": parts[0],
                    "Scheme Name": parts[3],
                    "Net Asset Value": parts[4],
                    "Date": parts[5],
                    "Category": current_category,
                    "Fund House": current_fund_house
                })
        
        df = pd.DataFrame(processed_data)
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Filter out any lingering header rows that might have text in numeric columns
        # Coerce Net Asset Value to numeric, turning errors (text) into NaN
        df['Net Asset Value'] = pd.to_numeric(df['Net Asset Value'], errors='coerce')
        
        # Drop rows where NAV became NaN after coercion
        df = df.dropna(subset=['Net Asset Value'])
        
        return df
    except Exception as e:
        st.error(f"Error fetching AMFI data: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_fund_history(scheme_code):
    """
    Fetches historical NAV data for a given scheme code from mfapi.in.
    Returns a DataFrame with 'date' and 'nav' columns.
    """
    url = f"https://api.mfapi.in/mf/{scheme_code}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        if data.get('status') != 'SUCCESS':
            return pd.DataFrame()
            
        nav_data = data.get('data', [])
        df = pd.DataFrame(nav_data)
        
        # Convert columns
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y')
        df['nav'] = pd.to_numeric(df['nav'])
        
        # Sort by date ascending
        df = df.sort_values('date')
        
        # Remove duplicate dates (keep latest entry if duplicates exist)
        df = df.drop_duplicates(subset='date', keep='last')
        
        # Standardize index
        df = df.set_index('date')
        
        # Apply Split Adjustment
        df['nav'] = adjust_nav_for_splits(df['nav'])
        
        return df[['nav']]
    except Exception as e:
        print(f"Error fetching history for {scheme_code}: {e}")
        return pd.DataFrame()

def fetch_scheme_details(scheme_code, nav_df=None):
    """
    Returns scheme details from the loaded NAV dataframe.
    """
    if nav_df is None or nav_df.empty:
        return {}
    
    # Filter for the specific scheme code
    scheme_info = nav_df[nav_df['Scheme Code'] == str(scheme_code)]
    
    if scheme_info.empty:
        return {}
    
    info_row = scheme_info.iloc[0]
    return {
        'scheme_name': info_row.get('Scheme Name', 'N/A'),
        'scheme_category': info_row.get('Category', 'N/A'),
        'fund_house': info_row.get('Fund House', 'N/A'),
        'nav': info_row.get('Net Asset Value', 'N/A')
    }

