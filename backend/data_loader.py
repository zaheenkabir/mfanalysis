import pandas as pd
import requests
from io import StringIO
import streamlit as st
import sqlite3
import os
from datetime import datetime, date

DB_PATH = os.path.join(os.path.dirname(__file__), "nav_cache.db")

def init_db():
    """Initializes the SQLite database for NAV caching."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS nav_history (
            scheme_code TEXT,
            date TEXT,
            nav REAL,
            PRIMARY KEY (scheme_code, date)
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolios (
            portfolio_name TEXT PRIMARY KEY,
            funds_json TEXT, -- JSON string of {fund_display: weight}
            last_updated TEXT
        )
    """)
    conn.commit()
    conn.close()

# Initialize on module load
init_db()

AMFI_NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt"

def adjust_nav_for_splits(series):
    """
    Enhanced split correction engine.
    Detects jumps (>70% drop or >300% rise) and applies a multiplier to align history.
    This version handles multiple splits in a single series.
    """
    if len(series) < 2:
        return series
    
    series = series.copy().sort_index()
    # Iterate backwards to adjust historical values to match the latest NAV
    for i in range(len(series) - 1, 0, -1):
        prev_val = series.iloc[i-1]
        curr_val = series.iloc[i]
        
        if prev_val == 0 or pd.isna(prev_val) or pd.isna(curr_val):
            continue
            
        ratio = curr_val / prev_val
        
        # Split Check: Drop of 70%+ or Rise of 300%+
        if ratio < 0.3 or ratio > 4.0:
            # Shift all historical data before this point by the ratio
            series.iloc[:i] = series.iloc[:i] * ratio
            
    return series

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
                
                # Strip "Other Scheme - " prefix for cleaner categorization (User Request)
                if cat.startswith("Other Scheme - "):
                    cat = cat.replace("Other Scheme - ", "").strip()
                
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
                scheme_name = parts[3]
                category = current_category
                
                # Refine FoF Domestic into Gold ETF, Silver ETF, and Other Domestic FoF
                if current_category == "FoF Domestic":
                    scheme_name_lower = scheme_name.lower()
                    if "gold" in scheme_name_lower:
                        category = "Gold ETF"
                    elif "silver" in scheme_name_lower:
                        category = "Silver ETF"
                    else:
                        category = "Other Domestic FoF"
                
                processed_data.append({
                    "Scheme Code": parts[0],
                    "Scheme Name": scheme_name,
                    "Net Asset Value": parts[4],
                    "Date": parts[5],
                    "Category": category,
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
    Fetches historical NAV data for a given scheme code.
    Uses a local SQLite cache and fetches incremental updates from mfapi.in if needed.
    """
    scheme_code = str(scheme_code)
    conn = sqlite3.connect(DB_PATH)
    
    # 1. Load from Cache
    try:
        cache_df = pd.read_sql(
            f"SELECT date, nav FROM nav_history WHERE scheme_code = '{scheme_code}'", 
            conn
        )
        if not cache_df.empty:
            cache_df['date'] = pd.to_datetime(cache_df['date']).dt.normalize().dt.tz_localize(None)
            cache_df = cache_df.set_index('date').sort_index()
    except Exception:
        cache_df = pd.DataFrame()

    # 2. Check if Cache is Up-to-Date
    today = date.today()
    # AMFI data is usually updated late evening or next morning. 
    # If we have data from yesterday or today, we consider it fresh enough for a "sync check"
    last_cached_date = cache_df.index.max().date() if not cache_df.empty else date(2000, 1, 1)
    
    if (today - last_cached_date).days > 1:
        # Cache is stale or empty, fetch from API
        url = f"https://api.mfapi.in/mf/{scheme_code}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if data.get('status') == 'SUCCESS':
                nav_data = data.get('data', [])
                api_df = pd.DataFrame(nav_data)
                api_df['date'] = pd.to_datetime(api_df['date'], format='%d-%m-%Y').dt.normalize().dt.tz_localize(None)
                api_df['nav'] = pd.to_numeric(api_df['nav'])
                
                # Filter for only new data to append
                new_data = api_df[api_df['date'].dt.date > last_cached_date].copy()
                
                if not new_data.empty:
                    # Save new data to SQLite
                    new_data_to_db = new_data.copy()
                    new_data_to_db['scheme_code'] = scheme_code
                    # Convert date to string for SQLite storage
                    new_data_to_db['date'] = new_data_to_db['date'].dt.strftime('%Y-%m-%d')
                    new_data_to_db[['scheme_code', 'date', 'nav']].to_sql(
                        'nav_history', conn, if_exists='append', index=False
                    )
                    
                    # Merge with existing cache for the return
                    new_data = new_data.set_index('date')[['nav']]
                    full_df = pd.concat([cache_df, new_data]).sort_index()
                    full_df = full_df[~full_df.index.duplicated(keep='last')]
                else:
                    full_df = cache_df
            else:
                full_df = cache_df # API failed, use what we have
        except Exception as e:
            print(f"Sync error for {scheme_code}: {e}")
            full_df = cache_df
    else:
        full_df = cache_df

    conn.close()
    
    if full_df.empty:
        return pd.DataFrame()
        
    # Apply cumulative split adjustment to the full series
    full_df['nav'] = adjust_nav_for_splits(full_df['nav'])
    return full_df[['nav']]

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

def save_portfolio(name, weights_dict):
    """Saves a named portfolio to the SQLite database."""
    import json
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    funds_json = json.dumps(weights_dict)
    last_updated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute("""
        INSERT INTO portfolios (portfolio_name, funds_json, last_updated)
        VALUES (?, ?, ?)
        ON CONFLICT(portfolio_name) DO UPDATE SET
            funds_json=excluded.funds_json,
            last_updated=excluded.last_updated
    """, (name, funds_json, last_updated))
    
    conn.commit()
    conn.close()

def load_all_portfolios():
    """Returns all saved portfolios from the database."""
    import json
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql("SELECT portfolio_name, funds_json FROM portfolios", conn)
        results = {}
        for _, row in df.iterrows():
            results[row['portfolio_name']] = json.loads(row['funds_json'])
        return results
    except Exception:
        return {}
    finally:
        conn.close()

def delete_portfolio(name):
    """Deletes a portfolio from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM portfolios WHERE portfolio_name = ?", (name,))
    conn.commit()
    conn.close()

