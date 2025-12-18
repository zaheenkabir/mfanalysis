import pandas as pd
import requests
from io import StringIO
import streamlit as st

AMFI_NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt"

@st.cache_data
def fetch_latest_nav():
    """
    Fetches the latest NAV data from AMFI.
    Returns a pandas DataFrame.
    """
    # TODO: Implement caching strategy
    try:
        response = requests.get(AMFI_NAV_URL)
        response.raise_for_status()
        
        # AMFI data is semicolon separated
        # Scheme Code;ISIN Div Payout/ISIN Growth;ISIN Div Reinvestment;Scheme Name;Net Asset Value;Date
        data = StringIO(response.text)
        df = pd.read_csv(data, sep=";", dtype={"Scheme Code": str})
        
        # Clean up column names and data
        df = df.dropna(subset=["Net Asset Value", "Date"])
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
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
        df = df.set_index('date')
        
        return df[['nav']]
    except Exception as e:
        print(f"Error fetching history for {scheme_code}: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_scheme_details(scheme_code):
    """
    Fetches scheme details - stub function (mftool is deprecated).
    Returns empty dict as scheme details are not critical for core functionality.
    """
    # mftool is deprecated, returning empty dict
    # The app works fine without detailed scheme metadata
    return {}
