from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import pandas as pd
from data_loader import fetch_latest_nav, fetch_fund_history
from analytics import get_fund_metrics, calculate_rolling_returns

app = FastAPI(title="Mutual Fund Dashboard API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global cache for fund list (simple in-memory cache)
FUND_LIST_CACHE = []

class FundItem(BaseModel):
    scheme_code: str
    scheme_name: str

class FundMetrics(BaseModel):
    cagr: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float

class FundDetail(BaseModel):
    scheme_code: str
    scheme_name: str
    metrics: FundMetrics
    history: List[dict] # List of {date: str, nav: float}

@app.on_event("startup")
async def load_funds():
    global FUND_LIST_CACHE
    print("Loading fund list...")
    df = fetch_latest_nav()
    if not df.empty:
        # Filter for Growth plans usually, but let's keep it broad for now
        # Just keep relevant columns
        funds = df[['Scheme Code', 'Scheme Name']].dropna().to_dict('records')
        FUND_LIST_CACHE = [
            FundItem(scheme_code=str(f['Scheme Code']), scheme_name=f['Scheme Name'])
            for f in funds
        ]
    print(f"Loaded {len(FUND_LIST_CACHE)} funds.")

@app.get("/funds", response_model=List[FundItem])
def get_funds(search: Optional[str] = None):
    if search:
        return [
            f for f in FUND_LIST_CACHE 
            if search.lower() in f.scheme_name.lower()
        ][:50]
    return FUND_LIST_CACHE[:100] # Limit default return size

@app.get("/funds/{scheme_code}", response_model=FundDetail)
def get_fund_details(scheme_code: str):
    # Find name
    fund_name = next((f.scheme_name for f in FUND_LIST_CACHE if f.scheme_code == scheme_code), "Unknown Fund")
    
    # Fetch history
    history_df = fetch_fund_history(scheme_code)
    if history_df.empty:
        raise HTTPException(status_code=404, detail="Fund history not found")
    
    # Calculate metrics
    metrics = get_fund_metrics(history_df['nav'])
    
    # Prepare history for chart (resample to reduce payload if needed, or send all)
    # sending all for now, but maybe last 5 years is enough
    history_data = history_df.reset_index()
    history_data['date'] = history_data['date'].dt.strftime('%Y-%m-%d')
    history_list = history_data.to_dict('records')
    
    return FundDetail(
        scheme_code=scheme_code,
        scheme_name=fund_name,
        metrics=FundMetrics(
            cagr=metrics.get("CAGR", 0),
            volatility=metrics.get("Volatility", 0),
            sharpe_ratio=metrics.get("Sharpe Ratio", 0),
            max_drawdown=metrics.get("Max Drawdown", 0)
        ),
        history=history_list[-1250:] # LAST 5 YEARS (approx 250*5)
    )
