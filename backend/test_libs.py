from mftool import Mftool
import yfinance as yf
import pandas as pd

def test_mftool():
    print("Testing mftool...")
    obj = Mftool()
    # Test getting details for a common fund, e.g., Quant Small Cap (check code)
    # Using a known code or searching
    try:
        # HDFC Banking and PSU (from previous interaction) code is 128628 (Check if mftool uses same codes)
        # mftool uses AMFI codes usually
        code = '128628' 
        quote = obj.get_scheme_quote(code)
        print(f"Quote for {code}: {quote}")
        
        details = obj.get_scheme_details(code)
        print(f"Details: {details}")
    except Exception as e:
        print(f"mftool error: {e}")

def test_yfinance():
    print("\nTesting yfinance...")
    try:
        # Nifty 50 ticker
        nifty = yf.Ticker("^NSEI")
        hist = nifty.history(period="1mo")
        print(f"Nifty 50 history:\n{hist.head()}")
    except Exception as e:
        print(f"yfinance error: {e}")

if __name__ == "__main__":
    test_mftool()
    test_yfinance()
