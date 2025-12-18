from mftool import Mftool
import yfinance as yf
import pandas as pd

def test_holdings():
    print("Testing mftool capabilities for holdings...")
    obj = Mftool()
    # Quant Small Cap: 120823 (Direct Plan Growth) ? Check codes.
    # Trying a known code or random
    code = '118550' # Some random code? 
    # Let's search for Quant Small Cap
    schemes = obj.get_scheme_codes()
    q_code = [k for k, v in schemes.items() if 'Quant Small Cap' in v and 'Growth' in v and 'Direct' in v]
    print(f"Quant Codes: {q_code}")
    
    if q_code:
        # Check if we can get anything more detailed
        # mftool library logic is often simple, let's look at available methods
        # It has 'get_scheme_details'
        det = obj.get_scheme_details(q_code[0])
        print("Scheme keys:", det.keys())
        
def test_benchmarks():
    tickers = {
        "Nifty 50": "^NSEI",
        "Nifty Bank": "^NSEBANK",
        "Smallcap": "^CNXSC",   # Often problematic on YF, might need symbol hunt
        "Midcap": "^NSEMDCP50"
    }
    
    print("\nTesting Benchmark Tickers on YF:")
    for name, ticker in tickers.items():
        try:
            dat = yf.Ticker(ticker).history(period="5d")
            print(f"{name} ({ticker}): {'Found' if not dat.empty else 'Empty'}")
        except Exception as e:
            print(f"{name} ({ticker}): Error {e}")

if __name__ == "__main__":
    test_holdings()
    test_benchmarks()
