# Welment Capital 💰
### Advanced Mutual Fund Analytics & Portfolio Builder

An institutional-grade dashboard for analyzing Indian mutual funds, constructing portfolios, and stress-testing against major market events.

## 🌟 Features

- **📊 Single Fund Analysis**: Deep dive into individual schemes wth performance metrics, risk ratios (Sharpe, Sortino, Alpha), and interactive charts.
- **⚖️ Compare Funds**: Head-to-head comparison of up to 5 funds with rolling returns, drawdown analysis, and capture ratios.
- **🎯 Build & Analyze Portfolio**: detailed portfolio construction with custom weights, correlation matrix, and efficient frontier analysis.
- **🧪 Risk Labs**: Stress test your portfolio against historical market crashes (COVID-19, 2008 Crisis, etc.).
- **🧮 Financial Calculators**: Built-in SIP, Lumpsum, and Step-up SIP calculators with inflation adjustment.

## 🚀 Live Demo
[Deploy this app on Streamlit Cloud](https://mfanalysis-dashboard.streamlit.app/)

## 🛠️ Tech Stack
- **Frontend**: Streamlit
- **Data**: yfinance, mftool, nsepython (fallback)
- **Analytics**: Pandas, NumPy, SciPy
- **Visualization**: Plotly

## 💻 Local Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/welment-capital.git
cd welment-capital
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Run the app
```bash
streamlit run dashboard/app.py
```
