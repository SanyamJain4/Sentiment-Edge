"""
Nifty 50 constituent tickers, in yfinance format (NSE suffix .NS).

NOTE: Nifty 50 constituents change periodically (index rebalancing,
roughly twice a year). Verify this list against the official NSE
index factsheet before production use:
https://www.niftyindices.com/indices/equity/broad-based-indices/nifty-50
"""

NIFTY_50_TICKERS = [
    "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS",
    "AXISBANK.NS", "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS",
    "BEL.NS", "BHARTIARTL.NS", "CIPLA.NS", "COALINDIA.NS",
    "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS", "HCLTECH.NS",
    "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
    "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS",
    "ITC.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS",
    "M&M.NS", "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS",
    "ONGC.NS", "POWERGRID.NS", "RELIANCE.NS", "SBILIFE.NS",
    "SBIN.NS", "SHRIRAMFIN.NS", "SUNPHARMA.NS", "TATACONSUM.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "TCS.NS", "TECHM.NS",
    "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "UPL.NS",
    "WIPRO.NS", "MARICO.NS",
]

# Human-readable company names, used for news search queries.
# Keep keys in sync with NIFTY_50_TICKERS above.
TICKER_TO_COMPANY_NAME = {
    "ADANIENT.NS": "Adani Enterprises",
    "ADANIPORTS.NS": "Adani Ports and Special Economic Zone",
    "APOLLOHOSP.NS": "Apollo Hospitals Enterprise",
    "ASIANPAINT.NS": "Asian Paints",
    "AXISBANK.NS": "Axis Bank",
    "BAJAJ-AUTO.NS": "Bajaj Auto",
    "BAJFINANCE.NS": "Bajaj Finance",
    "BAJAJFINSV.NS": "Bajaj Finserv",
    "BEL.NS": "Bharat Electronics",
    "BHARTIARTL.NS": "Bharti Airtel",
    "CIPLA.NS": "Cipla",
    "COALINDIA.NS": "Coal India",
    "DRREDDY.NS": "Dr. Reddy's Laboratories",
    "EICHERMOT.NS": "Eicher Motors",
    "GRASIM.NS": "Grasim Industries",
    "HCLTECH.NS": "HCL Technologies",
    "HDFCBANK.NS": "HDFC Bank",
    "HDFCLIFE.NS": "HDFC Life Insurance",
    "HEROMOTOCO.NS": "Hero MotoCorp",
    "HINDALCO.NS": "Hindalco Industries",
    "HINDUNILVR.NS": "Hindustan Unilever",
    "ICICIBANK.NS": "ICICI Bank",
    "INDUSINDBK.NS": "IndusInd Bank",
    "INFY.NS": "Infosys",
    "ITC.NS": "ITC",
    "JSWSTEEL.NS": "JSW Steel",
    "KOTAKBANK.NS": "Kotak Mahindra Bank",
    "LT.NS": "Larsen & Toubro",
    "M&M.NS": "Mahindra & Mahindra",
    "MARUTI.NS": "Maruti Suzuki India",
    "NESTLEIND.NS": "Nestle India",
    "NTPC.NS": "NTPC",
    "ONGC.NS": "Oil and Natural Gas Corporation",
    "POWERGRID.NS": "Power Grid Corporation of India",
    "RELIANCE.NS": "Reliance Industries",
    "SBILIFE.NS": "SBI Life Insurance",
    "SBIN.NS": "State Bank of India",
    "SHRIRAMFIN.NS": "Shriram Finance",
    "SUNPHARMA.NS": "Sun Pharmaceutical Industries",
    "TATACONSUM.NS": "Tata Consumer Products",
    "TATAMOTORS.NS": "Tata Motors",
    "TATASTEEL.NS": "Tata Steel",
    "TCS.NS": "Tata Consultancy Services",
    "TECHM.NS": "Tech Mahindra",
    "TITAN.NS": "Titan Company",
    "TRENT.NS": "Trent",
    "ULTRACEMCO.NS": "UltraTech Cement",
    "UPL.NS": "UPL",
    "WIPRO.NS": "Wipro",
    "MARICO.NS": "Marico",
}
