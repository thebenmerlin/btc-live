"""
Data fetching and loading utilities for BTC Paper Trading Bot.
"""

import requests
import os


# Request headers - required for cloud environments
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}


def fetch_btc_price() -> float:
    """
    Fetch current BTC price from multiple sources with fallbacks.
    Uses proper headers required by cloud environments like Streamlit Cloud.
    """
    
    # Source 1: Coinbase (most reliable, no restrictions)
    try:
        response = requests.get(
            "https://api.coinbase.com/v2/prices/BTC-USD/spot",
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        if "data" in data and "amount" in data["data"]:
            return float(data["data"]["amount"])
    except Exception:
        pass
    
    # Source 2: CoinGecko
    try:
        response = requests.get(
            "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        if "bitcoin" in data and "usd" in data["bitcoin"]:
            return float(data["bitcoin"]["usd"])
    except Exception:
        pass
    
    # Source 3: Kraken (also globally available)
    try:
        response = requests.get(
            "https://api.kraken.com/0/public/Ticker?pair=XBTUSD",
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        if "result" in data and "XXBTZUSD" in data["result"]:
            return float(data["result"]["XXBTZUSD"]["c"][0])
    except Exception:
        pass
    
    # Source 4: Binance (may be blocked in US)
    try:
        response = requests.get(
            "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT",
            headers=HEADERS,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        if "price" in data:
            return float(data["price"])
    except Exception:
        pass
    
    raise ConnectionError("Failed to fetch BTC price from all sources")


def load_historical_data(filepath: str) -> list:
    """
    Load and parse Bitcoin_Historical_Data.csv
    Returns list of prices (Close prices) in chronological order.
    """
    if not os.path.exists(filepath):
        return []
    
    prices = []
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                price_str = parts[1].replace('"', '').replace(',', '')
                try:
                    price = float(price_str)
                    prices.append(price)
                except ValueError:
                    continue
        
        # Reverse to get chronological order (CSV is newest first)
        prices = prices[::-1]
        return prices
    except Exception:
        return []
