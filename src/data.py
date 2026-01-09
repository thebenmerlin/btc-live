"""
Data fetching and loading utilities for BTC Paper Trading Bot.
"""

import requests
import os


COINGECKO_API_URL = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"


def fetch_btc_price() -> float:
    """
    Fetch current BTC price from CoinGecko API (primary) or Binance (fallback).
    CoinGecko works globally while Binance is restricted in some regions.
    """
    # Try CoinGecko first (works globally, including US/Streamlit Cloud)
    try:
        response = requests.get(COINGECKO_API_URL, timeout=10)
        response.raise_for_status()
        data = response.json()
        if "bitcoin" in data and "usd" in data["bitcoin"]:
            return float(data["bitcoin"]["usd"])
    except Exception:
        pass  # Fall through to Binance
    
    # Fallback to Binance
    try:
        response = requests.get(BINANCE_API_URL, timeout=10)
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
