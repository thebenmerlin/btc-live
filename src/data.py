"""
Data fetching and loading utilities for BTC Paper Trading Bot.
"""

import requests
import os


BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"


def fetch_btc_price() -> float:
    """Fetch current BTC price from Binance public API."""
    try:
        response = requests.get(BINANCE_API_URL, timeout=10)
        data = response.json()
        return float(data["price"])
    except Exception as e:
        raise ConnectionError(f"Failed to fetch BTC price: {e}")


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
