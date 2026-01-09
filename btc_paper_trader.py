"""
Enhanced Live Bitcoin Paper-Trading Bot
Volatility-Conditioned Trend/Regime Model

Features:
- SGDRegressor with L2 regularization for online learning
- Pre-training on historical CSV data
- 60 data points warmup
- Volatility-Conditioned Regime Detection:
  * High volatility regime → Trend-following signals
  * Low volatility regime → Mean-reversion signals
- Regime-aware feature engineering for stronger signals
"""

import requests
import time
import numpy as np
import os
from sklearn.linear_model import SGDRegressor

# ============================================================
# CONFIGURATION
# ============================================================
BINANCE_API_URL = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
POLL_INTERVAL = 1              # seconds between price fetches
WARMUP_DATA_POINTS = 60        # Number of data points to collect before trading
LOOKBACK_SHORT = 10            # Short-term lookback
LOOKBACK_MED = 20              # Medium-term lookback
LOOKBACK_LONG = 50             # Long-term lookback for regime detection
INITIAL_CASH = 10000.0         # starting paper trading balance in USD
TRADE_FRACTION = 0.30          # base fraction of available balance to trade (AGGRESSIVE)

# NO HOLD MODE - Always trade for maximum action
# Signal magnitude only affects position SIZE, not whether to trade
BASE_SIGNAL_MULTIPLIER = 1.0   # base position multiplier
STRONG_SIGNAL_MULTIPLIER = 2.0 # 2x on strong signals
WEAK_SIGNAL_MULTIPLIER = 0.5   # 0.5x on weak signals (but still trade!)

# Signal strength thresholds (for sizing, NOT for hold)
WEAK_THRESHOLD = 0.00005       # below this = weak (small trade)
STRONG_THRESHOLD = 0.0003      # above this = strong (big trade)

# Volatility regime thresholds
VOL_REGIME_LOOKBACK = 30       # lookback for volatility regime calculation
VOL_EXPANSION_PERCENTILE = 70  # percentile above which = high volatility regime

# Historical data file
HISTORICAL_CSV = "Bitcoin_Historical_Data.csv"

# ============================================================
# DATA LOADING - HISTORICAL CSV
# ============================================================
def load_historical_data(filepath):
    """Load and parse Bitcoin_Historical_Data.csv"""
    if not os.path.exists(filepath):
        print(f"[WARN] Historical data file not found: {filepath}")
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
        
        prices = prices[::-1]  # Chronological order
        print(f"[INFO] Loaded {len(prices)} historical prices from {filepath}")
        return prices
    except Exception as e:
        print(f"[WARN] Error loading historical data: {e}")
        return []

# ============================================================
# DATA FETCHING
# ============================================================
def fetch_btc_price():
    """Fetch current BTC price from Binance public API."""
    response = requests.get(BINANCE_API_URL, timeout=10)
    data = response.json()
    return float(data["price"])

# ============================================================
# REGIME DETECTION
# ============================================================
def compute_returns(prices, n=1):
    """Compute n-period returns."""
    if len(prices) < n + 1:
        return np.array([0.0])
    return np.diff(prices[-n-1:]) / np.array(prices[-n-1:-1])

def compute_volatility(prices, period):
    """Compute rolling volatility (std of returns)."""
    if len(prices) < period + 1:
        return 0.0
    returns = compute_returns(prices, period)
    return np.std(returns) if len(returns) > 0 else 0.0

def detect_volatility_regime(prices, current_vol, lookback=30):
    """
    Detect volatility regime: HIGH (expansion) or LOW (contraction).
    Returns regime score between 0 (low vol) and 1 (high vol).
    """
    if len(prices) < lookback + 10:
        return 0.5  # Neutral if insufficient data
    
    # Calculate historical volatility distribution
    vol_history = []
    for i in range(lookback, len(prices)):
        window = prices[i-lookback:i]
        returns = np.diff(window) / window[:-1]
        vol_history.append(np.std(returns))
    
    if not vol_history:
        return 0.5
    
    # Compute percentile of current volatility
    percentile = np.sum(np.array(vol_history) < current_vol) / len(vol_history)
    return percentile

def detect_trend_regime(prices, short_period=10, long_period=50):
    """
    Detect trend regime using dual moving average crossover.
    Returns: trend score between -1 (strong downtrend) and 1 (strong uptrend).
    """
    if len(prices) < long_period:
        return 0.0
    
    sma_short = np.mean(prices[-short_period:])
    sma_long = np.mean(prices[-long_period:])
    
    # Normalized trend strength
    trend_strength = (sma_short - sma_long) / sma_long
    
    # Scale to approximately -1 to 1 range
    return np.clip(trend_strength * 100, -1, 1)

# ============================================================
# VOLATILITY-CONDITIONED FEATURES
# ============================================================
def compute_ema(prices, period):
    """Compute Exponential Moving Average."""
    if len(prices) < period:
        return prices[-1] if prices else 0
    
    multiplier = 2 / (period + 1)
    ema = prices[-period]
    for price in prices[-period + 1:]:
        ema = (price - ema) * multiplier + ema
    return ema

def compute_atr(prices, period=14):
    """Compute Average True Range (simplified for single price series)."""
    if len(prices) < period + 1:
        return 0.0
    
    # Use high-low proxy: daily range approximated by price diff
    ranges = np.abs(np.diff(prices[-period-1:]))
    return np.mean(ranges)

def compute_features(prices):
    """
    Compute Volatility-Conditioned Regime features.
    
    Feature Set (12 features):
    1-2: Regime indicators (volatility regime, trend regime)
    3-4: Trend features (short-term, long-term trend strength)
    5-6: Volatility features (current vol, vol change)
    7-8: Regime-conditioned trend (trend × vol regime)
    9-10: Mean reversion features (z-score short, z-score long)
    11-12: Regime-conditioned reversion (z-score × inverse vol regime)
    """
    n = len(prices)
    
    # --- Volatility Regime ---
    current_vol = compute_volatility(prices, LOOKBACK_MED)
    vol_regime = detect_volatility_regime(prices, current_vol, VOL_REGIME_LOOKBACK)
    inv_vol_regime = 1 - vol_regime  # For mean reversion conditioning
    
    # --- Trend Regime ---
    trend_regime = detect_trend_regime(prices, LOOKBACK_SHORT, LOOKBACK_LONG)
    
    # --- Trend Features ---
    # Short-term trend: 10-period price change normalized
    short_trend = (prices[-1] - prices[-LOOKBACK_SHORT]) / prices[-LOOKBACK_SHORT] if n >= LOOKBACK_SHORT else 0
    
    # Long-term trend: EMA crossover signal
    ema_short = compute_ema(prices, LOOKBACK_SHORT)
    ema_long = compute_ema(prices, LOOKBACK_LONG) if n >= LOOKBACK_LONG else ema_short
    long_trend = (ema_short - ema_long) / ema_long if ema_long != 0 else 0
    
    # --- Volatility Features ---
    vol_short = compute_volatility(prices, LOOKBACK_SHORT)
    vol_long = compute_volatility(prices, LOOKBACK_LONG) if n >= LOOKBACK_LONG else vol_short
    vol_change = (vol_short - vol_long) / vol_long if vol_long > 0 else 0  # Vol expanding or contracting
    
    # Normalize volatility
    vol_normalized = current_vol / (vol_long + 1e-10) - 1 if n >= LOOKBACK_LONG else 0
    
    # --- Mean Reversion Features ---
    # Short-term z-score
    mean_short = np.mean(prices[-LOOKBACK_SHORT:])
    std_short = np.std(prices[-LOOKBACK_SHORT:])
    zscore_short = (prices[-1] - mean_short) / std_short if std_short > 0 else 0
    
    # Long-term z-score
    if n >= LOOKBACK_LONG:
        mean_long = np.mean(prices[-LOOKBACK_LONG:])
        std_long = np.std(prices[-LOOKBACK_LONG:])
        zscore_long = (prices[-1] - mean_long) / std_long if std_long > 0 else 0
    else:
        zscore_long = zscore_short
    
    # --- Regime-Conditioned Features ---
    # In high vol regime: emphasize trend-following
    trend_conditioned = short_trend * vol_regime + long_trend * vol_regime
    
    # In low vol regime: emphasize mean reversion
    reversion_conditioned = zscore_short * inv_vol_regime + zscore_long * inv_vol_regime
    
    # --- Trend Acceleration ---
    # Second derivative: is the trend accelerating or decelerating?
    if n >= LOOKBACK_SHORT + 5:
        trend_now = (prices[-1] - prices[-5]) / prices[-5]
        trend_prev = (prices[-5] - prices[-10]) / prices[-10]
        trend_accel = trend_now - trend_prev
    else:
        trend_accel = 0
    
    # --- Price Position Relative to ATR ---
    atr = compute_atr(prices, LOOKBACK_MED)
    if n >= LOOKBACK_MED and atr > 0:
        price_atr_ratio = (prices[-1] - prices[-2]) / atr  # Normalized move
    else:
        price_atr_ratio = 0
    
    return np.array([[
        vol_regime,              # 1. Volatility regime (0-1)
        trend_regime,            # 2. Trend regime (-1 to 1)
        short_trend,             # 3. Short-term trend
        long_trend,              # 4. Long-term trend (EMA crossover)
        vol_normalized,          # 5. Normalized current volatility
        vol_change,              # 6. Volatility change
        trend_conditioned,       # 7. Trend × volatility regime
        reversion_conditioned,   # 8. Z-score × inverse vol regime
        zscore_short,            # 9. Short-term z-score
        zscore_long,             # 10. Long-term z-score
        trend_accel,             # 11. Trend acceleration
        price_atr_ratio          # 12. Price move / ATR
    ]])

def compute_target(prices):
    """Compute target: the actual return that just occurred."""
    return (prices[-1] - prices[-2]) / prices[-2]

# ============================================================
# MODEL PRE-TRAINING
# ============================================================
def pretrain_on_historical(model, historical_prices):
    """Pre-train the model on historical daily data."""
    min_required = LOOKBACK_LONG + 10
    if len(historical_prices) < min_required:
        print(f"[INFO] Not enough historical data for pre-training (need {min_required})")
        return False
    
    print(f"[INFO] Pre-training model on {len(historical_prices)} historical data points...")
    
    train_count = 0
    for i in range(min_required, len(historical_prices)):
        prices_window = historical_prices[:i + 1]
        
        prev_features = compute_features(prices_window[:-1])
        target = compute_target(prices_window)
        
        model.partial_fit(prev_features, [target])
        train_count += 1
    
    print(f"[INFO] Pre-trained on {train_count} samples from historical data")
    return True

# ============================================================
# TRADING LOGIC
# ============================================================
def decide_action(predicted_return, vol_regime, trend_regime):
    """
    AGGRESSIVE MODE: Always trade, no HOLD.
    Signal strength determines position SIZE, not whether to trade.
    Returns: (action, signal_strength, confidence_note)
    """
    magnitude = abs(predicted_return)
    
    # Determine signal strength for position sizing
    if magnitude >= STRONG_THRESHOLD:
        signal_strength = "strong"  # Big position
    elif magnitude >= WEAK_THRESHOLD:
        signal_strength = "normal"  # Standard position
    else:
        signal_strength = "weak"    # Small position (but still trade!)
    
    # Check regime alignment for confidence note
    regime_aligned = (predicted_return > 0 and trend_regime > 0.2) or \
                     (predicted_return < 0 and trend_regime < -0.2) or \
                     abs(trend_regime) <= 0.2
    
    # ALWAYS TRADE - no hold
    if predicted_return >= 0:
        return "BUY", signal_strength, f"pred={predicted_return:+.6f}"
    else:
        return "SELL", signal_strength, f"pred={predicted_return:+.6f}"

def execute_trade(action, signal_strength, price, cash, btc_position):
    """
    AGGRESSIVE MODE: Execute trades with signal-based sizing.
    Always executes if resources available.
    """
    trade_btc = 0.0
    executed = False
    
    # Scale position by signal strength
    if signal_strength == "strong":
        fraction = TRADE_FRACTION * STRONG_SIGNAL_MULTIPLIER  # Big trade
    elif signal_strength == "weak":
        fraction = TRADE_FRACTION * WEAK_SIGNAL_MULTIPLIER    # Small trade
    else:
        fraction = TRADE_FRACTION * BASE_SIGNAL_MULTIPLIER    # Normal trade
    
    fraction = min(fraction, 0.6)  # Cap at 60%
    
    if action == "BUY" and cash > 10.0:  # Min $10 to trade
        spend_amount = cash * fraction
        trade_btc = spend_amount / price
        cash -= spend_amount
        btc_position += trade_btc
        executed = True
        
    elif action == "SELL" and btc_position > 0.0001:  # Min 0.0001 BTC
        trade_btc = btc_position * fraction
        cash += trade_btc * price
        btc_position -= trade_btc
        executed = True
    
    return cash, btc_position, trade_btc, executed

# ============================================================
# DISPLAY
# ============================================================
def get_regime_emoji(vol_regime, trend_regime):
    """Get visual indicators for current regime."""
    vol_icon = "🔥" if vol_regime > 0.7 else ("❄️" if vol_regime < 0.3 else "🌡️")
    
    if trend_regime > 0.3:
        trend_icon = "📈"
    elif trend_regime < -0.3:
        trend_icon = "📉"
    else:
        trend_icon = "↔️"
    
    return vol_icon, trend_icon

def print_status(price, prediction, action, signal_strength, cash, btc_position, 
                 initial_value, trade_btc, executed, features=None, confidence_note="", learn_count=0):
    """Print current trading status to terminal."""
    portfolio_value = cash + (btc_position * price)
    pnl = portfolio_value - initial_value
    pnl_pct = (pnl / initial_value) * 100
    
    # Color codes
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    action_color = GREEN if action == "BUY" else (RED if action == "SELL" else YELLOW)
    pnl_color = GREEN if pnl >= 0 else RED
    signal_indicator = "⚡" if signal_strength == "strong" else ("•" if signal_strength == "normal" else "○")
    
    # Regime indicators
    vol_regime = features[0][0] if features is not None else 0.5
    trend_regime = features[0][1] if features is not None else 0
    vol_icon, trend_icon = get_regime_emoji(vol_regime, trend_regime)
    
    print("\n" + "=" * 70)
    print(f"BTC Price:          ${price:,.2f}")
    print(f"Predicted Return:   {prediction:+.8f} {signal_indicator} [{signal_strength.upper()}]")
    
    if executed:
        trade_info = f"({trade_btc:.6f} BTC = ${trade_btc * price:,.2f})"
        print(f"Action:             {action_color}{BOLD}{action}{RESET} {trade_info}")
    else:
        status = "(no position)" if action == "SELL" else "(no cash)" if action == "BUY" else f"({confidence_note})"
        print(f"Action:             {action_color}{action}{RESET} {DIM}{status}{RESET}")
    
    print("-" * 70)
    print(f"Cash Balance:       ${cash:,.2f}")
    print(f"BTC Position:       {btc_position:.6f} BTC (${btc_position * price:,.2f})")
    print(f"Portfolio Value:    ${portfolio_value:,.2f}")
    print(f"PnL:                {pnl_color}${pnl:+,.2f} ({pnl_pct:+.2f}%){RESET}")
    
    # Show regime info and learning stats
    if features is not None:
        trend_cond = features[0][6]
        reversion_cond = features[0][7]
        print("-" * 70)
        regime_str = f"{vol_icon} Vol: {vol_regime:.0%} | {trend_icon} Trend: {trend_regime:+.2f}"
        learn_str = f"📚 Learned: {learn_count} samples"
        print(f"{DIM}{regime_str} | {learn_str}{RESET}")
    
    print("=" * 70)

# ============================================================
# MAIN LOOP
# ============================================================
def main():
    """Main trading loop."""
    # Initialize online learning model with L2 regularization
    # Higher learning rate for faster adaptation to new data
    model = SGDRegressor(
        penalty='l2',
        alpha=0.0005,              # Lower regularization for faster learning
        random_state=42,
        warm_start=True,
        learning_rate='invscaling',  # Learning rate decreases over time
        eta0=0.2,                    # Higher initial learning rate
        power_t=0.1,                 # Slower decay of learning rate
        tol=1e-5
    )
    model_trained = False
    learn_count = 0  # Track how many times we've learned
    
    print("=" * 70)
    print("  BTC PAPER TRADING BOT - CONTINUOUS LEARNING MODE")
    print("=" * 70)
    
    # Load historical data and pre-train
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, HISTORICAL_CSV)
    historical_prices = load_historical_data(csv_path)
    
    if historical_prices:
        model_trained = pretrain_on_historical(model, historical_prices)
        learn_count = len(historical_prices) - 60 if len(historical_prices) > 60 else 0
    
    # Initialize live price history
    prices = []
    
    # Portfolio state
    cash = INITIAL_CASH
    btc_position = 0.0
    trade_count = 0
    
    # Minimum data points needed
    min_data = max(LOOKBACK_LONG + 10, WARMUP_DATA_POINTS)
    
    print(f"\nInitial Cash: ${INITIAL_CASH:,.2f}")
    print(f"Poll Interval: {POLL_INTERVAL}s | Warmup: {WARMUP_DATA_POINTS} data points")
    print(f"Mode: AGGRESSIVE (NO HOLD) | Trade Size: {TRADE_FRACTION:.0%} base")
    print(f"⚡ Strong: >{STRONG_THRESHOLD:.5%} (2x) | • Normal | ○ Weak: <{WEAK_THRESHOLD:.5%} (0.5x)")
    print(f"\n🔥=High Vol  ❄️=Low Vol  📈=Uptrend  📉=Downtrend  ↔️=Ranging")
    print(f"📚 Model learns from EVERY new price tick (continuous online learning)")
    print(f"\n[WARMUP] Collecting {WARMUP_DATA_POINTS} data points...\n")
    
    while True:
        try:
            current_price = fetch_btc_price()
            prices.append(current_price)
            
            # Warmup phase: collect required data points
            if len(prices) < min_data:
                remaining = min_data - len(prices)
                print(f"[WARMUP] ${current_price:,.2f} | Data points: {len(prices)}/{min_data} ({remaining} left)")
                time.sleep(POLL_INTERVAL)
                continue
            
            if len(prices) == min_data:
                print(f"\n[READY] Warmup complete! Starting trading with {len(prices)} data points.\n")
            
            # Compute features for current state
            features = compute_features(prices)
            
            # ========================================
            # CONTINUOUS ONLINE LEARNING
            # Learn from EVERY new price tick
            # ========================================
            if len(prices) > min_data:
                # Calculate the actual return that occurred
                actual_return = compute_target(prices)
                
                # Get the features from the PREVIOUS state (before the price moved)
                prev_features = compute_features(prices[:-1])
                
                # LEARN: Update model with (previous_features -> actual_return)
                model.partial_fit(prev_features, [actual_return])
                learn_count += 1
                model_trained = True
            
            # Prediction based on CURRENT features
            if model_trained:
                predicted_return = model.predict(features)[0]
            else:
                # First time: bootstrap with zero
                predicted_return = 0.0
                model.partial_fit(features, [0.0])
                model_trained = True
                learn_count += 1
            
            # Extract regime info for decision
            vol_regime = features[0][0]
            trend_regime = features[0][1]
            
            # Decide and execute
            action, signal_strength, confidence_note = decide_action(
                predicted_return, vol_regime, trend_regime
            )
            cash, btc_position, trade_btc, executed = execute_trade(
                action, signal_strength, current_price, cash, btc_position
            )
            
            if executed:
                trade_count += 1
            
            # Display with learning stats
            print_status(
                current_price, predicted_return, action, signal_strength,
                cash, btc_position, INITIAL_CASH, trade_btc, executed, 
                features, confidence_note, learn_count
            )
            
            time.sleep(POLL_INTERVAL)
            
        except requests.RequestException as e:
            print(f"API Error: {e}. Retrying...")
            time.sleep(POLL_INTERVAL)
        except KeyboardInterrupt:
            print("\n\n" + "=" * 70)
            print("  BOT STOPPED")
            print("=" * 70)
            if prices:
                portfolio_value = cash + (btc_position * prices[-1])
                pnl = portfolio_value - INITIAL_CASH
                print(f"Total Trades:         {trade_count}")
                print(f"Final Cash:           ${cash:,.2f}")
                print(f"Final BTC:            {btc_position:.6f} BTC")
                print(f"Final Portfolio:      ${portfolio_value:,.2f}")
                print(f"Total PnL:            ${pnl:+,.2f} ({(pnl/INITIAL_CASH)*100:+.2f}%)")
            print("=" * 70)
            break

if __name__ == "__main__":
    main()
