"""
Enhanced Live Bitcoin Paper-Trading Bot
Phase 1: Hedge-Fund-Style Alpha Model

Key Features:
- Multi-horizon prediction (10s, 1m, 5m)
- Regime awareness via learned features
- Volatility-normalized signals
- Continuous position sizing (alpha / volatility)
- Real-time alpha and regime metrics display

Run: python btc_paper_trader.py
"""

import time
import numpy as np
import os
from src.data import fetch_btc_price, load_historical_data
from src.model import TradingBot, MIN_DATA_POINTS, ALPHA_WEIGHTS

# =============================================================================
# CONFIGURATION
# =============================================================================
POLL_INTERVAL = 1              # seconds between price fetches
HISTORICAL_CSV = "Bitcoin_Historical_Data.csv"

# Display settings
DISPLAY_WIDTH = 74


# =============================================================================
# DISPLAY FUNCTIONS
# =============================================================================

def get_alpha_arrow(alpha: float) -> str:
    """Get directional indicator for alpha."""
    if alpha > 0.001:
        return "▲▲"
    elif alpha > 0:
        return "▲"
    elif alpha < -0.001:
        return "▼▼"
    elif alpha < 0:
        return "▼"
    return "—"


def get_regime_indicators(vol_ratio: float, trend_strength: float) -> tuple:
    """Get visual indicators for regime state."""
    # Volatility regime
    if vol_ratio > 1.5:
        vol_icon = "🔥"  # High vol expansion
    elif vol_ratio > 1.1:
        vol_icon = "📈"  # Moderate expansion
    elif vol_ratio < 0.7:
        vol_icon = "❄️"   # Vol contraction
    else:
        vol_icon = "🌡️"   # Normal vol
    
    # Trend regime
    if trend_strength > 2.0:
        trend_icon = "🚀"  # Strong trend
    elif trend_strength > 1.0:
        trend_icon = "📊"  # Moderate trend
    else:
        trend_icon = "〰️"   # Ranging/noisy
    
    return vol_icon, trend_icon


def format_weight(weight: float) -> str:
    """Format position weight with color."""
    if weight > 0.1:
        return f"+{weight:.2f}"
    elif weight < -0.1:
        return f"{weight:.2f}"
    else:
        return f"{weight:+.2f}"


def print_status(result: dict, state, learn_count: int):
    """
    Print comprehensive trading status with alpha model metrics.
    
    Display sections:
    1. Price and blended alpha
    2. Multi-horizon alpha breakdown
    3. Regime metrics (learned features)
    4. Position sizing
    5. Portfolio stats
    """
    # ANSI color codes
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    price = result['price']
    alpha = result['alpha']
    regime = result['regime']
    target_weight = result['target_weight']
    current_weight = result['current_weight']
    pnl = result['pnl']
    pnl_pct = result['pnl_pct']
    drawdown = result['drawdown']
    portfolio_value = result['portfolio_value']
    
    # Determine colors
    alpha_color = GREEN if alpha.alpha_blended > 0 else RED if alpha.alpha_blended < 0 else YELLOW
    pnl_color = GREEN if pnl >= 0 else RED
    arrow = get_alpha_arrow(alpha.alpha_blended)
    vol_icon, trend_icon = get_regime_indicators(regime.vol_ratio, regime.trend_strength)
    
    # Build output
    print()
    print("═" * DISPLAY_WIDTH)
    
    # Price line
    print(f"  BTC Price:         ${price:,.2f}")
    
    print("─" * DISPLAY_WIDTH)
    
    # Alpha section
    print(f"  {BOLD}ALPHA (Blended):{RESET}    {alpha_color}{alpha.alpha_blended:+.5f} {arrow}{RESET}")
    
    # Multi-horizon breakdown
    print(f"  {DIM}├─ 10s ({int(ALPHA_WEIGHTS['short']*100)}%):{RESET}       {alpha.alpha_short:+.5f}")
    print(f"  {DIM}├─ 1m  ({int(ALPHA_WEIGHTS['medium']*100)}%):{RESET}       {alpha.alpha_medium:+.5f}")
    print(f"  {DIM}└─ 5m  ({int(ALPHA_WEIGHTS['long']*100)}%):{RESET}       {alpha.alpha_long:+.5f}")
    
    print("─" * DISPLAY_WIDTH)
    
    # Regime metrics
    print(f"  {BOLD}REGIME METRICS{RESET}  {vol_icon} {trend_icon}")
    print(f"  Vol Short:         {regime.vol_short:.6f}   Vol Long:    {regime.vol_long:.6f}")
    print(f"  Vol Ratio:         {regime.vol_ratio:.2f}         Trend Str:   {regime.trend_strength:.2f}")
    
    print("─" * DISPLAY_WIDTH)
    
    # Position section
    print(f"  {BOLD}POSITION{RESET}")
    
    # Color the weights
    target_str = f"{target_weight:+.2f}"
    current_str = f"{current_weight:+.2f}"
    
    if target_weight > 0.1:
        target_color = GREEN
    elif target_weight < -0.1:
        target_color = RED
    else:
        target_color = YELLOW
        
    print(f"  Target Weight:     {target_color}{target_str}{RESET}      Current:     {current_str}")
    
    # Trade info
    if result['trade_executed'] and result['trade_info']:
        trade = result['trade_info']
        action_color = GREEN if trade.action == "BUY" else RED
        print(f"  {BOLD}Trade Executed:{RESET}    {action_color}{trade.action}{RESET} {trade.btc_amount:.6f} BTC (${trade.usd_amount:,.2f})")
    else:
        delta = target_weight - current_weight
        if abs(delta) < 0.01:
            print(f"  {DIM}Trade Required:     None (on target){RESET}")
        else:
            print(f"  {DIM}Trade Required:     Pending rebalance (Δ{delta:+.2f}){RESET}")
    
    print("─" * DISPLAY_WIDTH)
    
    # Portfolio section
    print(f"  {BOLD}PORTFOLIO{RESET}")
    print(f"  Value:             ${portfolio_value:,.2f}")
    print(f"  PnL:               {pnl_color}${pnl:+,.2f} ({pnl_pct:+.2f}%){RESET}")
    print(f"  Max Drawdown:      {RED if drawdown > 0.02 else DIM}-{drawdown*100:.2f}%{RESET}")
    
    print("─" * DISPLAY_WIDTH)
    
    # Learning stats
    print(f"  {DIM}📚 Model Samples: {learn_count:,} | Trades: {state.trade_count}{RESET}")
    
    print("═" * DISPLAY_WIDTH)


def print_warmup(current: int, required: int, price: float):
    """Print warmup progress."""
    progress = current / required
    bar_width = 30
    filled = int(bar_width * progress)
    bar = "█" * filled + "░" * (bar_width - filled)
    print(f"\r[WARMUP] ${price:,.2f} | [{bar}] {current}/{required} ({progress*100:.0f}%)", end="", flush=True)


def print_header():
    """Print startup header."""
    print()
    print("═" * DISPLAY_WIDTH)
    print("  BTC PAPER TRADING BOT - ALPHA MODEL v1.0")
    print("  Phase 1: Hedge-Fund-Style Signal Construction")
    print("═" * DISPLAY_WIDTH)
    print()
    print("  Multi-Horizon Prediction:")
    print(f"    • 10-second (weight: {int(ALPHA_WEIGHTS['short']*100)}%)")
    print(f"    • 1-minute  (weight: {int(ALPHA_WEIGHTS['medium']*100)}%)")
    print(f"    • 5-minute  (weight: {int(ALPHA_WEIGHTS['long']*100)}%)")
    print()
    print("  Regime Features (learned, not hardcoded):")
    print("    • Volatility ratio (short/long)")
    print("    • Trend strength (signal-to-noise)")
    print()
    print("  Position Sizing:")
    print("    • Continuous: position ∝ alpha / volatility")
    print("    • Risk-adjusted: scales down in high vol")
    print()
    print("─" * DISPLAY_WIDTH)


def print_summary(state, prices: list):
    """Print final trading summary."""
    print("\n")
    print("═" * DISPLAY_WIDTH)
    print("  BOT STOPPED - FINAL SUMMARY")
    print("═" * DISPLAY_WIDTH)
    
    if prices:
        portfolio_value = state.cash + (state.btc_position * prices[-1])
        pnl = portfolio_value - state.initial_cash
        pnl_pct = (pnl / state.initial_cash) * 100
        
        # Color codes
        GREEN = "\033[92m"
        RED = "\033[91m"
        RESET = "\033[0m"
        pnl_color = GREEN if pnl >= 0 else RED
        
        print(f"  Total Trades:      {state.trade_count}")
        print(f"  Final Cash:        ${state.cash:,.2f}")
        print(f"  Final BTC:         {state.btc_position:.6f} BTC")
        print(f"  Final Portfolio:   ${portfolio_value:,.2f}")
        print(f"  Total PnL:         {pnl_color}${pnl:+,.2f} ({pnl_pct:+.2f}%){RESET}")
        print(f"  Max Drawdown:      -{state.drawdown*100:.2f}%")
        
        # Alpha stats
        if state.alpha_history:
            alphas = np.array(state.alpha_history)
            print()
            print(f"  Alpha Statistics:")
            print(f"    Mean:            {np.mean(alphas):+.5f}")
            print(f"    Std:             {np.std(alphas):.5f}")
            print(f"    Max:             {np.max(alphas):+.5f}")
            print(f"    Min:             {np.min(alphas):+.5f}")
    
    print("═" * DISPLAY_WIDTH)


# =============================================================================
# MAIN LOOP
# =============================================================================

def main():
    """Main trading loop."""
    
    # Print startup header
    print_header()
    
    # Initialize trading bot
    bot = TradingBot(initial_cash=10000.0)
    
    # Load and pretrain on historical data
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, HISTORICAL_CSV)
    historical_prices = load_historical_data(csv_path)
    
    if historical_prices:
        print(f"  Loading {len(historical_prices)} historical prices...")
        if bot.pretrain(historical_prices):
            print(f"  ✓ Pre-trained on historical data")
            print(f"  ✓ Model samples: {bot.alpha_model.total_learn_count:,}")
        else:
            print(f"  ⚠ Insufficient historical data for pre-training")
    else:
        print("  ⚠ No historical data found")
    
    print()
    print(f"  Starting live trading (poll interval: {POLL_INTERVAL}s)")
    print(f"  Warming up: need {MIN_DATA_POINTS} data points")
    print()
    
    warmup_complete = False
    
    while True:
        try:
            # Fetch current price
            current_price = fetch_btc_price()
            
            # Process price tick
            result = bot.add_price(current_price)
            
            # Handle warmup phase
            if result is None:
                current, required = bot.warmup_progress
                print_warmup(current, required, current_price)
                time.sleep(POLL_INTERVAL)
                continue
            
            # First time after warmup
            if not warmup_complete:
                print()  # Clear warmup line
                print(f"\n  ✓ Warmup complete! Starting trading.\n")
                warmup_complete = True
            
            # Display trading status
            print_status(result, bot.state, bot.alpha_model.total_learn_count)
            
            time.sleep(POLL_INTERVAL)
            
        except KeyboardInterrupt:
            print_summary(bot.state, bot.state.prices)
            break
            
        except Exception as e:
            print(f"\n  ⚠ Error: {e}")
            print(f"  Retrying in {POLL_INTERVAL}s...")
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
