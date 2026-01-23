"""
BTC Paper Trading Bot - Professional Trading Dashboard
Clean, professional trading terminal with live charts and PnL tracking.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.model import TradingBot
from src.data import fetch_btc_price, load_historical_data

# Page config
st.set_page_config(
    page_title="BTC Trading Terminal",
    page_icon="B",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Professional Light Mode CSS
st.markdown("""
<style>
    .stApp {
        background-color: #FAFAFA;
    }
    
    .terminal-header {
        background: linear-gradient(90deg, #1E3A5F 0%, #2D5A87 100%);
        color: white;
        padding: 1rem 2rem;
        margin: -1rem -1rem 1rem -1rem;
        border-radius: 0 0 8px 8px;
    }
    
    .terminal-title {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 1.6rem;
        font-weight: 600;
        color: #FFFFFF;
        margin: 0;
    }
    
    .terminal-subtitle {
        font-size: 0.8rem;
        color: #B0C4DE;
        margin: 0;
    }
    
    .metric-container {
        background: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        padding: 0.8rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        text-align: center;
    }
    
    .metric-label {
        font-size: 0.7rem;
        color: #666666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 0.2rem;
    }
    
    .metric-value {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1A1A1A;
        font-family: 'Consolas', 'Monaco', monospace;
    }
    
    .metric-delta {
        font-size: 0.75rem;
        margin-top: 0.2rem;
    }
    
    .pnl-positive {
        color: #00A152;
        background: rgba(0, 161, 82, 0.1);
        padding: 2px 6px;
        border-radius: 3px;
    }
    
    .pnl-negative {
        color: #DC3545;
        background: rgba(220, 53, 69, 0.1);
        padding: 2px 6px;
        border-radius: 3px;
    }
    
    .signal-box {
        background: #FFFFFF;
        border: 2px solid #E0E0E0;
        border-radius: 6px;
        padding: 0.8rem;
        text-align: center;
    }
    
    .signal-buy {
        border-color: #00A152;
        background: rgba(0, 161, 82, 0.05);
    }
    
    .signal-sell {
        border-color: #DC3545;
        background: rgba(220, 53, 69, 0.05);
    }
    
    .signal-action {
        font-size: 1.3rem;
        font-weight: 700;
    }
    
    .buy-text { color: #00A152; }
    .sell-text { color: #DC3545; }
    
    .section-header {
        font-family: 'Segoe UI', Arial, sans-serif;
        font-size: 0.85rem;
        font-weight: 600;
        color: #1E3A5F;
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 2px solid #1E3A5F;
        padding-bottom: 0.4rem;
        margin-bottom: 0.8rem;
    }
    
    .status-bar {
        background: #1E3A5F;
        color: #FFFFFF;
        padding: 0.4rem 1rem;
        font-family: 'Consolas', 'Monaco', monospace;
        font-size: 0.7rem;
        display: flex;
        justify-content: space-between;
        border-radius: 4px;
        margin-top: 0.8rem;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    [data-testid="stMetricValue"] {
        font-family: 'Consolas', 'Monaco', monospace;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize Streamlit session state."""
    if 'bot' not in st.session_state:
        st.session_state.bot = TradingBot(initial_cash=10000.0)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(script_dir, "Bitcoin_Historical_Data.csv")
        historical = load_historical_data(csv_path)
        if historical:
            st.session_state.bot.pretrain(historical)
    
    if 'start_time' not in st.session_state:
        st.session_state.start_time = datetime.now()
    
    if 'error_count' not in st.session_state:
        st.session_state.error_count = 0


def create_price_chart(prices: list, predictions: list) -> go.Figure:
    """Create professional price chart."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.7, 0.3]
    )
    
    if prices:
        x_vals = list(range(len(prices)))
        
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=prices,
                mode='lines',
                name='BTC/USD',
                line=dict(color='#1E3A5F', width=2),
                fill='tozeroy',
                fillcolor='rgba(30, 58, 95, 0.1)'
            ),
            row=1, col=1
        )
        
        if len(prices) > 0:
            fig.add_trace(
                go.Scatter(
                    x=[len(prices) - 1],
                    y=[prices[-1]],
                    mode='markers+text',
                    name='Current',
                    marker=dict(size=8, color='#FF6B00'),
                    text=[f"${prices[-1]:,.2f}"],
                    textposition='top center',
                    textfont=dict(size=11, color='#FF6B00', family='Consolas, Monaco')
                ),
                row=1, col=1
            )
    
    if predictions:
        x_vals = list(range(len(predictions)))
        colors = ['#00A152' if p > 0 else '#DC3545' for p in predictions]
        
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=predictions,
                name='Predictions',
                marker_color=colors,
                opacity=0.8
            ),
            row=2, col=1
        )
    
    fig.update_layout(
        template='plotly_white',
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FAFAFA',
        margin=dict(l=60, r=40, t=40, b=40),
        height=420,
        showlegend=False,
        font=dict(family='Segoe UI, Arial', size=11, color='#1A1A1A')
    )
    
    fig.update_xaxes(showgrid=False, showline=True, linecolor='#E0E0E0', zeroline=False)
    fig.update_yaxes(
        showgrid=True, 
        gridcolor='#F0F0F0', 
        showline=True, 
        linecolor='#E0E0E0',
        row=1, col=1
    )
    fig.update_yaxes(
        showgrid=True, 
        gridcolor='#F0F0F0', 
        showline=True, 
        linecolor='#E0E0E0',
        row=2, col=1
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0, y=1.05,
        text="<b>BTC/USD LIVE</b>",
        showarrow=False,
        font=dict(size=11, color='#1E3A5F'),
        xanchor='left'
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0, y=0.32,
        text="<b>SIGNAL</b>",
        showarrow=False,
        font=dict(size=10, color='#1E3A5F'),
        xanchor='left'
    )
    
    return fig


def create_pnl_chart(trades: list) -> go.Figure:
    """Create PnL evolution chart."""
    fig = go.Figure()
    
    if not trades:
        fig.add_annotation(
            text="Waiting for trades...",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=12, color='#888888')
        )
        fig.update_layout(
            height=160,
            paper_bgcolor='#FFFFFF',
            plot_bgcolor='#FAFAFA',
            margin=dict(l=50, r=30, t=30, b=25)
        )
        return fig
    
    pnl_values = [t.pnl for t in trades]
    x_vals = list(range(len(pnl_values)))
    colors = ['#00A152' if p >= 0 else '#DC3545' for p in pnl_values]
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=pnl_values,
        mode='lines',
        fill='tozeroy',
        line=dict(color='#1E3A5F', width=2),
        fillcolor='rgba(30, 58, 95, 0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=pnl_values,
        mode='markers',
        marker=dict(size=5, color=colors)
    ))
    
    fig.update_layout(
        height=160,
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FAFAFA',
        margin=dict(l=50, r=30, t=30, b=25),
        showlegend=False
    )
    
    fig.update_xaxes(showgrid=False, showline=True, linecolor='#E0E0E0')
    fig.update_yaxes(
        showgrid=True, 
        gridcolor='#F0F0F0',
        showline=True,
        linecolor='#E0E0E0',
        zeroline=True,
        zerolinecolor='#1E3A5F',
        zerolinewidth=1
    )
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0, y=1.15,
        text="<b>P&L</b>",
        showarrow=False,
        font=dict(size=10, color='#1E3A5F'),
        xanchor='left'
    )
    
    return fig


def display_header():
    """Display professional header."""
    st.markdown("""
    <div class="terminal-header">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <div>
                <p class="terminal-title">BTC TRADING TERMINAL</p>
                <p class="terminal-subtitle">Volatility-Conditioned Regime Model | Live Paper Trading</p>
            </div>
            <div style="text-align: right;">
                <p style="margin: 0; font-size: 0.9rem;">BTCUSD</p>
                <p style="margin: 0; font-size: 0.7rem; color: #B0C4DE;">Binance Spot</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def display_metrics(bot: TradingBot, current_price: float = None):
    """Display trading metrics."""
    state = bot.state
    
    price = current_price or (state.prices[-1] if state.prices else 0)
    btc_value = state.btc_position * price
    
    pnl_class = "pnl-positive" if state.pnl >= 0 else "pnl-negative"
    pnl_sign = "+" if state.pnl >= 0 else ""
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">CASH</div>
            <div class="metric-value">${state.cash:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">BTC HOLDINGS</div>
            <div class="metric-value">{state.btc_position:.6f}</div>
            <div class="metric-delta" style="color: #666;">${btc_value:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">PORTFOLIO</div>
            <div class="metric-value">${state.portfolio_value:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">P&L</div>
            <div class="metric-value"><span class="{pnl_class}">{pnl_sign}${abs(state.pnl):,.2f}</span></div>
            <div class="metric-delta {pnl_class}">{pnl_sign}{abs(state.pnl_pct):.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">TRADES</div>
            <div class="metric-value">{state.trade_count}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-label">SAMPLES</div>
            <div class="metric-value">{state.learn_count}</div>
        </div>
        """, unsafe_allow_html=True)


def display_current_signal(result: dict):
    """Display current trading signal."""
    if not result:
        return
    
    # Extract data from the result dictionary
    alpha = result.get('alpha')
    regime = result.get('regime')
    target_weight = result.get('target_weight', 0)
    trade_info = result.get('trade_info')
    
    # Determine action based on target weight
    if target_weight > 0.1:
        action = "BUY"
    elif target_weight < -0.1:
        action = "SELL"
    else:
        action = "HOLD"
    
    # Determine signal strength based on target weight magnitude
    abs_weight = abs(target_weight)
    if abs_weight > 0.5:
        signal = "strong"
    elif abs_weight > 0.2:
        signal = "medium"
    else:
        signal = "weak"
    
    # Get prediction from alpha object
    prediction = alpha.alpha_blended if alpha else 0.0
    
    signal_class = "signal-buy" if action == "BUY" else ("signal-sell" if action == "SELL" else "signal-box")
    action_class = "buy-text" if action == "BUY" else ("sell-text" if action == "SELL" else "")
    
    # Get regime metrics
    vol_regime = regime.vol_ratio if regime else 0.5
    trend_regime = regime.trend_strength if regime else 0.0
    
    vol_label = "HIGH" if vol_regime > 0.7 else ("LOW" if vol_regime < 0.3 else "MED")
    trend_label = "UP" if trend_regime > 0.3 else ("DOWN" if trend_regime < -0.3 else "FLAT")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="signal-box {signal_class}">
            <div class="metric-label">SIGNAL</div>
            <div class="signal-action {action_class}">{action}</div>
            <div style="font-size: 0.75rem; color: #666;">{signal.upper()}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        pred_color = "#00A152" if prediction >= 0 else "#DC3545"
        st.markdown(f"""
        <div class="signal-box">
            <div class="metric-label">PREDICTED RETURN</div>
            <div style="font-size: 1.1rem; font-weight: 600; color: {pred_color}; font-family: Consolas, Monaco;">
                {prediction:+.6f}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="signal-box">
            <div class="metric-label">VOL REGIME</div>
            <div style="font-size: 1.1rem; font-weight: 600;">
                {vol_label} ({vol_regime:.0%})
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="signal-box">
            <div class="metric-label">TREND REGIME</div>
            <div style="font-size: 1.1rem; font-weight: 600;">
                {trend_label} ({trend_regime:+.2f})
            </div>
        </div>
        """, unsafe_allow_html=True)


def display_trade_log(trades: list):
    """Display recent trade history."""
    st.markdown('<div class="section-header">TRADE LOG</div>', unsafe_allow_html=True)
    
    if not trades:
        st.info("Waiting for trades...")
        return
    
    recent_trades = trades[-8:][::-1]
    
    trade_data = []
    for t in recent_trades:
        pnl_display = f"+${t.pnl:.2f}" if t.pnl >= 0 else f"-${abs(t.pnl):.2f}"
        action_display = f"[+] {t.action}" if t.action == "BUY" else f"[-] {t.action}"
        trade_data.append({
            "Time": t.timestamp.strftime("%H:%M:%S"),
            "Action": action_display,
            "Signal": t.signal_strength.upper(),
            "BTC": f"{t.btc_amount:.6f}",
            "Price": f"${t.price:,.2f}",
            "P&L": pnl_display
        })
    
    df = pd.DataFrame(trade_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def display_status_bar(bot: TradingBot):
    """Display status bar."""
    state = bot.state
    elapsed = datetime.now() - st.session_state.start_time
    elapsed_str = str(elapsed).split('.')[0]
    current_time = datetime.now().strftime("%H:%M:%S")
    
    st.markdown(f"""
    <div class="status-bar">
        <span>SESSION: {elapsed_str}</span>
        <span>TICKS: {len(state.prices)}</span>
        <span>LEARNED: {state.learn_count}</span>
        <span>{current_time}</span>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit app."""
    init_session_state()
    bot = st.session_state.bot
    
    display_header()
    
    # Warmup phase
    if not bot.is_ready:
        current, required = bot.warmup_progress
        
        st.warning(f"Initializing... Collecting market data: {current}/{required} points")
        st.progress(current / required)
        
        try:
            price = fetch_btc_price()
            bot.add_price(price)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Current Price", f"${price:,.2f}")
            with col2:
                st.metric("Data Points", f"{current}/{required}")
                
        except Exception as e:
            st.error(f"Connection error: {e}")
        
        time.sleep(1)
        st.rerun()
        return
    
    # Fetch new price and execute trade
    result = None
    current_price = None
    try:
        current_price = fetch_btc_price()
        result = bot.add_price(current_price)
        st.session_state.error_count = 0
    except Exception as e:
        st.session_state.error_count += 1
        if st.session_state.error_count >= 3:
            st.error(f"API Error: {e}")
    
    # Display metrics
    display_metrics(bot, current_price)
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Current signal
    if result:
        display_current_signal(result)
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Main content
    col1, col2 = st.columns([2.2, 1])
    
    with col1:
        st.markdown('<div class="section-header">MARKET DATA</div>', unsafe_allow_html=True)
        
        prices = bot.state.prices[-100:] if len(bot.state.prices) > 100 else bot.state.prices
        predictions = bot.state.alpha_history[-100:] if len(bot.state.alpha_history) > 100 else bot.state.alpha_history
        
        fig = create_price_chart(prices, predictions)
        st.plotly_chart(fig, use_container_width=True)
        
        pnl_fig = create_pnl_chart(bot.state.trades)
        st.plotly_chart(pnl_fig, use_container_width=True)
    
    with col2:
        display_trade_log(bot.state.trades)
    
    display_status_bar(bot)
    
    time.sleep(1)
    st.rerun()


if __name__ == "__main__":
    main()
