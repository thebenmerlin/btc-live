# ₿ BTC Paper Trading Bot

A live Bitcoin paper trading bot with **Volatility-Conditioned Regime Model** and **Continuous Online Learning**. Features a beautiful Streamlit dashboard with real-time price charts, prediction graphs, and trading metrics.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

- **🤖 ML-Powered Trading**: SGDRegressor with L2 regularization for online learning
- **📊 Live Dashboard**: Real-time Streamlit interface with Plotly charts
- **🔄 Continuous Learning**: Model improves with every new price tick
- **📈 Regime Detection**: Volatility-conditioned trend/regime model
- **💹 Paper Trading**: Risk-free simulation with $10,000 starting balance
- **⚡ Aggressive Mode**: No-hold trading for maximum action

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/btc-live-trading.git
   cd btc-live-trading
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

5. **Open in browser**: Navigate to `http://localhost:8501`

## 📸 Dashboard Preview

The dashboard displays:
- **Trading Metrics**: Cash, BTC holdings, portfolio value, PnL
- **Live Price Chart**: Real-time BTC price with prediction overlay
- **Signal Indicators**: Current action, volatility regime, trend regime
- **Trade Log**: Recent trade history with timestamps

## 🧠 Model Architecture

### Volatility-Conditioned Regime Model

The bot uses a **regime-aware** approach to trading:

| Regime | Volatility | Strategy |
|--------|------------|----------|
| 🔥 High Vol | >70% | Trend-following signals |
| 🌡️ Medium Vol | 30-70% | Balanced approach |
| ❄️ Low Vol | <30% | Mean-reversion signals |

### Features (12 total)
1. Volatility regime score
2. Trend regime score
3. Short-term trend (10-period)
4. Long-term trend (EMA crossover)
5. Normalized volatility
6. Volatility change
7. Trend-conditioned signal
8. Reversion-conditioned signal
9. Short-term z-score
10. Long-term z-score
11. Trend acceleration
12. Price/ATR ratio

### Online Learning

The model learns from **every new price tick**:
- Uses `SGDRegressor.partial_fit()` for incremental updates
- Higher learning rate (0.2) for fast adaptation
- Lower regularization (0.0005) for quick learning

## 📁 Project Structure

```
btc-live-trading/
├── app.py                    # Streamlit dashboard
├── src/
│   ├── __init__.py
│   ├── model.py              # ML model & trading logic
│   └── data.py               # Data fetching utilities
├── Bitcoin_Historical_Data.csv
├── requirements.txt
├── .gitignore
└── README.md
```

## ⚙️ Configuration

Key parameters in `src/model.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `TRADE_FRACTION` | 0.30 | Base trade size (30%) |
| `STRONG_SIGNAL_MULTIPLIER` | 2.0 | 2x trades on strong signals |
| `WEAK_SIGNAL_MULTIPLIER` | 0.5 | 0.5x trades on weak signals |
| `STRONG_THRESHOLD` | 0.0003 | Strong signal threshold |
| `WEAK_THRESHOLD` | 0.00005 | Weak signal threshold |

## 🔧 Terminal Mode

For terminal-based trading (no GUI):

```bash
python btc_paper_trader.py
```

## 📝 License

MIT License - feel free to use and modify!

## ⚠️ Disclaimer

This is a **paper trading bot** for educational purposes only. It does not involve real money or actual trades. Past performance does not guarantee future results. Always do your own research before trading cryptocurrencies.

---

Made with ❤️ and ₿itcoin
