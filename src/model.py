"""
ML Model and Trading Logic for BTC Paper Trading Bot.
Implements Volatility-Conditioned Trend/Regime Model with continuous online learning.
"""

import numpy as np
from sklearn.linear_model import SGDRegressor
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from datetime import datetime


# Configuration
LOOKBACK_SHORT = 10
LOOKBACK_MED = 20
LOOKBACK_LONG = 50
VOL_REGIME_LOOKBACK = 30

# Trading thresholds
TRADE_FRACTION = 0.30
BASE_SIGNAL_MULTIPLIER = 1.0
STRONG_SIGNAL_MULTIPLIER = 2.0
WEAK_SIGNAL_MULTIPLIER = 0.5
WEAK_THRESHOLD = 0.00005
STRONG_THRESHOLD = 0.0003


@dataclass
class TradeRecord:
    """Record of a single trade."""
    timestamp: datetime
    action: str
    signal_strength: str
    btc_amount: float
    usd_amount: float
    price: float
    prediction: float
    portfolio_value: float
    pnl: float


@dataclass
class TradingState:
    """Current state of the trading bot."""
    cash: float = 10000.0
    btc_position: float = 0.0
    initial_cash: float = 10000.0
    trade_count: int = 0
    learn_count: int = 0
    prices: List[float] = field(default_factory=list)
    predictions: List[float] = field(default_factory=list)
    actual_returns: List[float] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)
    
    @property
    def portfolio_value(self) -> float:
        if not self.prices:
            return self.cash
        return self.cash + (self.btc_position * self.prices[-1])
    
    @property
    def pnl(self) -> float:
        return self.portfolio_value - self.initial_cash
    
    @property
    def pnl_pct(self) -> float:
        return (self.pnl / self.initial_cash) * 100


class TradingBot:
    """
    BTC Paper Trading Bot with Volatility-Conditioned Regime Model.
    Uses SGDRegressor for continuous online learning.
    """
    
    def __init__(self, initial_cash: float = 10000.0):
        self.model = SGDRegressor(
            penalty='l2',
            alpha=0.0005,
            random_state=42,
            warm_start=True,
            learning_rate='invscaling',
            eta0=0.2,
            power_t=0.1,
            tol=1e-5
        )
        self.model_trained = False
        self.state = TradingState(
            cash=initial_cash,
            initial_cash=initial_cash
        )
        self.min_data_points = max(LOOKBACK_LONG + 10, 60)
    
    def pretrain(self, historical_prices: List[float]) -> bool:
        """Pre-train the model on historical data."""
        if len(historical_prices) < self.min_data_points:
            return False
        
        for i in range(self.min_data_points, len(historical_prices)):
            prices_window = historical_prices[:i + 1]
            prev_features = self._compute_features(prices_window[:-1])
            target = self._compute_target(prices_window)
            self.model.partial_fit(prev_features, [target])
            self.state.learn_count += 1
        
        self.model_trained = True
        return True
    
    def add_price(self, price: float) -> Optional[dict]:
        """
        Add a new price and potentially execute a trade.
        Returns trade info dict if a trade was executed, None otherwise.
        """
        self.state.prices.append(price)
        
        # Check if we have enough data
        if len(self.state.prices) < self.min_data_points:
            return None
        
        # Compute features
        features = self._compute_features(self.state.prices)
        
        # Online learning from previous prediction
        if len(self.state.prices) > self.min_data_points and self.model_trained:
            actual_return = self._compute_target(self.state.prices)
            prev_features = self._compute_features(self.state.prices[:-1])
            self.model.partial_fit(prev_features, [actual_return])
            self.state.learn_count += 1
            self.state.actual_returns.append(actual_return)
        
        # Make prediction
        if self.model_trained:
            predicted_return = float(self.model.predict(features)[0])
        else:
            predicted_return = 0.0
            self.model.partial_fit(features, [0.0])
            self.model_trained = True
            self.state.learn_count += 1
        
        self.state.predictions.append(predicted_return)
        
        # Get regime info
        vol_regime = features[0][0]
        trend_regime = features[0][1]
        
        # Decide action
        action, signal_strength = self._decide_action(predicted_return, vol_regime, trend_regime)
        
        # Execute trade
        trade_info = self._execute_trade(action, signal_strength, price, predicted_return)
        
        return {
            'price': price,
            'prediction': predicted_return,
            'action': action,
            'signal_strength': signal_strength,
            'vol_regime': vol_regime,
            'trend_regime': trend_regime,
            'trade_executed': trade_info is not None,
            'trade_info': trade_info
        }
    
    def _compute_features(self, prices: List[float]) -> np.ndarray:
        """Compute Volatility-Conditioned Regime features."""
        n = len(prices)
        prices = np.array(prices)
        
        # Volatility regime
        current_vol = self._compute_volatility(prices, LOOKBACK_MED)
        vol_regime = self._detect_volatility_regime(prices, current_vol)
        inv_vol_regime = 1 - vol_regime
        
        # Trend regime
        trend_regime = self._detect_trend_regime(prices)
        
        # Trend features
        short_trend = (prices[-1] - prices[-LOOKBACK_SHORT]) / prices[-LOOKBACK_SHORT] if n >= LOOKBACK_SHORT else 0
        
        ema_short = self._compute_ema(prices, LOOKBACK_SHORT)
        ema_long = self._compute_ema(prices, LOOKBACK_LONG) if n >= LOOKBACK_LONG else ema_short
        long_trend = (ema_short - ema_long) / ema_long if ema_long != 0 else 0
        
        # Volatility features
        vol_short = self._compute_volatility(prices, LOOKBACK_SHORT)
        vol_long = self._compute_volatility(prices, LOOKBACK_LONG) if n >= LOOKBACK_LONG else vol_short
        vol_normalized = current_vol / (vol_long + 1e-10) - 1 if n >= LOOKBACK_LONG else 0
        vol_change = (vol_short - vol_long) / vol_long if vol_long > 0 else 0
        
        # Mean reversion features
        mean_short = np.mean(prices[-LOOKBACK_SHORT:])
        std_short = np.std(prices[-LOOKBACK_SHORT:])
        zscore_short = (prices[-1] - mean_short) / std_short if std_short > 0 else 0
        
        if n >= LOOKBACK_LONG:
            mean_long = np.mean(prices[-LOOKBACK_LONG:])
            std_long = np.std(prices[-LOOKBACK_LONG:])
            zscore_long = (prices[-1] - mean_long) / std_long if std_long > 0 else 0
        else:
            zscore_long = zscore_short
        
        # Regime-conditioned features
        trend_conditioned = short_trend * vol_regime + long_trend * vol_regime
        reversion_conditioned = zscore_short * inv_vol_regime + zscore_long * inv_vol_regime
        
        # Trend acceleration
        if n >= LOOKBACK_SHORT + 5:
            trend_now = (prices[-1] - prices[-5]) / prices[-5]
            trend_prev = (prices[-5] - prices[-10]) / prices[-10]
            trend_accel = trend_now - trend_prev
        else:
            trend_accel = 0
        
        # Price ATR ratio
        atr = self._compute_atr(prices, LOOKBACK_MED)
        price_atr_ratio = (prices[-1] - prices[-2]) / atr if n >= LOOKBACK_MED and atr > 0 else 0
        
        return np.array([[
            vol_regime, trend_regime, short_trend, long_trend,
            vol_normalized, vol_change, trend_conditioned, reversion_conditioned,
            zscore_short, zscore_long, trend_accel, price_atr_ratio
        ]])
    
    def _compute_target(self, prices: List[float]) -> float:
        """Compute target return."""
        return (prices[-1] - prices[-2]) / prices[-2]
    
    def _compute_volatility(self, prices: np.ndarray, period: int) -> float:
        if len(prices) < period + 1:
            return 0.0
        returns = np.diff(prices[-period-1:]) / prices[-period-1:-1]
        return float(np.std(returns))
    
    def _detect_volatility_regime(self, prices: np.ndarray, current_vol: float) -> float:
        if len(prices) < VOL_REGIME_LOOKBACK + 10:
            return 0.5
        
        vol_history = []
        for i in range(VOL_REGIME_LOOKBACK, len(prices)):
            window = prices[i-VOL_REGIME_LOOKBACK:i]
            returns = np.diff(window) / window[:-1]
            vol_history.append(np.std(returns))
        
        if not vol_history:
            return 0.5
        
        return float(np.sum(np.array(vol_history) < current_vol) / len(vol_history))
    
    def _detect_trend_regime(self, prices: np.ndarray) -> float:
        if len(prices) < LOOKBACK_LONG:
            return 0.0
        
        sma_short = np.mean(prices[-LOOKBACK_SHORT:])
        sma_long = np.mean(prices[-LOOKBACK_LONG:])
        trend_strength = (sma_short - sma_long) / sma_long
        return float(np.clip(trend_strength * 100, -1, 1))
    
    def _compute_ema(self, prices: np.ndarray, period: int) -> float:
        if len(prices) < period:
            return float(prices[-1])
        
        multiplier = 2 / (period + 1)
        ema = prices[-period]
        for price in prices[-period + 1:]:
            ema = (price - ema) * multiplier + ema
        return float(ema)
    
    def _compute_atr(self, prices: np.ndarray, period: int) -> float:
        if len(prices) < period + 1:
            return 0.0
        ranges = np.abs(np.diff(prices[-period-1:]))
        return float(np.mean(ranges))
    
    def _decide_action(self, predicted_return: float, vol_regime: float, trend_regime: float) -> Tuple[str, str]:
        """Decide trading action based on prediction."""
        magnitude = abs(predicted_return)
        
        if magnitude >= STRONG_THRESHOLD:
            signal_strength = "strong"
        elif magnitude >= WEAK_THRESHOLD:
            signal_strength = "normal"
        else:
            signal_strength = "weak"
        
        if predicted_return >= 0:
            return "BUY", signal_strength
        else:
            return "SELL", signal_strength
    
    def _execute_trade(self, action: str, signal_strength: str, price: float, prediction: float) -> Optional[TradeRecord]:
        """Execute a paper trade."""
        # Determine trade fraction
        if signal_strength == "strong":
            fraction = TRADE_FRACTION * STRONG_SIGNAL_MULTIPLIER
        elif signal_strength == "weak":
            fraction = TRADE_FRACTION * WEAK_SIGNAL_MULTIPLIER
        else:
            fraction = TRADE_FRACTION * BASE_SIGNAL_MULTIPLIER
        
        fraction = min(fraction, 0.6)
        
        trade_btc = 0.0
        trade_usd = 0.0
        executed = False
        
        if action == "BUY" and self.state.cash > 10.0:
            trade_usd = self.state.cash * fraction
            trade_btc = trade_usd / price
            self.state.cash -= trade_usd
            self.state.btc_position += trade_btc
            executed = True
            
        elif action == "SELL" and self.state.btc_position > 0.0001:
            trade_btc = self.state.btc_position * fraction
            trade_usd = trade_btc * price
            self.state.cash += trade_usd
            self.state.btc_position -= trade_btc
            executed = True
        
        if executed:
            self.state.trade_count += 1
            trade_record = TradeRecord(
                timestamp=datetime.now(),
                action=action,
                signal_strength=signal_strength,
                btc_amount=trade_btc,
                usd_amount=trade_usd,
                price=price,
                prediction=prediction,
                portfolio_value=self.state.portfolio_value,
                pnl=self.state.pnl
            )
            self.state.trades.append(trade_record)
            return trade_record
        
        return None
    
    @property
    def is_ready(self) -> bool:
        """Check if bot has enough data to trade."""
        return len(self.state.prices) >= self.min_data_points
    
    @property
    def warmup_progress(self) -> Tuple[int, int]:
        """Get warmup progress (current, required)."""
        return len(self.state.prices), self.min_data_points
