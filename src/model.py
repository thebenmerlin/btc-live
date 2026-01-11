"""
Alpha Model for BTC Paper Trading Bot - Phase 1
Hedge-Fund-Style Signal Construction

Key Design Principles:
1. Regime awareness via features (learned, not hardcoded)
2. Multi-horizon prediction (10s, 1m, 5m) for noise reduction
3. Volatility-normalized signals for scale-free comparison
4. Continuous position sizing based on expected value

Academic Framing:
- Signals are continuous expected returns, not binary directions
- All signals normalized by volatility → comparable across regimes
- Multi-horizon blending → reduces noise, reinforces persistent trends
- Position sizing ∝ alpha/volatility → Kelly-inspired risk management
"""

import numpy as np
from sklearn.linear_model import SGDRegressor
from dataclasses import dataclass, field
from typing import Tuple, List, Optional
from datetime import datetime


# =============================================================================
# CONFIGURATION
# =============================================================================

# Horizon definitions (in ticks, where 1 tick = poll interval)
HORIZON_SHORT = 10    # ~10 seconds at 1s poll
HORIZON_MED = 60      # ~1 minute
HORIZON_LONG = 300    # ~5 minutes

# Lookback periods for feature computation
LOOKBACK_VOL_SHORT = 20    # Short-term volatility window
LOOKBACK_VOL_LONG = 100    # Long-term volatility window
LOOKBACK_TREND = 50        # Trend calculation window

# Minimum data required before trading
MIN_DATA_POINTS = max(HORIZON_LONG + 50, 120)

# Alpha blending weights - how much each horizon contributes
# Rationale: Medium horizon is primary (most signal), short is noisy, long is stable
ALPHA_WEIGHTS = {
    'short': 0.2,   # Fast but noisy
    'medium': 0.5,  # Primary signal horizon
    'long': 0.3     # Slow but stable
}

# Position sizing constraints
MAX_POSITION_WEIGHT = 0.8   # Max 80% of portfolio in BTC
MIN_POSITION_WEIGHT = -0.8  # Min -80% (short, though we can't short in paper trading)
POSITION_SCALE = 100.0      # Scaling factor for alpha → position conversion

# Volatility floor to prevent division issues
VOL_FLOOR = 1e-6


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class TradeRecord:
    """Record of a single trade."""
    timestamp: datetime
    action: str
    btc_amount: float
    usd_amount: float
    price: float
    alpha: float
    target_weight: float
    current_weight: float
    portfolio_value: float
    pnl: float


@dataclass
class RegimeMetrics:
    """Current regime state - these are FEATURES, not trading rules."""
    vol_short: float = 0.0      # Short-term realized volatility
    vol_long: float = 0.0       # Long-term realized volatility
    vol_ratio: float = 1.0      # vol_short / vol_long (>1 = expansion, <1 = contraction)
    trend_strength: float = 0.0  # |return| / volatility (signal-to-noise ratio)
    
    def to_array(self) -> np.ndarray:
        """Convert regime metrics to feature array."""
        return np.array([self.vol_short, self.vol_long, self.vol_ratio, self.trend_strength])


@dataclass
class AlphaComponents:
    """Breakdown of alpha signal by horizon."""
    alpha_short: float = 0.0
    alpha_medium: float = 0.0
    alpha_long: float = 0.0
    alpha_blended: float = 0.0
    predicted_vol: float = 0.0


@dataclass
class TradingState:
    """Current state of the trading bot."""
    cash: float = 10000.0
    btc_position: float = 0.0
    initial_cash: float = 10000.0
    trade_count: int = 0
    learn_count: int = 0
    prices: List[float] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)
    
    # Tracking for analytics
    peak_value: float = 10000.0
    alpha_history: List[float] = field(default_factory=list)
    
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
    
    @property
    def btc_weight(self) -> float:
        """Current portfolio weight in BTC (-1 to 1 scale)."""
        if not self.prices or self.portfolio_value <= 0:
            return 0.0
        btc_value = self.btc_position * self.prices[-1]
        return btc_value / self.portfolio_value
    
    @property
    def drawdown(self) -> float:
        """Current drawdown from peak."""
        if self.peak_value <= 0:
            return 0.0
        return (self.peak_value - self.portfolio_value) / self.peak_value
    
    def update_peak(self):
        """Update peak value for drawdown tracking."""
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def compute_log_returns(prices: np.ndarray, horizon: int = 1) -> np.ndarray:
    """
    Compute log returns over specified horizon.
    Log returns are additive and more suitable for statistical modeling.
    """
    if len(prices) < horizon + 1:
        return np.array([0.0])
    
    # log(P_t / P_{t-horizon})
    return np.log(prices[horizon:] / prices[:-horizon])


def compute_realized_volatility(prices: np.ndarray, window: int) -> float:
    """
    Compute realized volatility as std of log returns.
    This is the standard measure used in quantitative finance.
    """
    if len(prices) < window + 1:
        return VOL_FLOOR
    
    returns = compute_log_returns(prices[-window-1:])
    vol = np.std(returns)
    return max(vol, VOL_FLOOR)


def compute_regime_metrics(prices: np.ndarray) -> RegimeMetrics:
    """
    Compute regime features that the model uses to learn when alpha works.
    
    Key insight: We don't hardcode "if vol > X, do Y". Instead, we provide
    these as features and let the model learn the relationship.
    """
    n = len(prices)
    
    # Short-term volatility (recent market state)
    vol_short = compute_realized_volatility(prices, min(LOOKBACK_VOL_SHORT, n - 1))
    
    # Long-term volatility (baseline)
    vol_long = compute_realized_volatility(prices, min(LOOKBACK_VOL_LONG, n - 1))
    
    # Volatility ratio: >1 means vol expansion, <1 means contraction
    # Model can learn: alpha works better in X regime
    vol_ratio = vol_short / max(vol_long, VOL_FLOOR)
    
    # Trend strength: how much signal vs noise?
    # High value = trending market, low value = noisy/ranging
    if n >= LOOKBACK_TREND:
        rolling_return = np.log(prices[-1] / prices[-LOOKBACK_TREND])
        trend_strength = abs(rolling_return) / max(vol_short * np.sqrt(LOOKBACK_TREND), VOL_FLOOR)
    else:
        trend_strength = 0.0
    
    return RegimeMetrics(
        vol_short=vol_short,
        vol_long=vol_long,
        vol_ratio=vol_ratio,
        trend_strength=trend_strength
    )


def compute_features(prices: np.ndarray, regime: RegimeMetrics) -> np.ndarray:
    """
    Compute feature vector for prediction.
    
    Features are designed to be:
    1. Normalized (comparable across time periods)
    2. Regime-aware (include volatility context)
    3. Multi-scale (different lookback periods)
    
    All momentum/trend features are NORMALIZED by volatility so they're
    comparable across different market regimes.
    """
    n = len(prices)
    current_vol = max(regime.vol_short, VOL_FLOOR)
    
    # =========================================================================
    # MOMENTUM FEATURES (volatility-normalized)
    # =========================================================================
    
    # Short-term momentum (last 10 ticks)
    if n >= 11:
        mom_short = np.log(prices[-1] / prices[-10]) / current_vol
    else:
        mom_short = 0.0
    
    # Medium-term momentum (last 30 ticks)
    if n >= 31:
        mom_med = np.log(prices[-1] / prices[-30]) / current_vol
    else:
        mom_med = 0.0
    
    # Long-term momentum (last 100 ticks)
    if n >= 101:
        mom_long = np.log(prices[-1] / prices[-100]) / current_vol
    else:
        mom_long = 0.0
    
    # =========================================================================
    # MEAN REVERSION FEATURES (z-scores)
    # =========================================================================
    
    # Z-score is already volatility-normalized by definition
    if n >= 20:
        mean_20 = np.mean(prices[-20:])
        std_20 = np.std(prices[-20:])
        zscore_20 = (prices[-1] - mean_20) / max(std_20, VOL_FLOOR * prices[-1])
    else:
        zscore_20 = 0.0
    
    if n >= 50:
        mean_50 = np.mean(prices[-50:])
        std_50 = np.std(prices[-50:])
        zscore_50 = (prices[-1] - mean_50) / max(std_50, VOL_FLOOR * prices[-1])
    else:
        zscore_50 = 0.0
    
    # =========================================================================
    # TREND FEATURES
    # =========================================================================
    
    # Moving average crossover (normalized)
    if n >= 50:
        ma_10 = np.mean(prices[-10:])
        ma_50 = np.mean(prices[-50:])
        ma_cross = (ma_10 - ma_50) / (ma_50 * current_vol)
    else:
        ma_cross = 0.0
    
    # Trend acceleration (second derivative of price)
    if n >= 20:
        ret_recent = np.log(prices[-1] / prices[-10])
        ret_prior = np.log(prices[-10] / prices[-20])
        trend_accel = (ret_recent - ret_prior) / current_vol
    else:
        trend_accel = 0.0
    
    # =========================================================================
    # VOLATILITY FEATURES (the regime context)
    # =========================================================================
    
    # Log volatility ratio (captures regime state)
    log_vol_ratio = np.log(max(regime.vol_ratio, 0.1))
    
    # Volatility change (is vol expanding or contracting?)
    if n >= LOOKBACK_VOL_SHORT + 10:
        vol_10_ago = compute_realized_volatility(prices[:-10], LOOKBACK_VOL_SHORT)
        vol_change = (regime.vol_short - vol_10_ago) / max(vol_10_ago, VOL_FLOOR)
    else:
        vol_change = 0.0
    
    # =========================================================================
    # COMBINED FEATURE VECTOR
    # 12 features total: 3 momentum + 2 reversion + 2 trend + 4 regime + 1 trend_strength
    # =========================================================================
    
    return np.array([[
        mom_short,              # Volatility-normalized short momentum
        mom_med,                # Volatility-normalized medium momentum
        mom_long,               # Volatility-normalized long momentum
        zscore_20,              # Mean reversion signal (short)
        zscore_50,              # Mean reversion signal (long)
        ma_cross,               # Trend signal (MA crossover)
        trend_accel,            # Trend acceleration
        regime.vol_short,       # Absolute volatility level
        log_vol_ratio,          # Regime state (vol expansion/contraction)
        vol_change,             # Volatility dynamics
        regime.trend_strength,  # Signal-to-noise ratio
        current_vol             # Current vol for reference
    ]])


# =============================================================================
# MULTI-HORIZON ALPHA MODEL
# =============================================================================

class AlphaModel:
    """
    Multi-horizon alpha model with online learning.
    
    Architecture:
    - 3 separate SGDRegressors, each predicting returns at different horizons
    - Features are identical across horizons (let model learn horizon-specific patterns)
    - Outputs are blended into a single alpha using fixed weights
    - All predictions are volatility-normalized
    
    Why this works:
    - Short horizon captures fast mean-reversion but is noisy
    - Long horizon captures trends but reacts slowly
    - Blending reduces noise while maintaining responsiveness
    """
    
    def __init__(self):
        # Create separate models for each horizon
        # Lower alpha (less regularization) for faster adaptation
        model_config = {
            'penalty': 'l2',
            'alpha': 0.001,
            'random_state': 42,
            'warm_start': True,
            'learning_rate': 'invscaling',
            'eta0': 0.1,
            'power_t': 0.25,
            'tol': 1e-6
        }
        
        self.model_short = SGDRegressor(**model_config)
        self.model_medium = SGDRegressor(**model_config)
        self.model_long = SGDRegressor(**model_config)
        
        self.trained = {
            'short': False,
            'medium': False,
            'long': False
        }
        
        # Track learning for each horizon
        self.learn_counts = {
            'short': 0,
            'medium': 0,
            'long': 0
        }
    
    def partial_fit_horizon(self, horizon: str, features: np.ndarray, target: float):
        """Update a specific horizon model with new data."""
        model = getattr(self, f'model_{horizon}')
        model.partial_fit(features, [target])
        self.trained[horizon] = True
        self.learn_counts[horizon] += 1
    
    def predict_horizon(self, horizon: str, features: np.ndarray) -> float:
        """Get prediction from a specific horizon model."""
        if not self.trained[horizon]:
            return 0.0
        model = getattr(self, f'model_{horizon}')
        return float(model.predict(features)[0])
    
    def compute_alpha(self, features: np.ndarray, current_vol: float) -> AlphaComponents:
        """
        Compute blended alpha from all horizons.
        
        Each horizon prediction is:
        1. Obtained from its model
        2. Divided by current volatility (normalization)
        3. Weighted according to ALPHA_WEIGHTS
        4. Summed to produce final alpha
        """
        vol = max(current_vol, VOL_FLOOR)
        
        # Get raw predictions from each horizon
        pred_short = self.predict_horizon('short', features)
        pred_medium = self.predict_horizon('medium', features)
        pred_long = self.predict_horizon('long', features)
        
        # Normalize by volatility → scale-free signals
        alpha_short = pred_short / vol
        alpha_medium = pred_medium / vol
        alpha_long = pred_long / vol
        
        # Blend with fixed weights
        alpha_blended = (
            ALPHA_WEIGHTS['short'] * alpha_short +
            ALPHA_WEIGHTS['medium'] * alpha_medium +
            ALPHA_WEIGHTS['long'] * alpha_long
        )
        
        return AlphaComponents(
            alpha_short=alpha_short,
            alpha_medium=alpha_medium,
            alpha_long=alpha_long,
            alpha_blended=alpha_blended,
            predicted_vol=vol
        )
    
    @property
    def total_learn_count(self) -> int:
        return sum(self.learn_counts.values())


# =============================================================================
# POSITION SIZING
# =============================================================================

def compute_target_position(alpha: float, volatility: float) -> float:
    """
    Convert alpha signal to target position weight.
    
    Position sizing logic (Kelly-inspired):
    - Position ∝ alpha / volatility
    - Higher alpha → larger position
    - Higher volatility → smaller position (risk management)
    
    This is a simplified Kelly criterion: we scale our bet by our edge
    divided by the variance of the outcome.
    
    Returns: target weight in [-MAX_POSITION_WEIGHT, +MAX_POSITION_WEIGHT]
    """
    vol = max(volatility, VOL_FLOOR)
    
    # Scale alpha to position
    # The POSITION_SCALE parameter converts alpha units to position units
    raw_position = alpha * POSITION_SCALE / (1 + vol * 100)
    
    # Clip to prevent excessive leverage
    return np.clip(raw_position, MIN_POSITION_WEIGHT, MAX_POSITION_WEIGHT)


# =============================================================================
# TRADING BOT
# =============================================================================

class TradingBot:
    """
    Hedge-Fund-Style BTC Paper Trading Bot.
    
    Combines:
    - Multi-horizon alpha predictions
    - Regime-aware features
    - Volatility-normalized signals
    - Continuous position sizing
    """
    
    def __init__(self, initial_cash: float = 10000.0):
        self.alpha_model = AlphaModel()
        self.state = TradingState(
            cash=initial_cash,
            initial_cash=initial_cash,
            peak_value=initial_cash
        )
        
        # Store targets for each horizon for online learning
        self._pending_targets = {
            'short': [],   # (features, timestamp_index) tuples
            'medium': [],
            'long': []
        }
    
    def pretrain(self, historical_prices: List[float]) -> bool:
        """
        Pre-train all horizon models on historical data.
        
        For each horizon, we create (features_t, return_{t→t+horizon}) pairs
        and train the corresponding model.
        """
        n = len(historical_prices)
        if n < MIN_DATA_POINTS + HORIZON_LONG:
            return False
        
        prices = np.array(historical_prices)
        
        # Train each horizon
        for i in range(MIN_DATA_POINTS, n - HORIZON_LONG):
            window = prices[:i+1]
            regime = compute_regime_metrics(window)
            features = compute_features(window, regime)
            
            # Short horizon target
            if i + HORIZON_SHORT < n:
                target_short = np.log(prices[i + HORIZON_SHORT] / prices[i])
                self.alpha_model.partial_fit_horizon('short', features, target_short)
            
            # Medium horizon target
            if i + HORIZON_MED < n:
                target_med = np.log(prices[i + HORIZON_MED] / prices[i])
                self.alpha_model.partial_fit_horizon('medium', features, target_med)
            
            # Long horizon target
            if i + HORIZON_LONG < n:
                target_long = np.log(prices[i + HORIZON_LONG] / prices[i])
                self.alpha_model.partial_fit_horizon('long', features, target_long)
        
        return True
    
    def add_price(self, price: float) -> Optional[dict]:
        """
        Process a new price tick.
        
        1. Add price to history
        2. Update models with any matured predictions
        3. Compute new alpha signal
        4. Determine target position
        5. Execute trades to reach target
        
        Returns: dict with all trading info, or None if still warming up
        """
        self.state.prices.append(price)
        self.state.update_peak()
        
        n = len(self.state.prices)
        
        # Warmup check
        if n < MIN_DATA_POINTS:
            return None
        
        prices = np.array(self.state.prices)
        
        # =====================================================================
        # ONLINE LEARNING: Update models with matured predictions
        # =====================================================================
        self._process_pending_targets(prices)
        
        # =====================================================================
        # COMPUTE FEATURES AND REGIME
        # =====================================================================
        regime = compute_regime_metrics(prices)
        features = compute_features(prices, regime)
        
        # =====================================================================
        # QUEUE NEW TARGETS FOR FUTURE LEARNING
        # =====================================================================
        # When these horizons mature, we'll have the actual returns to learn from
        self._pending_targets['short'].append((features.copy(), n - 1))
        self._pending_targets['medium'].append((features.copy(), n - 1))
        self._pending_targets['long'].append((features.copy(), n - 1))
        
        # Bootstrap if not yet trained
        if not self.alpha_model.trained['medium']:
            self.alpha_model.partial_fit_horizon('short', features, 0.0)
            self.alpha_model.partial_fit_horizon('medium', features, 0.0)
            self.alpha_model.partial_fit_horizon('long', features, 0.0)
        
        # =====================================================================
        # COMPUTE ALPHA
        # =====================================================================
        alpha = self.alpha_model.compute_alpha(features, regime.vol_short)
        self.state.alpha_history.append(alpha.alpha_blended)
        
        # =====================================================================
        # COMPUTE TARGET POSITION
        # =====================================================================
        target_weight = compute_target_position(alpha.alpha_blended, regime.vol_short)
        current_weight = self.state.btc_weight
        
        # =====================================================================
        # EXECUTE TRADE TO REACH TARGET
        # =====================================================================
        trade_info = self._execute_rebalance(price, target_weight, current_weight, alpha)
        
        return {
            'price': price,
            'alpha': alpha,
            'regime': regime,
            'target_weight': target_weight,
            'current_weight': current_weight,
            'trade_executed': trade_info is not None,
            'trade_info': trade_info,
            'portfolio_value': self.state.portfolio_value,
            'pnl': self.state.pnl,
            'pnl_pct': self.state.pnl_pct,
            'drawdown': self.state.drawdown,
            'learn_count': self.alpha_model.total_learn_count
        }
    
    def _process_pending_targets(self, prices: np.ndarray):
        """
        Process matured predictions and update models.
        
        When a prediction made H ticks ago has matured, we can now observe
        the actual return and use it to update the model.
        """
        n = len(prices)
        
        # Process each horizon
        for horizon_name, horizon_ticks in [('short', HORIZON_SHORT), 
                                              ('medium', HORIZON_MED), 
                                              ('long', HORIZON_LONG)]:
            new_pending = []
            for (features, idx) in self._pending_targets[horizon_name]:
                # Check if this prediction has matured
                if idx + horizon_ticks < n:
                    # Compute actual return that occurred
                    actual_return = np.log(prices[idx + horizon_ticks] / prices[idx])
                    self.alpha_model.partial_fit_horizon(horizon_name, features, actual_return)
                else:
                    # Not yet matured, keep waiting
                    new_pending.append((features, idx))
            
            self._pending_targets[horizon_name] = new_pending
            
            # Limit pending queue size to prevent memory issues
            if len(self._pending_targets[horizon_name]) > horizon_ticks + 100:
                self._pending_targets[horizon_name] = self._pending_targets[horizon_name][-horizon_ticks:]
    
    def _execute_rebalance(self, price: float, target_weight: float, 
                           current_weight: float, alpha: AlphaComponents) -> Optional[TradeRecord]:
        """
        Rebalance portfolio to reach target weight.
        
        Position sizing is CONTINUOUS, not binary:
        - If target > current: BUY to increase BTC exposure
        - If target < current: SELL to decrease BTC exposure
        - Trade size is proportional to the difference
        """
        portfolio_value = self.state.portfolio_value
        
        # Calculate target and current BTC values
        target_btc_value = portfolio_value * target_weight
        current_btc_value = self.state.btc_position * price
        
        # Difference we need to trade
        delta_value = target_btc_value - current_btc_value
        
        # Skip tiny trades (noise)
        if abs(delta_value) < 10.0:  # Min $10 trade
            return None
        
        # Execute the trade
        if delta_value > 0:
            # BUY: spend cash to acquire BTC
            spend = min(delta_value, self.state.cash - 1.0)  # Keep $1 buffer
            if spend <= 0:
                return None
            btc_acquired = spend / price
            self.state.cash -= spend
            self.state.btc_position += btc_acquired
            action = "BUY"
            trade_btc = btc_acquired
            trade_usd = spend
        else:
            # SELL: sell BTC for cash
            sell_value = min(abs(delta_value), current_btc_value)
            if sell_value <= 0:
                return None
            btc_sold = sell_value / price
            if btc_sold > self.state.btc_position:
                btc_sold = self.state.btc_position
            self.state.btc_position -= btc_sold
            self.state.cash += btc_sold * price
            action = "SELL"
            trade_btc = btc_sold
            trade_usd = btc_sold * price
        
        self.state.trade_count += 1
        
        trade_record = TradeRecord(
            timestamp=datetime.now(),
            action=action,
            btc_amount=trade_btc,
            usd_amount=trade_usd,
            price=price,
            alpha=alpha.alpha_blended,
            target_weight=target_weight,
            current_weight=current_weight,
            portfolio_value=self.state.portfolio_value,
            pnl=self.state.pnl
        )
        self.state.trades.append(trade_record)
        
        return trade_record
    
    @property
    def is_ready(self) -> bool:
        """Check if bot has enough data to trade."""
        return len(self.state.prices) >= MIN_DATA_POINTS
    
    @property
    def warmup_progress(self) -> Tuple[int, int]:
        """Get warmup progress (current, required)."""
        return len(self.state.prices), MIN_DATA_POINTS
