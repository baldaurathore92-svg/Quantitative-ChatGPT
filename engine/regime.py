"""
Market Regime Detection.

Determines current market state:
- TREND: Strong directional movement
- PULLBACK: Counter-trend within trend
- RANGE: Sideways/consolidation
- NOISE: Low signal, high variance

Uses deterministic rules only. No machine learning.
"""

from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum

from utils.types import Snapshot, Regime, RollingStats
from utils.math_utils import clamp
from buffers.rolling_variance import RollingVariance
from buffers.rolling_mean import RollingMean
from utils.constants import (
    TREND_MOMENTUM_THRESHOLD,
    TREND_OBI_THRESHOLD,
    RANGE_SPREAD_THRESHOLD,
    NOISE_VOL_THRESHOLD
)


@dataclass
class RegimeConfig:
    """Configuration for regime detection."""
    momentum_threshold: float = TREND_MOMENTUM_THRESHOLD
    obi_threshold: float = TREND_OBI_THRESHOLD
    spread_threshold: float = RANGE_SPREAD_THRESHOLD
    vol_threshold: float = NOISE_VOL_THRESHOLD
    
    momentum_window: int = 20
    obi_window: int = 20
    spread_window: int = 30
    price_window: int = 50
    
    trend_persistence: int = 3  # Consecutive readings for trend
    noise_persistence: int = 5  # Consecutive readings for noise


class RegimeDetector:
    """
    Detect market regime from order book dynamics.
    
    Deterministic rules based on:
    - Momentum strength and direction
    - OBI consistency
    - Spread behavior
    - Price volatility
    
    IMPORTANT:
    Uses ONLY observable snapshot data.
    No external indicators or ML models.
    
    Regime rules:
    - TREND: |momentum| > threshold, OBI aligned, spread stable
    - PULLBACK: Trend + counter-trend momentum
    - RANGE: Low momentum, spread stable, price bounded
    - NOISE: High volatility, conflicting signals
    """
    
    def __init__(self, config: RegimeConfig):
        self._config = config
        
        # Rolling statistics
        self._momentum_buffer = RollingMean(config.momentum_window)
        self._obi_buffer = RollingMean(config.obi_window)
        self._spread_buffer = RollingMean(config.spread_window)
        self._price_variance = RollingVariance(config.price_window)
        
        # State tracking
        self._current_regime = Regime.NOISE
        self._regime_count = 0
        self._prev_price: Optional[float] = None
        self._prev_momentum: float = 0.0
    
    def detect(
        self,
        momentum: float,
        obi: float,
        spread_ticks: float,
        price: float
    ) -> Regime:
        """
        Detect current market regime.
        
        Args:
            momentum: Current momentum value
            obi: Current OBI value
            spread_ticks: Current spread in ticks
            price: Current price (LTP or mid)
        
        Returns:
            Detected regime
        """
        # Update buffers
        self._momentum_buffer.update(momentum)
        self._obi_buffer.update(obi)
        self._spread_buffer.update(spread_ticks)
        
        if price > 0:
            self._price_variance.update(price)
        
        # Get smoothed values
        avg_momentum = self._momentum_buffer.value
        avg_obi = self._obi_buffer.value
        avg_spread = self._spread_buffer.value
        
        # Safe calculation of price volatility with division by zero protection
        price_mean = self._price_variance.mean
        price_vol = 0.0
        if self._price_variance.valid and abs(price_mean) > 1e-9:
            price_vol = self._price_variance.std / price_mean
        
        # Detect regime
        new_regime = self._classify_regime(
            avg_momentum, avg_obi, avg_spread, price_vol, momentum
        )
        
        # Require persistence for regime change
        if new_regime != self._current_regime:
            if new_regime == Regime.TREND:
                threshold = self._config.trend_persistence
            elif new_regime == Regime.NOISE:
                threshold = self._config.noise_persistence
            else:
                threshold = 2
            
            if self._regime_count >= threshold or self._current_regime == Regime.NOISE:
                self._current_regime = new_regime
                self._regime_count = 0
            else:
                self._regime_count += 1
        else:
            self._regime_count = 0
        
        self._prev_price = price
        self._prev_momentum = momentum
        
        return self._current_regime
    
    def _classify_regime(
        self,
        avg_momentum: float,
        avg_obi: float,
        avg_spread: float,
        price_vol: float,
        current_momentum: float
    ) -> Regime:
        """Classify regime from metrics."""
        momentum_abs = abs(avg_momentum)
        obi_abs = abs(avg_obi)
        
        # Check for noise first (high volatility, conflicting signals)
        if price_vol > self._config.vol_threshold:
            return Regime.NOISE
        
        # Check for trend
        if momentum_abs > self._config.momentum_threshold:
            # Momentum aligned with OBI
            momentum_dir = 1 if avg_momentum > 0 else -1
            obi_dir = 1 if avg_obi > 0 else -1
            
            if momentum_dir == obi_dir and obi_abs > self._config.obi_threshold:
                return Regime.TREND
            
            # Momentum strong but OBI diverging - pullback
            if momentum_dir != obi_dir and obi_abs > self._config.obi_threshold * 0.5:
                return Regime.PULLBACK
        
        # Check for range (low momentum, stable spread)
        if momentum_abs < self._config.momentum_threshold * 0.5:
            if avg_spread < self._config.spread_threshold:
                return Regime.RANGE
        
        # Default to noise if no clear pattern
        if momentum_abs < 0.2 and obi_abs < 0.2:
            return Regime.NOISE
        
        # Maintain current regime as fallback
        return self._current_regime
    
    @property
    def current_regime(self) -> Regime:
        """Get current regime."""
        return self._current_regime
    
    def get_regime_confidence(self) -> float:
        """
        Get confidence for current regime.
        
        Different regimes have different base confidence levels.
        """
        confidences = {
            Regime.TREND: 1.0,
            Regime.PULLBACK: 0.85,
            Regime.RANGE: 0.80,
            Regime.NOISE: 0.50
        }
        return confidences.get(self._current_regime, 0.5)
    
    def get_regime_threshold_multiplier(self) -> float:
        """
        Get threshold multiplier for current regime.
        
        Higher multiplier = harder to trigger signal.
        """
        multipliers = {
            Regime.TREND: 1.0,
            Regime.PULLBACK: 1.1,
            Regime.RANGE: 1.2,
            Regime.NOISE: 1.5
        }
        return multipliers.get(self._current_regime, 1.5)
    
    def reset(self) -> None:
        """Reset all state."""
        self._momentum_buffer.reset()
        self._obi_buffer.reset()
        self._spread_buffer.reset()
        self._price_variance.reset()
        self._current_regime = Regime.NOISE
        self._regime_count = 0
        self._prev_price = None
        self._prev_momentum = 0.0
    
    def get_stats(self) -> Dict:
        """Get regime statistics."""
        return {
            'current_regime': self._current_regime.name,
            'regime_confidence': self.get_regime_confidence(),
            'threshold_multiplier': self.get_regime_threshold_multiplier(),
            'avg_momentum': self._momentum_buffer.value,
            'avg_obi': self._obi_buffer.value,
            'avg_spread': self._spread_buffer.value,
            'price_volatility': self._price_variance.std
        }


class RegimeFeatureWeights:
    """
    Feature weight adjustments by regime.
    
    IMPORTANT: These are RELATIVE weights, not absolute scaling.
    Uniformly scaling all weights has NO effect on normalized composite.
    
    Instead, we adjust RELATIVE importance:
    - TREND: Increase momentum and OBI weights relative to others
    - RANGE: Increase microprice weight, decrease momentum
    - NOISE: Reduce relative weight of noisy features
    """
    
    def __init__(self):
        # Base relative weights (all 1.0 = equal importance)
        # These multiply the base weights from config
        self._regime_adjustments = {
            Regime.TREND: {
                'microprice': 0.8,
                'weighted_obi': 1.2,
                'momentum': 1.3,
                'acceleration': 1.0,
                'depth_slope': 0.8,
                'spread_compression': 0.6,
                'queue_persistence': 0.7,
                'refill_proxy': 0.8,
                'ltp_confirmation': 0.9
            },
            Regime.PULLBACK: {
                'microprice': 1.0,
                'weighted_obi': 1.0,
                'momentum': 0.8,
                'acceleration': 1.2,
                'depth_slope': 1.0,
                'spread_compression': 0.8,
                'queue_persistence': 1.0,
                'refill_proxy': 1.0,
                'ltp_confirmation': 1.1
            },
            Regime.RANGE: {
                'microprice': 1.1,
                'weighted_obi': 0.9,
                'momentum': 0.7,
                'acceleration': 0.8,
                'depth_slope': 1.0,
                'spread_compression': 1.0,
                'queue_persistence': 1.1,
                'refill_proxy': 1.0,
                'ltp_confirmation': 1.0
            },
            Regime.NOISE: {
                'microprice': 1.0,
                'weighted_obi': 1.0,
                'momentum': 0.8,
                'acceleration': 0.7,
                'depth_slope': 1.0,
                'spread_compression': 0.9,
                'queue_persistence': 1.0,
                'refill_proxy': 0.9,
                'ltp_confirmation': 1.1
            }
        }
    
    def get_adjusted_weights(
        self,
        regime: Regime,
        base_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Get regime-adjusted feature weights.
        
        Args:
            regime: Current market regime
            base_weights: Base feature weights from config
        
        Returns:
            Adjusted weights dictionary
        """
        adjustments = self._regime_adjustments.get(regime, {})
        
        adjusted = {}
        for feature, weight in base_weights.items():
            adjustment = adjustments.get(feature, 1.0)
            adjusted[feature] = weight * adjustment
        
        return adjusted
    
    def get_weight_adjustment(self, regime: Regime, feature: str) -> float:
        """Get weight adjustment for specific feature in regime."""
        adjustments = self._regime_adjustments.get(regime, {})
        return adjustments.get(feature, 1.0)
