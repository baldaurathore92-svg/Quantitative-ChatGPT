"""
Pressure Acceleration feature.

Time derivative of momentum.
Detects if pressure is building (acceleration) or fading (deceleration).

RETAIL API LIMITATION:
This is calculated from visible order book changes only.
Real acceleration would need trade flow direction.
"""

from typing import Optional, Tuple
from dataclasses import dataclass
import math

from utils.types import Snapshot, PriceKeyedBook, FeatureResult, RollingStats
from utils.math_utils import clamp
from buffers.rolling_ema import RollingEMA
from utils.constants import ACCELERATION_RANGE


@dataclass
class AccelerationConfig:
    """Configuration for acceleration calculation."""
    tick_size: float = 0.05
    momentum_tau: float = 10.0  # Momentum EMA tau
    derivative_tau: float = 5.0  # Derivative smoothing tau
    normalize_range: float = ACCELERATION_RANGE
    min_dt: float = 0.01  # Minimum time delta


class AccelerationCalculator:
    """
    Calculate pressure acceleration.
    
    Acceleration = d(Momentum) / dt
    
    Positive acceleration = pressure building
    Negative acceleration = pressure fading
    
    IMPORTANT:
    - Uses time-aware smoothing on derivative
    - Requires multiple samples for validity
    - Noisy by nature - use with caution
    
    Signal interpretation:
    - Acceleration + positive momentum = strong bullish
    - Acceleration + negative momentum = strong bearish
    - Deceleration + momentum = trend weakening
    """
    
    def __init__(self, config: AccelerationConfig):
        self._config = config
        
        # Internal state for momentum calculation
        # (Avoid circular import by computing momentum internally)
        self._momentum_ema = RollingEMA(tau=config.momentum_tau)
        self._tick_size = config.tick_size
        
        # Derivative smoothing EMA
        self._derivative_ema = RollingEMA(tau=config.derivative_tau)
        
        # State
        self._prev_momentum: Optional[float] = None
        self._prev_timestamp: Optional[float] = None
        self._initialized = False
        self._count = 0
    
    def _calculate_pressure(self, snapshot: Snapshot) -> float:
        """Calculate order book pressure for momentum."""
        if not snapshot.bids or not snapshot.asks:
            return 0.0
        
        best_bid = snapshot.best_bid
        best_ask = snapshot.best_ask
        
        if best_bid is None or best_ask is None:
            return 0.0
        
        # Calculate weighted OBI using L1-L2
        mid = snapshot.mid_price
        if mid is None:
            return 0.0
        
        bid_pressure = 0.0
        ask_pressure = 0.0
        
        levels = min(2, snapshot.depth)
        for i in range(levels):
            bid = snapshot.bids[i]
            ask = snapshot.asks[i]
            
            bid_dist = abs(bid.price - mid) / self._tick_size
            ask_dist = abs(ask.price - mid) / self._tick_size
            
            # Exponential decay by distance
            bid_weight = math.exp(-bid_dist / 3.0)  # lambda = 3 ticks
            ask_weight = math.exp(-ask_dist / 3.0)
            
            bid_pressure += bid_weight * bid.quantity
            ask_pressure += ask_weight * ask.quantity
        
        total = bid_pressure + ask_pressure
        if total < 1e-9:
            return 0.0
        
        return (bid_pressure - ask_pressure) / total
    
    def calculate(
        self,
        snapshot: Snapshot,
        prev_book: Optional[PriceKeyedBook] = None,
        stats: Optional[RollingStats] = None
    ) -> FeatureResult:
        """
        Calculate acceleration feature.
        
        Args:
            snapshot: Current snapshot
            prev_book: Previous book
            stats: Rolling statistics
        
        Returns:
            FeatureResult with acceleration value
        """
        if not snapshot.is_valid():
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="acceleration"
            )
        
        timestamp = snapshot.timestamp
        
        # Calculate current pressure and update momentum EMA
        current_pressure = self._calculate_pressure(snapshot)
        momentum_result = self._momentum_ema.update(current_pressure, timestamp)
        current_momentum = momentum_result.value
        
        # Need previous momentum to calculate derivative
        if self._prev_momentum is None or self._prev_timestamp is None:
            self._prev_momentum = current_momentum
            self._prev_timestamp = timestamp
            self._count = 1
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=True,
                raw_value=0.0,
                name="acceleration"
            )
        
        # Calculate time delta
        dt = timestamp - self._prev_timestamp
        
        if dt < self._config.min_dt:
            # Too small time delta - skip this update
            return FeatureResult(
                value=self._derivative_ema.value,
                confidence=0.5,
                valid=self._initialized,
                raw_value=self._derivative_ema.value,
                name="acceleration"
            )
        
        # Calculate raw derivative
        d_momentum = (current_momentum - self._prev_momentum) / dt
        
        # Smooth derivative with time-aware EMA
        ema_result = self._derivative_ema.update(d_momentum, timestamp)
        
        # Update state
        self._prev_momentum = current_momentum
        self._prev_timestamp = timestamp
        self._count += 1
        
        # Need at least a few samples
        if self._count < 3:
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=True,
                raw_value=ema_result.value,
                name="acceleration"
            )
        
        self._initialized = True
        
        # Normalize
        normalized = clamp(ema_result.value / self._config.normalize_range, -1.0, 1.0)
        
        # Confidence based on sample count
        count_factor = min(1.0, self._count / 10)
        confidence = count_factor * momentum_result.confidence if hasattr(momentum_result, 'confidence') else count_factor
        
        return FeatureResult(
            value=normalized,
            confidence=confidence,
            valid=True,
            raw_value=ema_result.value,
            name="acceleration"
        )
    
    @property
    def current_acceleration(self) -> float:
        """Get current acceleration value."""
        return self._derivative_ema.value
    
    @property
    def current_momentum(self) -> float:
        """Get current momentum value."""
        return self._momentum_ema.value
    
    def get_momentum_acceleration_pair(self) -> Tuple[float, float]:
        """Get both momentum and acceleration."""
        return self._momentum_ema.value, self._derivative_ema.value
    
    def reset(self) -> None:
        """Reset state."""
        self._momentum_ema.reset()
        self._derivative_ema.reset()
        self._prev_momentum = None
        self._prev_timestamp = None
        self._initialized = False
        self._count = 0


def calculate_acceleration(
    snapshot: Snapshot,
    momentum_tau: float = 10.0,
    derivative_tau: float = 5.0
) -> Tuple[float, float]:
    """
    Calculate acceleration.
    
    Returns:
        Tuple of (acceleration_value, confidence)
    """
    config = AccelerationConfig(momentum_tau=momentum_tau, derivative_tau=derivative_tau)
    calc = AccelerationCalculator(config)
    result = calc.calculate(snapshot)
    return result.value, result.confidence
