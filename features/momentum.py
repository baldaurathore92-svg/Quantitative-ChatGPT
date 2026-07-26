"""
Pressure Momentum feature.

Time-aware EMA of order book imbalance.
Tracks the persistence of buying/selling pressure.

CRITICAL: Uses TIME-AWARE EMA, not fixed-alpha EMA.
alpha = 1 - exp(-dt / tau)

This ensures consistent behavior regardless of snapshot frequency.
"""

from typing import Optional, Tuple
from dataclasses import dataclass
import math

from utils.types import Snapshot, PriceKeyedBook, FeatureResult, RollingStats
from utils.math_utils import clamp
from buffers.rolling_ema import RollingEMA
from utils.constants import MOMENTUM_RANGE


@dataclass
class MomentumConfig:
    """Configuration for momentum calculation."""
    tick_size: float = 0.05
    tau: float = 10.0  # Time constant in seconds
    normalize_range: float = MOMENTUM_RANGE
    use_weighted_obi: bool = True  # Use weighted OBI for pressure


class MomentumCalculator:
    """
    Calculate pressure momentum using time-aware EMA.
    
    Momentum = EMA(order_book_pressure, tau)
    
    Where order_book_pressure is typically OBI or microprice deviation.
    
    This is the PRIMARY signal for trend detection.
    
    IMPORTANT:
    - Uses time-aware alpha, not fixed alpha
    - Requires timestamps for proper calculation
    - Resets gracefully on gap detection
    
    Signal interpretation:
    - Momentum > 0 = bullish pressure building
    - Momentum < 0 = bearish pressure building
    - Momentum magnitude indicates strength
    """
    
    def __init__(self, config: MomentumConfig):
        self._config = config
        self._ema = RollingEMA(tau=config.tau)
        self._last_timestamp: Optional[float] = None
        self._prev_obi = 0.0
    
    def calculate(
        self,
        snapshot: Snapshot,
        prev_book: Optional[PriceKeyedBook] = None,
        stats: Optional[RollingStats] = None
    ) -> FeatureResult:
        """
        Calculate momentum feature.
        
        Args:
            snapshot: Current snapshot
            prev_book: Previous book (for OBI calculation)
            stats: Rolling statistics (not used)
        
        Returns:
            FeatureResult with momentum value
        """
        if not snapshot.is_valid():
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="momentum"
            )
        
        timestamp = snapshot.timestamp
        
        # Calculate current pressure (OBI)
        current_obi = self._calculate_pressure(snapshot)
        
        # Check for time gap (reconnect, etc.)
        if self._last_timestamp is not None:
            dt = timestamp - self._last_timestamp
            if dt > self._config.tau * 3:  # Gap > 3 tau
                # Significant gap - reset state
                self._ema.reset()
                self._prev_obi = current_obi
        
        # Update EMA with time-aware decay
        result = self._ema.update(current_obi, timestamp)
        self._last_timestamp = timestamp
        
        # Normalize momentum
        normalized = clamp(result.value / self._config.normalize_range, -1.0, 1.0)
        
        # Calculate confidence
        confidence = self._calculate_confidence(result.valid, self._ema.count)
        
        return FeatureResult(
            value=normalized,
            confidence=confidence,
            valid=result.valid,
            raw_value=result.value,
            name="momentum"
        )
    
    def _calculate_pressure(self, snapshot: Snapshot) -> float:
        """
        Calculate order book pressure.
        
        Uses simple OBI if use_weighted_obi is False.
        Uses weighted OBI (L1-L2) otherwise.
        """
        if not snapshot.bids or not snapshot.asks:
            return 0.0
        
        best_bid = snapshot.best_bid
        best_ask = snapshot.best_ask
        
        if best_bid is None or best_ask is None:
            return 0.0
        
        if self._config.use_weighted_obi:
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
                
                bid_dist = abs(bid.price - mid) / self._config.tick_size
                ask_dist = abs(ask.price - mid) / self._config.tick_size
                
                # Exponential decay by distance
                bid_weight = math.exp(-bid_dist / 3.0)  # lambda = 3 ticks
                ask_weight = math.exp(-ask_dist / 3.0)
                
                bid_pressure += bid_weight * bid.quantity
                ask_pressure += ask_weight * ask.quantity
            
            total = bid_pressure + ask_pressure
            if total < 1e-9:
                return 0.0
            
            return (bid_pressure - ask_pressure) / total
        else:
            # Simple L1 OBI
            bid_qty = best_bid.quantity
            ask_qty = best_ask.quantity
            total = bid_qty + ask_qty
            
            if total < 1e-9:
                return 0.0
            
            return (bid_qty - ask_qty) / total
    
    def _calculate_confidence(self, valid: bool, sample_count: int) -> float:
        """Calculate confidence based on sample count."""
        if not valid:
            return 0.0
        
        # Need at least a few samples for confidence
        if sample_count >= 20:
            return 1.0
        elif sample_count >= 10:
            return 0.8
        elif sample_count >= 5:
            return 0.6
        else:
            return 0.4
    
    @property
    def current_momentum(self) -> float:
        """Get current momentum value."""
        return self._ema.value
    
    @property
    def direction(self) -> int:
        """Get momentum direction: +1 bullish, -1 bearish, 0 neutral."""
        val = self._ema.value
        if val > 0.1:
            return 1
        elif val < -0.1:
            return -1
        return 0
    
    def reset(self) -> None:
        """Reset momentum state."""
        self._ema.reset()
        self._last_timestamp = None
        self._prev_obi = 0.0
    
    def get_ema_stats(self) -> dict:
        """Get EMA statistics."""
        return {
            'value': self._ema.value,
            'valid': self._ema.valid,
            'count': self._ema.count,
            'tau': self._ema.tau
        }


class MomentumDerivative:
    """
    Calculate momentum derivative (rate of change).
    
    Useful for detecting momentum acceleration/deceleration.
    """
    
    def __init__(self, tau: float = 10.0):
        self._momentum = MomentumCalculator(MomentumConfig(tau=tau))
        self._prev_momentum = 0.0
        self._prev_timestamp: Optional[float] = None
    
    def calculate(
        self,
        snapshot: Snapshot,
        prev_book: Optional[PriceKeyedBook] = None,
        stats: Optional[RollingStats] = None
    ) -> FeatureResult:
        """
        Calculate momentum derivative.
        
        Returns rate of change of momentum.
        """
        result = self._momentum.calculate(snapshot, prev_book, stats)
        
        if not result.valid:
            self._prev_momentum = 0.0
            self._prev_timestamp = snapshot.timestamp
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="momentum_derivative"
            )
        
        timestamp = snapshot.timestamp
        
        if self._prev_timestamp is None:
            self._prev_momentum = result.raw_value
            self._prev_timestamp = timestamp
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=True,
                raw_value=0.0,
                name="momentum_derivative"
            )
        
        dt = timestamp - self._prev_timestamp
        
        if dt <= 0:
            return FeatureResult(
                value=0.0,
                confidence=result.confidence,
                valid=True,
                raw_value=0.0,
                name="momentum_derivative"
            )
        
        # Calculate derivative
        d_momentum = (result.raw_value - self._prev_momentum) / dt
        
        # Normalize
        normalized = clamp(d_momentum * 10, -1.0, 1.0)  # Scale factor for typical values
        
        self._prev_momentum = result.raw_value
        self._prev_timestamp = timestamp
        
        return FeatureResult(
            value=normalized,
            confidence=result.confidence,
            valid=True,
            raw_value=d_momentum,
            name="momentum_derivative"
        )
    
    def reset(self) -> None:
        """Reset state."""
        self._momentum.reset()
        self._prev_momentum = 0.0
        self._prev_timestamp = None


def calculate_momentum(
    snapshot: Snapshot,
    tau: float = 10.0
) -> Tuple[float, float]:
    """
    Calculate momentum.
    
    Note: Creates new calculator each time.
    For proper tracking, use MomentumCalculator class.
    
    Returns:
        Tuple of (momentum_value, confidence)
    """
    config = MomentumConfig(tau=tau)
    calc = MomentumCalculator(config)
    result = calc.calculate(snapshot)
    return result.value, result.confidence
