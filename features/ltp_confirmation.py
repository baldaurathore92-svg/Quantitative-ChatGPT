"""
LTP Confirmation feature.

Validates signals against Last Traded Price (LTP) movement.
Detects divergence between signal and actual trades.

If Microprice suggests upward movement but LTP is dropping,
the signal is suspect.

RETAIL API LIMITATION:
We cannot see trade direction (buyer/seller initiated).
We only see LTP and volume, not aggressor side.
"""

from typing import Optional, Tuple
from dataclasses import dataclass
import math

from utils.types import Snapshot, PriceKeyedBook, FeatureResult, RollingStats
from utils.math_utils import clamp, ticks_from_mid
from utils.constants import EPSILON


@dataclass
class LTPConfirmationConfig:
    """Configuration for LTP confirmation."""
    tick_size: float = 0.05
    ltp_change_threshold: float = 2.0  # Ticks to consider significant
    divergence_penalty: float = 0.5  # Confidence reduction on divergence
    history_size: int = 10


class LTPConfirmationCalculator:
    """
    Calculate LTP confirmation feature.
    
    Validates directional signals against actual trade prices.
    
    Confirmation rules:
    - Bullish signal + LTP rising = CONFIRMED (high confidence)
    - Bullish signal + LTP falling = DIVERGENCE (low confidence)
    - Bearish signal + LTP falling = CONFIRMED (high confidence)
    - Bearish signal + LTP rising = DIVERGENCE (low confidence)
    
    IMPORTANT:
    We cannot see aggressor side, so:
    - Rising LTP could be buyer-initiated OR seller lifting offers
    - Falling LTP could be seller-initiated OR buyer hitting bids
    
    This is a PROXY validation, not definitive confirmation.
    """
    
    def __init__(self, config: LTPConfirmationConfig):
        self._config = config
        self._ltp_history: list = []
        self._prev_ltp: Optional[float] = None
        
        # Cached calculator for microprice (avoid allocation in hot path)
        from features.microprice import MicropriceCalculator, MicropriceConfig
        self._microprice_calc = MicropriceCalculator(
            MicropriceConfig(tick_size=config.tick_size)
        )
    
    def calculate(
        self,
        snapshot: Snapshot,
        prev_book: Optional[PriceKeyedBook] = None,
        stats: Optional[RollingStats] = None
    ) -> FeatureResult:
        """
        Calculate LTP confirmation.
        
        This is typically used to MODIFY other signal confidence,
        not as a standalone feature.
        
        Returns:
            FeatureResult with confirmation value (+1 confirmed, -1 divergent)
        """
        if not snapshot.is_valid():
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="ltp_confirmation"
            )
        
        ltp = snapshot.ltp
        timestamp = snapshot.timestamp
        
        # Track LTP history
        self._ltp_history.append((timestamp, ltp))
        if len(self._ltp_history) > self._config.history_size:
            self._ltp_history.pop(0)
        
        # Need history for confirmation
        if len(self._ltp_history) < 2:
            self._prev_ltp = ltp
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=True,
                raw_value=0.0,
                name="ltp_confirmation"
            )
        
        # Calculate LTP trend
        ltp_trend = self._calculate_ltp_trend()
        
        # Calculate microprice deviation (signal)
        microprice_signal = self._calculate_microprice_signal(snapshot)
        
        # Calculate confirmation
        confirmation = self._calculate_confirmation(ltp_trend, microprice_signal)
        
        # Confidence based on LTP change magnitude
        ltp_change = abs(ltp - self._prev_ltp) if self._prev_ltp else 0
        ltp_change_ticks = ltp_change / self._config.tick_size
        confidence = min(1.0, ltp_change_ticks / self._config.ltp_change_threshold)
        
        self._prev_ltp = ltp
        
        return FeatureResult(
            value=confirmation,
            confidence=confidence,
            valid=True,
            raw_value=ltp_trend,
            name="ltp_confirmation"
        )
    
    def _calculate_ltp_trend(self) -> float:
        """Calculate LTP trend from history."""
        if len(self._ltp_history) < 2:
            return 0.0
        
        # Simple linear trend
        first_ltp = self._ltp_history[0][1]
        last_ltp = self._ltp_history[-1][1]
        
        if first_ltp <= 0:
            return 0.0
        
        change = (last_ltp - first_ltp) / first_ltp
        
        # Normalize
        return clamp(change * 100, -1.0, 1.0)  # % change scaled
    
    def _calculate_microprice_signal(self, snapshot: Snapshot) -> float:
        """Calculate microprice-based signal (uses cached calculator)."""
        result = self._microprice_calc.calculate(snapshot)
        return result.value
    
    def _calculate_confirmation(self, ltp_trend: float, microprice_signal: float) -> float:
        """
        Calculate confirmation value.
        
        Returns +1 if confirmed, -1 if divergent, 0 if neutral.
        """
        # Determine LTP direction
        ltp_direction = 1 if ltp_trend > 0.1 else (-1 if ltp_trend < -0.1 else 0)
        
        # Determine signal direction
        signal_direction = 1 if microprice_signal > 0.1 else (-1 if microprice_signal < -0.1 else 0)
        
        # Check for confirmation
        if signal_direction == 0 or ltp_direction == 0:
            return 0.0
        
        if signal_direction == ltp_direction:
            # Confirmed
            return 1.0
        else:
            # Divergent
            return -1.0
    
    def validate_signal(
        self,
        signal_value: float,
        snapshot: Snapshot
    ) -> Tuple[float, float]:
        """
        Validate a signal against LTP movement.
        
        Args:
            signal_value: Value of signal to validate (+ bullish, - bearish)
            snapshot: Current snapshot
        
        Returns:
            Tuple of (adjusted_signal, confidence_modifier)
        """
        if not snapshot.is_valid() or self._prev_ltp is None:
            return signal_value, 1.0
        
        ltp = snapshot.ltp
        ltp_change = ltp - self._prev_ltp
        ltp_change_ticks = ltp_change / self._config.tick_size
        
        # Determine signal direction
        signal_direction = 1 if signal_value > 0.1 else (-1 if signal_value < -0.1 else 0)
        
        if signal_direction == 0:
            return signal_value, 1.0
        
        # Check LTP direction
        ltp_direction = 1 if ltp_change_ticks > 0.5 else (-1 if ltp_change_ticks < -0.5 else 0)
        
        if ltp_direction == 0:
            return signal_value, 1.0
        
        # Check for divergence
        if signal_direction != ltp_direction:
            # Divergence detected
            # Reduce confidence and dampen signal
            return signal_value * 0.5, 1.0 - self._config.divergence_penalty
        
        # Confirmed
        return signal_value, 1.0 + 0.2 * abs(ltp_change_ticks)  # Slight boost
    
    def reset(self) -> None:
        """Reset state."""
        self._ltp_history.clear()
        self._prev_ltp = None
    
    def get_ltp_stats(self) -> dict:
        """Get LTP statistics."""
        if not self._ltp_history:
            return {'count': 0}
        
        prices = [ltp for _, ltp in self._ltp_history]
        return {
            'count': len(prices),
            'first': prices[0],
            'last': prices[-1],
            'change': prices[-1] - prices[0],
            'min': min(prices),
            'max': max(prices)
        }


def calculate_ltp_confirmation(
    snapshot: Snapshot,
    signal_value: float = 0.0
) -> Tuple[float, float]:
    """
    Calculate LTP confirmation.
    
    Returns:
        Tuple of (confirmation_value, confidence)
    """
    config = LTPConfirmationConfig()
    calc = LTPConfirmationCalculator(config)
    result = calc.calculate(snapshot)
    return result.value, result.confidence


def validate_signal_with_ltp(
    signal_value: float,
    snapshot: Snapshot,
    prev_ltp: Optional[float] = None
) -> Tuple[float, float]:
    """
    Validate signal against LTP movement.
    
    Convenience function for one-off validation.
    
    Returns:
        Tuple of (adjusted_signal, confidence_modifier)
    """
    config = LTPConfirmationConfig()
    calc = LTPConfirmationCalculator(config)
    if prev_ltp is not None:
        calc._prev_ltp = prev_ltp
    return calc.validate_signal(signal_value, snapshot)
