"""
Spread Compression feature.

Detects when spread is shrinking over time, which can indicate:
- Decreased uncertainty
- Upcoming price move direction being established
- Market maker confidence

RETAIL API LIMITATION:
We cannot see quote updates or cancellations.
This is a PROXY based on observable spread changes.
"""

from typing import Optional, Tuple
from dataclasses import dataclass

from utils.types import Snapshot, PriceKeyedBook, FeatureResult, RollingStats
from utils.math_utils import clamp
from buffers.ring_buffer import NumericRingBuffer
from utils.constants import SPREAD_COMPRESSION_RANGE


@dataclass
class SpreadCompressionConfig:
    """Configuration for spread compression calculation."""
    tick_size: float = 0.05
    window_size: int = 20  # Number of snapshots to analyze
    compression_threshold: float = 0.5  # Ratio threshold for compression
    normalize_range: float = SPREAD_COMPRESSION_RANGE


class SpreadCompressionCalculator:
    """
    Calculate spread compression feature.
    
    Tracks spread over time and detects compression (shrinking spread).
    
    Compression is calculated as:
    current_spread / average_historical_spread
    
    Values < 1.0 indicate compression (spreading tightening).
    
    Signal interpretation:
    - Compression + bullish signals = stronger bullish signal
    - Compression + bearish signals = stronger bearish signal
    - Compression alone = potential move, direction unclear
    
    IMPORTANT:
    This is a PROXY for market maker behavior.
    Real analysis would need quote update timestamps.
    """
    
    def __init__(self, config: SpreadCompressionConfig):
        self._config = config
        self._spread_buffer = NumericRingBuffer(config.window_size)
        self._initialized = False
    
    def calculate(
        self,
        snapshot: Snapshot,
        prev_book: Optional[PriceKeyedBook] = None,
        stats: Optional[RollingStats] = None
    ) -> FeatureResult:
        """
        Calculate spread compression.
        
        Args:
            snapshot: Validated market snapshot
            prev_book: Previous book (not used)
            stats: Rolling statistics (not used)
        
        Returns:
            FeatureResult with value indicating compression degree
        """
        if not snapshot.is_valid():
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="spread_compression"
            )
        
        spread = snapshot.spread
        if spread is None or spread <= 0:
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="spread_compression"
            )
        
        spread_ticks = spread / self._config.tick_size
        
        # Add to buffer
        self._spread_buffer.push(spread_ticks)
        
        # Need enough history
        if self._spread_buffer.count < 5:
            self._initialized = False
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=True,
                raw_value=1.0,  # Neutral
                name="spread_compression"
            )
        
        self._initialized = True
        
        # Calculate compression ratio
        avg_spread = self._spread_buffer.mean
        if avg_spread <= 0:
            return FeatureResult(
                value=0.0,
                confidence=0.5,
                valid=True,
                raw_value=1.0,
                name="spread_compression"
            )
        
        compression_ratio = spread_ticks / avg_spread
        
        # Normalize to [-1, +1]
        # compression_ratio < 1 = compression = bullish tendency
        # compression_ratio > 1 = expansion = bearish tendency
        
        # Map ratio to signal:
        # ratio 0.5 (50% compression) -> +1.0 (strong bullish)
        # ratio 1.0 (no change) -> 0.0 (neutral)
        # ratio 1.5 (50% expansion) -> -1.0 (strong bearish)
        
        # Linear mapping: (ratio - 1) / range
        deviation = compression_ratio - 1.0
        normalized = -deviation / self._config.compression_threshold
        normalized = clamp(normalized, -1.0, 1.0)
        
        # Confidence based on sample count and consistency
        buffer_stats = self._spread_buffer.get_stats()
        confidence = self._calculate_confidence(
            buffer_stats.count, buffer_stats.std, avg_spread
        )
        
        return FeatureResult(
            value=normalized,
            confidence=confidence,
            valid=True,
            raw_value=compression_ratio,
            name="spread_compression"
        )
    
    def _calculate_confidence(
        self,
        sample_count: int,
        spread_std: float,
        avg_spread: float
    ) -> float:
        """Calculate confidence based on sample count and spread stability."""
        # Sample count factor
        if sample_count >= self._config.window_size:
            count_factor = 1.0
        elif sample_count >= 10:
            count_factor = 0.8
        else:
            count_factor = sample_count / 10 * 0.8
        
        # Stability factor (low std = stable = high confidence)
        if avg_spread > 0:
            cv = spread_std / avg_spread  # Coefficient of variation
            if cv <= 0.1:
                stability_factor = 1.0
            elif cv <= 0.3:
                stability_factor = 0.8
            else:
                stability_factor = 0.5
        else:
            stability_factor = 0.5
        
        return count_factor * 0.5 + stability_factor * 0.5
    
    def reset(self) -> None:
        """Reset spread buffer."""
        self._spread_buffer.clear()
        self._initialized = False
    
    def get_trend(self) -> str:
        """Get current spread trend."""
        if not self._initialized:
            return "unknown"
        
        recent = list(self._spread_buffer)[-5:]
        if len(recent) < 5:
            return "insufficient_data"
        
        # Simple trend: compare first half to second half
        first_half_avg = sum(recent[:2]) / 2
        second_half_avg = sum(recent[3:]) / 2
        
        if second_half_avg < first_half_avg * 0.9:
            return "compressing"
        elif second_half_avg > first_half_avg * 1.1:
            return "expanding"
        else:
            return "stable"
    
    def get_statistics(self) -> dict:
        """Get spread statistics."""
        stats = self._spread_buffer.get_stats()
        return {
            'mean': stats.mean,
            'std': stats.std,
            'count': stats.count,
            'latest': self._spread_buffer.get_latest()
        }


def calculate_spread_compression(
    snapshot: Snapshot,
    tick_size: float = 0.05,
    window_size: int = 20
) -> Tuple[float, float]:
    """
    Calculate spread compression.
    
    Note: This creates a new calculator each time for convenience.
    For proper tracking, use SpreadCompressionCalculator class.
    
    Returns:
        Tuple of (compression_value, confidence)
    """
    config = SpreadCompressionConfig(
        tick_size=tick_size,
        window_size=window_size
    )
    calc = SpreadCompressionCalculator(config)
    result = calc.calculate(snapshot)
    return result.value, result.confidence
