"""
Buffers package for Snapshot Quant Engine.

All O(1) rolling statistics implementations.
"""

from .ring_buffer import RingBuffer, NumericRingBuffer, BufferStats
from .rolling_mean import RollingMean, TimeAwareRollingMean, RollingMeanResult
from .rolling_variance import RollingVariance, WelfordVariance, RollingVarianceResult
from .rolling_ema import RollingEMA, DualEMA, TripleEMA, EMAResult
from .monotonic_queue import (
    MonotonicQueue, RollingMinMax, RollingPercentile, MinMaxResult
)

__all__ = [
    # Ring buffer
    'RingBuffer', 'NumericRingBuffer', 'BufferStats',
    # Rolling mean
    'RollingMean', 'TimeAwareRollingMean', 'RollingMeanResult',
    # Rolling variance
    'RollingVariance', 'WelfordVariance', 'RollingVarianceResult',
    # Rolling EMA
    'RollingEMA', 'DualEMA', 'TripleEMA', 'EMAResult',
    # Monotonic queue
    'MonotonicQueue', 'RollingMinMax', 'RollingPercentile', 'MinMaxResult',
]
