"""
Features package for Snapshot Quant Engine.

All feature calculations for composite scoring.
"""

from .microprice import (
    MicropriceCalculator, MicropriceConfig, calculate_microprice_deviation
)
from .weighted_obi import (
    WeightedOBICalculator, OBIConfig, SimpleOBI, calculate_obi
)
from .depth_slope import (
    DepthSlopeCalculator, DepthSlopeConfig, CumulativeDepthRatio, calculate_depth_slope
)
from .spread import (
    SpreadCalculator, SpreadConfig, calculate_spread_quality
)
from .spread_compression import (
    SpreadCompressionCalculator, SpreadCompressionConfig, calculate_spread_compression
)
from .queue_persistence import (
    QueuePersistenceCalculator, QueuePersistenceConfig, calculate_queue_persistence
)
from .momentum import (
    MomentumCalculator, MomentumConfig, MomentumDerivative, calculate_momentum
)
from .acceleration import (
    AccelerationCalculator, AccelerationConfig, calculate_acceleration
)
from .refill import (
    RefillProxyCalculator, RefillConfig, calculate_refill_proxy
)
from .ltp_confirmation import (
    LTPConfirmationCalculator, LTPConfirmationConfig, 
    calculate_ltp_confirmation, validate_signal_with_ltp
)

__all__ = [
    # Microprice
    'MicropriceCalculator', 'MicropriceConfig', 'calculate_microprice_deviation',
    # Weighted OBI
    'WeightedOBICalculator', 'OBIConfig', 'SimpleOBI', 'calculate_obi',
    # Depth slope
    'DepthSlopeCalculator', 'DepthSlopeConfig', 'CumulativeDepthRatio', 'calculate_depth_slope',
    # Spread
    'SpreadCalculator', 'SpreadConfig', 'calculate_spread_quality',
    # Spread compression
    'SpreadCompressionCalculator', 'SpreadCompressionConfig', 'calculate_spread_compression',
    # Queue persistence
    'QueuePersistenceCalculator', 'QueuePersistenceConfig', 'calculate_queue_persistence',
    # Momentum
    'MomentumCalculator', 'MomentumConfig', 'MomentumDerivative', 'calculate_momentum',
    # Acceleration
    'AccelerationCalculator', 'AccelerationConfig', 'calculate_acceleration',
    # Refill proxy
    'RefillProxyCalculator', 'RefillConfig', 'calculate_refill_proxy',
    # LTP confirmation
    'LTPConfirmationCalculator', 'LTPConfirmationConfig', 
    'calculate_ltp_confirmation', 'validate_signal_with_ltp',
]
