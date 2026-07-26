"""
Utilities package for Snapshot Quant Engine.
"""

from .types import (
    Regime, State, SignalType,
    PriceLevel, DepthLevel, Snapshot, PriceKeyedBook,
    FeatureResult, CompositeScore, ExecutionSignal, StateTransition,
    RollingStats, MarketState, ValidationResult,
    FeatureCalculator, BufferProtocol, DataSource,
    MarketSubscription, SnapshotDeliveryMode
)
from .constants import (
    EPSILON,
    HIGH_CONFIDENCE,
    LOW_CONFIDENCE,
    MAX_CONFIDENCE,
    MAX_SPREAD_RATIO,
    MEDIUM_CONFIDENCE,
    MIN_CONFIDENCE,
    MIN_PRICE,
    MIN_QUANTITY,
    NSE_CASH_TICK,
    NSE_FNO_TICK,
)
from .math_utils import (
    clamp, normalize_to_range, safe_divide,
    exponential_decay_weight, time_aware_ema_alpha, update_ema,
    weighted_mid_price, microprice, linear_regression_slope,
    sign, ticks_from_mid, is_price_better, spread_to_ticks,
    volatility_from_returns, queue_imbalance_ratio, decayed_sum
)
from .logging_utils import (
    StructuredLogger, setup_logging, log_performance, PerformanceTracker
)

__all__ = [
    # Types
    'Regime', 'State', 'SignalType',
    'PriceLevel', 'DepthLevel', 'Snapshot', 'PriceKeyedBook',
    'FeatureResult', 'CompositeScore', 'ExecutionSignal', 'StateTransition',
    'RollingStats', 'MarketState', 'ValidationResult',
    'FeatureCalculator', 'BufferProtocol', 'DataSource',
    'MarketSubscription', 'SnapshotDeliveryMode',
    # Constants
    'NSE_CASH_TICK', 'NSE_FNO_TICK', 'MAX_SPREAD_RATIO',
    'MIN_CONFIDENCE', 'MAX_CONFIDENCE', 'HIGH_CONFIDENCE',
    'MEDIUM_CONFIDENCE', 'LOW_CONFIDENCE',
    'EPSILON', 'MIN_PRICE', 'MIN_QUANTITY',
    # Math utils
    'clamp', 'normalize_to_range', 'safe_divide',
    'exponential_decay_weight', 'time_aware_ema_alpha', 'update_ema',
    'weighted_mid_price', 'microprice', 'linear_regression_slope',
    'sign', 'ticks_from_mid', 'is_price_better', 'spread_to_ticks',
    'volatility_from_returns', 'queue_imbalance_ratio', 'decayed_sum',
    # Logging
    'StructuredLogger', 'setup_logging', 'log_performance', 'PerformanceTracker',
]
