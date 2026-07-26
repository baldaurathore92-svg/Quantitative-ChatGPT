"""
Utilities package for Snapshot Quant Engine.
"""

from .types import (
    Regime, State, SignalType,
    PriceLevel, DepthLevel, Snapshot, PriceKeyedBook,
    FeatureResult, CompositeScore, ExecutionSignal, StateTransition,
    RollingStats, MarketState, ValidationResult,
    FeatureCalculator, BufferProtocol, DataSource
)
from .constants import *
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
