"""
Engine package for Snapshot Quant Engine.

Core signal generation components.
"""

from .threshold import DynamicThreshold, RegimeThreshold, ThresholdConfig
from .regime import RegimeDetector, RegimeConfig, RegimeFeatureWeights
from .state_machine import TradingStateMachine, StateMachineConfig
from .confidence import (
    FeatureConfidenceCalculator, ConfidenceConfig, RegimeConfidenceModifier
)
from .composite import CompositeCalculator, CompositeConfig, calculate_composite
from .execution import ExecutionModel, ExecutionConfig
from .quant_engine import QuantEngine, SnapshotValidator

__all__ = [
    # Threshold
    'DynamicThreshold', 'RegimeThreshold', 'ThresholdConfig',
    # Regime
    'RegimeDetector', 'RegimeConfig', 'RegimeFeatureWeights',
    # State machine
    'TradingStateMachine', 'StateMachineConfig',
    # Confidence
    'FeatureConfidenceCalculator', 'ConfidenceConfig', 'RegimeConfidenceModifier',
    # Composite
    'CompositeCalculator', 'CompositeConfig', 'calculate_composite',
    # Execution
    'ExecutionModel', 'ExecutionConfig',
    # Main engine
    'QuantEngine', 'SnapshotValidator',
]
