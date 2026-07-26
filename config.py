"""
Configuration management for Snapshot Quant Engine.

All configurable parameters loaded from config.json with sensible defaults.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List
import json
import math
from pathlib import Path

from utils.constants import (
    NOISE_VOL_THRESHOLD,
    RANGE_SPREAD_THRESHOLD,
    TREND_MOMENTUM_THRESHOLD,
    TREND_OBI_THRESHOLD,
)
from utils.types import MarketSubscription


@dataclass
class APIConfig:
    """Angel One SmartAPI configuration."""
    api_key: str = ""
    auth_token: str = ""
    client_code: str = ""
    feed_token: str = ""
    heartbeat_interval: float = 10.0
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10
    snapshot_timeout: float = 30.0
    correlation_id: str = "snapquote1"


@dataclass
class TickConfig:
    """NSE tick size configuration."""
    tick_size: float = 0.05
    price_band_pct: float = 0.05  # 5% max spread validation
    
    def price_to_ticks(self, price_diff: float) -> float:
        """Convert price difference to number of ticks."""
        return abs(price_diff) / self.tick_size
    
    def ticks_to_price(self, ticks: float) -> float:
        """Convert ticks to price."""
        return ticks * self.tick_size


@dataclass
class SpreadConfig:
    """Spread calculation configuration."""
    max_spread_ticks: float = 100.0  # Max valid spread in ticks
    compression_window: int = 20  # Snapshots for spread compression detection
    compression_threshold: float = 0.5  # Spread ratio threshold


@dataclass
class BufferConfig:
    """Ring buffer configuration."""
    default_size: int = 100
    price_history_size: int = 50
    spread_history_size: int = 30
    momentum_window: int = 20


@dataclass
class ContextConfig:
    """Lifecycle limits for per-instrument mutable engine state."""

    max_active_contexts: int = 256
    idle_timeout_seconds: float = 1800.0
    timing_window_size: int = 100

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_active_contexts, bool)
            or not isinstance(self.max_active_contexts, int)
            or self.max_active_contexts <= 0
        ):
            raise ValueError("max_active_contexts must be a positive integer")
        if (
            isinstance(self.idle_timeout_seconds, bool)
            or not isinstance(self.idle_timeout_seconds, (int, float))
            or not math.isfinite(self.idle_timeout_seconds)
            or self.idle_timeout_seconds < 0
        ):
            raise ValueError(
                "idle_timeout_seconds must be finite and non-negative"
            )
        if (
            isinstance(self.timing_window_size, bool)
            or not isinstance(self.timing_window_size, int)
            or self.timing_window_size <= 0
        ):
            raise ValueError("timing_window_size must be a positive integer")


@dataclass
class EMAConfig:
    """Exponential moving average configuration."""
    short_tau: float = 5.0  # seconds
    medium_tau: float = 15.0  # seconds
    long_tau: float = 60.0  # seconds
    momentum_tau: float = 10.0  # seconds


@dataclass
class FeatureWeights:
    """Relative weights for composite scoring."""
    microprice: float = 1.0
    weighted_obi: float = 1.2
    depth_slope: float = 0.6
    spread_compression: float = 0.4
    queue_persistence: float = 0.5
    momentum: float = 0.8
    acceleration: float = 0.4
    refill_proxy: float = 0.3
    ltp_confirmation: float = 0.2


@dataclass
class RegimeWeights:
    """Feature weight multipliers per regime."""
    trend: Dict[str, float] = field(default_factory=lambda: {
        'microprice': 0.8,
        'weighted_obi': 1.2,
        'momentum': 1.3,
        'acceleration': 1.0,
    })
    pullback: Dict[str, float] = field(default_factory=lambda: {
        'microprice': 1.0,
        'weighted_obi': 1.0,
        'momentum': 0.8,
        'acceleration': 1.2,
    })
    range: Dict[str, float] = field(default_factory=lambda: {
        'microprice': 1.1,
        'weighted_obi': 0.9,
        'momentum': 0.7,
        'depth_slope': 1.0,
    })
    noise: Dict[str, float] = field(default_factory=lambda: {
        'microprice': 1.0,
        'weighted_obi': 1.0,
        'momentum': 0.8,
    })


@dataclass
class ThresholdConfig:
    """Dynamic threshold configuration."""
    base_threshold: float = 0.6
    min_threshold: float = 0.3
    max_threshold: float = 0.9
    volatility_multiplier: float = 0.2
    spread_multiplier: float = 0.15
    queue_stability_multiplier: float = 0.1
    min_samples: int = 10
    warmup_samples: int = 5
    spread_vol_window: int = 30
    obi_vol_window: int = 30
    microprice_vol_window: int = 30


@dataclass
class RegimeThresholdConfig:
    """Threshold multipliers per regime."""
    trend: float = 1.0
    pullback: float = 1.1
    range: float = 1.2
    noise: float = 1.5


@dataclass
class RegimeConfidenceConfig:
    """Confidence multipliers per regime."""
    trend: float = 1.0
    pullback: float = 0.85
    range: float = 0.80
    noise: float = 0.50


@dataclass
class RegimeDetectionConfig:
    """Market regime classifier thresholds and persistence."""

    momentum_threshold: float = TREND_MOMENTUM_THRESHOLD
    obi_threshold: float = TREND_OBI_THRESHOLD
    spread_threshold: float = RANGE_SPREAD_THRESHOLD
    vol_threshold: float = NOISE_VOL_THRESHOLD
    trend_persistence: int = 3
    noise_persistence: int = 5

    def __post_init__(self) -> None:
        for name in (
            "momentum_threshold",
            "obi_threshold",
            "spread_threshold",
            "vol_threshold",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("trend_persistence", "noise_persistence"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass
class StateMachineConfig:
    """State machine configuration."""
    warmup_samples: int = 10
    watch_timeout_seconds: float = 5.0
    cooldown_seconds: float = 3.0
    min_hold_time: float = 1.0
    signal_persistence: int = 3  # Consecutive signals required
    watch_threshold: float = 0.5
    position_threshold: float = 0.7
    exit_threshold: float = 0.3


@dataclass
class ExecutionConfig:
    """Execution model configuration."""
    slippage_ticks: float = 1.0
    max_depth_walk: int = 3  # Max levels to walk
    min_fill_ratio: float = 0.5
    execution_cost_long: float = 0.01  # % cost for long
    execution_cost_short: float = 0.01  # % cost for short


@dataclass
class ConfidenceConfig:
    """Feature confidence calculation configuration."""
    min_liquidity_qty: int = 100
    max_spread_ticks: float = 20.0
    min_queue_stability: float = 0.3
    volatility_window: int = 20
    base_confidence: float = 0.5


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "snapshot_quant.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class EngineConfig:
    """Complete engine configuration."""
    api: APIConfig = field(default_factory=APIConfig)
    tick: TickConfig = field(default_factory=TickConfig)
    spread: SpreadConfig = field(default_factory=SpreadConfig)
    buffer: BufferConfig = field(default_factory=BufferConfig)
    ema: EMAConfig = field(default_factory=EMAConfig)
    weights: FeatureWeights = field(default_factory=FeatureWeights)
    regime_weights: RegimeWeights = field(default_factory=RegimeWeights)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    regime_threshold: RegimeThresholdConfig = field(default_factory=RegimeThresholdConfig)
    regime_confidence: RegimeConfidenceConfig = field(default_factory=RegimeConfidenceConfig)
    state_machine: StateMachineConfig = field(default_factory=StateMachineConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    subscriptions: List[MarketSubscription] = field(default_factory=list)
    context: ContextConfig = field(default_factory=ContextConfig)
    regime_detection: RegimeDetectionConfig = field(
        default_factory=RegimeDetectionConfig
    )
    
    @classmethod
    def from_json(cls, path: str) -> 'EngineConfig':
        """Load configuration from JSON file."""
        config_path = Path(path)
        if not config_path.exists():
            return cls()
        
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        return cls._from_dict(data)
    
    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> 'EngineConfig':
        """Parse configuration from dictionary."""
        config = cls()
        
        if 'api' in data:
            config.api = APIConfig(**{k: v for k, v in data['api'].items() 
                                       if k in APIConfig.__dataclass_fields__})
        
        if 'tick' in data:
            config.tick = TickConfig(**{k: v for k, v in data['tick'].items()
                                          if k in TickConfig.__dataclass_fields__})
        
        if 'spread' in data:
            config.spread = SpreadConfig(**{k: v for k, v in data['spread'].items()
                                             if k in SpreadConfig.__dataclass_fields__})
        
        if 'buffer' in data:
            config.buffer = BufferConfig(**{k: v for k, v in data['buffer'].items()
                                             if k in BufferConfig.__dataclass_fields__})

        if 'context' in data:
            config.context = ContextConfig(**{
                k: v for k, v in data['context'].items()
                if k in ContextConfig.__dataclass_fields__
            })

        if 'regime_detection' in data:
            config.regime_detection = RegimeDetectionConfig(**{
                k: v for k, v in data['regime_detection'].items()
                if k in RegimeDetectionConfig.__dataclass_fields__
            })
        
        if 'ema' in data:
            config.ema = EMAConfig(**{k: v for k, v in data['ema'].items()
                                        if k in EMAConfig.__dataclass_fields__})
        
        if 'weights' in data:
            config.weights = FeatureWeights(**{k: v for k, v in data['weights'].items()
                                                 if k in FeatureWeights.__dataclass_fields__})
        
        if 'regime_weights' in data:
            rw = data['regime_weights']
            config.regime_weights = RegimeWeights(
                trend=rw.get('trend', config.regime_weights.trend),
                pullback=rw.get('pullback', config.regime_weights.pullback),
                range=rw.get('range', config.regime_weights.range),
                noise=rw.get('noise', config.regime_weights.noise),
            )
        
        if 'threshold' in data:
            config.threshold = ThresholdConfig(**{k: v for k, v in data['threshold'].items()
                                                   if k in ThresholdConfig.__dataclass_fields__})

        if 'regime_threshold' in data:
            config.regime_threshold = RegimeThresholdConfig(**{
                k: v for k, v in data['regime_threshold'].items()
                if k in RegimeThresholdConfig.__dataclass_fields__
            })

        if 'regime_confidence' in data:
            config.regime_confidence = RegimeConfidenceConfig(**{
                k: v for k, v in data['regime_confidence'].items()
                if k in RegimeConfidenceConfig.__dataclass_fields__
            })

        if 'state_machine' in data:
            config.state_machine = StateMachineConfig(**{k: v for k, v in data['state_machine'].items()
                                                          if k in StateMachineConfig.__dataclass_fields__})
        
        if 'execution' in data:
            config.execution = ExecutionConfig(**{k: v for k, v in data['execution'].items()
                                                   if k in ExecutionConfig.__dataclass_fields__})
        
        if 'confidence' in data:
            config.confidence = ConfidenceConfig(**{k: v for k, v in data['confidence'].items()
                                                     if k in ConfidenceConfig.__dataclass_fields__})
        
        if 'logging' in data:
            config.logging = LoggingConfig(**{k: v for k, v in data['logging'].items()
                                               if k in LoggingConfig.__dataclass_fields__})
        
        if 'subscriptions' in data:
            raw_subscriptions = data['subscriptions']
            if not isinstance(raw_subscriptions, list):
                raise ValueError("subscriptions must be a list")
            config.subscriptions = [
                MarketSubscription.from_config(value)
                for value in raw_subscriptions
            ]

        return config
    
    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        def dataclass_to_dict(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
            elif isinstance(obj, dict):
                return {k: dataclass_to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [dataclass_to_dict(v) for v in obj]
            return obj
        
        data = dataclass_to_dict(self)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
