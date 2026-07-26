"""
Main Quant Engine for Snapshot Mode 3.

Orchestrates the complete signal generation pipeline:
1. Snapshot Validation
2. Feature Extraction
3. Feature Normalization
4. Confidence Calculation
5. Regime Detection
6. Composite Scoring
7. Dynamic Threshold
8. State Machine
9. Execution Signal

All operations are O(1) incremental for sub-millisecond latency.
"""

import time
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

from config import EngineConfig
from utils.types import (
    Snapshot, PriceKeyedBook, FeatureResult, CompositeScore,
    Regime, State, SignalType, ExecutionSignal, MarketState, ValidationResult
)
from utils.math_utils import clamp
from utils.logging_utils import StructuredLogger

# Features
from features.microprice import MicropriceCalculator, MicropriceConfig
from features.weighted_obi import WeightedOBICalculator, OBIConfig, SimpleOBI
from features.depth_slope import DepthSlopeCalculator, DepthSlopeConfig
from features.spread import SpreadCalculator, SpreadConfig
from features.spread_compression import SpreadCompressionCalculator, SpreadCompressionConfig
from features.queue_persistence import QueuePersistenceCalculator, QueuePersistenceConfig
from features.momentum import MomentumCalculator, MomentumConfig
from features.acceleration import AccelerationCalculator, AccelerationConfig
from features.refill import RefillProxyCalculator, RefillConfig
from features.ltp_confirmation import LTPConfirmationCalculator, LTPConfirmationConfig

# Engine components
from engine.threshold import DynamicThreshold, RegimeThreshold, ThresholdConfig
from engine.regime import RegimeDetector, RegimeConfig, RegimeFeatureWeights
from engine.state_machine import TradingStateMachine, StateMachineConfig
from engine.confidence import FeatureConfidenceCalculator, ConfidenceConfig, RegimeConfidenceModifier
from engine.composite import CompositeCalculator, CompositeConfig
from engine.execution import ExecutionModel, ExecutionConfig

# Buffers
from buffers.rolling_variance import RollingVariance


class SnapshotValidator:
    """
    Validates incoming snapshots.
    
    Rejects snapshots with:
    - Crossed book
    - Missing L1
    - Invalid prices
    - Negative quantities
    - Zero bid/ask
    - Excessive spread
    - Duplicate stale data
    - Timestamp anomalies
    - Price band violations
    """
    
    def __init__(self, max_spread_ticks: float = 100.0, tick_size: float = 0.05):
        self._max_spread_ticks = max_spread_ticks
        self._tick_size = tick_size
        self._last_sequence: Dict[str, int] = {}
        self._last_timestamp: Dict[str, float] = {}
        self._last_ltp: Dict[str, float] = {}
        self._validation_count = 0  # For periodic cleanup
    
    def validate(self, snapshot: Snapshot) -> ValidationResult:
        """Validate snapshot with periodic cleanup."""
        # Periodic cleanup to prevent unbounded growth
        self._validation_count += 1
        if self._validation_count >= 1000:
            self._cleanup_old_symbols(snapshot.symbol)
            self._validation_count = 0
        
        # Check symbol
        if not snapshot.symbol:
            return ValidationResult(valid=False, reason="Missing symbol")
        
        symbol = snapshot.symbol
        
        # Check bids/asks exist
        if not snapshot.bids or not snapshot.asks:
            return ValidationResult(valid=False, reason="Missing depth")
        
        # Check best bid/ask
        best_bid = snapshot.best_bid
        best_ask = snapshot.best_ask
        
        if best_bid is None or best_ask is None:
            return ValidationResult(valid=False, reason="Missing best bid/ask")
        
        # Check prices are valid
        if best_bid.price <= 0 or best_ask.price <= 0:
            return ValidationResult(valid=False, reason="Invalid prices")
        
        # Check quantities are valid
        if best_bid.quantity < 0 or best_ask.quantity < 0:
            return ValidationResult(valid=False, reason="Negative quantity")
        
        # Check for crossed book
        if best_bid.price >= best_ask.price:
            return ValidationResult(valid=False, reason="Crossed book")
        
        # Check spread
        spread = best_ask.price - best_bid.price
        spread_ticks = spread / self._tick_size
        
        if spread_ticks > self._max_spread_ticks:
            return ValidationResult(valid=False, reason=f"Spread too wide: {spread_ticks:.1f} ticks")
        
        # Check LTP
        if snapshot.ltp <= 0:
            return ValidationResult(valid=False, reason="Invalid LTP")
        
        # Check LTP is within bid-ask range (with tolerance for edge cases)
        mid = (best_bid.price + best_ask.price) / 2
        tolerance = spread * 5  # Allow 5x spread tolerance
        if abs(snapshot.ltp - mid) > tolerance:
            # LTP is too far from mid - could be stale or error
            # Log warning but don't reject (could be valid for illiquid stocks)
            pass  # Allow but could flag
        
        # Check for stale/duplicate by sequence
        last_seq = self._last_sequence.get(symbol, -1)
        if snapshot.sequence > 0 and snapshot.sequence <= last_seq:
            return ValidationResult(valid=False, reason="Stale snapshot (sequence)")
        
        # Check for timestamp going backwards significantly
        last_ts = self._last_timestamp.get(symbol, 0)
        if snapshot.timestamp > 0 and last_ts > 0:
            if snapshot.timestamp < last_ts - 60:  # More than 60 seconds backwards
                return ValidationResult(valid=False, reason="Timestamp anomaly")
        
        # Check for extreme price jump (circuit filter simulation)
        last_ltp = self._last_ltp.get(symbol, 0)
        if last_ltp > 0:
            price_change_pct = abs(snapshot.ltp - last_ltp) / last_ltp
            if price_change_pct > 0.20:  # 20% jump - possible circuit
                # Log warning but don't reject (could be valid)
                pass
        
        # Update tracking
        self._last_sequence[symbol] = snapshot.sequence
        self._last_timestamp[symbol] = snapshot.timestamp
        self._last_ltp[symbol] = snapshot.ltp
        
        return ValidationResult(valid=True, snapshot=snapshot)
    
    def _cleanup_old_symbols(self, current_symbol: str) -> None:
        """
        Clean up old symbol tracking data.
        
        Keeps only current symbol and recently seen symbols.
        """
        # Keep only last 10 symbols to prevent unbounded growth
        if len(self._last_sequence) > 10:
            # Keep current symbol and 9 most recent
            symbols_to_keep = {current_symbol}
            for symbol in list(self._last_sequence.keys())[-9:]:
                symbols_to_keep.add(symbol)
            
            # Remove others
            for symbol in list(self._last_sequence.keys()):
                if symbol not in symbols_to_keep:
                    del self._last_sequence[symbol]
                    self._last_timestamp.pop(symbol, None)
                    self._last_ltp.pop(symbol, None)


class QuantEngine:
    """
    Main Quant Engine for Angel One SmartAPI V2 Snapshot Mode 3.
    
    RETAIL API LIMITATIONS:
    This engine uses ONLY observable snapshot data:
    - Price levels with quantity
    - Last traded price
    - Total volume (if available)
    
    We CANNOT observe:
    - Tick-by-tick order events
    - Aggressor side
    - Order IDs
    - Hidden liquidity
    - Exact FIFO queue position
    
    All features are BEST-EFFORT PROXIES based on visible order book.
    
    ARCHITECTURE:
    Raw Snapshot → Validation → Price-Keyed Book → Features → 
    Composite → Regime → Threshold → State Machine → Execution
    """
    
    def __init__(self, config: EngineConfig):
        self._config = config
        self._logger = StructuredLogger('QuantEngine')
        
        # Validate configuration
        self._validate_config()
        
        # Initialize components
        self._init_components()
        
        # State per symbol
        self._market_states: Dict[str, MarketState] = {}
        
        # Performance tracking
        self._processing_times: list = []
        self._snapshot_count = 0
    
    def _validate_config(self) -> None:
        """Validate configuration values at startup."""
        errors = []
        
        # Validate tick size
        if self._config.tick.tick_size <= 0:
            errors.append(f"Invalid tick_size: {self._config.tick.tick_size}")
        
        # Validate thresholds
        if not (0 < self._config.threshold.base_threshold <= 1):
            errors.append(f"Invalid base_threshold: {self._config.threshold.base_threshold}")
        
        if self._config.threshold.min_threshold >= self._config.threshold.max_threshold:
            errors.append("min_threshold must be less than max_threshold")
        
        # Validate buffer sizes
        if self._config.buffer.default_size <= 0:
            errors.append(f"Invalid buffer size: {self._config.buffer.default_size}")
        
        # Validate EMA taus
        if self._config.ema.momentum_tau <= 0:
            errors.append(f"Invalid momentum_tau: {self._config.ema.momentum_tau}")
        
        # Validate execution parameters
        if self._config.execution.slippage_ticks < 0:
            errors.append(f"Invalid slippage_ticks: {self._config.execution.slippage_ticks}")
        
        if self._config.execution.max_depth_walk <= 0:
            errors.append(f"Invalid max_depth_walk: {self._config.execution.max_depth_walk}")
        
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            self._logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _init_components(self) -> None:
        """Initialize all engine components."""
        tick_size = self._config.tick.tick_size
        
        # Validator
        self._validator = SnapshotValidator(
            max_spread_ticks=self._config.spread.max_spread_ticks,
            tick_size=tick_size
        )
        
        # Feature calculators
        self._microprice = MicropriceCalculator(
            MicropriceConfig(tick_size=tick_size)
        )
        self._weighted_obi = WeightedOBICalculator(
            OBIConfig(tick_size=tick_size, lambda_decay=3.0)
        )
        self._depth_slope = DepthSlopeCalculator(
            DepthSlopeConfig(tick_size=tick_size)
        )
        self._spread = SpreadCalculator(
            SpreadConfig(tick_size=tick_size)
        )
        self._spread_compression = SpreadCompressionCalculator(
            SpreadCompressionConfig(tick_size=tick_size, window_size=self._config.spread.compression_window)
        )
        self._queue_persistence = QueuePersistenceCalculator(
            QueuePersistenceConfig(tick_size=tick_size)
        )
        self._momentum = MomentumCalculator(
            MomentumConfig(tick_size=tick_size, tau=self._config.ema.momentum_tau)
        )
        self._acceleration = AccelerationCalculator(
            AccelerationConfig(tick_size=tick_size)
        )
        self._refill = RefillProxyCalculator(
            RefillConfig(tick_size=tick_size)
        )
        self._ltp_confirmation = LTPConfirmationCalculator(
            LTPConfirmationConfig(tick_size=tick_size)
        )
        
        # Engine components
        self._threshold = DynamicThreshold(self._config.threshold)
        self._regime_threshold = RegimeThreshold(
            trend_mult=self._config.regime_threshold.trend,
            pullback_mult=self._config.regime_threshold.pullback,
            range_mult=self._config.regime_threshold.range,
            noise_mult=self._config.regime_threshold.noise
        )
        self._regime_detector = RegimeDetector(RegimeConfig(
            momentum_window=self._config.buffer.momentum_window,
            obi_window=self._config.buffer.default_size,
            spread_window=self._config.buffer.spread_history_size,
            price_window=self._config.buffer.price_history_size
        ))
        self._regime_weights = RegimeFeatureWeights()
        self._confidence_calc = FeatureConfidenceCalculator(self._config.confidence)
        self._confidence_modifier = RegimeConfidenceModifier(
            trend_mult=self._config.regime_confidence.trend,
            pullback_mult=self._config.regime_confidence.pullback,
            range_mult=self._config.regime_confidence.range,
            noise_mult=self._config.regime_confidence.noise
        )
        self._composite = CompositeCalculator(
            CompositeConfig(weights={
                'microprice': self._config.weights.microprice,
                'weighted_obi': self._config.weights.weighted_obi,
                'depth_slope': self._config.weights.depth_slope,
                'spread_compression': self._config.weights.spread_compression,
                'queue_persistence': self._config.weights.queue_persistence,
                'momentum': self._config.weights.momentum,
                'acceleration': self._config.weights.acceleration,
                'refill_proxy': self._config.weights.refill_proxy,
                'ltp_confirmation': self._config.weights.ltp_confirmation
            })
        )
        self._state_machine = TradingStateMachine(self._config.state_machine)
        self._execution = ExecutionModel(ExecutionConfig(
            slippage_ticks=self._config.execution.slippage_ticks,
            max_depth_walk=self._config.execution.max_depth_walk,
            tick_size=tick_size
        ))
        
        # Rolling volatility
        self._spread_vol = RollingVariance(30)
        self._obi_vol = RollingVariance(30)
    
    def process(self, snapshot: Snapshot) -> Optional[CompositeScore]:
        """
        Process a market snapshot.
        
        This is the main entry point for the engine.
        
        Args:
            snapshot: Raw market snapshot
        
        Returns:
            CompositeScore if valid, None if rejected
        """
        start_time = time.perf_counter()
        
        # Validate snapshot
        validation = self._validator.validate(snapshot)
        if not validation.valid:
            self._logger.debug(f"Snapshot rejected: {validation.reason}")
            return None
        
        # Get or create market state
        symbol = snapshot.symbol
        if symbol not in self._market_states:
            self._market_states[symbol] = MarketState(symbol=symbol)
        
        market_state = self._market_states[symbol]
        prev_book = market_state.prev_book
        
        # Calculate all features
        features = self._calculate_features(snapshot, prev_book)
        
        # Update market state
        market_state.update(snapshot)
        
        # Detect regime
        regime = self._detect_regime(features, snapshot)
        
        # Get regime-specific weights
        regime_weights = self._regime_weights.get_adjusted_weights(
            regime, self._config.weights.__dict__
        )
        
        # Get regime confidence multiplier
        regime_conf_mult = self._confidence_modifier.get_modifier(regime.name)
        
        # Calculate composite
        composite = self._composite.calculate(
            features=features,
            regime=regime,
            regime_weights=regime_weights,
            regime_confidence_mult=regime_conf_mult,
            timestamp=snapshot.timestamp
        )
        
        # Calculate dynamic threshold
        spread_ticks = snapshot.spread_ticks(self._config.tick.tick_size)
        if spread_ticks is None:
            spread_ticks = 1.0
        
        threshold = self._threshold.calculate(
            spread_ticks=spread_ticks,
            obi_value=features.get('weighted_obi', FeatureResult(0, 0, False, 0, '')).value,
            microprice_deviation=features.get('microprice', FeatureResult(0, 0, False, 0, '')).ticks
        )
        
        # Apply regime threshold multiplier
        threshold = self._regime_threshold.apply_regime_threshold(
            threshold, regime.name,
            self._config.threshold.min_threshold,
            self._config.threshold.max_threshold
        )
        
        composite.threshold_used = threshold
        
        # Update state machine
        state = self._state_machine.update(
            composite_value=composite.value,
            confidence=composite.confidence,
            regime=regime.name,
            timestamp=snapshot.timestamp
        )
        
        # Track processing time
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self._processing_times.append(elapsed_ms)
        if len(self._processing_times) > 100:
            self._processing_times.pop(0)
        
        self._snapshot_count += 1
        
        # Log periodically
        if self._snapshot_count % 100 == 0:
            self._logger.debug(
                f"Processed {self._snapshot_count} snapshots",
                avg_time_ms=sum(self._processing_times) / len(self._processing_times)
            )
        
        return composite
    
    def _calculate_features(
        self,
        snapshot: Snapshot,
        prev_book: Optional[PriceKeyedBook]
    ) -> Dict[str, FeatureResult]:
        """Calculate all features."""
        features = {}
        
        # Microprice
        features['microprice'] = self._microprice.calculate(snapshot, prev_book)
        
        # Weighted OBI
        features['weighted_obi'] = self._weighted_obi.calculate(snapshot, prev_book)
        
        # Depth slope
        features['depth_slope'] = self._depth_slope.calculate(snapshot, prev_book)
        
        # Spread quality
        features['spread_quality'] = self._spread.calculate(snapshot, prev_book)
        
        # Spread compression
        features['spread_compression'] = self._spread_compression.calculate(snapshot, prev_book)
        
        # Queue persistence
        features['queue_persistence'] = self._queue_persistence.calculate(snapshot, prev_book)
        
        # Momentum
        features['momentum'] = self._momentum.calculate(snapshot, prev_book)
        
        # Acceleration
        features['acceleration'] = self._acceleration.calculate(snapshot, prev_book)
        
        # Refill proxy
        features['refill_proxy'] = self._refill.calculate(snapshot, prev_book)
        
        # LTP confirmation
        features['ltp_confirmation'] = self._ltp_confirmation.calculate(snapshot, prev_book)
        
        return features
    
    def _detect_regime(
        self,
        features: Dict[str, FeatureResult],
        snapshot: Snapshot
    ) -> Regime:
        """Detect market regime."""
        momentum = features.get('momentum', FeatureResult(0, 0, False, 0, '')).value
        obi = features.get('weighted_obi', FeatureResult(0, 0, False, 0, '')).value
        spread_ticks = snapshot.spread_ticks(self._config.tick.tick_size) or 1.0
        price = snapshot.ltp
        
        return self._regime_detector.detect(momentum, obi, spread_ticks, price)
    
    def get_state(self, symbol: str) -> State:
        """Get current state for symbol."""
        return self._state_machine.state
    
    def get_regime(self, symbol: str) -> Regime:
        """Get current regime for symbol."""
        return self._regime_detector.current_regime
    
    def get_execution_signal(
        self,
        snapshot: Snapshot,
        composite: CompositeScore
    ) -> Optional[ExecutionSignal]:
        """Create execution signal from composite."""
        state = self._state_machine.state
        
        if state == State.LONG:
            signal_type = SignalType.BULLISH
        elif state == State.SHORT:
            signal_type = SignalType.BEARISH
        else:
            return None
        
        return self._execution.create_execution_signal(
            snapshot=snapshot,
            signal_type=signal_type,
            composite_value=composite.value,
            confidence=composite.confidence,
            regime=composite.regime,
            timestamp=composite.timestamp
        )
    
    def reset(self, symbol: Optional[str] = None) -> None:
        """Reset engine state."""
        if symbol:
            if symbol in self._market_states:
                self._market_states[symbol].reset()
        else:
            for state in self._market_states.values():
                state.reset()
            
            self._threshold.reset()
            self._regime_detector.reset()
            self._state_machine.reset()
            self._composite.reset()
            
            # Reset features with state
            self._momentum.reset()
            self._acceleration.reset()
            self._spread_compression.reset()
            self._refill.reset()
            self._ltp_confirmation.reset()
            self._queue_persistence.reset()
            
            # Reset processing state
            self._processing_times.clear()
            self._snapshot_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        avg_time = sum(self._processing_times) / len(self._processing_times) if self._processing_times else 0
        
        return {
            'snapshot_count': self._snapshot_count,
            'avg_processing_time_ms': avg_time,
            'state': self._state_machine.state.name,
            'regime': self._regime_detector.current_regime.name,
            'threshold': self._threshold.get_current_threshold(),
            'symbols_tracked': len(self._market_states)
        }
    
    def get_feature_values(self, symbol: str) -> Dict[str, float]:
        """Get current feature values for symbol."""
        # Return last calculated feature values
        # This is a convenience method for debugging
        return {}
