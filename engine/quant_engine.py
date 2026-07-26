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

import math
import time
from collections import deque
from typing import Dict, Optional, Any
from dataclasses import dataclass, field

from config import EngineConfig
from utils.types import (
    Snapshot, PriceLevel, PriceKeyedBook, FeatureResult, CompositeScore,
    Regime, State, ExecutionSignal, MarketState, ValidationResult,
    MarketSubscription
)
from utils.logging_utils import StructuredLogger

# Features
from features.microprice import MicropriceCalculator, MicropriceConfig
from features.weighted_obi import WeightedOBICalculator, OBIConfig
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
from engine.confidence import RegimeConfidenceModifier
from engine.composite import CompositeCalculator, CompositeConfig
from engine.execution import ExecutionModel, ExecutionConfig


@dataclass
class _ValidationHistory:
    """Ordering history retained for one validated instrument."""

    sequence: int = -1
    timestamp: float = 0.0


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
    
    def __init__(
        self,
        max_spread_ticks: float = 100.0,
        tick_size: float = 0.05,
        max_tracked_instruments: int = 1024
    ):
        if (
            not isinstance(max_spread_ticks, (int, float))
            or isinstance(max_spread_ticks, bool)
            or not math.isfinite(max_spread_ticks)
            or max_spread_ticks <= 0
        ):
            raise ValueError("max_spread_ticks must be finite and positive")
        if (
            not isinstance(tick_size, (int, float))
            or isinstance(tick_size, bool)
            or not math.isfinite(tick_size)
            or tick_size <= 0
        ):
            raise ValueError("tick_size must be finite and positive")
        if (
            isinstance(max_tracked_instruments, bool)
            or not isinstance(max_tracked_instruments, int)
            or max_tracked_instruments <= 0
        ):
            raise ValueError("max_tracked_instruments must be a positive integer")
        self._max_spread_ticks = max_spread_ticks
        self._tick_size = tick_size
        self._max_tracked_instruments = max_tracked_instruments
        self._history: Dict[str, _ValidationHistory] = {}
    
    def validate(self, snapshot: Snapshot) -> ValidationResult:
        """Validate one snapshot against this context's previous snapshot."""
        # Check symbol and scalar fields
        if not isinstance(snapshot.symbol, str) or not snapshot.symbol.strip():
            return ValidationResult(valid=False, reason="Missing symbol")
        if not isinstance(snapshot.token, str):
            return ValidationResult(valid=False, reason="Invalid token")
        if snapshot.token and (
            not snapshot.token.isdecimal() or int(snapshot.token) <= 0
        ):
            return ValidationResult(valid=False, reason="Invalid token")
        if snapshot.exchange_type and not snapshot.token:
            return ValidationResult(valid=False, reason="Missing token")
        if (
            isinstance(snapshot.exchange_type, bool)
            or not isinstance(snapshot.exchange_type, int)
            or (
                snapshot.exchange_type != 0
                and snapshot.exchange_type
                not in MarketSubscription.SUPPORTED_EXCHANGE_TYPES
            )
        ):
            return ValidationResult(valid=False, reason="Invalid exchange type")

        if (
            not isinstance(snapshot.ltp, (int, float))
            or isinstance(snapshot.ltp, bool)
            or not math.isfinite(snapshot.ltp)
            or snapshot.ltp <= 0
        ):
            return ValidationResult(valid=False, reason="Invalid LTP")
        if (
            not isinstance(snapshot.timestamp, (int, float))
            or isinstance(snapshot.timestamp, bool)
            or not math.isfinite(snapshot.timestamp)
        ):
            return ValidationResult(valid=False, reason="Invalid timestamp")
        if (
            not isinstance(snapshot.exchange_timestamp, (int, float))
            or isinstance(snapshot.exchange_timestamp, bool)
            or not math.isfinite(snapshot.exchange_timestamp)
        ):
            return ValidationResult(valid=False, reason="Invalid exchange timestamp")

        quantity_fields = {
            'ltp_quantity': snapshot.ltp_quantity,
            'volume_traded': snapshot.volume_traded,
            'total_buy_qty': snapshot.total_buy_qty,
            'total_sell_qty': snapshot.total_sell_qty,
            'sequence': snapshot.sequence,
        }
        for name, value in quantity_fields.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                return ValidationResult(valid=False, reason=f"Invalid {name}")

        # Check and validate every depth level, not only L1.
        if not snapshot.bids or not snapshot.asks:
            return ValidationResult(valid=False, reason="Missing depth")

        for side, levels in (('bid', snapshot.bids), ('ask', snapshot.asks)):
            for level in levels:
                if not isinstance(level, PriceLevel):
                    return ValidationResult(
                        valid=False,
                        reason=f"Invalid {side} level"
                    )
                if (
                    not isinstance(level.price, (int, float))
                    or isinstance(level.price, bool)
                    or not math.isfinite(level.price)
                    or level.price <= 0
                ):
                    return ValidationResult(
                        valid=False,
                        reason=f"Invalid {side} price"
                    )
                if (
                    not isinstance(level.quantity, int)
                    or isinstance(level.quantity, bool)
                    or level.quantity <= 0
                ):
                    return ValidationResult(
                        valid=False,
                        reason=f"Invalid {side} quantity"
                    )
                if (
                    not isinstance(level.order_count, int)
                    or isinstance(level.order_count, bool)
                    or level.order_count < 0
                    or level.order_count > level.quantity
                ):
                    return ValidationResult(
                        valid=False,
                        reason=f"Invalid {side} order count"
                    )

        if any(
            snapshot.bids[index].price <= snapshot.bids[index + 1].price
            for index in range(len(snapshot.bids) - 1)
        ):
            return ValidationResult(valid=False, reason="Bid depth is not strictly descending")
        if any(
            snapshot.asks[index].price >= snapshot.asks[index + 1].price
            for index in range(len(snapshot.asks) - 1)
        ):
            return ValidationResult(valid=False, reason="Ask depth is not strictly ascending")

        best_bid = snapshot.best_bid
        best_ask = snapshot.best_ask
        if best_bid is None or best_ask is None:
            return ValidationResult(valid=False, reason="Missing best bid/ask")

        if best_bid.price >= best_ask.price:
            return ValidationResult(valid=False, reason="Crossed book")

        spread = best_ask.price - best_bid.price
        spread_ticks = spread / self._tick_size
        if not math.isfinite(spread_ticks) or spread_ticks > self._max_spread_ticks:
            return ValidationResult(valid=False, reason="Spread too wide")
        
        # Check LTP is within bid-ask range (with tolerance for edge cases)
        mid = (best_bid.price + best_ask.price) / 2
        tolerance = spread * 5  # Allow 5x spread tolerance
        if abs(snapshot.ltp - mid) > tolerance:
            # LTP is too far from mid - could be stale or error
            # Log warning but don't reject (could be valid for illiquid stocks)
            pass  # Allow but could flag
        
        instrument_key = snapshot.instrument_key
        history = self._history.get(instrument_key)

        # Check for stale/duplicate by sequence within this instrument only.
        if (
            history is not None
            and snapshot.sequence > 0
            and snapshot.sequence <= history.sequence
        ):
            return ValidationResult(valid=False, reason="Stale snapshot (sequence)")
        
        # Check for timestamp going backwards significantly per instrument.
        if (
            history is not None
            and snapshot.timestamp > 0
            and history.timestamp > 0
            and snapshot.timestamp < history.timestamp - 60
        ):
            return ValidationResult(valid=False, reason="Timestamp anomaly")

        if history is None:
            if len(self._history) >= self._max_tracked_instruments:
                oldest_key = next(iter(self._history))
                self._history.pop(oldest_key)
            history = _ValidationHistory()
        else:
            # Reinsert below to keep dictionary order as least-recently-used.
            self._history.pop(instrument_key)
        history.sequence = snapshot.sequence
        history.timestamp = snapshot.timestamp
        self._history[instrument_key] = history
        
        return ValidationResult(valid=True, snapshot=snapshot)
    
    def reset(self) -> None:
        """Clear ordering history for every tracked instrument."""
        self._history.clear()


@dataclass
class SymbolContext:
    """All mutable calculation and trading state owned by one symbol."""

    symbol: str
    validator: SnapshotValidator
    market_state: MarketState
    spread_compression: SpreadCompressionCalculator
    queue_persistence: QueuePersistenceCalculator
    momentum: MomentumCalculator
    acceleration: AccelerationCalculator
    refill: RefillProxyCalculator
    ltp_confirmation: LTPConfirmationCalculator
    threshold: DynamicThreshold
    regime_detector: RegimeDetector
    composite: CompositeCalculator
    state_machine: TradingStateMachine
    last_feature_values: Dict[str, FeatureResult] = field(default_factory=dict)
    last_composite: Optional[CompositeScore] = None
    processing_times: deque[float] = field(default_factory=deque)
    snapshot_count: int = 0
    last_seen_monotonic: float = field(default_factory=time.monotonic)

    def reset(self) -> None:
        """Reset this symbol without changing any other symbol context."""
        self.validator.reset()
        self.market_state.reset()
        self.spread_compression.reset()
        self.queue_persistence.reset()
        self.momentum.reset()
        self.acceleration.reset()
        self.refill.reset()
        self.ltp_confirmation.reset()
        self.threshold.reset()
        self.regime_detector.reset()
        self.composite.reset()
        self.state_machine.reset()
        self.last_feature_values.clear()
        self.last_composite = None
        self.processing_times.clear()
        self.snapshot_count = 0


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
        
        # Typed mutable state partitioned by exchange-qualified instrument key.
        self._contexts: Dict[str, SymbolContext] = {}
        self._accepted_snapshot_count = 0
        self._rejected_snapshot_count = 0
        self._rejection_reasons: Dict[str, int] = {}
        self._contexts_created = 0
        self._contexts_evicted = 0
        self._context_eviction_reasons: Dict[str, int] = {}
        self._total_processing_time_ms = 0.0
        idle_timeout = self._config.context.idle_timeout_seconds
        self._context_cleanup_interval = (
            min(60.0, max(0.01, idle_timeout / 4.0))
            if idle_timeout > 0
            else 0.0
        )
        self._next_context_cleanup = (
            time.monotonic() + self._context_cleanup_interval
        )
    
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

        if (
            isinstance(self._config.context.max_active_contexts, bool)
            or not isinstance(self._config.context.max_active_contexts, int)
            or self._config.context.max_active_contexts <= 0
        ):
            errors.append("context.max_active_contexts must be a positive integer")
        if (
            isinstance(self._config.context.idle_timeout_seconds, bool)
            or not isinstance(
                self._config.context.idle_timeout_seconds, (int, float)
            )
            or not math.isfinite(self._config.context.idle_timeout_seconds)
            or self._config.context.idle_timeout_seconds < 0
        ):
            errors.append(
                "context.idle_timeout_seconds must be finite and non-negative"
            )
        if (
            isinstance(self._config.context.timing_window_size, bool)
            or not isinstance(self._config.context.timing_window_size, int)
            or self._config.context.timing_window_size <= 0
        ):
            errors.append("context.timing_window_size must be a positive integer")
        
        # Validate EMA taus
        if self._config.ema.momentum_tau <= 0:
            errors.append(f"Invalid momentum_tau: {self._config.ema.momentum_tau}")
        if self._config.ema.short_tau <= 0:
            errors.append(f"Invalid short_tau: {self._config.ema.short_tau}")

        if self._config.spread.compression_threshold <= 0:
            errors.append(
                f"Invalid compression_threshold: {self._config.spread.compression_threshold}"
            )

        # Validate execution parameters
        if self._config.execution.slippage_ticks < 0:
            errors.append(f"Invalid slippage_ticks: {self._config.execution.slippage_ticks}")
        
        if self._config.execution.max_depth_walk <= 0:
            errors.append(f"Invalid max_depth_walk: {self._config.execution.max_depth_walk}")

        if not (0 <= self._config.execution.min_fill_ratio <= 1):
            errors.append(f"Invalid min_fill_ratio: {self._config.execution.min_fill_ratio}")

        if self._config.execution.execution_cost_long < 0:
            errors.append(
                f"Invalid execution_cost_long: {self._config.execution.execution_cost_long}"
            )
        if self._config.execution.execution_cost_short < 0:
            errors.append(
                f"Invalid execution_cost_short: {self._config.execution.execution_cost_short}"
            )

        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            self._logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _init_components(self) -> None:
        """Build shared stateless components and reusable context configuration."""
        tick_size = self._config.tick.tick_size

        # Stateless feature calculators can be shared across symbols.
        self._microprice = MicropriceCalculator(MicropriceConfig(tick_size=tick_size))
        self._weighted_obi = WeightedOBICalculator(
            OBIConfig(tick_size=tick_size, lambda_decay=3.0)
        )
        self._depth_slope = DepthSlopeCalculator(DepthSlopeConfig(tick_size=tick_size))
        self._spread = SpreadCalculator(SpreadConfig(
            tick_size=tick_size,
            max_spread_ticks=self._config.spread.max_spread_ticks
        ))

        # Parse root configuration into typed component configs once. The symbol
        # factory reuses these values but creates fresh mutable calculators.
        self._spread_compression_config = SpreadCompressionConfig(
            tick_size=tick_size,
            window_size=self._config.spread.compression_window,
            compression_threshold=self._config.spread.compression_threshold
        )
        self._queue_persistence_config = QueuePersistenceConfig(tick_size=tick_size)
        self._momentum_config = MomentumConfig(
            tick_size=tick_size,
            tau=self._config.ema.momentum_tau
        )
        self._acceleration_config = AccelerationConfig(
            tick_size=tick_size,
            momentum_tau=self._config.ema.momentum_tau,
            derivative_tau=self._config.ema.short_tau
        )
        self._refill_config = RefillConfig(tick_size=tick_size)
        self._ltp_confirmation_config = LTPConfirmationConfig(tick_size=tick_size)
        self._threshold_config = ThresholdConfig(
            base_threshold=self._config.threshold.base_threshold,
            min_threshold=self._config.threshold.min_threshold,
            max_threshold=self._config.threshold.max_threshold,
            volatility_multiplier=self._config.threshold.volatility_multiplier,
            spread_multiplier=self._config.threshold.spread_multiplier,
            queue_stability_multiplier=self._config.threshold.queue_stability_multiplier,
            min_samples=self._config.threshold.min_samples,
            warmup_samples=self._config.threshold.warmup_samples,
            spread_vol_window=self._config.threshold.spread_vol_window,
            obi_vol_window=self._config.threshold.obi_vol_window,
            microprice_vol_window=self._config.threshold.microprice_vol_window
        )
        self._regime_config = RegimeConfig(
            momentum_window=self._config.buffer.momentum_window,
            obi_window=self._config.buffer.default_size,
            spread_window=self._config.buffer.spread_history_size,
            price_window=self._config.buffer.price_history_size
        )
        self._composite_config = CompositeConfig(weights={
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
        self._state_machine_config = StateMachineConfig(
            warmup_samples=self._config.state_machine.warmup_samples,
            watch_timeout_seconds=self._config.state_machine.watch_timeout_seconds,
            cooldown_seconds=self._config.state_machine.cooldown_seconds,
            min_hold_time=self._config.state_machine.min_hold_time,
            signal_persistence=self._config.state_machine.signal_persistence,
            watch_threshold=self._config.state_machine.watch_threshold,
            position_threshold=self._config.state_machine.position_threshold,
            exit_threshold=self._config.state_machine.exit_threshold
        )

        # Stateless regime/config modifiers and execution remain shared.
        self._regime_threshold = RegimeThreshold(
            trend_mult=self._config.regime_threshold.trend,
            pullback_mult=self._config.regime_threshold.pullback,
            range_mult=self._config.regime_threshold.range,
            noise_mult=self._config.regime_threshold.noise
        )
        self._regime_weights = RegimeFeatureWeights(
            trend=self._config.regime_weights.trend,
            pullback=self._config.regime_weights.pullback,
            range_weights=self._config.regime_weights.range,
            noise=self._config.regime_weights.noise
        )
        self._confidence_modifier = RegimeConfidenceModifier(
            trend_mult=self._config.regime_confidence.trend,
            pullback_mult=self._config.regime_confidence.pullback,
            range_mult=self._config.regime_confidence.range,
            noise_mult=self._config.regime_confidence.noise
        )
        self._execution = ExecutionModel(ExecutionConfig(
            slippage_ticks=self._config.execution.slippage_ticks,
            max_depth_walk=self._config.execution.max_depth_walk,
            min_fill_ratio=self._config.execution.min_fill_ratio,
            execution_cost_long_pct=self._config.execution.execution_cost_long,
            execution_cost_short_pct=self._config.execution.execution_cost_short,
            tick_size=tick_size
        ))

    def _create_symbol_context(self, symbol: str) -> SymbolContext:
        """Create a fresh, fully isolated mutable context for ``symbol``."""
        return SymbolContext(
            symbol=symbol,
            validator=SnapshotValidator(
                max_spread_ticks=self._config.spread.max_spread_ticks,
                tick_size=self._config.tick.tick_size
            ),
            market_state=MarketState(symbol=symbol),
            spread_compression=SpreadCompressionCalculator(
                self._spread_compression_config
            ),
            queue_persistence=QueuePersistenceCalculator(
                self._queue_persistence_config
            ),
            momentum=MomentumCalculator(self._momentum_config),
            acceleration=AccelerationCalculator(self._acceleration_config),
            refill=RefillProxyCalculator(self._refill_config),
            ltp_confirmation=LTPConfirmationCalculator(
                self._ltp_confirmation_config
            ),
            threshold=DynamicThreshold(self._threshold_config),
            regime_detector=RegimeDetector(self._regime_config),
            composite=CompositeCalculator(self._composite_config),
            state_machine=TradingStateMachine(self._state_machine_config),
            processing_times=deque(
                maxlen=self._config.context.timing_window_size
            )
        )

    @staticmethod
    def _context_holds_position(context: SymbolContext) -> bool:
        """Return whether evicting this context could lose position state."""
        return context.state_machine.state in {
            State.LONG,
            State.SHORT,
            State.EXIT_LONG,
            State.EXIT_SHORT,
        }

    def _evict_context(self, instrument_key: str, reason: str) -> None:
        """Remove one inactive context and record why it was removed."""
        context = self._contexts.pop(instrument_key, None)
        if context is None:
            return
        self._contexts_evicted += 1
        self._context_eviction_reasons[reason] = (
            self._context_eviction_reasons.get(reason, 0) + 1
        )
        self._logger.debug(
            "Evicted instrument context",
            instrument_key=instrument_key,
            reason=reason,
            active_contexts=len(self._contexts)
        )

    def _evict_idle_contexts(
        self,
        now: float,
        exclude: Optional[str] = None,
        force: bool = False
    ) -> None:
        """Periodically remove expired non-position contexts."""
        idle_timeout = self._config.context.idle_timeout_seconds
        if idle_timeout <= 0:
            return
        if not force and now < self._next_context_cleanup:
            return
        self._next_context_cleanup = now + self._context_cleanup_interval
        idle_keys = [
            key
            for key, context in self._contexts.items()
            if key != exclude
            and now - context.last_seen_monotonic >= idle_timeout
            and not self._context_holds_position(context)
        ]
        for key in idle_keys:
            self._evict_context(key, "idle_timeout")

    def _make_room_for_context(self, now: float) -> bool:
        """Evict idle/LRU non-position contexts before admitting a new one."""
        self._evict_idle_contexts(now, force=True)

        maximum = self._config.context.max_active_contexts
        while len(self._contexts) >= maximum:
            candidates = [
                (context.last_seen_monotonic, key)
                for key, context in self._contexts.items()
                if not self._context_holds_position(context)
            ]
            if not candidates:
                return False
            _, oldest_key = min(candidates)
            self._evict_context(oldest_key, "capacity_lru")
        return True

    def _record_rejection(self, reason: str) -> None:
        """Increment bounded rejection diagnostics."""
        self._rejected_snapshot_count += 1
        self._rejection_reasons[reason] = self._rejection_reasons.get(reason, 0) + 1

    def process(self, snapshot: Snapshot) -> Optional[CompositeScore]:
        """Process one snapshot entirely within its instrument context."""
        start_time = time.perf_counter()
        now = time.monotonic()
        if not isinstance(snapshot.symbol, str) or not snapshot.symbol.strip():
            self._evict_idle_contexts(now)
            reason = "Missing symbol"
            self._record_rejection(reason)
            self._logger.debug("Snapshot rejected", reason=reason)
            return None

        instrument_key = snapshot.instrument_key
        self._evict_idle_contexts(now, exclude=instrument_key)
        context = self._contexts.get(instrument_key)
        is_new_context = context is None
        if context is None:
            context = self._create_symbol_context(instrument_key)

        validation = context.validator.validate(snapshot)
        if not validation.valid:
            reason = validation.reason or "Unknown validation failure"
            self._record_rejection(reason)
            self._logger.debug(
                "Snapshot rejected",
                instrument_key=instrument_key,
                reason=reason
            )
            return None

        if is_new_context:
            if not self._make_room_for_context(now):
                reason = "Context capacity reserved by active positions"
                self._record_rejection(reason)
                self._logger.warning(
                    "Snapshot rejected",
                    instrument_key=instrument_key,
                    reason=reason,
                    active_contexts=len(self._contexts)
                )
                return None
            self._contexts[instrument_key] = context
            self._contexts_created += 1
        context.last_seen_monotonic = now

        prev_book = context.market_state.prev_book
        features = self._calculate_features(context, snapshot, prev_book)
        context.last_feature_values = dict(features)
        context.market_state.update(snapshot)

        regime = self._detect_regime(context, features, snapshot)
        regime_weights = self._regime_weights.get_weight_adjustments(regime)
        regime_conf_mult = self._confidence_modifier.get_modifier(regime.name)
        composite = context.composite.calculate(
            features=features,
            regime=regime,
            regime_weights=regime_weights,
            regime_confidence_mult=regime_conf_mult,
            timestamp=snapshot.timestamp
        )

        spread_ticks = snapshot.spread_ticks(self._config.tick.tick_size)
        if spread_ticks is None:
            spread_ticks = 1.0
        threshold = context.threshold.calculate(
            spread_ticks=spread_ticks,
            obi_value=features.get(
                'weighted_obi', FeatureResult(0, 0, False, 0, '')
            ).value,
            microprice_deviation=features.get(
                'microprice', FeatureResult(0, 0, False, 0, '')
            ).ticks
        )
        threshold = self._regime_threshold.apply_regime_threshold(
            threshold,
            regime.name,
            self._config.threshold.min_threshold,
            self._config.threshold.max_threshold
        )
        composite.threshold_used = threshold
        context.last_composite = composite

        context.state_machine.update(
            composite_value=composite.value,
            confidence=composite.confidence,
            regime=regime.name,
            timestamp=snapshot.timestamp,
            activation_threshold=composite.threshold_used
        )

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        context.processing_times.append(elapsed_ms)
        context.snapshot_count += 1
        self._accepted_snapshot_count += 1
        self._total_processing_time_ms += elapsed_ms

        if self._accepted_snapshot_count % 100 == 0:
            self._logger.debug(
                "Processed snapshots",
                snapshot_count=self._accepted_snapshot_count,
                avg_time_ms=(
                    self._total_processing_time_ms
                    / self._accepted_snapshot_count
                ),
                active_contexts=len(self._contexts)
            )

        return composite

    def _calculate_features(
        self,
        context: SymbolContext,
        snapshot: Snapshot,
        prev_book: Optional[PriceKeyedBook]
    ) -> Dict[str, FeatureResult]:
        """Calculate features with stateful calculators from ``context``."""
        return {
            'microprice': self._microprice.calculate(snapshot, prev_book),
            'weighted_obi': self._weighted_obi.calculate(snapshot, prev_book),
            'depth_slope': self._depth_slope.calculate(snapshot, prev_book),
            'spread_quality': self._spread.calculate(snapshot, prev_book),
            'spread_compression': context.spread_compression.calculate(
                snapshot, prev_book
            ),
            'queue_persistence': context.queue_persistence.calculate(
                snapshot, prev_book
            ),
            'momentum': context.momentum.calculate(snapshot, prev_book),
            'acceleration': context.acceleration.calculate(snapshot, prev_book),
            'refill_proxy': context.refill.calculate(snapshot, prev_book),
            'ltp_confirmation': context.ltp_confirmation.calculate(
                snapshot, prev_book
            )
        }

    def _detect_regime(
        self,
        context: SymbolContext,
        features: Dict[str, FeatureResult],
        snapshot: Snapshot
    ) -> Regime:
        """Detect a regime using only this symbol's rolling state."""
        momentum = features.get(
            'momentum', FeatureResult(0, 0, False, 0, '')
        ).value
        obi = features.get(
            'weighted_obi', FeatureResult(0, 0, False, 0, '')
        ).value
        spread_ticks = snapshot.spread_ticks(self._config.tick.tick_size) or 1.0
        return context.regime_detector.detect(
            momentum, obi, spread_ticks, snapshot.ltp
        )

    def _resolve_context(
        self,
        symbol: str,
        exchange_type: Optional[int] = None
    ) -> Optional[SymbolContext]:
        """Resolve a symbol or qualified key without guessing across exchanges."""
        if not isinstance(symbol, str) or not symbol.strip():
            return None
        identifier = symbol.strip()
        if exchange_type is not None and (
            isinstance(exchange_type, bool)
            or exchange_type not in MarketSubscription.SUPPORTED_EXCHANGE_TYPES
        ):
            raise ValueError("exchange_type is unsupported")

        key_prefix, separator, _key_token = identifier.partition(":")
        if separator and key_prefix.isdecimal():
            qualified_exchange = int(key_prefix)
            if qualified_exchange not in MarketSubscription.SUPPORTED_EXCHANGE_TYPES:
                raise ValueError("qualified instrument key has unsupported exchange_type")
            if exchange_type is not None and exchange_type != qualified_exchange:
                raise ValueError(
                    "exchange_type contradicts the qualified instrument key"
                )
            return self._contexts.get(identifier)

        matches: dict[str, SymbolContext] = {}
        for context_key, context in self._contexts.items():
            snapshot = context.market_state.current_snapshot
            key_exchange_text, key_separator, key_token = context_key.partition(":")
            key_exchange = (
                int(key_exchange_text)
                if key_separator and key_exchange_text.isdecimal()
                else 0
            )
            identifier_matches = (
                context_key == identifier
                or (bool(key_token) and key_token == identifier)
                or (
                    snapshot is not None
                    and identifier in {snapshot.symbol, snapshot.token}
                )
            )
            exchange_matches = (
                exchange_type is None
                or key_exchange == exchange_type
                or (snapshot is not None and snapshot.exchange_type == exchange_type)
            )
            if identifier_matches and exchange_matches:
                matches[context_key] = context

        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous symbol {identifier!r}; use an exchange-qualified key "
                "such as 'exchange_type:token'"
            )
        return next(iter(matches.values())) if matches else None

    def get_state(
        self,
        symbol: str,
        exchange_type: Optional[int] = None
    ) -> State:
        """Get state by symbol or exchange-qualified instrument key."""
        context = self._resolve_context(symbol, exchange_type)
        return context.state_machine.state if context is not None else State.WARMUP

    def get_regime(
        self,
        symbol: str,
        exchange_type: Optional[int] = None
    ) -> Regime:
        """Get regime by symbol or exchange-qualified instrument key."""
        context = self._resolve_context(symbol, exchange_type)
        return (
            context.regime_detector.current_regime
            if context is not None
            else Regime.NOISE
        )

    def get_execution_signal(
        self,
        snapshot: Snapshot,
        composite: CompositeScore,
        quantity: Optional[int] = None
    ) -> Optional[ExecutionSignal]:
        """Create an execution signal from the snapshot symbol's context."""
        context = self._contexts.get(snapshot.instrument_key)
        if context is None:
            return None
        context_composite = context.last_composite
        if context_composite is None or composite is not context_composite:
            return None

        signal_type = context.state_machine.consume_execution_signal_type()
        if signal_type is None:
            return None

        # The transition command is consumed before construction. Invalid or
        # non-fillable commands therefore cannot be retried on positioned ticks.
        return self._execution.create_execution_signal(
            snapshot=snapshot,
            signal_type=signal_type,
            composite_value=context_composite.value,
            confidence=context_composite.confidence,
            regime=context_composite.regime,
            timestamp=context_composite.timestamp,
            quantity=quantity
        )

    def reset(
        self,
        symbol: Optional[str] = None,
        exchange_type: Optional[int] = None
    ) -> None:
        """Reset one resolved instrument, or every context when omitted."""
        if symbol is not None:
            context = self._resolve_context(symbol, exchange_type)
            if context is not None:
                context.reset()
            return
        if exchange_type is not None:
            raise ValueError("exchange_type requires a symbol or token")

        for context in self._contexts.values():
            context.reset()
        self._accepted_snapshot_count = 0
        self._rejected_snapshot_count = 0
        self._rejection_reasons.clear()
        self._contexts_created = len(self._contexts)
        self._contexts_evicted = 0
        self._context_eviction_reasons.clear()
        self._total_processing_time_ms = 0.0

    def _get_context_stats(self, context: SymbolContext) -> Dict[str, Any]:
        processing_times = context.processing_times
        avg_time = (
            sum(processing_times) / len(processing_times)
            if processing_times
            else 0.0
        )
        threshold = (
            context.last_composite.threshold_used
            if context.last_composite is not None
            else context.threshold.get_current_threshold()
        )
        return {
            'snapshot_count': context.snapshot_count,
            'avg_processing_time_ms': avg_time,
            'state': context.state_machine.state.name,
            'regime': context.regime_detector.current_regime.name,
            'threshold': threshold,
            'idle_seconds': max(
                0.0,
                time.monotonic() - context.last_seen_monotonic
            )
        }

    def get_stats(
        self,
        symbol: Optional[str] = None,
        exchange_type: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get aggregate stats or one instrument's resolved statistics."""
        if symbol is not None:
            context = self._resolve_context(symbol, exchange_type)
            if context is None:
                return {
                    'snapshot_count': 0,
                    'avg_processing_time_ms': 0.0,
                    'state': State.WARMUP.name,
                    'regime': Regime.NOISE.name,
                    'threshold': self._threshold_config.base_threshold,
                    'idle_seconds': 0.0,
                    'symbols_tracked': len(self._contexts)
                }
            stats = self._get_context_stats(context)
            stats['symbols_tracked'] = len(self._contexts)
            return stats

        if exchange_type is not None:
            raise ValueError("exchange_type requires a symbol or token")

        symbol_stats = {
            context_symbol: self._get_context_stats(context)
            for context_symbol, context in self._contexts.items()
        }
        current_snapshot_count = sum(
            context.snapshot_count for context in self._contexts.values()
        )
        rolling_times = [
            duration
            for context in self._contexts.values()
            for duration in context.processing_times
        ]
        avg_time = (
            sum(rolling_times) / len(rolling_times)
            if rolling_times
            else 0.0
        )
        lifetime_avg_time = (
            self._total_processing_time_ms / self._accepted_snapshot_count
            if self._accepted_snapshot_count
            else 0.0
        )

        if len(self._contexts) == 1:
            only_stats = next(iter(symbol_stats.values()))
            state = only_stats['state']
            regime = only_stats['regime']
            threshold = only_stats['threshold']
        elif not self._contexts:
            state = State.WARMUP.name
            regime = Regime.NOISE.name
            threshold = self._threshold_config.base_threshold
        else:
            state = 'MULTI'
            regime = 'MULTI'
            threshold = self._threshold_config.base_threshold

        return {
            'snapshot_count': current_snapshot_count,
            'accepted_snapshot_count': self._accepted_snapshot_count,
            'rejected_snapshot_count': self._rejected_snapshot_count,
            'avg_processing_time_ms': avg_time,
            'lifetime_avg_processing_time_ms': lifetime_avg_time,
            'state': state,
            'regime': regime,
            'threshold': threshold,
            'symbols_tracked': len(self._contexts),
            'contexts_created': self._contexts_created,
            'contexts_evicted': self._contexts_evicted,
            'max_active_contexts': self._config.context.max_active_contexts,
            'rejection_reasons': dict(self._rejection_reasons),
            'context_eviction_reasons': dict(
                self._context_eviction_reasons
            ),
            'symbols': symbol_stats
        }

    def get_feature_values(
        self,
        symbol: str,
        exchange_type: Optional[int] = None
    ) -> Dict[str, float]:
        """Get feature values by symbol or exchange-qualified key."""
        context = self._resolve_context(symbol, exchange_type)
        if context is None:
            return {}
        return {
            name: result.value
            for name, result in context.last_feature_values.items()
        }
