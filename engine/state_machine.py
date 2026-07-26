"""
State Machine for signal generation.

Manages state transitions:
WARMUP -> NEUTRAL -> WATCH_LONG/WATCH_SHORT -> LONG/SHORT -> EXIT -> COOLDOWN

Ensures signals are persistent and not noise.

THREAD SAFETY:
This state machine is used in multi-threaded WebSocket context.
All state mutations are protected by threading.Lock().
"""

from typing import Optional, Dict, List
from dataclasses import dataclass
import math
import time
import threading

from utils.types import State, SignalType, StateTransition
from utils.constants import (
    WATCH_TIMEOUT,
    COOLDOWN_TIME,
    MIN_HOLD_TIME,
    WARMUP_SAMPLES,
    MIN_CONSECUTIVE_SIGNALS
)


@dataclass
class StateMachineConfig:
    """Configuration for state machine."""
    warmup_samples: int = WARMUP_SAMPLES
    watch_timeout_seconds: float = WATCH_TIMEOUT
    cooldown_seconds: float = COOLDOWN_TIME
    min_hold_time: float = MIN_HOLD_TIME
    signal_persistence: int = MIN_CONSECUTIVE_SIGNALS
    
    # Threshold for triggering WATCH state
    watch_threshold: float = 0.5
    
    # Threshold for triggering position
    position_threshold: float = 0.7
    
    # Threshold for exiting position
    exit_threshold: float = 0.3


class TradingStateMachine:
    """
    State machine for trading signal generation.
    
    States:
    - WARMUP: Collecting initial samples
    - NEUTRAL: No position, no strong signal
    - WATCH_LONG: Bullish signal building, waiting for confirmation
    - WATCH_SHORT: Bearish signal building, waiting for confirmation
    - LONG: Active long position
    - SHORT: Active short position
    - EXIT_LONG: Exiting long position
    - EXIT_SHORT: Exiting short position
    - COOLDOWN: Post-exit cooldown
    
    Transitions are triggered by composite score and confidence.
    """
    
    def __init__(self, config: StateMachineConfig):
        self._config = config
        
        # Thread safety lock
        self._lock = threading.Lock()
        
        # Current state
        self._state = State.WARMUP
        self._sample_count = 0
        
        # State timing - initialize with current time
        self._state_enter_time: float = time.time()
        self._watch_start_time: Optional[float] = None
        self._position_start_time: Optional[float] = None
        
        # Signal tracking
        self._consecutive_bullish = 0
        self._consecutive_bearish = 0
        self._last_composite = 0.0
        
        # Transition history and one-shot command for the latest update.
        self._transitions: List[StateTransition] = []
        self._pending_signal_type: Optional[SignalType] = None
    
    def update(
        self,
        composite_value: float,
        confidence: float,
        regime: str,
        timestamp: Optional[float] = None,
        activation_threshold: Optional[float] = None
    ) -> State:
        """
        Update state machine with new composite.

        ``activation_threshold`` overrides the configured position threshold for
        directional tracking and entry confirmation. The watch threshold keeps
        the configured watch/position ratio, while exits continue to use the
        configured exit threshold and minimum hold time.
        
        Thread-safe: Uses lock to protect state mutations.
        
        Args:
            composite_value: Normalized composite score [-1, +1]
            confidence: Signal confidence [0, 1]
            regime: Current market regime
            timestamp: Current timestamp (defaults to current time)
            activation_threshold: Optional per-update directional entry threshold
        
        Returns:
            Current state
        """
        with self._lock:
            return self._update_internal(
                composite_value,
                confidence,
                regime,
                timestamp,
                activation_threshold
            )
    
    def _update_internal(
        self,
        composite_value: float,
        confidence: float,
        regime: str,
        timestamp: Optional[float],
        activation_threshold: Optional[float]
    ) -> State:
        """Internal update (must be called with lock held)."""
        if timestamp is None:
            timestamp = time.time()

        entry_threshold, watch_threshold = self._resolve_activation_thresholds(
            activation_threshold
        )

        # A command belongs only to the transition produced by this update.
        # If it was not consumed, advancing the machine expires it rather than
        # allowing a stale or failed command to be retried indefinitely.
        self._pending_signal_type = None

        self._sample_count += 1
        self._last_composite = composite_value
        
        # Track signal direction using the same activation policy as entry.
        if composite_value >= watch_threshold:
            self._consecutive_bullish += 1
            self._consecutive_bearish = 0
        elif composite_value <= -watch_threshold:
            self._consecutive_bearish += 1
            self._consecutive_bullish = 0
        else:
            self._consecutive_bullish = max(0, self._consecutive_bullish - 1)
            self._consecutive_bearish = max(0, self._consecutive_bearish - 1)
        
        # Process based on current state
        new_state = self._process_state(
            composite_value, confidence, regime, timestamp, entry_threshold
        )
        
        # Handle state transition
        if new_state != self._state:
            self._transition(new_state, composite_value, timestamp)
        
        return self._state

    def _resolve_activation_thresholds(
        self,
        activation_threshold: Optional[float]
    ) -> tuple[float, float]:
        """Return safe entry/watch thresholds for one update."""
        configured_entry = self._config.position_threshold
        if (
            activation_threshold is None
            or isinstance(activation_threshold, bool)
            or not isinstance(activation_threshold, (int, float))
            or not math.isfinite(activation_threshold)
        ):
            return configured_entry, self._config.watch_threshold

        entry_threshold = min(1.0, max(0.0, float(activation_threshold)))
        if configured_entry > 0 and math.isfinite(configured_entry):
            watch_ratio = self._config.watch_threshold / configured_entry
        else:
            watch_ratio = 1.0
        watch_ratio = min(1.0, max(0.0, watch_ratio))
        watch_threshold = entry_threshold * watch_ratio
        return entry_threshold, watch_threshold
    
    def _process_state(
        self,
        composite: float,
        confidence: float,
        regime: str,
        timestamp: float,
        entry_threshold: float
    ) -> State:
        """Process state transitions."""
        
        # WARMUP: Collect samples
        if self._state == State.WARMUP:
            if self._sample_count >= self._config.warmup_samples:
                return State.NEUTRAL
            return State.WARMUP
        
        # NEUTRAL: Check for signal emergence
        elif self._state == State.NEUTRAL:
            if self._consecutive_bullish >= self._config.signal_persistence:
                self._watch_start_time = timestamp
                return State.WATCH_LONG
            elif self._consecutive_bearish >= self._config.signal_persistence:
                self._watch_start_time = timestamp
                return State.WATCH_SHORT
            return State.NEUTRAL
        
        # WATCH_LONG: Confirm bullish signal
        elif self._state == State.WATCH_LONG:
            # Check timeout
            if self._watch_start_time is not None:
                elapsed = timestamp - self._watch_start_time
                if elapsed > self._config.watch_timeout_seconds:
                    return State.NEUTRAL
            
            # Check for confirmation
            if composite >= entry_threshold and confidence > 0.5:
                self._position_start_time = timestamp
                return State.LONG
            
            # Check for reversal
            if composite < 0:
                return State.NEUTRAL
            
            return State.WATCH_LONG
        
        # WATCH_SHORT: Confirm bearish signal
        elif self._state == State.WATCH_SHORT:
            # Check timeout
            if self._watch_start_time is not None:
                elapsed = timestamp - self._watch_start_time
                if elapsed > self._config.watch_timeout_seconds:
                    return State.NEUTRAL
            
            # Check for confirmation
            if composite <= -entry_threshold and confidence > 0.5:
                self._position_start_time = timestamp
                return State.SHORT
            
            # Check for reversal
            if composite > 0:
                return State.NEUTRAL
            
            return State.WATCH_SHORT
        
        # LONG: Check for exit
        elif self._state == State.LONG:
            # Minimum hold time
            if self._position_start_time is not None:
                elapsed = timestamp - self._position_start_time
                if elapsed < self._config.min_hold_time:
                    return State.LONG
            
            # Check for exit signal
            if composite < self._config.exit_threshold:
                return State.EXIT_LONG
            
            return State.LONG
        
        # SHORT: Check for exit
        elif self._state == State.SHORT:
            # Minimum hold time
            if self._position_start_time is not None:
                elapsed = timestamp - self._position_start_time
                if elapsed < self._config.min_hold_time:
                    return State.SHORT
            
            # Check for exit signal
            if composite > -self._config.exit_threshold:
                return State.EXIT_SHORT
            
            return State.SHORT
        
        # EXIT_LONG/EXIT_SHORT: Transition to cooldown
        elif self._state == State.EXIT_LONG or self._state == State.EXIT_SHORT:
            return State.COOLDOWN
        
        # COOLDOWN: Wait before next signal
        elif self._state == State.COOLDOWN:
            elapsed = timestamp - self._state_enter_time
            if elapsed >= self._config.cooldown_seconds:
                return State.NEUTRAL
            return State.COOLDOWN
        
        return self._state
    
    def _transition(self, new_state: State, composite: float, timestamp: float) -> None:
        """Handle state transition."""
        old_state = self._state
        
        # Record transition
        transition = StateTransition(
            from_state=old_state,
            to_state=new_state,
            trigger=f"composite={composite:.3f}",
            composite_value=composite,
            timestamp=timestamp,
            reason=self._get_transition_reason(old_state, new_state)
        )
        self._transitions.append(transition)

        command_by_transition = {
            (State.WATCH_LONG, State.LONG): SignalType.BULLISH,
            (State.WATCH_SHORT, State.SHORT): SignalType.BEARISH,
            (State.LONG, State.EXIT_LONG): SignalType.EXIT_LONG,
            (State.SHORT, State.EXIT_SHORT): SignalType.EXIT_SHORT,
        }
        self._pending_signal_type = command_by_transition.get((old_state, new_state))

        # Update state
        self._state = new_state
        self._state_enter_time = timestamp
        
        # Reset counters on transition
        if new_state == State.NEUTRAL:
            self._consecutive_bullish = 0
            self._consecutive_bearish = 0
            self._watch_start_time = None
        
        # Keep transition history limited
        if len(self._transitions) > 100:
            self._transitions = self._transitions[-50:]
    
    def _get_transition_reason(self, old_state: State, new_state: State) -> str:
        """Get human-readable reason for transition."""
        reasons = {
            (State.WARMUP, State.NEUTRAL): "Warmup complete",
            (State.NEUTRAL, State.WATCH_LONG): "Bullish signal emerging",
            (State.NEUTRAL, State.WATCH_SHORT): "Bearish signal emerging",
            (State.WATCH_LONG, State.LONG): "Bullish signal confirmed",
            (State.WATCH_SHORT, State.SHORT): "Bearish signal confirmed",
            (State.WATCH_LONG, State.NEUTRAL): "Bullish signal faded",
            (State.WATCH_SHORT, State.NEUTRAL): "Bearish signal faded",
            (State.LONG, State.EXIT_LONG): "Exit signal triggered",
            (State.SHORT, State.EXIT_SHORT): "Exit signal triggered",
            (State.EXIT_LONG, State.COOLDOWN): "Long position exited",
            (State.EXIT_SHORT, State.COOLDOWN): "Short position exited",
            (State.COOLDOWN, State.NEUTRAL): "Cooldown complete"
        }
        return reasons.get((old_state, new_state), "State transition")
    
    @property
    def state(self) -> State:
        """Get current state (thread-safe)."""
        with self._lock:
            return self._state
    
    @property
    def is_positioned(self) -> bool:
        """Check if in active position (thread-safe)."""
        with self._lock:
            return self._state in (State.LONG, State.SHORT)
    
    @property
    def position_direction(self) -> Optional[int]:
        """Get position direction: +1 long, -1 short, None otherwise (thread-safe)."""
        with self._lock:
            if self._state == State.LONG:
                return 1
            elif self._state == State.SHORT:
                return -1
            return None
    
    @property
    def time_in_state(self) -> float:
        """Get time spent in current state."""
        with self._lock:
            return time.time() - self._state_enter_time
    
    def consume_execution_signal_type(self) -> Optional[SignalType]:
        """Consume the command created by the latest qualifying transition."""
        with self._lock:
            signal_type = self._pending_signal_type
            self._pending_signal_type = None
            return signal_type

    def get_signal_type(self) -> SignalType:
        """Get the directional signal represented by the current state."""
        with self._lock:
            if self._state in (State.LONG, State.WATCH_LONG):
                return SignalType.BULLISH
            if self._state in (State.SHORT, State.WATCH_SHORT):
                return SignalType.BEARISH
            if self._state == State.EXIT_LONG:
                return SignalType.EXIT_LONG
            if self._state == State.EXIT_SHORT:
                return SignalType.EXIT_SHORT
        return SignalType.NEUTRAL
    
    def reset(self) -> None:
        """Reset state machine (thread-safe)."""
        with self._lock:
            self._state = State.WARMUP
            self._sample_count = 0
            self._state_enter_time = 0.0
            self._watch_start_time = None
            self._position_start_time = None
            self._consecutive_bullish = 0
            self._consecutive_bearish = 0
            self._last_composite = 0.0
            self._pending_signal_type = None
            self._transitions.clear()
    
    def force_state(self, state: State) -> None:
        """Force state transition (for testing/recovery - thread-safe)."""
        with self._lock:
            self._state = state
            self._state_enter_time = time.time()
            self._pending_signal_type = None
    
    def get_transitions(self, limit: int = 10) -> List[StateTransition]:
        """Get recent transitions (thread-safe)."""
        with self._lock:
            return self._transitions[-limit:].copy()
    
    def get_state_info(self) -> Dict:
        """Get current state information (thread-safe)."""
        with self._lock:
            return {
                'state': self._state.name,
                'sample_count': self._sample_count,
                'time_in_state': time.time() - self._state_enter_time,
                'consecutive_bullish': self._consecutive_bullish,
                'consecutive_bearish': self._consecutive_bearish,
                'last_composite': self._last_composite,
                'is_positioned': self._state in (State.LONG, State.SHORT),
                'position_direction': 1 if self._state == State.LONG else (-1 if self._state == State.SHORT else None)
            }
