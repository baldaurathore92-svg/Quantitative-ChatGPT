"""
Refill Proxy feature.

Detects consumption followed by recovery at price levels.
Indicates aggressive participants being filled and market makers replenishing.

RETAIL API LIMITATION:
We CANNOT see:
- Individual order additions
- Order cancellations vs executions
- Hidden liquidity

This is a PROXY based on observable quantity changes.
We can only detect NET changes at each price level.

A "refill" detection means:
1. Quantity at a price level decreased (consumption detected)
2. Quantity subsequently increased (recovery detected)

This suggests aggressive order flow followed by liquidity provision.
"""

from typing import Optional, Dict, Tuple
from dataclasses import dataclass
from enum import Enum

from utils.types import Snapshot, PriceKeyedBook, FeatureResult, RollingStats, PriceLevel
from utils.math_utils import clamp
from utils.constants import REFILL_PROXY_RANGE


class RefillState(Enum):
    """State of refill detection."""
    NEUTRAL = 0
    CONSUMED = 1  # Quantity dropped
    RECOVERED = 2  # Quantity recovered after drop


@dataclass
class LevelState:
    """State for a single price level."""
    price: float
    prev_qty: int = 0
    min_qty: int = 0  # Minimum seen since consumption
    max_qty: int = 0  # Maximum seen
    state: RefillState = RefillState.NEUTRAL
    consumption_ratio: float = 0.0
    recovery_ratio: float = 0.0
    timestamp: float = 0.0


@dataclass
class RefillConfig:
    """Configuration for refill detection."""
    tick_size: float = 0.05
    consumption_threshold: float = 0.3  # % drop to detect consumption
    recovery_threshold: float = 0.5  # % recovery to detect refill
    levels_to_track: int = 3
    max_age_seconds: float = 30.0  # Max age for consumption event
    normalize_range: float = REFILL_PROXY_RANGE


class RefillProxyCalculator:
    """
    Calculate refill proxy feature.
    
    CRITICAL IMPLEMENTATION:
    - Uses PRICE-KEYED tracking (not index-based)
    - Only triggers AFTER confirmed consumption
    - Does NOT trigger on simple queue growth
    - Cleans up old price level entries to prevent memory leak
    
    Detection flow:
    1. Track quantity at each price level
    2. Detect consumption (quantity drops significantly)
    3. Detect recovery (quantity increases after drop)
    4. Signal = bid refill (bullish) or ask refill (bearish)
    
    Signal interpretation:
    - Bid refill = bid wall rebuilt after consumption = bullish support
    - Ask refill = ask wall rebuilt after consumption = bearish resistance
    
    IMPORTANT CAVEAT:
    We cannot distinguish:
    - New orders vs existing order increases
    - Market maker replenishment vs new participant entry
    
    The proxy assumes refill indicates conviction at that price level.
    """
    
    def __init__(self, config: RefillConfig):
        self._config = config
        self._bid_states: Dict[float, LevelState] = {}
        self._ask_states: Dict[float, LevelState] = {}
        self._prev_book: Optional[PriceKeyedBook] = None
        self._last_refill_signal = 0.0
        self._cleanup_counter = 0  # For periodic cleanup
    
    def calculate(
        self,
        snapshot: Snapshot,
        prev_book: Optional[PriceKeyedBook] = None,
        stats: Optional[RollingStats] = None
    ) -> FeatureResult:
        """
        Calculate refill proxy feature.
        
        Args:
            snapshot: Current snapshot
            prev_book: Previous price-keyed book
            stats: Rolling statistics
        
        Returns:
            FeatureResult indicating refill signal
        """
        if not snapshot.is_valid():
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="refill_proxy"
            )
        
        timestamp = snapshot.timestamp
        book = prev_book if prev_book is not None else self._prev_book
        
        # First snapshot - initialize state
        if book is None:
            self._initialize_states(snapshot)
            self._prev_book = PriceKeyedBook.from_snapshot(snapshot)
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=True,
                raw_value=0.0,
                name="refill_proxy"
            )
        
        # Periodic cleanup to prevent memory leak
        self._cleanup_counter += 1
        if self._cleanup_counter >= 100:
            self._cleanup_old_states(snapshot)
            self._cleanup_counter = 0
        
        # Update level states with current snapshot
        bid_refill = self._update_side_states(
            snapshot.bids, book.bids, self._bid_states, timestamp, 'bid'
        )
        ask_refill = self._update_side_states(
            snapshot.asks, book.asks, self._ask_states, timestamp, 'ask'
        )
        
        # Update previous book
        self._prev_book = PriceKeyedBook.from_snapshot(snapshot)
        
        # Calculate signal
        # Bid refill = bullish, Ask refill = bearish
        refill_signal = bid_refill - ask_refill
        
        # Smooth signal slightly
        self._last_refill_signal = 0.7 * self._last_refill_signal + 0.3 * refill_signal
        
        normalized = clamp(self._last_refill_signal, -1.0, 1.0)
        
        # Confidence based on detection strength
        confidence = self._calculate_confidence(bid_refill, ask_refill)
        
        return FeatureResult(
            value=normalized,
            confidence=confidence,
            valid=True,
            raw_value=refill_signal,
            name="refill_proxy"
        )
    
    def _cleanup_old_states(self, snapshot: Snapshot) -> None:
        """
        Clean up old price level states that are no longer in the book.
        
        This prevents unbounded memory growth.
        """
        current_bid_prices = set(level.price for level in snapshot.bids[:self._config.levels_to_track])
        current_ask_prices = set(level.price for level in snapshot.asks[:self._config.levels_to_track])
        
        # Remove old bid states
        prices_to_remove = [p for p in self._bid_states if p not in current_bid_prices]
        for price in prices_to_remove:
            del self._bid_states[price]
        
        # Remove old ask states
        prices_to_remove = [p for p in self._ask_states if p not in current_ask_prices]
        for price in prices_to_remove:
            del self._ask_states[price]
    
    def _initialize_states(self, snapshot: Snapshot) -> None:
        """Initialize level states from first snapshot."""
        self._bid_states.clear()
        self._ask_states.clear()
        
        for level in snapshot.bids[:self._config.levels_to_track]:
            self._bid_states[level.price] = LevelState(
                price=level.price,
                prev_qty=level.quantity,
                min_qty=level.quantity,
                max_qty=level.quantity,
                state=RefillState.NEUTRAL
            )
        
        for level in snapshot.asks[:self._config.levels_to_track]:
            self._ask_states[level.price] = LevelState(
                price=level.price,
                prev_qty=level.quantity,
                min_qty=level.quantity,
                max_qty=level.quantity,
                state=RefillState.NEUTRAL
            )
    
    def _update_side_states(
        self,
        current_levels: Tuple[PriceLevel, ...],
        prev_prices: Dict[float, Tuple[int, int]],
        states: Dict[float, LevelState],
        timestamp: float,
        side: str
    ) -> float:
        """
        Update states and calculate refill signal for one side.
        
        Returns:
            Refill signal (0 to 1) for this side
        """
        if not current_levels:
            return 0.0
        
        refill_detected = 0.0
        levels_analyzed = 0
        
        for level in current_levels[:self._config.levels_to_track]:
            price = level.price
            current_qty = level.quantity
            
            # Get previous data (PRICE-KEYED)
            prev_data = prev_prices.get(price)
            prev_qty = prev_data[0] if prev_data else current_qty
            
            # Get or create state
            state = states.get(price)
            if state is None:
                state = LevelState(price=price)
                states[price] = state
            
            # Update max
            if current_qty > state.max_qty:
                state.max_qty = current_qty
            
            # Detect consumption
            if state.state == RefillState.NEUTRAL:
                if prev_qty > 0:
                    drop_ratio = (prev_qty - current_qty) / prev_qty
                    if drop_ratio >= self._config.consumption_threshold:
                        # Consumption detected
                        state.state = RefillState.CONSUMED
                        state.min_qty = current_qty
                        state.consumption_ratio = drop_ratio
                        state.timestamp = timestamp
            
            # Detect recovery after consumption
            elif state.state == RefillState.CONSUMED:
                if current_qty < state.min_qty:
                    state.min_qty = current_qty

                # Expiry is independent of whether a recovery ratio is computable.
                if timestamp - state.timestamp > self._config.max_age_seconds:
                    state.state = RefillState.NEUTRAL
                    state.min_qty = current_qty
                elif state.min_qty > 0:
                    recovery = (current_qty - state.min_qty) / state.min_qty

                    if recovery >= self._config.recovery_threshold:
                        # Refill detected!
                        state.state = RefillState.RECOVERED
                        state.recovery_ratio = recovery

                        # Signal strength
                        strength = state.consumption_ratio * state.recovery_ratio
                        refill_detected += strength

                        # Reset to neutral after detection
                        state.state = RefillState.NEUTRAL
                        state.min_qty = current_qty
                        state.max_qty = current_qty
            
            state.prev_qty = current_qty
            levels_analyzed += 1
        
        # Normalize by levels analyzed
        if levels_analyzed > 0:
            refill_detected = min(1.0, refill_detected / levels_analyzed)
        
        return refill_detected
    
    def _calculate_confidence(self, bid_refill: float, ask_refill: float) -> float:
        """Calculate confidence based on signal strength."""
        max_signal = max(bid_refill, ask_refill)
        
        if max_signal >= 0.5:
            return 0.8
        elif max_signal >= 0.3:
            return 0.6
        elif max_signal >= 0.1:
            return 0.4
        else:
            return 0.2
    
    def reset(self) -> None:
        """Reset all state."""
        self._bid_states.clear()
        self._ask_states.clear()
        self._prev_book = None
        self._last_refill_signal = 0.0
    
    def get_level_states(self) -> Dict[str, Dict[float, LevelState]]:
        """Get current level states for debugging."""
        return {
            'bid': dict(self._bid_states),
            'ask': dict(self._ask_states)
        }


def calculate_refill_proxy(
    snapshot: Snapshot,
    prev_book: Optional[PriceKeyedBook] = None
) -> Tuple[float, float]:
    """
    Calculate refill proxy.
    
    Returns:
        Tuple of (refill_signal, confidence)
    """
    config = RefillConfig()
    calc = RefillProxyCalculator(config)
    result = calc.calculate(snapshot, prev_book)
    return result.value, result.confidence
