"""
Queue Persistence feature.

Tracks stability of queue (quantity) at specific price levels.
Persistent queues suggest strong support/resistance.

RETAIL API LIMITATION:
We cannot see order additions/cancellations directly.
We only see net quantity changes at each price level.

This is a PROXY for queue stability based on:
- Price-keyed comparison (not index-based)
- Quantity persistence ratio
"""

from typing import Optional, Dict, Tuple
from dataclasses import dataclass
import math

from utils.types import Snapshot, PriceKeyedBook, FeatureResult, RollingStats, PriceLevel
from utils.math_utils import clamp, safe_divide
from utils.constants import QUEUE_PERSISTENCE_RANGE


@dataclass
class QueuePersistenceConfig:
    """Configuration for queue persistence calculation."""
    tick_size: float = 0.05
    min_persistence_ratio: float = 0.3
    max_persistence_ratio: float = 0.95
    levels_to_track: int = 5
    normalize_range: float = QUEUE_PERSISTENCE_RANGE


class QueuePersistenceCalculator:
    """
    Calculate queue persistence feature.
    
    IMPORTANT: Uses PRICE-KEYED comparison, not index-based.
    This avoids false signals when best bid/ask shifts.
    
    Persistence = quantity_at_price / previous_quantity_at_same_price
    
    High persistence at bid = bid support is stable (bullish)
    High persistence at ask = ask resistance is stable (bearish)
    
    We calculate relative persistence between sides:
    - Bid persistence > Ask persistence = bullish
    - Ask persistence > Bid persistence = bearish
    
    CRITICAL: This only captures NET changes (additions - cancellations - fills).
    We cannot distinguish:
    - New order added vs existing order increased
    - Cancellation vs execution
    
    The proxy assumes stable queues indicate informed participants.
    """
    
    def __init__(self, config: QueuePersistenceConfig):
        self._config = config
        self._prev_book: Optional[PriceKeyedBook] = None
    
    def calculate(
        self,
        snapshot: Snapshot,
        prev_book: Optional[PriceKeyedBook] = None,
        stats: Optional[RollingStats] = None
    ) -> FeatureResult:
        """
        Calculate queue persistence.
        
        Args:
            snapshot: Current snapshot
            prev_book: Previous price-keyed book (CRITICAL: price-keyed, not index-based)
            stats: Rolling statistics (not used)
        
        Returns:
            FeatureResult indicating relative queue stability
        """
        if not snapshot.is_valid():
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="queue_persistence"
            )
        
        # Use provided prev_book or internal state
        book = prev_book if prev_book is not None else self._prev_book
        
        if book is None:
            # First snapshot - no persistence to measure
            self._prev_book = PriceKeyedBook.from_snapshot(snapshot)
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=True,
                raw_value=1.0,
                name="queue_persistence"
            )
        
        # Calculate persistence for each side
        bid_persistence, bid_conf = self._calculate_side_persistence(
            snapshot.bids, book.bids, 'bid'
        )
        ask_persistence, ask_conf = self._calculate_side_persistence(
            snapshot.asks, book.asks, 'ask'
        )
        
        # Update internal state
        self._prev_book = PriceKeyedBook.from_snapshot(snapshot)
        
        # Compare relative persistence
        # Higher bid persistence = bid queue more stable = bullish
        # Higher ask persistence = ask queue more stable = bearish
        
        if bid_persistence is None or ask_persistence is None:
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=True,
                raw_value=0.5,
                name="queue_persistence"
            )
        
        # Normalize: bid_persistence - ask_persistence in [-1, +1]
        persistence_diff = bid_persistence - ask_persistence
        
        # Scale by combined confidence
        confidence = (bid_conf + ask_conf) / 2
        
        return FeatureResult(
            value=clamp(persistence_diff, -1.0, 1.0),
            confidence=confidence,
            valid=True,
            raw_value=persistence_diff,
            name="queue_persistence"
        )
    
    def _calculate_side_persistence(
        self,
        current_levels: Tuple[PriceLevel, ...],
        prev_prices: Dict[float, Tuple[int, int]],
        side: str
    ) -> Tuple[Optional[float], float]:
        """
        Calculate persistence for one side.
        
        Uses PRICE-KEYED lookup, not index-based.
        
        Args:
            current_levels: Current price levels
            prev_prices: Previous price -> (qty, orders) mapping
            side: 'bid' or 'ask'
        
        Returns:
            Tuple of (persistence_ratio, confidence)
        """
        if not current_levels:
            return None, 0.0
        
        total_current = 0
        total_persistent = 0
        levels_checked = 0
        
        for i, level in enumerate(current_levels[:self._config.levels_to_track]):
            current_qty = level.quantity
            total_current += current_qty
            
            # PRICE-KEYED lookup (CRITICAL)
            prev_data = prev_prices.get(level.price)
            if prev_data is not None:
                prev_qty = prev_data[0]
                # Persistence: how much of previous quantity remains
                if prev_qty > 0:
                    persistence = min(1.0, current_qty / prev_qty)
                    total_persistent += persistence * current_qty
                else:
                    # New quantity at this price (no previous)
                    total_persistent += 0.5 * current_qty  # Neutral
            
            levels_checked += 1
        
        if total_current <= 0 or levels_checked == 0:
            return None, 0.0
        
        # Weighted average persistence
        persistence_ratio = total_persistent / total_current
        
        # Confidence based on levels checked
        confidence = min(1.0, levels_checked / self._config.levels_to_track)
        
        return persistence_ratio, confidence
    
    def get_bid_persistence(
        self,
        snapshot: Snapshot,
        prev_book: PriceKeyedBook
    ) -> Tuple[float, float]:
        """Get bid side persistence only."""
        persistence, conf = self._calculate_side_persistence(
            snapshot.bids, prev_book.bids, 'bid'
        )
        return persistence if persistence else 0.0, conf
    
    def get_ask_persistence(
        self,
        snapshot: Snapshot,
        prev_book: PriceKeyedBook
    ) -> Tuple[float, float]:
        """Get ask side persistence only."""
        persistence, conf = self._calculate_side_persistence(
            snapshot.asks, prev_book.asks, 'ask'
        )
        return persistence if persistence else 0.0, conf
    
    def reset(self) -> None:
        """Reset internal state."""
        self._prev_book = None
    
    def get_queue_changes(
        self,
        snapshot: Snapshot,
        prev_book: PriceKeyedBook
    ) -> Dict[str, Dict[float, int]]:
        """
        Get queue changes at each price level.
        
        Returns:
            Dict with 'bid' and 'ask' keys, each mapping price -> qty_change
        """
        changes = {'bid': {}, 'ask': {}}
        
        for level in snapshot.bids[:self._config.levels_to_track]:
            prev_data = prev_book.bids.get(level.price)
            if prev_data:
                change = level.quantity - prev_data[0]
                changes['bid'][level.price] = change
            else:
                changes['bid'][level.price] = level.quantity  # New level
        
        for level in snapshot.asks[:self._config.levels_to_track]:
            prev_data = prev_book.asks.get(level.price)
            if prev_data:
                change = level.quantity - prev_data[0]
                changes['ask'][level.price] = change
            else:
                changes['ask'][level.price] = level.quantity  # New level
        
        return changes


def calculate_queue_persistence(
    snapshot: Snapshot,
    prev_book: Optional[PriceKeyedBook] = None
) -> Tuple[float, float]:
    """
    Calculate queue persistence.
    
    Returns:
        Tuple of (persistence_value, confidence)
    """
    config = QueuePersistenceConfig()
    calc = QueuePersistenceCalculator(config)
    result = calc.calculate(snapshot, prev_book)
    return result.value, result.confidence
