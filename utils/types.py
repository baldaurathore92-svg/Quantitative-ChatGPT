"""
Type definitions for Snapshot Quant Engine.

Protocol definitions, dataclasses, and type aliases for the entire system.
"""

from dataclasses import dataclass, field
from typing import Protocol, Dict, List, Optional, Tuple, Any
from enum import Enum, auto
from datetime import datetime
import time


class Regime(Enum):
    """Market regime states."""
    TREND = auto()
    PULLBACK = auto()
    RANGE = auto()
    NOISE = auto()


class State(Enum):
    """Trading state machine states."""
    WARMUP = auto()
    NEUTRAL = auto()
    WATCH_LONG = auto()
    WATCH_SHORT = auto()
    LONG = auto()
    SHORT = auto()
    EXIT_LONG = auto()
    EXIT_SHORT = auto()
    COOLDOWN = auto()


class SignalType(Enum):
    """Signal types from composite score."""
    BULLISH = auto()
    BEARISH = auto()
    NEUTRAL = auto()


@dataclass(frozen=True)
class PriceLevel:
    """Single price level in order book."""
    price: float
    quantity: int
    order_count: int = 0
    
    def __post_init__(self):
        if self.price < 0:
            raise ValueError(f"Price cannot be negative: {self.price}")
        if self.quantity < 0:
            raise ValueError(f"Quantity cannot be negative: {self.quantity}")
        if self.order_count < 0:
            raise ValueError(f"Order count cannot be negative: {self.order_count}")


@dataclass(frozen=True)
class DepthLevel:
    """Depth level with both bid and ask at same level."""
    bid: PriceLevel
    ask: PriceLevel


@dataclass
class Snapshot:
    """
    Validated market snapshot from Angel One SmartAPI V2.
    
    All prices and quantities validated. Missing/invalid fields rejected upstream.
    """
    symbol: str
    timestamp: float  # Unix timestamp in seconds
    ltp: float  # Last traded price
    ltp_quantity: int = 0  # Last traded quantity (if available)
    volume_traded: int = 0  # Total volume for the day
    total_buy_qty: int = 0  # Total buy quantity (if available)
    total_sell_qty: int = 0  # Total sell quantity (if available)
    bids: Tuple[PriceLevel, ...] = field(default_factory=tuple)  # Bid levels (0 = best)
    asks: Tuple[PriceLevel, ...] = field(default_factory=tuple)  # Ask levels (0 = best)
    sequence: int = 0  # Sequence number for ordering
    exchange_timestamp: float = 0.0  # Exchange timestamp if available
    
    @property
    def best_bid(self) -> Optional[PriceLevel]:
        """Get best bid level."""
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[PriceLevel]:
        """Get best ask level."""
        return self.asks[0] if self.asks else None
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid.price + self.best_ask.price) / 2.0
        return None
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask.price - self.best_bid.price
        return None
    
    def spread_ticks(self, tick_size: float = 0.05) -> Optional[float]:
        """
        Calculate spread in ticks.
        
        NOTE: This is a method, not a property, because it requires tick_size argument.
        For convenience, use snapshot.spread / tick_size directly.
        """
        if self.spread:
            return self.spread / tick_size
        return None
    
    @property
    def depth(self) -> int:
        """Get available depth levels."""
        return min(len(self.bids), len(self.asks))
    
    def is_valid(self) -> bool:
        """Basic validity check."""
        if not self.bids or not self.asks:
            return False
        if self.best_bid is None or self.best_ask is None:
            return False
        if self.best_bid.price <= 0 or self.best_ask.price <= 0:
            return False
        if self.best_bid.price >= self.best_ask.price:
            return False  # Crossed book
        if self.ltp <= 0:
            return False
        return True


@dataclass
class PriceKeyedBook:
    """
    Price-keyed order book for O(1) lookups.
    
    Maps price -> (quantity, order_count) for both sides.
    Enables proper comparison between snapshots without false refill/consumption.
    """
    bids: Dict[float, Tuple[int, int]]  # price -> (qty, order_count)
    asks: Dict[float, Tuple[int, int]]  # price -> (qty, order_count)
    timestamp: float
    
    @classmethod
    def from_snapshot(cls, snapshot: Snapshot) -> 'PriceKeyedBook':
        """Create price-keyed book from snapshot."""
        bids = {level.price: (level.quantity, level.order_count) for level in snapshot.bids}
        asks = {level.price: (level.quantity, level.order_count) for level in snapshot.asks}
        return cls(bids=bids, asks=asks, timestamp=snapshot.timestamp)
    
    def get_bid_qty(self, price: float) -> int:
        """Get bid quantity at price."""
        data = self.bids.get(price)
        return data[0] if data else 0
    
    def get_ask_qty(self, price: float) -> int:
        """Get ask quantity at price."""
        data = self.asks.get(price)
        return data[0] if data else 0
    
    def get_bid_orders(self, price: float) -> int:
        """Get bid order count at price."""
        data = self.bids.get(price)
        return data[1] if data else 0
    
    def get_ask_orders(self, price: float) -> int:
        """Get ask order count at price."""
        data = self.asks.get(price)
        return data[1] if data else 0


@dataclass
class FeatureResult:
    """
    Output from any feature calculation.
    
    Includes value, confidence, validity flag, and raw value for debugging.
    """
    value: float  # Normalized value in [-1, +1]
    confidence: float  # Confidence in [0, 1]
    valid: bool  # Whether result is valid
    raw_value: float  # Raw unnormalized value for debugging
    name: str  # Feature name
    ticks: float = 0.0  # Price deviation in ticks (if applicable)
    
    def __post_init__(self):
        # Clamp value to valid range
        if self.valid:
            self.value = max(-1.0, min(1.0, self.value))
            self.confidence = max(0.0, min(1.0, self.confidence))


@dataclass
class CompositeScore:
    """
    Composite signal score.
    
    Weighted average of all features, always normalized to [-1, +1].
    """
    value: float  # Composite value in [-1, +1]
    confidence: float  # Overall confidence
    regime: Regime
    threshold_used: float
    features: Dict[str, FeatureResult]  # Individual feature results
    timestamp: float
    samples: int  # Number of samples used
    
    @property
    def signal(self) -> SignalType:
        """Determine signal type from value."""
        if self.value > 0.3:
            return SignalType.BULLISH
        elif self.value < -0.3:
            return SignalType.BEARISH
        return SignalType.NEUTRAL
    
    def is_actionable(self, threshold: float) -> bool:
        """Check if score exceeds threshold."""
        return abs(self.value) >= threshold


@dataclass
class ExecutionSignal:
    """
    Execution-ready signal.
    
    Contains all information needed for order placement.
    """
    symbol: str
    signal_type: SignalType
    target_price: float  # Execution price (not mid, actual order price)
    confidence: float
    composite_value: float
    regime: Regime
    timestamp: float
    slippage_ticks: float
    max_depth_walk: int
    
    @property
    def direction(self) -> int:
        """Get direction as +1 (long) or -1 (short)."""
        if self.signal_type == SignalType.BULLISH:
            return 1
        elif self.signal_type == SignalType.BEARISH:
            return -1
        return 0


@dataclass
class StateTransition:
    """Record of state machine transition."""
    from_state: State
    to_state: State
    trigger: str
    composite_value: float
    timestamp: float
    reason: str = ""


@dataclass
class RollingStats:
    """Rolling statistics for a metric."""
    mean: float = 0.0
    variance: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    samples: int = 0
    
    @property
    def valid(self) -> bool:
        """Check if stats are valid."""
        return self.samples > 0


class FeatureCalculator(Protocol):
    """Protocol for feature calculators."""
    
    def calculate(
        self,
        snapshot: Snapshot,
        prev_book: Optional[PriceKeyedBook],
        stats: Optional[RollingStats]
    ) -> FeatureResult:
        """Calculate feature from snapshot."""
        ...


class BufferProtocol(Protocol):
    """Protocol for rolling buffers."""
    
    def push(self, value: float) -> None:
        """Add value to buffer."""
        ...
    
    def get_stats(self) -> RollingStats:
        """Get current statistics."""
        ...
    
    def reset(self) -> None:
        """Reset buffer state."""
        ...
    
    @property
    def full(self) -> bool:
        """Check if buffer is full."""
        ...


class DataSource(Protocol):
    """Protocol for data sources (websocket, replay, etc.)."""
    
    def connect(self) -> bool:
        """Connect to data source."""
        ...
    
    def disconnect(self) -> None:
        """Disconnect from data source."""
        ...
    
    def subscribe(self, symbols: List[str]) -> bool:
        """Subscribe to symbols."""
        ...
    
    def get_snapshot(self) -> Optional[Snapshot]:
        """Get latest snapshot."""
        ...


@dataclass
class MarketState:
    """
    Current market state for a single symbol.
    
    Maintains all rolling state needed for feature calculation.
    """
    symbol: str
    current_snapshot: Optional[Snapshot] = None
    prev_book: Optional[PriceKeyedBook] = None
    prev_prev_book: Optional[PriceKeyedBook] = None
    last_update_time: float = 0.0
    snapshot_count: int = 0
    consecutive_signals: int = 0
    last_signal_type: SignalType = SignalType.NEUTRAL
    
    # Rolling features for momentum/acceleration
    prev_composite: float = 0.0
    prev_momentum: float = 0.0
    
    def update(self, snapshot: Snapshot) -> None:
        """Update market state with new snapshot."""
        self.prev_prev_book = self.prev_book
        self.prev_book = PriceKeyedBook.from_snapshot(snapshot)
        self.current_snapshot = snapshot
        self.last_update_time = snapshot.timestamp
        self.snapshot_count += 1
    
    def reset(self) -> None:
        """Reset state (on reconnect)."""
        self.current_snapshot = None
        self.prev_book = None
        self.prev_prev_book = None
        self.snapshot_count = 0
        self.consecutive_signals = 0
        self.last_signal_type = SignalType.NEUTRAL
        self.prev_composite = 0.0
        self.prev_momentum = 0.0


@dataclass
class ValidationResult:
    """Result of snapshot validation."""
    valid: bool
    reason: str = ""
    snapshot: Optional[Snapshot] = None
