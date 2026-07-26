"""
Type definitions for Snapshot Quant Engine.

Protocol definitions, dataclasses, and type aliases for the entire system.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol, Dict, Optional, Tuple, Any
from enum import Enum, auto
from numbers import Real
import math


def _is_finite_number(value: Any) -> bool:
    """Return whether a runtime value is a finite, non-boolean real number."""
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _is_nonnegative_int(value: Any) -> bool:
    """Return whether a runtime value is a non-boolean non-negative integer."""
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


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
    """Execution command types."""
    BULLISH = auto()
    BEARISH = auto()
    EXIT_LONG = auto()
    EXIT_SHORT = auto()
    NEUTRAL = auto()


class SnapshotDeliveryMode(Enum):
    """Exclusive snapshot delivery path selected for an adapter instance."""
    CALLBACK = auto()
    PULL = auto()


@dataclass(frozen=True)
class MarketSubscription:
    """Validated SmartAPI exchange segment and numeric instrument token."""

    SUPPORTED_EXCHANGE_TYPES = frozenset({1, 2, 3, 4, 5, 7, 13})

    exchange_type: int
    token: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.exchange_type, bool)
            or not isinstance(self.exchange_type, int)
            or self.exchange_type not in self.SUPPORTED_EXCHANGE_TYPES
        ):
            supported = ", ".join(
                str(value) for value in sorted(self.SUPPORTED_EXCHANGE_TYPES)
            )
            raise ValueError(f"exchange_type must be one of: {supported}")
        if not isinstance(self.token, str):
            raise TypeError("token must be a string")
        normalized_token = self.token.strip()
        if not normalized_token.isdecimal() or int(normalized_token) <= 0:
            raise ValueError("token must be a positive numeric string")
        object.__setattr__(self, "token", normalized_token)

    @classmethod
    def from_config(cls, value: Any) -> "MarketSubscription":
        """Convert a typed object, configuration mapping, or two-item pair."""
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            exchange_type = value.get("exchange_type", value.get("exchangeType"))
            token = value.get("token")
            if not isinstance(token, str):
                raise TypeError("subscription token must be a string")
            return cls(exchange_type=exchange_type, token=token)
        if (
            isinstance(value, Sequence)
            and not isinstance(value, (str, bytes, bytearray))
            and len(value) == 2
        ):
            return cls(exchange_type=value[0], token=value[1])
        raise TypeError("subscription must provide exchange_type and token")

    def as_dict(self) -> Dict[str, object]:
        """Return the snake-case representation used by local configuration."""
        return {"exchange_type": self.exchange_type, "token": self.token}


@dataclass(frozen=True)
class PriceLevel:
    """Single price level in order book."""
    price: float
    quantity: int
    order_count: int = 0

    def __post_init__(self) -> None:
        if not _is_finite_number(self.price) or self.price < 0:
            raise ValueError(f"Price must be finite and non-negative: {self.price}")
        if not _is_nonnegative_int(self.quantity):
            raise ValueError(f"Quantity must be a non-negative integer: {self.quantity}")
        if not _is_nonnegative_int(self.order_count):
            raise ValueError(f"Order count must be a non-negative integer: {self.order_count}")
        if self.order_count > self.quantity:
            raise ValueError("Order count cannot exceed quantity")


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
    token: str = ""  # SmartAPI numeric instrument token, when available
    exchange_type: int = 0  # SmartAPI exchange segment, 0 for non-live data
    ltp_quantity: int = 0  # Last traded quantity (if available)
    volume_traded: int = 0  # Total volume for the day
    total_buy_qty: int = 0  # Total buy quantity (if available)
    total_sell_qty: int = 0  # Total sell quantity (if available)
    bids: Tuple[PriceLevel, ...] = field(default_factory=tuple)  # Bid levels (0 = best)
    asks: Tuple[PriceLevel, ...] = field(default_factory=tuple)  # Ask levels (0 = best)
    sequence: int = 0  # Sequence number for ordering
    exchange_timestamp: float = 0.0  # Exchange timestamp if available

    @property
    def instrument_key(self) -> str:
        """Return an exchange-qualified key for mutable engine state."""
        if self.exchange_type and self.token:
            return f"{self.exchange_type}:{self.token}"
        return self.symbol.strip()

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
        """Check scalar fields and the complete ordered depth book."""
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            return False
        if not isinstance(self.token, str):
            return False
        if self.token and (not self.token.isdecimal() or int(self.token) <= 0):
            return False
        if self.exchange_type and not self.token:
            return False
        if (
            isinstance(self.exchange_type, bool)
            or not isinstance(self.exchange_type, int)
            or (
                self.exchange_type != 0
                and self.exchange_type not in MarketSubscription.SUPPORTED_EXCHANGE_TYPES
            )
        ):
            return False

        if not _is_finite_number(self.ltp) or self.ltp <= 0:
            return False
        if not _is_finite_number(self.timestamp):
            return False
        if not _is_finite_number(self.exchange_timestamp):
            return False

        integer_fields = (
            self.ltp_quantity,
            self.volume_traded,
            self.total_buy_qty,
            self.total_sell_qty,
            self.sequence,
        )
        if not all(_is_nonnegative_int(value) for value in integer_fields):
            return False

        if not isinstance(self.bids, tuple) or not isinstance(self.asks, tuple):
            return False
        if not self.bids or not self.asks:
            return False

        for levels in (self.bids, self.asks):
            for level in levels:
                if not isinstance(level, PriceLevel):
                    return False
                if not _is_finite_number(level.price) or level.price <= 0:
                    return False
                if not _is_nonnegative_int(level.quantity) or level.quantity <= 0:
                    return False
                if not _is_nonnegative_int(level.order_count):
                    return False
                if level.order_count > level.quantity:
                    return False

        if any(
            self.bids[index].price <= self.bids[index + 1].price
            for index in range(len(self.bids) - 1)
        ):
            return False
        if any(
            self.asks[index].price >= self.asks[index + 1].price
            for index in range(len(self.asks) - 1)
        ):
            return False

        if self.best_bid is None or self.best_ask is None:
            return False
        if self.best_bid.price >= self.best_ask.price:
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
    token: str = ""
    exchange_type: int = 0

    @property
    def instrument_key(self) -> str:
        """Return the exchange-qualified identifier used for order routing."""
        if self.exchange_type and self.token:
            return f"{self.exchange_type}:{self.token}"
        return self.symbol

    @property
    def direction(self) -> int:
        """Get order side as +1 (buy) or -1 (sell)."""
        if self.signal_type in (SignalType.BULLISH, SignalType.EXIT_SHORT):
            return 1
        if self.signal_type in (SignalType.BEARISH, SignalType.EXIT_LONG):
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
    """Protocol for data sources with one exclusive snapshot delivery path."""

    @property
    def delivery_mode(self) -> SnapshotDeliveryMode:
        """Return the selected snapshot delivery mode."""
        ...

    def connect(self) -> bool:
        """Connect to data source."""
        ...

    def disconnect(self) -> None:
        """Disconnect from data source."""
        ...

    def subscribe(self, subscriptions: Sequence[MarketSubscription]) -> bool:
        """Subscribe to validated exchange/token pairs."""
        ...

    def get_snapshot(self, timeout: float = 1.0) -> Optional[Snapshot]:
        """Get a snapshot only when pull delivery is selected."""
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
