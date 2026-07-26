"""
Ring buffer implementation for O(1) rolling statistics.

Never uses del or rebuilds lists inside processing loop.
All operations are O(1) amortized.
"""

from typing import Generic, TypeVar, Optional, Iterator, List, cast
from dataclasses import dataclass
import math
import threading

T = TypeVar('T')


@dataclass
class BufferStats:
    """Statistics for numeric ring buffer."""
    mean: float = 0.0
    variance: float = 0.0
    std: float = 0.0
    min: float = 0.0
    max: float = 0.0
    sum: float = 0.0
    count: int = 0

    @property
    def valid(self) -> bool:
        return self.count > 0


class RingBuffer(Generic[T]):
    """
    Thread-safe ring buffer with O(1) push operation.

    Features:
    - Fixed capacity
    - O(1) push
    - O(1) access by index
    - O(1) is_full check
    - No allocations after initialization
    - Thread-safe for single producer, single consumer

    Usage:
        buffer = RingBuffer[float](100)
        buffer.push(1.5)
        if buffer.full:
            avg = sum(buffer) / len(buffer)
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")

        self._capacity = capacity
        self._buffer: List[Optional[T]] = [None] * capacity
        self._head = 0  # Next write position
        self._count = 0  # Number of elements
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        """Get buffer capacity."""
        return self._capacity

    @property
    def count(self) -> int:
        """Get current number of elements."""
        with self._lock:
            return self._count

    @property
    def full(self) -> bool:
        """Check if buffer is full."""
        with self._lock:
            return self._count >= self._capacity

    @property
    def empty(self) -> bool:
        """Check if buffer is empty."""
        with self._lock:
            return self._count == 0

    def push(self, value: T) -> Optional[T]:
        """
        Push value to buffer.

        Returns:
            Evicted value if buffer was full, None otherwise
        """
        with self._lock:
            evicted = None

            if self._count >= self._capacity:
                evicted = self._buffer[self._head]

            self._buffer[self._head] = value
            self._head = (self._head + 1) % self._capacity

            if self._count < self._capacity:
                self._count += 1

            return evicted

    def get(self, index: int) -> Optional[T]:
        """
        Get element by index (0 = oldest, count-1 = newest).

        Returns None if index out of range.
        """
        with self._lock:
            if index < 0 or index >= self._count:
                return None

            if self._count < self._capacity:
                return self._buffer[index]

            actual_index = (self._head + index) % self._capacity
            return self._buffer[actual_index]

    def get_latest(self) -> Optional[T]:
        """Get most recently added element."""
        with self._lock:
            if self._count == 0:
                return None
            latest_index = (self._head - 1 + self._capacity) % self._capacity
            return self._buffer[latest_index]

    def get_oldest(self) -> Optional[T]:
        """Get oldest element."""
        return self.get(0)

    def clear(self) -> None:
        """Clear all elements."""
        with self._lock:
            self._buffer = [None] * self._capacity
            self._head = 0
            self._count = 0

    def __len__(self) -> int:
        with self._lock:
            return self._count

    def __iter__(self) -> Iterator[T]:
        """Iterate over a stable snapshot from oldest to newest."""
        with self._lock:
            start = self._head if self._count == self._capacity else 0
            values = [
                cast(T, self._buffer[(start + i) % self._capacity])
                for i in range(self._count)
            ]
        return iter(values)

    def to_list(self) -> List[T]:
        """Convert to list (oldest to newest)."""
        return list(self)

    def __repr__(self) -> str:
        return f"RingBuffer(capacity={self._capacity}, count={self._count})"


class NumericRingBuffer(RingBuffer[float]):
    """
    Ring buffer optimized for numeric values.

    Provides O(1) rolling statistics through incremental updates.
    Never scans complete buffer.

    The variance uses moments relative to a nearby origin rather than raw
    ``sum(x*x)``. This preserves precision for prices with a large absolute
    value and a small rolling variance while still supporting exact O(1)
    addition and eviction.
    """

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._sum = 0.0
        self._origin: Optional[float] = None
        self._centered_sum = 0.0
        self._centered_sum_squares = 0.0

    def push(self, value: float) -> Optional[float]:
        """Push a finite value and update exact rolling statistics in O(1)."""
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("value must be a finite number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("value must be a finite number")

        with self._lock:
            evicted: Optional[float] = None

            if self._count == self._capacity:
                stored = self._buffer[self._head]
                if stored is None:
                    raise RuntimeError("Full ring buffer contains an empty slot")
                evicted = float(stored)
                self._remove_value(evicted)
                self._count -= 1

                # Keep the origin close to the retained rolling window. The
                # moment translation itself is O(1).
                if self._count > 0:
                    next_oldest = self._buffer[(self._head + 1) % self._capacity]
                    if next_oldest is None:
                        raise RuntimeError("Ring buffer contains an empty retained slot")
                    self._rebase(float(next_oldest))

            self._buffer[self._head] = value
            self._head = (self._head + 1) % self._capacity
            self._add_value(value)
            self._count += 1

            return evicted

    def _add_value(self, value: float) -> None:
        """Add one value to the running moments; caller holds ``_lock``."""
        if self._count == 0 or self._origin is None:
            self._origin = value
            deviation = 0.0
        else:
            deviation = value - self._origin

        self._sum += value
        self._centered_sum += deviation
        self._centered_sum_squares += deviation * deviation

    def _remove_value(self, value: float) -> None:
        """Remove one value from the running moments; caller holds ``_lock``."""
        if self._count <= 1 or self._origin is None:
            self._reset_statistics()
            return

        deviation = value - self._origin
        self._sum -= value
        self._centered_sum -= deviation
        self._centered_sum_squares -= deviation * deviation

    def _rebase(self, new_origin: float) -> None:
        """Translate centered moments to ``new_origin`` in O(1)."""
        if self._count == 0 or self._origin is None:
            self._origin = new_origin
            self._centered_sum = 0.0
            self._centered_sum_squares = 0.0
            return

        shift = self._origin - new_origin
        old_centered_sum = self._centered_sum
        self._centered_sum_squares += (
            2.0 * shift * old_centered_sum + self._count * shift * shift
        )
        self._centered_sum = old_centered_sum + self._count * shift
        self._origin = new_origin

    def _mean_unlocked(self) -> float:
        if self._count == 0 or self._origin is None:
            return 0.0
        return self._origin + self._centered_sum / self._count

    def _variance_unlocked(self) -> float:
        if self._count < 2:
            return 0.0
        variance = (
            self._centered_sum_squares
            - self._centered_sum * self._centered_sum / self._count
        ) / self._count
        return max(0.0, variance)

    def _reset_statistics(self) -> None:
        self._sum = 0.0
        self._origin = None
        self._centered_sum = 0.0
        self._centered_sum_squares = 0.0

    def get_stats(self) -> BufferStats:
        """Get current statistics (O(1))."""
        with self._lock:
            if self._count == 0:
                return BufferStats()

            variance = self._variance_unlocked()
            return BufferStats(
                mean=self._mean_unlocked(),
                variance=variance,
                std=variance ** 0.5,
                min=0.0,  # Use MonotonicQueue for min/max
                max=0.0,
                sum=self._sum,
                count=self._count
            )

    @property
    def mean(self) -> float:
        """Get current mean (O(1))."""
        with self._lock:
            return self._mean_unlocked()

    @property
    def variance(self) -> float:
        """Get current population variance (O(1))."""
        with self._lock:
            return self._variance_unlocked()

    @property
    def std(self) -> float:
        """Get current population standard deviation (O(1))."""
        with self._lock:
            return self._variance_unlocked() ** 0.5

    @property
    def sum(self) -> float:
        """Get current sum (O(1))."""
        with self._lock:
            return self._sum

    def clear(self) -> None:
        """Clear all elements and reset statistics."""
        with self._lock:
            self._buffer = [None] * self._capacity
            self._head = 0
            self._count = 0
            self._reset_statistics()

    def reset(self) -> None:
        """Alias for clear()."""
        self.clear()
