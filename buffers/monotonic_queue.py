"""
Monotonic queue for O(1) rolling min/max.

Implements deque-based monotonic queue for efficient rolling statistics.
"""

from typing import Generic, TypeVar, Optional
from dataclasses import dataclass
from collections import deque

T = TypeVar('T')


@dataclass
class MinMaxResult:
    """Result of rolling min/max calculation."""
    min_val: float
    max_val: float
    range_val: float
    count: int
    valid: bool


class MonotonicQueue(Generic[T]):
    """
    Monotonic queue for O(1) rolling min/max.
    
    Maintains elements in monotonic order, allowing O(1) access to min/max.
    
    For minimum: elements stored in increasing order
    For maximum: elements stored in decreasing order
    
    Time complexity:
    - push: O(1) amortized
    - get_min/get_max: O(1)
    - pop: O(1) amortized
    
    Usage:
        # For rolling minimum
        mq = MonotonicQueue(mode='min')
        mq.push(5, 0)
        mq.push(3, 1)
        print(mq.get())  # 3
        
        # For rolling maximum
        mq = MonotonicQueue(mode='max')
        mq.push(5, 0)
        mq.push(3, 1)
        print(mq.get())  # 5
    """
    
    def __init__(self, mode: str = 'min', capacity: int = 1000):
        if mode not in ('min', 'max'):
            raise ValueError(f"Mode must be 'min' or 'max', got {mode}")
        if capacity <= 0:
            raise ValueError(f"Capacity must be positive, got {capacity}")
        
        self._mode = mode
        self._capacity = capacity
        self._deque: deque = deque()
        self._count = 0
    
    @property
    def count(self) -> int:
        return self._count
    
    @property
    def empty(self) -> bool:
        return len(self._deque) == 0
    
    def push(self, value: T, index: int) -> None:
        """
        Push value with associated index.
        
        Removes elements that cannot be min/max in any future window.
        
        Args:
            value: Value to push
            index: Index (timestamp or sequence number) for eviction
        """
        if self._mode == 'min':
            # Remove all elements >= new value
            while self._deque and self._deque[-1][0] >= value:
                self._deque.pop()
        else:
            # Remove all elements <= new value
            while self._deque and self._deque[-1][0] <= value:
                self._deque.pop()
        
        self._deque.append((value, index))
        self._count += 1
    
    def get(self) -> Optional[T]:
        """Get current min or max value."""
        if not self._deque:
            return None
        return self._deque[0][0]
    
    def pop_expired(self, oldest_valid_index: int) -> None:
        """
        Remove elements that are outside the window.
        
        Args:
            oldest_valid_index: Elements with index < this should be removed
        """
        while self._deque and self._deque[0][1] < oldest_valid_index:
            self._deque.popleft()
    
    def clear(self) -> None:
        """Clear the queue."""
        self._deque.clear()
        self._count = 0
    
    def __len__(self) -> int:
        return len(self._deque)
    
    def __repr__(self) -> str:
        return f"MonotonicQueue(mode={self._mode}, size={len(self._deque)})"


class RollingMinMax:
    """
    Rolling minimum and maximum with O(1) operations.
    
    Uses two monotonic queues internally.
    
    Usage:
        rmm = RollingMinMax(window_size=100)
        rmm.update(1.5)
        rmm.update(2.3)
        print(f"Min: {rmm.min}, Max: {rmm.max}")
    """
    
    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError(f"Window size must be positive, got {window_size}")
        
        self._window_size = window_size
        self._min_queue = MonotonicQueue[float](mode='min', capacity=window_size)
        self._max_queue = MonotonicQueue[float](mode='max', capacity=window_size)
        self._buffer: list = []
        self._index = 0
        self._count = 0
    
    @property
    def window_size(self) -> int:
        return self._window_size
    
    @property
    def min(self) -> float:
        """Get current minimum."""
        result = self._min_queue.get()
        return result if result is not None else 0.0
    
    @property
    def max(self) -> float:
        """Get current maximum."""
        result = self._max_queue.get()
        return result if result is not None else 0.0
    
    @property
    def range(self) -> float:
        """Get current range (max - min)."""
        return self.max - self.min
    
    @property
    def mid_range(self) -> float:
        """Get mid of range."""
        return (self.min + self.max) / 2.0
    
    @property
    def count(self) -> int:
        return self._count
    
    @property
    def full(self) -> bool:
        return self._count >= self._window_size
    
    @property
    def valid(self) -> bool:
        return self._count > 0
    
    def update(self, value: float) -> MinMaxResult:
        """
        Update with new value.
        
        O(1) amortized operation.
        
        Returns:
            MinMaxResult with current state
        """
        # Store value in circular buffer
        if len(self._buffer) < self._window_size:
            self._buffer.append(value)
        else:
            self._buffer[self._index % self._window_size] = value
        
        # Push to monotonic queues
        self._min_queue.push(value, self._index)
        self._max_queue.push(value, self._index)
        
        self._index += 1
        self._count = min(self._count + 1, self._window_size)
        
        # Remove expired elements
        if self._count >= self._window_size:
            oldest_valid_index = self._index - self._window_size
            self._min_queue.pop_expired(oldest_valid_index)
            self._max_queue.pop_expired(oldest_valid_index)
        
        return MinMaxResult(
            min_val=self.min,
            max_val=self.max,
            range_val=self.range,
            count=self._count,
            valid=self.valid
        )
    
    def get_result(self) -> MinMaxResult:
        """Get current result without updating."""
        return MinMaxResult(
            min_val=self.min,
            max_val=self.max,
            range_val=self.range,
            count=self._count,
            valid=self.valid
        )
    
    def reset(self) -> None:
        """Reset to initial state."""
        self._min_queue.clear()
        self._max_queue.clear()
        self._buffer.clear()
        self._index = 0
        self._count = 0
    
    def __repr__(self) -> str:
        return (f"RollingMinMax(window_size={self._window_size}, "
                f"min={self.min:.4f}, max={self.max:.4f}, count={self._count})")


class RollingPercentile:
    """
    Rolling percentile calculator.
    
    WARNING: This is O(n log n), NOT O(1)!
    
    For exact percentiles, we maintain a sorted list with bisect.
    This is intentional - O(1) percentile algorithms like P² or KLL
    are approximate and complex to implement correctly.
    
    For production with large windows, consider:
    - P² algorithm for streaming quantiles
    - KLL sketch for bounded memory
    - T-Digest for distributed systems
    
    This implementation is suitable for small windows (< 1000).
    """
    
    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError(f"Window size must be positive, got {window_size}")
        
        self._window_size = window_size
        self._sorted_values: list = []
        self._buffer: list = [None] * window_size
        self._index = 0
        self._count = 0
    
    def update(self, value: float) -> None:
        """Update with new value (O(n log n) for exact implementation)."""
        import bisect
        
        # If buffer is full, remove oldest
        if self._count >= self._window_size:
            oldest = self._buffer[self._index % self._window_size]
            if oldest is not None:
                # Remove from sorted list
                idx = bisect.bisect_left(self._sorted_values, oldest)
                if idx < len(self._sorted_values) and self._sorted_values[idx] == oldest:
                    self._sorted_values.pop(idx)
        
        # Add new value
        bisect.insort(self._sorted_values, value)
        self._buffer[self._index % self._window_size] = value
        self._index += 1
        self._count = min(self._count + 1, self._window_size)
    
    def percentile(self, p: float) -> float:
        """Get value at percentile p (0-100)."""
        if not self._sorted_values:
            return 0.0
        
        p = max(0, min(100, p))
        idx = int(len(self._sorted_values) * p / 100)
        idx = min(idx, len(self._sorted_values) - 1)
        return self._sorted_values[idx]
    
    @property
    def median(self) -> float:
        return self.percentile(50)
    
    @property
    def q1(self) -> float:
        return self.percentile(25)
    
    @property
    def q3(self) -> float:
        return self.percentile(75)
    
    @property
    def iqr(self) -> float:
        return self.q3 - self.q1
    
    def reset(self) -> None:
        self._sorted_values = []
        self._buffer = [None] * self._window_size
        self._index = 0
        self._count = 0
