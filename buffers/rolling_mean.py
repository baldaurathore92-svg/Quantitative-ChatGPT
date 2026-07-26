"""
Rolling mean with O(1) update.

Incremental computation without scanning entire buffer.
"""

from typing import Optional
from dataclasses import dataclass


@dataclass
class RollingMeanResult:
    """Result of rolling mean calculation."""
    value: float
    samples: int
    valid: bool


class RollingMean:
    """
    Rolling mean with O(1) incremental update.
    
    Never scans complete buffer.
    Uses simple incremental mean update:
        mean = mean + (new_value - old_value) / n
    
    Usage:
        rm = RollingMean(20)
        rm.update(1.5)
        rm.update(2.3)
        print(rm.value)  # O(1)
    """
    
    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError(f"Window size must be positive, got {window_size}")
        
        self._window_size = window_size
        self._buffer: list = [0.0] * window_size
        self._sum = 0.0
        self._count = 0
        self._index = 0  # Current write position
    
    @property
    def window_size(self) -> int:
        return self._window_size
    
    @property
    def value(self) -> float:
        """Get current mean (O(1))."""
        if self._count == 0:
            return 0.0
        return self._sum / self._count
    
    @property
    def count(self) -> int:
        """Get current sample count."""
        return self._count
    
    @property
    def full(self) -> bool:
        """Check if window is full."""
        return self._count >= self._window_size
    
    @property
    def valid(self) -> bool:
        """Check if mean is valid (at least one sample)."""
        return self._count > 0
    
    def update(self, value: float) -> RollingMeanResult:
        """
        Update rolling mean with new value.
        
        O(1) operation.
        
        Returns:
            RollingMeanResult with current state
        """
        if self._count < self._window_size:
            # Window not yet full
            self._buffer[self._index] = value
            self._sum += value
            self._count += 1
        else:
            # Window full, evict oldest
            evicted = self._buffer[self._index]
            self._sum += value - evicted
            self._buffer[self._index] = value
        
        self._index = (self._index + 1) % self._window_size
        
        return RollingMeanResult(
            value=self.value,
            samples=self._count,
            valid=self.valid
        )
    
    def get_result(self) -> RollingMeanResult:
        """Get current result without updating."""
        return RollingMeanResult(
            value=self.value,
            samples=self._count,
            valid=self.valid
        )
    
    def reset(self) -> None:
        """Reset to initial state."""
        self._buffer = [0.0] * self._window_size
        self._sum = 0.0
        self._count = 0
        self._index = 0
    
    def __repr__(self) -> str:
        return f"RollingMean(window_size={self._window_size}, value={self.value:.4f}, count={self._count})"


class TimeAwareRollingMean:
    """
    Time-aware rolling mean with exponential decay.
    
    Instead of a fixed window, uses time-based exponential decay.
    Older values have less weight.
    
    alpha = 1 - exp(-dt / tau)
    mean = alpha * new_value + (1 - alpha) * mean
    
    Usage:
        rm = TimeAwareRollingMean(tau=10.0)  # 10 second time constant
        rm.update(1.5, timestamp=100.0)
        rm.update(2.3, timestamp=101.0)
        print(rm.value)
    """
    
    def __init__(self, tau: float):
        if tau <= 0:
            raise ValueError(f"Tau must be positive, got {tau}")
        
        self._tau = tau
        self._mean = 0.0
        self._last_timestamp: Optional[float] = None
        self._initialized = False
        self._count = 0
    
    @property
    def tau(self) -> float:
        return self._tau
    
    @property
    def value(self) -> float:
        """Get current mean."""
        return self._mean
    
    @property
    def count(self) -> int:
        return self._count
    
    @property
    def valid(self) -> bool:
        return self._initialized
    
    def update(self, value: float, timestamp: float) -> RollingMeanResult:
        """
        Update with time-aware exponential decay.
        
        Args:
            value: New value
            timestamp: Current timestamp in seconds
        
        Returns:
            RollingMeanResult with current state
        """
        if not self._initialized:
            self._mean = value
            self._last_timestamp = timestamp
            self._initialized = True
            self._count = 1
        else:
            # Calculate time delta after initialization established the timestamp.
            last_timestamp = self._last_timestamp
            if last_timestamp is None:
                raise RuntimeError("initialized rolling mean has no timestamp")
            dt = timestamp - last_timestamp
            if dt < 0:
                # Timestamp went backwards - use small positive dt
                dt = 0.001
            
            # Calculate alpha
            import math
            alpha = 1.0 - math.exp(-dt / self._tau)
            
            # Update mean
            self._mean = alpha * value + (1.0 - alpha) * self._mean
            self._last_timestamp = timestamp
            self._count += 1
        
        return RollingMeanResult(
            value=self._mean,
            samples=self._count,
            valid=self._initialized
        )
    
    def reset(self) -> None:
        """Reset to initial state."""
        self._mean = 0.0
        self._last_timestamp = None
        self._initialized = False
        self._count = 0
    
    def __repr__(self) -> str:
        return f"TimeAwareRollingMean(tau={self._tau}, value={self.value:.4f}, valid={self._initialized})"
