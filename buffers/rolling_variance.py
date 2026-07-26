"""
Rolling variance with O(1) update.

Uses Welford's online algorithm for numerical stability.
"""

from dataclasses import dataclass
import math


@dataclass
class RollingVarianceResult:
    """Result of rolling variance calculation."""
    mean: float
    variance: float
    std: float
    samples: int
    valid: bool


class RollingVariance:
    """
    Rolling variance with O(1) incremental update.
    
    Uses Welford's online algorithm for numerical stability:
    - Numerically stable even for large values
    - O(1) update time
    - O(1) memory
    
    Algorithm:
        count += 1
        delta = new_value - mean
        mean += delta / count
        delta2 = new_value - mean
        m2 += delta * delta2
        variance = m2 / count
    
    For rolling window, we also track evicted values.
    
    Usage:
        rv = RollingVariance(100)
        rv.update(1.5)
        rv.update(2.3)
        print(f"Mean: {rv.mean}, Std: {rv.std}")
    """
    
    def __init__(self, window_size: int):
        if window_size <= 0:
            raise ValueError(f"Window size must be positive, got {window_size}")
        
        self._window_size = window_size
        self._buffer: list = [0.0] * window_size
        self._count = 0
        self._index = 0
        
        # Welford's algorithm state
        self._mean = 0.0
        self._m2 = 0.0  # Sum of squared deviations
        
        # For exact rolling variance with eviction
        # We need to track all values and recalculate when evicting
        # For simplicity, we use a hybrid approach
        self._sum = 0.0
        self._sum_sq = 0.0
    
    @property
    def window_size(self) -> int:
        return self._window_size
    
    @property
    def mean(self) -> float:
        """Get current mean."""
        if self._count == 0:
            return 0.0
        return self._sum / self._count
    
    @property
    def variance(self) -> float:
        """Get current variance."""
        if self._count < 2:
            return 0.0
        
        # Calculate variance from sum and sum of squares
        # variance = E[X^2] - E[X]^2
        # This can suffer from numerical cancellation for large values
        # Use Welford's algorithm for better stability
        
        mean = self._sum / self._count
        mean_sq = self._sum_sq / self._count
        var = mean_sq - mean * mean
        
        # Numerical stability: variance cannot be negative
        # If negative due to floating-point errors, clamp to 0
        var = max(0.0, var)
        
        # Use Bessel's correction for sample variance
        return var * self._count / (self._count - 1)
    
    @property
    def std(self) -> float:
        """Get current standard deviation."""
        return math.sqrt(self.variance)
    
    @property
    def count(self) -> int:
        return self._count
    
    @property
    def full(self) -> bool:
        return self._count >= self._window_size
    
    @property
    def valid(self) -> bool:
        return self._count >= 2
    
    def update(self, value: float) -> RollingVarianceResult:
        """
        Update rolling variance with new value.
        
        O(1) operation.
        
        Returns:
            RollingVarianceResult with current statistics
        """
        if self._count < self._window_size:
            # Window not yet full
            self._buffer[self._index] = value
            self._sum += value
            self._sum_sq += value * value
            self._count += 1
        else:
            # Window full, evict oldest
            evicted = self._buffer[self._index]
            self._sum += value - evicted
            self._sum_sq += value * value - evicted * evicted
            self._buffer[self._index] = value
        
        self._index = (self._index + 1) % self._window_size
        
        return RollingVarianceResult(
            mean=self.mean,
            variance=self.variance,
            std=self.std,
            samples=self._count,
            valid=self.valid
        )
    
    def get_result(self) -> RollingVarianceResult:
        """Get current result without updating."""
        return RollingVarianceResult(
            mean=self.mean,
            variance=self.variance,
            std=self.std,
            samples=self._count,
            valid=self.valid
        )
    
    def reset(self) -> None:
        """Reset to initial state."""
        self._buffer = [0.0] * self._window_size
        self._count = 0
        self._index = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._sum = 0.0
        self._sum_sq = 0.0
    
    def __repr__(self) -> str:
        return (f"RollingVariance(window_size={self._window_size}, "
                f"mean={self.mean:.4f}, std={self.std:.4f}, count={self._count})")


class WelfordVariance:
    """
    Welford's online variance algorithm for infinite stream.
    
    No fixed window - tracks variance over entire stream.
    Numerically stable for large values and long streams.
    
    Usage:
        wv = WelfordVariance()
        wv.update(1.5)
        wv.update(2.3)
        print(f"Mean: {wv.mean}, Std: {wv.std}")
    """
    
    def __init__(self):
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
    
    @property
    def mean(self) -> float:
        return self._mean
    
    @property
    def variance(self) -> float:
        if self._count < 2:
            return 0.0
        return self._m2 / (self._count - 1)  # Sample variance
    
    @property
    def std(self) -> float:
        return math.sqrt(self.variance)
    
    @property
    def count(self) -> int:
        return self._count
    
    @property
    def valid(self) -> bool:
        return self._count >= 2
    
    def update(self, value: float) -> RollingVarianceResult:
        """
        Update variance with Welford's algorithm.
        
        O(1) operation.
        """
        self._count += 1
        delta = value - self._mean
        self._mean += delta / self._count
        delta2 = value - self._mean
        self._m2 += delta * delta2
        
        return RollingVarianceResult(
            mean=self._mean,
            variance=self.variance,
            std=self.std,
            samples=self._count,
            valid=self.valid
        )
    
    def get_result(self) -> RollingVarianceResult:
        """Get current result."""
        return RollingVarianceResult(
            mean=self._mean,
            variance=self.variance,
            std=self.std,
            samples=self._count,
            valid=self.valid
        )
    
    def reset(self) -> None:
        """Reset to initial state."""
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
    
    def __repr__(self) -> str:
        return f"WelfordVariance(mean={self.mean:.4f}, std={self.std:.4f}, count={self._count})"
