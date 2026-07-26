"""
Rolling EMA with time-aware decay.

Implements proper exponential moving average with time-based alpha.
"""

from typing import Optional
from dataclasses import dataclass
import math


@dataclass
class EMAResult:
    """Result of EMA calculation."""
    value: float
    alpha_used: float
    dt: float
    valid: bool


class RollingEMA:
    """
    Time-aware Exponential Moving Average.
    
    Uses time-based alpha calculation:
        alpha = 1 - exp(-dt / tau)
        ema = alpha * new_value + (1 - alpha) * ema
    
    This ensures consistent smoothing regardless of sampling rate.
    
    Usage:
        ema = RollingEMA(tau=10.0)  # 10 second time constant
        ema.update(1.5, timestamp=100.0)
        ema.update(2.3, timestamp=101.5)
        print(ema.value)
    """
    
    def __init__(self, tau: float):
        if tau <= 0:
            raise ValueError(f"Tau must be positive, got {tau}")
        
        self._tau = tau
        self._ema: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._initialized = False
        self._count = 0
    
    @property
    def tau(self) -> float:
        return self._tau
    
    @property
    def value(self) -> float:
        """Get current EMA value."""
        return self._ema if self._ema is not None else 0.0
    
    @property
    def valid(self) -> bool:
        return self._initialized
    
    @property
    def count(self) -> int:
        return self._count
    
    def update(self, value: float, timestamp: float) -> EMAResult:
        """
        Update EMA with time-aware decay.
        
        Args:
            value: New value
            timestamp: Current timestamp in seconds
        
        Returns:
            EMAResult with current state
        """
        if not self._initialized:
            self._ema = value
            self._last_timestamp = timestamp
            self._initialized = True
            self._count = 1
            return EMAResult(
                value=self._ema,
                alpha_used=1.0,
                dt=0.0,
                valid=True
            )
        
        # Calculate time delta
        dt = timestamp - self._last_timestamp
        if dt < 0:
            # Timestamp went backwards - reset
            self._ema = value
            self._last_timestamp = timestamp
            return EMAResult(
                value=self._ema,
                alpha_used=1.0,
                dt=dt,
                valid=True
            )
        
        # Calculate alpha
        alpha = 1.0 - math.exp(-dt / self._tau)
        
        # Update EMA
        self._ema = alpha * value + (1.0 - alpha) * self._ema
        self._last_timestamp = timestamp
        self._count += 1
        
        return EMAResult(
            value=self._ema,
            alpha_used=alpha,
            dt=dt,
            valid=True
        )
    
    def force_update(self, value: float) -> EMAResult:
        """
        Force update with alpha = 1 / (count + 1).
        
        Used when timestamp is not available.
        """
        self._count += 1
        alpha = 1.0 / self._count
        
        if self._ema is None:
            self._ema = value
            self._initialized = True
        else:
            self._ema = alpha * value + (1.0 - alpha) * self._ema
        
        return EMAResult(
            value=self._ema,
            alpha_used=alpha,
            dt=0.0,
            valid=True
        )
    
    def get_result(self) -> EMAResult:
        """Get current result without updating."""
        return EMAResult(
            value=self.value,
            alpha_used=0.0,
            dt=0.0,
            valid=self._initialized
        )
    
    def reset(self) -> None:
        """Reset to initial state."""
        self._ema = None
        self._last_timestamp = None
        self._initialized = False
        self._count = 0
    
    def __repr__(self) -> str:
        return f"RollingEMA(tau={self._tau}, value={self.value:.4f}, valid={self._initialized})"


class DualEMA:
    """
    Dual EMA for crossover signals.
    
    Maintains two EMAs with different time constants.
    Crossover can indicate momentum direction.
    
    Usage:
        dema = DualEMA(fast_tau=5.0, slow_tau=20.0)
        dema.update(1.5, timestamp=100.0)
        if dema.crossover == 1:
            print("Fast crossed above slow - bullish")
    """
    
    def __init__(self, fast_tau: float, slow_tau: float):
        if fast_tau <= 0 or slow_tau <= 0:
            raise ValueError("Taus must be positive")
        if fast_tau >= slow_tau:
            raise ValueError(f"fast_tau ({fast_tau}) must be less than slow_tau ({slow_tau})")
        
        self._fast_ema = RollingEMA(fast_tau)
        self._slow_ema = RollingEMA(slow_tau)
        self._prev_fast = None
        self._prev_slow = None
    
    @property
    def fast(self) -> float:
        return self._fast_ema.value
    
    @property
    def slow(self) -> float:
        return self._slow_ema.value
    
    @property
    def crossover(self) -> int:
        """
        Get crossover signal.
        
        Returns:
            1 if fast just crossed above slow (bullish)
            -1 if fast just crossed below slow (bearish)
            0 otherwise
        """
        if not self._fast_ema.valid or not self._slow_ema.valid:
            return 0
        if self._prev_fast is None or self._prev_slow is None:
            return 0
        
        # Check for crossover
        if self._prev_fast <= self._prev_slow and self._fast_ema.value > self._slow_ema.value:
            return 1  # Bullish crossover
        elif self._prev_fast >= self._prev_slow and self._fast_ema.value < self._slow_ema.value:
            return -1  # Bearish crossover
        
        return 0
    
    @property
    def spread(self) -> float:
        """Get spread between fast and slow EMA."""
        return self._fast_ema.value - self._slow_ema.value
    
    @property
    def valid(self) -> bool:
        return self._fast_ema.valid and self._slow_ema.valid
    
    def update(self, value: float, timestamp: float) -> tuple:
        """
        Update both EMAs.
        
        Returns:
            (fast_result, slow_result, crossover)
        """
        self._prev_fast = self._fast_ema.value if self._fast_ema.valid else None
        self._prev_slow = self._slow_ema.value if self._slow_ema.valid else None
        
        fast_result = self._fast_ema.update(value, timestamp)
        slow_result = self._slow_ema.update(value, timestamp)
        
        return fast_result, slow_result, self.crossover
    
    def reset(self) -> None:
        """Reset both EMAs."""
        self._fast_ema.reset()
        self._slow_ema.reset()
        self._prev_fast = None
        self._prev_slow = None
    
    def __repr__(self) -> str:
        return (f"DualEMA(fast={self.fast:.4f}, slow={self.slow:.4f}, "
                f"spread={self.spread:.4f}, valid={self.valid})")


class TripleEMA:
    """
    Triple EMA for trend detection with reduced lag.
    
    TEMA = 3*EMA1 - 3*EMA2 + EMA3
    where EMA1, EMA2, EMA3 are EMAs of EMAs.
    
    Usage:
        tema = TripleEMA(tau=10.0)
        tema.update(1.5, timestamp=100.0)
        print(f"TEMA: {tema.value}")
    """
    
    def __init__(self, tau: float):
        if tau <= 0:
            raise ValueError(f"Tau must be positive, got {tau}")
        
        self._tau = tau
        self._ema1 = RollingEMA(tau)
        self._ema2 = RollingEMA(tau)
        self._ema3 = RollingEMA(tau)
    
    @property
    def value(self) -> float:
        """Get TEMA value."""
        if not self._ema1.valid:
            return 0.0
        
        e1 = self._ema1.value
        e2 = self._ema2.value if self._ema2.valid else e1
        e3 = self._ema3.value if self._ema3.valid else e2
        
        return 3 * e1 - 3 * e2 + e3
    
    @property
    def valid(self) -> bool:
        return self._ema1.valid
    
    def update(self, value: float, timestamp: float) -> float:
        """
        Update TEMA.
        
        Returns TEMA value.
        """
        # Update first EMA
        self._ema1.update(value, timestamp)
        
        # Update second EMA (EMA of EMA)
        if self._ema1.valid:
            self._ema2.update(self._ema1.value, timestamp)
        
        # Update third EMA (EMA of EMA of EMA)
        if self._ema2.valid:
            self._ema3.update(self._ema2.value, timestamp)
        
        return self.value
    
    def reset(self) -> None:
        """Reset all EMAs."""
        self._ema1.reset()
        self._ema2.reset()
        self._ema3.reset()
    
    def __repr__(self) -> str:
        return f"TripleEMA(tau={self._tau}, value={self.value:.4f}, valid={self.valid})"
