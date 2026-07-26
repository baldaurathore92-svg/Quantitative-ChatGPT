"""
Dynamic Threshold calculation.

Threshold adapts to market conditions:
- Spread volatility
- Microprice volatility
- OBI volatility
- Queue stability
- Minimum sample requirements

The threshold determines when a composite score is actionable.
"""

from typing import Dict
from dataclasses import dataclass

from utils.math_utils import clamp
from buffers.rolling_variance import RollingVariance
from utils.constants import MIN_THRESHOLD, MAX_THRESHOLD


@dataclass
class ThresholdConfig:
    """Configuration for dynamic threshold."""
    base_threshold: float = 0.6
    min_threshold: float = MIN_THRESHOLD
    max_threshold: float = MAX_THRESHOLD
    volatility_multiplier: float = 0.2
    spread_multiplier: float = 0.15
    queue_stability_multiplier: float = 0.1
    min_samples: int = 10
    warmup_samples: int = 5
    spread_vol_window: int = 30
    obi_vol_window: int = 30
    microprice_vol_window: int = 30


class DynamicThreshold:
    """
    Calculate dynamic threshold for signal activation.
    
    Threshold = base_threshold * volatility_adjustment * spread_adjustment * queue_adjustment
    
    Higher volatility -> Higher threshold (need stronger signal)
    Wider spread -> Higher threshold (more uncertainty)
    Unstable queues -> Higher threshold (less reliable)
    
    The threshold is clamped to [min_threshold, max_threshold].
    """
    
    def __init__(self, config: ThresholdConfig):
        self._config = config
        
        # Rolling statistics for volatility
        self._spread_variance = RollingVariance(config.spread_vol_window)
        self._obi_variance = RollingVariance(config.obi_vol_window)
        self._microprice_variance = RollingVariance(config.microprice_vol_window)
        
        # Sample count
        self._sample_count = 0
    
    def calculate(
        self,
        spread_ticks: float,
        obi_value: float,
        microprice_deviation: float,
        queue_stability: float = 1.0
    ) -> float:
        """
        Calculate dynamic threshold.
        
        Args:
            spread_ticks: Current spread in ticks
            obi_value: Current OBI value
            microprice_deviation: Current microprice deviation in ticks
            queue_stability: Queue stability metric [0, 1]
        
        Returns:
            Dynamic threshold value
        """
        self._sample_count += 1
        
        # Update volatility measures
        self._spread_variance.update(spread_ticks)
        self._obi_variance.update(obi_value)
        self._microprice_variance.update(microprice_deviation)
        
        # Check if we have enough samples
        if self._sample_count < self._config.warmup_samples:
            return self._config.base_threshold
        
        # Calculate volatility adjustments
        vol_adjustment = self._calculate_volatility_adjustment()
        spread_adjustment = self._calculate_spread_adjustment(spread_ticks)
        queue_adjustment = self._calculate_queue_adjustment(queue_stability)
        
        # Combine adjustments
        threshold = self._config.base_threshold
        threshold *= vol_adjustment
        threshold *= spread_adjustment
        threshold *= queue_adjustment
        
        # Clamp to limits
        threshold = clamp(threshold, self._config.min_threshold, self._config.max_threshold)
        
        return threshold
    
    def _calculate_volatility_adjustment(self) -> float:
        """
        Calculate volatility-based adjustment.
        
        Higher volatility -> Higher threshold.
        """
        spread_vol = self._spread_variance.std if self._spread_variance.valid else 0
        obi_vol = self._obi_variance.std if self._obi_variance.valid else 0
        microprice_vol = self._microprice_variance.std if self._microprice_variance.valid else 0
        
        # Normalize volatilities
        # Spread vol: 0-5 ticks is normal
        spread_vol_norm = min(1.0, spread_vol / 5.0)
        
        # OBI vol: already normalized
        obi_vol_norm = min(1.0, obi_vol * 2)
        
        # Microprice vol: 0-5 ticks is normal
        microprice_vol_norm = min(1.0, microprice_vol / 5.0)
        
        # Combined volatility
        combined_vol = (spread_vol_norm + obi_vol_norm + microprice_vol_norm) / 3
        
        # Adjustment: higher vol = higher threshold
        # vol = 0 -> adjustment = 1.0
        # vol = 1 -> adjustment = 1 + volatility_multiplier
        adjustment = 1.0 + combined_vol * self._config.volatility_multiplier
        
        return adjustment
    
    def _calculate_spread_adjustment(self, spread_ticks: float) -> float:
        """
        Calculate spread-based adjustment.
        
        Wider spread -> Higher threshold.
        """
        # Spread: 0-5 ticks is tight, 5-20 is normal, >20 is wide
        if spread_ticks <= 2:
            adjustment = 0.9  # Tight spread, lower threshold
        elif spread_ticks <= 5:
            adjustment = 1.0  # Normal spread
        elif spread_ticks <= 10:
            adjustment = 1.0 + self._config.spread_multiplier * (spread_ticks - 5) / 5
        else:
            adjustment = 1.0 + self._config.spread_multiplier * 2  # Cap at 2x multiplier
        
        return adjustment
    
    def _calculate_queue_adjustment(self, queue_stability: float) -> float:
        """
        Calculate queue stability adjustment.
        
        Less stable queues -> Higher threshold.
        """
        # queue_stability: 0 = unstable, 1 = stable
        if queue_stability >= 0.7:
            adjustment = 0.95  # Stable queues, slightly lower threshold
        elif queue_stability >= 0.3:
            adjustment = 1.0  # Normal stability
        else:
            # Unstable queues
            instability = 1.0 - queue_stability
            adjustment = 1.0 + self._config.queue_stability_multiplier * instability
        
        return adjustment
    
    def get_current_threshold(self) -> float:
        """Get current threshold without updating."""
        if self._sample_count < self._config.warmup_samples:
            return self._config.base_threshold
        
        vol_adjustment = self._calculate_volatility_adjustment()
        threshold = self._config.base_threshold * vol_adjustment
        return clamp(threshold, self._config.min_threshold, self._config.max_threshold)
    
    def reset(self) -> None:
        """Reset all state."""
        self._spread_variance.reset()
        self._obi_variance.reset()
        self._microprice_variance.reset()
        self._sample_count = 0
    
    def get_stats(self) -> Dict:
        """Get threshold statistics."""
        return {
            'sample_count': self._sample_count,
            'current_threshold': self.get_current_threshold(),
            'spread_vol': self._spread_variance.std,
            'obi_vol': self._obi_variance.std,
            'microprice_vol': self._microprice_variance.std
        }


class RegimeThreshold:
    """
    Regime-aware threshold multiplier.
    
    Different regimes have different threshold requirements:
    - TREND: Standard threshold (trend provides confidence)
    - PULLBACK: Slightly higher threshold
    - RANGE: Higher threshold (more noise)
    - NOISE: Highest threshold (need strong signal)
    
    IMPORTANT: Only THRESHOLD multiplier changes, not feature weights.
    Uniformly scaling all feature weights has no effect on normalized composite.
    """
    
    def __init__(
        self,
        trend_mult: float = 1.0,
        pullback_mult: float = 1.1,
        range_mult: float = 1.2,
        noise_mult: float = 1.5
    ):
        self._multipliers = {
            'TREND': trend_mult,
            'PULLBACK': pullback_mult,
            'RANGE': range_mult,
            'NOISE': noise_mult
        }
    
    def get_multiplier(self, regime: str) -> float:
        """Get threshold multiplier for regime."""
        return self._multipliers.get(regime.upper(), 1.0)
    
    def apply_regime_threshold(
        self,
        base_threshold: float,
        regime: str,
        min_threshold: float = MIN_THRESHOLD,
        max_threshold: float = MAX_THRESHOLD
    ) -> float:
        """Apply regime multiplier to base threshold."""
        multiplier = self.get_multiplier(regime)
        threshold = base_threshold * multiplier
        return clamp(threshold, min_threshold, max_threshold)
