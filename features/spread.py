"""
Spread feature calculations.

Various spread-related metrics for market quality assessment.
"""

from typing import Optional, Tuple
from dataclasses import dataclass
import math

from utils.types import Snapshot, PriceKeyedBook, FeatureResult, RollingStats
from utils.math_utils import clamp, safe_divide
from utils.constants import MAX_SPREAD_RATIO


@dataclass
class SpreadConfig:
    """Configuration for spread calculations."""
    tick_size: float = 0.05
    max_spread_ticks: float = 100.0
    normalize_range: float = 20.0  # Normalize spread to [0, 1] range


class SpreadCalculator:
    """
    Calculate spread-related metrics.
    
    Features:
    - Absolute spread (in ticks)
    - Relative spread (as % of mid)
    - Spread quality assessment
    """
    
    def __init__(self, config: SpreadConfig):
        self._config = config
    
    def calculate(
        self,
        snapshot: Snapshot,
        prev_book: Optional[PriceKeyedBook] = None,
        stats: Optional[RollingStats] = None
    ) -> FeatureResult:
        """
        Calculate spread feature.
        
        Returns normalized spread quality (inverse of spread).
        High value = tight spread = good quality.
        """
        if not snapshot.is_valid():
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="spread_quality"
            )
        
        spread = snapshot.spread
        if spread is None or spread <= 0:
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="spread_quality"
            )
        
        spread_ticks = spread / self._config.tick_size
        mid = snapshot.mid_price
        
        # Spread quality: inverse, normalized
        # 1 tick spread = 1.0 quality
        # 5 tick spread = 0.5 quality
        # 10+ tick spread = 0.0 quality
        if spread_ticks <= 1:
            quality = 1.0
        elif spread_ticks <= 5:
            quality = 1.0 - (spread_ticks - 1) / 8
        else:
            quality = max(0.0, 0.5 - (spread_ticks - 5) / 10)
        
        quality = clamp(quality, 0.0, 1.0)
        
        return FeatureResult(
            value=quality,
            confidence=1.0,  # Spread is directly observable
            valid=True,
            raw_value=spread_ticks,
            name="spread_quality"
        )
    
    def get_spread_ticks(self, snapshot: Snapshot) -> Optional[float]:
        """Get spread in ticks."""
        if not snapshot.is_valid():
            return None
        spread = snapshot.spread
        if spread is None:
            return None
        return spread / self._config.tick_size
    
    def get_relative_spread(self, snapshot: Snapshot) -> Optional[float]:
        """Get spread as percentage of mid price."""
        if not snapshot.is_valid():
            return None
        
        spread = snapshot.spread
        mid = snapshot.mid_price
        
        if spread is None or mid is None or mid <= 0:
            return None
        
        return (spread / mid) * 100  # As percentage
    
    def is_valid_spread(self, snapshot: Snapshot) -> bool:
        """Check if spread is within valid range."""
        spread_ticks = self.get_spread_ticks(snapshot)
        if spread_ticks is None:
            return False
        return spread_ticks <= self._config.max_spread_ticks
    
    def get_spread_confidence(self, spread_ticks: float) -> float:
        """
        Get confidence based on spread tightness.
        
        Used by other features for their confidence calculation.
        """
        if spread_ticks <= 1:
            return 1.0
        elif spread_ticks <= 3:
            return 0.9
        elif spread_ticks <= 5:
            return 0.7
        elif spread_ticks <= 10:
            return 0.5
        else:
            return 0.3


def calculate_spread_quality(
    snapshot: Snapshot,
    tick_size: float = 0.05
) -> Tuple[float, float]:
    """
    Calculate spread quality.
    
    Returns:
        Tuple of (quality, spread_ticks)
    """
    config = SpreadConfig(tick_size=tick_size)
    calc = SpreadCalculator(config)
    result = calc.calculate(snapshot)
    return result.value, result.raw_value
