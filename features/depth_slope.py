"""
Depth Slope feature calculation.

Measures the decay of quantity across depth levels.
A steep slope suggests thin order book (price likely to move).
A flat slope suggests thick order book (price likely to stay).

RETAIL API LIMITATION:
We only see visible depth. Hidden liquidity is not captured.
This is a PROXY for order book depth quality.
"""

import math
from typing import Optional, Tuple
from dataclasses import dataclass

from utils.types import Snapshot, PriceKeyedBook, FeatureResult, RollingStats, PriceLevel
from utils.math_utils import clamp, linear_regression_slope
from utils.constants import DEPTH_SLOPE_RANGE


@dataclass
class DepthSlopeConfig:
    """Configuration for depth slope calculation."""
    tick_size: float = 0.05
    min_levels: int = 3
    max_levels: int = 5
    normalize_range: float = DEPTH_SLOPE_RANGE
    use_log_qty: bool = True  # Use log(quantity) for better linearity


class DepthSlopeCalculator:
    """
    Calculate depth slope across order book levels.
    
    For each side, fit a line to:
    x = price distance from mid (in ticks)
    y = cumulative quantity (or log quantity)
    
    Slope interpretation:
    - Large negative slope = thin book (price moves easily)
    - Small slope = thick book (price resists moves)
    
    The feature compares bid slope vs ask slope:
    - Bid slope > Ask slope = more bid support (bullish)
    - Bid slope < Ask slope = more ask support (bearish)
    
    IMPORTANT:
    This uses ONLY visible depth from snapshot.
    Real exchange order book may have hidden liquidity.
    """
    
    def __init__(self, config: DepthSlopeConfig):
        self._config = config
    
    def calculate(
        self,
        snapshot: Snapshot,
        prev_book: Optional[PriceKeyedBook] = None,
        stats: Optional[RollingStats] = None
    ) -> FeatureResult:
        """
        Calculate depth slope feature.
        
        Args:
            snapshot: Validated market snapshot
            prev_book: Previous book (not used)
            stats: Rolling statistics (not used)
        
        Returns:
            FeatureResult with normalized value in [-1, +1]
        """
        if not snapshot.is_valid():
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="depth_slope"
            )
        
        depth = snapshot.depth
        if depth < self._config.min_levels:
            return FeatureResult(
                value=0.0,
                confidence=0.3,  # Low confidence but still valid
                valid=True,
                raw_value=0.0,
                name="depth_slope"
            )
        
        mid = snapshot.mid_price
        if mid is None:
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="depth_slope"
            )
        
        # Calculate slopes for each side
        bid_slope, bid_r2 = self._calculate_side_slope(snapshot.bids, mid, 'bid')
        ask_slope, ask_r2 = self._calculate_side_slope(snapshot.asks, mid, 'ask')
        
        # Both slopes are negative (quantity decreases with distance)
        # More negative = thinner book
        
        # Feature: relative difference in slopes
        # If bid_slope is less negative (thicker bid book) -> bullish
        # If ask_slope is less negative (thicker ask book) -> bearish
        
        if abs(bid_slope) < 1e-9 and abs(ask_slope) < 1e-9:
            return FeatureResult(
                value=0.0,
                confidence=0.5,
                valid=True,
                raw_value=0.0,
                name="depth_slope"
            )
        
        # Normalize slope difference
        # bid_slope - ask_slope > 0 means bid book is thicker relative to ask
        slope_diff = bid_slope - ask_slope  # Less negative - more negative
        
        # Slope is typically in range [-10, 0] for each side
        # Diff is typically in [-5, +5]
        normalized = slope_diff / self._config.normalize_range
        normalized = clamp(normalized, -1.0, 1.0)
        
        # Confidence based on R-squared and depth
        r2_avg = (bid_r2 + ask_r2) / 2
        depth_factor = min(1.0, depth / self._config.max_levels)
        confidence = r2_avg * 0.5 + depth_factor * 0.5
        
        return FeatureResult(
            value=normalized,
            confidence=confidence,
            valid=True,
            raw_value=slope_diff,
            name="depth_slope"
        )
    
    def _calculate_side_slope(
        self,
        levels: Tuple[PriceLevel, ...],
        mid: float,
        side: str
    ) -> Tuple[float, float]:
        """
        Calculate slope for one side of the book.
        
        Args:
            levels: Price levels
            mid: Mid price
            side: 'bid' or 'ask'
        
        Returns:
            Tuple of (slope, r_squared)
        """
        n = min(len(levels), self._config.max_levels)
        if n < 2:
            return 0.0, 0.0
        
        x_values = []
        y_values = []
        
        cum_qty = 0.0
        
        for i in range(n):
            level = levels[i]
            
            # X: distance from mid in ticks
            distance = abs(level.price - mid) / self._config.tick_size
            x_values.append(distance)
            
            # Y: cumulative quantity (or log)
            cum_qty += level.quantity
            if self._config.use_log_qty:
                y_values.append(math.log(max(cum_qty, 1)))
            else:
                y_values.append(cum_qty)
        
        # Calculate slope
        slope, intercept, r2 = linear_regression_slope(
            tuple(x_values), tuple(y_values)
        )
        
        return slope, r2
    
    def calculate_separate_slopes(
        self,
        snapshot: Snapshot
    ) -> Tuple[float, float, float, float]:
        """
        Calculate separate bid and ask slopes.
        
        Returns:
            Tuple of (bid_slope, ask_slope, bid_r2, ask_r2)
        """
        if not snapshot.is_valid():
            return 0.0, 0.0, 0.0, 0.0
        
        mid = snapshot.mid_price
        if mid is None:
            return 0.0, 0.0, 0.0, 0.0
        
        bid_slope, bid_r2 = self._calculate_side_slope(snapshot.bids, mid, 'bid')
        ask_slope, ask_r2 = self._calculate_side_slope(snapshot.asks, mid, 'ask')
        
        return bid_slope, ask_slope, bid_r2, ask_r2


class CumulativeDepthRatio:
    """
    Calculate cumulative depth ratio between sides.
    
    Alternative to slope-based approach.
    """
    
    def __init__(self, tick_size: float = 0.05, levels: int = 5):
        self._tick_size = tick_size
        self._levels = levels
    
    def calculate(self, snapshot: Snapshot) -> FeatureResult:
        """
        Calculate cumulative depth ratio.
        
        Returns normalized ratio in [-1, +1].
        """
        if not snapshot.is_valid():
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="cum_depth_ratio"
            )
        
        n = min(snapshot.depth, self._levels)
        if n == 0:
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="cum_depth_ratio"
            )
        
        cum_bid = sum(level.quantity for level in snapshot.bids[:n])
        cum_ask = sum(level.quantity for level in snapshot.asks[:n])
        
        total = cum_bid + cum_ask
        if total < 1:
            return FeatureResult(
                value=0.0,
                confidence=0.5,
                valid=True,
                raw_value=0.0,
                name="cum_depth_ratio"
            )
        
        ratio = (cum_bid - cum_ask) / total
        
        return FeatureResult(
            value=clamp(ratio, -1.0, 1.0),
            confidence=0.8,
            valid=True,
            raw_value=ratio,
            name="cum_depth_ratio"
        )


def calculate_depth_slope(
    snapshot: Snapshot,
    tick_size: float = 0.05,
    levels: int = 5
) -> Tuple[float, float]:
    """
    Calculate depth slope.
    
    Convenience function.
    
    Returns:
        Tuple of (slope_value, confidence)
    """
    config = DepthSlopeConfig(tick_size=tick_size, max_levels=levels)
    calc = DepthSlopeCalculator(config)
    result = calc.calculate(snapshot)
    return result.value, result.confidence
