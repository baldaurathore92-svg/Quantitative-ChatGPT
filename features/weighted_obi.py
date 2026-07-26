"""
Weighted Order Book Imbalance (OBI) calculation.

OBI measures the relative pressure between bid and ask sides across depth levels.
Weighted by distance from mid to give more importance to levels closer to spread.

CRITICAL IMPLEMENTATION NOTE:
Distance MUST be measured in TICKS, not percentage.
Using percentage like exp(-distance/(mid*0.005)) is INCORRECT.

Correct formula:
    distance_ticks = |price - mid| / TickSize
    weight = exp(-distance_ticks / lambda)

where lambda is a configurable decay factor in ticks.
"""

import math
from typing import Optional, List, Tuple
from dataclasses import dataclass

from utils.types import Snapshot, PriceKeyedBook, FeatureResult, RollingStats, PriceLevel
from utils.math_utils import clamp, safe_divide, exponential_decay_weight
from utils.constants import OBI_RANGE


@dataclass
class OBIConfig:
    """Configuration for OBI calculation."""
    tick_size: float = 0.05
    lambda_decay: float = 3.0  # Decay factor in ticks
    min_levels: int = 1
    max_levels: int = 5
    use_quantity_weights: bool = True
    normalize_range: float = OBI_RANGE


class WeightedOBICalculator:
    """
    Calculate weighted order book imbalance.
    
    OBI = Σ(qty_at_level * distance_weight) for each side
    
    Distance weight = exp(-distance_ticks / lambda)
    
    The result is normalized to [-1, +1]:
    - Positive = bid pressure (bullish)
    - Negative = ask pressure (bearish)
    
    IMPORTANT FOR RETAIL API:
    This uses ONLY visible depth from snapshot.
    Hidden liquidity and iceberg orders are NOT captured.
    This is a PROXY for order flow pressure, not actual flow.
    """
    
    def __init__(self, config: OBIConfig):
        self._config = config
    
    def calculate(
        self,
        snapshot: Snapshot,
        prev_book: Optional[PriceKeyedBook] = None,
        stats: Optional[RollingStats] = None
    ) -> FeatureResult:
        """
        Calculate weighted OBI.
        
        Args:
            snapshot: Validated market snapshot
            prev_book: Previous book (not used for OBI)
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
                name="weighted_obi"
            )
        
        mid = snapshot.mid_price
        if mid is None:
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="weighted_obi"
            )
        
        # Calculate weighted OBI
        weighted_bid_qty, bid_confidence = self._calculate_weighted_sum(
            snapshot.bids, mid, 'bid'
        )
        weighted_ask_qty, ask_confidence = self._calculate_weighted_sum(
            snapshot.asks, mid, 'ask'
        )
        
        # Calculate imbalance
        total_weighted = weighted_bid_qty + weighted_ask_qty
        
        if total_weighted < 1e-9:
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=True,
                raw_value=0.0,
                name="weighted_obi"
            )
        
        # OBI in [-1, +1]
        obi = (weighted_bid_qty - weighted_ask_qty) / total_weighted
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(
            snapshot, weighted_bid_qty, weighted_ask_qty, bid_confidence, ask_confidence
        )
        
        return FeatureResult(
            value=clamp(obi, -1.0, 1.0),
            confidence=confidence,
            valid=True,
            raw_value=obi,
            name="weighted_obi"
        )
    
    def _calculate_weighted_sum(
        self,
        levels: Tuple[PriceLevel, ...],
        mid: float,
        side: str
    ) -> Tuple[float, float]:
        """
        Calculate weighted sum of quantities across levels.
        
        Args:
            levels: Price levels (bids or asks)
            mid: Mid price
            side: 'bid' or 'ask'
        
        Returns:
            Tuple of (weighted_sum, confidence)
        """
        if not levels:
            return 0.0, 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        levels_used = 0
        
        max_levels = min(len(levels), self._config.max_levels)
        
        for i, level in enumerate(levels[:max_levels]):
            # Calculate distance in TICKS
            distance_ticks = abs(level.price - mid) / self._config.tick_size
            
            # Calculate weight using exponential decay
            # Closer levels get more weight
            distance_weight = exponential_decay_weight(
                distance_ticks, self._config.lambda_decay
            )
            
            # Optionally add quantity weighting
            if self._config.use_quantity_weights:
                # Use sqrt to dampen the effect of very large quantities
                qty_factor = math.sqrt(max(level.quantity, 1))
                weight = distance_weight * qty_factor
            else:
                weight = distance_weight * level.quantity
            
            weighted_sum += weight
            total_weight += distance_weight
            levels_used += 1
        
        # Confidence based on how many levels we used
        confidence = min(1.0, levels_used / max(1, self._config.min_levels))
        
        return weighted_sum, confidence
    
    def _calculate_confidence(
        self,
        snapshot: Snapshot,
        weighted_bid: float,
        weighted_ask: float,
        bid_conf: float,
        ask_conf: float
    ) -> float:
        """
        Calculate overall confidence.
        
        Higher confidence when:
        - Good depth
        - Balanced weights from both sides
        - Tight spread
        """
        # Depth confidence
        depth = snapshot.depth
        depth_conf = min(1.0, depth / self._config.max_levels)
        
        # Balance confidence (both sides contributing)
        total = weighted_bid + weighted_ask
        if total > 0:
            balance = min(weighted_bid, weighted_ask) / total
            balance_conf = 2 * balance  # 0.5 balance = 1.0 confidence
        else:
            balance_conf = 0.0
        
        # Spread confidence
        spread = snapshot.spread_ticks(self._config.tick_size)
        if spread is not None:
            if spread <= 1:
                spread_conf = 1.0
            elif spread <= 3:
                spread_conf = 0.8
            elif spread <= 5:
                spread_conf = 0.6
            else:
                spread_conf = 0.4
        else:
            spread_conf = 0.5
        
        # Combine
        confidence = (
            depth_conf * 0.3 +
            balance_conf * 0.3 +
            spread_conf * 0.2 +
            (bid_conf + ask_conf) / 2 * 0.2
        )
        
        return clamp(confidence, 0.0, 1.0)
    
    def calculate_level_contributions(
        self,
        snapshot: Snapshot
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """
        Calculate contribution of each level to OBI.
        
        Returns:
            Tuple of (bid_contributions, ask_contributions)
            Each is list of (distance_ticks, weight) tuples
        """
        if not snapshot.is_valid():
            return [], []
        
        mid = snapshot.mid_price
        if mid is None:
            return [], []
        
        bid_contribs = []
        for level in snapshot.bids[:self._config.max_levels]:
            distance = abs(level.price - mid) / self._config.tick_size
            weight = exponential_decay_weight(distance, self._config.lambda_decay)
            bid_contribs.append((distance, weight * level.quantity))
        
        ask_contribs = []
        for level in snapshot.asks[:self._config.max_levels]:
            distance = abs(level.price - mid) / self._config.tick_size
            weight = exponential_decay_weight(distance, self._config.lambda_decay)
            ask_contribs.append((distance, weight * level.quantity))
        
        return bid_contribs, ask_contribs


class SimpleOBI:
    """
    Simple OBI using only L1-L2 levels.
    
    Primary signal for the engine.
    """
    
    def __init__(self, tick_size: float = 0.05, lambda_decay: float = 3.0):
        self._tick_size = tick_size
        self._lambda = lambda_decay
    
    def calculate(self, snapshot: Snapshot) -> FeatureResult:
        """
        Calculate simple OBI from L1-L2.
        
        This is the primary OBI signal.
        """
        if not snapshot.is_valid() or snapshot.depth < 1:
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="simple_obi"
            )
        
        mid = snapshot.mid_price
        if mid is None:
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="simple_obi"
            )
        
        # Get L1 and L2
        levels_to_use = min(2, snapshot.depth)
        
        weighted_bid = 0.0
        weighted_ask = 0.0
        
        for i in range(levels_to_use):
            bid = snapshot.bids[i]
            ask = snapshot.asks[i]
            
            # Distance in ticks
            bid_dist = abs(bid.price - mid) / self._tick_size
            ask_dist = abs(ask.price - mid) / self._tick_size
            
            # Weight
            bid_weight = math.exp(-bid_dist / self._lambda)
            ask_weight = math.exp(-ask_dist / self._lambda)
            
            weighted_bid += bid_weight * bid.quantity
            weighted_ask += ask_weight * ask.quantity
        
        total = weighted_bid + weighted_ask
        if total < 1e-9:
            obi = 0.0
        else:
            obi = (weighted_bid - weighted_ask) / total
        
        # Higher confidence for L1-L2 only
        confidence = 0.9 if levels_to_use == 2 else 0.7
        
        return FeatureResult(
            value=clamp(obi, -1.0, 1.0),
            confidence=confidence,
            valid=True,
            raw_value=obi,
            name="simple_obi"
        )


def calculate_obi(
    snapshot: Snapshot,
    tick_size: float = 0.05,
    lambda_decay: float = 3.0
) -> Tuple[float, float]:
    """
    Calculate weighted OBI.
    
    Convenience function.
    
    Returns:
        Tuple of (obi_value, confidence)
    """
    config = OBIConfig(tick_size=tick_size, lambda_decay=lambda_decay)
    calc = WeightedOBICalculator(config)
    result = calc.calculate(snapshot)
    return result.value, result.confidence
