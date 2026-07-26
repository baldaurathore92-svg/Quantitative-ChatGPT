"""
Microprice feature calculation.

Microprice is the imbalance-weighted mid price, indicating where the "true" price
should be based on queue imbalance.

RETAIL API LIMITATION:
This is NOT true fair price. Retail snapshot data lacks:
- Aggressor side
- Trade direction
- Hidden liquidity
- Order IDs for queue position

The microprice here is a best-effort proxy using visible queue imbalance.
"""

import math
from typing import Optional, Tuple
from dataclasses import dataclass

from utils.types import Snapshot, PriceKeyedBook, FeatureResult, RollingStats
from utils.math_utils import safe_divide, ticks_from_mid, clamp
from utils.constants import MICROPRICE_TICK_RANGE


@dataclass
class MicropriceConfig:
    """Configuration for microprice calculation."""
    tick_size: float = 0.05
    max_deviation_ticks: float = MICROPRICE_TICK_RANGE
    use_sqrt_weight: bool = True  # Use sqrt(qty) instead of qty for less aggressive weighting
    normalize_range: float = MICROPRICE_TICK_RANGE  # Ticks to normalize to [-1, 1]


class MicropriceCalculator:
    """
    Calculate microprice from order book snapshot.
    
    Microprice = (bid_price * w_ask + ask_price * w_bid) / (w_bid + w_ask)
    
    Where weights are typically:
    - w_bid = f(ask_qty) - weight from opposite queue
    - w_ask = f(bid_qty) - weight from opposite queue
    
    This formulation makes microprice move toward the side with less queue,
    indicating where price pressure exists.
    
    IMPORTANT:
    This is ONLY for scoring, not execution.
    Real execution uses best ask + slippage for longs, best bid - slippage for shorts.
    """
    
    def __init__(self, config: MicropriceConfig):
        self._config = config
    
    def calculate(
        self,
        snapshot: Snapshot,
        prev_book: Optional[PriceKeyedBook] = None,
        stats: Optional[RollingStats] = None
    ) -> FeatureResult:
        """
        Calculate microprice feature.
        
        Args:
            snapshot: Validated market snapshot
            prev_book: Previous price-keyed book (not used for microprice)
            stats: Rolling statistics (not used for microprice)
        
        Returns:
            FeatureResult with normalized value in [-1, +1]
        """
        if not snapshot.is_valid():
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="microprice"
            )
        
        best_bid = snapshot.best_bid
        best_ask = snapshot.best_ask
        
        if best_bid is None or best_ask is None:
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="microprice"
            )
        
        # Calculate microprice
        microprice_val = self._calculate_microprice(
            best_bid.price, best_bid.quantity,
            best_ask.price, best_ask.quantity
        )
        
        # Calculate deviation from mid in ticks
        mid = snapshot.mid_price
        if mid is None:
            return FeatureResult(
                value=0.0,
                confidence=0.0,
                valid=False,
                raw_value=0.0,
                name="microprice"
            )
        
        deviation_ticks = (microprice_val - mid) / self._config.tick_size
        
        # Normalize to [-1, +1]
        # Positive = bullish (microprice above mid, bid pressure)
        # Negative = bearish (microprice below mid, ask pressure)
        normalized = deviation_ticks / self._config.normalize_range
        normalized = clamp(normalized, -1.0, 1.0)
        
        # Calculate confidence based on spread and queue sizes
        confidence = self._calculate_confidence(snapshot, best_bid, best_ask)
        
        return FeatureResult(
            value=normalized,
            confidence=confidence,
            valid=True,
            raw_value=microprice_val,
            name="microprice",
            ticks=deviation_ticks
        )
    
    def _calculate_microprice(
        self,
        bid_price: float,
        bid_qty: int,
        ask_price: float,
        ask_qty: int
    ) -> float:
        """
        Calculate microprice using weighted mid.
        
        Uses inverse of opposite queue as weight.
        Microprice = (bid * w_ask + ask * w_bid) / (w_bid + w_ask)
        
        With sqrt weighting:
        w_bid = sqrt(ask_qty), w_ask = sqrt(bid_qty)
        
        This makes microprice move toward the side with LESS queue.
        """
        if self._config.use_sqrt_weight:
            # Use sqrt for less aggressive weighting
            w_bid = math.sqrt(max(bid_qty, 1))
            w_ask = math.sqrt(max(ask_qty, 1))
        else:
            # Use linear weighting
            w_bid = float(max(ask_qty, 1))  # Weight from opposite queue
            w_ask = float(max(bid_qty, 1))
        
        denominator = w_bid + w_ask
        
        if denominator < 1e-9:
            return (bid_price + ask_price) / 2.0
        
        microprice = (bid_price * w_ask + ask_price * w_bid) / denominator
        
        return microprice
    
    def _calculate_confidence(
        self,
        snapshot: Snapshot,
        best_bid,
        best_ask
    ) -> float:
        """
        Calculate confidence based on book quality.
        
        Higher confidence when:
        - Tight spread
        - Large queue sizes
        - Good depth
        """
        spread = snapshot.spread
        if spread is None:
            return 0.0
        
        spread_ticks = spread / self._config.tick_size
        
        # Spread factor: tighter spread = higher confidence
        # Max confidence at 1 tick spread, decreasing after
        if spread_ticks <= 1:
            spread_factor = 1.0
        elif spread_ticks <= 5:
            spread_factor = 1.0 - (spread_ticks - 1) / 10
        else:
            spread_factor = 0.5
        
        # Queue size factor
        total_qty = best_bid.quantity + best_ask.quantity
        if total_qty >= 10000:
            queue_factor = 1.0
        elif total_qty >= 1000:
            queue_factor = 0.8
        elif total_qty >= 100:
            queue_factor = 0.6
        else:
            queue_factor = 0.4
        
        # Depth factor
        depth = snapshot.depth
        if depth >= 5:
            depth_factor = 1.0
        elif depth >= 3:
            depth_factor = 0.8
        else:
            depth_factor = 0.5
        
        confidence = spread_factor * 0.4 + queue_factor * 0.4 + depth_factor * 0.2
        
        return clamp(confidence, 0.0, 1.0)
    
    def calculate_multilevel(
        self,
        snapshot: Snapshot,
        levels: int = 5
    ) -> Tuple[float, float]:
        """
        Calculate microprice using multiple depth levels.
        
        More stable than single-level microprice.
        
        Args:
            snapshot: Validated snapshot
            levels: Number of depth levels to use
        
        Returns:
            Tuple of (microprice, confidence)
        """
        if not snapshot.is_valid():
            return snapshot.mid_price or 0.0, 0.0
        
        levels = min(levels, snapshot.depth)
        if levels == 0:
            return snapshot.mid_price or 0.0, 0.0
        
        total_bid_qty = sum(level.quantity for level in snapshot.bids[:levels])
        total_ask_qty = sum(level.quantity for level in snapshot.asks[:levels])
        
        # Weight each level by distance and quantity
        weighted_bid_sum = 0.0
        weighted_ask_sum = 0.0
        total_weight = 0.0
        
        mid = snapshot.mid_price
        if mid is None:
            return snapshot.mid_price or 0.0, 0.0
        
        for i in range(levels):
            bid = snapshot.bids[i]
            ask = snapshot.asks[i]
            
            # Weight by inverse of distance (closer = more weight)
            # and by quantity
            bid_dist = abs(bid.price - mid) / self._config.tick_size
            ask_dist = abs(ask.price - mid) / self._config.tick_size
            
            # Exponential decay by distance
            bid_weight = math.exp(-bid_dist / 2) * math.sqrt(bid.quantity)
            ask_weight = math.exp(-ask_dist / 2) * math.sqrt(ask.quantity)
            
            weighted_bid_sum += bid.price * ask_weight  # Weight from opposite
            weighted_ask_sum += ask.price * bid_weight
            total_weight += ask_weight + bid_weight
        
        if total_weight < 1e-9:
            return mid, 0.0
        
        microprice = (weighted_bid_sum + weighted_ask_sum) / total_weight
        
        # Higher confidence for multilevel
        confidence = min(1.0, 0.8 + levels * 0.04)
        
        return microprice, confidence


def calculate_microprice_deviation(
    snapshot: Snapshot,
    tick_size: float = 0.05
) -> Tuple[float, float]:
    """
    Calculate microprice deviation from mid.
    
    Convenience function for quick calculation.
    
    Args:
        snapshot: Validated snapshot
        tick_size: Tick size for normalization
    
    Returns:
        Tuple of (deviation_ticks, confidence)
    """
    config = MicropriceConfig(tick_size=tick_size)
    calc = MicropriceCalculator(config)
    result = calc.calculate(snapshot)
    return result.ticks, result.confidence
