"""
Feature Confidence calculation.

Each feature outputs:
- value: Normalized value
- confidence: How reliable is the value
- valid: Is the result usable

Confidence depends on:
- Spread tightness
- Liquidity depth
- Queue stability
- Book quality
- Volatility
"""

from typing import Dict, Optional
from dataclasses import dataclass

from utils.types import Snapshot, FeatureResult, RollingStats
from utils.math_utils import clamp, safe_divide
from utils.constants import (
    MIN_CONFIDENCE,
    HIGH_CONFIDENCE,
    MEDIUM_CONFIDENCE,
    LOW_CONFIDENCE,
    MIN_LIQUIDITY_QTY,
    MIN_QUEUE_STABILITY
)


@dataclass
class ConfidenceConfig:
    """Configuration for confidence calculation."""
    min_liquidity_qty: int = MIN_LIQUIDITY_QTY
    max_spread_ticks: float = 20.0
    min_queue_stability: float = MIN_QUEUE_STABILITY
    volatility_window: int = 20
    base_confidence: float = MEDIUM_CONFIDENCE


class FeatureConfidenceCalculator:
    """
    Calculate confidence for feature values.
    
    Confidence determines how much weight to give each feature
    in the composite calculation.
    
    Higher confidence when:
    - Spread is tight
    - Liquidity is high
    - Queue is stable
    - Book is well-formed
    
    Lower confidence when:
    - Spread is wide
    - Liquidity is thin
    - Queue is volatile
    - Book has gaps or anomalies
    """
    
    def __init__(self, config: ConfidenceConfig):
        self._config = config
    
    def calculate_spread_confidence(self, spread_ticks: float) -> float:
        """
        Calculate confidence based on spread.
        
        Tighter spread = higher confidence.
        """
        if spread_ticks <= 1:
            return HIGH_CONFIDENCE
        elif spread_ticks <= 2:
            return 0.9
        elif spread_ticks <= 5:
            return MEDIUM_CONFIDENCE
        elif spread_ticks <= 10:
            return LOW_CONFIDENCE
        elif spread_ticks <= 20:
            return 0.2
        else:
            return MIN_CONFIDENCE
    
    def calculate_liquidity_confidence(
        self,
        bid_qty: int,
        ask_qty: int,
        depth: int
    ) -> float:
        """
        Calculate confidence based on liquidity.
        
        More liquidity = higher confidence.
        """
        total_qty = bid_qty + ask_qty
        min_qty = self._config.min_liquidity_qty
        
        # Quantity factor
        if total_qty >= min_qty * 10:
            qty_factor = 1.0
        elif total_qty >= min_qty * 5:
            qty_factor = 0.9
        elif total_qty >= min_qty:
            qty_factor = 0.7
        else:
            qty_factor = total_qty / min_qty * 0.7
        
        # Balance factor
        if total_qty > 0:
            balance = min(bid_qty, ask_qty) / total_qty
            balance_factor = balance * 2  # 0.5 balance = 1.0
        else:
            balance_factor = 0.5
        
        # Depth factor
        if depth >= 5:
            depth_factor = 1.0
        elif depth >= 3:
            depth_factor = 0.8
        else:
            depth_factor = 0.6
        
        return (qty_factor * 0.5 + balance_factor * 0.3 + depth_factor * 0.2)
    
    def calculate_queue_stability_confidence(
        self,
        bid_persistence: Optional[float],
        ask_persistence: Optional[float]
    ) -> float:
        """
        Calculate confidence based on queue stability.
        
        More stable queues = higher confidence.
        """
        if bid_persistence is None or ask_persistence is None:
            return MEDIUM_CONFIDENCE
        
        avg_persistence = (bid_persistence + ask_persistence) / 2
        
        if avg_persistence >= 0.8:
            return HIGH_CONFIDENCE
        elif avg_persistence >= 0.6:
            return 0.85
        elif avg_persistence >= 0.4:
            return MEDIUM_CONFIDENCE
        elif avg_persistence >= self._config.min_queue_stability:
            return LOW_CONFIDENCE
        else:
            return MIN_CONFIDENCE
    
    def calculate_volatility_confidence(
        self,
        spread_vol: float,
        obi_vol: float,
        microprice_vol: float
    ) -> float:
        """
        Calculate confidence based on volatility.
        
        Lower volatility = higher confidence.
        """
        # Normalize volatilities
        spread_vol_norm = min(1.0, spread_vol / 5.0)
        obi_vol_norm = min(1.0, obi_vol * 2)
        microprice_vol_norm = min(1.0, microprice_vol / 5.0)
        
        # Combined volatility (0 = stable, 1 = very volatile)
        combined_vol = (spread_vol_norm + obi_vol_norm + microprice_vol_norm) / 3
        
        # Invert: low volatility = high confidence
        confidence = 1.0 - combined_vol * 0.5
        
        return clamp(confidence, MIN_CONFIDENCE, HIGH_CONFIDENCE)
    
    def calculate_book_quality_confidence(self, snapshot: Snapshot) -> float:
        """
        Calculate confidence based on overall book quality.
        
        Checks for:
        - Crossed book
        - Gaps in depth
        - Inverted levels
        """
        if not snapshot.is_valid():
            return MIN_CONFIDENCE
        
        confidence = 1.0
        
        # Check for gaps (price differences too large)
        for i in range(min(3, snapshot.depth - 1)):
            # Safety check: ensure we have enough levels
            if i + 1 >= len(snapshot.bids) or i + 1 >= len(snapshot.asks):
                break
            
            bid_price = snapshot.bids[i].price if i < len(snapshot.bids) else 0.0
            bid_next_price = snapshot.bids[i + 1].price if i + 1 < len(snapshot.bids) else 0.0
            ask_price = snapshot.asks[i].price if i < len(snapshot.asks) else 0.0
            ask_next_price = snapshot.asks[i + 1].price if i + 1 < len(snapshot.asks) else 0.0
            
            # Skip if prices are zero or invalid
            if bid_price <= 0 or bid_next_price <= 0 or ask_price <= 0 or ask_next_price <= 0:
                continue
            
            bid_gap = bid_price - bid_next_price
            ask_gap = ask_next_price - ask_price
            
            # Gaps should be positive and reasonable
            if bid_gap <= 0 or ask_gap <= 0:
                confidence *= 0.7
            elif bid_gap > bid_price * 0.01:
                # Gap > 1% of price
                confidence *= 0.85
            elif ask_gap > ask_price * 0.01:
                confidence *= 0.85
        
        # Check for reasonable quantities
        for level in list(snapshot.bids[:3]) + list(snapshot.asks[:3]):
            if level.quantity <= 0:
                confidence *= 0.5
            elif level.quantity > 1e9:  # Suspiciously large
                confidence *= 0.8
        
        return clamp(confidence, MIN_CONFIDENCE, HIGH_CONFIDENCE)
    
    def calculate_combined_confidence(
        self,
        snapshot: Snapshot,
        spread_ticks: float,
        bid_persistence: Optional[float] = None,
        ask_persistence: Optional[float] = None,
        spread_vol: float = 0.0,
        obi_vol: float = 0.0,
        microprice_vol: float = 0.0
    ) -> float:
        """
        Calculate combined confidence from all factors.
        
        Returns weighted average of individual confidences.
        """
        # Individual confidences
        spread_conf = self.calculate_spread_confidence(spread_ticks)
        
        best_bid = snapshot.best_bid
        best_ask = snapshot.best_ask
        
        if best_bid and best_ask:
            liquidity_conf = self.calculate_liquidity_confidence(
                best_bid.quantity, best_ask.quantity, snapshot.depth
            )
        else:
            liquidity_conf = LOW_CONFIDENCE
        
        queue_conf = self.calculate_queue_stability_confidence(
            bid_persistence, ask_persistence
        )
        
        volatility_conf = self.calculate_volatility_confidence(
            spread_vol, obi_vol, microprice_vol
        )
        
        book_conf = self.calculate_book_quality_confidence(snapshot)
        
        # Weighted average
        confidence = (
            spread_conf * 0.25 +
            liquidity_conf * 0.25 +
            queue_conf * 0.20 +
            volatility_conf * 0.15 +
            book_conf * 0.15
        )
        
        return clamp(confidence, MIN_CONFIDENCE, HIGH_CONFIDENCE)
    
    def adjust_feature_confidence(
        self,
        feature_result: FeatureResult,
        base_confidence: float
    ) -> float:
        """
        Adjust feature-specific confidence with base confidence.
        
        Uses geometric mean to combine confidences.
        """
        if not feature_result.valid:
            return MIN_CONFIDENCE
        
        # Geometric mean of feature confidence and base confidence
        combined = (feature_result.confidence * base_confidence) ** 0.5
        
        return clamp(combined, MIN_CONFIDENCE, HIGH_CONFIDENCE)


class RegimeConfidenceModifier:
    """
    Modify confidence based on market regime.
    
    Different regimes warrant different confidence levels:
    - TREND: Full confidence
    - PULLBACK: Slightly reduced
    - RANGE: Moderately reduced
    - NOISE: Significantly reduced
    """
    
    def __init__(
        self,
        trend_mult: float = 1.0,
        pullback_mult: float = 0.85,
        range_mult: float = 0.80,
        noise_mult: float = 0.50
    ):
        self._modifiers = {
            'TREND': trend_mult,
            'PULLBACK': pullback_mult,
            'RANGE': range_mult,
            'NOISE': noise_mult
        }
    
    def modify(self, confidence: float, regime: str) -> float:
        """Apply regime modifier to confidence."""
        modifier = self._modifiers.get(regime.upper(), 0.5)
        return clamp(confidence * modifier, MIN_CONFIDENCE, HIGH_CONFIDENCE)
    
    def get_modifier(self, regime: str) -> float:
        """Get confidence modifier for regime."""
        return self._modifiers.get(regime.upper(), 0.5)
