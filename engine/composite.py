"""
Composite Score calculation.

Combines all features into a single normalized score.

Composite = Σ(feature × weight × confidence) / Σ(weight × confidence)

The composite is ALWAYS normalized to [-1, +1].
"""

from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field
import time

from utils.types import FeatureResult, CompositeScore, Regime
from utils.math_utils import clamp
from utils.constants import EPSILON


@dataclass
class CompositeConfig:
    """Configuration for composite calculation."""
    # Feature weights (relative importance)
    weights: Dict[str, float] = field(default_factory=lambda: {
        'microprice': 1.0,
        'weighted_obi': 1.2,
        'depth_slope': 0.6,
        'spread_compression': 0.4,
        'queue_persistence': 0.5,
        'momentum': 0.8,
        'acceleration': 0.4,
        'refill_proxy': 0.3,
        'ltp_confirmation': 0.2
    })


class CompositeCalculator:
    """
    Calculate composite score from features.
    
    IMPORTANT IMPLEMENTATION NOTES:
    
    1. Composite is ALWAYS normalized to [-1, +1]
    
    2. Uniformly scaling ALL weights has NO effect on normalized composite:
       - If every weight is multiplied by K, both numerator and denominator
         are multiplied by K, so the ratio is unchanged.
    
    3. Therefore, regime-specific adjustments use:
       - RELATIVE weight changes (some up, some down)
       - Confidence multiplier
       - Threshold multiplier
    
    4. The weighted average formula:
       composite = Σ(f × w × c) / Σ(w × c)
       
       where:
       - f = feature value in [-1, +1]
       - w = feature weight (relative importance)
       - c = feature confidence in [0, 1]
    
    Features with low confidence contribute less to the composite.
    """
    
    def __init__(self, config: CompositeConfig):
        self._config = config
        self._sample_count = 0
    
    def calculate(
        self,
        features: Dict[str, FeatureResult],
        regime: Regime,
        regime_weights: Optional[Dict[str, float]] = None,
        regime_confidence_mult: float = 1.0,
        timestamp: Optional[float] = None
    ) -> CompositeScore:
        """
        Calculate composite score.
        
        Args:
            features: Dictionary of feature name -> FeatureResult
            regime: Current market regime
            regime_weights: Optional regime-specific weight adjustments
            regime_confidence_mult: Confidence multiplier for regime
            timestamp: Current timestamp
        
        Returns:
            CompositeScore with normalized value
        """
        if timestamp is None:
            timestamp = time.time()
        
        self._sample_count += 1
        
        # Get base weights
        base_weights = self._config.weights.copy()
        
        # Apply regime-specific weight adjustments (RELATIVE changes)
        if regime_weights:
            for feature, adjustment in regime_weights.items():
                if feature in base_weights:
                    base_weights[feature] *= adjustment
        
        # Calculate weighted sum
        numerator = 0.0
        denominator = 0.0
        valid_features = 0
        
        for feature_name, feature_result in features.items():
            if not feature_result.valid:
                continue
            
            weight = base_weights.get(feature_name, 0.0)
            if weight <= 0:
                continue
            
            # Apply regime confidence multiplier to feature confidence
            adjusted_confidence = feature_result.confidence * regime_confidence_mult
            adjusted_confidence = clamp(adjusted_confidence, 0.0, 1.0)
            
            # Weighted contribution
            contribution = feature_result.value * weight * adjusted_confidence
            weight_contribution = weight * adjusted_confidence
            
            numerator += contribution
            denominator += weight_contribution
            valid_features += 1
        
        # Calculate composite
        if denominator < EPSILON or valid_features == 0:
            composite_value = 0.0
            overall_confidence = 0.0
        else:
            composite_value = numerator / denominator
            composite_value = clamp(composite_value, -1.0, 1.0)
            
            # Overall confidence is weighted average of feature confidences
            # denominator already contains weight * confidence
            # Divide by sum of weights to get average confidence
            total_weight = sum(w for w in base_weights.values() if w > 0)
            if total_weight > 0:
                overall_confidence = denominator / total_weight
            else:
                overall_confidence = 0.0
            
            # Clamp confidence to valid range
            overall_confidence = clamp(overall_confidence, 0.0, 1.0)
        
        return CompositeScore(
            value=composite_value,
            confidence=overall_confidence,
            regime=regime,
            threshold_used=0.0,  # Set by caller
            features=features,
            timestamp=timestamp,
            samples=self._sample_count
        )
    
    def calculate_simple(
        self,
        features: Dict[str, FeatureResult],
        weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate simple composite without regime adjustments.
        
        Returns just the value, no metadata.
        """
        if weights is None:
            weights = self._config.weights
        
        numerator = 0.0
        denominator = 0.0
        
        for feature_name, feature_result in features.items():
            if not feature_result.valid:
                continue
            
            weight = weights.get(feature_name, 0.0)
            if weight <= 0:
                continue
            
            numerator += feature_result.value * weight * feature_result.confidence
            denominator += weight * feature_result.confidence
        
        if denominator < EPSILON:
            return 0.0
        
        return clamp(numerator / denominator, -1.0, 1.0)
    
    def get_feature_contributions(
        self,
        features: Dict[str, FeatureResult],
        weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Get contribution of each feature to composite.
        
        Useful for debugging and understanding signal composition.
        """
        if weights is None:
            weights = self._config.weights
        
        contributions = {}
        
        for feature_name, feature_result in features.items():
            if not feature_result.valid:
                contributions[feature_name] = 0.0
                continue
            
            weight = weights.get(feature_name, 0.0)
            contribution = feature_result.value * weight * feature_result.confidence
            contributions[feature_name] = contribution
        
        return contributions
    
    def get_feature_rankings(
        self,
        features: Dict[str, FeatureResult]
    ) -> list:
        """
        Get features ranked by absolute contribution.
        
        Returns list of (feature_name, abs_contribution, value).
        """
        contributions = self.get_feature_contributions(features)
        
        rankings = [
            (name, abs(contrib), features[name].value if name in features else 0)
            for name, contrib in contributions.items()
        ]
        
        rankings.sort(key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def reset(self) -> None:
        """Reset sample count."""
        self._sample_count = 0


def calculate_composite(
    features: Dict[str, FeatureResult],
    weights: Optional[Dict[str, float]] = None
) -> Tuple[float, float]:
    """
    Calculate composite score.
    
    Convenience function.
    
    Returns:
        Tuple of (composite_value, confidence)
    """
    config = CompositeConfig()
    if weights is not None:
        config.weights = weights
    calc = CompositeCalculator(config)
    score = calc.calculate(features, Regime.NOISE)
    return score.value, score.confidence
