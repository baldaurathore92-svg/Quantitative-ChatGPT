"""
Execution Model for signal execution.

Calculates actual execution prices and handles:
- Depth walking
- Slippage estimation
- Cost accounting

IMPORTANT: Microprice is ONLY for scoring.
Real execution uses best ask + slippage for longs,
best bid - slippage for shorts.
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass

from utils.types import Snapshot, PriceLevel, ExecutionSignal, SignalType, Regime
from utils.math_utils import clamp, safe_divide
from utils.constants import (
    DEFAULT_SLIPPAGE_TICKS,
    MAX_DEPTH_WALK,
    MIN_FILL_RATIO,
    EPSILON
)


@dataclass
class ExecutionConfig:
    """Configuration for execution model."""
    slippage_ticks: float = DEFAULT_SLIPPAGE_TICKS
    max_depth_walk: int = MAX_DEPTH_WALK
    min_fill_ratio: float = MIN_FILL_RATIO
    execution_cost_long_pct: float = 0.01  # % cost
    execution_cost_short_pct: float = 0.01  # % cost
    tick_size: float = 0.05


class ExecutionModel:
    """
    Calculate execution prices for signals.
    
    IMPORTANT PRINCIPLE:
    - Microprice is used for SCORING only
    - Execution uses actual order book prices
    - We must cross the spread to get filled
    
    For LONG positions:
    - Target price = Best Ask + slippage
    - This is the price we pay to buy immediately
    
    For SHORT positions:
    - Target price = Best Bid - slippage
    - This is the price we receive when selling immediately
    
    The execution model is SEPARATE from signal logic.
    """
    
    def __init__(self, config: ExecutionConfig):
        self._config = config
    
    def calculate_long_execution(
        self,
        snapshot: Snapshot,
        quantity: Optional[int] = None
    ) -> Tuple[float, float, int]:
        """
        Calculate execution for long position.
        
        Args:
            snapshot: Current market snapshot
            quantity: Target quantity (optional, for depth walking)
        
        Returns:
            Tuple of (execution_price, avg_price, fillable_qty)
        """
        if not snapshot.is_valid():
            return 0.0, 0.0, 0
        
        if quantity is None:
            # Single-level execution
            best_ask = snapshot.best_ask
            if best_ask is None:
                return 0.0, 0.0, 0
            
            execution_price = best_ask.price + self._config.slippage_ticks * self._config.tick_size
            return execution_price, execution_price, best_ask.quantity
        
        # Depth walking for larger quantity
        return self._walk_book_for_long(snapshot, quantity)
    
    def calculate_short_execution(
        self,
        snapshot: Snapshot,
        quantity: Optional[int] = None
    ) -> Tuple[float, float, int]:
        """
        Calculate execution for short position.
        
        Args:
            snapshot: Current market snapshot
            quantity: Target quantity (optional, for depth walking)
        
        Returns:
            Tuple of (execution_price, avg_price, fillable_qty)
        """
        if not snapshot.is_valid():
            return 0.0, 0.0, 0
        
        if quantity is None:
            # Single-level execution
            best_bid = snapshot.best_bid
            if best_bid is None:
                return 0.0, 0.0, 0
            
            execution_price = best_bid.price - self._config.slippage_ticks * self._config.tick_size
            return execution_price, execution_price, best_bid.quantity
        
        # Depth walking for larger quantity
        return self._walk_book_for_short(snapshot, quantity)
    
    def _walk_book_for_long(
        self,
        snapshot: Snapshot,
        target_qty: int
    ) -> Tuple[float, float, int]:
        """
        Walk the ask book for long execution.
        
        Returns:
            Tuple of (worst_price, avg_price, total_fillable)
        """
        total_qty = 0
        total_value = 0.0
        worst_price = 0.0
        levels_walked = 0
        
        for level in snapshot.asks[:self._config.max_depth_walk]:
            fill_qty = min(level.quantity, target_qty - total_qty)
            
            if fill_qty <= 0:
                break
            
            total_qty += fill_qty
            total_value += fill_qty * level.price
            worst_price = max(worst_price, level.price)
            levels_walked += 1
            
            if total_qty >= target_qty:
                break
        
        if total_qty == 0:
            return 0.0, 0.0, 0
        
        # Add slippage to worst price
        execution_price = worst_price + self._config.slippage_ticks * self._config.tick_size
        
        # Average price paid
        avg_price = total_value / total_qty
        
        return execution_price, avg_price, total_qty
    
    def _walk_book_for_short(
        self,
        snapshot: Snapshot,
        target_qty: int
    ) -> Tuple[float, float, int]:
        """
        Walk the bid book for short execution.
        
        Returns:
            Tuple of (worst_price, avg_price, total_fillable)
        """
        total_qty = 0
        total_value = 0.0
        worst_price = float('inf')
        levels_walked = 0
        
        for level in snapshot.bids[:self._config.max_depth_walk]:
            fill_qty = min(level.quantity, target_qty - total_qty)
            
            if fill_qty <= 0:
                break
            
            total_qty += fill_qty
            total_value += fill_qty * level.price
            worst_price = min(worst_price, level.price)
            levels_walked += 1
            
            if total_qty >= target_qty:
                break
        
        if total_qty == 0:
            return 0.0, 0.0, 0
        
        # Subtract slippage from worst price
        execution_price = worst_price - self._config.slippage_ticks * self._config.tick_size
        
        # Average price received
        avg_price = total_value / total_qty
        
        return execution_price, avg_price, total_qty
    
    def calculate_execution_cost(
        self,
        snapshot: Snapshot,
        signal_type: SignalType
    ) -> float:
        """
        Calculate execution cost for a signal.
        
        Cost includes:
        - Spread cost (crossing the spread)
        - Slippage
        - Configured execution cost
        """
        if not snapshot.is_valid():
            return float('inf')
        
        spread = snapshot.spread
        if spread is None:
            return float('inf')
        
        mid = snapshot.mid_price
        if mid is None or mid <= 0:
            return float('inf')
        
        # Spread cost as % of mid
        spread_cost_pct = (spread / mid) * 100
        
        # Slippage cost
        slippage_cost_pct = (self._config.slippage_ticks * self._config.tick_size / mid) * 100
        
        # Fixed execution cost
        if signal_type == SignalType.BULLISH:
            fixed_cost = self._config.execution_cost_long_pct
        else:
            fixed_cost = self._config.execution_cost_short_pct
        
        total_cost = spread_cost_pct + slippage_cost_pct + fixed_cost
        
        return total_cost
    
    def create_execution_signal(
        self,
        snapshot: Snapshot,
        signal_type: SignalType,
        composite_value: float,
        confidence: float,
        regime: Regime,
        timestamp: float
    ) -> Optional[ExecutionSignal]:
        """
        Create execution signal from composite.
        
        Args:
            snapshot: Current snapshot
            signal_type: BULLISH or BEARISH
            composite_value: Composite score value
            confidence: Signal confidence
            regime: Current market regime
            timestamp: Current timestamp
        
        Returns:
            ExecutionSignal or None if invalid
        """
        if not snapshot.is_valid():
            return None
        
        if signal_type == SignalType.NEUTRAL:
            return None
        
        symbol = snapshot.symbol
        
        if signal_type == SignalType.BULLISH:
            target_price, avg_price, fillable = self.calculate_long_execution(snapshot)
        else:
            target_price, avg_price, fillable = self.calculate_short_execution(snapshot)
        
        if target_price <= 0:
            return None
        
        return ExecutionSignal(
            symbol=symbol,
            signal_type=signal_type,
            target_price=target_price,
            confidence=confidence,
            composite_value=composite_value,
            regime=regime,
            timestamp=timestamp,
            slippage_ticks=self._config.slippage_ticks,
            max_depth_walk=self._config.max_depth_walk
        )
    
    def estimate_fill_ratio(
        self,
        snapshot: Snapshot,
        quantity: int,
        side: str
    ) -> float:
        """
        Estimate fill ratio for given quantity.
        
        Args:
            snapshot: Current snapshot
            quantity: Target quantity
            side: 'bid' or 'ask'
        
        Returns:
            Estimated fill ratio [0, 1]
        """
        if not snapshot.is_valid():
            return 0.0
        
        if side.lower() == 'ask':
            # Buying, walking asks
            levels = snapshot.asks
        else:
            # Selling, walking bids
            levels = snapshot.bids
        
        available_qty = sum(
            level.quantity for level in levels[:self._config.max_depth_walk]
        )
        
        if available_qty <= 0:
            return 0.0
        
        return min(1.0, quantity / available_qty)
    
    def get_execution_info(self, snapshot: Snapshot) -> dict:
        """Get execution information for snapshot."""
        long_exec, long_avg, long_qty = self.calculate_long_execution(snapshot)
        short_exec, short_avg, short_qty = self.calculate_short_execution(snapshot)
        
        return {
            'long_target': long_exec,
            'long_avg': long_avg,
            'long_qty': long_qty,
            'short_target': short_exec,
            'short_avg': short_avg,
            'short_qty': short_qty,
            'slippage_ticks': self._config.slippage_ticks,
            'max_depth_walk': self._config.max_depth_walk
        }
