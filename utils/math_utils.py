"""
Mathematical utilities for Snapshot Quant Engine.

All operations optimized for incremental computation.
No O(N) scans inside hot paths.
"""

import math
from typing import Tuple, Optional
from .constants import EPSILON


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range [min_val, max_val]."""
    return max(min_val, min(max_val, value))


def normalize_to_range(
    value: float,
    min_val: float,
    max_val: float,
    target_min: float = -1.0,
    target_max: float = 1.0
) -> float:
    """
    Normalize value from [min_val, max_val] to [target_min, target_max].
    
    Handles edge cases:
    - Division by zero when min_val == max_val
    - Values outside source range (clamped)
    """
    if abs(max_val - min_val) < EPSILON:
        # Source range is zero-width, return midpoint of target
        return (target_min + target_max) / 2.0
    
    # Clamp to source range
    clamped = clamp(value, min_val, max_val)
    
    # Linear interpolation
    ratio = (clamped - min_val) / (max_val - min_val)
    return target_min + ratio * (target_max - target_min)


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value for zero denominator.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value to return if denominator is zero
    
    Returns:
        numerator / denominator, or default if denominator is zero
    """
    if abs(denominator) < EPSILON:
        return default
    return numerator / denominator


def exponential_decay_weight(
    distance_ticks: float,
    lambda_factor: float
) -> float:
    """
    Calculate exponential decay weight based on tick distance.
    
    weight = exp(-distance_ticks / lambda)
    
    Args:
        distance_ticks: Distance in number of ticks (must be non-negative)
        lambda_factor: Decay factor (higher = slower decay)
    
    Returns:
        Weight in (0, 1]
    
    Raises:
        ValueError: If lambda_factor is not positive
    """
    if distance_ticks < 0:
        # Log warning but handle gracefully
        distance_ticks = abs(distance_ticks)
    
    if lambda_factor <= 0:
        raise ValueError(f"Lambda must be positive, got {lambda_factor}")
    
    return math.exp(-distance_ticks / lambda_factor)


def time_aware_ema_alpha(
    dt: float,
    tau: float
) -> float:
    """
    Calculate time-aware EMA alpha.
    
    alpha = 1 - exp(-dt / tau)
    
    This ensures consistent smoothing regardless of sampling rate.
    
    Args:
        dt: Time elapsed since last update (seconds)
        tau: Time constant (seconds)
    
    Returns:
        Alpha in (0, 1)
    """
    if dt < 0:
        raise ValueError(f"dt cannot be negative: {dt}")
    if tau <= 0:
        raise ValueError(f"tau must be positive: {tau}")
    
    return 1.0 - math.exp(-dt / tau)


def update_ema(
    current_ema: float,
    new_value: float,
    alpha: float
) -> float:
    """
    Update EMA with new value.
    
    new_ema = alpha * new_value + (1 - alpha) * current_ema
    
    Args:
        current_ema: Current EMA value
        new_value: New observation
        alpha: Smoothing factor in (0, 1)
    
    Returns:
        Updated EMA
    """
    if alpha < 0 or alpha > 1:
        raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
    
    return alpha * new_value + (1.0 - alpha) * current_ema


def weighted_mid_price(
    bid_price: float,
    bid_qty: int,
    ask_price: float,
    ask_qty: int
) -> float:
    """
    Calculate quantity-weighted mid price.
    
    Args:
        bid_price: Best bid price
        bid_qty: Best bid quantity
        ask_price: Best ask price
        ask_qty: Best ask quantity
    
    Returns:
        Weighted mid price
    """
    total_qty = bid_qty + ask_qty
    if total_qty <= 0:
        return (bid_price + ask_price) / 2.0
    
    return (bid_price * ask_qty + ask_price * bid_qty) / total_qty


def microprice(
    bid_price: float,
    bid_qty: int,
    ask_price: float,
    ask_qty: int
) -> float:
    """
    Calculate microprice (imbalance-weighted mid).
    
    Uses inverse of opposite queue as weight.
    
    Microprice = (bid * sqrt(ask_qty) + ask * sqrt(bid_qty)) / 
                 (sqrt(ask_qty) + sqrt(bid_qty))
    
    This formulation makes microprice move toward the side with less queue,
    indicating where the "true" price should be.
    
    Args:
        bid_price: Best bid price
        bid_qty: Best bid quantity
        ask_price: Best ask price
        ask_qty: Best ask quantity
    
    Returns:
        Microprice
    """
    # Use sqrt for less aggressive weighting
    sqrt_bid_qty = math.sqrt(max(bid_qty, 1))
    sqrt_ask_qty = math.sqrt(max(ask_qty, 1))
    
    denominator = sqrt_bid_qty + sqrt_ask_qty
    if denominator < EPSILON:
        return (bid_price + ask_price) / 2.0
    
    return (bid_price * sqrt_ask_qty + ask_price * sqrt_bid_qty) / denominator


def linear_regression_slope(
    x_values: Tuple[float, ...],
    y_values: Tuple[float, ...]
) -> Tuple[float, float, float]:
    """
    Calculate linear regression slope.
    
    Uses simple linear regression for slope estimation.
    
    Args:
        x_values: X coordinates
        y_values: Y coordinates
    
    Returns:
        Tuple of (slope, intercept, r_squared)
    """
    n = len(x_values)
    if n < 2:
        return 0.0, 0.0, 0.0
    
    if n != len(y_values):
        raise ValueError("x_values and y_values must have same length")
    
    # Calculate means
    sum_x = sum(x_values)
    sum_y = sum(y_values)
    mean_x = sum_x / n
    mean_y = sum_y / n
    
    # Calculate slope components
    sum_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_values, y_values))
    sum_xx = sum((x - mean_x) ** 2 for x in x_values)
    
    if abs(sum_xx) < EPSILON:
        return 0.0, mean_y, 0.0
    
    slope = sum_xy / sum_xx
    intercept = mean_y - slope * mean_x
    
    # Calculate R-squared
    if abs(sum_y) < EPSILON:
        r_squared = 0.0
    else:
        y_mean = mean_y
        ss_tot = sum((y - y_mean) ** 2 for y in y_values)
        ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(x_values, y_values))
        r_squared = 1.0 - safe_divide(ss_res, ss_tot, 0.0)
    
    return slope, intercept, r_squared


def sign(value: float) -> int:
    """Return sign of value as -1, 0, or 1."""
    if value > EPSILON:
        return 1
    elif value < -EPSILON:
        return -1
    return 0


def ticks_from_mid(
    price: float,
    mid: float,
    tick_size: float
) -> float:
    """
    Calculate distance from mid in ticks.
    
    Args:
        price: Target price
        mid: Mid price
        tick_size: Tick size
    
    Returns:
        Distance in ticks (can be negative if price < mid)
    """
    if tick_size <= 0:
        raise ValueError(f"Tick size must be positive: {tick_size}")
    
    return (price - mid) / tick_size


def is_price_better(
    price1: float,
    price2: float,
    side: str
) -> bool:
    """
    Check if price1 is better than price2 for given side.
    
    Args:
        price1: First price
        price2: Second price
        side: 'bid' or 'ask'
    
    Returns:
        True if price1 is better
    """
    if side.lower() == 'bid':
        return price1 > price2
    elif side.lower() == 'ask':
        return price1 < price2
    else:
        raise ValueError(f"Invalid side: {side}. Must be 'bid' or 'ask'")


def spread_to_ticks(
    spread: float,
    tick_size: float
) -> float:
    """Convert spread to number of ticks."""
    if tick_size <= 0:
        raise ValueError(f"Tick size must be positive: {tick_size}")
    return spread / tick_size


def volatility_from_returns(
    returns: Tuple[float, ...],
    annualize: bool = True,
    periods_per_year: int = 252 * 390 * 60  # ~1 second snapshots
) -> float:
    """
    Calculate volatility from return series.
    
    Args:
        returns: Series of returns
        annualize: Whether to annualize
        periods_per_year: Number of periods per year for annualization
    
    Returns:
        Volatility (annualized if requested)
    """
    n = len(returns)
    if n < 2:
        return 0.0
    
    # Calculate mean
    mean_return = sum(returns) / n
    
    # Calculate variance
    variance = sum((r - mean_return) ** 2 for r in returns) / (n - 1)
    
    vol = math.sqrt(variance)
    
    if annualize:
        vol *= math.sqrt(periods_per_year)
    
    return vol


def queue_imbalance_ratio(
    bid_qty: int,
    ask_qty: int
) -> float:
    """
    Calculate queue imbalance ratio in [-1, 1].
    
    Positive = bid pressure
    Negative = ask pressure
    
    Args:
        bid_qty: Total bid quantity
        ask_qty: Total ask quantity
    
    Returns:
        Imbalance ratio in [-1, 1]
    """
    total = bid_qty + ask_qty
    if total <= 0:
        return 0.0
    
    return (bid_qty - ask_qty) / total


def decayed_sum(
    values: Tuple[float, ...],
    timestamps: Tuple[float, ...],
    current_time: float,
    half_life: float
) -> float:
    """
    Calculate time-decayed sum.
    
    Each value is weighted by exp(-dt / tau) where tau = half_life / ln(2).
    
    Args:
        values: Values to sum
        timestamps: Timestamps of values
        current_time: Current timestamp
        half_life: Half-life in seconds
    
    Returns:
        Decayed sum
    """
    if len(values) != len(timestamps):
        raise ValueError("values and timestamps must have same length")
    
    if half_life <= 0:
        return sum(values)
    
    tau = half_life / math.log(2)
    total = 0.0
    
    for value, ts in zip(values, timestamps):
        dt = current_time - ts
        if dt >= 0:
            weight = math.exp(-dt / tau)
            total += value * weight
    
    return total
