"""Deterministic tick-by-tick market scenarios for engine verification."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, replace
from enum import Enum
from typing import Iterator

from utils.types import MarketSubscription, PriceLevel, Snapshot


class MarketPattern(str, Enum):
    """Supported synthetic market paths."""

    UPWARD = "upward"
    DOWNWARD = "downward"
    NOISE = "noise"
    RANDOM = "random"


@dataclass(frozen=True)
class MarketScenarioConfig:
    """Parameters for one reproducible, exchange-grid-aligned tick stream."""

    pattern: MarketPattern
    symbol: str = "SIM"
    tick_count: int = 120
    start_price: float = 100.0
    tick_size: float = 0.05
    interval_seconds: float = 0.25
    spread_ticks: int = 2
    depth_levels: int = 5
    trend_step_ticks: int = 1
    pressure: float = 0.90
    noise_amplitude_ticks: int = 2
    random_seed: int = 7
    start_timestamp: float = 1_700_000_000.0
    start_sequence: int = 1
    start_volume_traded: int = 0
    base_quantity: int = 1000
    volume_per_tick: int = 100
    token: str = ""
    exchange_type: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.pattern, MarketPattern):
            raise TypeError("pattern must be a MarketPattern")
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        object.__setattr__(self, "symbol", self.symbol.strip())

        self._positive_int("tick_count", self.tick_count)
        self._positive_finite("start_price", self.start_price)
        self._positive_finite("tick_size", self.tick_size)
        self._positive_finite("interval_seconds", self.interval_seconds)
        self._positive_int("spread_ticks", self.spread_ticks)
        self._positive_int("depth_levels", self.depth_levels)
        self._positive_int("trend_step_ticks", self.trend_step_ticks)
        self._positive_int("noise_amplitude_ticks", self.noise_amplitude_ticks)
        self._positive_int("start_sequence", self.start_sequence)
        self._nonnegative_int("start_volume_traded", self.start_volume_traded)
        self._positive_int("base_quantity", self.base_quantity)
        self._positive_int("volume_per_tick", self.volume_per_tick)

        if (
            isinstance(self.pressure, bool)
            or not isinstance(self.pressure, (int, float))
            or not math.isfinite(self.pressure)
            or not 0.0 <= self.pressure <= 1.0
        ):
            raise ValueError("pressure must be finite and between 0 and 1")
        if isinstance(self.random_seed, bool) or not isinstance(self.random_seed, int):
            raise ValueError("random_seed must be an integer")
        if isinstance(self.exchange_type, bool) or not isinstance(
            self.exchange_type, int
        ):
            raise TypeError("exchange_type must be an integer")
        if not isinstance(self.token, str):
            raise TypeError("token must be a string")
        if not math.isfinite(self.start_timestamp) or self.start_timestamp <= 0:
            raise ValueError("start_timestamp must be finite and positive")
        next_timestamp = self.start_timestamp + self.interval_seconds
        end_timestamp = (
            self.start_timestamp
            + (self.tick_count - 1) * self.interval_seconds
        )
        maximum_timestamp = max(next_timestamp, end_timestamp)
        if not math.isfinite(maximum_timestamp):
            raise ValueError("scenario timestamps must remain finite")
        if (
            next_timestamp <= self.start_timestamp
            or math.ulp(maximum_timestamp) >= self.interval_seconds
        ):
            raise ValueError(
                "interval_seconds is not representable across the timestamp range"
            )

        start_ticks = self._price_ticks(self.start_price, self.tick_size)
        minimum_center = self._minimum_center_ticks(start_ticks)
        maximum_center = self._maximum_center_ticks(start_ticks)
        deepest_bid_offset = (self.spread_ticks + 1) // 2 + self.depth_levels - 1
        if minimum_center <= deepest_bid_offset:
            raise ValueError(
                "scenario can reach a price too low for its spread and depth"
            )
        highest_ask_offset = (
            self.spread_ticks - (self.spread_ticks + 1) // 2
            + self.depth_levels
            - 1
        )
        extreme_ticks = (
            minimum_center - deepest_bid_offset,
            maximum_center + highest_ask_offset,
        )
        for price_ticks in extreme_ticks:
            price = price_ticks * self.tick_size
            adjacent = (price_ticks + 1) * self.tick_size
            if not math.isfinite(price) or not math.isfinite(adjacent):
                raise ValueError("scenario prices must remain finite")
            if price == adjacent or math.ulp(price) >= self.tick_size:
                raise ValueError(
                    "tick_size is not representable across the scenario price range"
                )

        if self.token:
            subscription = MarketSubscription(self.exchange_type, self.token)
            object.__setattr__(self, "token", subscription.token)
            object.__setattr__(self, "exchange_type", subscription.exchange_type)
        elif self.exchange_type != 0:
            raise ValueError("exchange_type requires a numeric token")

    def _minimum_center_ticks(self, start_ticks: int) -> int:
        intervals = self.tick_count - 1
        if self.pattern == MarketPattern.DOWNWARD:
            extras = sum(1 for index in range(1, self.tick_count) if index % 4 == 0)
            return start_ticks - intervals * self.trend_step_ticks - extras
        if self.pattern == MarketPattern.NOISE:
            return start_ticks - self.noise_amplitude_ticks
        if self.pattern == MarketPattern.RANDOM:
            return start_ticks - intervals * self.noise_amplitude_ticks
        return start_ticks

    def _maximum_center_ticks(self, start_ticks: int) -> int:
        intervals = self.tick_count - 1
        if self.pattern == MarketPattern.UPWARD:
            extras = sum(1 for index in range(1, self.tick_count) if index % 4 == 0)
            return start_ticks + intervals * self.trend_step_ticks + extras
        if self.pattern == MarketPattern.NOISE:
            return start_ticks + self.noise_amplitude_ticks
        if self.pattern == MarketPattern.RANDOM:
            return start_ticks + intervals * self.noise_amplitude_ticks
        return start_ticks

    @staticmethod
    def _price_ticks(price: float, tick_size: float) -> int:
        ticks = price / tick_size
        if not math.isfinite(ticks):
            raise ValueError("start_price and tick_size produce a non-finite grid")
        rounded = round(ticks)
        if not math.isclose(ticks, rounded, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("start_price must align with tick_size")
        return int(rounded)

    @staticmethod
    def _positive_int(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    @staticmethod
    def _nonnegative_int(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    @staticmethod
    def _positive_finite(name: str, value: float) -> None:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")


class TickMarketGenerator:
    """Generate deterministic, valid snapshots for a configured market path."""

    def __init__(self, config: MarketScenarioConfig):
        self._config = config

    def __iter__(self) -> Iterator[Snapshot]:
        return self.generate()

    def generate(self) -> Iterator[Snapshot]:
        """Yield one complete deterministic stream."""
        config = self._config
        rng = random.Random(config.random_seed)
        start_ticks = MarketScenarioConfig._price_ticks(
            config.start_price,
            config.tick_size,
        )
        center_ticks = start_ticks

        for index in range(config.tick_count):
            if index > 0:
                center_ticks, step_ticks = self._next_center_ticks(
                    index,
                    start_ticks,
                    center_ticks,
                    rng,
                )
            else:
                step_ticks = 0
            pressure = self._pressure(index, step_ticks, rng)
            timestamp = config.start_timestamp + index * config.interval_seconds
            yield self._snapshot(
                index=index,
                center_ticks=center_ticks,
                pressure=pressure,
                timestamp=timestamp,
            )

    def _next_center_ticks(
        self,
        index: int,
        start_ticks: int,
        center_ticks: int,
        rng: random.Random,
    ) -> tuple[int, int]:
        config = self._config
        if config.pattern == MarketPattern.UPWARD:
            step = config.trend_step_ticks + (1 if index % 4 == 0 else 0)
            return center_ticks + step, step
        if config.pattern == MarketPattern.DOWNWARD:
            step = -(config.trend_step_ticks + (1 if index % 4 == 0 else 0))
            return center_ticks + step, step
        if config.pattern == MarketPattern.NOISE:
            amplitude = config.noise_amplitude_ticks
            offsets = (
                0,
                amplitude,
                -amplitude,
                amplitude // 2,
                -(amplitude // 2),
                amplitude,
                0,
                -amplitude,
            )
            target = start_ticks + offsets[index % len(offsets)]
            return target, target - center_ticks

        step = rng.randint(
            -config.noise_amplitude_ticks,
            config.noise_amplitude_ticks,
        )
        return center_ticks + step, step

    def _pressure(
        self,
        index: int,
        step_ticks: int,
        rng: random.Random,
    ) -> float:
        config = self._config
        if config.pattern == MarketPattern.UPWARD:
            return config.pressure
        if config.pattern == MarketPattern.DOWNWARD:
            return -config.pressure
        if config.pattern == MarketPattern.NOISE:
            return 0.15 if index % 2 == 0 else -0.15

        scale = max(1, config.noise_amplitude_ticks)
        directional = step_ticks / scale * 0.55
        return max(-1.0, min(1.0, directional + rng.uniform(-0.35, 0.35)))

    def _snapshot(
        self,
        index: int,
        center_ticks: int,
        pressure: float,
        timestamp: float,
    ) -> Snapshot:
        config = self._config
        best_bid_ticks = center_ticks - (config.spread_ticks + 1) // 2
        best_ask_ticks = best_bid_ticks + config.spread_ticks
        quantity_skew = int(config.base_quantity * 0.90 * pressure)
        bid_base = max(config.depth_levels + 1, config.base_quantity + quantity_skew)
        ask_base = max(config.depth_levels + 1, config.base_quantity - quantity_skew)

        bids = tuple(
            self._price_level(best_bid_ticks - level, bid_base, level)
            for level in range(config.depth_levels)
        )
        asks = tuple(
            self._price_level(best_ask_ticks + level, ask_base, level)
            for level in range(config.depth_levels)
        )

        return Snapshot(
            symbol=config.symbol,
            token=config.token,
            exchange_type=config.exchange_type,
            timestamp=timestamp,
            ltp=center_ticks * config.tick_size,
            ltp_quantity=config.volume_per_tick,
            volume_traded=(
                config.start_volume_traded
                + (index + 1) * config.volume_per_tick
            ),
            total_buy_qty=sum(level.quantity for level in bids),
            total_sell_qty=sum(level.quantity for level in asks),
            bids=bids,
            asks=asks,
            sequence=config.start_sequence + index,
            exchange_timestamp=timestamp,
        )

    def _price_level(
        self,
        price_ticks: int,
        base_quantity: int,
        level: int,
    ) -> PriceLevel:
        order_count = max(1, 10 - level)
        return PriceLevel(
            price=price_ticks * self._config.tick_size,
            quantity=max(order_count, base_quantity - level * 50),
            order_count=order_count,
        )


def generate_market_scenario(config: MarketScenarioConfig) -> tuple[Snapshot, ...]:
    """Materialize a configured scenario for replay or assertions."""
    return tuple(TickMarketGenerator(config))


def continue_market_scenario(
    previous: Snapshot,
    config: MarketScenarioConfig,
) -> tuple[Snapshot, ...]:
    """Generate a stream that continues ordering, identity, time, and volume."""
    continuation = replace(
        config,
        symbol=previous.symbol,
        token=previous.token,
        exchange_type=previous.exchange_type,
        start_price=previous.ltp,
        start_timestamp=previous.timestamp + config.interval_seconds,
        start_sequence=previous.sequence + 1,
        start_volume_traded=previous.volume_traded,
    )
    return generate_market_scenario(continuation)
