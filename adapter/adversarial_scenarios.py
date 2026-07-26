"""Deterministic adversarial tick streams for engine stress verification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from adapter.market_scenarios import MarketPattern, MarketScenarioConfig
from utils.types import PriceLevel, Snapshot


_TIMESTAMP_REGRESSION_SECONDS = 60.0


class AdversarialPattern(str, Enum):
    """Extreme market and malformed-feed scenarios."""

    FLASH_CRASH = "flash_crash"
    WHIPSAW = "whipsaw"
    LIQUIDITY_VACUUM = "liquidity_vacuum"
    SPREAD_SHOCK = "spread_shock"
    STALE_FEED = "stale_feed"
    OUT_OF_ORDER = "out_of_order"
    CROSSED_BOOK = "crossed_book"
    PACKET_BURST = "packet_burst"


@dataclass(frozen=True)
class AdversarialTick:
    """One generated tick plus its expected validation outcome."""

    snapshot: Snapshot
    phase: str
    expected_rejection: str | None = None

    @property
    def should_accept(self) -> bool:
        return self.expected_rejection is None


@dataclass(frozen=True)
class AdversarialScenarioConfig:
    """Configuration shared by every deterministic adversarial stream."""

    pattern: AdversarialPattern
    symbol: str = "STRESS"
    tick_count: int = 80
    start_price: float = 100.0
    tick_size: float = 0.05
    validation_tick_size: float = 0.05
    max_spread_ticks: float = 100.0
    interval_seconds: float = 0.25
    depth_levels: int = 5
    base_quantity: int = 1000
    volume_per_tick: int = 100
    start_timestamp: float = 1_700_100_000.0
    start_sequence: int = 1
    token: str = ""
    exchange_type: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.pattern, AdversarialPattern):
            raise TypeError("pattern must be an AdversarialPattern")
        if (
            isinstance(self.tick_count, bool)
            or not isinstance(self.tick_count, int)
            or self.tick_count < 32
        ):
            raise ValueError("tick_count must be an integer of at least 32")

        baseline = MarketScenarioConfig(
            pattern=MarketPattern.NOISE,
            symbol=self.symbol,
            tick_count=1,
            start_price=self.start_price,
            tick_size=self.tick_size,
            interval_seconds=self.interval_seconds,
            spread_ticks=2,
            depth_levels=self.depth_levels,
            noise_amplitude_ticks=1,
            start_timestamp=self.start_timestamp,
            start_sequence=self.start_sequence,
            base_quantity=self.base_quantity,
            volume_per_tick=self.volume_per_tick,
            token=self.token,
            exchange_type=self.exchange_type,
        )
        object.__setattr__(self, "symbol", baseline.symbol)
        object.__setattr__(self, "token", baseline.token)
        object.__setattr__(self, "exchange_type", baseline.exchange_type)

        for name in ("validation_tick_size", "max_spread_ticks"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")

        start_ticks = MarketScenarioConfig._price_ticks(
            self.start_price,
            self.tick_size,
        )
        quarter = self.tick_count // 4
        maximum_drop = quarter * 4 if self.pattern is AdversarialPattern.FLASH_CRASH else 0
        widest_bid_offset = self.depth_levels
        if self.pattern is AdversarialPattern.SPREAD_SHOCK:
            _, rejected_spread = self.spread_tick_limits()
            widest_bid_offset = (rejected_spread + 1) // 2 + self.depth_levels - 1
        if start_ticks - maximum_drop <= widest_bid_offset:
            raise ValueError(
                "start_price is too low for the selected adversarial depth"
            )
        end_timestamp = (
            self.start_timestamp
            + (self.tick_count - 1) * self.interval_seconds
        )
        if not math.isfinite(end_timestamp):
            raise ValueError("scenario end timestamp must remain finite")
        if self.pattern is AdversarialPattern.OUT_OF_ORDER:
            midpoint = self.tick_count // 2
            last_accepted_timestamp = (
                self.start_timestamp + (midpoint - 1) * self.interval_seconds
            )
            if last_accepted_timestamp <= _TIMESTAMP_REGRESSION_SECONDS + 1.0:
                raise ValueError(
                    "start_timestamp is too low for a positive timestamp regression"
                )

    def spread_tick_limits(self) -> tuple[int, int]:
        """Return adjacent scenario-grid spreads bracketing the validator limit."""
        limit_price = self.validation_tick_size * self.max_spread_ticks
        maximum_valid = math.floor(limit_price / self.tick_size)

        while maximum_valid > 0 and not self._spread_is_validator_accepted(
            maximum_valid
        ):
            maximum_valid -= 1
        while self._spread_is_validator_accepted(maximum_valid + 1):
            maximum_valid += 1

        if maximum_valid < 2:
            raise ValueError(
                "validator spread policy must accept the normal two-tick spread"
            )
        return maximum_valid, maximum_valid + 1

    def _spread_is_validator_accepted(self, spread_ticks: int) -> bool:
        """Evaluate spread acceptance using the validator's float arithmetic."""
        center_ticks = MarketScenarioConfig._price_ticks(
            self.start_price,
            self.tick_size,
        )
        best_bid_ticks = center_ticks - (spread_ticks + 1) // 2
        best_ask_ticks = best_bid_ticks + spread_ticks
        spread_price = (
            best_ask_ticks * self.tick_size
            - best_bid_ticks * self.tick_size
        )
        return (
            spread_price / self.validation_tick_size
            <= self.max_spread_ticks
        )


class AdversarialScenarioGenerator:
    """Build one complete adversarial stream with explicit fault metadata."""

    def __init__(self, config: AdversarialScenarioConfig):
        self._config = config
        self._start_ticks = MarketScenarioConfig._price_ticks(
            config.start_price,
            config.tick_size,
        )

    def generate(self) -> tuple[AdversarialTick, ...]:
        handlers = {
            AdversarialPattern.FLASH_CRASH: self._flash_crash,
            AdversarialPattern.WHIPSAW: self._whipsaw,
            AdversarialPattern.LIQUIDITY_VACUUM: self._liquidity_vacuum,
            AdversarialPattern.SPREAD_SHOCK: self._spread_shock,
            AdversarialPattern.STALE_FEED: self._stale_feed,
            AdversarialPattern.OUT_OF_ORDER: self._out_of_order,
            AdversarialPattern.CROSSED_BOOK: self._crossed_book,
            AdversarialPattern.PACKET_BURST: self._packet_burst,
        }
        events = tuple(handlers[self._config.pattern]())
        for event in events:
            if not event.should_accept:
                continue
            spread_ticks = event.snapshot.spread_ticks(
                self._config.validation_tick_size
            )
            if (
                spread_ticks is None
                or spread_ticks > self._config.max_spread_ticks
            ):
                raise ValueError(
                    "validator spread policy rejects an expected-valid event"
                )
        return events

    def _flash_crash(self) -> list[AdversarialTick]:
        count = self._config.tick_count
        quarter = count // 4
        drop_ticks = quarter * 4
        bottom = self._start_ticks - drop_ticks
        events: list[AdversarialTick] = []
        for index in range(count):
            if index < quarter:
                center, pressure, phase = self._start_ticks, 0.05, "stable"
            elif index < quarter * 2:
                elapsed = index - quarter + 1
                center, pressure, phase = (
                    self._start_ticks - elapsed * 4,
                    -0.95,
                    "crash",
                )
            elif index < quarter * 3:
                center, pressure, phase = bottom, -0.20, "bottom"
            else:
                recovery_index = index - quarter * 3 + 1
                recovery_count = count - quarter * 3
                recovered = round(drop_ticks * recovery_index / recovery_count)
                center, pressure, phase = (
                    bottom + recovered,
                    0.90,
                    "recovery",
                )
            events.append(self._event(index, center, pressure=pressure, phase=phase))
        return events

    def _whipsaw(self) -> list[AdversarialTick]:
        events: list[AdversarialTick] = []
        for index in range(self._config.tick_count):
            if index == 0:
                center, pressure = self._start_ticks, 0.0
            else:
                direction = 1 if index % 2 else -1
                center = self._start_ticks + direction * 8
                pressure = direction * 0.95
            events.append(self._event(index, center, pressure=pressure, phase="whipsaw"))
        return events

    def _liquidity_vacuum(self) -> list[AdversarialTick]:
        count = self._config.tick_count
        quarter = count // 4
        events: list[AdversarialTick] = []
        for index in range(count):
            if index < quarter:
                quantity, spread, phase = self._config.base_quantity, 2, "liquid"
            elif index < quarter * 2:
                progress = (index - quarter + 1) / quarter
                quantity = max(1, round(self._config.base_quantity * (1 - progress)))
                spread = 2 + round(progress * 18)
                phase = "drain"
            elif index < quarter * 3:
                quantity, spread, phase = 1, 20, "vacuum"
            else:
                progress = (index - quarter * 3 + 1) / (count - quarter * 3)
                quantity = max(1, round(self._config.base_quantity * progress))
                spread = max(2, 20 - round(progress * 18))
                phase = "recovery"
            events.append(
                self._event(
                    index,
                    self._start_ticks,
                    spread_ticks=spread,
                    pressure=0.0,
                    quantity=quantity,
                    phase=phase,
                )
            )
        return events

    def _spread_shock(self) -> list[AdversarialTick]:
        midpoint = self._config.tick_count // 2
        maximum_valid_spread, rejected_spread = self._config.spread_tick_limits()
        events: list[AdversarialTick] = []
        for index in range(self._config.tick_count):
            rejection: str | None = None
            if index == midpoint - 1:
                spread, phase = maximum_valid_spread, "maximum_valid_spread"
            elif index == midpoint:
                spread, phase = rejected_spread, "rejected_spread"
                rejection = "Spread too wide"
            elif index == midpoint + 1:
                spread, phase = 2, "recovery"
            else:
                spread, phase = 2, "normal"
            events.append(
                self._event(
                    index,
                    self._start_ticks,
                    spread_ticks=spread,
                    phase=phase,
                    expected_rejection=rejection,
                )
            )
        return events

    def _stale_feed(self) -> list[AdversarialTick]:
        midpoint = self._config.tick_count // 2
        events: list[AdversarialTick] = []
        for index in range(self._config.tick_count):
            sequence = None
            timestamp = None
            rejection: str | None = None
            phase = "normal"
            if index == midpoint:
                sequence = self._config.start_sequence + index - 1
                timestamp = self._timestamp(index - 1)
                rejection = "Stale snapshot (sequence)"
                phase = "duplicate_stale"
            elif index == midpoint + 1:
                phase = "recovery"
            events.append(
                self._event(
                    index,
                    self._start_ticks,
                    sequence=sequence,
                    timestamp=timestamp,
                    phase=phase,
                    expected_rejection=rejection,
                )
            )
        return events

    def _out_of_order(self) -> list[AdversarialTick]:
        midpoint = self._config.tick_count // 2
        events: list[AdversarialTick] = []
        for index in range(self._config.tick_count):
            sequence = None
            timestamp = None
            rejection: str | None = None
            phase = "normal"
            if index == midpoint:
                sequence = self._config.start_sequence + index - 3
                rejection = "Stale snapshot (sequence)"
                phase = "sequence_regression"
            elif index == midpoint + 1:
                last_accepted_timestamp = self._timestamp(midpoint - 1)
                timestamp = (
                    last_accepted_timestamp
                    - _TIMESTAMP_REGRESSION_SECONDS
                    - 1.0
                )
                rejection = "Timestamp anomaly"
                phase = "timestamp_regression"
            elif index == midpoint + 2:
                phase = "recovery"
            events.append(
                self._event(
                    index,
                    self._start_ticks,
                    sequence=sequence,
                    timestamp=timestamp,
                    phase=phase,
                    expected_rejection=rejection,
                )
            )
        return events

    def _crossed_book(self) -> list[AdversarialTick]:
        midpoint = self._config.tick_count // 2
        return [
            self._event(
                index,
                self._start_ticks,
                crossed=index == midpoint,
                phase=(
                    "crossed_book"
                    if index == midpoint
                    else "recovery"
                    if index == midpoint + 1
                    else "normal"
                ),
                expected_rejection="Crossed book" if index == midpoint else None,
            )
            for index in range(self._config.tick_count)
        ]

    def _packet_burst(self) -> list[AdversarialTick]:
        midpoint = self._config.tick_count // 2
        burst_end = min(self._config.tick_count, midpoint + 12)
        burst_timestamp = self._timestamp(midpoint)
        sequence_gap = 25
        events: list[AdversarialTick] = []
        for index in range(self._config.tick_count):
            timestamp = burst_timestamp if midpoint <= index < burst_end else None
            sequence = (
                self._config.start_sequence + index + sequence_gap
                if index >= midpoint
                else None
            )
            phase = (
                "packet_burst"
                if midpoint <= index < burst_end
                else "post_gap_recovery"
                if index == burst_end
                else "normal"
            )
            events.append(
                self._event(
                    index,
                    self._start_ticks + (index % 3) - 1,
                    sequence=sequence,
                    timestamp=timestamp,
                    phase=phase,
                )
            )
        return events

    def _event(
        self,
        index: int,
        center_ticks: int,
        *,
        spread_ticks: int = 2,
        pressure: float = 0.0,
        quantity: int | None = None,
        sequence: int | None = None,
        timestamp: float | None = None,
        crossed: bool = False,
        phase: str,
        expected_rejection: str | None = None,
    ) -> AdversarialTick:
        config = self._config
        base_quantity = config.base_quantity if quantity is None else quantity
        skew = int(base_quantity * 0.85 * pressure)
        bid_quantity = max(1, base_quantity + skew)
        ask_quantity = max(1, base_quantity - skew)

        if crossed:
            best_bid_ticks = center_ticks
            best_ask_ticks = center_ticks
        else:
            best_bid_ticks = center_ticks - (spread_ticks + 1) // 2
            best_ask_ticks = best_bid_ticks + spread_ticks

        bids = tuple(
            self._level(best_bid_ticks - level, bid_quantity, level)
            for level in range(config.depth_levels)
        )
        asks = tuple(
            self._level(best_ask_ticks + level, ask_quantity, level)
            for level in range(config.depth_levels)
        )
        snapshot_timestamp = self._timestamp(index) if timestamp is None else timestamp
        snapshot = Snapshot(
            symbol=config.symbol,
            token=config.token,
            exchange_type=config.exchange_type,
            timestamp=snapshot_timestamp,
            ltp=center_ticks * config.tick_size,
            ltp_quantity=config.volume_per_tick,
            volume_traded=(index + 1) * config.volume_per_tick,
            total_buy_qty=sum(level.quantity for level in bids),
            total_sell_qty=sum(level.quantity for level in asks),
            bids=bids,
            asks=asks,
            sequence=(
                config.start_sequence + index
                if sequence is None
                else sequence
            ),
            exchange_timestamp=snapshot_timestamp,
        )
        return AdversarialTick(
            snapshot=snapshot,
            phase=phase,
            expected_rejection=expected_rejection,
        )

    def _level(self, price_ticks: int, quantity: int, level: int) -> PriceLevel:
        level_quantity = max(1, quantity // (level + 1))
        order_count = min(max(1, 10 - level), level_quantity)
        return PriceLevel(
            price=price_ticks * self._config.tick_size,
            quantity=level_quantity,
            order_count=order_count,
        )

    def _timestamp(self, index: int) -> float:
        return self._config.start_timestamp + index * self._config.interval_seconds


def generate_adversarial_scenario(
    config: AdversarialScenarioConfig,
) -> tuple[AdversarialTick, ...]:
    """Materialize one deterministic adversarial stream."""
    return AdversarialScenarioGenerator(config).generate()
