"""Fail-safe verification for deterministic adversarial tick streams."""

from __future__ import annotations

import math
import time
import unittest
from collections import Counter
from dataclasses import replace

from adapter.adversarial_scenarios import (
    AdversarialPattern,
    AdversarialScenarioConfig,
    generate_adversarial_scenario,
)
from adapter.market_scenarios import (
    MarketPattern,
    MarketScenarioConfig,
    generate_market_scenario,
)
from config import ContextConfig, EngineConfig
from engine.quant_engine import QuantEngine
from simulate_adversarial import (
    STRESS_LATENCY_BUDGET_MS,
    run_stress_scenario,
)
from simulate_trends import create_signal_verification_config
from utils.types import SignalType, State


class AdversarialGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = AdversarialScenarioConfig(
            pattern=AdversarialPattern.FLASH_CRASH,
        )

    def test_every_pattern_is_deterministic_grid_aligned_and_recovers(self) -> None:
        for pattern in AdversarialPattern:
            with self.subTest(pattern=pattern.value):
                config = replace(self.base, pattern=pattern)
                first = generate_adversarial_scenario(config)
                second = generate_adversarial_scenario(config)

                self.assertEqual(first, second)
                self.assertEqual(len(first), config.tick_count)
                self.assertTrue(first[-1].should_accept)
                self.assertTrue(
                    all(
                        math.isclose(
                            event.snapshot.ltp / config.tick_size,
                            round(event.snapshot.ltp / config.tick_size),
                            abs_tol=1e-9,
                        )
                        and all(
                            math.isclose(
                                level.price / config.tick_size,
                                round(level.price / config.tick_size),
                                abs_tol=1e-9,
                            )
                            for level in (
                                event.snapshot.bids + event.snapshot.asks
                            )
                        )
                        for event in first
                    )
                )
                self.assertEqual(
                    [event.snapshot.volume_traded for event in first],
                    [
                        config.volume_per_tick * index
                        for index in range(1, config.tick_count + 1)
                    ],
                )

                for event in first:
                    structurally_valid = event.snapshot.is_valid()
                    if event.phase == "crossed_book":
                        self.assertFalse(structurally_valid)
                    else:
                        self.assertTrue(structurally_valid)

    def test_flash_crash_falls_holds_and_fully_recovers(self) -> None:
        events = generate_adversarial_scenario(self.base)
        phases = Counter(event.phase for event in events)
        prices = [event.snapshot.ltp for event in events]

        self.assertEqual(prices[0], prices[-1])
        self.assertLess(min(prices), prices[0] * 0.98)
        self.assertEqual(
            set(phases),
            {"stable", "crash", "bottom", "recovery"},
        )
        crash_prices = [
            event.snapshot.ltp for event in events if event.phase == "crash"
        ]
        recovery_prices = [
            event.snapshot.ltp for event in events if event.phase == "recovery"
        ]
        self.assertTrue(
            all(left > right for left, right in zip(crash_prices, crash_prices[1:]))
        )
        self.assertTrue(
            all(
                left < right
                for left, right in zip(recovery_prices, recovery_prices[1:])
            )
        )

    def test_whipsaw_alternates_price_and_depth_pressure(self) -> None:
        events = generate_adversarial_scenario(
            replace(self.base, pattern=AdversarialPattern.WHIPSAW)
        )
        for index, event in enumerate(events[1:], start=1):
            previous = events[index - 1]
            if index > 1:
                self.assertNotEqual(
                    math.copysign(1, event.snapshot.ltp - self.base.start_price),
                    math.copysign(
                        1,
                        previous.snapshot.ltp - self.base.start_price,
                    ),
                )
            if index % 2:
                self.assertGreater(
                    event.snapshot.total_buy_qty,
                    event.snapshot.total_sell_qty,
                )
            else:
                self.assertLess(
                    event.snapshot.total_buy_qty,
                    event.snapshot.total_sell_qty,
                )

    def test_liquidity_vacuum_reaches_minimum_depth_then_recovers(self) -> None:
        events = generate_adversarial_scenario(
            replace(self.base, pattern=AdversarialPattern.LIQUIDITY_VACUUM)
        )
        vacuum = [event for event in events if event.phase == "vacuum"]
        recovery = [event for event in events if event.phase == "recovery"]

        self.assertTrue(vacuum)
        self.assertTrue(
            all(
                event.snapshot.best_bid is not None
                and event.snapshot.best_bid.quantity == 1
                and event.snapshot.best_ask is not None
                and event.snapshot.best_ask.quantity == 1
                for event in vacuum
            )
        )
        self.assertGreater(
            recovery[-1].snapshot.total_buy_qty,
            vacuum[-1].snapshot.total_buy_qty,
        )
        recovered_spread = recovery[-1].snapshot.spread_ticks(self.base.tick_size)
        self.assertIsNotNone(recovered_spread)
        assert recovered_spread is not None
        self.assertTrue(math.isclose(recovered_spread, 2, abs_tol=1e-9))

    def test_fault_metadata_matches_exact_adversarial_payloads(self) -> None:
        expected = {
            AdversarialPattern.SPREAD_SHOCK: ["Spread too wide"],
            AdversarialPattern.STALE_FEED: ["Stale snapshot (sequence)"],
            AdversarialPattern.OUT_OF_ORDER: [
                "Stale snapshot (sequence)",
                "Timestamp anomaly",
            ],
            AdversarialPattern.CROSSED_BOOK: ["Crossed book"],
        }
        for pattern, reasons in expected.items():
            with self.subTest(pattern=pattern.value):
                events = generate_adversarial_scenario(
                    replace(self.base, pattern=pattern)
                )
                faults = [event for event in events if not event.should_accept]
                self.assertEqual(
                    [event.expected_rejection for event in faults],
                    reasons,
                )

        spread_events = generate_adversarial_scenario(
            replace(self.base, pattern=AdversarialPattern.SPREAD_SHOCK)
        )
        max_valid = next(
            event for event in spread_events if event.phase == "maximum_valid_spread"
        )
        rejected = next(
            event for event in spread_events if event.phase == "rejected_spread"
        )
        max_valid_spread = max_valid.snapshot.spread_ticks(self.base.tick_size)
        rejected_spread = rejected.snapshot.spread_ticks(self.base.tick_size)
        self.assertIsNotNone(max_valid_spread)
        self.assertIsNotNone(rejected_spread)
        assert max_valid_spread is not None
        assert rejected_spread is not None
        self.assertTrue(math.isclose(max_valid_spread, 100, abs_tol=1e-9))
        self.assertTrue(math.isclose(rejected_spread, 101, abs_tol=1e-9))

    def test_spread_metadata_tracks_validator_policy_on_custom_price_grid(self) -> None:
        policies = (
            (0.01, 0.05, 100.0),
            (0.02, 0.01, 10.0),
        )
        for scenario_tick, validation_tick, maximum_spread in policies:
            with self.subTest(
                scenario_tick=scenario_tick,
                validation_tick=validation_tick,
                maximum_spread=maximum_spread,
            ):
                config = replace(
                    self.base,
                    pattern=AdversarialPattern.SPREAD_SHOCK,
                    tick_size=scenario_tick,
                    validation_tick_size=validation_tick,
                    max_spread_ticks=maximum_spread,
                )
                events = generate_adversarial_scenario(config)
                engine_config = EngineConfig()
                engine_config.tick.tick_size = validation_tick
                engine_config.spread.max_spread_ticks = maximum_spread
                engine = QuantEngine(engine_config)

                observed = [
                    engine.process(event.snapshot) is not None
                    for event in events
                ]

                self.assertEqual(
                    observed,
                    [event.should_accept for event in events],
                )
                self.assertEqual(
                    engine.get_stats()["rejection_reasons"],
                    {"Spread too wide": 1},
                )

        incompatible = replace(
            self.base,
            pattern=AdversarialPattern.LIQUIDITY_VACUUM,
            max_spread_ticks=10.0,
        )
        with self.assertRaisesRegex(
            ValueError,
            "validator spread policy rejects an expected-valid event",
        ):
            generate_adversarial_scenario(incompatible)

    def test_packet_burst_models_sequence_loss_and_equal_timestamps(self) -> None:
        events = generate_adversarial_scenario(
            replace(self.base, pattern=AdversarialPattern.PACKET_BURST)
        )
        burst = [event for event in events if event.phase == "packet_burst"]
        burst_timestamps = {event.snapshot.timestamp for event in burst}

        self.assertGreaterEqual(len(burst), 10)
        self.assertEqual(len(burst_timestamps), 1)
        first_burst_index = events.index(burst[0])
        self.assertGreater(
            burst[0].snapshot.sequence,
            events[first_burst_index - 1].snapshot.sequence + 1,
        )
        self.assertTrue(
            all(
                left.snapshot.sequence < right.snapshot.sequence
                for left, right in zip(events, events[1:])
            )
        )

    def test_invalid_configuration_is_rejected(self) -> None:
        invalid = (
            {"pattern": "flash_crash"},
            {"tick_count": 31},
            {"tick_count": True},
            {"start_price": 1.0},
            {"depth_levels": 0},
            {"interval_seconds": 1e-9},
            {"validation_tick_size": 0.0},
            {"max_spread_ticks": float("nan")},
            {
                "pattern": AdversarialPattern.SPREAD_SHOCK,
                "max_spread_ticks": 1.0,
            },
            {
                "pattern": AdversarialPattern.OUT_OF_ORDER,
                "start_timestamp": 1.0,
            },
            {"token": 0},
            {"exchange_type": False},
        )
        for changes in invalid:
            with self.subTest(changes=changes):
                with self.assertRaises((TypeError, ValueError)):
                    replace(self.base, **changes)


class AdversarialEngineTests(unittest.TestCase):
    def test_report_contract_matches_every_expected_rejection(self) -> None:
        summaries = {
            pattern: run_stress_scenario(pattern)
            for pattern in AdversarialPattern
        }
        self.assertTrue(all(summary.contract_ok for summary in summaries.values()))
        self.assertTrue(all(summary.validation_ok for summary in summaries.values()))
        self.assertTrue(all(summary.telemetry_ok for summary in summaries.values()))
        self.assertTrue(all(summary.signal_safe for summary in summaries.values()))
        self.assertTrue(all(summary.terminal_state_ok for summary in summaries.values()))
        self.assertTrue(all(summary.latency_ok for summary in summaries.values()))
        self.assertTrue(all(summary.recovery_ok for summary in summaries.values()))
        baseline = summaries[AdversarialPattern.FLASH_CRASH]
        broken_contracts = (
            replace(baseline, engine_accepted=baseline.engine_accepted - 1),
            replace(baseline, signals=(SignalType.BULLISH.name,)),
            replace(baseline, final_state=State.LONG.name),
            replace(
                baseline,
                max_latency_ms=baseline.latency_budget_ms + 1.0,
            ),
            replace(baseline, recovery_ok=False),
        )
        self.assertTrue(all(not summary.contract_ok for summary in broken_contracts))
        self.assertEqual(summaries[AdversarialPattern.OUT_OF_ORDER].rejected, 2)
        self.assertEqual(summaries[AdversarialPattern.SPREAD_SHOCK].rejected, 1)
        for pattern in (
            AdversarialPattern.FLASH_CRASH,
            AdversarialPattern.WHIPSAW,
            AdversarialPattern.LIQUIDITY_VACUUM,
            AdversarialPattern.PACKET_BURST,
        ):
            self.assertEqual(summaries[pattern].rejected, 0)

    def test_rejected_ticks_do_not_mutate_state_or_features(self) -> None:
        fault_patterns = (
            AdversarialPattern.SPREAD_SHOCK,
            AdversarialPattern.STALE_FEED,
            AdversarialPattern.OUT_OF_ORDER,
            AdversarialPattern.CROSSED_BOOK,
        )
        for pattern in fault_patterns:
            with self.subTest(pattern=pattern.value):
                engine = QuantEngine(EngineConfig())
                control = QuantEngine(EngineConfig())
                events = generate_adversarial_scenario(
                    replace(self.base_config(pattern), symbol=pattern.value)
                )
                faults_seen = 0
                final_composite = None
                for index, event in enumerate(events):
                    symbol = event.snapshot.symbol
                    before_stats = engine.get_stats(symbol)
                    before_features = engine.get_feature_values(symbol)
                    before_state = engine.get_state(symbol)
                    before_regime = engine.get_regime(symbol)
                    aggregate_before = engine.get_stats()
                    composite = engine.process(event.snapshot)
                    final_composite = composite

                    if event.should_accept:
                        control_composite = control.process(event.snapshot)
                        self.assertIsNotNone(composite)
                        self.assertEqual(composite, control_composite)
                        self.assertEqual(
                            engine.get_feature_values(symbol),
                            control.get_feature_values(symbol),
                        )
                        self.assertEqual(engine.get_state(symbol), control.get_state(symbol))
                        self.assertEqual(
                            engine.get_regime(symbol),
                            control.get_regime(symbol),
                        )
                        engine_stats = engine.get_stats(symbol)
                        control_stats = control.get_stats(symbol)
                        for key in ("snapshot_count", "state", "regime", "threshold"):
                            self.assertEqual(engine_stats[key], control_stats[key])
                        continue

                    faults_seen += 1
                    self.assertIsNone(composite)
                    after_stats = engine.get_stats(symbol)
                    self.assertEqual(
                        after_stats["snapshot_count"],
                        before_stats["snapshot_count"],
                    )
                    self.assertEqual(engine.get_feature_values(symbol), before_features)
                    self.assertEqual(engine.get_state(symbol), before_state)
                    self.assertEqual(engine.get_regime(symbol), before_regime)
                    aggregate_after = engine.get_stats()
                    self.assertEqual(
                        aggregate_after["accepted_snapshot_count"],
                        aggregate_before["accepted_snapshot_count"],
                    )
                    self.assertEqual(
                        aggregate_after["rejected_snapshot_count"],
                        aggregate_before["rejected_snapshot_count"] + 1,
                    )
                    self.assertTrue(
                        any(later.should_accept for later in events[index + 1 :])
                    )
                self.assertGreater(faults_seen, 0)
                self.assertTrue(events[-1].should_accept)
                self.assertIsNotNone(final_composite)
                self.assertEqual(
                    engine.get_stats()["accepted_snapshot_count"],
                    control.get_stats()["accepted_snapshot_count"],
                )

    def test_sensitive_policy_emits_no_false_signal_during_adversarial_runs(self) -> None:
        for pattern in AdversarialPattern:
            with self.subTest(pattern=pattern.value):
                engine = QuantEngine(create_signal_verification_config())
                signals = []
                for event in generate_adversarial_scenario(self.base_config(pattern)):
                    composite = engine.process(event.snapshot)
                    if composite is None:
                        continue
                    signal = engine.get_execution_signal(event.snapshot, composite)
                    if signal is not None:
                        signals.append(signal.signal_type.name)
                    self.assertIsNone(
                        engine.get_execution_signal(event.snapshot, composite)
                    )
                self.assertEqual(signals, [])
                self.assertEqual(engine.get_state("STRESS"), State.NEUTRAL)

    def test_signal_is_consumed_once_across_rejection_and_recovery(self) -> None:
        engine = QuantEngine(create_signal_verification_config())
        directional = generate_market_scenario(
            MarketScenarioConfig(
                pattern=MarketPattern.UPWARD,
                symbol="ADVERSARIAL_SIGNAL",
                tick_count=80,
            )
        )
        transition = None
        transition_snapshot = None
        for snapshot in directional:
            composite = engine.process(snapshot)
            self.assertIsNotNone(composite)
            assert composite is not None
            transition = engine.get_execution_signal(snapshot, composite)
            self.assertIsNone(engine.get_execution_signal(snapshot, composite))
            if transition is not None:
                transition_snapshot = snapshot
                break

        self.assertIsNotNone(transition)
        assert transition is not None
        self.assertEqual(transition.signal_type, SignalType.BULLISH)
        self.assertIsNotNone(transition_snapshot)
        assert transition_snapshot is not None

        fault_events = generate_adversarial_scenario(
            AdversarialScenarioConfig(
                pattern=AdversarialPattern.CROSSED_BOOK,
                symbol=transition_snapshot.symbol,
                tick_count=32,
                start_price=transition_snapshot.ltp,
                start_sequence=transition_snapshot.sequence + 1,
                start_timestamp=transition_snapshot.timestamp + 0.25,
            )
        )
        fault = next(event for event in fault_events if not event.should_accept)
        recovery = next(
            event for event in fault_events
            if event.phase == "recovery"
        )

        self.assertIsNone(engine.process(fault.snapshot))
        recovery_composite = engine.process(recovery.snapshot)
        self.assertIsNotNone(recovery_composite)
        assert recovery_composite is not None
        recovery_transition = engine.get_execution_signal(
            recovery.snapshot,
            recovery_composite,
        )
        self.assertIsNotNone(recovery_transition)
        assert recovery_transition is not None
        self.assertEqual(recovery_transition.signal_type, SignalType.EXIT_LONG)
        self.assertIsNone(
            engine.get_execution_signal(recovery.snapshot, recovery_composite)
        )
        self.assertEqual(
            engine.get_state(transition_snapshot.symbol),
            State.EXIT_LONG,
        )

    def test_interleaved_faults_remain_isolated_across_symbols(self) -> None:
        engine = QuantEngine(EngineConfig())
        flash = generate_adversarial_scenario(
            replace(
                self.base_config(AdversarialPattern.FLASH_CRASH),
                symbol="FLASH",
            )
        )
        ordered = generate_adversarial_scenario(
            replace(
                self.base_config(AdversarialPattern.OUT_OF_ORDER),
                symbol="FAULTY",
            )
        )

        for flash_event, ordered_event in zip(flash, ordered):
            flash_composite = engine.process(flash_event.snapshot)
            flash_features = engine.get_feature_values("FLASH")
            flash_state = engine.get_state("FLASH")
            flash_stats = engine.get_stats("FLASH")
            ordered_composite = engine.process(ordered_event.snapshot)
            self.assertIsNotNone(flash_composite)
            self.assertEqual(
                ordered_composite is not None,
                ordered_event.should_accept,
            )
            if not ordered_event.should_accept:
                self.assertEqual(engine.get_feature_values("FLASH"), flash_features)
                self.assertEqual(engine.get_state("FLASH"), flash_state)
                after_flash_stats = engine.get_stats("FLASH")
                for key in ("snapshot_count", "state", "regime", "threshold"):
                    self.assertEqual(after_flash_stats[key], flash_stats[key])

        stats = engine.get_stats()
        self.assertEqual(stats["symbols_tracked"], 2)
        self.assertEqual(stats["symbols"]["FLASH"]["snapshot_count"], 80)
        self.assertEqual(stats["symbols"]["FAULTY"]["snapshot_count"], 78)
        self.assertEqual(stats["rejected_snapshot_count"], 2)
        self.assertEqual(
            stats["rejection_reasons"],
            {"Stale snapshot (sequence)": 1, "Timestamp anomaly": 1},
        )

    def test_context_memory_remains_bounded_under_adversarial_symbols(self) -> None:
        config = EngineConfig(context=ContextConfig(max_active_contexts=3))
        engine = QuantEngine(config)
        for index, pattern in enumerate(AdversarialPattern):
            events = generate_adversarial_scenario(
                replace(
                    self.base_config(pattern),
                    symbol=f"STRESS_{index}",
                )
            )
            for event in events:
                engine.process(event.snapshot)

        stats = engine.get_stats()
        self.assertLessEqual(stats["symbols_tracked"], 3)
        self.assertGreaterEqual(stats["contexts_evicted"], 5)
        self.assertEqual(stats["contexts_created"], 8)

    def test_combined_stress_latency_stays_within_engine_budget(self) -> None:
        engine = QuantEngine(EngineConfig())
        durations: list[float] = []
        processed = 0
        for pattern in AdversarialPattern:
            events = generate_adversarial_scenario(
                replace(self.base_config(pattern), symbol=f"LAT_{pattern.value}")
            )
            for event in events:
                started = time.perf_counter()
                engine.process(event.snapshot)
                durations.append((time.perf_counter() - started) * 1000.0)
                processed += 1

        stats = engine.get_stats()
        ordered_durations = sorted(durations)
        p99_index = math.ceil(len(ordered_durations) * 0.99) - 1
        self.assertEqual(processed, len(AdversarialPattern) * 80)
        self.assertLess(sum(durations) / len(durations), 5.0)
        self.assertLess(ordered_durations[p99_index], 10.0)
        self.assertLessEqual(max(durations), STRESS_LATENCY_BUDGET_MS)
        self.assertLess(stats["lifetime_avg_processing_time_ms"], 5.0)

    @staticmethod
    def base_config(pattern: AdversarialPattern) -> AdversarialScenarioConfig:
        return AdversarialScenarioConfig(pattern=pattern)


if __name__ == "__main__":
    unittest.main()
