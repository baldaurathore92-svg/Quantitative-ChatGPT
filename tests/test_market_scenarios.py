"""End-to-end verification for deterministic tick-by-tick market scenarios."""

from __future__ import annotations

import math
import os
import tempfile
import unittest
from dataclasses import replace

from adapter.market_scenarios import (
    MarketPattern,
    MarketScenarioConfig,
    TickMarketGenerator,
    continue_market_scenario,
    generate_market_scenario,
)
from adapter.replay import ReplayAdapter, ReplayConfig, SnapshotRecorder
from config import EngineConfig
from engine.quant_engine import QuantEngine
from simulate_trends import create_signal_verification_config, run_scenario
from utils.constants import MOMENTUM_RANGE
from utils.types import SignalType, SnapshotDeliveryMode, State


class MarketScenarioGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.base = MarketScenarioConfig(
            pattern=MarketPattern.UPWARD,
            tick_count=32,
        )

    def test_all_patterns_produce_valid_ordered_ticks(self) -> None:
        for pattern in MarketPattern:
            with self.subTest(pattern=pattern.value):
                config = replace(self.base, pattern=pattern)
                snapshots = generate_market_scenario(config)

                self.assertEqual(len(snapshots), config.tick_count)
                self.assertTrue(all(snapshot.is_valid() for snapshot in snapshots))
                self.assertEqual(
                    [snapshot.sequence for snapshot in snapshots],
                    list(
                        range(
                            config.start_sequence,
                            config.start_sequence + config.tick_count,
                        )
                    ),
                )
                self.assertTrue(
                    all(
                        math.isclose(
                            snapshots[index].timestamp
                            - snapshots[index - 1].timestamp,
                            config.interval_seconds,
                        )
                        for index in range(1, len(snapshots))
                    )
                )
                self.assertTrue(
                    all(
                        math.isclose(
                            snapshot.ltp / config.tick_size,
                            round(snapshot.ltp / config.tick_size),
                            abs_tol=1e-9,
                        )
                        and all(
                            math.isclose(
                                level.price / config.tick_size,
                                round(level.price / config.tick_size),
                                abs_tol=1e-9,
                            )
                            for level in snapshot.bids + snapshot.asks
                        )
                        for snapshot in snapshots
                    )
                )
                self.assertEqual(
                    [snapshot.volume_traded for snapshot in snapshots],
                    [
                        config.volume_per_tick * index
                        for index in range(1, config.tick_count + 1)
                    ],
                )
                self.assertTrue(
                    all(snapshot.depth == config.depth_levels for snapshot in snapshots)
                )

    def test_directional_paths_and_depth_pressure_are_aligned(self) -> None:
        upward = generate_market_scenario(
            replace(self.base, pattern=MarketPattern.UPWARD)
        )
        downward = generate_market_scenario(
            replace(self.base, pattern=MarketPattern.DOWNWARD)
        )

        self.assertTrue(
            all(left.ltp < right.ltp for left, right in zip(upward, upward[1:]))
        )
        self.assertTrue(
            all(left.ltp > right.ltp for left, right in zip(downward, downward[1:]))
        )
        self.assertTrue(
            all(tick.total_buy_qty > tick.total_sell_qty for tick in upward)
        )
        self.assertTrue(
            all(tick.total_buy_qty < tick.total_sell_qty for tick in downward)
        )
        self.assertAlmostEqual(
            upward[-1].ltp - upward[0].ltp,
            downward[0].ltp - downward[-1].ltp,
            places=9,
        )

    def test_noise_is_bounded_and_random_walk_is_seeded(self) -> None:
        noise_config = replace(self.base, pattern=MarketPattern.NOISE, tick_count=128)
        noise = generate_market_scenario(noise_config)
        noise_prices = [snapshot.ltp for snapshot in noise]
        maximum_range = (
            noise_config.noise_amplitude_ticks * 2.0 * noise_config.tick_size
        )
        self.assertLessEqual(
            max(noise_prices) - min(noise_prices),
            maximum_range + 1e-12,
        )

        random_config = replace(self.base, pattern=MarketPattern.RANDOM, tick_count=128)
        first = generate_market_scenario(random_config)
        second = generate_market_scenario(random_config)
        different = generate_market_scenario(replace(random_config, random_seed=99))
        self.assertEqual(first, second)
        self.assertNotEqual(
            [snapshot.ltp for snapshot in first],
            [snapshot.ltp for snapshot in different],
        )
        for seed in range(20):
            seeded_config = replace(random_config, random_seed=seed)
            seeded = generate_market_scenario(seeded_config)
            self.assertEqual(seeded, generate_market_scenario(seeded_config))
            self.assertTrue(all(snapshot.is_valid() for snapshot in seeded))

        maximum_step = random_config.noise_amplitude_ticks * random_config.tick_size
        self.assertTrue(
            all(
                abs(right.ltp - left.ltp) <= maximum_step + 1e-12
                for left, right in zip(first, first[1:])
            )
        )

    def test_continuation_and_exchange_identity(self) -> None:
        config = replace(
            self.base,
            symbol="NIFTY",
            token=" 26000 ",
            exchange_type=1,
            start_sequence=501,
            start_volume_traded=10_000,
            start_timestamp=1_800_000_000.0,
        )
        snapshots = tuple(TickMarketGenerator(config))
        continuation = continue_market_scenario(
            snapshots[-1],
            replace(config, pattern=MarketPattern.NOISE, tick_count=10),
        )
        self.assertEqual(snapshots[0].sequence, 501)
        self.assertEqual(snapshots[0].token, "26000")
        self.assertEqual(snapshots[-1].instrument_key, "1:26000")
        self.assertTrue(all(snapshot.symbol == "NIFTY" for snapshot in snapshots))
        self.assertEqual(continuation[0].sequence, snapshots[-1].sequence + 1)
        self.assertGreater(continuation[0].timestamp, snapshots[-1].timestamp)
        self.assertGreater(
            continuation[0].volume_traded,
            snapshots[-1].volume_traded,
        )

    def test_record_and_replay_round_trip_preserves_generated_ticks(self) -> None:
        snapshots = generate_market_scenario(
            replace(
                self.base,
                pattern=MarketPattern.RANDOM,
                tick_count=12,
                token="26000",
                exchange_type=1,
            )
        )
        handle = tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False)
        path = handle.name
        handle.close()
        try:
            with SnapshotRecorder(path, compress=False) as recorder:
                for snapshot in snapshots:
                    recorder.record(snapshot)

            replay = ReplayAdapter(
                ReplayConfig(path, speed=0.0),
                delivery_mode=SnapshotDeliveryMode.PULL,
            )
            self.assertTrue(replay.load())
            self.assertTrue(replay.start())
            replayed = []
            while replay.running:
                replayed_snapshot = replay.get_snapshot(timeout=0.0)
                if replayed_snapshot is not None:
                    replayed.append(replayed_snapshot)
            self.assertEqual(tuple(replayed), snapshots)
        finally:
            os.unlink(path)

    def test_invalid_configuration_is_rejected(self) -> None:
        invalid_changes = (
            {"tick_count": 0},
            {"start_price": 0.0},
            {"tick_size": float("nan")},
            {"interval_seconds": 0.0},
            {"spread_ticks": -1},
            {"spread_ticks": 1.5},
            {"depth_levels": 0},
            {"trend_step_ticks": 0},
            {"pressure": 1.1},
            {"noise_amplitude_ticks": 0},
            {"random_seed": True},
            {"start_timestamp": float("inf")},
            {"start_sequence": 0},
            {"base_quantity": 0},
            {"volume_per_tick": 0},
            {"start_price": 0.25},
            {"start_price": 100.01},
            {"pattern": MarketPattern.DOWNWARD, "tick_count": 6000},
            {"start_volume_traded": -1},
            {"token": 0},
            {"token": None},
            {"exchange_type": False},
            {"exchange_type": 1},
            {"start_price": 1e16, "tick_size": 1.0},
            {
                "start_price": 9_007_199_254_740_988.0,
                "tick_size": 1.0,
                "tick_count": 1,
            },
            {"interval_seconds": 1e-9, "tick_count": 3},
        )
        for changes in invalid_changes:
            with self.subTest(changes=changes):
                with self.assertRaises((TypeError, ValueError)):
                    replace(self.base, **changes)


class MarketScenarioEngineTests(unittest.TestCase):
    def test_production_calibration_is_unchanged_by_simulation_policy(self) -> None:
        production = EngineConfig()
        verification = create_signal_verification_config()

        self.assertEqual(MOMENTUM_RANGE, 2.0)
        self.assertEqual(production.regime_detection.momentum_threshold, 0.5)
        self.assertEqual(production.regime_detection.obi_threshold, 0.4)
        self.assertEqual(verification.regime_detection.momentum_threshold, 0.3)
        self.assertEqual(verification.regime_detection.obi_threshold, 0.3)

    def test_default_engine_is_conservative_for_generated_paths(self) -> None:
        results = {pattern: run_scenario(pattern) for pattern in MarketPattern}

        self.assertGreater(results[MarketPattern.UPWARD].average_composite, 0.15)
        self.assertLess(results[MarketPattern.DOWNWARD].average_composite, -0.15)
        self.assertLess(abs(results[MarketPattern.NOISE].average_composite), 0.05)
        self.assertLess(abs(results[MarketPattern.RANDOM].average_composite), 0.05)
        self.assertTrue(all(result.rejected_snapshots == 0 for result in results.values()))
        self.assertTrue(all(result.final_state == State.NEUTRAL.name for result in results.values()))
        self.assertTrue(all(not result.signals for result in results.values()))

    def test_signal_policy_handles_seed_7_random_fixture(self) -> None:
        config = create_signal_verification_config()
        results = {
            pattern: run_scenario(pattern, engine_config=config)
            for pattern in MarketPattern
        }

        self.assertEqual(results[MarketPattern.UPWARD].dominant_regime, "TREND")
        self.assertEqual(results[MarketPattern.UPWARD].final_state, State.LONG.name)
        self.assertEqual(
            results[MarketPattern.UPWARD].signals,
            (SignalType.BULLISH.name,),
        )
        self.assertEqual(results[MarketPattern.DOWNWARD].dominant_regime, "TREND")
        self.assertEqual(results[MarketPattern.DOWNWARD].final_state, State.SHORT.name)
        self.assertEqual(
            results[MarketPattern.DOWNWARD].signals,
            (SignalType.BEARISH.name,),
        )
        for pattern in (MarketPattern.NOISE, MarketPattern.RANDOM):
            self.assertEqual(results[pattern].dominant_regime, "NOISE")
            self.assertEqual(results[pattern].final_state, State.NEUTRAL.name)
            self.assertEqual(results[pattern].signals, ())

    def test_directional_positions_exit_once_when_pressure_fades(self) -> None:
        for direction, expected in (
            (
                MarketPattern.UPWARD,
                (SignalType.BULLISH.name, SignalType.EXIT_LONG.name),
            ),
            (
                MarketPattern.DOWNWARD,
                (SignalType.BEARISH.name, SignalType.EXIT_SHORT.name),
            ),
        ):
            with self.subTest(direction=direction.value):
                engine = QuantEngine(create_signal_verification_config())
                directional = generate_market_scenario(
                    MarketScenarioConfig(
                        pattern=direction,
                        symbol="LIFECYCLE",
                        tick_count=60,
                    )
                )
                neutral = continue_market_scenario(
                    directional[-1],
                    MarketScenarioConfig(
                        pattern=MarketPattern.NOISE,
                        symbol="LIFECYCLE",
                        tick_count=80,
                    ),
                )
                signals: list[str] = []
                for snapshot in directional + neutral:
                    composite = engine.process(snapshot)
                    self.assertIsNotNone(composite)
                    assert composite is not None
                    signal = engine.get_execution_signal(snapshot, composite)
                    if signal is not None:
                        signals.append(signal.signal_type.name)
                    self.assertIsNone(
                        engine.get_execution_signal(snapshot, composite),
                        "transition commands must be consumable exactly once",
                    )

                cooldown_release = continue_market_scenario(
                    neutral[-1],
                    MarketScenarioConfig(
                        pattern=MarketPattern.NOISE,
                        symbol="LIFECYCLE",
                        tick_count=1,
                        interval_seconds=1001.0,
                    ),
                )[0]
                release_composite = engine.process(cooldown_release)
                self.assertIsNotNone(release_composite)
                assert release_composite is not None
                self.assertIsNone(
                    engine.get_execution_signal(cooldown_release, release_composite)
                )

                self.assertEqual(tuple(signals), expected)
                self.assertEqual(engine.get_state("LIFECYCLE"), State.NEUTRAL)

    def test_interleaved_upward_and_downward_streams_remain_isolated(self) -> None:
        engine = QuantEngine(create_signal_verification_config())
        upward = generate_market_scenario(
            MarketScenarioConfig(
                pattern=MarketPattern.UPWARD,
                symbol="UP",
                tick_count=80,
            )
        )
        downward = generate_market_scenario(
            MarketScenarioConfig(
                pattern=MarketPattern.DOWNWARD,
                symbol="DOWN",
                tick_count=80,
            )
        )
        signals: dict[str, list[str]] = {"UP": [], "DOWN": []}

        for up_tick, down_tick in zip(upward, downward):
            for snapshot in (up_tick, down_tick):
                composite = engine.process(snapshot)
                self.assertIsNotNone(composite)
                assert composite is not None
                signal = engine.get_execution_signal(snapshot, composite)
                if signal is not None:
                    signals[snapshot.symbol].append(signal.signal_type.name)
                self.assertIsNone(engine.get_execution_signal(snapshot, composite))

        self.assertEqual(engine.get_state("UP"), State.LONG)
        self.assertEqual(engine.get_state("DOWN"), State.SHORT)
        self.assertEqual(signals["UP"], [SignalType.BULLISH.name])
        self.assertEqual(signals["DOWN"], [SignalType.BEARISH.name])
        stats = engine.get_stats()
        self.assertEqual(stats["snapshot_count"], 160)
        self.assertEqual(stats["accepted_snapshot_count"], 160)
        self.assertEqual(stats["rejected_snapshot_count"], 0)
        self.assertEqual(stats["symbols_tracked"], 2)
        self.assertGreaterEqual(stats["avg_processing_time_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
