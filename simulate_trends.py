"""Run deterministic market scenarios through the complete quant engine."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from adapter.market_scenarios import (
    MarketPattern,
    MarketScenarioConfig,
    generate_market_scenario,
)
from config import EngineConfig
from engine.quant_engine import QuantEngine


@dataclass(frozen=True)
class ScenarioSummary:
    """Compact end-to-end result for one generated market path."""

    pattern: MarketPattern
    tick_count: int
    start_price: float
    end_price: float
    min_price: float
    max_price: float
    average_composite: float
    min_composite: float
    max_composite: float
    final_state: str
    dominant_regime: str
    signals: tuple[str, ...]
    rejected_snapshots: int


def create_signal_verification_config() -> EngineConfig:
    """Return a sensitive policy used only to exercise signal transitions."""
    config = EngineConfig()
    config.threshold.base_threshold = 0.22
    config.threshold.min_threshold = 0.15
    config.threshold.max_threshold = 0.35
    config.threshold.warmup_samples = 3
    config.threshold.min_samples = 5
    config.threshold.volatility_multiplier = 0.0
    config.threshold.spread_multiplier = 0.0
    config.threshold.queue_stability_multiplier = 0.0
    config.state_machine.warmup_samples = 5
    config.state_machine.watch_threshold = 0.15
    config.state_machine.position_threshold = 0.22
    config.state_machine.exit_threshold = 0.10
    config.state_machine.cooldown_seconds = 1000.0
    config.state_machine.signal_persistence = 2
    config.regime_detection.momentum_threshold = 0.30
    config.regime_detection.obi_threshold = 0.30
    config.regime_detection.trend_persistence = 1
    return config


def run_scenario(
    pattern: MarketPattern,
    tick_count: int = 120,
    seed: int = 7,
    engine_config: EngineConfig | None = None,
) -> ScenarioSummary:
    """Generate one scenario and process every tick through the engine."""
    scenario = MarketScenarioConfig(
        pattern=pattern,
        symbol=f"SIM_{pattern.value.upper()}",
        tick_count=tick_count,
        random_seed=seed,
    )
    snapshots = generate_market_scenario(scenario)
    engine = QuantEngine(engine_config or EngineConfig())
    composites: list[float] = []
    regimes: list[str] = []
    signals: list[str] = []

    for snapshot in snapshots:
        composite = engine.process(snapshot)
        if composite is None:
            continue
        composites.append(composite.value)
        regimes.append(composite.regime.name)
        signal = engine.get_execution_signal(snapshot, composite)
        if signal is not None:
            signals.append(signal.signal_type.name)

    if not composites:
        raise RuntimeError(f"scenario {pattern.value} produced no composites")
    symbol_stats = engine.get_stats(scenario.symbol)
    aggregate_stats = engine.get_stats()
    dominant_regime = Counter(regimes).most_common(1)[0][0]
    prices = [snapshot.ltp for snapshot in snapshots]
    return ScenarioSummary(
        pattern=pattern,
        tick_count=len(snapshots),
        start_price=prices[0],
        end_price=prices[-1],
        min_price=min(prices),
        max_price=max(prices),
        average_composite=sum(composites) / len(composites),
        min_composite=min(composites),
        max_composite=max(composites),
        final_state=str(symbol_stats["state"]),
        dominant_regime=dominant_regime,
        signals=tuple(signals),
        rejected_snapshots=int(aggregate_stats["rejected_snapshot_count"]),
    )


def _print_report(
    title: str,
    engine_config: EngineConfig,
    seed: int = 7,
) -> None:
    """Print one comparable report under the supplied engine policy."""
    print(f"\n{title}")
    print(
        "pattern   ticks   start      end      change   avg_comp  "
        "range_comp       regime   state       signals"
    )
    print("-" * 116)
    for pattern in MarketPattern:
        result = run_scenario(
            pattern,
            seed=seed,
            engine_config=engine_config,
        )
        change = result.end_price - result.start_price
        signal_text = ",".join(result.signals) or "-"
        display_pattern = (
            f"random[{seed}]" if pattern == MarketPattern.RANDOM else pattern.value
        )
        print(
            f"{display_pattern:<9} {result.tick_count:>5} "
            f"{result.start_price:>8.3f} {result.end_price:>8.3f} "
            f"{change:>+8.3f} {result.average_composite:>+9.3f} "
            f"[{result.min_composite:>+6.3f},{result.max_composite:>+6.3f}] "
            f"{result.dominant_regime:<8} {result.final_state:<11} "
            f"{signal_text}"
        )


def main() -> None:
    """Print default-policy and signal-lifecycle reports for every pattern."""
    _print_report("DEFAULT ENGINE POLICY", EngineConfig())
    _print_report(
        "SIGNAL VERIFICATION POLICY (simulation only)",
        create_signal_verification_config(),
    )


if __name__ == "__main__":
    main()
