"""Run every adversarial tick stream through the complete quant engine."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass

from adapter.adversarial_scenarios import (
    AdversarialPattern,
    AdversarialScenarioConfig,
    generate_adversarial_scenario,
)
from config import EngineConfig
from engine.quant_engine import QuantEngine


STRESS_LATENCY_BUDGET_MS = 50.0


@dataclass(frozen=True)
class StressSummary:
    """Observed fail-safe behavior for one adversarial stream."""

    pattern: AdversarialPattern
    ticks: int
    accepted: int
    engine_accepted: int
    rejected: int
    expected_rejections: tuple[tuple[str, int], ...]
    actual_rejections: tuple[tuple[str, int], ...]
    final_state: str
    signals: tuple[str, ...]
    recovery_ok: bool
    max_latency_ms: float
    latency_budget_ms: float = STRESS_LATENCY_BUDGET_MS

    @property
    def validation_ok(self) -> bool:
        """Return whether every expected validation outcome was observed."""
        expected_rejected = sum(count for _, count in self.expected_rejections)
        return (
            self.expected_rejections == self.actual_rejections
            and self.rejected == expected_rejected
            and self.accepted == self.ticks - expected_rejected
        )

    @property
    def telemetry_ok(self) -> bool:
        """Return whether local and engine acceptance counts agree."""
        return self.engine_accepted == self.accepted

    @property
    def signal_safe(self) -> bool:
        """Return whether the default policy emitted no trade command."""
        return not self.signals

    @property
    def terminal_state_ok(self) -> bool:
        """Return whether the engine returned to its fail-safe neutral state."""
        return self.final_state == "NEUTRAL"

    @property
    def latency_ok(self) -> bool:
        """Return whether every measured tick stayed inside the stress budget."""
        return self.max_latency_ms <= self.latency_budget_ms

    @property
    def contract_ok(self) -> bool:
        return (
            self.validation_ok
            and self.telemetry_ok
            and self.recovery_ok
            and self.signal_safe
            and self.terminal_state_ok
            and self.latency_ok
        )


def run_stress_scenario(pattern: AdversarialPattern) -> StressSummary:
    """Generate, process, and summarize one adversarial scenario."""
    engine_config = EngineConfig()
    config = AdversarialScenarioConfig(
        pattern=pattern,
        symbol=f"STRESS_{pattern.value.upper()}",
        validation_tick_size=engine_config.tick.tick_size,
        max_spread_ticks=engine_config.spread.max_spread_ticks,
    )
    events = generate_adversarial_scenario(config)
    engine = QuantEngine(engine_config)
    expected = Counter(
        event.expected_rejection
        for event in events
        if event.expected_rejection is not None
    )
    signals: list[str] = []
    accepted = 0
    latencies: list[float] = []
    recovery_ok = True
    awaiting_recovery = False

    for event in events:
        started = time.perf_counter()
        composite = engine.process(event.snapshot)
        latencies.append((time.perf_counter() - started) * 1000.0)
        accepted_now = composite is not None
        if event.expected_rejection is not None:
            awaiting_recovery = True
        elif awaiting_recovery:
            recovery_ok = recovery_ok and accepted_now
            awaiting_recovery = False
        if composite is None:
            continue
        accepted += 1
        signal = engine.get_execution_signal(event.snapshot, composite)
        if signal is not None:
            signals.append(signal.signal_type.name)

    stats = engine.get_stats()
    actual_rejections = {
        str(reason): int(count)
        for reason, count in dict(stats["rejection_reasons"]).items()
    }
    return StressSummary(
        pattern=pattern,
        ticks=len(events),
        accepted=accepted,
        engine_accepted=int(stats["accepted_snapshot_count"]),
        rejected=int(stats["rejected_snapshot_count"]),
        expected_rejections=tuple(sorted(expected.items())),
        actual_rejections=tuple(sorted(actual_rejections.items())),
        final_state=str(stats["state"]),
        signals=tuple(signals),
        recovery_ok=recovery_ok and not awaiting_recovery,
        max_latency_ms=max(latencies, default=0.0),
    )


def main() -> None:
    """Print a fail-safe contract report for every adversarial pattern."""
    print(
        "pattern             ticks accepted rejected state      max_ms "
        "recovery contract rejection_reasons"
    )
    print("-" * 126)
    for pattern in AdversarialPattern:
        result = run_stress_scenario(pattern)
        reasons = ",".join(
            f"{reason}:{count}" for reason, count in result.actual_rejections
        ) or "-"
        print(
            f"{pattern.value:<19} {result.ticks:>5} {result.accepted:>8} "
            f"{result.rejected:>8} {result.final_state:<10} "
            f"{result.max_latency_ms:>7.3f} "
            f"{str(result.recovery_ok):<8} {str(result.contract_ok):<8} "
            f"{reasons}"
        )


if __name__ == "__main__":
    main()
