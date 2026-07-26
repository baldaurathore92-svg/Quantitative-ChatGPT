"""Adapter package for live, replay, and deterministic simulation data."""

from .angel_v2 import AngelOneWebSocket, SmartAPIConfig, SmartAPIParser
from .market_scenarios import (
    MarketPattern,
    MarketScenarioConfig,
    TickMarketGenerator,
    continue_market_scenario,
    generate_market_scenario,
)
from .replay import ReplayAdapter, ReplayConfig, SnapshotRecorder, create_mock_snapshot

__all__ = [
    "SmartAPIConfig",
    "SmartAPIParser",
    "AngelOneWebSocket",
    "ReplayConfig",
    "ReplayAdapter",
    "SnapshotRecorder",
    "create_mock_snapshot",
    "MarketPattern",
    "MarketScenarioConfig",
    "TickMarketGenerator",
    "generate_market_scenario",
    "continue_market_scenario",
]
