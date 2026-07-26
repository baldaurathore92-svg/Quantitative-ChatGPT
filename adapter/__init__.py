"""
Adapter package for Snapshot Quant Engine.

Data source adapters for live and replay data.
"""

from .angel_v2 import (
    SmartAPIConfig, SmartAPIParser, AngelOneWebSocket
)
from .replay import (
    ReplayConfig, ReplayAdapter, SnapshotRecorder, create_mock_snapshot
)

__all__ = [
    # Angel One SmartAPI
    'SmartAPIConfig', 'SmartAPIParser', 'AngelOneWebSocket',
    # Replay
    'ReplayConfig', 'ReplayAdapter', 'SnapshotRecorder', 'create_mock_snapshot',
]
