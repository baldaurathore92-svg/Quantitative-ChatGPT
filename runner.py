"""
Runner script for Snapshot Quant Engine.

Provides convenience functions for starting the engine
with different configurations.
"""

import time
from typing import Optional, Callable, Sequence

from config import EngineConfig
from engine.quant_engine import QuantEngine
from adapter.angel_v2 import SmartAPIConfig, AngelOneWebSocket
from adapter.replay import ReplayAdapter, ReplayConfig, create_mock_snapshot
from utils.types import (
    Snapshot, CompositeScore, ExecutionSignal, SnapshotDeliveryMode,
    MarketSubscription
)
from utils.logging_utils import StructuredLogger


class QuantEngineRunner:
    """
    High-level interface for running the Quant Engine.
    
    Simplifies the common use cases:
    - Live trading with Angel One
    - Backtesting with replay data
    - Paper trading with callbacks
    """
    
    def __init__(self, config: Optional[EngineConfig] = None):
        self._config = config or EngineConfig()
        self._engine = QuantEngine(self._config)
        self._logger = StructuredLogger('QuantEngineRunner')
        
        # Callbacks
        self._on_signal: Optional[Callable[[ExecutionSignal], None]] = None
        self._on_composite: Optional[Callable[[CompositeScore], None]] = None
    
    @property
    def engine(self) -> QuantEngine:
        """Get the underlying engine."""
        return self._engine
    
    @property
    def config(self) -> EngineConfig:
        """Get configuration."""
        return self._config
    
    def on_signal(self, callback: Callable[[ExecutionSignal], None]) -> None:
        """Set signal callback."""
        self._on_signal = callback
    
    def on_composite(self, callback: Callable[[CompositeScore], None]) -> None:
        """Set composite callback."""
        self._on_composite = callback
    
    def process_snapshot(self, snapshot: Snapshot) -> Optional[CompositeScore]:
        """
        Process a single snapshot.
        
        This is the main entry point for custom data pipelines.
        
        Args:
            snapshot: Market snapshot
        
        Returns:
            CompositeScore if valid, None otherwise
        """
        composite = self._engine.process(snapshot)
        
        if composite and self._on_composite:
            self._on_composite(composite)
        
        if composite:
            # Always ask the state machine for its one-shot transition command;
            # exits may occur below the outer composite activation threshold.
            signal = self._engine.get_execution_signal(snapshot, composite)
            if signal and self._on_signal:
                self._on_signal(signal)
        
        return composite
    
    def process_mock(
        self,
        symbol: str = 'TEST',
        ltp: float = 100.0
    ) -> Optional[CompositeScore]:
        """
        Process a mock snapshot for testing.
        
        Args:
            symbol: Symbol name
            ltp: Last traded price
        
        Returns:
            CompositeScore
        """
        snapshot = create_mock_snapshot(
            symbol=symbol,
            ltp=ltp,
            spread_ticks=2.0,
            depth_levels=5,
            tick_size=self._config.tick.tick_size
        )
        
        return self.process_snapshot(snapshot)
    
    def run_live(
        self,
        subscriptions: Optional[Sequence[MarketSubscription]] = None,
        on_signal: Optional[Callable[[ExecutionSignal], None]] = None,
        on_disconnect: Optional[Callable[[], None]] = None
    ) -> None:
        """
        Run with live Angel One data.
        
        Args:
            subscriptions: Exchange/token pairs; defaults to configured subscriptions
            on_signal: Signal callback
            on_disconnect: Disconnect callback
        """
        if on_signal:
            self._on_signal = on_signal

        selected_subscriptions = (
            tuple(subscriptions)
            if subscriptions is not None
            else tuple(self._config.subscriptions)
        )
        if not selected_subscriptions:
            self._logger.error("At least one live subscription is required")
            return
        
        # Create WebSocket adapter
        api_config = SmartAPIConfig(
            api_key=self._config.api.api_key,
            auth_token=self._config.api.auth_token,
            client_code=self._config.api.client_code,
            feed_token=self._config.api.feed_token,
            heartbeat_interval=self._config.api.heartbeat_interval,
            reconnect_delay=self._config.api.reconnect_delay,
            max_reconnect_attempts=self._config.api.max_reconnect_attempts,
            snapshot_timeout=self._config.api.snapshot_timeout,
            correlation_id=self._config.api.correlation_id,
        )
        
        ws = AngelOneWebSocket(
            api_config,
            delivery_mode=SnapshotDeliveryMode.CALLBACK
        )
        
        def on_snapshot(snapshot: Snapshot):
            self.process_snapshot(snapshot)
        
        ws.on_snapshot = on_snapshot
        ws.on_disconnect = on_disconnect
        
        if ws.connect():
            if not ws.subscribe(selected_subscriptions):
                self._logger.error("Failed to subscribe to configured tokens")
                ws.disconnect()
                return
            
            # Keep running through transient disconnects while the adapter retries.
            try:
                while ws.running:
                    time.sleep(1)
            except KeyboardInterrupt:
                pass
            finally:
                ws.disconnect()
    
    def run_replay(
        self,
        file_path: str,
        speed: float = 1.0,
        loop: bool = False,
        on_signal: Optional[Callable[[ExecutionSignal], None]] = None,
        on_complete: Optional[Callable[[], None]] = None
    ) -> None:
        """
        Run with replay data.
        
        Args:
            file_path: Path to recorded data
            speed: Replay speed (1.0 = real-time, 0 = instant)
            loop: Loop replay
            on_signal: Signal callback
            on_complete: Completion callback
        """
        if on_signal:
            self._on_signal = on_signal
        
        config = ReplayConfig(
            file_path=file_path,
            speed=speed,
            loop=loop,
            skip_invalid=True
        )
        
        replay = ReplayAdapter(
            config,
            delivery_mode=SnapshotDeliveryMode.CALLBACK
        )
        
        def on_snapshot(snapshot: Snapshot):
            self.process_snapshot(snapshot)
        
        replay.on_snapshot = on_snapshot
        
        if replay.load() and replay.start():
            try:
                while replay.running:
                    replay.get_snapshot(timeout=1.0)
            except KeyboardInterrupt:
                pass
            finally:
                replay.stop()
        
        if on_complete:
            on_complete()
    
    def reset(self) -> None:
        """Reset engine state."""
        self._engine.reset()
    
    def get_stats(self) -> dict:
        """Get engine statistics."""
        return self._engine.get_stats()


def quick_test():
    """
    Quick test function.
    
    Creates mock data and processes it to verify the engine works.
    """
    print("Running quick test...")
    
    # Create runner
    runner = QuantEngineRunner()
    
    # Process some mock snapshots
    for i in range(10):
        ltp = 100.0 + i * 0.05  # Slightly increasing price
        composite = runner.process_mock(symbol='TEST', ltp=ltp)
        
        if composite:
            print(f"Snapshot {i+1}: Composite={composite.value:.3f}, "
                  f"Regime={composite.regime.name}, "
                  f"Threshold={composite.threshold_used:.3f}")
    
    # Print stats
    stats = runner.get_stats()
    print(f"\nStats: {stats}")
    
    print("\nQuick test complete!")


if __name__ == '__main__':
    quick_test()
