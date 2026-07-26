"""
Main entry point for Snapshot Quant Engine.

Usage:
    python main.py --config config.json
    
Or programmatically:
    from main import run_engine
    
    run_engine(config_path='config.json')
"""

import sys
import signal
import time
import argparse
from typing import Optional, Union

from config import EngineConfig
from engine.quant_engine import QuantEngine
from adapter.angel_v2 import SmartAPIConfig, AngelOneWebSocket
from adapter.replay import ReplayConfig, ReplayAdapter
from utils.types import Snapshot, CompositeScore, SnapshotDeliveryMode
from utils.logging_utils import setup_logging


class EngineRunner:
    """
    Orchestrates the complete quant engine execution.
    
    Handles:
    - Configuration loading
    - Data source connection (live or replay)
    - Signal processing loop
    - Graceful shutdown
    - Signal output
    """
    
    def __init__(self, config: EngineConfig, use_replay: bool = False):
        self._config = config
        self._logger = setup_logging(config)
        self._use_replay = use_replay
        
        # Initialize engine
        self._engine = QuantEngine(config)
        
        # Data source
        self._data_source: Optional[
            Union[AngelOneWebSocket, ReplayAdapter]
        ] = None
        self._running = False
        self._processed_count = 0
        
        # Statistics
        self._start_time: Optional[float] = None
        self._signals_generated = 0
    
    def start_live(self) -> bool:
        """
        Start with live Angel One SmartAPI connection.
        
        Returns:
            True if started successfully
        """
        # Validate API config
        if not self._config.api.api_key:
            self._logger.error("Missing API key in configuration")
            return False
        
        if not self._config.api.auth_token:
            self._logger.error("Missing auth token in configuration")
            return False

        if not self._config.subscriptions:
            self._logger.error("At least one live subscription is required")
            return False

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
        
        data_source = AngelOneWebSocket(
            api_config,
            delivery_mode=SnapshotDeliveryMode.PULL
        )

        # Lifecycle callbacks only; snapshots are owned by the main pull loop.
        data_source.on_error = self._on_error
        data_source.on_connect = self._on_connect
        data_source.on_disconnect = self._on_disconnect

        # Connect
        if not data_source.connect():
            self._logger.error("Failed to connect to data source")
            return False
        
        # Subscribe to configured exchange segments and numeric instrument tokens.
        if not data_source.subscribe(self._config.subscriptions):
            self._logger.error("Failed to subscribe to configured tokens")
            data_source.disconnect()
            return False

        self._data_source = data_source
        self._running = True
        self._start_time = time.time()
        
        self._logger.info(
            "Engine started (live)",
            subscriptions=[
                subscription.as_dict()
                for subscription in self._config.subscriptions
            ]
        )
        
        return True
    
    def start_replay(self, file_path: str, speed: float = 1.0) -> bool:
        """
        Start with replay data.
        
        Args:
            file_path: Path to recorded data file
            speed: Replay speed (1.0 = real-time, 0 = instant)
        
        Returns:
            True if started successfully
        """
        replay_config = ReplayConfig(
            file_path=file_path,
            speed=speed,
            loop=False,
            skip_invalid=True
        )
        
        data_source = ReplayAdapter(
            replay_config,
            delivery_mode=SnapshotDeliveryMode.PULL
        )

        # Lifecycle callbacks only; snapshots are owned by the main pull loop.
        data_source.on_error = self._on_error
        data_source.on_connect = self._on_connect
        data_source.on_disconnect = self._on_disconnect

        # Load and start
        if not data_source.load():
            self._logger.error("Failed to load replay data")
            return False
        
        if not data_source.start():
            self._logger.error("Failed to start replay")
            return False

        self._data_source = data_source
        self._running = True
        self._start_time = time.time()
        
        self._logger.info(
            "Engine started (replay)",
            file=file_path,
            total_snapshots=data_source.total_snapshots
        )
        
        return True
    
    def run(self) -> None:
        """
        Run the main processing loop.
        
        Blocks until shutdown signal or data source ends.
        """
        data_source = self._data_source
        if not self._running or data_source is None:
            self._logger.error("Engine not started")
            return
        
        # Setup signal handlers
        self._setup_signal_handlers()
        
        # Main loop
        while self._running:
            try:
                snapshot = data_source.get_snapshot(timeout=1.0)

                if snapshot:
                    self._process_snapshot(snapshot)
                elif not data_source.is_connected and not data_source.running:
                    # Replay completed or the live adapter exhausted retries.
                    self._logger.info("Data source stopped")
                    break
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                self._logger.error(f"Processing error: {e}")
        
        self._stop()
    
    def _process_snapshot(self, snapshot: Snapshot) -> None:
        """Process a single snapshot."""
        # Process through engine
        composite = self._engine.process(snapshot)
        
        self._processed_count += 1
        
        if composite is None:
            return
        
        # Log periodically
        if self._processed_count % 100 == 0:
            self._logger.info(
                f"Processed {self._processed_count} snapshots",
                composite=composite.value,
                regime=composite.regime.name
            )
        
        # The state-machine transition policy is authoritative for both entries
        # and exits, including exits below the composite activation threshold.
        self._handle_signal(snapshot, composite)
    
    def _handle_signal(self, snapshot: Snapshot, composite: CompositeScore) -> None:
        """Emit a one-shot command created by the latest state transition."""
        exec_signal = self._engine.get_execution_signal(snapshot, composite)

        if exec_signal:
            self._signals_generated += 1
            self._logger.log_signal(
                symbol=exec_signal.symbol,
                signal_type=exec_signal.signal_type.name,
                target_price=exec_signal.target_price,
                composite=exec_signal.composite_value,
                confidence=exec_signal.confidence
            )
            
            # Here you would send to your execution system
            # For now, just log
            self._output_signal(exec_signal)
    
    def _output_signal(self, signal) -> None:
        """Output signal to external system."""
        # Implement your signal output here
        # Examples:
        # - Send to order management system
        # - Write to signal file
        # - Publish to message queue
        pass
    
    def _on_snapshot(self, snapshot: Snapshot) -> None:
        """Callback for new snapshot (used by live WebSocket)."""
        self._process_snapshot(snapshot)
    
    def _on_error(self, error: Exception) -> None:
        """Callback for errors."""
        self._logger.error(f"Data source error: {error}")
    
    def _on_connect(self) -> None:
        """Callback for connection."""
        self._logger.info("Data source connected")
    
    def _on_disconnect(self) -> None:
        """Callback for disconnection."""
        self._logger.info("Data source disconnected")
    
    def _setup_signal_handlers(self) -> None:
        """Setup OS signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self._logger.info(f"Received signal {signum}, shutting down")
            self._running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _stop(self) -> None:
        """Stop the engine."""
        self._running = False
        
        if self._data_source:
            self._data_source.disconnect()
        
        # Print statistics
        elapsed = time.time() - self._start_time if self._start_time else 0
        
        self._logger.info(
            "Engine stopped",
            processed=self._processed_count,
            signals=self._signals_generated,
            elapsed_seconds=elapsed,
            rate=self._processed_count / elapsed if elapsed > 0 else 0
        )
    
    def stop(self) -> None:
        """External stop command."""
        self._running = False
    
    def get_stats(self) -> dict:
        """Get engine statistics."""
        stats = self._engine.get_stats()
        stats.update({
            'processed_count': self._processed_count,
            'signals_generated': self._signals_generated,
            'running': self._running
        })
        return stats


def run_engine(
    config_path: str = 'config.json',
    replay_path: Optional[str] = None,
    replay_speed: float = 1.0
) -> None:
    """
    Run the quant engine.
    
    Args:
        config_path: Path to configuration file
        replay_path: If set, use replay instead of live
        replay_speed: Speed for replay
    """
    # Load configuration
    config = EngineConfig.from_json(config_path)
    
    # Create runner
    runner = EngineRunner(config, use_replay=replay_path is not None)
    
    # Start
    if replay_path:
        success = runner.start_replay(replay_path, replay_speed)
    else:
        success = runner.start_live()
    
    if not success:
        sys.exit(1)
    
    # Run
    runner.run()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description='Snapshot Quant Engine')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--replay', help='Replay file path (instead of live)')
    parser.add_argument('--speed', type=float, default=1.0, help='Replay speed')
    
    args = parser.parse_args()
    
    run_engine(
        config_path=args.config,
        replay_path=args.replay,
        replay_speed=args.speed
    )


if __name__ == '__main__':
    main()
