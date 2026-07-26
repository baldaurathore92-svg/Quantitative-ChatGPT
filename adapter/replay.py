"""
Replay Adapter for historical data and backtesting.

Reads recorded snapshots and replays them with proper timing.
Useful for:
- Feature development
- Strategy backtesting
- Debugging

The replay adapter implements the same interface as AngelOneWebSocket
for easy swapping in testing.
"""

import json
import time
import gzip
from typing import Optional, Callable, List, Dict, Any, Iterator
from pathlib import Path
from dataclasses import dataclass

from utils.types import Snapshot, PriceLevel
from utils.logging_utils import StructuredLogger


@dataclass
class ReplayConfig:
    """Configuration for replay adapter."""
    file_path: str
    speed: float = 1.0  # 1.0 = real-time, 0.0 = instant, 2.0 = 2x speed
    loop: bool = False  # Loop back to start when finished
    skip_invalid: bool = True  # Skip invalid snapshots


class SnapshotRecorder:
    """
    Record snapshots to file for later replay.
    
    File format: JSON Lines (one JSON object per line)
    Each line contains a serialized snapshot.
    """
    
    def __init__(self, output_path: str, compress: bool = True):
        self._output_path = Path(output_path)
        self._compress = compress
        self._file = None
        self._count = 0
        self._logger = StructuredLogger('SnapshotRecorder')
    
    def open(self) -> None:
        """Open file for writing."""
        if self._compress:
            self._file = gzip.open(self._output_path, 'wt', encoding='utf-8')
        else:
            self._file = open(self._output_path, 'w', encoding='utf-8')
    
    def record(self, snapshot: Snapshot) -> None:
        """Record a snapshot."""
        if self._file is None:
            self.open()
        
        # Serialize snapshot
        data = self._serialize_snapshot(snapshot)
        self._file.write(json.dumps(data) + '\n')
        self._count += 1
    
    def _serialize_snapshot(self, snapshot: Snapshot) -> Dict[str, Any]:
        """Serialize snapshot to dict."""
        return {
            'symbol': snapshot.symbol,
            'timestamp': snapshot.timestamp,
            'ltp': snapshot.ltp,
            'ltp_quantity': snapshot.ltp_quantity,
            'volume_traded': snapshot.volume_traded,
            'total_buy_qty': snapshot.total_buy_qty,
            'total_sell_qty': snapshot.total_sell_qty,
            'bids': [{'price': b.price, 'quantity': b.quantity, 'order_count': b.order_count} 
                     for b in snapshot.bids],
            'asks': [{'price': a.price, 'quantity': a.quantity, 'order_count': a.order_count} 
                     for a in snapshot.asks],
            'sequence': snapshot.sequence,
            'exchange_timestamp': snapshot.exchange_timestamp
        }
    
    def close(self) -> None:
        """Close file."""
        if self._file:
            self._file.close()
            self._file = None
        self._logger.info(f"Recorded {self._count} snapshots to {self._output_path}")
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class ReplayAdapter:
    """
    Replay adapter for historical data.
    
    Reads recorded snapshots and provides them through the same interface
    as the live WebSocket adapter.
    
    Features:
    - Speed control (real-time, instant, custom speed)
    - Loop option
    - Filter by symbol
    - Pause/resume
    - Seek to timestamp
    
    USAGE:
        config = ReplayConfig('recordings/2024-01-15.jsonl.gz', speed=2.0)
        replay = ReplayAdapter(config)
        
        replay.on_snapshot = my_callback
        
        replay.start()
        while replay.running:
            snapshot = replay.get_snapshot(timeout=1.0)
            if snapshot:
                process(snapshot)
        
        replay.stop()
    """
    
    def __init__(self, config: ReplayConfig):
        self._config = config
        self._logger = StructuredLogger('ReplayAdapter')
        
        # State
        self._running = False
        self._paused = False
        self._position = 0
        
        # Data
        self._snapshots: List[Dict] = []
        self._current_index = 0
        
        # Callbacks (same interface as WebSocket)
        self.on_snapshot: Optional[Callable[[Snapshot], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None
        
        # Timing
        self._last_replay_time: Optional[float] = None
        self._start_time: Optional[float] = None
    
    def load(self) -> bool:
        """
        Load snapshots from file.
        
        Returns:
            True if loaded successfully
        """
        path = Path(self._config.file_path)
        
        if not path.exists():
            self._logger.error(f"File not found: {path}")
            return False
        
        try:
            # Detect compression
            if path.suffix == '.gz' or path.suffixes[-2:] == ['.jsonl', '.gz']:
                opener = gzip.open
            else:
                opener = open
            
            self._snapshots.clear()
            
            with opener(path, 'rt', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        self._snapshots.append(data)
                    except json.JSONDecodeError as e:
                        if not self._config.skip_invalid:
                            self._logger.warning(f"Invalid JSON: {e}")
            
            self._logger.info(f"Loaded {len(self._snapshots)} snapshots from {path}")
            return True
            
        except Exception as e:
            self._logger.error(f"Load error: {e}")
            return False
    
    def start(self) -> bool:
        """
        Start replay.
        
        Returns:
            True if started successfully
        """
        if not self._snapshots:
            if not self.load():
                return False
        
        if not self._snapshots:
            self._logger.error("No snapshots to replay")
            return False
        
        self._running = True
        self._paused = False
        self._current_index = 0
        self._start_time = time.time()
        self._last_replay_time = None
        
        if self.on_connect:
            self.on_connect()
        
        return True
    
    def stop(self) -> None:
        """Stop replay."""
        self._running = False
        
        if self.on_disconnect:
            self.on_disconnect()
    
    def pause(self) -> None:
        """Pause replay."""
        self._paused = True
    
    def resume(self) -> None:
        """Resume replay."""
        self._paused = False
    
    def get_snapshot(self, timeout: float = 1.0) -> Optional[Snapshot]:
        """
        Get next snapshot.
        
        Respects speed setting for timing.
        """
        if not self._running or self._paused:
            return None
        
        if self._current_index >= len(self._snapshots):
            # End of recording
            if self._config.loop:
                self._current_index = 0
                self._logger.info("Looping replay")
            else:
                self._running = False
                return None
        
        if self._current_index >= len(self._snapshots):
            return None
        
        # Get snapshot data
        data = self._snapshots[self._current_index]
        snapshot = self._parse_snapshot(data)
        
        if snapshot is None:
            if self._config.skip_invalid:
                self._current_index += 1
                return self.get_snapshot(timeout)
            return None
        
        # Apply speed control
        if self._config.speed > 0 and self._last_replay_time is not None:
            # Calculate required delay
            current_ts = data.get('timestamp', 0)
            prev_ts = self._snapshots[self._current_index - 1].get('timestamp', 0) if self._current_index > 0 else current_ts
            
            actual_dt = time.time() - self._last_replay_time
            required_dt = (current_ts - prev_ts) / self._config.speed if self._config.speed > 0 else 0
            
            if actual_dt < required_dt:
                time.sleep(required_dt - actual_dt)
        
        self._current_index += 1
        self._last_replay_time = time.time()
        
        if self.on_snapshot:
            self.on_snapshot(snapshot)
        
        return snapshot
    
    def _parse_snapshot(self, data: Dict) -> Optional[Snapshot]:
        """Parse snapshot from dict."""
        try:
            bids = tuple(
                PriceLevel(
                    price=float(b['price']),
                    quantity=int(b['quantity']),
                    order_count=int(b.get('order_count', 0))
                )
                for b in data.get('bids', [])
            )
            
            asks = tuple(
                PriceLevel(
                    price=float(a['price']),
                    quantity=int(a['quantity']),
                    order_count=int(a.get('order_count', 0))
                )
                for a in data.get('asks', [])
            )
            
            return Snapshot(
                symbol=data['symbol'],
                timestamp=float(data['timestamp']),
                ltp=float(data['ltp']),
                ltp_quantity=int(data.get('ltp_quantity', 0)),
                volume_traded=int(data.get('volume_traded', 0)),
                total_buy_qty=int(data.get('total_buy_qty', 0)),
                total_sell_qty=int(data.get('total_sell_qty', 0)),
                bids=bids,
                asks=asks,
                sequence=int(data.get('sequence', 0)),
                exchange_timestamp=float(data.get('exchange_timestamp', 0))
            )
            
        except (KeyError, ValueError, TypeError) as e:
            self._logger.warning(f"Parse error: {e}")
            return None
    
    def seek_to_time(self, timestamp: float) -> bool:
        """Seek to specific timestamp."""
        for i, data in enumerate(self._snapshots):
            if data.get('timestamp', 0) >= timestamp:
                self._current_index = i
                return True
        return False
    
    def seek_to_index(self, index: int) -> bool:
        """Seek to specific index."""
        if 0 <= index < len(self._snapshots):
            self._current_index = index
            return True
        return False
    
    @property
    def running(self) -> bool:
        return self._running
    
    @property
    def position(self) -> float:
        """Get replay position (0.0 to 1.0)."""
        if not self._snapshots:
            return 0.0
        return self._current_index / len(self._snapshots)
    
    @property
    def total_snapshots(self) -> int:
        return len(self._snapshots)
    
    @property
    def current_index(self) -> int:
        return self._current_index
    
    @property
    def is_connected(self) -> bool:
        """Implement same interface as WebSocket."""
        return self._running
    
    def subscribe(self, symbols: List[str]) -> bool:
        """Implement same interface as WebSocket (no-op for replay)."""
        return True
    
    def disconnect(self) -> None:
        """Implement same interface as WebSocket."""
        self.stop()


def create_mock_snapshot(
    symbol: str,
    ltp: float,
    spread_ticks: float = 2.0,
    depth_levels: int = 5,
    tick_size: float = 0.05
) -> Snapshot:
    """
    Create a mock snapshot for testing.
    
    Useful for unit tests and feature development.
    """
    spread = spread_ticks * tick_size
    
    bids = tuple(
        PriceLevel(
            price=ltp - spread/2 - i * tick_size,
            quantity=1000 - i * 100,
            order_count=10 - i
        )
        for i in range(depth_levels)
    )
    
    asks = tuple(
        PriceLevel(
            price=ltp + spread/2 + i * tick_size,
            quantity=1000 - i * 100,
            order_count=10 - i
        )
        for i in range(depth_levels)
    )
    
    return Snapshot(
        symbol=symbol,
        timestamp=time.time(),
        ltp=ltp,
        ltp_quantity=100,
        volume_traded=1000000,
        total_buy_qty=50000,
        total_sell_qty=45000,
        bids=bids,
        asks=asks,
        sequence=0,
        exchange_timestamp=time.time()
    )
