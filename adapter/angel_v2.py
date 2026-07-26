"""
Angel One SmartAPI V2 WebSocket Adapter for Snapshot Mode 3.

Handles:
- WebSocket connection management
- Snapshot parsing (Mode 3 / SnapQuote)
- Heartbeat handling
- Reconnection logic
- Payload format handling (dict and JSON)

IMPORTANT: This adapter accepts ONLY Mode 3 (SnapQuote) data.
Mode 1 (LTP only) and Mode 2 (Quote) are NOT supported.
"""

import json
import time
import threading
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from queue import Queue, Empty
import socket

from utils.types import Snapshot, PriceLevel
from utils.logging_utils import StructuredLogger


@dataclass
class SmartAPIConfig:
    """SmartAPI configuration."""
    api_key: str
    auth_token: str
    client_code: str
    feed_token: str
    heartbeat_interval: float = 10.0
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10
    snapshot_timeout: float = 30.0


class SmartAPIParser:
    """
    Parse Angel One SmartAPI V2 Mode 3 (SnapQuote) responses.
    
    Mode 3 provides:
    - Exchange (NSE/BSE)
    - Trading Symbol
    - Last Traded Price (LTP)
    - Last Traded Quantity
    - Volume Traded for the Day
    - Total Buy Quantity
    - Total Sell Quantity
    - 5 Levels of Depth (Price, Quantity, Order Count)
    
    IMPORTANT: This parser handles BOTH dict and JSON string payloads.
    """
    
    @staticmethod
    def parse_snapquote(data: Any) -> Optional[Snapshot]:
        """
        Parse Mode 3 SnapQuote data.
        
        Accepts both dict and JSON string formats.
        
        Args:
            data: Raw API response (dict or JSON string)
        
        Returns:
            Snapshot if valid, None otherwise
        """
        # Handle JSON string
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return None
        
        # Ensure we have a dict
        if not isinstance(data, dict):
            return None
        
        # Extract fields
        try:
            # Token/symbol
            symbol = data.get('symbol', data.get('token', ''))
            if not symbol:
                return None
            
            # LTP
            ltp = float(data.get('ltp', 0))
            if ltp <= 0:
                return None
            
            # Timestamp
            timestamp = float(data.get('timestamp', time.time()))
            
            # Volume fields
            ltp_quantity = int(data.get('ltq', data.get('last_traded_quantity', 0)))
            volume_traded = int(data.get('v', data.get('volume_trade_for_the_day', 0)))
            total_buy_qty = int(data.get('tbq', data.get('total_buy_quantity', 0)))
            total_sell_qty = int(data.get('tsq', data.get('total_sell_quantity', 0)))
            
            # Parse depth levels
            bids = SmartAPIParser._parse_depth(data, 'bid', 'bids', 'bp', 'bq', 'bo')
            asks = SmartAPIParser._parse_depth(data, 'ask', 'asks', 'sp', 'sq', 'so')
            
            # Sequence number
            sequence = int(data.get('seq', data.get('sequence', 0)))
            
            # Exchange timestamp
            exchange_ts = float(data.get('exch_ts', data.get('exchange_timestamp', 0)))
            
            return Snapshot(
                symbol=str(symbol),
                timestamp=timestamp,
                ltp=ltp,
                ltp_quantity=ltp_quantity,
                volume_traded=volume_traded,
                total_buy_qty=total_buy_qty,
                total_sell_qty=total_sell_qty,
                bids=tuple(bids),
                asks=tuple(asks),
                sequence=sequence,
                exchange_timestamp=exchange_ts
            )
            
        except (KeyError, ValueError, TypeError) as e:
            return None
    
    @staticmethod
    def _parse_depth(
        data: dict,
        side1: str,
        side2: str,
        price_key: str,
        qty_key: str,
        orders_key: str
    ) -> List[PriceLevel]:
        """Parse depth levels from snapshot."""
        levels = []
        
        # Try different field naming conventions
        # API may use 'bp1', 'bp2', etc. or 'bids' array
        
        for i in range(1, 6):
            price = 0.0
            qty = 0
            orders = 0
            
            # Try array format first
            depth_array = data.get(side2, [])
            if isinstance(depth_array, list) and len(depth_array) >= i:
                level_data = depth_array[i-1]
                if isinstance(level_data, dict):
                    price = float(level_data.get('price', level_data.get('p', 0)))
                    qty = int(level_data.get('quantity', level_data.get('q', 0)))
                    orders = int(level_data.get('orders', level_data.get('o', 0)))
            
            # Try indexed format
            if price <= 0:
                price = float(data.get(f'{price_key}{i}', data.get(f'{side1[0]}p{i}', 0)))
            if qty <= 0:
                qty = int(data.get(f'{qty_key}{i}', data.get(f'{side1[0]}q{i}', 0)))
            if orders <= 0:
                orders = int(data.get(f'{orders_key}{i}', data.get(f'{side1[0]}o{i}', 0)))
            
            if price > 0 and qty > 0:
                levels.append(PriceLevel(price=price, quantity=qty, order_count=orders))
        
        return levels


class AngelOneWebSocket:
    """
    WebSocket adapter for Angel One SmartAPI V2.
    
    Features:
    - Mode 3 (SnapQuote) subscription
    - Heartbeat handling
    - Reconnection with loop (never recursive)
    - Dict and JSON payload handling
    - Thread-safe message queue
    
    USAGE:
        config = SmartAPIConfig(api_key='...', auth_token='...', ...)
        ws = AngelOneWebSocket(config)
        
        ws.on_snapshot = my_callback
        
        if ws.connect():
            ws.subscribe(['RELIANCE', 'TCS'])
            # Process messages
            while running:
                snapshot = ws.get_snapshot(timeout=1.0)
                if snapshot:
                    # Process snapshot
                    pass
        
        ws.disconnect()
    """
    
    def __init__(self, config: SmartAPIConfig):
        self._config = config
        self._logger = StructuredLogger('AngelOneWebSocket')
        
        # Connection state with thread-safe lock
        self._state_lock = threading.Lock()
        self._connected = False
        self._reconnect_count = 0
        self._ws = None
        self._ws_thread = None
        self._running = False
        
        # Message queue
        self._message_queue: Queue = Queue(maxsize=1000)
        
        # Callbacks
        self.on_snapshot: Optional[Callable[[Snapshot], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None
        
        # Subscriptions
        self._subscribed_symbols: List[str] = []
        
        # Last heartbeat
        self._last_heartbeat = time.time()
        
        # Heartbeat thread reference
        self._heartbeat_thread: Optional[threading.Thread] = None
        
        # Parser
        self._parser = SmartAPIParser()
    
    @property
    def connected(self) -> bool:
        """Thread-safe check for connection status."""
        with self._state_lock:
            return self._connected
    
    @property
    def running(self) -> bool:
        """Thread-safe check for running status."""
        with self._state_lock:
            return self._running
    
    def connect(self) -> bool:
        """
        Connect to WebSocket.
        
        Returns True if connected successfully.
        """
        try:
            # Import WebSocket library
            try:
                import websocket
            except ImportError:
                self._logger.error("websocket-client not installed. Run: pip install websocket-client")
                return False
            
            # WebSocket URL for Angel One
            ws_url = self._build_ws_url()
            
            self._running = True
            self._ws = websocket.WebSocketApp(
                ws_url,
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close
            )
            
            # Start WebSocket thread
            self._ws_thread = threading.Thread(target=self._run_forever, daemon=True)
            self._ws_thread.start()
            
            # Wait for connection
            timeout = 10.0
            start = time.time()
            while not self.connected and time.time() - start < timeout:
                time.sleep(0.1)
            
            return self.connected
            
        except Exception as e:
            self._logger.error(f"Connection error: {e}")
            return False
    
    def _build_ws_url(self) -> str:
        """Build WebSocket URL."""
        # Angel One SmartAPI WebSocket URL
        # Format: wss://smartapisocket.angelone.in/smart-stream
        base_url = "wss://smartapisocket.angelone.in/smart-stream"
        return base_url
    
    def _run_forever(self) -> None:
        """Run WebSocket connection loop (never recursive reconnect)."""
        while self.running:
            try:
                self._ws.run_forever()
            except Exception as e:
                self._logger.error(f"WebSocket error: {e}")
            
            # Check if we should reconnect
            if self.running and self._reconnect_count < self._config.max_reconnect_attempts:
                self._reconnect_count += 1
                self._logger.info(f"Reconnecting attempt {self._reconnect_count}")
                time.sleep(self._config.reconnect_delay)
            elif self.running:
                self._logger.error("Max reconnect attempts reached")
                with self._state_lock:
                    self._running = False
                break
    
    def _on_open(self, ws) -> None:
        """Handle WebSocket open."""
        with self._state_lock:
            self._connected = True
        self._reconnect_count = 0
        self._last_heartbeat = time.time()
        
        self._logger.info("WebSocket connected")
        
        # Send authentication
        self._send_auth()
        
        # Re-subscribe to symbols
        if self._subscribed_symbols:
            self._subscribe_internal(self._subscribed_symbols)
        
        # Start heartbeat thread
        self._start_heartbeat()
        
        if self.on_connect:
            self.on_connect()
    
    def _on_close(self, ws, close_status_code, close_msg) -> None:
        """Handle WebSocket close."""
        was_connected = self._connected
        self._connected = False
        
        self._logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        
        # Clear message queue on disconnect
        while not self._message_queue.empty():
            try:
                self._message_queue.get_nowait()
            except Exception:
                break
        
        if self.on_disconnect and was_connected:
            self.on_disconnect()
    
    def _send_auth(self) -> None:
        """Send authentication message."""
        # Angel One SmartAPI V2 WebSocket authentication format
        # Reference: https://smartapi.angelone.in/docs/websocket
        auth_msg = {
            "action": 1,  # Subscribe action
            "params": {
                "mode": 3,  # Mode 3 (SnapQuote)
                "tokenKeys": []
            },
            "jwttoken": self._config.auth_token,
            "clientcode": self._config.client_code,
            "key": self._config.api_key
        }
        
        self._send_json(auth_msg)
    
    def subscribe(self, symbols: List[str]) -> bool:
        """
        Subscribe to symbols.
        
        Args:
            symbols: List of trading symbols (e.g., ['RELIANCE', 'TCS'])
        
        Returns:
            True if subscription sent successfully
        """
        self._subscribed_symbols = symbols
        
        if not self._connected:
            return False
        
        return self._subscribe_internal(symbols)
    
    def _subscribe_internal(self, symbols: List[str]) -> bool:
        """Internal subscribe."""
        try:
            # Angel One SmartAPI V2 subscription format
            # Note: Uses token list, not symbol names directly
            # In production, you would map symbols to tokens first
            token_keys = []
            for symbol in symbols:
                token_keys.append({
                    "tokenType": "NSE",  # NSE Cash
                    "tokens": [symbol]    # Token/symbol identifier
                })
            
            sub_msg = {
                "action": 1,  # Subscribe
                "params": {
                    "mode": 3,  # Mode 3 (SnapQuote)
                    "tokenKeys": token_keys
                }
            }
            
            self._send_json(sub_msg)
            return True
            
        except Exception as e:
            self._logger.error(f"Subscribe error: {e}")
            return False
    
    def _on_message(self, ws, message) -> None:
        """Handle incoming message."""
        try:
            # Parse message
            data = json.loads(message) if isinstance(message, str) else message
            
            # Check message type
            msg_type = data.get('type', data.get('t', ''))
            
            if msg_type == 'hb' or msg_type == 'heartbeat':
                # Heartbeat response
                self._last_heartbeat = time.time()
                return
            
            if msg_type == 'error':
                # Error message
                error_msg = data.get('message', data.get('msg', 'Unknown error'))
                error_code = data.get('code', data.get('errorcode', ''))
                self._logger.error(f"API error: {error_code} - {error_msg}")
                if self.on_error:
                    self.on_error(Exception(f"{error_code}: {error_msg}"))
                return
            
            # Parse as snapshot
            snapshot = self._parser.parse_snapquote(data)
            
            if snapshot:
                # Put in queue (thread-safe)
                try:
                    self._message_queue.put_nowait(snapshot)
                except Exception:
                    # Queue full, drop oldest and retry
                    try:
                        self._message_queue.get_nowait()
                        self._message_queue.put_nowait(snapshot)
                    except Exception:
                        pass  # Queue operations failed, skip this snapshot
                
                # Callback
                if self.on_snapshot:
                    try:
                        self.on_snapshot(snapshot)
                    except Exception as e:
                        self._logger.error(f"Callback error: {e}")
            else:
                # Unrecognized message format - log for debugging
                self._logger.debug(f"Unrecognized message: {str(data)[:100]}")
            
        except json.JSONDecodeError as e:
            self._logger.warning(f"Invalid JSON message: {str(message)[:100]}")
        except KeyError as e:
            self._logger.warning(f"Missing expected field: {e}")
        except ValueError as e:
            self._logger.warning(f"Value error in message: {e}")
        except TypeError as e:
            self._logger.warning(f"Type error in message: {e}")
        except Exception as e:
            self._logger.error(f"Unexpected message handling error: {e}")
    
    def _on_error(self, ws, error) -> None:
        """Handle WebSocket error."""
        self._logger.error(f"WebSocket error: {error}")
        if self.on_error:
            self.on_error(error)
    

    
    def _send_json(self, data: dict) -> bool:
        """
        Send JSON message.
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self._ws:
            self._logger.warning("WebSocket not initialized")
            return False
        
        if not self._connected:
            self._logger.warning("WebSocket not connected")
            return False
        
        try:
            self._ws.send(json.dumps(data))
            return True
        except Exception as e:
            self._logger.error(f"Send error: {e}")
            return False
    
    def _start_heartbeat(self) -> None:
        """Start heartbeat thread with timeout detection."""
        def heartbeat_loop():
            missed_heartbeats = 0
            max_missed = 3  # Max missed heartbeats before considering dead
            
            while self._running and self._connected:
                try:
                    # Check for heartbeat timeout
                    time_since_last = time.time() - self._last_heartbeat
                    expected_interval = self._config.heartbeat_interval * 2  # Allow 2x tolerance
                    
                    if time_since_last > expected_interval:
                        missed_heartbeats += 1
                        self._logger.warning(
                            f"Heartbeat timeout: {time_since_last:.1f}s since last response "
                            f"(missed: {missed_heartbeats}/{max_missed})"
                        )
                        
                        if missed_heartbeats >= max_missed:
                            self._logger.error("Max missed heartbeats - connection likely dead")
                            # Force reconnection by closing
                            if self._ws:
                                try:
                                    self._ws.close()
                                except Exception:
                                    pass
                            break
                    else:
                        missed_heartbeats = 0  # Reset on successful heartbeat
                    
                    # Send heartbeat
                    hb_msg = {"action": 0, "type": "hb"}
                    self._send_json(hb_msg)
                    time.sleep(self._config.heartbeat_interval)
                    
                except Exception as e:
                    self._logger.error(f"Heartbeat error: {e}")
                    break
        
        self._heartbeat_thread = threading.Thread(target=heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()
    
    def unsubscribe(self, symbols: List[str]) -> bool:
        """Unsubscribe from symbols."""
        try:
            token_keys = []
            for symbol in symbols:
                token_keys.append({
                    "tokenType": "NSE",
                    "tokens": [symbol]
                })
            
            unsub_msg = {
                "action": 0,  # Unsubscribe
                "params": {
                    "mode": 3,
                    "tokenKeys": token_keys
                }
            }
            
            self._send_json(unsub_msg)
            
            # Update subscribed list
            self._subscribed_symbols = [s for s in self._subscribed_symbols if s not in symbols]
            return True
            
        except Exception as e:
            self._logger.error(f"Unsubscribe error: {e}")
            return False
    
    def get_snapshot(self, timeout: float = 1.0) -> Optional[Snapshot]:
        """
        Get next snapshot from queue.
        
        Args:
            timeout: Max wait time in seconds
        
        Returns:
            Snapshot or None if timeout
        """
        try:
            return self._message_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def disconnect(self) -> None:
        """Disconnect WebSocket (thread-safe)."""
        with self._state_lock:
            self._running = False
            self._connected = False
        
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        
        # Wait for threads to finish
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=2.0)
        
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=1.0)
        
        self._logger.info("WebSocket disconnected")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._connected
    
    @property
    def queued_messages(self) -> int:
        """Get number of queued messages."""
        return self._message_queue.qsize()
