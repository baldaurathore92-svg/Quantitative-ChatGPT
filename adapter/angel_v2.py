"""Angel One SmartAPI WebSocket V2 adapter for Mode 3 SnapQuote data."""

from __future__ import annotations

import json
import math
import struct
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from queue import Empty, Full, Queue
from typing import Any, Callable, Optional

from utils.logging_utils import StructuredLogger
from utils.types import MarketSubscription, PriceLevel, Snapshot, SnapshotDeliveryMode

SmartAPISubscription = MarketSubscription


@dataclass(frozen=True)
class SmartAPIConfig:
    """Authentication and connection settings for SmartAPI WebSocket V2."""

    api_key: str
    auth_token: str
    client_code: str
    feed_token: str
    heartbeat_interval: float = 10.0
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 10
    snapshot_timeout: float = 30.0
    correlation_id: str = "snapquote1"

    def __post_init__(self) -> None:
        required = {
            "api_key": self.api_key,
            "auth_token": self.auth_token,
            "client_code": self.client_code,
            "feed_token": self.feed_token,
        }
        for name, value in required.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.correlation_id, str) or not self.correlation_id.isalnum():
            raise ValueError("correlation_id must be alphanumeric")
        if not 1 <= len(self.correlation_id) <= 10:
            raise ValueError("correlation_id must contain at most 10 characters")
        if self.heartbeat_interval <= 0:
            raise ValueError("heartbeat_interval must be positive")
        if self.reconnect_delay < 0:
            raise ValueError("reconnect_delay cannot be negative")
        if self.max_reconnect_attempts < 0:
            raise ValueError("max_reconnect_attempts cannot be negative")
        if self.snapshot_timeout <= 0:
            raise ValueError("snapshot_timeout must be positive")


class SmartAPIParser:
    """Parse native V2 SnapQuote frames and explicit replay compatibility data."""

    SNAP_QUOTE_MODE = 3
    SNAP_QUOTE_PACKET_LENGTH = 379
    TOKEN_START = 2
    TOKEN_END = 27
    BEST_FIVE_START = 147
    BEST_FIVE_END = 347
    DEPTH_ENTRY_LENGTH = 20
    PRICE_SCALE = 100.0

    @classmethod
    def parse_snapquote(cls, data: Any) -> Optional[Snapshot]:
        """Parse binary production data, or JSON/dict replay compatibility data."""
        if isinstance(data, (bytes, bytearray, memoryview)):
            return cls.parse_binary_snapquote(bytes(data))
        return cls.parse_compat_snapquote(data)

    @classmethod
    def parse_binary_snapquote(cls, frame: bytes) -> Optional[Snapshot]:
        """Parse one little-endian 379-byte SmartAPI V2 Mode 3 frame safely."""
        if not isinstance(frame, bytes) or len(frame) != cls.SNAP_QUOTE_PACKET_LENGTH:
            return None

        try:
            mode = cls._unpack(frame, 0, "B")
            exchange_type_value = cls._unpack(frame, 1, "B")
            if mode != cls.SNAP_QUOTE_MODE:
                return None
            if (
                isinstance(exchange_type_value, bool)
                or not isinstance(exchange_type_value, int)
                or exchange_type_value
                not in MarketSubscription.SUPPORTED_EXCHANGE_TYPES
            ):
                return None
            exchange_type = exchange_type_value

            token = cls._parse_token(frame[cls.TOKEN_START:cls.TOKEN_END])
            if token is None:
                return None

            sequence = cls._nonnegative_int(cls._unpack(frame, 27, "q"))
            exchange_timestamp_ms = cls._nonnegative_int(cls._unpack(frame, 35, "q"))
            ltp_raw = cls._nonnegative_int(cls._unpack(frame, 43, "q"))
            ltp_quantity = cls._nonnegative_int(cls._unpack(frame, 51, "q"))
            volume_traded = cls._nonnegative_int(cls._unpack(frame, 67, "q"))
            total_buy_qty = cls._quantity_from_double(cls._unpack(frame, 75, "d"))
            total_sell_qty = cls._quantity_from_double(cls._unpack(frame, 83, "d"))
            last_traded_timestamp_ms = cls._nonnegative_int(cls._unpack(frame, 123, "q"))

            if (
                sequence is None
                or exchange_timestamp_ms is None
                or ltp_raw is None
                or ltp_quantity is None
                or volume_traded is None
                or total_buy_qty is None
                or total_sell_qty is None
                or last_traded_timestamp_ms is None
            ):
                return None
            if ltp_raw == 0:
                return None

            depth = cls._parse_best_five(frame[cls.BEST_FIVE_START:cls.BEST_FIVE_END])
            if depth is None:
                return None
            bids, asks = depth

            exchange_timestamp = cls._milliseconds_to_seconds(exchange_timestamp_ms)
            traded_timestamp = cls._milliseconds_to_seconds(last_traded_timestamp_ms)
            timestamp = exchange_timestamp or traded_timestamp
            if timestamp <= 0:
                return None

            snapshot = Snapshot(
                symbol=token,
                token=token,
                exchange_type=exchange_type,
                timestamp=timestamp,
                ltp=ltp_raw / cls.PRICE_SCALE,
                ltp_quantity=ltp_quantity,
                volume_traded=volume_traded,
                total_buy_qty=total_buy_qty,
                total_sell_qty=total_sell_qty,
                bids=bids,
                asks=asks,
                sequence=sequence,
                exchange_timestamp=exchange_timestamp,
            )
            return snapshot if snapshot.is_valid() else None
        except (OverflowError, struct.error, TypeError, ValueError):
            return None

    @classmethod
    def parse_compat_snapquote(cls, data: Any) -> Optional[Snapshot]:
        """Parse explicit JSON/dict data used by tests and recorded replays."""
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return None
        if not isinstance(data, Mapping):
            return None

        try:
            token_value = data.get("token", "")
            symbol_value = data.get("symbol", token_value)
            symbol = str(symbol_value).strip()
            token = str(token_value).strip()
            exchange_type = int(
                data.get("exchange_type", data.get("exchangeType", 0)) or 0
            )
            if not symbol:
                return None

            ltp = float(data.get("ltp", 0) or 0)
            timestamp = float(data.get("timestamp", time.time()) or time.time())
            snapshot = Snapshot(
                symbol=symbol,
                token=token,
                exchange_type=exchange_type,
                timestamp=timestamp,
                ltp=ltp,
                ltp_quantity=int(
                    data.get("ltq", data.get("last_traded_quantity", 0)) or 0
                ),
                volume_traded=int(
                    data.get("v", data.get("volume_trade_for_the_day", 0)) or 0
                ),
                total_buy_qty=int(
                    data.get("tbq", data.get("total_buy_quantity", 0)) or 0
                ),
                total_sell_qty=int(
                    data.get("tsq", data.get("total_sell_quantity", 0)) or 0
                ),
                bids=tuple(cls._parse_compat_depth(data, "bid", "bids", "bp", "bq", "bo")),
                asks=tuple(cls._parse_compat_depth(data, "ask", "asks", "sp", "sq", "so")),
                sequence=int(data.get("seq", data.get("sequence", 0)) or 0),
                exchange_timestamp=float(
                    data.get("exch_ts", data.get("exchange_timestamp", 0)) or 0
                ),
            )
            return snapshot if snapshot.is_valid() else None
        except (OverflowError, TypeError, ValueError):
            return None

    @staticmethod
    def _unpack(frame: bytes, offset: int, format_code: str) -> int | float:
        return struct.unpack_from(f"<{format_code}", frame, offset)[0]

    @staticmethod
    def _parse_token(raw_token: bytes) -> Optional[str]:
        try:
            token = raw_token.split(b"\x00", 1)[0].decode("ascii").strip()
        except UnicodeDecodeError:
            return None
        if not token.isdecimal() or int(token) <= 0:
            return None
        return token

    @staticmethod
    def _nonnegative_int(value: int | float) -> Optional[int]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return None
        return value

    @staticmethod
    def _quantity_from_double(value: int | float) -> Optional[int]:
        if not isinstance(value, float) or not math.isfinite(value) or value < 0:
            return None
        rounded = round(value)
        if not math.isclose(value, rounded, abs_tol=1e-6):
            return None
        return int(rounded)

    @staticmethod
    def _milliseconds_to_seconds(value: int) -> float:
        if value <= 0:
            return 0.0
        return value / 1000.0

    @classmethod
    def _parse_best_five(
        cls,
        depth_bytes: bytes,
    ) -> Optional[tuple[tuple[PriceLevel, ...], tuple[PriceLevel, ...]]]:
        if len(depth_bytes) != cls.BEST_FIVE_END - cls.BEST_FIVE_START:
            return None

        bids: list[PriceLevel] = []
        asks: list[PriceLevel] = []
        try:
            for index in range(10):
                offset = index * cls.DEPTH_ENTRY_LENGTH
                side = cls._unpack(depth_bytes, offset, "H")
                quantity = cls._nonnegative_int(cls._unpack(depth_bytes, offset + 2, "q"))
                price_raw = cls._nonnegative_int(cls._unpack(depth_bytes, offset + 10, "q"))
                order_count = cls._nonnegative_int(cls._unpack(depth_bytes, offset + 18, "H"))
                if (
                    side not in (0, 1)
                    or quantity is None
                    or price_raw is None
                    or order_count is None
                ):
                    return None
                if quantity == 0 or price_raw == 0:
                    return None
                level = PriceLevel(
                    price=price_raw / cls.PRICE_SCALE,
                    quantity=quantity,
                    order_count=order_count,
                )
                # SmartAPI's official V2 decoder swaps the flag-labelled
                # collections when exposing SnapQuote buy/sell depth: flag 0
                # is effective sell depth and flag 1 is effective buy depth.
                (asks if side == 0 else bids).append(level)
        except (struct.error, TypeError, ValueError):
            return None

        if len(bids) != 5 or len(asks) != 5:
            return None
        bids.sort(key=lambda level: level.price, reverse=True)
        asks.sort(key=lambda level: level.price)
        return tuple(bids), tuple(asks)

    @staticmethod
    def _parse_compat_depth(
        data: Mapping[str, Any],
        side_name: str,
        array_name: str,
        price_key: str,
        quantity_key: str,
        orders_key: str,
    ) -> list[PriceLevel]:
        levels: list[PriceLevel] = []
        depth_array = data.get(array_name, [])
        for index in range(1, 6):
            price = 0.0
            quantity = 0
            orders = 0
            if isinstance(depth_array, list) and len(depth_array) >= index:
                raw_level = depth_array[index - 1]
                if isinstance(raw_level, Mapping):
                    price = float(raw_level.get("price", raw_level.get("p", 0)) or 0)
                    quantity = int(
                        raw_level.get("quantity", raw_level.get("q", 0)) or 0
                    )
                    orders = int(raw_level.get("orders", raw_level.get("o", 0)) or 0)
            if price <= 0:
                price = float(
                    data.get(f"{price_key}{index}", data.get(f"{side_name[0]}p{index}", 0))
                    or 0
                )
            if quantity <= 0:
                quantity = int(
                    data.get(
                        f"{quantity_key}{index}",
                        data.get(f"{side_name[0]}q{index}", 0),
                    )
                    or 0
                )
            if orders <= 0:
                orders = int(
                    data.get(
                        f"{orders_key}{index}",
                        data.get(f"{side_name[0]}o{index}", 0),
                    )
                    or 0
                )
            if price > 0 and quantity > 0:
                levels.append(PriceLevel(price=price, quantity=quantity, order_count=orders))
        return levels


class AngelOneWebSocket:
    """Thread-safe SmartAPI V2 WebSocket client with exclusive delivery modes."""

    URL = "wss://smartapisocket.angelone.in/smart-stream"
    SNAP_QUOTE_MODE = 3
    SUBSCRIBE_ACTION = 1
    UNSUBSCRIBE_ACTION = 0

    def __init__(
        self,
        config: SmartAPIConfig,
        delivery_mode: SnapshotDeliveryMode = SnapshotDeliveryMode.PULL,
    ) -> None:
        if not isinstance(delivery_mode, SnapshotDeliveryMode):
            raise ValueError("delivery_mode must be a SnapshotDeliveryMode")
        self._config = config
        self._delivery_mode = delivery_mode
        self._logger = StructuredLogger("AngelOneWebSocket")
        self._state_lock = threading.RLock()
        self._connected = False
        self._running = False
        self._reconnect_count = 0
        self._ws: Any = None
        self._ws_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._message_queue: Queue[Snapshot] = Queue(maxsize=1000)
        self._subscriptions: tuple[MarketSubscription, ...] = ()
        now = time.monotonic()
        self._last_heartbeat = now
        self._last_snapshot = now
        self._stale_close_requested = False
        self._parser = SmartAPIParser()

        self.on_snapshot: Optional[Callable[[Snapshot], None]] = None
        self.on_error: Optional[Callable[[Exception], None]] = None
        self.on_connect: Optional[Callable[[], None]] = None
        self.on_disconnect: Optional[Callable[[], None]] = None

    @property
    def connected(self) -> bool:
        with self._state_lock:
            return self._connected

    @property
    def running(self) -> bool:
        with self._state_lock:
            return self._running

    @property
    def is_connected(self) -> bool:
        return self.connected

    @property
    def delivery_mode(self) -> SnapshotDeliveryMode:
        return self._delivery_mode

    @property
    def queued_messages(self) -> int:
        return self._message_queue.qsize()

    @property
    def subscriptions(self) -> tuple[MarketSubscription, ...]:
        with self._state_lock:
            return self._subscriptions

    def connect(self) -> bool:
        """Construct the authenticated WebSocketApp and wait for connection."""
        try:
            import websocket  # type: ignore[import-not-found]
        except ImportError:
            self._logger.error("websocket-client is not installed")
            return False

        with self._state_lock:
            if self._running:
                return self._connected
            self._running = True
            self._reconnect_count = 0
            self._ws = websocket.WebSocketApp(
                self.URL,
                header=self._auth_headers(),
                on_open=self._on_open,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
            )
            ws_thread = threading.Thread(
                target=self._run_forever,
                name="angel-websocket",
                daemon=True,
            )
            self._ws_thread = ws_thread

        ws_thread.start()
        self._start_heartbeat_monitor()

        deadline = time.monotonic() + 10.0
        while not self.connected and self.running and time.monotonic() < deadline:
            time.sleep(0.05)
        if self.connected:
            return True
        self.disconnect()
        return False

    def _auth_headers(self) -> dict[str, str]:
        return {
            "Authorization": self._config.auth_token,
            "x-api-key": self._config.api_key,
            "x-client-code": self._config.client_code,
            "x-feed-token": self._config.feed_token,
        }

    def _run_forever(self) -> None:
        while self.running:
            with self._state_lock:
                ws = self._ws
            if ws is None:
                break
            try:
                ws.run_forever()
            except Exception as exc:
                self._notify_error(exc)

            with self._state_lock:
                if not self._running:
                    break
                if self._reconnect_count >= self._config.max_reconnect_attempts:
                    self._running = False
                    exhausted = True
                else:
                    self._reconnect_count += 1
                    attempt = self._reconnect_count
                    exhausted = False
            if exhausted:
                self._notify_error(RuntimeError("maximum reconnect attempts reached"))
                break
            self._logger.info(f"Reconnecting attempt {attempt}")
            time.sleep(self._config.reconnect_delay)

    def _on_open(self, ws: Any) -> None:
        now = time.monotonic()
        with self._state_lock:
            self._connected = True
            self._reconnect_count = 0
            self._last_heartbeat = now
            self._last_snapshot = now
            self._stale_close_requested = False
            subscriptions = self._subscriptions
        self._logger.info("WebSocket connected")
        if subscriptions:
            self._subscribe_internal(subscriptions, self.SUBSCRIBE_ACTION)
        if self.on_connect is not None:
            self.on_connect()

    def _on_close(self, ws: Any, close_status_code: Any, close_msg: Any) -> None:
        with self._state_lock:
            was_connected = self._connected
            self._connected = False
            self._stale_close_requested = False
        self._logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self._clear_message_queue()
        if was_connected and self.on_disconnect is not None:
            self.on_disconnect()

    def _on_error(self, ws: Any, error: Any) -> None:
        exception = error if isinstance(error, Exception) else RuntimeError(str(error))
        self._notify_error(exception)

    def _on_message(self, ws: Any, message: Any) -> None:
        if isinstance(message, str):
            if message.strip().lower() == "pong":
                with self._state_lock:
                    self._last_heartbeat = time.monotonic()
                return
            try:
                control = json.loads(message)
            except json.JSONDecodeError:
                self._logger.warning("Invalid JSON compatibility message")
                return
            if isinstance(control, Mapping) and self._handle_control_message(control):
                return
            snapshot = self._parser.parse_compat_snapquote(control)
        elif isinstance(message, Mapping):
            if self._handle_control_message(message):
                return
            snapshot = self._parser.parse_compat_snapquote(message)
        elif isinstance(message, (bytes, bytearray, memoryview)):
            snapshot = self._parser.parse_binary_snapquote(bytes(message))
        else:
            snapshot = None

        if snapshot is None:
            self._logger.warning("Rejected malformed or unsupported WebSocket frame")
            return
        with self._state_lock:
            self._last_snapshot = time.monotonic()
            self._last_heartbeat = self._last_snapshot
        self._deliver(snapshot)

    def _handle_control_message(self, message: Mapping[str, Any]) -> bool:
        message_type = str(message.get("type", message.get("t", ""))).lower()
        if message_type in {"hb", "heartbeat", "pong"}:
            with self._state_lock:
                self._last_heartbeat = time.monotonic()
            return True
        if message_type == "error" or message.get("errorcode"):
            code = message.get("code", message.get("errorcode", ""))
            detail = message.get("message", message.get("msg", "Unknown error"))
            self._notify_error(RuntimeError(f"{code}: {detail}"))
            return True
        return False

    def _deliver(self, snapshot: Snapshot) -> None:
        if self._delivery_mode == SnapshotDeliveryMode.CALLBACK:
            if self.on_snapshot is not None:
                try:
                    self.on_snapshot(snapshot)
                except Exception as exc:
                    self._notify_error(exc)
            return

        try:
            self._message_queue.put_nowait(snapshot)
        except Full:
            try:
                self._message_queue.get_nowait()
                self._message_queue.put_nowait(snapshot)
            except (Empty, Full):
                self._logger.warning("Snapshot queue remained full; dropped snapshot")

    def subscribe(self, subscriptions: Sequence[Any]) -> bool:
        """Validate exchange/token subscriptions and send a Mode 3 request."""
        try:
            normalized = self._normalize_subscriptions(subscriptions)
        except (TypeError, ValueError) as exc:
            self._logger.error(f"Invalid subscription: {exc}")
            return False
        with self._state_lock:
            self._subscriptions = normalized
            connected = self._connected
        if not connected:
            return False
        return self._subscribe_internal(normalized, self.SUBSCRIBE_ACTION)

    def unsubscribe(self, subscriptions: Sequence[Any]) -> bool:
        """Unsubscribe validated exchange/token pairs and update reconnect state."""
        try:
            normalized = self._normalize_subscriptions(subscriptions)
        except (TypeError, ValueError) as exc:
            self._logger.error(f"Invalid subscription: {exc}")
            return False
        if not self.connected:
            return False
        if not self._subscribe_internal(normalized, self.UNSUBSCRIBE_ACTION):
            return False
        remove = set(normalized)
        with self._state_lock:
            self._subscriptions = tuple(
                subscription
                for subscription in self._subscriptions
                if subscription not in remove
            )
        return True

    @staticmethod
    def _normalize_subscriptions(
        subscriptions: Sequence[Any],
    ) -> tuple[MarketSubscription, ...]:
        if isinstance(subscriptions, (str, bytes, bytearray)):
            raise TypeError("subscriptions must contain exchange/token objects")
        normalized = tuple(
            MarketSubscription.from_config(subscription)
            for subscription in subscriptions
        )
        if not normalized:
            raise ValueError("at least one subscription is required")
        return tuple(dict.fromkeys(normalized))

    def _subscribe_internal(
        self,
        subscriptions: Sequence[MarketSubscription],
        action: int = SUBSCRIBE_ACTION,
    ) -> bool:
        grouped: dict[int, list[str]] = {}
        for subscription in subscriptions:
            grouped.setdefault(subscription.exchange_type, []).append(subscription.token)
        payload = {
            "correlationID": self._config.correlation_id,
            "action": action,
            "params": {
                "mode": self.SNAP_QUOTE_MODE,
                "tokenList": [
                    {"exchangeType": exchange_type, "tokens": tokens}
                    for exchange_type, tokens in grouped.items()
                ],
            },
        }
        return self._send_json(payload)

    def _send_json(self, data: Mapping[str, Any]) -> bool:
        return self._send_text(json.dumps(data, separators=(",", ":")))

    def _send_text(self, message: str) -> bool:
        with self._state_lock:
            ws = self._ws
            connected = self._connected
        if ws is None or not connected:
            return False
        try:
            ws.send(message)
            return True
        except Exception as exc:
            self._notify_error(exc)
            return False

    def _start_heartbeat_monitor(self) -> None:
        with self._state_lock:
            current = self._heartbeat_thread
            if current is not None and current.is_alive():
                return
            heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                name="angel-heartbeat",
                daemon=True,
            )
            self._heartbeat_thread = heartbeat_thread
        heartbeat_thread.start()

    def _heartbeat_loop(self) -> None:
        while self.running:
            time.sleep(self._config.heartbeat_interval)
            with self._state_lock:
                connected = self._connected
                has_subscriptions = bool(self._subscriptions)
                last_snapshot = self._last_snapshot
                stale_close_requested = self._stale_close_requested
                ws = self._ws
            if not connected:
                continue
            if has_subscriptions:
                stale_for = time.monotonic() - last_snapshot
                if stale_for > self._config.snapshot_timeout and not stale_close_requested:
                    with self._state_lock:
                        self._stale_close_requested = True
                    self._notify_error(
                        TimeoutError(f"snapshot stream stale for {stale_for:.1f} seconds")
                    )
                    if ws is not None:
                        try:
                            ws.close()
                        except Exception as exc:
                            self._notify_error(exc)
                    continue
            self._send_text("ping")

    def get_snapshot(self, timeout: float = 1.0) -> Optional[Snapshot]:
        """Return the next snapshot only for pull delivery."""
        if self._delivery_mode != SnapshotDeliveryMode.PULL:
            return None
        try:
            return self._message_queue.get(timeout=timeout)
        except Empty:
            return None

    def disconnect(self) -> None:
        """Stop reconnect/heartbeat loops and close the current connection."""
        with self._state_lock:
            self._running = False
            self._connected = False
            ws = self._ws
            ws_thread = self._ws_thread
            heartbeat_thread = self._heartbeat_thread
        if ws is not None:
            try:
                ws.close()
            except Exception as exc:
                self._notify_error(exc)

        current_thread = threading.current_thread()
        if ws_thread is not None and ws_thread is not current_thread and ws_thread.is_alive():
            ws_thread.join(timeout=2.0)
        if (
            heartbeat_thread is not None
            and heartbeat_thread is not current_thread
            and heartbeat_thread.is_alive()
        ):
            heartbeat_thread.join(timeout=min(1.0, self._config.heartbeat_interval))
        self._logger.info("WebSocket disconnected")

    def _clear_message_queue(self) -> None:
        while True:
            try:
                self._message_queue.get_nowait()
            except Empty:
                return

    def _notify_error(self, error: Exception) -> None:
        self._logger.error(f"WebSocket error: {error}")
        if self.on_error is not None:
            self.on_error(error)
