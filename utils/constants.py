"""
Constants for Snapshot Quant Engine.

All magic numbers centralized here for maintainability.
"""

# NSE market hours
MARKET_OPEN_HOUR: int = 9
MARKET_OPEN_MINUTE: int = 15
MARKET_CLOSE_HOUR: int = 15
MARKET_CLOSE_MINUTE: int = 30

# Default tick sizes for NSE segments
NSE_CASH_TICK: float = 0.05
NSE_FNO_TICK: float = 0.05

# Price band limits (as percentages)
MAX_DAILY_BAND_PCT: float = 20.0
CIRCUIT_LIMIT_PCT: float = 5.0

# Spread thresholds
MAX_SPREAD_RATIO: float = 0.02  # 2% of price
MIN_LIQUIDITY_QTY: int = 100

# Confidence thresholds
MIN_CONFIDENCE: float = 0.1
MAX_CONFIDENCE: float = 1.0
HIGH_CONFIDENCE: float = 0.8
MEDIUM_CONFIDENCE: float = 0.5
LOW_CONFIDENCE: float = 0.3

# Feature normalization ranges
# These define the expected range of raw values before normalization
MICROPRICE_TICK_RANGE: float = 10.0  # Microprice deviation in ticks
OBI_RANGE: float = 2.0  # OBI is already [-1, 1], allow some amplification
DEPTH_SLOPE_RANGE: float = 5.0  # Slope coefficient range
SPREAD_COMPRESSION_RANGE: float = 1.0  # Already normalized
QUEUE_PERSISTENCE_RANGE: float = 1.0  # Already normalized
MOMENTUM_RANGE: float = 2.0  # Momentum can exceed [-1, 1]
ACCELERATION_RANGE: float = 1.0  # Normalized acceleration
REFILL_PROXY_RANGE: float = 1.0  # Already normalized

# EMA decay constants (in seconds)
FAST_EMA_TAU: float = 5.0
SLOW_EMA_TAU: float = 20.0
MOMENTUM_EMA_TAU: float = 10.0

# Default buffer sizes
DEFAULT_BUFFER_SIZE: int = 100
PRICE_HISTORY_SIZE: int = 50
SPREAD_HISTORY_SIZE: int = 30

# State machine timeouts (in seconds)
WATCH_TIMEOUT: float = 5.0
COOLDOWN_TIME: float = 3.0
MIN_HOLD_TIME: float = 1.0
WARMUP_SAMPLES: int = 10

# Composite thresholds
DEFAULT_THRESHOLD: float = 0.6
MIN_THRESHOLD: float = 0.3
MAX_THRESHOLD: float = 0.9

# Volatility thresholds (annualized)
LOW_VOL_THRESHOLD: float = 0.15  # 15%
HIGH_VOL_THRESHOLD: float = 0.40  # 40%

# Queue stability thresholds
MIN_QUEUE_STABILITY: float = 0.3
HIGH_QUEUE_STABILITY: float = 0.7

# Book quality thresholds
MIN_DEPTH_LEVELS: int = 3
MAX_CROSSED_SPREAD_TICKS: int = 0

# Execution constants
DEFAULT_SLIPPAGE_TICKS: float = 1.0
MAX_DEPTH_WALK: int = 3
MIN_FILL_RATIO: float = 0.5

# Regime thresholds
TREND_MOMENTUM_THRESHOLD: float = 0.5
TREND_OBI_THRESHOLD: float = 0.4
RANGE_SPREAD_THRESHOLD: float = 0.3
NOISE_VOL_THRESHOLD: float = 0.7

# WebSocket constants
HEARTBEAT_INTERVAL: float = 10.0
RECONNECT_DELAY: float = 5.0
MAX_RECONNECT_ATTEMPTS: int = 10
SNAPSHOT_TIMEOUT: float = 30.0

# Logging levels
LOG_DEBUG: str = "DEBUG"
LOG_INFO: str = "INFO"
LOG_WARNING: str = "WARNING"
LOG_ERROR: str = "ERROR"
LOG_CRITICAL: str = "CRITICAL"

# Performance targets (milliseconds)
TARGET_PROCESSING_TIME_MS: float = 1.0  # Sub-millisecond target
MAX_PROCESSING_TIME_MS: float = 5.0  # Maximum acceptable

# Numerical stability
EPSILON: float = 1e-9
MIN_PRICE: float = 0.01
MIN_QUANTITY: int = 1

# Feature weight defaults
DEFAULT_WEIGHT_MICROPRICE: float = 1.0
DEFAULT_WEIGHT_OBI: float = 1.2
DEFAULT_WEIGHT_DEPTH_SLOPE: float = 0.6
DEFAULT_WEIGHT_SPREAD_COMPRESSION: float = 0.4
DEFAULT_WEIGHT_QUEUE_PERSISTENCE: float = 0.5
DEFAULT_WEIGHT_MOMENTUM: float = 0.8
DEFAULT_WEIGHT_ACCELERATION: float = 0.4
DEFAULT_WEIGHT_REFILL: float = 0.3
DEFAULT_WEIGHT_LTP_CONFIRM: float = 0.2

# Regime-specific thresholds
REGIME_TREND_THRESHOLD_MULT: float = 1.0
REGIME_PULLBACK_THRESHOLD_MULT: float = 1.1
REGIME_RANGE_THRESHOLD_MULT: float = 1.2
REGIME_NOISE_THRESHOLD_MULT: float = 1.5

# Regime-specific confidence
REGIME_TREND_CONFIDENCE: float = 1.0
REGIME_PULLBACK_CONFIDENCE: float = 0.85
REGIME_RANGE_CONFIDENCE: float = 0.80
REGIME_NOISE_CONFIDENCE: float = 0.50

# Signal persistence
MIN_CONSECUTIVE_SIGNALS: int = 3

# Data freshness
MAX_SNAPSHOT_AGE_SECONDS: float = 5.0

# Duplicate detection
DUPLICATE_SEQUENCE_TOLERANCE: int = 0

# Price validation
MIN_VALID_PRICE: float = 0.01
MAX_VALID_PRICE: float = 1000000.0  # 10 lakh rupees

# Quantity validation
MIN_VALID_QUANTITY: int = 0
MAX_VALID_QUANTITY: int = 100000000  # 10 crore shares

# Order count validation
MIN_VALID_ORDERS: int = 0
MAX_VALID_ORDERS: int = 10000
