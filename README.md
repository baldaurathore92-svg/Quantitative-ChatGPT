# Snapshot Quant Engine

A production-grade quantitative trading engine for NSE using Angel One SmartAPI V2 Mode 3 (SnapQuote) with O(1) incremental processing.

## Features

### Core Features (11)
- **Microprice** - Imbalance-weighted mid price calculation
- **Weighted OBI** - Order book imbalance with depth weighting
- **Depth Slope** - Price-quantity relationship across depth
- **Spread** - Bid-ask spread analysis
- **Spread Compression** - Spread narrowing detection
- **Queue Persistence** - Order queue stability measurement
- **Momentum** - Price momentum with time-aware EMA
- **Acceleration** - Rate of change of momentum
- **Refill Proxy** - Liquidity refill detection
- **LTP Confirmation** - Last traded price alignment

### O(1) Buffers
- Ring buffer with incremental statistics
- Rolling mean/variance (Welford's algorithm)
- Time-aware exponential moving average
- Monotonic queue for rolling min/max

### Engine Components
- Composite score calculation with regime-aware weights
- Dynamic threshold based on market volatility
- Market regime detection (Trend/Pullback/Range/Noise)
- State machine for signal generation
- Feature confidence calculation
- Execution model with depth walking

## Performance

- **~0.15ms per snapshot** (~6,500 snapshots/second)
- **O(1) incremental processing** - no full scans
- **Thread-safe** - all shared state protected
- **Memory stable** - periodic cleanup prevents leaks

## Usage

```python
from engine.quant_engine import QuantEngine
from config import EngineConfig
from adapter.angel_v2 import AngelOneWebSocket, SmartAPIConfig

# Initialize engine
config = EngineConfig()
engine = QuantEngine(config)

# Connect to Angel One SmartAPI
api_config = SmartAPIConfig(
    api_key='your_api_key',
    auth_token='your_jwt_token',
    client_code='your_client_code',
    feed_token='your_feed_token'
)

ws = AngelOneWebSocket(api_config)
ws.on_snapshot = engine.process

if ws.connect():
    ws.subscribe(['RELIANCE', 'TCS', 'INFY'])
```

## Project Structure

```
├── adapter/           # Data source adapters
│   ├── angel_v2.py    # Angel One SmartAPI WebSocket
│   └── replay.py      # Backtesting adapter
├── buffers/           # O(1) buffer implementations
├── engine/            # Core engine components
├── features/          # Feature calculators
├── utils/             # Utilities and types
├── config.py          # Configuration management
├── main.py            # Main entry point
└── runner.py          # Convenience runner
```

## Requirements

- Python 3.11+
- websocket-client

## Installation

```bash
pip install websocket-client
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## License

MIT
