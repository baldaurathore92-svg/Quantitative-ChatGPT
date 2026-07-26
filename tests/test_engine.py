"""
Tests for Snapshot Quant Engine.

Tests all major components:
- Snapshot validation
- Feature calculations
- Composite scoring
- State machine transitions
"""

import unittest
import time

# Import engine components
from config import EngineConfig
from utils.types import Snapshot, PriceLevel, PriceKeyedBook
from adapter.replay import create_mock_snapshot


class TestSnapshotValidation(unittest.TestCase):
    """Test snapshot validation."""
    
    def test_valid_snapshot(self):
        """Test that valid snapshot is accepted."""
        snapshot = create_mock_snapshot(
            symbol='TEST',
            ltp=100.0,
            spread_ticks=2.0,
            depth_levels=5
        )
        
        self.assertTrue(snapshot.is_valid())
        self.assertEqual(snapshot.symbol, 'TEST')
        self.assertIsNotNone(snapshot.best_bid)
        self.assertIsNotNone(snapshot.best_ask)
        self.assertIsNotNone(snapshot.mid_price)
        self.assertIsNotNone(snapshot.spread)
    
    def test_invalid_crossed_book(self):
        """Test that crossed book is rejected."""
        # Create snapshot with crossed book
        bids = tuple([
            PriceLevel(price=100.0, quantity=1000, order_count=10)
        ])
        asks = tuple([
            PriceLevel(price=99.0, quantity=1000, order_count=10)  # Ask < Bid
        ])
        
        snapshot = Snapshot(
            symbol='TEST',
            timestamp=time.time(),
            ltp=100.0,
            bids=bids,
            asks=asks
        )
        
        self.assertFalse(snapshot.is_valid())
    
    def test_missing_depth(self):
        """Test that missing depth is rejected."""
        snapshot = Snapshot(
            symbol='TEST',
            timestamp=time.time(),
            ltp=100.0,
            bids=tuple(),
            asks=tuple()
        )
        
        self.assertFalse(snapshot.is_valid())


class TestPriceKeyedBook(unittest.TestCase):
    """Test price-keyed book operations."""
    
    def test_price_keyed_lookup(self):
        """Test that price-keyed lookup works correctly."""
        snapshot = create_mock_snapshot('TEST', 100.0)
        book = PriceKeyedBook.from_snapshot(snapshot)
        
        # Test bid lookup
        for level in snapshot.bids:
            qty = book.get_bid_qty(level.price)
            self.assertEqual(qty, level.quantity)
        
        # Test ask lookup
        for level in snapshot.asks:
            qty = book.get_ask_qty(level.price)
            self.assertEqual(qty, level.quantity)
    
    def test_missing_price_returns_zero(self):
        """Test that missing price returns 0."""
        snapshot = create_mock_snapshot('TEST', 100.0)
        book = PriceKeyedBook.from_snapshot(snapshot)
        
        # Test missing price
        qty = book.get_bid_qty(50.0)  # Price not in book
        self.assertEqual(qty, 0)


class TestMicroprice(unittest.TestCase):
    """Test microprice calculation."""
    
    def test_microprice_calculation(self):
        """Test basic microprice calculation."""
        from features.microprice import MicropriceCalculator, MicropriceConfig
        
        config = MicropriceConfig(tick_size=0.05)
        calc = MicropriceCalculator(config)
        
        snapshot = create_mock_snapshot('TEST', 100.0)
        result = calc.calculate(snapshot)
        
        self.assertTrue(result.valid)
        self.assertGreaterEqual(result.value, -1.0)
        self.assertLessEqual(result.value, 1.0)
    
    def test_microprice_balanced_book(self):
        """Test microprice with balanced book."""
        from features.microprice import MicropriceCalculator, MicropriceConfig
        
        config = MicropriceConfig(tick_size=0.05)
        calc = MicropriceCalculator(config)
        
        # Create balanced book
        bids = tuple([PriceLevel(price=99.0, quantity=1000, order_count=10)])
        asks = tuple([PriceLevel(price=101.0, quantity=1000, order_count=10)])
        
        snapshot = Snapshot(
            symbol='TEST',
            timestamp=time.time(),
            ltp=100.0,
            bids=bids,
            asks=asks
        )
        
        result = calc.calculate(snapshot)
        self.assertTrue(result.valid)
        
        # With balanced quantities, microprice should be near mid
        mid = snapshot.mid_price
        self.assertIsNotNone(mid)
        self.assertAlmostEqual(result.raw_value, mid, delta=0.5)


class TestWeightedOBI(unittest.TestCase):
    """Test weighted OBI calculation."""
    
    def test_obi_calculation(self):
        """Test basic OBI calculation."""
        from features.weighted_obi import WeightedOBICalculator, OBIConfig
        
        config = OBIConfig(tick_size=0.05, lambda_decay=3.0)
        calc = WeightedOBICalculator(config)
        
        snapshot = create_mock_snapshot('TEST', 100.0)
        result = calc.calculate(snapshot)
        
        self.assertTrue(result.valid)
        self.assertGreaterEqual(result.value, -1.0)
        self.assertLessEqual(result.value, 1.0)
    
    def test_obi_bid_pressure(self):
        """Test OBI with more bid pressure."""
        from features.weighted_obi import WeightedOBICalculator, OBIConfig
        
        config = OBIConfig(tick_size=0.05)
        calc = WeightedOBICalculator(config)
        
        # Create book with more bid quantity
        bids = tuple([
            PriceLevel(price=99.0, quantity=5000, order_count=50),
            PriceLevel(price=98.5, quantity=3000, order_count=30)
        ])
        asks = tuple([
            PriceLevel(price=101.0, quantity=1000, order_count=10),
            PriceLevel(price=101.5, quantity=500, order_count=5)
        ])
        
        snapshot = Snapshot(
            symbol='TEST',
            timestamp=time.time(),
            ltp=100.0,
            bids=bids,
            asks=asks
        )
        
        result = calc.calculate(snapshot)
        self.assertTrue(result.valid)
        self.assertGreater(result.value, 0)  # Should be bullish


class TestMomentum(unittest.TestCase):
    """Test momentum calculation."""
    
    def test_momentum_calculation(self):
        """Test momentum with time-aware EMA."""
        from features.momentum import MomentumCalculator, MomentumConfig
        
        config = MomentumConfig(tick_size=0.05, tau=5.0)
        calc = MomentumCalculator(config)
        
        # Process multiple snapshots
        for i in range(10):
            snapshot = create_mock_snapshot('TEST', 100.0 + i * 0.1)
            result = calc.calculate(snapshot)
            
            self.assertTrue(result.valid)
            self.assertGreaterEqual(result.value, -1.0)
            self.assertLessEqual(result.value, 1.0)


class TestCompositeScore(unittest.TestCase):
    """Test composite scoring."""
    
    def test_composite_calculation(self):
        """Test composite score calculation."""
        from engine.composite import CompositeCalculator, CompositeConfig
        from utils.types import FeatureResult
        
        config = CompositeConfig()
        calc = CompositeCalculator(config)
        
        features = {
            'microprice': FeatureResult(value=0.3, confidence=0.8, valid=True, raw_value=0.03, name='microprice'),
            'weighted_obi': FeatureResult(value=0.5, confidence=0.9, valid=True, raw_value=0.5, name='weighted_obi'),
            'momentum': FeatureResult(value=0.2, confidence=0.7, valid=True, raw_value=0.02, name='momentum'),
            'depth_slope': FeatureResult(value=0.0, confidence=0.5, valid=True, raw_value=0.0, name='depth_slope'),
            'spread_compression': FeatureResult(value=0.0, confidence=0.5, valid=True, raw_value=0.0, name='spread_compression'),
            'queue_persistence': FeatureResult(value=0.0, confidence=0.5, valid=True, raw_value=0.0, name='queue_persistence'),
            'acceleration': FeatureResult(value=0.0, confidence=0.5, valid=True, raw_value=0.0, name='acceleration'),
            'refill_proxy': FeatureResult(value=0.0, confidence=0.5, valid=True, raw_value=0.0, name='refill_proxy'),
            'ltp_confirmation': FeatureResult(value=0.0, confidence=0.5, valid=True, raw_value=0.0, name='ltp_confirmation'),
        }
        
        from utils.types import Regime
        score = calc.calculate(features, Regime.RANGE)
        
        self.assertIsNotNone(score)
        self.assertGreaterEqual(score.value, -1.0)
        self.assertLessEqual(score.value, 1.0)


class TestStateMachine(unittest.TestCase):
    """Test state machine."""
    
    def test_warmup_to_neutral(self):
        """Test transition from WARMUP to NEUTRAL."""
        from engine.state_machine import TradingStateMachine, StateMachineConfig
        
        config = StateMachineConfig(warmup_samples=5)
        sm = TradingStateMachine(config)
        
        # Initial state
        from utils.types import State
        self.assertEqual(sm.state, State.WARMUP)
        
        # Process samples
        for i in range(10):
            sm.update(0.0, 0.5, 'NOISE', time.time())
        
        self.assertEqual(sm.state, State.NEUTRAL)
    
    def test_neutral_to_watch(self):
        """Test transition from NEUTRAL to WATCH_LONG."""
        from engine.state_machine import TradingStateMachine, StateMachineConfig
        
        config = StateMachineConfig(warmup_samples=3, signal_persistence=2)
        sm = TradingStateMachine(config)
        
        # Get to NEUTRAL
        for i in range(5):
            sm.update(0.0, 0.5, 'NOISE', time.time())
        
        from utils.types import State
        self.assertEqual(sm.state, State.NEUTRAL)
        
        # Generate bullish signals
        for i in range(3):
            sm.update(0.6, 0.7, 'TREND', time.time())
        
        # Should be in WATCH_LONG or higher
        self.assertIn(sm.state, [State.WATCH_LONG, State.LONG])


class TestFullEngine(unittest.TestCase):
    """Test complete engine integration."""
    
    def test_engine_processing(self):
        """Test full engine processing."""
        from engine.quant_engine import QuantEngine
        
        config = EngineConfig()
        engine = QuantEngine(config)
        
        # Process multiple snapshots
        for i in range(20):
            snapshot = create_mock_snapshot('TEST', 100.0 + i * 0.05)
            composite = engine.process(snapshot)
            
            self.assertIsNotNone(composite)
            self.assertGreaterEqual(composite.value, -1.0)
            self.assertLessEqual(composite.value, 1.0)
        
        # Check stats
        stats = engine.get_stats()
        self.assertEqual(stats['snapshot_count'], 20)
    
    def test_engine_reset(self):
        """Test engine reset."""
        from engine.quant_engine import QuantEngine
        
        config = EngineConfig()
        engine = QuantEngine(config)
        
        # Process snapshots
        for i in range(10):
            snapshot = create_mock_snapshot('TEST', 100.0)
            engine.process(snapshot)
        
        # Reset
        engine.reset()
        
        # Stats should be reset
        stats = engine.get_stats()
        self.assertEqual(stats['snapshot_count'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
