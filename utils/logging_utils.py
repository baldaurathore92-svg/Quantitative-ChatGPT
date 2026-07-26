"""
Logging utilities for Snapshot Quant Engine.

Structured logging with performance tracking.
"""

import logging
import time
from typing import Optional, Dict, Any
from contextlib import contextmanager
from functools import wraps
import json


class StructuredLogger:
    """
    Structured logger for quant engine.
    
    Provides:
    - JSON-structured logs for machine parsing
    - Performance timing
    - Context enrichment
    """
    
    def __init__(self, name: str, level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        self._context: Dict[str, Any] = {}
    
    def set_context(self, **kwargs: Any) -> None:
        """Set persistent context for all log messages."""
        self._context.update(kwargs)
    
    def clear_context(self) -> None:
        """Clear persistent context."""
        self._context.clear()
    
    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        """Internal logging with context."""
        log_data = {
            'message': message,
            'level': level,
            **self._context,
            **kwargs
        }
        
        log_method = getattr(self.logger, level.lower())
        log_method(json.dumps(log_data))
    
    def debug(self, message: str, **kwargs: Any) -> None:
        self._log('DEBUG', message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        self._log('INFO', message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        self._log('WARNING', message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        self._log('ERROR', message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        self._log('CRITICAL', message, **kwargs)
    
    @contextmanager
    def timer(self, operation: str, **kwargs: Any):
        """
        Context manager for timing operations.
        
        Usage:
            with logger.timer('feature_calculation', feature='microprice'):
                # ... code ...
        """
        start_time = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self._log('DEBUG', f'{operation}_complete', 
                     operation=operation, 
                     elapsed_ms=elapsed_ms, 
                     **kwargs)
    
    def log_snapshot(self, symbol: str, ltp: float, spread_ticks: float, 
                     processing_time_ms: float) -> None:
        """Log snapshot processing."""
        self.debug('snapshot_processed', 
                   symbol=symbol, 
                   ltp=ltp, 
                   spread_ticks=spread_ticks,
                   processing_time_ms=processing_time_ms)
    
    def log_feature(self, feature_name: str, value: float, confidence: float, 
                    valid: bool) -> None:
        """Log feature calculation."""
        self.debug('feature_calculated',
                   feature=feature_name,
                   value=round(value, 4),
                   confidence=round(confidence, 4),
                   valid=valid)
    
    def log_composite(self, value: float, confidence: float, regime: str) -> None:
        """Log composite score."""
        self.info('composite_score',
                  value=round(value, 4),
                  confidence=round(confidence, 4),
                  regime=regime)
    
    def log_state_transition(self, from_state: str, to_state: str, 
                              trigger: str, reason: str) -> None:
        """Log state machine transition."""
        self.info('state_transition',
                  from_state=from_state,
                  to_state=to_state,
                  trigger=trigger,
                  reason=reason)
    
    def log_signal(self, symbol: str, signal_type: str, target_price: float,
                   composite: float, confidence: float) -> None:
        """Log execution signal."""
        self.info('execution_signal',
                  symbol=symbol,
                  signal_type=signal_type,
                  target_price=target_price,
                  composite=round(composite, 4),
                  confidence=round(confidence, 4))
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]) -> None:
        """Log error with full context."""
        self.error('exception_occurred',
                   error_type=type(error).__name__,
                   error_message=str(error),
                   **context)


def setup_logging(config: Any) -> StructuredLogger:
    """
    Setup logging from configuration.
    
    Args:
        config: Configuration object with logging settings
    
    Returns:
        Configured StructuredLogger
    """
    from logging.handlers import RotatingFileHandler
    
    # Create root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, config.logging.level))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(config.logging.format)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if config.logging.file_path:
        file_handler = RotatingFileHandler(
            config.logging.file_path,
            maxBytes=config.logging.max_file_size,
            backupCount=config.logging.backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    return StructuredLogger('snapshot_quant', config.logging.level)


def log_performance(func):
    """
    Decorator to log function performance.
    
    Usage:
        @log_performance
        def calculate_feature(...):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = StructuredLogger(func.__module__)
        start_time = time.perf_counter()
        
        try:
            result = func(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.debug(f'{func.__name__}_complete', elapsed_ms=elapsed_ms)
            return result
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            logger.error(f'{func.__name__}_failed', 
                        error=str(e), 
                        elapsed_ms=elapsed_ms)
            raise
    
    return wrapper


class PerformanceTracker:
    """
    Track performance metrics across snapshots.
    
    Provides rolling statistics on processing times.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.processing_times: list = []
        self.total_snapshots = 0
        self.max_time = 0.0
        self.min_time = float('inf')
    
    def record(self, processing_time_ms: float) -> None:
        """Record processing time."""
        self.processing_times.append(processing_time_ms)
        if len(self.processing_times) > self.window_size:
            self.processing_times.pop(0)
        
        self.total_snapshots += 1
        self.max_time = max(self.max_time, processing_time_ms)
        self.min_time = min(self.min_time, processing_time_ms)
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics."""
        if not self.processing_times:
            return {
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'p99': 0.0,
                'total': 0
            }
        
        n = len(self.processing_times)
        mean = sum(self.processing_times) / n
        variance = sum((t - mean) ** 2 for t in self.processing_times) / n
        std = variance ** 0.5
        
        # Calculate p99
        sorted_times = sorted(self.processing_times)
        p99_idx = int(n * 0.99)
        p99 = sorted_times[p99_idx]
        
        return {
            'mean': mean,
            'std': std,
            'min': self.min_time if self.min_time != float('inf') else 0.0,
            'max': self.max_time,
            'p99': p99,
            'total': self.total_snapshots
        }
