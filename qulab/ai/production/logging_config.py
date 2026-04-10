"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Production-Ready Logging Configuration
Implements centralized logging with structured JSON output, rotation, and monitoring
"""
import logging
import logging.handlers
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

class StructuredJsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging
    Outputs logs in JSON format for easy parsing and analysis
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields
        if hasattr(record, "extra_fields"):
            log_data.update(record.extra_fields)

        return json.dumps(log_data)


class ProductionLogger:
    """
    Production-ready logger with structured output and rotation
    """

    def __init__(
        self,
        name: str = "qulab_ai",
        log_dir: str = "/Users/noone/QuLabInfinite/logs",
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 10,
        console_output: bool = True,
    ):
        """
        Initialize production logger

        Args:
            name: Logger name
            log_dir: Directory for log files
            max_bytes: Max size per log file before rotation
            backup_count: Number of backup files to keep
            console_output: Whether to also output to console
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers.clear()

        # File handler with rotation
        log_file = self.log_dir / f"{name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(StructuredJsonFormatter())
        self.logger.addHandler(file_handler)

        # Console handler (optional)
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_formatter = logging.Formatter(
                '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(logging.DEBUG, message, kwargs)

    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(logging.INFO, message, kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(logging.WARNING, message, kwargs)

    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log(logging.ERROR, message, kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log(logging.CRITICAL, message, kwargs)

    def _log(self, level: int, message: str, extra_fields: Dict[str, Any]):
        """Internal logging with extra fields"""
        self.logger.log(level, message, extra={"extra_fields": extra_fields})

    def log_operation(
        self,
        operation: str,
        status: str,
        duration_ms: float = None,
        **kwargs
    ):
        """
        Log an operation with structured metadata

        Args:
            operation: Operation name (e.g., "parse_smiles", "convert_units")
            status: Status (success, error, warning)
            duration_ms: Operation duration in milliseconds
            **kwargs: Additional metadata
        """
        data = {
            "operation": operation,
            "status": status,
        }

        if duration_ms is not None:
            data["duration_ms"] = duration_ms

        data.update(kwargs)

        if status == "success":
            self.info(f"Operation {operation} completed", **data)
        elif status == "error":
            self.error(f"Operation {operation} failed", **data)
        else:
            self.warning(f"Operation {operation} completed with warnings", **data)


# Global logger instance
_logger = None

def get_logger(name: str = "qulab_ai") -> ProductionLogger:
    """Get or create global logger instance"""
    global _logger
    if _logger is None:
        _logger = ProductionLogger(name=name)
    return _logger


# Example usage
if __name__ == "__main__":
    logger = get_logger()

    # Test different log levels
    logger.debug("Debug message", user_id=123, action="test")
    logger.info("Info message", component="parser", status="running")
    logger.warning("Warning message", resource="memory", usage="85%")
    logger.error("Error message", error_code="E001", details="Sample error")

    # Log an operation
    logger.log_operation(
        operation="parse_smiles",
        status="success",
        duration_ms=25.3,
        input_smiles="CCO",
        n_atoms=3
    )

    print(f"\nLogs written to: {logger.log_dir / 'qulab_ai.log'}")
