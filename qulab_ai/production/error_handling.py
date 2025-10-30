"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Production-Ready Error Handling
Implements comprehensive error handling with retries, circuit breakers, and fallbacks
"""
import time
import functools
from typing import Callable, Any, Optional, Type, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from qulab_ai.production.logging_config import get_logger

logger = get_logger()


class QuLabException(Exception):
    """Base exception for QuLab AI"""
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "QULAB_ERROR"
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class ParserException(QuLabException):
    """Exception during parsing operations"""
    def __init__(self, message: str, parser_type: str = None, **kwargs):
        super().__init__(message, error_code="PARSER_ERROR", details=kwargs)
        self.parser_type = parser_type


class ValidationException(QuLabException):
    """Exception during validation"""
    def __init__(self, message: str, field: str = None, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", details=kwargs)
        self.field = field


class ResourceException(QuLabException):
    """Exception related to resource limits"""
    def __init__(self, message: str, resource_type: str = None, **kwargs):
        super().__init__(message, error_code="RESOURCE_ERROR", details=kwargs)
        self.resource_type = resource_type


@dataclass
class CircuitBreakerState:
    """Circuit breaker state"""
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    is_open: bool = False
    half_open_attempts: int = 0


class CircuitBreaker:
    """
    Circuit breaker pattern implementation
    Prevents cascading failures by temporarily disabling failing operations
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout_seconds: int = 60,
        half_open_attempts: int = 3
    ):
        """
        Initialize circuit breaker

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout_seconds: Seconds to wait before attempting recovery
            half_open_attempts: Number of test attempts in half-open state
        """
        self.failure_threshold = failure_threshold
        self.timeout = timedelta(seconds=timeout_seconds)
        self.half_open_attempts = half_open_attempts
        self.state = CircuitBreakerState()

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        Call function through circuit breaker

        Args:
            func: Function to call
            *args, **kwargs: Function arguments

        Returns:
            Function result

        Raises:
            ResourceException: If circuit is open
        """
        # Check if circuit is open
        if self.state.is_open:
            if self._should_attempt_reset():
                logger.info("Circuit breaker: Attempting reset", function=func.__name__)
                self.state.half_open_attempts += 1
            else:
                raise ResourceException(
                    "Circuit breaker is open",
                    resource_type="circuit_breaker",
                    function=func.__name__,
                    failure_count=self.state.failure_count
                )

        # Attempt to call function
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise

    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit"""
        if self.state.last_failure_time is None:
            return False

        elapsed = datetime.utcnow() - self.state.last_failure_time
        return elapsed >= self.timeout and self.state.half_open_attempts < self.half_open_attempts

    def _on_success(self):
        """Handle successful call"""
        if self.state.half_open_attempts > 0:
            logger.info("Circuit breaker: Reset successful")

        self.state.failure_count = 0
        self.state.is_open = False
        self.state.half_open_attempts = 0

    def _on_failure(self):
        """Handle failed call"""
        self.state.failure_count += 1
        self.state.last_failure_time = datetime.utcnow()

        if self.state.failure_count >= self.failure_threshold:
            self.state.is_open = True
            logger.error(
                "Circuit breaker: OPEN",
                failure_count=self.state.failure_count,
                threshold=self.failure_threshold
            )


def retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_multiplier: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """
    Retry decorator with exponential backoff

    Args:
        max_attempts: Maximum retry attempts
        delay_seconds: Initial delay between retries
        backoff_multiplier: Multiplier for exponential backoff
        exceptions: Exception types to catch and retry

    Example:
        @retry(max_attempts=3, delay_seconds=1.0)
        def unreliable_operation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 1
            current_delay = delay_seconds

            while attempt <= max_attempts:
                try:
                    logger.debug(
                        f"Attempting {func.__name__}",
                        attempt=attempt,
                        max_attempts=max_attempts
                    )
                    result = func(*args, **kwargs)
                    if attempt > 1:
                        logger.info(
                            f"{func.__name__} succeeded after retry",
                            attempt=attempt
                        )
                    return result

                except exceptions as e:
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts",
                            error=str(e),
                            error_type=type(e).__name__
                        )
                        raise

                    logger.warning(
                        f"{func.__name__} failed, retrying",
                        attempt=attempt,
                        next_retry_in=current_delay,
                        error=str(e)
                    )

                    time.sleep(current_delay)
                    current_delay *= backoff_multiplier
                    attempt += 1

        return wrapper
    return decorator


def safe_execution(
    fallback_value: Any = None,
    log_errors: bool = True,
    raise_on_error: bool = False
):
    """
    Safe execution decorator with fallback value

    Args:
        fallback_value: Value to return on error
        log_errors: Whether to log errors
        raise_on_error: Whether to raise exception after logging

    Example:
        @safe_execution(fallback_value={})
        def risky_operation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    logger.error(
                        f"Error in {func.__name__}",
                        error=str(e),
                        error_type=type(e).__name__,
                        fallback_value=fallback_value
                    )

                if raise_on_error:
                    raise

                return fallback_value

        return wrapper
    return decorator


def timed_execution(log_threshold_ms: float = 100.0):
    """
    Decorator to time function execution and log slow operations

    Args:
        log_threshold_ms: Log warning if execution exceeds this threshold

    Example:
        @timed_execution(log_threshold_ms=50.0)
        def slow_operation():
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                if duration_ms > log_threshold_ms:
                    logger.warning(
                        f"{func.__name__} execution exceeded threshold",
                        duration_ms=duration_ms,
                        threshold_ms=log_threshold_ms
                    )
                else:
                    logger.debug(
                        f"{func.__name__} execution time",
                        duration_ms=duration_ms
                    )

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.error(
                    f"{func.__name__} failed after {duration_ms:.2f}ms",
                    error=str(e)
                )
                raise

        return wrapper
    return decorator


# Example usage
if __name__ == "__main__":
    # Test retry decorator
    @retry(max_attempts=3, delay_seconds=0.5)
    def unreliable_function():
        import random
        if random.random() < 0.7:
            raise Exception("Random failure")
        return "Success!"

    # Test safe execution
    @safe_execution(fallback_value="fallback")
    def risky_function():
        raise ValueError("This will return fallback")

    # Test timed execution
    @timed_execution(log_threshold_ms=100.0)
    def slow_function():
        time.sleep(0.15)
        return "Done"

    print("Testing error handling...")
    try:
        result = unreliable_function()
        print(f"Retry test: {result}")
    except:
        print("Retry test: Failed after all attempts")

    print(f"Safe execution test: {risky_function()}")
    print(f"Timed execution test: {slow_function()}")
