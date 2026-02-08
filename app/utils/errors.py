"""Centralized error handling utilities for the journal app.

Provides:
- Custom exception classes for different error categories
- Standardized API error response helpers
- Retry logic with exponential backoff
- Structured logging configuration
- Input validation utilities
"""

import functools
import logging
import time
import traceback
from enum import Enum
from typing import Any, Callable, Optional, TypeVar

from flask import jsonify, request

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error Categories
# ---------------------------------------------------------------------------


class ErrorCategory(Enum):
    """Categories for classifying errors."""

    SERVICE_UNAVAILABLE = "service_unavailable"
    VALIDATION = "validation"
    DATABASE = "database"
    AI_SERVICE = "ai_service"
    VOICE = "voice"
    IMAGE = "image"
    NETWORK = "network"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    NOT_FOUND = "not_found"
    INTERNAL = "internal"


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------


class JournalAppError(Exception):
    """Base exception for all journal app errors."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        details: Optional[dict] = None,
        recoverable: bool = True,
        user_message: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.details = details or {}
        self.recoverable = recoverable
        # User-friendly message (may differ from technical message)
        self.user_message = user_message or message

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON response."""
        return {
            "error": self.user_message,
            "category": self.category.value,
            "recoverable": self.recoverable,
            "details": self.details,
        }


class ServiceUnavailableError(JournalAppError):
    """Raised when an external service is not available."""

    def __init__(
        self,
        service_name: str,
        message: Optional[str] = None,
        setup_instructions: Optional[str] = None,
    ):
        user_msg = message or f"{service_name} is currently unavailable"
        details = {"service": service_name}
        if setup_instructions:
            details["setup_instructions"] = setup_instructions
        super().__init__(
            message=f"{service_name} unavailable: {message}",
            category=ErrorCategory.SERVICE_UNAVAILABLE,
            details=details,
            recoverable=True,
            user_message=user_msg,
        )
        self.service_name = service_name
        self.setup_instructions = setup_instructions


class ValidationError(JournalAppError):
    """Raised when input validation fails."""

    def __init__(
        self,
        field: str,
        message: str,
        value: Any = None,
    ):
        details = {"field": field}
        if value is not None:
            # Don't include sensitive values
            details["value_type"] = type(value).__name__
        super().__init__(
            message=f"Validation error on '{field}': {message}",
            category=ErrorCategory.VALIDATION,
            details=details,
            recoverable=True,
            user_message=message,
        )
        self.field = field


class DatabaseError(JournalAppError):
    """Raised when database operations fail."""

    def __init__(
        self,
        operation: str,
        message: str,
        original_error: Optional[Exception] = None,
    ):
        details = {"operation": operation}
        if original_error:
            details["original_error"] = str(original_error)
        super().__init__(
            message=f"Database error during {operation}: {message}",
            category=ErrorCategory.DATABASE,
            details=details,
            recoverable=False,
            user_message="A database error occurred. Your data may not have been saved.",
        )
        self.operation = operation
        self.original_error = original_error


class AIServiceError(JournalAppError):
    """Raised when AI service calls fail."""

    def __init__(
        self,
        service: str,
        message: str,
        timeout: bool = False,
        retry_possible: bool = True,
    ):
        category = ErrorCategory.TIMEOUT if timeout else ErrorCategory.AI_SERVICE
        details = {"service": service, "timeout": timeout}
        super().__init__(
            message=f"AI service error ({service}): {message}",
            category=category,
            details=details,
            recoverable=retry_possible,
            user_message=message,
        )
        self.service = service
        self.timeout = timeout


class VoiceError(JournalAppError):
    """Raised when voice processing fails."""

    def __init__(
        self,
        message: str,
        error_type: str = "transcription",
        guidance: Optional[str] = None,
    ):
        details = {"error_type": error_type}
        if guidance:
            details["guidance"] = guidance
        super().__init__(
            message=f"Voice error ({error_type}): {message}",
            category=ErrorCategory.VOICE,
            details=details,
            recoverable=True,
            user_message=message,
        )
        self.guidance = guidance


class ImageGenerationError(JournalAppError):
    """Raised when image generation fails."""

    def __init__(
        self,
        message: str,
        fallback_available: bool = True,
    ):
        super().__init__(
            message=f"Image generation error: {message}",
            category=ErrorCategory.IMAGE,
            details={"fallback_available": fallback_available},
            recoverable=fallback_available,
            user_message=message,
        )
        self.fallback_available = fallback_available


class TimeoutError(JournalAppError):
    """Raised when an operation times out."""

    def __init__(
        self,
        operation: str,
        timeout_seconds: int,
        allow_retry: bool = True,
    ):
        super().__init__(
            message=f"Operation '{operation}' timed out after {timeout_seconds}s",
            category=ErrorCategory.TIMEOUT,
            details={"operation": operation, "timeout_seconds": timeout_seconds},
            recoverable=allow_retry,
            user_message=f"The operation took too long and was cancelled. You can try again.",
        )
        self.operation = operation
        self.timeout_seconds = timeout_seconds


# ---------------------------------------------------------------------------
# API Response Helpers
# ---------------------------------------------------------------------------


def error_response(
    message: str,
    status_code: int = 400,
    category: Optional[ErrorCategory] = None,
    details: Optional[dict] = None,
    recoverable: bool = True,
):
    """Create a standardized JSON error response."""
    response = {
        "error": message,
        "recoverable": recoverable,
    }
    if category:
        response["category"] = category.value
    if details:
        response["details"] = details

    # Log the error
    log.warning(
        "API error response [%d]: %s | Path: %s | Details: %s",
        status_code,
        message,
        request.path if request else "unknown",
        details,
    )

    return jsonify(response), status_code


def success_response(
    data: Any = None,
    message: Optional[str] = None,
    status_code: int = 200,
):
    """Create a standardized JSON success response."""
    response = {"success": True}
    if message:
        response["message"] = message
    if data is not None:
        response["data"] = data
    return jsonify(response), status_code


# ---------------------------------------------------------------------------
# Retry Logic
# ---------------------------------------------------------------------------

T = TypeVar("T")


def retry_with_backoff(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    retryable_exceptions: tuple = (Exception,),
    on_retry: Optional[Callable[[Exception, int], None]] = None,
):
    """Decorator for retry logic with exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff calculation
        retryable_exceptions: Tuple of exception types that trigger retry
        on_retry: Optional callback called on each retry with (exception, attempt)
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e

                    if attempt >= max_retries:
                        log.error(
                            "All %d retry attempts failed for %s: %s",
                            max_retries + 1,
                            func.__name__,
                            str(e),
                        )
                        raise

                    # Calculate delay with exponential backoff and jitter
                    delay = min(
                        base_delay * (exponential_base**attempt),
                        max_delay,
                    )
                    # Add jitter (0-25% of delay)
                    import random

                    delay = delay * (1 + random.random() * 0.25)

                    log.warning(
                        "Attempt %d/%d for %s failed: %s. Retrying in %.1fs",
                        attempt + 1,
                        max_retries + 1,
                        func.__name__,
                        str(e),
                        delay,
                    )

                    if on_retry:
                        on_retry(e, attempt)

                    time.sleep(delay)

            # Should not reach here, but just in case
            raise last_exception

        return wrapper

    return decorator


# ---------------------------------------------------------------------------
# Input Validation Utilities
# ---------------------------------------------------------------------------


def validate_entry_content(content: str) -> tuple[bool, Optional[str]]:
    """Validate journal entry content.

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not content:
        return False, "Entry content cannot be empty"

    if not content.strip():
        return False, "Entry content cannot be only whitespace"

    # Import here to avoid circular import
    from config import Config

    max_length = Config.MAX_ENTRY_LENGTH

    if len(content) > max_length:
        return False, f"Entry exceeds maximum length of {max_length:,} characters"

    # Check for extremely long entries that might cause issues
    word_count = len(content.split())
    if word_count > 50000:
        return False, (
            f"Entry has {word_count:,} words which may cause performance issues. "
            "Consider splitting into multiple entries."
        )

    return True, None


def validate_uuid(value: str, field_name: str = "id") -> tuple[bool, Optional[str]]:
    """Validate that a string is a valid UUID format."""
    import re

    uuid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
        re.IGNORECASE,
    )

    if not value:
        return False, f"{field_name} is required"

    if not uuid_pattern.match(value):
        return False, f"{field_name} must be a valid UUID"

    return True, None


def validate_audio_file(
    file_data: bytes, filename: str, max_size_mb: int = 50, max_duration_minutes: int = 10
) -> tuple[bool, Optional[str], Optional[dict]]:
    """Validate an audio file for transcription.

    Returns:
        Tuple of (is_valid, error_message, metadata)
    """
    # Check file size
    size_mb = len(file_data) / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"Audio file too large ({size_mb:.1f}MB). Maximum is {max_size_mb}MB.", None

    # Check file extension
    allowed_extensions = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in allowed_extensions:
        return (
            False,
            f"Unsupported audio format '{ext}'. Supported: {', '.join(sorted(allowed_extensions))}",
            None,
        )

    metadata = {
        "size_mb": round(size_mb, 2),
        "extension": ext,
    }

    return True, None, metadata


def sanitize_string(value: str, max_length: int = 1000, strip: bool = True) -> str:
    """Sanitize a string input."""
    if not isinstance(value, str):
        value = str(value)

    if strip:
        value = value.strip()

    if len(value) > max_length:
        value = value[:max_length]

    return value


# ---------------------------------------------------------------------------
# Logging Utilities
# ---------------------------------------------------------------------------


def log_request_error(
    error: Exception,
    context: Optional[dict] = None,
    include_traceback: bool = True,
):
    """Log an error with request context."""
    error_info = {
        "error_type": type(error).__name__,
        "error_message": str(error),
        "path": request.path if request else "unknown",
        "method": request.method if request else "unknown",
    }

    if context:
        error_info.update(context)

    if include_traceback:
        error_info["traceback"] = traceback.format_exc()

    log.error("Request error: %s", error_info)


def log_api_call(
    service: str,
    operation: str,
    success: bool,
    duration_ms: Optional[float] = None,
    details: Optional[dict] = None,
):
    """Log an API call to an external service."""
    log_data = {
        "service": service,
        "operation": operation,
        "success": success,
    }

    if duration_ms is not None:
        log_data["duration_ms"] = round(duration_ms, 2)

    if details:
        log_data.update(details)

    if success:
        log.info("API call: %s", log_data)
    else:
        log.warning("API call failed: %s", log_data)


# ---------------------------------------------------------------------------
# Progress Tracking
# ---------------------------------------------------------------------------


class ProgressTracker:
    """Track progress of long-running operations."""

    def __init__(self, operation_id: str, total_steps: int = 100):
        self.operation_id = operation_id
        self.total_steps = total_steps
        self.current_step = 0
        self.status = "pending"
        self.message = ""
        self.started_at = time.time()
        self.cancelled = False

    def update(self, step: int, message: str = ""):
        """Update progress."""
        self.current_step = min(step, self.total_steps)
        self.status = "in_progress"
        self.message = message

    def complete(self, message: str = "Complete"):
        """Mark operation as complete."""
        self.current_step = self.total_steps
        self.status = "completed"
        self.message = message

    def fail(self, message: str):
        """Mark operation as failed."""
        self.status = "failed"
        self.message = message

    def cancel(self):
        """Cancel the operation."""
        self.cancelled = True
        self.status = "cancelled"
        self.message = "Operation cancelled by user"

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.started_at

    @property
    def progress_percent(self) -> int:
        """Get progress as percentage."""
        if self.total_steps == 0:
            return 0
        return int((self.current_step / self.total_steps) * 100)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON response."""
        return {
            "operation_id": self.operation_id,
            "progress": self.progress_percent,
            "status": self.status,
            "message": self.message,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "cancelled": self.cancelled,
        }


# Store for tracking in-progress operations
_progress_trackers: dict[str, ProgressTracker] = {}


def create_progress_tracker(operation_id: str, total_steps: int = 100) -> ProgressTracker:
    """Create and store a new progress tracker."""
    tracker = ProgressTracker(operation_id, total_steps)
    _progress_trackers[operation_id] = tracker
    return tracker


def get_progress_tracker(operation_id: str) -> Optional[ProgressTracker]:
    """Get an existing progress tracker."""
    return _progress_trackers.get(operation_id)


def remove_progress_tracker(operation_id: str):
    """Remove a progress tracker."""
    _progress_trackers.pop(operation_id, None)
