"""Simple in-memory rate limiting for API endpoints."""

import logging
from collections import defaultdict
from functools import wraps
from time import time

from flask import jsonify, request

log = logging.getLogger(__name__)

_requests = defaultdict(list)


def rate_limit(max_requests=60, window_seconds=60):
    """Decorator for rate limiting API endpoints.

    Uses in-memory storage - resets on app restart.
    Rate limits are per-client IP address.

    Args:
        max_requests: Maximum number of requests allowed in the window
        window_seconds: Time window in seconds

    Returns:
        Decorated function that returns 429 if rate limit exceeded
    """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            client_ip = request.remote_addr or "unknown"
            endpoint = request.endpoint or f.__name__
            key = f"{client_ip}:{endpoint}"
            now = time()

            # Clean expired entries
            _requests[key] = [t for t in _requests[key] if t > now - window_seconds]

            if len(_requests[key]) >= max_requests:
                log.warning("Rate limit exceeded for %s on %s", client_ip, endpoint)
                return jsonify({
                    "error": "Rate limit exceeded",
                    "message": f"Too many requests. Please wait before trying again.",
                    "retry_after": window_seconds,
                }), 429

            _requests[key].append(now)
            return f(*args, **kwargs)

        return wrapper

    return decorator


def get_rate_limit_stats():
    """Return current rate limit statistics for monitoring."""
    now = time()
    stats = {}
    for key, timestamps in _requests.items():
        active = [t for t in timestamps if t > now - 60]
        if active:
            stats[key] = len(active)
    return stats


def clear_rate_limits():
    """Clear all rate limit data. Useful for testing."""
    global _requests
    _requests = defaultdict(list)
