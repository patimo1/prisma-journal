"""Simple TTL cache utilities for database query results."""

import logging
from functools import wraps
from time import time

log = logging.getLogger(__name__)

_cache = {}
_cache_ttl = {}


def cached(ttl_seconds=60):
    """Decorator for caching function results with TTL.

    Args:
        ttl_seconds: Time-to-live in seconds for cached results.

    Returns:
        Decorated function with caching behavior.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = (func.__name__, args, tuple(sorted(kwargs.items())))
            now = time()
            if key in _cache and _cache_ttl.get(key, 0) > now:
                return _cache[key]
            result = func(*args, **kwargs)
            _cache[key] = result
            _cache_ttl[key] = now + ttl_seconds
            return result

        wrapper.invalidate = lambda: _cache.clear()
        return wrapper

    return decorator


def invalidate_cache():
    """Clear all cached results.

    Call this after create/update/delete operations to ensure
    fresh data on next read.
    """
    global _cache, _cache_ttl
    _cache = {}
    _cache_ttl = {}
    log.debug("Cache invalidated")


def get_cache_stats():
    """Return current cache statistics for monitoring."""
    now = time()
    active = sum(1 for ttl in _cache_ttl.values() if ttl > now)
    return {
        "total_entries": len(_cache),
        "active_entries": active,
        "expired_entries": len(_cache) - active,
    }
