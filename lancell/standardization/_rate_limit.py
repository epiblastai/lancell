"""Shared rate-limiting and retry logic for API-calling resolvers.

Provides a token-bucket rate limiter per endpoint and a ``@rate_limited`` decorator
that automatically sleeps to stay within limits and retries on 429/503.
"""

import functools
import threading
import time
from collections.abc import Callable
from typing import Any

import requests

# Default per-endpoint rate limits (requests per second)
DEFAULT_LIMITS: dict[str, float] = {
    "pubchem": 5,
    "ensembl": 15,
    "uniprot": 10,
    "hgnc": 10,
    "mygene": 10,
    "ols4": 10,
    "cellosaurus": 5,
    "chembl": 10,
    "ncbi": 3,
}


class TokenBucket:
    """Thread-safe token-bucket rate limiter."""

    def __init__(self, rate: float):
        self.rate = rate  # tokens per second
        self.max_tokens = rate  # burst size = 1 second worth
        self._tokens = float(rate)
        self._last_refill = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Block until a token is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self.max_tokens, self._tokens + elapsed * self.rate)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
            # Sleep a short interval before retrying
            time.sleep(1.0 / self.rate)


_buckets: dict[str, TokenBucket] = {}
_buckets_lock = threading.Lock()


def _get_bucket(endpoint: str, max_per_second: float | None = None) -> TokenBucket:
    """Get or create the token bucket for *endpoint*."""
    with _buckets_lock:
        if endpoint not in _buckets:
            rate = max_per_second or DEFAULT_LIMITS.get(endpoint, 10)
            _buckets[endpoint] = TokenBucket(rate)
        return _buckets[endpoint]


def rate_limited(
    endpoint: str,
    max_per_second: float | None = None,
    max_retries: int = 3,
    backoff_base: float = 1.0,
) -> Callable:
    """Decorator that applies rate limiting and retries with exponential backoff.

    Retries on ``requests.HTTPError`` with status 429 or 503.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            bucket = _get_bucket(endpoint, max_per_second)
            last_exc: Exception | None = None
            for attempt in range(max_retries + 1):
                bucket.acquire()
                try:
                    return fn(*args, **kwargs)
                except requests.HTTPError as exc:
                    status = exc.response.status_code if exc.response is not None else 0
                    if status in (429, 503) and attempt < max_retries:
                        last_exc = exc
                        time.sleep(backoff_base * (2**attempt))
                        continue
                    raise
                except Exception:
                    raise
            # Should not reach here, but just in case
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator
