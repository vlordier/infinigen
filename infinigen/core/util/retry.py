# Copyright (C) 2024, Princeton University.
# This source code is licensed under the BSD 3-Clause license found in the LICENSE file in the root directory
# of this source tree.

"""
Retry with exponential backoff and jitter.

Provides a decorator and a context-manager-style helper for retrying
operations that may fail transiently (network timeouts, file locks,
SLURM API hiccups, etc.).

Usage::

    from infinigen.core.util.retry import retry

    # As a decorator
    @retry(max_attempts=5, base_delay=1.0, max_delay=60.0)
    def flaky_upload(path):
        ...

    # As a callable wrapper
    result = retry(max_attempts=3)(lambda: requests.get(url))()
"""

from __future__ import annotations

import logging
import random
import time
from functools import wraps

logger = logging.getLogger(__name__)


class MaxRetriesExceeded(Exception):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, attempts: int, last_exception: BaseException):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Failed after {attempts} attempt(s); last error: {last_exception}"
        )


def compute_delay(
    attempt: int,
    base_delay: float,
    max_delay: float,
    jitter: bool = True,
) -> float:
    """Return the sleep duration for the given *attempt* index (0-based).

    Uses full-jitter exponential backoff:
    ``delay = random(0, min(max_delay, base_delay * 2 ** attempt))``

    When *jitter* is ``False`` the delay is deterministic (useful for tests).
    """
    exp_delay = min(max_delay, base_delay * (2 ** attempt))
    if jitter:
        return random.uniform(0, exp_delay)
    return exp_delay


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: bool = True,
    retryable: tuple[type[BaseException], ...] = (Exception,),
    on_retry: None | callable = None,
):
    """Decorator / wrapper factory that retries a callable on failure.

    Parameters
    ----------
    max_attempts:
        Total number of attempts (including the first). Must be >= 1.
    base_delay:
        Initial backoff in seconds.  Must be >= 0.
    max_delay:
        Cap for the exponential growth.  Must be >= *base_delay*.
    jitter:
        Add randomised jitter to prevent thundering-herd effects.
    retryable:
        Tuple of exception types that trigger a retry. Exceptions not
        listed here propagate immediately.
    on_retry:
        Optional callback ``(attempt, exception, delay) -> None`` invoked
        before each sleep. Useful for custom logging or metrics.

    Raises
    ------
    ValueError
        If parameters are out of range.
    MaxRetriesExceeded
        When all attempts fail (wraps the last exception).
    """
    if max_attempts < 1:
        raise ValueError(f"max_attempts must be >= 1, got {max_attempts}")
    if base_delay < 0:
        raise ValueError(f"base_delay must be >= 0, got {base_delay}")
    if max_delay < base_delay:
        raise ValueError(
            f"max_delay ({max_delay}) must be >= base_delay ({base_delay})"
        )

    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: BaseException | None = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except retryable as exc:
                    last_exc = exc
                    if attempt + 1 >= max_attempts:
                        break
                    delay = compute_delay(attempt, base_delay, max_delay, jitter)
                    logger.warning(
                        "Attempt %d/%d for %s failed (%s); retrying in %.2fs",
                        attempt + 1,
                        max_attempts,
                        fn.__qualname__,
                        exc,
                        delay,
                    )
                    if on_retry is not None:
                        on_retry(attempt, exc, delay)
                    time.sleep(delay)

            raise MaxRetriesExceeded(max_attempts, last_exc)

        return wrapper

    return decorator
