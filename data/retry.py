"""
data/retry.py

Small, dependency-free retry decorator for the two network boundaries in
this project that talk to a real external service: yfinance (global_factors.py)
and Kite Connect (kite_fetcher.py). Both are known-but-untested live network
paths (see EOD_JOB.md / PROJECT_NOTES.md) -- transient failures (a dropped
connection, a rate-limit blip, a slow response) shouldn't take down an EOD
run that would otherwise have succeeded on a second try.

Deliberately not using a third-party library (tenacity, backoff, etc.) --
the policy needed here is simple enough (fixed attempt count, exponential
delay, retry on any exception) that a ~30-line decorator is easier to audit
than adding a new dependency for it.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry_with_backoff(
    attempts: int = 3,
    initial_delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[BaseException], ...] = (Exception,),
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator: retry the wrapped call up to `attempts` times on any of
    `exceptions`, sleeping `initial_delay * backoff_factor**n` between
    attempts (so with the defaults: fail, wait 1s, fail, wait 2s, fail ->
    raise -- about 3s of added latency in the worst case, which is
    negligible next to a job that runs once a day).

    The last exception is re-raised unchanged if every attempt fails, so
    callers see the real error (a bad ticker, an expired Kite token, etc.)
    rather than a generic "retries exhausted" wrapper.

    `on_retry(attempt_number, exception, delay)` is called before each
    sleep, if provided -- defaults to a WARNING log line. Tests pass a stub
    here to assert retry behaviour without actually sleeping.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            delay = initial_delay
            last_exc: BaseException | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # noqa: PERF203 - clarity over micro-perf here
                    last_exc = exc
                    if attempt >= attempts:
                        break
                    if on_retry is not None:
                        on_retry(attempt, exc, delay)
                    else:
                        logger.warning(
                            "%s: attempt %d/%d failed (%r) -- retrying in %.1fs",
                            func.__qualname__, attempt, attempts, exc, delay,
                        )
                    time.sleep(delay)
                    delay *= backoff_factor
            assert last_exc is not None  # loop always runs >=1 time
            raise last_exc

        return wrapper

    return decorator
