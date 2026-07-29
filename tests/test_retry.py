"""
tests/test_retry.py

Tests data.retry.retry_with_backoff() in isolation. Uses the on_retry
callback hook (rather than mocking time.sleep) to both avoid slowing the
test suite down with real sleeps and to assert exactly how many times /
with what backoff the decorator retried.
"""

from __future__ import annotations

import pytest

from data.retry import retry_with_backoff


def test_succeeds_on_first_try_without_retrying():
    calls = []

    @retry_with_backoff(attempts=3, initial_delay=0, on_retry=lambda *a: calls.append(a))
    def always_ok():
        return "fine"

    assert always_ok() == "fine"
    assert calls == []  # never needed to retry


def test_succeeds_after_transient_failures():
    attempts_made = {"n": 0}

    @retry_with_backoff(attempts=3, initial_delay=0, on_retry=lambda *a: None)
    def fails_twice_then_ok():
        attempts_made["n"] += 1
        if attempts_made["n"] < 3:
            raise ConnectionError("transient")
        return "recovered"

    assert fails_twice_then_ok() == "recovered"
    assert attempts_made["n"] == 3


def test_raises_original_exception_after_exhausting_attempts():
    @retry_with_backoff(attempts=3, initial_delay=0, on_retry=lambda *a: None)
    def always_fails():
        raise ValueError("still broken")

    with pytest.raises(ValueError, match="still broken"):
        always_fails()


def test_retries_exactly_attempts_minus_one_times():
    retry_calls = []

    @retry_with_backoff(attempts=4, initial_delay=0, on_retry=lambda n, exc, delay: retry_calls.append(n))
    def always_fails():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        always_fails()

    # 4 attempts total -> retried (i.e. on_retry called) after attempts 1, 2, 3
    # (not after the 4th, since that's the final failure that gets raised).
    assert retry_calls == [1, 2, 3]


def test_backoff_delay_doubles_each_retry():
    delays = []

    @retry_with_backoff(
        attempts=4, initial_delay=1.0, backoff_factor=2.0,
        on_retry=lambda n, exc, delay: delays.append(delay),
    )
    def always_fails():
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        always_fails()

    assert delays == [1.0, 2.0, 4.0]


def test_only_specified_exceptions_are_retried():
    @retry_with_backoff(attempts=3, initial_delay=0, exceptions=(ConnectionError,), on_retry=lambda *a: None)
    def raises_value_error():
        raise ValueError("not a connection issue")

    # ValueError isn't in `exceptions`, so it should propagate immediately
    # without retrying.
    with pytest.raises(ValueError):
        raises_value_error()


def test_preserves_function_metadata():
    @retry_with_backoff()
    def documented_function():
        """A docstring that should survive wrapping."""

    assert documented_function.__name__ == "documented_function"
    assert documented_function.__doc__ == "A docstring that should survive wrapping."
