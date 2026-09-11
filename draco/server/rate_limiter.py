"""
Rate Limiter

Token bucket rate limiter for the Draco OpenAI-compatible server.
Supports per-user and global rate limits.

Usage:
    limiter = RateLimiter(requests_per_second=10, burst=20)
    if limiter.allow("user-123"):
        # Process request
    else:
        # Return 429
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    limit: int
    retry_after_ms: Optional[float] = None

    @property
    def headers(self) -> dict:
        """HTTP headers for rate limiting."""
        h = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(self.remaining),
        }
        if not self.allowed and self.retry_after_ms is not None:
            h["Retry-After"] = str(int(self.retry_after_ms / 1000) + 1)
        return h


class TokenBucket:
    """Single token bucket for rate limiting."""

    def __init__(self, capacity: int, refill_rate: float):
        """
        Args:
            capacity: Maximum tokens in bucket (burst size)
            refill_rate: Tokens added per second
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.tokens = float(capacity)
        self.last_refill = time.monotonic()

    def consume(self, tokens: int = 1) -> RateLimitResult:
        """Try to consume tokens from the bucket."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

        if self.tokens >= tokens:
            self.tokens -= tokens
            return RateLimitResult(
                allowed=True,
                remaining=int(self.tokens),
                limit=self.capacity,
            )
        else:
            # Calculate retry after
            deficit = tokens - self.tokens
            retry_after = (deficit / self.refill_rate) * 1000  # ms
            return RateLimitResult(
                allowed=False,
                remaining=0,
                limit=self.capacity,
                retry_after_ms=retry_after,
            )

    @property
    def available(self) -> int:
        return int(self.tokens)


class RateLimiter:
    """
    Token bucket rate limiter with per-user and global limits.

    Tracks rate limits per user key and globally. Each user gets
    an independent token bucket. The global bucket limits total throughput.
    """

    def __init__(
        self,
        requests_per_second: float = 10.0,
        burst: int = 20,
        per_user_rps: Optional[float] = None,
        per_user_burst: Optional[int] = None,
    ):
        """
        Args:
            requests_per_second: Global requests per second limit
            burst: Global burst capacity
            per_user_rps: Per-user requests per second (defaults to global)
            per_user_burst: Per-user burst capacity (defaults to burst)
        """
        self.requests_per_second = requests_per_second
        self.burst = burst
        self.per_user_rps = per_user_rps or requests_per_second
        self.per_user_burst = per_user_burst or burst

        # Global bucket
        self._global_bucket = TokenBucket(burst, requests_per_second)

        # Per-user buckets
        self._user_buckets: dict[str, TokenBucket] = {}

        # Statistics
        self._total_requests = 0
        self._total_rejected = 0

    def allow(self, user_key: Optional[str] = None) -> RateLimitResult:
        """
        Check if a request is allowed.

        Args:
            user_key: User identifier (None for anonymous/global-only)

        Returns:
            RateLimitResult with allowed status and remaining quota
        """
        self._total_requests += 1

        # Check global limit first
        global_result = self._global_bucket.consume()
        if not global_result.allowed:
            self._total_rejected += 1
            return global_result

        # Check per-user limit
        if user_key is not None:
            bucket = self._get_user_bucket(user_key)
            user_result = bucket.consume()
            if not user_result.allowed:
                self._total_rejected += 1
                # Restore global tokens
                self._global_bucket.tokens = min(
                    self._global_bucket.capacity,
                    self._global_bucket.tokens + 1,
                )
                return user_result

        return global_result

    def _get_user_bucket(self, user_key: str) -> TokenBucket:
        if user_key not in self._user_buckets:
            self._user_buckets[user_key] = TokenBucket(
                self.per_user_burst, self.per_user_rps,
            )
        return self._user_buckets[user_key]

    def reset_user(self, user_key: str) -> None:
        """Reset a user's rate limit bucket."""
        self._user_buckets.pop(user_key, None)

    def reset_all(self) -> None:
        """Reset all rate limit buckets."""
        self._global_bucket = TokenBucket(self.burst, self.requests_per_second)
        self._user_buckets.clear()

    @property
    def stats(self) -> dict:
        return {
            "total_requests": self._total_requests,
            "total_rejected": self._total_rejected,
            "active_users": len(self._user_buckets),
            "global_available": self._global_bucket.available,
        }

    def __repr__(self) -> str:
        return (
            f"RateLimiter("
            f"rps={self.requests_per_second}, "
            f"burst={self.burst}, "
            f"per_user_rps={self.per_user_rps})"
        )
