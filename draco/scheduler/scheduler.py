"""
Continuous Batching Scheduler

Manages request queuing, dynamic batch formation, and preemption.
Groups requests into batches for efficient GPU utilization.

Supports:
- Dynamic batch formation from waiting requests
- Request priority and preemption
- Batch size limits
- Token budget management
"""

from __future__ import annotations

import heapq
import itertools
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("draco.scheduler")


class RequestStatus(Enum):
    """Status of a generation request."""
    WAITING = "waiting"
    RUNNING = "running"
    PREEMPTED = "preempted"
    FINISHED = "finished"


@dataclass
class SchedulerRequest:
    """A single generation request managed by the scheduler."""
    request_id: int
    prompt_token_ids: List[int]
    max_tokens: int
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    stop_token_ids: Optional[List[int]] = None
    status: RequestStatus = RequestStatus.WAITING
    priority: int = 0
    created_at: float = field(default_factory=time.monotonic)
    num_generated_tokens: int = 0

    # Accumulated output
    generated_token_ids: List[int] = field(default_factory=list)

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def total_length(self) -> int:
        return self.prompt_length + self.num_generated_tokens


@dataclass
class BatchState:
    """Current batch state."""
    request_ids: List[int] = field(default_factory=list)
    batch_size: int = 0
    total_tokens: int = 0


class _PriorityWaitQueue:
    """Priority-ordered wait queue backed by a heap.

    `add_request` previously did a linear scan + `deque.insert(i, ...)`
    to keep the waiting queue sorted by priority — O(n) per insertion.
    With many queued requests (exactly the situation a busy scheduler is
    in) that turns enqueueing into an O(n^2) pattern. This keeps the same
    "higher priority first, FCFS among equal priority" ordering but with
    O(log n) push/pop via heapq.
    """

    def __init__(self) -> None:
        self._heap: List[Any] = []
        self._counter = itertools.count()

    def push(self, req: "SchedulerRequest") -> None:
        # Negate priority since heapq is a min-heap and higher priority
        # should come out first; the monotonic counter breaks ties in
        # insertion order without ever comparing SchedulerRequest objects.
        heapq.heappush(self._heap, (-req.priority, next(self._counter), req))

    def peek(self) -> "SchedulerRequest":
        return self._heap[0][2]

    def popleft(self) -> "SchedulerRequest":
        return heapq.heappop(self._heap)[2]

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def __iter__(self):
        # Order isn't guaranteed to be priority-sorted here (heap
        # invariant only guarantees the root), which is fine — this is
        # only used for lookups (get_request), not scheduling decisions.
        return (item[2] for item in self._heap)

    def __getitem__(self, idx: int) -> "SchedulerRequest":
        if idx == 0:
            return self.peek()
        raise IndexError("_PriorityWaitQueue only supports peeking index 0")


class ContinuousBatchScheduler:
    """Scheduler that forms dynamic batches from waiting requests.

    Implements a simple FCFS scheduler with optional priority and preemption.
    """

    def __init__(
        self,
        max_num_seqs: int = 256,
        max_num_batched_tokens: int = 8192,
        max_batch_size: int = 32,
        preemption_enabled: bool = False,
    ):
        self.max_num_seqs = max_num_seqs
        self.max_num_batched_tokens = max_num_batched_tokens
        self.max_batch_size = max_batch_size
        self.preemption_enabled = preemption_enabled

        # Request queues
        self._waiting: _PriorityWaitQueue = _PriorityWaitQueue()
        self._running: Dict[int, SchedulerRequest] = {}
        self._preempted: deque[SchedulerRequest] = deque()
        self._finished: List[SchedulerRequest] = []

        # Current batch
        self._current_batch = BatchState()

        # Statistics
        self._total_requests_scheduled = 0
        self._total_tokens_processed = 0
        self._total_batches = 0

    def add_request(
        self,
        request_id: int,
        prompt_token_ids: List[int],
        max_tokens: int = 16,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop_token_ids: Optional[List[int]] = None,
        priority: int = 0,
    ) -> SchedulerRequest:
        """Add a new request to the waiting queue.

        Args:
            request_id: Unique request ID
            prompt_token_ids: Token IDs for the prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling
            stop_token_ids: Token IDs that stop generation
            priority: Higher = scheduled first

        Returns:
            The created SchedulerRequest
        """
        req = SchedulerRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            stop_token_ids=stop_token_ids,
            priority=priority,
        )

        # O(log n) priority-ordered insert (see _PriorityWaitQueue).
        self._waiting.push(req)

        self._total_requests_scheduled += 1
        logger.debug("Added request %d (priority=%d, prompt_len=%d)", request_id, priority, len(prompt_token_ids))
        return req

    def schedule(self) -> BatchState:
        """Form the next batch from waiting/preempted requests.

        Returns:
            BatchState with the scheduled request IDs.
        """
        new_batch_ids: List[int] = []
        new_batch_tokens = 0

        # `max_num_seqs` bounds total concurrently-running sequences; it
        # was previously stored but never enforced, so a scheduler could
        # admit unboundedly many running requests as long as batch/token
        # limits allowed it.
        def _room_for_more() -> bool:
            return (
                len(new_batch_ids) < self.max_batch_size
                and len(self._running) < self.max_num_seqs
            )

        # First, try to resume preempted requests
        while self._preempted and _room_for_more():
            req = self._preempted.popleft()
            prompt_tokens = req.prompt_length
            if new_batch_tokens + prompt_tokens <= self.max_num_batched_tokens:
                req.status = RequestStatus.RUNNING
                self._running[req.request_id] = req
                new_batch_ids.append(req.request_id)
                new_batch_tokens += prompt_tokens
            else:
                self._preempted.appendleft(req)
                break

        # Then, add from waiting queue
        while self._waiting and _room_for_more():
            req = self._waiting[0]
            prompt_tokens = req.prompt_length
            if new_batch_tokens + prompt_tokens <= self.max_num_batched_tokens:
                self._waiting.popleft()
                req.status = RequestStatus.RUNNING
                self._running[req.request_id] = req
                new_batch_ids.append(req.request_id)
                new_batch_tokens += prompt_tokens
            else:
                break

        self._current_batch = BatchState(
            request_ids=new_batch_ids,
            batch_size=len(new_batch_ids),
            total_tokens=new_batch_tokens,
        )
        self._total_batches += 1

        logger.debug(
            "Scheduled batch %d: %d requests, %d tokens",
            self._total_batches, len(new_batch_ids), new_batch_tokens,
        )
        return self._current_batch

    def update_request(
        self,
        request_id: int,
        generated_token_id: int,
    ) -> bool:
        """Update a request with a newly generated token.

        Args:
            request_id: The request to update
            generated_token_id: The token that was generated

        Returns:
            True if generation should continue, False if done.
        """
        if request_id not in self._running:
            return False

        req = self._running[request_id]
        req.generated_token_ids.append(generated_token_id)
        req.num_generated_tokens += 1

        # Check stop conditions
        if req.num_generated_tokens >= req.max_tokens:
            self._finish_request(request_id)
            return False

        if req.stop_token_ids and generated_token_id in req.stop_token_ids:
            self._finish_request(request_id)
            return False

        return True

    def _finish_request(self, request_id: int) -> None:
        """Move a request from running to finished."""
        if request_id in self._running:
            req = self._running.pop(request_id)
            req.status = RequestStatus.FINISHED
            self._finished.append(req)
            self._total_tokens_processed += req.num_generated_tokens

    def preempt_request(self, request_id: int) -> None:
        """Preempt a running request (move to preempted queue)."""
        if not self.preemption_enabled:
            return
        if request_id in self._running:
            req = self._running.pop(request_id)
            req.status = RequestStatus.PREEMPTED
            self._preempted.append(req)
            logger.debug("Preempted request %d", request_id)

    def is_finished(self) -> bool:
        """Check if all requests are done."""
        return (
            len(self._waiting) == 0
            and len(self._running) == 0
            and len(self._preempted) == 0
        )

    def get_request(self, request_id: int) -> Optional[SchedulerRequest]:
        """Get a request by ID from any queue."""
        if request_id in self._running:
            return self._running[request_id]
        for req in self._waiting:
            if req.request_id == request_id:
                return req
        for req in self._preempted:
            if req.request_id == request_id:
                return req
        for req in self._finished:
            if req.request_id == request_id:
                return req
        return None

    def get_finished(self) -> List[SchedulerRequest]:
        """Get all finished requests and clear the finished list."""
        result = list(self._finished)
        self._finished.clear()
        return result

    @property
    def waiting_count(self) -> int:
        return len(self._waiting)

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def preempted_count(self) -> int:
        return len(self._preempted)

    @property
    def current_batch(self) -> BatchState:
        return self._current_batch

    @property
    def utilization(self) -> float:
        """Scheduler utilization: running / (running + waiting)."""
        total = len(self._running) + len(self._waiting)
        if total == 0:
            return 0.0
        return len(self._running) / total

    def __repr__(self) -> str:
        return (
            f"ContinuousBatchScheduler("
            f"waiting={len(self._waiting)}, "
            f"running={len(self._running)}, "
            f"preempted={len(self._preempted)}, "
            f"finished={len(self._finished)}, "
            f"total_batches={self._total_batches})"
        )
