"""
Chunked Prefill

Breaks long prompts into smaller chunks for scheduling-friendly prefill.
Instead of processing an entire 8K-token prompt in one forward pass,
chunked prefill splits it into smaller pieces (e.g., 512 tokens each)
that can be interleaved with decode steps from other requests.

This improves scheduler utilization and reduces latency for other requests
waiting in the queue.

Usage:
    chunker = ChunkedPrefiller(chunk_size=512)
    chunks = chunker.split_prompt([1, 2, 3, ..., 8000])
    # chunks == [[1..512], [513..1024], ..., [7681..8000]]

    # Use with scheduler
    for chunk in chunks:
        scheduler.schedule_chunk(request_id, chunk)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("draco.chunked_prefill")


@dataclass
class PrefillChunk:
    """A single chunk of a prefill prompt."""
    chunk_index: int
    token_ids: List[int]
    is_first: bool = False
    is_last: bool = False
    start_offset: int = 0
    total_tokens: int = 0

    @property
    def num_tokens(self) -> int:
        return len(self.token_ids)

    def __repr__(self) -> str:
        return (
            f"PrefillChunk("
            f"index={self.chunk_index}, "
            f"tokens={self.num_tokens}, "
            f"first={self.is_first}, "
            f"last={self.is_last})"
        )


@dataclass
class ChunkedPrefillState:
    """Tracks the state of chunked prefill for a single request."""
    request_id: int
    total_tokens: int
    chunks: List[PrefillChunk] = field(default_factory=list)
    current_chunk_index: int = 0
    completed: bool = False

    @property
    def num_completed_tokens(self) -> int:
        """Number of tokens that have been prefilled so far."""
        completed = 0
        for chunk in self.chunks[:self.current_chunk_index]:
            completed += chunk.num_tokens
        return completed

    @property
    def remaining_tokens(self) -> int:
        return self.total_tokens - self.num_completed_tokens

    @property
    def progress(self) -> float:
        if self.total_tokens == 0:
            return 1.0
        return self.num_completed_tokens / self.total_tokens

    def get_next_chunk(self) -> Optional[PrefillChunk]:
        """Get the next chunk to process."""
        if self.current_chunk_index >= len(self.chunks):
            return None
        return self.chunks[self.current_chunk_index]

    def advance(self) -> bool:
        """Advance to the next chunk. Returns True if more chunks remain."""
        self.current_chunk_index += 1
        if self.current_chunk_index >= len(self.chunks):
            self.completed = True
            return False
        return True


class ChunkedPrefiller:
    """
    Splits long prompts into smaller chunks for scheduling-friendly prefill.

    The chunker maintains chunk_size (in tokens) and tracks per-request
    prefill state to coordinate with the continuous batching scheduler.
    """

    def __init__(self, chunk_size: int = 512):
        """
        Args:
            chunk_size: Maximum tokens per prefill chunk.
        """
        self.chunk_size = chunk_size
        self._states: Dict[int, ChunkedPrefillState] = {}

    def split_prompt(self, token_ids: List[int]) -> List[PrefillChunk]:
        """Split a token sequence into chunks.

        Args:
            token_ids: Full token sequence to split

        Returns:
            List of PrefillChunk objects
        """
        if not token_ids:
            return [PrefillChunk(
                chunk_index=0, token_ids=[], is_first=True, is_last=True,
                start_offset=0, total_tokens=0,
            )]

        chunks = []
        total = len(token_ids)

        for i in range(0, total, self.chunk_size):
            chunk_tokens = token_ids[i:i + self.chunk_size]
            chunk = PrefillChunk(
                chunk_index=len(chunks),
                token_ids=chunk_tokens,
                is_first=(i == 0),
                is_last=(i + self.chunk_size >= total),
                start_offset=i,
                total_tokens=total,
            )
            chunks.append(chunk)

        return chunks

    def start_prefill(self, request_id: int, token_ids: List[int]) -> ChunkedPrefillState:
        """Start chunked prefill for a new request.

        Args:
            request_id: Unique request ID
            token_ids: Full prompt token IDs

        Returns:
            ChunkedPrefillState tracking the prefill progress
        """
        chunks = self.split_prompt(token_ids)
        state = ChunkedPrefillState(
            request_id=request_id,
            total_tokens=len(token_ids),
            chunks=chunks,
        )
        self._states[request_id] = state
        logger.debug(
            "Started chunked prefill for request %d: %d tokens in %d chunks",
            request_id, len(token_ids), len(chunks),
        )
        return state

    def get_next_chunk(self, request_id: int) -> Optional[PrefillChunk]:
        """Get the next prefill chunk for a request."""
        state = self._states.get(request_id)
        if state is None:
            return None
        return state.get_next_chunk()

    def advance(self, request_id: int) -> bool:
        """Advance to the next chunk. Returns True if more chunks remain."""
        state = self._states.get(request_id)
        if state is None:
            return False
        return state.advance()

    def is_done(self, request_id: int) -> bool:
        """Check if prefill is complete for a request."""
        state = self._states.get(request_id)
        if state is None:
            return True
        return state.completed

    def get_state(self, request_id: int) -> Optional[ChunkedPrefillState]:
        """Get the prefill state for a request."""
        return self._states.get(request_id)

    def remove_request(self, request_id: int) -> None:
        """Remove tracking state for a completed/cancelled request."""
        self._states.pop(request_id, None)

    @property
    def active_requests(self) -> int:
        """Number of requests with in-progress prefill."""
        return sum(1 for s in self._states.values() if not s.completed)

    @property
    def stats(self) -> Dict[str, Any]:
        total_tokens = sum(s.total_tokens for s in self._states.values())
        completed_tokens = sum(s.num_completed_tokens for s in self._states.values())
        return {
            "chunk_size": self.chunk_size,
            "active_requests": self.active_requests,
            "total_prefill_tokens": total_tokens,
            "completed_prefill_tokens": completed_tokens,
            "progress": completed_tokens / max(total_tokens, 1),
        }

    def __repr__(self) -> str:
        return (
            f"ChunkedPrefiller("
            f"chunk_size={self.chunk_size}, "
            f"active={self.active_requests})"
        )
