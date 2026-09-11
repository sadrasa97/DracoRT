"""
Continuous batching for the native CPU runtime (spec: "continuous
batching" requirement).

Reuses draco.scheduler.scheduler.ContinuousBatchScheduler as-is rather
than forking a CPU-specific copy (spec section 54's "don't fork" rule)
— the scheduler's request bookkeeping is device-agnostic. This module
supplies the CPU-specific half: turning scheduled batches into prefill/
decode calls against CPUExecutionBackend, admitting new sequences only
when the KV cache has room, and freeing KV blocks as sequences finish.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from draco.runtime.cpu.executor import CPUExecutionBackend
from draco.scheduler.scheduler import ContinuousBatchScheduler


class CPUBatchEngine:
    def __init__(
        self,
        executor: CPUExecutionBackend,
        max_num_seqs: int = 16,
        max_num_batched_tokens: int = 8192,
    ) -> None:
        self.executor = executor
        self.scheduler = ContinuousBatchScheduler(
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_batch_size=max_num_seqs,
        )
        self._seq_ids: Dict[int, int] = {}  # request_id -> KV cache seq_id
        self._prefilled: Dict[int, bool] = {}
        self._results: Dict[int, List[int]] = {}
        self._active_request_ids: set = set()

    def add_request(
        self, request_id: int, prompt_token_ids: List[int], max_tokens: int = 32, **kw
    ) -> None:
        self.scheduler.add_request(
            request_id=request_id, prompt_token_ids=prompt_token_ids, max_tokens=max_tokens, **kw
        )
        self._results[request_id] = list(prompt_token_ids)

    def step(self) -> bool:
        """Run one scheduling step. Returns True if there's still work to do."""
        batch = self.scheduler.schedule()  # newly-admitted request ids this call only

        # The scheduler only reports *newly* admitted requests each call, so
        # we track the full set of currently-running requests ourselves —
        # everything admitted so far that hasn't finished yet.
        self._active_request_ids.update(batch.request_ids)

        for request_id in batch.request_ids:
            self._seq_ids[request_id] = self.executor.new_sequence()
            self._prefilled[request_id] = False

        active_ids = list(self._active_request_ids)

        decode_request_ids: List[int] = []
        decode_batch_ids: List[int] = []
        decode_tokens: List[int] = []
        decode_positions: List[int] = []

        for request_id in active_ids:
            req = self.scheduler.get_request(request_id)
            seq_id = self._seq_ids[request_id]

            if not self._prefilled[request_id]:
                # Prefill the full prompt for this sequence in one call.
                logits = self.executor.forward_step(
                    seq_id, np.array(req.prompt_token_ids, dtype=np.int64), start_position=0
                )
                self._prefilled[request_id] = True
                next_id = int(np.argmax(logits))
                self._emit(request_id, next_id)
            else:
                decode_request_ids.append(request_id)
                decode_batch_ids.append(seq_id)
                last_token = self._results[request_id][-1]
                position = len(self._results[request_id]) - 1
                decode_tokens.append(last_token)
                decode_positions.append(position)

        if decode_batch_ids:
            logits = self.executor.forward_batch_decode(decode_batch_ids, decode_tokens, decode_positions)
            next_ids = np.argmax(logits, axis=-1)
            for rid, next_id in zip(decode_request_ids, next_ids):
                self._emit(rid, int(next_id))

        for request_id in self.scheduler.get_finished():
            seq_id = self._seq_ids.get(request_id.request_id)
            if seq_id is not None:
                self.executor.end_sequence(seq_id)
            self._active_request_ids.discard(request_id.request_id)

        return not self.scheduler.is_finished()

    def _emit(self, request_id: int, token_id: int) -> None:
        self._results[request_id].append(token_id)
        still_going = self.scheduler.update_request(request_id, token_id)
        if not still_going:
            pass  # scheduler already moved it to finished; freed in step()

    def run_to_completion(self, max_steps: int = 100_000) -> Dict[int, List[int]]:
        steps = 0
        while self.scheduler.running_count or self.scheduler.waiting_count or self.scheduler.preempted_count:
            if not self.step():
                break
            steps += 1
            if steps >= max_steps:
                raise RuntimeError(f"CPUBatchEngine exceeded max_steps={max_steps} without finishing.")
        return dict(self._results)
