"""
Adaptive speculative decoding for the native CPU runtime, draft-model-free.

Uses prompt-lookup decoding (a.k.a. self-speculation): instead of a
separate small draft model — which would need its own conversion/loading
pipeline, out of scope here — draft tokens are proposed by matching the
last few generated tokens against earlier context and copying whatever
followed that match previously. This is a real, known-effective technique
for repetitive/structured generation (e.g. code, quoting back input) and
needs no second model. See draco.engine.speculative for the GPU engine's
separate-draft-model speculative decoder this complements; the two share
the same greedy accept/reject math but not code, since the CPU version's
"draft" comes from n-gram lookup rather than a second forward pass.

Adaptivity: draft length is widened after a run of good acceptance and
narrowed after a run of poor acceptance, via an exponential moving
average of the per-round acceptance rate — cheap drafts stay cheap when
they're not paying off, and get more ambitious when they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from draco.runtime.cpu.executor import CPUExecutionBackend


@dataclass
class AdaptiveSpeculativeConfig:
    ngram_size: int = 3
    min_draft_len: int = 1
    max_draft_len: int = 8
    initial_draft_len: int = 4
    ema_alpha: float = 0.3
    grow_threshold: float = 0.7  # acceptance rate above which draft length grows
    shrink_threshold: float = 0.3  # acceptance rate below which draft length shrinks


@dataclass
class SpeculativeStats:
    rounds: int = 0
    proposed_tokens: int = 0
    accepted_tokens: int = 0
    bonus_tokens: int = 0
    draft_len_history: List[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        if self.proposed_tokens == 0:
            return 0.0
        return self.accepted_tokens / self.proposed_tokens

    @property
    def tokens_per_round(self) -> float:
        if self.rounds == 0:
            return 0.0
        return (self.accepted_tokens + self.bonus_tokens) / self.rounds


def _propose_ngram_draft(context: List[int], ngram_size: int, draft_len: int) -> Optional[List[int]]:
    """Look for the most recent earlier occurrence of the last `ngram_size`
    tokens elsewhere in `context`, and propose whatever followed it."""
    if len(context) < ngram_size + 1:
        return None
    needle = context[-ngram_size:]
    # search from the end backwards (excluding the needle's own position)
    search_end = len(context) - ngram_size
    for start in range(search_end - 1, -1, -1):
        if context[start : start + ngram_size] == needle:
            candidate = context[start + ngram_size : start + ngram_size + draft_len]
            if candidate:
                return candidate
    return None


class AdaptiveSpeculativeDecoder:
    def __init__(self, executor: CPUExecutionBackend, config: Optional[AdaptiveSpeculativeConfig] = None) -> None:
        self.executor = executor
        self.config = config or AdaptiveSpeculativeConfig()
        self._draft_len = self.config.initial_draft_len
        self._acceptance_ema = 0.5
        self.stats = SpeculativeStats()

    def _adapt(self, accepted: int, proposed: int) -> None:
        if proposed == 0:
            return
        rate = accepted / proposed
        alpha = self.config.ema_alpha
        self._acceptance_ema = alpha * rate + (1 - alpha) * self._acceptance_ema

        if self._acceptance_ema >= self.config.grow_threshold:
            self._draft_len = min(self.config.max_draft_len, self._draft_len + 1)
        elif self._acceptance_ema <= self.config.shrink_threshold:
            self._draft_len = max(self.config.min_draft_len, self._draft_len - 1)

    def generate(self, prompt_ids: List[int], max_new_tokens: int) -> List[int]:
        cfg = self.config
        seq_id = self.executor.new_sequence()
        try:
            context = list(prompt_ids)
            logits = self.executor.forward_step(
                seq_id, np.array(context, dtype=np.int64), start_position=0
            )
            generated = 0

            while generated < max_new_tokens:
                draft = _propose_ngram_draft(context, cfg.ngram_size, self._draft_len)
                if not draft:
                    # No repeated pattern to exploit this round: fall back to
                    # plain greedy decoding for one token (still correct,
                    # just no speedup this round).
                    next_id = int(np.argmax(logits))
                    context.append(next_id)
                    generated += 1
                    self.stats.rounds += 1
                    self.stats.bonus_tokens += 1
                    if generated >= max_new_tokens:
                        break
                    logits = self.executor.forward_step(
                        seq_id, np.array([next_id], dtype=np.int64), start_position=len(context) - 1
                    )
                    continue

                draft = draft[: max(1, min(len(draft), max_new_tokens - generated))]
                verify_start = len(context)
                all_logits = self.executor.forward_step(
                    seq_id,
                    np.array(draft, dtype=np.int64),
                    start_position=verify_start,
                    return_all_positions=True,
                )  # (len(draft), vocab); all_logits[i] predicts token after draft[i]

                accepted = 0
                cur_logits = logits  # prediction for the token at verify_start (before draft[0])
                for i, drafted_token in enumerate(draft):
                    predicted = int(np.argmax(cur_logits))
                    if predicted != drafted_token:
                        break
                    accepted += 1
                    cur_logits = all_logits[i]

                self.stats.rounds += 1
                self.stats.proposed_tokens += len(draft)
                self.stats.accepted_tokens += accepted
                self.stats.draft_len_history.append(self._draft_len)

                # Roll back KV for any rejected suffix of the draft.
                self.executor.truncate_sequence(seq_id, verify_start + accepted)

                context.extend(draft[:accepted])
                generated += accepted

                # Bonus token: the model's own greedy choice at the last
                # accepted position is always correct-by-definition and
                # free (no extra forward pass needed).
                if generated < max_new_tokens:
                    bonus_token = int(np.argmax(cur_logits))
                    context.append(bonus_token)
                    generated += 1
                    self.stats.bonus_tokens += 1
                    if generated >= max_new_tokens:
                        break
                    logits = self.executor.forward_step(
                        seq_id, np.array([bonus_token], dtype=np.int64), start_position=len(context) - 1
                    )
                else:
                    break

                self._adapt(accepted, len(draft))

            return context
        finally:
            self.executor.end_sequence(seq_id)
