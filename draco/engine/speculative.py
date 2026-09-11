"""
Speculative Decoding

Uses a small draft model to propose multiple tokens, then verifies them
with the target model in a single forward pass. Tokens that match are
accepted; mismatches cause rollback to the first disagreement.

Speculative decoding improves throughput by reducing the number of
sequential target model forward passes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

logger = logging.getLogger("draco.speculative")


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""
    num_speculative_tokens: int = 5  # Tokens proposed by draft model
    draft_temperature: float = 0.0   # Draft uses greedy for speed
    target_temperature: float = 1.0  # Target uses specified sampling
    top_p: float = 1.0
    top_k: int = -1
    rejection_free: bool = False     # If True, always accept (for benchmarking)


@dataclass
class SpeculativeStep:
    """Result of a single speculative decoding step."""
    draft_tokens: List[int]
    draft_probs: List[float]
    target_tokens: List[int]
    target_probs: List[float]
    accepted_mask: List[bool]
    num_accepted: int
    num_rejected: int
    first_rejection_idx: Optional[int]
    tokens_generated: int
    step_time_ms: float
    bonus_token: Optional[int] = None


@dataclass
class SpeculativeStats:
    """Aggregated statistics across multiple speculative steps."""
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_rejected_tokens: int = 0
    total_steps: int = 0
    total_target_forward_passes: int = 0
    total_draft_forward_passes: int = 0
    total_time_ms: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        total = self.total_accepted_tokens + self.total_rejected_tokens
        if total == 0:
            return 0.0
        return self.total_accepted_tokens / total

    @property
    def tokens_per_step(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_steps

    @property
    def speedup(self) -> float:
        """Estimated speedup over standard autoregressive decoding.

        Standard: 1 token per target forward pass.
        Speculative: avg_accepted_tokens per (1 draft + 1 target) pass.
        """
        if self.total_target_forward_passes == 0:
            return 1.0
        standard_tokens = self.total_target_forward_passes
        speculative_tokens = self.total_accepted_tokens
        return speculative_tokens / standard_tokens if standard_tokens > 0 else 1.0


class SpeculativeDecoder:
    """Speculative decoding engine.

    Orchestrates draft model proposal, target model verification,
    and token acceptance/rejection.
    """

    def __init__(
        self,
        target_model: Any,
        draft_model: Any,
        config: Optional[SpeculativeConfig] = None,
        tokenizer: Any = None,
    ):
        """
        Args:
            target_model: The main (large) model — must have forward() returning logits
            draft_model: The small draft model — same interface
            config: Speculative decoding configuration
            tokenizer: Tokenizer for EOS detection
        """
        self.target_model = target_model
        self.draft_model = draft_model
        self.config = config or SpeculativeConfig()
        self.tokenizer = tokenizer
        self.stats = SpeculativeStats()

        # Move draft to same device as target
        target_device = next(target_model.parameters()).device
        self.draft_model = self.draft_model.to(target_device)
        self.draft_model.eval()
        self.target_model.eval()

    @staticmethod
    def _sample_from_logits(
        logits: torch.Tensor,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = -1,
    ) -> Tuple[int, float]:
        """Sample a single token from a logits vector.

        Shared by draft proposal and target verification/bonus sampling so
        the greedy-vs-stochastic decision is made the same way everywhere.
        Previously the target-side check was `temperature > 0 and
        temperature < 1e-7`, which is never true for `temperature == 0.0`
        (fails `> 0`) — so a caller asking for greedy decoding via
        `temperature=0.0` silently fell through to the stochastic branch
        instead. Fixed to a single `<= 1e-7` greedy threshold.
        """
        if temperature <= 1e-7:
            token = int(logits.argmax().item())
            prob = float(torch.softmax(logits, dim=-1)[token].item())
            return token, prob

        probs = torch.softmax(logits / max(temperature, 1e-7), dim=-1)

        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            mask = cumulative > top_p
            mask[0] = False
            probs[sorted_indices[mask]] = 0.0
            probs = probs / probs.sum()

        if top_k > 0:
            top_k_vals, _ = torch.topk(probs, top_k)
            min_val = top_k_vals[-1]
            probs[probs < min_val] = 0.0
            probs = probs / probs.sum()

        token = int(torch.multinomial(probs, 1).item())
        prob = float(probs[token].item())
        return token, prob

    @torch.no_grad()
    def generate(
        self,
        prompt_token_ids: List[int],
        max_tokens: int = 128,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop_token_ids: Optional[List[int]] = None,
    ) -> List[int]:
        """Generate tokens using speculative decoding.

        Args:
            prompt_token_ids: Starting token IDs
            max_tokens: Maximum tokens to generate
            temperature: Target model temperature
            top_p: Target model top_p
            top_k: Target model top_k
            stop_token_ids: Tokens that stop generation

        Returns:
            List of generated token IDs (excluding prompt)
        """
        device = next(self.target_model.parameters()).device
        generated: List[int] = []
        all_ids = list(prompt_token_ids)
        past_length = len(prompt_token_ids)

        self.stats = SpeculativeStats()

        while len(generated) < max_tokens:
            remaining = max_tokens - len(generated)
            num_spec = min(self.config.num_speculative_tokens, remaining)

            if num_spec <= 0:
                break

            step_result = self._speculative_step(
                all_ids=all_ids,
                past_length=past_length,
                num_speculative=num_spec,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

            if step_result is None:
                break

            # Apply accepted tokens
            rejected = False
            for i, (accepted, token) in enumerate(
                zip(step_result.accepted_mask, step_result.target_tokens)
            ):
                if accepted:
                    generated.append(token)
                    all_ids.append(token)
                    past_length += 1

                    # Check stop conditions
                    if stop_token_ids and token in stop_token_ids:
                        return generated
                    if self._is_eos(token):
                        return generated
                else:
                    # First rejection: use target's token at that position
                    rejected = True
                    generated.append(token)
                    all_ids.append(token)
                    past_length += 1

                    if stop_token_ids and token in stop_token_ids:
                        return generated
                    if self._is_eos(token):
                        return generated
                    break

            # All draft tokens were accepted: the target model's forward
            # pass already computed the distribution for the position
            # right after the last accepted draft token, so one more
            # token comes "for free" with no extra forward pass — the
            # classic speculative-decoding bonus token.
            if not rejected and step_result.bonus_token is not None and len(generated) < max_tokens:
                token = step_result.bonus_token
                generated.append(token)
                all_ids.append(token)
                past_length += 1
                if stop_token_ids and token in stop_token_ids:
                    return generated
                if self._is_eos(token):
                    return generated

            # Update stats
            self.stats.total_steps += 1
            self.stats.total_time_ms += step_result.step_time_ms

        return generated

    def _speculative_step(
        self,
        all_ids: List[int],
        past_length: int,
        num_speculative: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> Optional[SpeculativeStep]:
        """Execute one speculative decoding step.

        1. Draft model proposes `num_speculative` tokens
        2. Target model verifies all tokens in one forward pass
        3. Accept matching tokens, reject at first mismatch
        """
        t0 = time.perf_counter()
        device = next(self.target_model.parameters()).device

        # --- Step 1: Draft model proposes tokens ---
        # Feed the full growing sequence on every step. The draft/target
        # model interface here is stateless (forward(input_ids) -> logits,
        # no KV cache handle passed in), so re-using only the last token
        # after the first iteration — as before — silently discarded all
        # prior context for every token after the first, making draft
        # proposals close to random beyond position 1.
        draft_tokens: List[int] = []
        draft_probs: List[float] = []
        draft_ids = list(all_ids)

        for _ in range(num_speculative):
            draft_input = torch.tensor([draft_ids], dtype=torch.long, device=device)
            draft_logits = self.draft_model(draft_input)
            if isinstance(draft_logits, tuple):
                draft_logits = draft_logits[0]

            next_logits = draft_logits[0, -1, :]
            token, prob = self._sample_from_logits(
                next_logits, temperature=self.config.draft_temperature,
            )

            draft_tokens.append(token)
            draft_probs.append(prob)
            draft_ids.append(token)

        self.stats.total_draft_forward_passes += num_speculative

        # --- Step 2: Target model verifies ---
        # Build input: original sequence + draft tokens
        verify_ids = all_ids + draft_tokens
        verify_input = torch.tensor([verify_ids], dtype=torch.long, device=device)

        target_logits = self.target_model(verify_input)
        if isinstance(target_logits, tuple):
            target_logits = target_logits[0]

        # --- Step 3: Accept/reject ---
        # `target_logits[0, t, :]` is the model's predicted distribution
        # for the token at position t+1 given tokens[0..t] — i.e. it
        # verifies draft_tokens[i] using the distribution conditioned on
        # everything *before* draft_tokens[i], which sits at index
        # `past_length + i - 1` in verify_input (past_length - 1 for the
        # first draft token, using the last real context token).
        # The previous code used `past_length + i`, which is the
        # distribution for draft_tokens[i + 1] instead — verifying every
        # draft token against the wrong target distribution.
        target_tokens: List[int] = []
        target_probs: List[float] = []
        accepted_mask: List[bool] = []
        num_accepted = 0
        num_rejected = 0
        first_rejection_idx: Optional[int] = None

        for i in range(num_speculative):
            position = past_length + i - 1
            target_logit = target_logits[0, position, :]
            target_token, target_prob = self._sample_from_logits(
                target_logit, temperature=temperature, top_p=top_p, top_k=top_k,
            )

            target_tokens.append(target_token)
            target_probs.append(target_prob)

            # Acceptance criterion: target token matches draft
            accepted = True if self.config.rejection_free else (target_token == draft_tokens[i])
            accepted_mask.append(accepted)

            if accepted:
                num_accepted += 1
            else:
                num_rejected += 1
                if first_rejection_idx is None:
                    first_rejection_idx = i
                break  # Stop at first rejection

        # Bonus token: if every proposed draft token was accepted, the
        # target model's forward pass already computed the distribution
        # for the token right after the last draft token (at the final
        # position of verify_input) — sample it for a free extra token
        # with zero additional forward passes.
        bonus_token: Optional[int] = None
        if num_rejected == 0 and num_accepted == num_speculative:
            bonus_position = past_length + num_speculative - 1
            bonus_logit = target_logits[0, bonus_position, :]
            bonus_token, _ = self._sample_from_logits(
                bonus_logit, temperature=temperature, top_p=top_p, top_k=top_k,
            )

        self.stats.total_accepted_tokens += num_accepted
        self.stats.total_rejected_tokens += num_rejected
        self.stats.total_target_forward_passes += 1

        step_time = (time.perf_counter() - t0) * 1000

        return SpeculativeStep(
            draft_tokens=draft_tokens,
            draft_probs=draft_probs,
            target_tokens=target_tokens,
            target_probs=target_probs,
            accepted_mask=accepted_mask,
            num_accepted=num_accepted,
            num_rejected=num_rejected,
            first_rejection_idx=first_rejection_idx,
            tokens_generated=num_accepted + (1 if num_rejected > 0 else 0) + (1 if bonus_token is not None else 0),
            step_time_ms=step_time,
            bonus_token=bonus_token,
        )

    def _is_eos(self, token_id: int) -> bool:
        """Check if token is end-of-sequence."""
        if self.tokenizer is not None:
            eos_id = getattr(self.tokenizer, "eos_token_id", None)
            if eos_id is not None and token_id == eos_id:
                return True
        return False

    def get_stats(self) -> SpeculativeStats:
        """Get current speculative decoding statistics."""
        return self.stats

    def __repr__(self) -> str:
        return (
            f"SpeculativeDecoder("
            f"draft={type(self.draft_model).__name__}, "
            f"target={type(self.target_model).__name__}, "
            f"num_speculative={self.config.num_speculative_tokens})"
        )
