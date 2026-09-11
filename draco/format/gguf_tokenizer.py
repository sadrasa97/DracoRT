"""
Byte-level BPE tokenizer for GGUF models (llama / gpt2 tokenizer models).

GGUF files embed their tokenizer (tokens, scores, token types, BPE merges,
chat template) so the model is self-contained. This module implements the
byte-level BPE algorithm those files use — the same algorithm as GPT-2 /
Llama's ``bytes_to_unicode`` vocabulary — plus a small chat-template
renderer for the two common template families (Qwen ``<|im_start|>`` and
Llama-3 ``<|start_header_id|>``), so a .gguf model can be used for text
generation without any external tokenizer.

The GGUF-stored token strings are the *mapped* byte-level forms
(e.g. ``"Ġthe"`` for ``b" the"``), so ``encode()`` maps input text to the
same representation before applying BPE merges.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:  # the `regex` module gives the exact GPT-2 \p{L}/\p{N} classes
    import regex as _regex  # type: ignore

    _HAS_REGEX = True
except ImportError:  # pragma: no cover - depends on environment
    _regex = None
    _HAS_REGEX = False

# GPT-2 pre-tokenization pattern. With the `regex` module this is the exact
# upstream pattern; the stdlib fallback approximates \p{L} with \w.
_GPT2_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
_FALLBACK_PATTERN = (
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+"""
)

_SPECIAL_TOKEN_RE = re.compile(r"^<\|.*\|>$|^<\|endoftext\|>$")
# token types used by llama.cpp (llama_token_type)
_TOKEN_TYPE_CONTROL = 3
_TOKEN_TYPE_USER_DEFINED = 4


def bytes_to_unicode() -> Dict[int, str]:
    """The GPT-2 byte -> unicode-char mapping."""
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def _get_pairs(word: Tuple[str, ...]) -> set:
    return set(zip(word, word[1:]))


class GGUFBPETokenizer:
    """Minimal byte-level BPE tokenizer over GGUF-embedded vocab data."""

    def __init__(
        self,
        tokens: Sequence[str],
        merges: Optional[Sequence[str]] = None,
        scores: Optional[Sequence[float]] = None,
        token_type: Optional[Sequence[int]] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        unknown_token_id: Optional[int] = None,
        chat_template: Optional[str] = None,
        model_type: str = "llama",
    ) -> None:
        self._tokens = list(tokens)
        self._scores = list(scores) if scores is not None else None
        self._token_type = list(token_type) if token_type is not None else None
        self.bos_token_id = bos_token_id
        self.unknown_token_id = unknown_token_id
        self.chat_template = chat_template or None
        self.model_type = model_type

        self._b2u = bytes_to_unicode()
        self._rev = {v: k for k, v in self._b2u.items()}

        self._vocab: Dict[str, int] = {}
        self._byte_to_id: Dict[int, int] = {}
        for i, token in enumerate(self._tokens):
            self._vocab.setdefault(token, i)
            if len(token) == 1 and token in self._rev:
                self._byte_to_id.setdefault(self._rev[token], i)

        self._rank: Dict[Tuple[str, str], int] = {}
        for rank, merge in enumerate(merges or []):
            parts = merge.split(" ")
            if len(parts) == 2:
                self._rank[(parts[0], parts[1])] = rank

        self._specials: List[str] = []
        for i, token in enumerate(self._tokens):
            tt = self._token_type[i] if self._token_type else None
            if tt in (_TOKEN_TYPE_CONTROL, _TOKEN_TYPE_USER_DEFINED) or _SPECIAL_TOKEN_RE.match(
                token
            ):
                self._specials.append(token)
        # longest-first so greedy matching picks the longest special token
        self._specials_sorted = sorted(self._specials, key=len, reverse=True)
        self._special_ids = {self._vocab.get(s) for s in self._specials if s in self._vocab}

        if _HAS_REGEX:
            self._pat = _regex.compile(_GPT2_PATTERN, _regex.IGNORECASE)
        else:
            self._pat = re.compile(_FALLBACK_PATTERN, re.IGNORECASE)

        # eos: metadata id if given, else the chat end token, else <|endoftext|>
        self.eos_token_id = eos_token_id
        if self.eos_token_id is None:
            self.eos_token_id = self._id_of("<|im_end|>") or self._id_of("<|endoftext|>")
        extra = set()
        for t in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>", "<|end_of_text|>"):
            tid = self._id_of(t)
            if tid is not None:
                extra.add(tid)
        self.extra_eos_ids = sorted(extra)

        self._bpe_cache: Dict[str, Tuple[str, ...]] = {}

    # ------------------------------------------------------------------
    # public API (mirrors the transformers tokenizer surface the engine uses)
    # ------------------------------------------------------------------

    @property
    def all_special_tokens(self) -> List[str]:
        return list(self._specials)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def _id_of(self, token: str) -> Optional[int]:
        return self._vocab.get(token)

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        i = 0
        n = len(text)
        while i < n:
            matched: Optional[str] = None
            for s in self._specials_sorted:
                if text.startswith(s, i):
                    matched = s
                    break
            if matched is not None:
                ids.append(self._vocab[matched])
                i += len(matched)
                continue
            next_special = n
            for s in self._specials_sorted:
                idx = text.find(s, i)
                if idx != -1 and idx < next_special:
                    next_special = idx
            seg = text[i:next_special]
            if seg:
                ids.extend(self._encode_segment(seg))
            i = next_special
        return ids

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        out = bytearray()
        for tid in token_ids:
            if tid is None or tid < 0 or tid >= len(self._tokens):
                continue
            if skip_special_tokens and tid in self._special_ids:
                continue
            token = self._tokens[tid]
            for ch in token:
                b = self._rev.get(ch)
                if b is not None:
                    out.append(b)
                else:
                    out.extend(ch.encode("utf-8"))
        return out.decode("utf-8", errors="replace")

    def apply_chat_template(
        self,
        messages: Sequence[Dict[str, Any]],
        add_generation_prompt: bool = True,
        tokenize: bool = False,
    ) -> str:
        """Render ``messages`` with the GGUF chat template.

        Only the two dominant template families are rendered explicitly
        (Qwen ``<|im_start|>`` and Llama-3 ``<|start_header_id|>``); any
        other template returns None so callers can fall back to raw text.
        """
        if not self.chat_template:
            return None  # type: ignore[return-value]
        if "<|im_start|>" in self.chat_template:
            default_system: Optional[str] = None
            m = re.search(r"<\|im_start\|>system\n(.*?)<\|im_end\|>", self.chat_template, re.S)
            if m:
                default_system = m.group(1)
            parts: List[str] = []
            if not any(msg.get("role") == "system" for msg in messages) and default_system:
                parts.append(f"<|im_start|>system\n{default_system}<|im_end|>\n")
            for msg in messages:
                parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n")
            return "".join(parts)
        if "<|start_header_id|>" in self.chat_template:
            bos = "<|begin_of_text|>" if "<|begin_of_text|>" in self.chat_template else ""
            parts = [bos]
            for msg in messages:
                parts.append(
                    f"<|start_header_id|>{msg['role']}<|end_header_id|>\n\n"
                    f"{msg['content']}<|eot_id|>"
                )
            if add_generation_prompt:
                parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
            return "".join(parts)
        return None  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _encode_segment(self, seg: str) -> List[int]:
        mapped: List[str] = []
        for ch in seg:
            o = ord(ch)
            if o in self._b2u:
                mapped.append(self._b2u[o])
            else:
                for b in ch.encode("utf-8"):
                    mapped.append(self._b2u[b])
        mapped_text = "".join(mapped)
        ids: List[int] = []
        for token in self._pat.findall(mapped_text):
            pieces = self._bpe(token)
            for piece in pieces:
                ids.extend(self._piece_to_ids(piece))
        return ids

    def _piece_to_ids(self, piece: str) -> List[int]:
        if piece in self._vocab:
            return [self._vocab[piece]]
        if self.unknown_token_id is not None:
            return [self.unknown_token_id]
        ids: List[int] = []
        for ch in piece:
            b = self._rev.get(ch)
            if b is not None and b in self._byte_to_id:
                ids.append(self._byte_to_id[b])
            else:
                for bb in ch.encode("utf-8"):
                    if bb in self._byte_to_id:
                        ids.append(self._byte_to_id[bb])
        return ids

    def _bpe(self, token: str) -> Tuple[str, ...]:
        if token in self._bpe_cache:
            return self._bpe_cache[token]
        word = tuple(token)
        pairs = _get_pairs(word)
        if not pairs:
            self._bpe_cache[token] = (token,)
            return (token,)
        while True:
            bigram = min(pairs, key=lambda p: self._rank.get(p, float("inf")))
            if bigram not in self._rank:
                break
            first, second = bigram
            new_word: List[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if (
                    i < len(word) - 1
                    and word[i] == first
                    and word[i + 1] == second
                ):
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        self._bpe_cache[token] = word
        return word