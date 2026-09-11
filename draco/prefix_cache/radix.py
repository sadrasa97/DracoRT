"""
Prefix Cache (Radix Tree)

Implements automatic prompt prefix caching using a radix tree.
When multiple requests share a common prefix, the KV cache for that
prefix is computed once and reused, avoiding redundant computation.

The radix tree stores token sequences as edge labels. Each node
represents a shared prefix and stores the associated KV cache state.

Usage:
    cache = PrefixCache(max_size=1024)

    # Insert a prompt prefix
    token_ids = [1, 2, 3, 4, 5]
    kv_state = {"key": ..., "value": ...}
    cache.insert(token_ids, kv_state, seq_id=0)

    # Lookup a prefix — returns cached KV if prefix exists
    result = cache.lookup([1, 2, 3, 4, 5, 6, 7])
    # result.matched_length == 5 (first 5 tokens cached)
    # result.kv_state == {"key": ..., "value": ...}
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("draco.prefix_cache")


class RadixNode:
    """A node in the radix tree.

    Each node stores a token sequence (edge label) and optionally
    a cached KV state for the prefix up to this node.
    """

    def __init__(
        self,
        token_ids: Optional[List[int]] = None,
        parent: Optional[RadixNode] = None,
    ):
        self.token_ids: List[int] = token_ids or []
        self.parent: Optional[RadixNode] = parent
        self.children: Dict[int, RadixNode] = {}  # first_token -> child node
        self.kv_state: Optional[Any] = None  # Cached KV state
        self.seq_id: Optional[int] = None  # Associated sequence ID
        self.is_leaf: bool = True
        self.access_count: int = 0  # LRU eviction support

    @property
    def depth(self) -> int:
        """Total token count from root to this node."""
        if self.parent is None:
            return len(self.token_ids)
        return self.parent.depth + len(self.token_ids)

    @property
    def has_kv(self) -> bool:
        return self.kv_state is not None

    def __repr__(self) -> str:
        return (
            f"RadixNode("
            f"tokens={self.token_ids[:5]}{'...' if len(self.token_ids) > 5 else ''}, "
            f"depth={self.depth}, "
            f"children={len(self.children)}, "
            f"has_kv={self.has_kv})"
        )


@dataclass
class LookupResult:
    """Result of a prefix cache lookup."""
    found: bool
    matched_length: int = 0
    kv_state: Optional[Any] = None
    seq_id: Optional[int] = None
    remaining_tokens: Optional[List[int]] = None

    @property
    def has_cache(self) -> bool:
        return self.found and self.kv_state is not None


class PrefixCache:
    """
    Radix-tree based prefix cache for KV cache reuse.

    Provides O(1) amortized prefix lookup for sequences sharing
    common prefixes. Supports LRU eviction when cache is full.
    """

    def __init__(self, max_size: int = 1024):
        """
        Args:
            max_size: Maximum number of cached prefix states.
        """
        self.max_size = max_size
        self._root = RadixNode(token_ids=[])
        self._num_entries = 0
        self._num_hits = 0
        self._num_misses = 0
        # O(1) LRU tracking: maps id(node) -> node, ordered by recency.
        # Avoids a full tree scan + sort on every eviction.
        self._lru: "OrderedDict[int, RadixNode]" = OrderedDict()

    @property
    def root(self) -> RadixNode:
        return self._root

    def insert(
        self,
        token_ids: List[int],
        kv_state: Any,
        seq_id: Optional[int] = None,
    ) -> RadixNode:
        """
        Insert a token sequence with its KV state into the cache.

        Handles radix tree splitting/merging automatically.

        Args:
            token_ids: Token IDs to cache
            kv_state: KV cache state to store
            seq_id: Optional sequence ID

        Returns:
            The node where the KV state was stored
        """
        if not token_ids:
            return self._root

        node = self._insert_tokens(token_ids, self._root)

        had_kv = node.has_kv
        node.kv_state = kv_state
        node.seq_id = seq_id
        node.access_count += 1

        # Only a transition from "no cached state" to "has cached state"
        # is a new entry — re-inserting the same prefix must not inflate
        # the count (that previously caused spurious premature evictions).
        if not had_kv:
            self._num_entries += 1
        self._lru[id(node)] = node
        self._lru.move_to_end(id(node))

        # Evict if over capacity — O(1) via the LRU ordering instead of
        # a full tree scan + sort on every insert.
        while self._num_entries > self.max_size:
            self._evict_lru()

        return node

    def lookup(self, token_ids: List[int]) -> LookupResult:
        """
        Look up the longest matching prefix in the cache.

        Args:
            token_ids: Full token sequence to look up

        Returns:
            LookupResult with matched prefix and remaining tokens
        """
        if not token_ids:
            return LookupResult(found=True, matched_length=0, remaining_tokens=[])

        matched_length, node = self._traverse(token_ids)

        if matched_length == 0:
            self._num_misses += 1
            return LookupResult(
                found=False,
                matched_length=0,
                remaining_tokens=list(token_ids),
            )

        node.access_count += 1
        self._num_hits += 1
        if node.has_kv and id(node) in self._lru:
            self._lru.move_to_end(id(node))

        remaining = token_ids[matched_length:] if matched_length < len(token_ids) else []
        return LookupResult(
            found=True,
            matched_length=matched_length,
            kv_state=node.kv_state,
            seq_id=node.seq_id,
            remaining_tokens=remaining,
        )

    def contains_prefix(self, token_ids: List[int]) -> bool:
        """Check if a prefix exists in the cache."""
        matched, _ = self._traverse(token_ids)
        return matched == len(token_ids)

    def remove(self, token_ids: List[int]) -> bool:
        """Remove a cached prefix entry."""
        if not token_ids:
            return False

        matched, node = self._traverse(token_ids)
        if matched < len(token_ids) or not node.has_kv:
            return False

        node.kv_state = None
        node.seq_id = None
        self._lru.pop(id(node), None)
        self._num_entries = max(0, self._num_entries - 1)
        return True

    def clear(self) -> None:
        """Clear the entire cache."""
        self._root = RadixNode(token_ids=[])
        self._num_entries = 0
        self._lru.clear()

    @property
    def size(self) -> int:
        return self._num_entries

    @property
    def hit_rate(self) -> float:
        total = self._num_hits + self._num_misses
        if total == 0:
            return 0.0
        return self._num_hits / total

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "entries": self._num_entries,
            "max_size": self.max_size,
            "hits": self._num_hits,
            "misses": self._num_misses,
            "hit_rate": self.hit_rate,
        }

    # ------------------------------------------------------------------
    # Internal methods
    # ------------------------------------------------------------------

    def _insert_tokens(self, tokens: List[int], root: RadixNode) -> RadixNode:
        """Insert tokens into the radix tree, splitting nodes as needed."""
        current = root
        offset = 0

        while offset < len(tokens):
            first_token = tokens[offset]

            if first_token not in current.children:
                # Create a new leaf node
                new_node = RadixNode(
                    token_ids=tokens[offset:],
                    parent=current,
                )
                current.children[first_token] = new_node
                new_node._was_new = True
                return new_node

            child = current.children[first_token]
            # Find common prefix between child's tokens and remaining tokens
            common_len = 0
            child_tokens = child.token_ids
            remaining = tokens[offset:]
            while (
                common_len < len(child_tokens)
                and common_len < len(remaining)
                and child_tokens[common_len] == remaining[common_len]
            ):
                common_len += 1

            if common_len == len(child_tokens):
                # Fully consumed child — go deeper
                current = child
                offset += common_len
            elif common_len < len(child_tokens):
                # Split the child node
                split_node = RadixNode(
                    token_ids=child_tokens[:common_len],
                    parent=current,
                )
                current.children[first_token] = split_node

                # Remainder of old child
                remainder = RadixNode(
                    token_ids=child_tokens[common_len:],
                    parent=split_node,
                )
                remainder.children = child.children
                remainder.kv_state = child.kv_state
                remainder.seq_id = child.seq_id
                remainder.access_count = child.access_count

                # `child` is being replaced by `remainder` as the node that
                # owns this cached KV state — carry the LRU entry over so
                # eviction ordering and future move_to_end() calls stay
                # correct (previously this state silently fell out of LRU
                # tracking on every split, making it un-evictable and
                # unrefreshable).
                if id(child) in self._lru:
                    del self._lru[id(child)]
                    self._lru[id(remainder)] = remainder

                split_node.children = {remainder.token_ids[0]: remainder}

                # New tokens after the split
                remaining_after_split = remaining[common_len:]
                if remaining_after_split:
                    new_node = RadixNode(
                        token_ids=remaining_after_split,
                        parent=split_node,
                    )
                    split_node.children[remaining_after_split[0]] = new_node
                    new_node._was_new = True
                    return new_node
                else:
                    split_node._was_new = True
                    return split_node
            else:
                # Exact match, go deeper
                current = child
                offset += common_len

        return current

    def _traverse(self, token_ids: List[int]) -> Tuple[int, RadixNode]:
        """Traverse the tree matching as many tokens as possible.

        Returns (matched_length, last_matching_node).
        """
        current = self._root
        matched = 0
        offset = 0

        while offset < len(token_ids):
            first_token = token_ids[offset]
            if first_token not in current.children:
                break

            child = current.children[first_token]
            child_tokens = child.token_ids

            # Count matching tokens
            common_len = 0
            remaining = token_ids[offset:]
            while (
                common_len < len(child_tokens)
                and common_len < len(remaining)
                and child_tokens[common_len] == remaining[common_len]
            ):
                common_len += 1

            if common_len < len(child_tokens):
                # Partial match — stop here
                matched += common_len
                break

            matched += common_len
            offset += common_len
            current = child

        return matched, current

    def _evict_lru(self) -> None:
        """Evict the least recently used entry.

        Uses the `_lru` OrderedDict (oldest entry at the front) instead of
        walking the whole tree and sorting on every eviction. That made
        eviction O(n log n) in the number of cached entries; this is O(1)
        amortized.
        """
        while self._lru:
            victim_id, victim = next(iter(self._lru.items()))
            del self._lru[victim_id]
            if victim.has_kv:
                victim.kv_state = None
                victim.seq_id = None
                self._num_entries = max(0, self._num_entries - 1)
                logger.debug("Evicted LRU node at depth %d", victim.depth)
                return
        # LRU tracking empty but count says otherwise (shouldn't happen) —
        # fall back to a full scan so we never get permanently stuck.
        candidates = []
        self._collect_leaves(self._root, candidates)
        if not candidates:
            self._num_entries = 0
            return
        candidates.sort(key=lambda n: n.access_count)
        victim = candidates[0]
        victim.kv_state = None
        victim.seq_id = None
        self._num_entries = max(0, self._num_entries - 1)

    def _collect_leaves(self, node: RadixNode, leaves: List[RadixNode]) -> None:
        """Collect all leaf nodes with KV state."""
        if node.has_kv and not node.children:
            leaves.append(node)
        for child in node.children.values():
            self._collect_leaves(child, leaves)

    def __repr__(self) -> str:
        return (
            f"PrefixCache("
            f"entries={self._num_entries}, "
            f"max={self.max_size}, "
            f"hit_rate={self.hit_rate:.1%})"
        )
