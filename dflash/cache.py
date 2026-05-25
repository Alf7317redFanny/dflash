"""KV cache implementations for dflash inference.

Supports both static (pre-allocated) and dynamic (growing) cache strategies,
along with sliding window variants for long-context generation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch


@dataclass
class CacheConfig:
    """Configuration for KV cache behavior."""
    max_seq_len: int = 2048
    num_layers: int = 32
    num_heads: int = 32
    num_kv_heads: int = 32  # for GQA/MQA
    head_dim: int = 128
    dtype: torch.dtype = torch.float16
    device: str = "cuda"
    # sliding window size; None means full attention cache
    window_size: Optional[int] = None


class StaticKVCache:
    """Pre-allocated KV cache for efficient inference.

    Allocates all memory upfront to avoid fragmentation during generation.
    Suitable when max sequence length is known in advance.
    """

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self._seq_len = 0

        shape = (
            config.num_layers,
            config.num_kv_heads,
            config.max_seq_len,
            config.head_dim,
        )
        self.k_cache = torch.zeros(shape, dtype=config.dtype, device=config.device)
        self.v_cache = torch.zeros(shape, dtype=config.dtype, device=config.device)

    @property
    def seq_len(self) -> int:
        return self._seq_len

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write new keys/values and return the full cached sequence.

        Args:
            layer_idx: Which transformer layer is updating the cache.
            key: New key tensor of shape (batch, num_kv_heads, seq, head_dim).
            value: New value tensor matching key shape.

        Returns:
            Tuple of (full_keys, full_values) up to current position.
        """
        bsz, nkv, seq, hdim = key.shape
        end = self._seq_len + seq

        if end > self.config.max_seq_len:
            raise ValueError(
                f"Cache overflow: tried to write position {end} into cache of "
                f"size {self.config.max_seq_len}"
            )

        self.k_cache[layer_idx, :, self._seq_len:end] = key[0]  # assume bsz=1
        self.v_cache[layer_idx, :, self._seq_len:end] = value[0]

        if layer_idx == self.config.num_layers - 1:
            self._seq_len = end

        k_out = self.k_cache[layer_idx, :, :end].unsqueeze(0)
        v_out = self.v_cache[layer_idx, :, :end].unsqueeze(0)
        return k_out, v_out

    def reset(self) -> None:
        """Clear the cache for a new sequence."""
        self._seq_len = 0
        self.k_cache.zero_()
        self.v_cache.zero_()

    def memory_bytes(self) -> int:
        """Return total bytes allocated by this cache."""
        return (self.k_cache.numel() + self.v_cache.numel()) * self.k_cache.element_size()


class SlidingWindowCache(StaticKVCache):
    """KV cache with a fixed-size sliding window.

    Older tokens outside the window are evicted, enabling O(1) memory
    usage for arbitrarily long sequences at the cost of losing distant context.
    """

    def __init__(self, config: CacheConfig) -> None:
        if config.window_size is None:
            raise ValueError("SlidingWindowCache requires window_size to be set.")
        # override max_seq_len to the window size for allocation
        window_config = CacheConfig(
            max_seq_len=config.window_size,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            num_kv_heads=config.num_kv_heads,
            head_dim=config.head_dim,
            dtype=config.dtype,
            device=config.device,
            window_size=config.window_size,
        )
        super().__init__(window_config)
        self._total_seen = 0

    def update(
        self,
        layer_idx: int,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Slide the window and insert new tokens."""
        w = self.config.max_seq_len
        _, _, seq, _ = key.shape

        write_pos = self._total_seen % w
        end_pos = write_pos + seq

        if end_pos <= w:
            self.k_cache[layer_idx, :, write_pos:end_pos] = key[0]
            self.v_cache[layer_idx, :, write_pos:end_pos] = value[0]
        else:
            # wrap around
            first = w - write_pos
            self.k_cache[layer_idx, :, write_pos:] = key[0, :, :first]
            self.v_cache[layer_idx, :, write_pos:] = value[0, :, :first]
            self.k_cache[layer_idx, :, : seq - first] = key[0, :, first:]
            self.v_cache[layer_idx, :, : seq - first] = value[0, :, first:]

        if layer_idx == self.config.num_layers - 1:
            self._total_seen += seq
            self._seq_len = min(self._total_seen, w)

        filled = min(self._total_seen + seq, w)
        k_out = self.k_cache[layer_idx, :, :filled].unsqueeze(0)
        v_out = self.v_cache[layer_idx, :, :filled].unsqueeze(0)
        return k_out, v_out


def build_cache(config: CacheConfig) -> StaticKVCache:
    """Factory: return the appropriate cache type based on config."""
    if config.window_size is not None:
        return SlidingWindowCache(config)
    return StaticKVCache(config)
