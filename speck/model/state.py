"""Persistent attention, convolution, and recurrent decoding state."""

from collections import defaultdict

import torch


class AttentionState:
    """Maintain a bounded key-value cache in chronological ring-buffer order."""

    def __init__(self, batch_size, kv_heads, capacity, head_dim, device, dtype, storage_dtype=None):
        if capacity < 1:
            raise ValueError("attention state capacity must be positive")
        shape = (batch_size, kv_heads, capacity, head_dim)
        self.compute_dtype = dtype
        self.storage_dtype = storage_dtype or dtype
        if self.storage_dtype not in {torch.float32, torch.float16, torch.bfloat16, torch.int8}:
            raise ValueError("unsupported KV cache dtype")
        self.keys = torch.empty(shape, device=device, dtype=self.storage_dtype)
        self.values = torch.empty(shape, device=device, dtype=self.storage_dtype)
        scale_shape = (batch_size, kv_heads, capacity, 1)
        self.key_scales = (
            torch.empty(scale_shape, device=device, dtype=torch.float16)
            if self.storage_dtype == torch.int8
            else None
        )
        self.value_scales = (
            torch.empty(scale_shape, device=device, dtype=torch.float16)
            if self.storage_dtype == torch.int8
            else None
        )
        self.capacity = capacity
        self.used = 0
        self.write_position = 0

    def current(self):
        if self.used < self.capacity:
            return self._decode(
                self.keys[:, :, : self.used],
                self.values[:, :, : self.used],
                self.key_scales[:, :, : self.used] if self.key_scales is not None else None,
                self.value_scales[:, :, : self.used] if self.value_scales is not None else None,
            )
        if self.write_position == 0:
            return self._decode(self.keys, self.values, self.key_scales, self.value_scales)
        keys = torch.cat(
            (self.keys[:, :, self.write_position :], self.keys[:, :, : self.write_position]),
            dim=2,
        )
        values = torch.cat(
            (self.values[:, :, self.write_position :], self.values[:, :, : self.write_position]),
            dim=2,
        )
        key_scales = (
            torch.cat(
                (
                    self.key_scales[:, :, self.write_position :],
                    self.key_scales[:, :, : self.write_position],
                ),
                dim=2,
            )
            if self.key_scales is not None
            else None
        )
        value_scales = (
            torch.cat(
                (
                    self.value_scales[:, :, self.write_position :],
                    self.value_scales[:, :, : self.write_position],
                ),
                dim=2,
            )
            if self.value_scales is not None
            else None
        )
        return self._decode(keys, values, key_scales, value_scales)

    def _decode(self, keys, values, key_scales, value_scales):
        if self.storage_dtype != torch.int8:
            return keys.to(self.compute_dtype), values.to(self.compute_dtype)
        assert key_scales is not None and value_scales is not None
        return (
            keys.to(self.compute_dtype) * key_scales.to(self.compute_dtype),
            values.to(self.compute_dtype) * value_scales.to(self.compute_dtype),
        )

    def _encode(self, tensor):
        if self.storage_dtype != torch.int8:
            return tensor.to(self.storage_dtype), None
        # Round scales upward to a nonzero representable FP16 value. Rounding
        # downward can saturate the largest value; underflow used to erase a head.
        minimum_scale = torch.finfo(torch.float16).tiny * torch.finfo(torch.float16).eps
        scale = (tensor.float().abs().amax(dim=-1, keepdim=True) / 127).clamp_min(minimum_scale)
        stored_scale = scale.to(torch.float16)
        stored_scale = torch.where(
            stored_scale.float() < scale,
            torch.nextafter(stored_scale, torch.full_like(stored_scale, float("inf"))),
            stored_scale,
        )
        quantized = (tensor.float() / stored_scale.float()).round().clamp(-127, 127).to(torch.int8)
        return quantized, stored_scale

    def append(self, keys, values):
        keys, key_scales = self._encode(keys)
        values, value_scales = self._encode(values)
        length = keys.size(2)
        if length >= self.capacity:
            self.keys.copy_(keys[:, :, -self.capacity :])
            self.values.copy_(values[:, :, -self.capacity :])
            if self.key_scales is not None:
                assert key_scales is not None and value_scales is not None
                self.key_scales.copy_(key_scales[:, :, -self.capacity :])
                self.value_scales.copy_(value_scales[:, :, -self.capacity :])
            self.used = self.capacity
            self.write_position = 0
            return
        first = min(length, self.capacity - self.write_position)
        end = self.write_position + first
        self.keys[:, :, self.write_position : end] = keys[:, :, :first]
        self.values[:, :, self.write_position : end] = values[:, :, :first]
        if self.key_scales is not None:
            assert key_scales is not None and value_scales is not None
            self.key_scales[:, :, self.write_position : end] = key_scales[:, :, :first]
            self.value_scales[:, :, self.write_position : end] = value_scales[:, :, :first]
        remaining = length - first
        if remaining:
            self.keys[:, :, :remaining] = keys[:, :, first:]
            self.values[:, :, :remaining] = values[:, :, first:]
            if self.key_scales is not None:
                self.key_scales[:, :, :remaining] = key_scales[:, :, first:]
                self.value_scales[:, :, :remaining] = value_scales[:, :, first:]
        self.write_position = (self.write_position + length) % self.capacity
        self.used = min(self.capacity, self.used + length)

    def allocated_bytes(self):
        tensors = (self.keys, self.values, self.key_scales, self.value_scales)
        return sum(
            tensor.numel() * tensor.element_size() for tensor in tensors if tensor is not None
        )


class ConvolutionState:
    """Hold causal convolution history for incremental decoding."""

    def __init__(self, batch_size, inner_size, history, device, dtype):
        self.values = torch.zeros(batch_size, inner_size, history, device=device, dtype=dtype)

    def allocated_bytes(self):
        return self.values.numel() * self.values.element_size()


class DeltaNetState:
    """Hold fixed-size recurrent and local-convolution delta-rule state."""

    def __init__(
        self,
        batch_size,
        num_heads,
        key_head_dim,
        value_head_dim,
        conv_dim,
        conv_history,
        device,
        dtype,
        kind="gated_deltanet",
    ):
        self.kind = kind
        self.recurrent = torch.zeros(
            batch_size,
            num_heads,
            key_head_dim,
            value_head_dim,
            device=device,
            dtype=torch.float32,
        )
        self.convolution = torch.zeros(
            batch_size,
            conv_dim,
            conv_history,
            device=device,
            dtype=dtype,
        )

    def reset(self):
        self.recurrent.zero_()
        self.convolution.zero_()

    def allocated_bytes(self):
        tensors = (self.recurrent, self.convolution)
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)


class SequenceState:
    """Track incremental-decoding position and per-operation caches."""

    def __init__(self, entries, length):
        self.entries = entries
        self.position = 0
        self.length = length

    def reset(self):
        self.position = 0
        for entry in self.entries.values():
            if isinstance(entry, AttentionState):
                entry.used = 0
                entry.write_position = 0
            elif isinstance(entry, DeltaNetState):
                entry.reset()
            else:
                entry.values.zero_()

    def allocated_bytes(self):
        return sum(entry.allocated_bytes() for entry in self.entries.values())

    def memory_report(self):
        by_kind = defaultdict(int)
        for entry in self.entries.values():
            if isinstance(entry, AttentionState):
                kind = "attention_kv"
            elif isinstance(entry, DeltaNetState):
                kind = entry.kind
            else:
                kind = "convolution"
            by_kind[kind] += entry.allocated_bytes()
        return {"total_bytes": sum(by_kind.values()), "by_kind": dict(sorted(by_kind.items()))}
