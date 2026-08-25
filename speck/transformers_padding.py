"""Validate attention masks used by the published Transformers integration."""

import torch


def validate_right_padding(attention_mask, batch_size, sequence_length):
    """Return whether a binary mask contains only trailing padding."""

    if attention_mask is None:
        return False
    if attention_mask.ndim != 2 or attention_mask.shape != (batch_size, sequence_length):
        raise ValueError("attention_mask must match the input shape")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)).item():
        raise ValueError("attention_mask must be binary")
    if not torch.all(attention_mask[:, 0] == 1).item():
        raise ValueError("attention_mask rows must start with a token")
    if torch.any(attention_mask[:, 1:] > attention_mask[:, :-1]).item():
        raise ValueError("Speck supports right padding only")
    return torch.any(attention_mask == 0).item()
