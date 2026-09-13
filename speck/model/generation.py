"""Generate token sequences with the native Speck decoding cache."""

import math

import torch


def validate_sampling(max_tokens, temperature, top_k):
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(temperature)
        or temperature < 0
    ):
        raise ValueError("temperature must be finite and non-negative")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")


@torch.inference_mode()
def generate_tokens(model, tokens, *, max_tokens, eos_token_id, device, temperature=0.0, top_k=50):
    """Return up to max_tokens new IDs, excluding EOS, from a nonempty prompt.

    Callers supply an evaluation-mode model on the requested device. Temperature
    zero selects greedy decoding; positive temperatures sample within top_k.
    """

    validate_sampling(max_tokens, temperature, top_k)
    if not tokens:
        raise ValueError("generation requires a nonempty prompt")
    length = len(tokens) + max_tokens
    if length > model.config.max_position_embeddings:
        raise ValueError("prompt and generated tokens exceed the model context")
    state = model.state(length=length, device=device)
    inputs = torch.tensor([tokens], device=device, dtype=torch.long)
    generated = []
    for _ in range(max_tokens):
        logits = model(inputs, state=state, last_token_only=True)[:, -1]
        if temperature == 0:
            token = logits.argmax(dim=-1)
        else:
            values, indices = torch.topk(logits, min(top_k, logits.size(-1)))
            probabilities = torch.softmax(values / temperature, dim=-1)
            token = indices.gather(-1, torch.multinomial(probabilities, 1)).squeeze(-1)
        token_id = token.item()
        if token_id == eos_token_id:
            break
        generated.append(token_id)
        inputs = token[:, None]
    return generated
