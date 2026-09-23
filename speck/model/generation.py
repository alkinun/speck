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
def generate_tokens(
    model, tokens, *, max_tokens, eos_token_id, device, temperature=0.0, top_k=50, vocab_size=None
):
    """Return up to max_tokens new IDs, excluding EOS, from a nonempty prompt.

    Callers supply an evaluation-mode model on the requested device. Temperature
    zero selects greedy decoding; positive temperatures sample within top_k.
    """

    validate_sampling(max_tokens, temperature, top_k)
    if vocab_size is not None and (
        type(vocab_size) is not int or vocab_size < 1 or not 0 <= eos_token_id < vocab_size
    ):
        raise ValueError("generation vocabulary must be positive and contain EOS")
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
        if vocab_size is not None:
            if vocab_size > logits.size(-1):
                raise ValueError("generation vocabulary exceeds model output rows")
            logits = logits[:, :vocab_size]
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


@torch.inference_mode()
def sample_group(
    model, tokens, *, group, max_tokens, eos_token_id, temperature=1.0, generator=None
):
    """Sample `group` completions of one prompt together, each ending at its first EOS.

    The prompt is prefilled once per row and every row decodes in lockstep; a finished row keeps
    decoding into a discarded tail. Sampling covers the full vocabulary, so at temperature one the
    completions are drawn from the policy itself. Returned completions include their EOS.
    """

    validate_sampling(max_tokens, temperature, 1)
    if temperature == 0 or isinstance(group, bool) or not isinstance(group, int) or group < 1:
        raise ValueError("group sampling needs a positive group and a positive temperature")
    if len(tokens) + max_tokens > model.config.max_position_embeddings:
        raise ValueError("prompt and generated tokens exceed the model context")
    device = next(model.parameters()).device
    state = model.state(batch_size=group, length=len(tokens) + max_tokens, device=device)
    inputs = torch.tensor([tokens] * group, device=device, dtype=torch.long)
    completions = torch.empty((group, 0), dtype=torch.long, device=device)
    finished = torch.zeros(group, dtype=torch.bool, device=device)
    for _ in range(max_tokens):
        logits = model(inputs, state=state, last_token_only=True)[:, -1].float()
        probabilities = torch.softmax(logits / temperature, dim=-1)
        token = torch.multinomial(probabilities, 1, generator=generator).squeeze(-1)
        completions = torch.cat((completions, token[:, None]), dim=1)
        finished |= token == eos_token_id
        if finished.all():
            break
        inputs = token[:, None]
    result = []
    for row in completions.tolist():
        end = row.index(eos_token_id) + 1 if eos_token_id in row else len(row)
        result.append(row[:end])
    return result
