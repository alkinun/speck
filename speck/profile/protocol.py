"""request-level model profiling with growing decode state."""

import time

from speck.profile.schema import ProfileResult
from speck.profile.stats import summarize


def _milliseconds(started):
    return (time.perf_counter() - started) * 1000


def _request(session, scenario, prompt, generated):
    state = session.allocate_state(
        scenario.batch_size,
        scenario.prompt_tokens + scenario.generated_tokens,
        scenario.cache_dtype,
    )
    session.synchronize()
    request_started = time.perf_counter()
    started = time.perf_counter()
    session.prefill(prompt, state)
    session.synchronize()
    prefill = _milliseconds(started)
    decode = []
    for index in range(scenario.generated_tokens):
        started = time.perf_counter()
        session.decode(generated[:, index:index + 1], state)
        session.synchronize()
        decode.append(_milliseconds(started))
    request = _milliseconds(request_started)
    return prefill, decode, request, state.allocated_bytes()


def profile_session(session, scenario, architecture_digest, prompt, generated, weight_bytes):
    if prompt.shape != (scenario.batch_size, scenario.prompt_tokens):
        raise ValueError("profile prompt shape does not match the scenario")
    if generated.shape != (scenario.batch_size, scenario.generated_tokens):
        raise ValueError("profile decode shape does not match the scenario")
    for _ in range(scenario.warmup_requests):
        _request(session, scenario, prompt, generated)
    session.reset_peak_memory()
    prefills = []
    first_decode = []
    decode = []
    requests = []
    state_bytes = None
    for _ in range(scenario.measured_requests):
        prefill, decoded, request, allocated = _request(
            session,
            scenario,
            prompt,
            generated,
        )
        prefills.append(prefill)
        first_decode.append(decoded[0])
        decode.extend(decoded[1:] or decoded)
        requests.append(request)
        if state_bytes is None:
            state_bytes = allocated
        elif state_bytes != allocated:
            raise ValueError("profile state allocation changed between requests")
    return ProfileResult(
        scenario_digest=scenario.digest,
        architecture_digest=architecture_digest,
        model_prefill_ms=summarize(prefills),
        first_decode_ms=summarize(first_decode),
        decode_ms=summarize(decode),
        request_ms=summarize(requests),
        weight_bytes=weight_bytes,
        state_bytes=state_bytes or 0,
        peak_memory_bytes=session.peak_memory_bytes(),
    )
