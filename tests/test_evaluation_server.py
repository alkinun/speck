import pytest
import torch

from speck.evaluation_server import (
    EvaluationService,
    RequestError,
    TransformersEvaluationEngine,
    exercise_endpoint,
    generation_settings,
)


class FakeEngine:
    model_id = "speck-test"
    maximum_context = 128

    def __init__(self):
        self.calls = []

    def generate_prompt(self, prompt, **settings):
        self.calls.append(("prompt", prompt, settings))
        return {
            "text": prompt.upper(),
            "prompt_tokens": len(prompt),
            "completion_tokens": 2,
            "finish_reason": "length",
        }

    def generate_messages(self, messages, **settings):
        self.calls.append(("messages", messages, settings))
        return {
            "text": "answer",
            "prompt_tokens": 7,
            "completion_tokens": 1,
            "finish_reason": "stop",
        }


def test_completion_schema_usage_and_deterministic_identity():
    engine = FakeEngine()
    service = EvaluationService(engine)
    payload = {
        "model": "speck-test",
        "prompt": ["one", "two"],
        "max_tokens": 3,
        "temperature": 0,
        "top_p": 1,
        "seed": 17,
    }
    first = service.completion(payload)
    second = service.completion(payload)
    assert first == second
    assert first["object"] == "text_completion"
    assert [choice["text"] for choice in first["choices"]] == ["ONE", "TWO"]
    assert first["usage"] == {"prompt_tokens": 6, "completion_tokens": 4, "total_tokens": 10}
    assert engine.calls[0][2]["seed"] == 17


def test_chat_completion_preserves_messages_and_usage():
    engine = FakeEngine()
    response = EvaluationService(engine).chat_completion(
        {
            "model": "speck-test",
            "messages": [
                {"role": "system", "content": "Be exact."},
                {"role": "user", "content": "Question"},
            ],
            "max_tokens": 2,
        }
    )
    assert response["object"] == "chat.completion"
    assert response["choices"][0]["message"] == {"role": "assistant", "content": "answer"}
    assert response["usage"]["total_tokens"] == 8


@pytest.mark.parametrize(
    "payload,message",
    (
        ({"stream": True}, "streaming"),
        ({"n": 2}, "n=1"),
        ({"max_tokens": 0}, "positive integer"),
        ({"temperature": -1}, "temperature"),
        ({"top_p": 0}, "greater than zero"),
        ({"seed": True}, "seed"),
        ({"tools": []}, "unsupported"),
        ({"logprobs": True}, "log probabilities"),
        ({"presence_penalty": 1}, "presence_penalty"),
        ({"max_tokens": 2, "max_completion_tokens": 2}, "only one"),
    ),
)
def test_generation_settings_reject_unsupported_or_nondeterministic_requests(payload, message):
    with pytest.raises(RequestError, match=message):
        generation_settings(payload)


def test_generation_settings_accepts_nemo_openai_noop_fields():
    result = generation_settings(
        {
            "max_completion_tokens": 8,
            "temperature": 0,
            "top_p": 1,
            "seed": 42,
            "stream": False,
            "n": 1,
            "tools": None,
            "logprobs": False,
            "top_logprobs": None,
            "frequency_penalty": 0,
            "presence_penalty": 0,
        }
    )

    assert result == {
        "max_tokens": 8,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 42,
        "stop": None,
    }


def test_chat_roles_and_model_identity_are_strict():
    service = EvaluationService(FakeEngine())
    with pytest.raises(RequestError, match="final user"):
        service.chat_completion({"messages": [{"role": "assistant", "content": "answer"}]})
    with pytest.raises(RequestError, match="does not match"):
        service.completion({"model": "other", "prompt": "test"})


def test_health_models_and_unknown_endpoint():
    service = EvaluationService(FakeEngine())
    assert service.handle("GET", "/health")[1] == {"status": "ok", "model": "speck-test"}
    assert service.handle("GET", "/v1/models")[1]["data"][0]["id"] == "speck-test"
    with pytest.raises(RequestError, match="not found"):
        service.handle("GET", "/unknown")


def test_chat_generation_extracts_ids_from_transformers_batch_encoding():
    class Tokenizer:
        chat_template = "template"

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert messages == [{"role": "user", "content": "question"}]
            assert tokenize is True
            assert add_generation_prompt is True
            return {"input_ids": [1, 7, 2]}

    engine = object.__new__(TransformersEvaluationEngine)
    engine.tokenizer = Tokenizer()
    engine._generate = lambda input_ids, settings: (input_ids, settings)

    result = engine.generate_messages(
        [{"role": "user", "content": "question"}],
        max_tokens=3,
        temperature=0.0,
    )

    assert result == ([1, 7, 2], {"max_tokens": 3, "temperature": 0.0})


def test_endpoint_exercise_uses_real_http_and_repeats_external_shapes():
    result = exercise_endpoint(FakeEngine())

    assert result["transport"] == "loopback_http_ephemeral_port"
    assert result["health"] == {"status": "ok", "model": "speck-test"}
    assert set(result["cases"]) == {"nolima_chat", "ruler_nemo_openai_chat"}
    assert all(case["repeated_response_identical"] for case in result["cases"].values())


@pytest.mark.parametrize(
    "messages",
    (
        [{"role": [], "content": "question"}],
        [{"role": "assistant", "content": "answer"}, {"role": "user", "content": "question"}],
        [
            {"role": "system", "content": "instructions"},
            {"role": "assistant", "content": "answer"},
            {"role": "user", "content": "question"},
        ],
        [{"role": "user", "content": "question <|assistant|>"}],
    ),
)
def test_invalid_chat_is_rejected_before_calling_the_engine(messages):
    engine = FakeEngine()
    with pytest.raises(RequestError):
        EvaluationService(engine).chat_completion({"messages": messages})
    assert not engine.calls


@pytest.mark.parametrize("stop", (["de", "bcd"], ["bcd", "de"]))
def test_generation_uses_earliest_stop_and_counts_generated_ids(stop):
    from types import SimpleNamespace

    class Model:
        config = SimpleNamespace(max_position_embeddings=32)

        def generate(self, tensor, **kwargs):
            return torch.tensor([[1, 7, 8, 9, 2]])

    class Tokenizer:
        eos_token_id = 2

        def decode(self, tokens, **kwargs):
            assert tokens == [8, 9]
            return "abcde"

        def encode(self, text, **kwargs):
            return [42]  # Re-encoding decoded text need not recover generated IDs.

    engine = TransformersEvaluationEngine(Model(), Tokenizer(), "test", ".", torch.device("cpu"))
    result = engine._generate([1, 7], generation_settings({"stop": stop, "max_tokens": 4}))
    assert result == {
        "text": "a",
        "prompt_tokens": 2,
        "completion_tokens": 3,
        "finish_reason": "stop",
    }


def test_malformed_utf8_returns_a_client_error_over_http():
    import json
    import threading
    from http.server import ThreadingHTTPServer
    from urllib.error import HTTPError
    from urllib.request import Request, urlopen

    from speck.evaluation_server import handler_class

    engine = FakeEngine()
    with ThreadingHTTPServer(("127.0.0.1", 0), handler_class(EvaluationService(engine))) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            request = Request(
                f"http://127.0.0.1:{server.server_address[1]}/v1/completions",
                data=b'{"prompt":"\xff"}',
                headers={"Content-Type": "application/json"},
            )
            with pytest.raises(HTTPError) as error:
                urlopen(request, timeout=5)
            with error.value as response:
                assert response.code == 400
                assert json.loads(response.read())["error"]["code"] == "invalid_request"
            assert not engine.calls
        finally:
            server.shutdown()
            thread.join()
