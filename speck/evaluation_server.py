"""Serve a local Transformers export through a strict OpenAI-compatible evaluation API."""

import hashlib
import json
import math
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen


class RequestError(ValueError):
    """Represent a client-visible evaluation request error."""

    def __init__(self, message, *, code="invalid_request", status=400):
        super().__init__(message)
        self.code = code
        self.status = status


def _number(value, name, minimum, maximum=None):
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        interval = f"[{minimum}, {maximum}]" if maximum is not None else f"[{minimum}, infinity)"
        raise RequestError(f"{name} must be a finite number in {interval}")
    return value


def generation_settings(payload):
    """Resolve the deliberately small deterministic generation surface."""

    if payload.get("stream", False):
        raise RequestError("streaming is not supported by the evaluation server")
    if payload.get("n", 1) != 1:
        raise RequestError("the evaluation server requires n=1")
    unsupported = sorted(
        field
        for field in ("tools", "tool_choice", "functions")
        if field in payload and payload[field] is not None
    )
    if unsupported:
        raise RequestError(f"unsupported evaluation fields: {', '.join(unsupported)}")
    if payload.get("logprobs") not in (None, False) or payload.get("top_logprobs") is not None:
        raise RequestError("log probabilities are not supported by the evaluation server")
    for field, expected in (
        ("frequency_penalty", 0.0),
        ("presence_penalty", 0.0),
    ):
        if field in payload and (isinstance(payload[field], bool) or payload[field] != expected):
            raise RequestError(f"{field} must be {expected:g} for evaluation")
    if "max_tokens" in payload and "max_completion_tokens" in payload:
        raise RequestError("provide only one completion-token limit")
    max_tokens = payload.get("max_tokens", payload.get("max_completion_tokens", 16))
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens < 1:
        raise RequestError("max_tokens must be a positive integer")
    temperature = _number(payload.get("temperature", 0.0), "temperature", 0.0)
    top_p = _number(payload.get("top_p", 1.0), "top_p", 0.0, 1.0)
    if top_p == 0:
        raise RequestError("top_p must be greater than zero")
    seed = payload.get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise RequestError("seed must be an integer")
    stop = payload.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    if stop is not None and (
        not isinstance(stop, list)
        or not stop
        or len(stop) > 4
        or any(not isinstance(value, str) or not value for value in stop)
    ):
        raise RequestError("stop must be a non-empty string or up to four non-empty strings")
    return {
        "max_tokens": max_tokens,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "seed": seed,
        "stop": stop,
    }


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise RequestError("messages must be a non-empty list")
    previous = None
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or set(message) != {"role", "content"}:
            raise RequestError("each message must contain only role and content")
        role, content = message["role"], message["content"]
        if role not in {"system", "user", "assistant"}:
            raise RequestError(f"unsupported message role: {role!r}")
        if not isinstance(content, str) or not content:
            raise RequestError("message content must be a non-empty string")
        if role == "system" and index != 0:
            raise RequestError("system message must be first")
        if role != "system" and previous == role:
            raise RequestError("user and assistant roles must alternate")
        previous = role
    if messages[-1]["role"] != "user":
        raise RequestError("chat completion requires a final user message")
    return messages


def _response_id(kind, model, content, prompt_tokens, completion_tokens):
    payload = f"{kind}\0{model}\0{content}\0{prompt_tokens}\0{completion_tokens}".encode()
    return f"speck-{hashlib.sha256(payload).hexdigest()[:24]}"


class EvaluationService:
    """Translate strict OpenAI request subsets to one serialized generation engine."""

    def __init__(self, engine):
        self.engine = engine

    def models(self):
        return {
            "object": "list",
            "data": [
                {
                    "id": self.engine.model_id,
                    "object": "model",
                    "created": 0,
                    "owned_by": "specklabs",
                }
            ],
        }

    def completion(self, payload):
        if not isinstance(payload, dict):
            raise RequestError("request body must be a JSON object")
        requested_model = payload.get("model", self.engine.model_id)
        if requested_model != self.engine.model_id:
            raise RequestError("requested model does not match the loaded Speck export")
        prompt = payload.get("prompt")
        prompts = [prompt] if isinstance(prompt, str) else prompt
        if (
            not isinstance(prompts, list)
            or not prompts
            or any(not isinstance(value, str) or not value for value in prompts)
        ):
            raise RequestError("prompt must be a non-empty string or list of non-empty strings")
        settings = generation_settings(payload)
        outputs = [self.engine.generate_prompt(value, **settings) for value in prompts]
        prompt_tokens = sum(output["prompt_tokens"] for output in outputs)
        completion_tokens = sum(output["completion_tokens"] for output in outputs)
        joined = "\0".join(output["text"] for output in outputs)
        return {
            "id": _response_id(
                "completion", self.engine.model_id, joined, prompt_tokens, completion_tokens
            ),
            "object": "text_completion",
            "created": 0,
            "model": self.engine.model_id,
            "choices": [
                {
                    "index": index,
                    "text": output["text"],
                    "finish_reason": output["finish_reason"],
                    "logprobs": None,
                }
                for index, output in enumerate(outputs)
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }

    def chat_completion(self, payload):
        if not isinstance(payload, dict):
            raise RequestError("request body must be a JSON object")
        requested_model = payload.get("model", self.engine.model_id)
        if requested_model != self.engine.model_id:
            raise RequestError("requested model does not match the loaded Speck export")
        messages = validate_messages(payload.get("messages"))
        settings = generation_settings(payload)
        output = self.engine.generate_messages(messages, **settings)
        return {
            "id": _response_id(
                "chat",
                self.engine.model_id,
                output["text"],
                output["prompt_tokens"],
                output["completion_tokens"],
            ),
            "object": "chat.completion",
            "created": 0,
            "model": self.engine.model_id,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output["text"]},
                    "finish_reason": output["finish_reason"],
                }
            ],
            "usage": {
                "prompt_tokens": output["prompt_tokens"],
                "completion_tokens": output["completion_tokens"],
                "total_tokens": output["prompt_tokens"] + output["completion_tokens"],
            },
        }

    def handle(self, method, path, payload=None):
        if method == "GET" and path in {"/health", "/healthz"}:
            return 200, {"status": "ok", "model": self.engine.model_id}
        if method == "GET" and path == "/v1/models":
            return 200, self.models()
        if method == "POST" and path == "/v1/completions":
            return 200, self.completion(payload)
        if method == "POST" and path == "/v1/chat/completions":
            return 200, self.chat_completion(payload)
        raise RequestError("endpoint not found", code="not_found", status=404)


class TransformersEvaluationEngine:
    """Run deterministic local generation without Accelerate or a serving framework."""

    def __init__(self, model, tokenizer, model_id, export_dir, device):
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.export_dir = str(export_dir)
        self.device = device
        self.maximum_context = model.config.max_position_embeddings
        self.lock = threading.Lock()

    @classmethod
    def load(cls, export_dir, *, device="cpu", dtype="float32", require_attestation=True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        export_dir = Path(export_dir).expanduser().resolve()
        if not export_dir.is_dir():
            raise ValueError(f"Transformers export does not exist: {export_dir}")
        attestation_path = export_dir / "speck_parity.json"
        if require_attestation:
            if not attestation_path.is_file():
                raise ValueError("Speck export is missing speck_parity.json")
            attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
            if attestation.get("format") != "speck_export_parity" or not attestation.get("passed"):
                raise ValueError("Speck export parity attestation is invalid")
            config = json.loads((export_dir / "config.json").read_text(encoding="utf-8"))
            if attestation.get("parameters") != config.get("expected_parameters"):
                raise ValueError("Speck export parity parameter count does not match config")
        dtypes = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        if dtype not in dtypes:
            raise ValueError("evaluation server dtype must be float32, bfloat16, or float16")
        tokenizer = AutoTokenizer.from_pretrained(
            export_dir,
            trust_remote_code=True,
            local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            export_dir,
            trust_remote_code=True,
            local_files_only=True,
            dtype=dtypes[dtype],
        ).to(device)
        model.eval()
        return cls(model, tokenizer, export_dir.name, export_dir, torch.device(device))

    def _generate(self, input_ids, settings):
        import torch

        if len(input_ids) + settings["max_tokens"] > self.maximum_context:
            raise RequestError("prompt and completion exceed the model context")
        with self.lock, torch.inference_mode():
            torch.manual_seed(settings["seed"])
            tensor = torch.tensor([input_ids], device=self.device)
            kwargs = {
                "max_new_tokens": settings["max_tokens"],
                "do_sample": settings["temperature"] > 0,
                "use_cache": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            if settings["temperature"] > 0:
                kwargs.update(temperature=settings["temperature"], top_p=settings["top_p"])
            sequence = self.model.generate(tensor, **kwargs)[0]
        generated = sequence[len(input_ids) :].tolist()
        finish_reason = (
            "stop" if generated and generated[-1] == self.tokenizer.eos_token_id else "length"
        )
        if generated and generated[-1] == self.tokenizer.eos_token_id:
            generated = generated[:-1]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        for stop in settings["stop"] or ():
            if stop in text:
                text = text.split(stop, 1)[0]
                finish_reason = "stop"
        completion_tokens = len(self.tokenizer.encode(text, add_special_tokens=False))
        return {
            "text": text,
            "prompt_tokens": len(input_ids),
            "completion_tokens": completion_tokens,
            "finish_reason": finish_reason,
        }

    def generate_prompt(self, prompt, **settings):
        input_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        return self._generate(input_ids, settings)

    def generate_messages(self, messages, **settings):
        if not getattr(self.tokenizer, "chat_template", None):
            raise RequestError(
                "chat completions require an instruction export with a chat template"
            )
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        if hasattr(input_ids, "get"):
            input_ids = input_ids.get("input_ids")
        if not isinstance(input_ids, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in input_ids
        ):
            raise RuntimeError("chat template did not return one token-id sequence")
        return self._generate(input_ids, settings)


def _error(error):
    return {
        "error": {
            "message": str(error),
            "type": "invalid_request_error",
            "param": None,
            "code": error.code,
        }
    }


def handler_class(service, maximum_request_bytes=16 * 1024 * 1024):
    class Handler(BaseHTTPRequestHandler):
        server_version = "SpeckEvaluationServer/1"

        def log_message(self, format, *args):
            return None

        def respond(self, status, payload):
            data = json.dumps(payload, separators=(",", ":")).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def dispatch(self, method):
            try:
                payload = None
                if method == "POST":
                    content_type = self.headers.get("Content-Type", "").split(";", 1)[0]
                    if content_type != "application/json":
                        raise RequestError("Content-Type must be application/json")
                    try:
                        length = int(self.headers.get("Content-Length", ""))
                    except ValueError as error:
                        raise RequestError("Content-Length must be an integer") from error
                    if not 0 < length <= maximum_request_bytes:
                        raise RequestError("request body size is invalid")
                    try:
                        payload = json.loads(self.rfile.read(length))
                    except json.JSONDecodeError as error:
                        raise RequestError("request body is not valid JSON") from error
                status, response = service.handle(method, self.path, payload)
            except RequestError as error:
                status, response = error.status, _error(error)
            except Exception:
                traceback.print_exc()
                error = RequestError(
                    "internal evaluation server error", code="internal_error", status=500
                )
                status, response = error.status, _error(error)
            self.respond(status, response)

        def do_GET(self):
            self.dispatch("GET")

        def do_POST(self):
            self.dispatch("POST")

    return Handler


def _local_json_request(port, path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    with urlopen(request, timeout=30) as response:
        return response.status, json.loads(response.read())


def exercise_endpoint(engine):
    """Exercise the external-suite request shapes over a real loopback HTTP socket."""

    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        handler_class(EvaluationService(engine)),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    cases = {
        "nolima_chat": {
            "model": engine.model_id,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "Reply with one word: blue"},
            ],
            "seed": 43,
            "max_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "ruler_nemo_openai_chat": {
            "model": engine.model_id,
            "messages": [{"role": "user", "content": "Reply with one word: blue"}],
            "seed": 42,
            "max_completion_tokens": 8,
            "temperature": 0.0,
            "top_p": 1.0,
            "stream": False,
            "n": 1,
            "tools": None,
            "logprobs": False,
            "top_logprobs": None,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        },
    }
    try:
        health_status, health = _local_json_request(port, "/health")
        models_status, models = _local_json_request(port, "/v1/models")
        results = {}
        for name, payload in cases.items():
            first_status, first = _local_json_request(
                port,
                "/v1/chat/completions",
                payload,
            )
            second_status, second = _local_json_request(
                port,
                "/v1/chat/completions",
                payload,
            )
            if first_status != 200 or second_status != 200 or first != second:
                raise RuntimeError(f"evaluation endpoint qualification failed for {name}")
            if first.get("model") != engine.model_id or first.get("object") != "chat.completion":
                raise RuntimeError(f"evaluation endpoint returned an invalid schema for {name}")
            results[name] = {
                "request": payload,
                "response": first,
                "repeated_response_identical": True,
            }
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    if health_status != 200 or health.get("status") != "ok" or models_status != 200:
        raise RuntimeError("evaluation endpoint health or model discovery failed")
    return {
        "transport": "loopback_http_ephemeral_port",
        "health": health,
        "models": models,
        "cases": results,
    }


def serve(engine, host="127.0.0.1", port=8000):
    """Serve until interrupted; intended only for controlled local evaluation."""

    service = EvaluationService(engine)
    server = ThreadingHTTPServer((host, port), handler_class(service))
    print(
        json.dumps(
            {
                "status": "serving",
                "host": host,
                "port": port,
                "model": engine.model_id,
                "maximum_context": engine.maximum_context,
                "created": int(time.time()),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever()
