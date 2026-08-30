"""Define the Speck instruction tokenizer and chat serialization contract."""

import hashlib
import json
import os
import shutil
from pathlib import Path

ROLE_TOKENS = {
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
}
RESERVED_TOKENS = ("<s>", "</s>", "<unk>", *ROLE_TOKENS.values())

CHAT_TEMPLATE = """{%- if messages|length == 0 %}
    {{- raise_exception('messages must not be empty') }}
{%- endif %}
{{- bos_token }}
{%- set system_offset = 1 if messages[0]['role'] == 'system' else 0 %}
{%- for message in messages %}
    {%- if message['role'] == 'system' %}
        {%- if not loop.first %}
            {{- raise_exception('system message must be first') }}
        {%- endif %}
    {%- elif message['role'] in ['user', 'assistant'] %}
        {%- set turn_index = loop.index0 - system_offset %}
        {%- if (message['role'] == 'user') != (turn_index % 2 == 0) %}
            {{- raise_exception('conversation roles must alternate user/assistant') }}
        {%- endif %}
    {%- else %}
        {{- raise_exception('unsupported message role') }}
    {%- endif %}
    {%- for reserved_token in ['<s>', '</s>', '<unk>', '<|system|>', '<|user|>', '<|assistant|>'] %}
        {%- if reserved_token in message['content'] %}
            {{- raise_exception('message content contains a reserved chat token') }}
        {%- endif %}
    {%- endfor %}
    {{- '<|' + message['role'] + '|>\n' }}
    {%- if message['role'] == 'assistant' %}
        {% generation %}{{- message['content'] + eos_token }}{% endgeneration %}
    {%- else %}
        {{- message['content'] + eos_token }}
    {%- endif %}
    {{- '\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
    {%- if messages[-1]['role'] != 'user' %}
        {{- raise_exception('generation prompt requires a final user message') }}
    {%- endif %}
    {{- '<|assistant|>\n' }}
{%- endif %}"""


class ChatFormatError(ValueError):
    """Indicate that a conversation cannot satisfy the chat template."""


class ChatTokenizer:
    """Extend the base SentencePiece tokenizer with three fixed role tokens."""

    def __init__(self, base):
        self.base = base
        self.role_ids = {role: base.vocab_size + index for index, role in enumerate(ROLE_TOKENS)}
        self._newline = base.encode("\n")

    @property
    def model_path(self):
        return self.base.model_path

    @property
    def vocab_size(self):
        return self.base.vocab_size + len(ROLE_TOKENS)

    @property
    def bos_id(self):
        return self.base.bos_id

    @property
    def eos_id(self):
        return self.base.eos_id

    @property
    def newline_ids(self):
        return tuple(self._newline)

    def _validate(self, messages, add_generation_prompt):
        if not isinstance(messages, list) or not messages:
            raise ChatFormatError("messages must be a non-empty list")
        if not isinstance(messages[0], dict):
            raise ChatFormatError("each message must be an object")
        offset = int(messages[0].get("role") == "system")
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ChatFormatError("each message must be an object")
            role = message.get("role")
            content = message.get("content")
            if role == "system":
                if index != 0:
                    raise ChatFormatError("system message must be first")
            elif role in {"user", "assistant"}:
                expected = "user" if (index - offset) % 2 == 0 else "assistant"
                if role != expected:
                    raise ChatFormatError(
                        "conversation roles must alternate user/assistant"
                    )
            else:
                raise ChatFormatError(f"unsupported message role: {role!r}")
            if not isinstance(content, str) or not content:
                raise ChatFormatError("message content must be a non-empty string")
            if any(token in content for token in RESERVED_TOKENS):
                raise ChatFormatError("message content contains a reserved chat token")
        if add_generation_prompt and messages[-1]["role"] != "user":
            raise ChatFormatError("generation prompt requires a final user message")

    def encode_messages(self, messages, add_generation_prompt=False):
        """Return template tokens and a mask selecting assistant content and EOS."""

        self._validate(messages, add_generation_prompt)
        tokens = [self.bos_id]
        assistant_mask = [False]
        for message in messages:
            role = message["role"]
            content = self.base.encode("\n" + message["content"])
            supervised = role == "assistant"
            if content[: len(self._newline)] != self._newline:
                raise ChatFormatError("tokenizer does not preserve the chat newline delimiter")
            tokens.append(self.role_ids[role])
            assistant_mask.append(False)
            tokens.extend(content)
            assistant_mask.extend([False] * len(self._newline))
            assistant_mask.extend([supervised] * (len(content) - len(self._newline)))
            tokens.append(self.eos_id)
            assistant_mask.append(supervised)
            tokens.extend(self._newline)
            assistant_mask.extend([False] * len(self._newline))
        if add_generation_prompt:
            tokens.append(self.role_ids["assistant"])
            assistant_mask.append(False)
            tokens.extend(self._newline)
            assistant_mask.extend([False] * len(self._newline))
        return tokens, assistant_mask

    def render(self, messages, add_generation_prompt=False):
        self._validate(messages, add_generation_prompt)
        parts = ["<s>"]
        for message in messages:
            parts.append(f"{ROLE_TOKENS[message['role']]}\n{message['content']}</s>\n")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "".join(parts)

    def decode(self, tokens, skip_special_tokens=False):
        if skip_special_tokens:
            tokens = [token for token in tokens if token < self.base.vocab_size]
            return self.base.decode(tokens)
        output = []
        ordinary = []
        role_by_id = {token_id: ROLE_TOKENS[role] for role, token_id in self.role_ids.items()}
        for token in tokens:
            if token in role_by_id:
                if ordinary:
                    output.append(self.base.decode(ordinary))
                    ordinary = []
                output.append(role_by_id[token])
            else:
                ordinary.append(token)
        if ordinary:
            output.append(self.base.decode(ordinary))
        return "".join(output)

    def metadata(self):
        values = {
            "format_version": 1,
            "base_fingerprint": self.base.fingerprint(),
            "vocab_size": self.vocab_size,
            "bos_token_id": self.bos_id,
            "eos_token_id": self.eos_id,
            "role_tokens": {
                role: {"content": token, "id": self.role_ids[role]}
                for role, token in ROLE_TOKENS.items()
            },
            "chat_template": CHAT_TEMPLATE,
        }
        payload = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
        values["fingerprint"] = hashlib.sha256(payload).hexdigest()
        return values

    def fingerprint(self):
        return self.metadata()["fingerprint"]

    def save_pretrained(self, directory, model_max_length=4096):
        """Write a standard LlamaTokenizer artifact with the Speck chat template."""

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        shutil.copy2(self.model_path, directory / "tokenizer.model")
        added_tokens = {
            str(self.role_ids[role]): {
                "content": token,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True,
            }
            for role, token in ROLE_TOKENS.items()
        }
        tokenizer_config = {
            "add_bos_token": True,
            "add_eos_token": False,
            "added_tokens_decoder": added_tokens,
            "additional_special_tokens": list(ROLE_TOKENS.values()),
            "bos_token": "<s>",
            "chat_template": CHAT_TEMPLATE,
            "clean_up_tokenization_spaces": False,
            "eos_token": "</s>",
            "legacy": True,
            "model_max_length": model_max_length,
            "pad_token": None,
            "split_special_tokens": False,
            "tokenizer_class": "LlamaTokenizer",
            "unk_token": "<unk>",
        }
        special_tokens = {
            "additional_special_tokens": list(ROLE_TOKENS.values()),
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        }
        files = {
            "chat_template.jinja": CHAT_TEMPLATE,
            "special_tokens_map.json": json.dumps(special_tokens, indent=2, sort_keys=True) + "\n",
            "tokenizer_config.json": json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n",
            "tokenizer_metadata.json": json.dumps(self.metadata(), indent=2, sort_keys=True) + "\n",
        }
        for name, content in files.items():
            path = directory / name
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(content, encoding="utf-8")
            os.replace(temporary, path)


def get_chat_tokenizer(**config):
    from speck.tokenizer import get_tokenizer

    return ChatTokenizer(get_tokenizer(**config))
