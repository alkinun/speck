"""Diagnostic whitespace counts; never a production normalization policy."""

import re


def inspect_structure(text, tokenizer):
    if not isinstance(text, str) or not 0 < len(text) <= 2_000_000:
        raise ValueError("structure review requires a bounded nonempty candidate")
    # Preserve newlines and all non-whitespace. This only diagnoses padding sensitivity.
    collapsed = re.sub(r"[^\S\n]+", " ", text)
    tokens = len(tokenizer.encode(collapsed, bos=True, eos=True))
    return {
        "characters": len(text),
        "whitespace_characters": sum(c.isspace() for c in text),
        "whitespace_fraction": sum(c.isspace() for c in text) / len(text),
        "longest_horizontal_whitespace_run": max(
            (m.end() - m.start() for m in re.finditer(r"[^\S\n]+", text)), default=0
        ),
        "pipe_characters": text.count("|"),
        "lines": text.count("\n") + 1,
        "horizontal_whitespace_collapsed_characters": len(collapsed),
        "horizontal_whitespace_collapsed_tokens_including_bos_eos": tokens,
        "collapsed_length_plus_lookahead": {
            str(length): tokens >= length + 1 for length in (32768, 65536, 131072)
        },
    }
