"""Inspect bounded long-document candidates without changing any production filter or stock."""

import hashlib
import heapq


def probe_rows(rows, tokenizer, *, minimum_chars, maximum_chars, sample_size):
    """Census all physical rows; tokenize the largest bounded English candidates only."""
    if any(type(x) is not int for x in (minimum_chars, maximum_chars, sample_size)) or not (
        0 < minimum_chars <= maximum_chars <= 2_000_000 and 1 <= sample_size <= 32
    ):
        raise ValueError("invalid bounded long-document probe")
    count = accepted = over_cap = missing = wrong_language = 0
    maximum = 0
    candidates = []
    for ordinal, row in enumerate(rows):
        count += 1
        text = row.get("text")
        if not isinstance(text, str) or not text:
            missing += 1
            continue
        if row.get("in_language") != "en":
            wrong_language += 1
            continue
        size = len(text)
        maximum = max(maximum, size)
        if size < minimum_chars:
            continue
        accepted += 1
        if size > maximum_chars:
            over_cap += 1
            continue
        key = (size, -ordinal)
        if len(candidates) < sample_size or key > candidates[0][:2]:
            item = (*key, text, row.get("id"))
            if len(candidates) < sample_size:
                heapq.heappush(candidates, item)
            else:
                heapq.heapreplace(candidates, item)
    examples = []
    for size, negative_ordinal, text, identifier in sorted(candidates, reverse=True):
        tokens = tokenizer.encode(text, bos=True, eos=True)
        examples.append(
            {
                "physical_row": -negative_ordinal,
                "upstream_id": identifier,
                "characters": size,
                "utf8_bytes": len(text.encode()),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "tokens_including_bos_eos": len(tokens),
                "fits_intact_context_plus_lookahead": {
                    str(length): len(tokens) >= length + 1 for length in (32768, 65536, 131072)
                },
            }
        )
    return {
        "physical_rows": count,
        "missing_or_empty_text_rows": missing,
        "non_english_metadata_rows": wrong_language,
        "maximum_english_document_characters": maximum,
        "english_rows_at_least_minimum_chars": accepted,
        "english_rows_above_probe_character_cap": over_cap,
        "sample": examples,
        "sampling": "Largest character lengths within the explicit probe range, earliest physical row breaks ties. This is a biased feasibility sample, not a random yield or quality estimate.",
        "qualified_stock_tokens": None,
        "boundary": "English metadata and character bounds only. Security, repetition, family identity, full reference/candidate exclusion and model usefulness have not been qualified. No source text is published or moved into a training bank. Unexamined candidates and over-cap documents remain unqualified.",
    }
