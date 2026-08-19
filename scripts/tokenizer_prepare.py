"""download and verify the pinned mistral tokenizer."""

from speck.tokenizer import prepare


tokenizer = prepare()
print(f"prepared mistral tokenizer with {tokenizer.vocab_size:,} tokens")
