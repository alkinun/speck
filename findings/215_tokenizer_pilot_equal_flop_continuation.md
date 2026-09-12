# 215 — Both custom tokenizer pilots can now reach the equal-FLOP endpoint

The corrected shared continuation adds 68,970 whole documents and 72,024,616 Mistral-reference tokens
after the fixed-document boundary. It preserves 72M total tokens while adapting only the continuation
mixture to measured science availability. The 40,960 pack adds 66,611,628 tokens, leaving 49,189,466
tokens beyond its equal-FLOP stop; exact-32K adds 68,693,132, leaving 11,306,975 beyond its stop.

All source continuity, shared-document, tokenizer identity, minimum-capacity, and shard gates pass. The
first unpublished science-exhaustion stop remains recorded separately. No model run has started and D5
remains unopened.

Artifact: [continuation result](../results/data/tokenizer-pilot-continuation-20260911.json).
