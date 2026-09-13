# 213 — The firewall-excluded tokenizer-pilot corpus retains 2.056B reference tokens

The complete 2B calibration corpus was reprocessed with all twelve prepared firewall candidate views
taking precedence. This candidate superset is intentionally larger than the materialized tokenizer,
selection, D5, and E2 partitions, so it excludes every possible firewall record without opening either
sealed audit.

Across 2,481,271 records, the pass retains 2,232,542 and removes 248,718 exact plus 11 verified-near
duplicates. The six retained training outputs contain 2,056,192,176 Mistral-reference tokens, leaving
856,192,176 tokens of headroom above the fixed 1.2B pilot requirement. Cleanup and all input identities
pass. The conservative superset may remove extra otherwise eligible documents; that cost is accepted for
the matched tokenizer pilot and is not presented as a final-corpus yield estimate.

This result authorizes fixed-stream materialization only. It does not authorize the seven model runs,
open D5, select a tokenizer, or authorize flagship training.

Artifact: [pilot corpus result](../results/data/tokenizer-pilot-corpus-20260911.json).
