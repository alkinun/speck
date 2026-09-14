# FineWiki now covers the proposed reference-background requirement

The [checked result](../../results/data/finewiki-stock-preparation-20260914.json) completes the
[fixed first-shard plan](../flagship/finewiki_stock_preparation_v1.json) under clean implementation
`47c8635`. Its complete pinned Parquet contains **421,456** physical rows and matches the declared
2,510,037,970-byte size and SHA-256.

| Stage | Records |
| --- | ---: |
| Physical rows | 421,456 |
| English reader/length yield | 392,774 |
| After benchmark/security/repetition/Gitleaks filters | 390,564 |
| After reference exclusion and candidate deduplication | **387,313** |

Retained text is **2,137,398,513 UTF-8 bytes** and **582,070,378 tokens**, including BOS/EOS.
The counter's model SHA-256 matches the now-frozen Mistral base tokenizer exactly. This exceeds the
proposed 400M-token reference-background requirement by 182.1M tokens before joint-view losses and
packing. The new stock replaces the older 64.7M measurement for this capacity comparison; it is not
added to that overlapping bank as distinct supply.

Sequential acquisition removes 1,324 benchmark matches, 404 repeated-line records, 475 raw email/IP
records, and seven records affected by eight Gitleaks findings. Exclusion removes 1,129 exact and
2,090 near duplicates within candidates, plus 17 exact and 15 verified-near reference matches. Both
positive controls pass, the reference outputs remain unchanged, and final exact overlap is zero.
All published output, removal-ledger, index, raw-file, plan, and source-use hashes were verified.

## Interruption and measurement boundary

Acquisition completed in **1,570.07 seconds** before the foreground tool was interrupted during
exclusion. The process was confirmed absent before resuming from the original clean revision and
committed checkpoint. Acquisition was reused; the source was not downloaded or filtered again.

The **283.77-second** exclusion measurement covers the resumed invocation only. The **508.01 MiB**
observed WAL peak also covers that invocation; the interrupted invocation's complete timing and WAL
peak were not published. The report's observed-storage gate applies to its retained observations,
not an established whole-run maximum. These measurements are not a clean-versus-resume speed comparison,
a complete preparation elapsed time, or a production forecast. Source-stock completeness and artifact
integrity are verified independently of these missing operating measurements.

Next, bind experiment-specific common-background membership and joint deduplication before final
packing. Other proposed sources, especially peS2o science and the major web/code treatments, still
need source-identical eligible supply.
