# Research notebook

This directory is append-only chronological process memory. It captures context that does not belong in
code documentation or a scientific finding: attempted approaches, discussions, unexpected failures,
open questions, and handoffs.

Use one file per consequential work session or decision:

```text
YYYY-MM-DD-short-topic.md
```

Required template:

```markdown
# YYYY-MM-DD — Short topic

## Context

## Work performed

## Decisions

## Evidence and links

## Open questions

## Next actions
```

Rules:

- Write enough for another researcher to resume the work without chat history.
- Link exact commits, Linear issues, contracts, results, and W&B runs when available.
- Distinguish observed output from interpretation and speculation.
- Do not use notebook prose as experiment authority or paper evidence.
- Correct material errors with a dated successor note; do not silently rewrite historical context.
- Promote stable scientific conclusions into `findings/` and paper-level conclusions into
  `paper/claims.json`.
