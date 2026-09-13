# Research archive

This is the preserved research collection immediately before the repository reorganization.
The original checkout is commit `e3ed10c` (`precleanup-2026-09-13`).

## Browse

| Collection | Contents |
| --- | --- |
| [Findings](pregrant-history/findings/ARCHIVE.md) | All 217 numbered findings, including failures and negative results |
| [Research](pregrant-history/research/README.md) | Original program, protocols, and contract versions |
| [Experiments](pregrant-history/experiments/README.md) | Completed screens, pilots, and release recipes |
| [Results](pregrant-history/results/README.md) | Original checked measurements and qualification records |
| [Notebook](pregrant-history/research/notebook/README.md) | Original attempts, decisions, and handoffs |
| [Previous cleanup](pregrant-history/docs/code_cleanup_2026-09-08.md) | Earlier engineering audits |

The [manifest](manifest.json) lists original paths and SHA-256 identities for **every tracked file**
at the preserved revision. Non-Python artifacts are readable here with their original relative tree
structure. Original Python sources and tests are retained in Git at that revision.

## Verify and reproduce

```bash
python -m scripts.archive check
python -m scripts.archive restore /path/to/precleanup-checkout
```

Restore creates a detached Git worktree containing the complete original source, lockfile, tests,
and research paths. Install its environment from that checkout's lockfile (`uv sync` with the
appropriate extras/groups); use that environment when running the historical commands. Use a result's
producing revision when reproducing work older than this snapshot.
The tokenizer pilot can continue from the preserved snapshot without changing its implementation
authorization. It still requires its original external datasets and checkpoint volume.

Historical paths inside JSON and Markdown retain their original meaning. Some findings reference
tooling removed before this snapshot: the retired Paper 1 tools were removed in `7322c95` and can
be inspected in `7322c95^` or the relevant producing revision. Use
`python -m scripts.archive locate ORIGINAL_PATH` to resolve a preserved path or removed tool.

Normal formatting, linting, and focused searches exclude this directory. Archive integrity is a
separate check included in `make quality`. Do not edit archived bytes to repair a link or change a
conclusion; update the index or add a correction to the current research record.
