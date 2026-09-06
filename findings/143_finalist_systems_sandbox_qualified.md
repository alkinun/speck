# 143 — Disposable kernel read-only isolation qualifies

## One real, temporary namespace test

A unique transient systemd user service attempted to create a file beside a temporary sentinel under
`ReadOnlyPaths`. Its receipt directory was separately declared with `ReadWritePaths`. The service used
`Type=exec`, `Restart=no`, `--wait`, `--collect`, and `--pipe`; no status loop was used.

On systemd 261 and kernel 7.1.3, the protected write failed with `EROFS`. The sentinel SHA-256 was
identical before and after, the forbidden path was absent, the service returned zero after recording the
expected failure, and a single post-collection lookup returned `not-found`. The temporary directory was
then removed automatically.

Five mocked tests separately bind command construction, path separation, receipt semantics, and service
failure. This upgrades the workload boundary from posthoc mutation detection to a qualified kernel
mechanism on a disposable path.

## Remaining gate

No finalist experiment or checkpoint path was opened or sandboxed. The benchmark engine, GPU access
inside the sandbox, and live checkpoint loading remain unqualified. After the language sequence, one
no-output actual-path preflight is still mandatory before measured systems trials.

## Artifact

- [Sandbox qualification](../results/Speck-Paper1/finalist-systems-sandbox-qualified-v1.json)
