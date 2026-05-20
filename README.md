# oxideav-theora

Pure-Rust Theora video codec.

## Status — 2026-05-20

**Orphan-rebuild scaffold.** The crate's prior implementation was
retired under the workspace clean-room policy: provenance for several
data tables and at least one structural decode loop could not be
defended against the "no external library source as reference" rule
that governs every crate in this workspace.

Per workspace policy, the only acceptable response is a full
clean-room re-implementation against the Theora I Specification and
black-box validator binaries. That work has not yet been scheduled.

Every public entry point currently returns `Error::NotImplemented`.

## Planned clean-room sources

The clean-room rebuild will consult only:

* Theora I Specification (Xiph) — the authoritative format spec.
* Black-box invocations of `ffmpeg` (the binary — not its source) and
  the `theoradec` / `theoraenc` CLI binaries as opaque validators.

No external library source — libtheora, Xiph reference source, FFmpeg
theora source, etc. — is permitted as a reference under the workspace
clean-room policy.

## License

MIT. See `LICENSE`.
