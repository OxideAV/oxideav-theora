# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Tests

- docs corpus: add SHA-only fixture mode (`evaluate_sha`) and un-ignore the
  1920x1080 HD fixture so the decoder is exercised end-to-end at HD against
  the SHA-256 recorded in `notes.md` without shipping the 6 MB raw `.yuv`.
- docs corpus: promote four bit-exact-clean fixtures to `Tier::BitExact`
  (`tiny-i-only-16x16`, `bitstream-version-3.2.1`,
  `picture-region-non-mb-aligned`, `q-high`) so any regression on the
  intra / picture-region-crop / weak-quant paths now fails CI.

### Encoder

- `send_frame` now validates each frame's plane count, per-plane stride
  and data length against the encoder's configured pixel format and
  dimensions, returning a descriptive `Error::invalid` instead of
  panicking deep inside the encode loop on misshapen input.

## [0.0.5](https://github.com/OxideAV/oxideav-theora/compare/v0.0.4...v0.0.5) - 2026-05-03

### Other

- replace never-match regex with semver_check = false
- migrate to centralized OxideAV/.github reusable workflows
- round 19 — iterative half-pel refinement on all ME paths
- adopt slim VideoFrame shape
- pin release-plz to patch-only bumps

## [0.0.4](https://github.com/OxideAV/oxideav-theora/compare/v0.0.3...v0.0.4) - 2026-04-25

### Other

- drop oxideav-codec/oxideav-container shims, import from oxideav-core
- add simple CBR rate-control loop to encoder
- docs — encoder now emits INTER_MV_FOUR
- add 4-MV round-trip test + mode-count introspection helper
- encoder emits INTER_MV_FOUR when RD beats 1-MV
- drop Cargo.lock — this crate is a library
- bump oxideav-core / oxideav-codec dep examples to "0.1"
- bump to oxideav-core 0.1.1 + codec 0.1.1
- migrate register() to CodecInfo builder
- bump oxideav-core + oxideav-codec deps to "0.1"
- claim AVI FourCC via oxideav-codec CodecTag registry
- rewrite README + crate description to match current encode/decode state
