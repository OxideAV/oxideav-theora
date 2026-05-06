# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.7](https://github.com/OxideAV/oxideav-theora/compare/v0.0.6...v0.0.7) - 2026-05-06

### Other

- drop dead `linkme` dep
- scene-change detection, two-pass stats, full token coverage (round 20)
- auto-register via oxideav_core::register! macro (linkme distributed slice)
- unify entry point on register(&mut RuntimeContext) ([#502](https://github.com/OxideAV/oxideav-theora/pull/502))

### Encoder

- **Scene-change detection**: the encoder now computes average MB SAD
  against the previous reference frame before determining the frame type.
  If the SAD exceeds `SCENE_CHANGE_MB_SAD` (2048 per MB) the frame is
  forced to a keyframe regardless of the keyint schedule, so the golden
  reference tracks real scene boundaries. Opt-out via
  `EncoderOptions::scene_change_detect = false`.

- **Two-pass encoding**: new public function `run_first_pass` collects
  per-frame complexity (intra MAD for frame 0; inter luma SAD for
  subsequent frames). Passing the result as `EncoderOptions::two_pass_stats`
  pre-biases the per-frame QI before the normal rate-control loop runs,
  allocating more bits to complex frames and fewer to easy ones.

- **Full DCT token coverage** (spec §7.7.3): the encoder now uses the
  complete Theora 32-token set.
  - Multi-block EOB tokens (tokens 1-6): consecutive coded blocks whose
    entire residual is zero are collapsed into a single multi-EOB token
    instead of one EOB codeword per block. Saves 1-5 Huffman codewords per
    run on low-motion inter frames.
  - Combined zero-run+value tokens (tokens 23-31): patterns such as
    "N zeros then ±1" or "N zeros then ±{2,3}" are encoded in a single
    codeword (tokens 23-31) instead of a separate zero-run token followed
    by a value token. Reduces bit cost on sparse high-frequency residuals.

- Cross-validate: libtheora/ffmpeg decodes outputs at 256×256 with PSNR_Y
  ≥ 35 dB (new `psnr_256x256_libtheora_cross_decode` test).

## [0.0.6](https://github.com/OxideAV/oxideav-theora/compare/v0.0.5...v0.0.6) - 2026-05-04

### Other

- validate frame shape in send_frame, return Error not panic
- promote 4 bit-exact-clean fixtures to BitExact + cargo fmt
- add SHA-only fixture mode + un-ignore 1080p HD
- silence dead-code on Yuv444
- wire docs/video/theora/ fixture corpus as docs_corpus.rs

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
