# Changelog

All notable changes to `oxideav-theora` are recorded here.

## [Unreleased]

### Added

* **VP3 hardcoded `LFLIMS` / `ACSCALE` / `DCSCALE` tables (2026-05-22,
  round 4).** Transcribed from Appendix B.2 + B.3 of `Theora.pdf`
  ("The hard-coded loop filter limit values used in VP3 are defined as
  follows: …" / "The hard-coded quantization parameters used by VP3 are
  defined as follows: …") and exposed as public constants:
  * `LFLIMS_VP3: [u8; 64]` — 7-bit loop-filter limits (range 0..=30),
    monotonically non-increasing across `qi`.
  * `ACSCALE_VP3: [u16; 64]` — AC dequantization scale values (range
    10..=500), monotonically non-increasing across `qi`.
  * `DCSCALE_VP3: [u16; 64]` — DC dequantization scale values (range
    10..=220), monotonically non-increasing across `qi`.

  Reshaped `TheoraSetupHeader` to the round-4 contract:
  `{ loop_filter_limits: [u8; 64], ac_scale: [u16; 64],
  dc_scale: [u16; 64] }` (dropping the round-3
  `Option<[u8; 64]>` / `Vec<[u8; 64]>` placeholders). A new
  `TheoraSetupHeader::vp3_defaults()` constructor returns the
  Appendix-B-typed fallback for `version < 0x030200` streams (per
  Appendix B.1 first bullet — VP3-compatible decode). Base matrices,
  NQRS / QRSIZES / QRBMIS, and the Huffman tables are deferred to
  round 5.

  5 new tests bring the total from 63 to 68: transcription-fidelity
  monotonicity + spot-value asserts on each of the three Appendix B
  tables, an independent row-sum re-tally, plus the
  `vp3_defaults()` round-trip.

  **§6.4.1 spec gap still unresolved.** `parse_setup_header` continues
  to surface `Error::SetupHeaderBodyNotImplemented` for the per-stream
  decode path: the numbered procedure body for §6.4.1 ("It is decoded
  as follows:") remains absent from the spec PDF (page 50 ends with
  the sentence, page 51 begins with "VP3 Compatibility" / §6.4.2).
  Round 4's Appendix B fallback applies cleanly to streams that
  predate the per-stream tables (`vp3-compat-decode` fixture); a
  later round will populate the body once the docs collaborator
  recovers the §6.4.1 procedure steps.

* **Setup-header entrypoint + MSb-first bit reader (2026-05-21, round 3).**
  `parse_setup_header` returns a typed `TheoraSetupHeader` whose
  body fields (`loop_filter_limits`, `base_matrices`) are reserved
  placeholders, and surfaces a new
  `Error::SetupHeaderBodyNotImplemented` once §6.4.5 step 1 (the
  `0x82`+"theora" common-header guard from §6.1) succeeds. Reject
  paths cover wrong header type (`0x80` / `0x81` / video-data),
  bad magic, and truncation at every prefix.
  A crate-private MSb-first `BitReader` implementing §5.2 is also
  landed, ready for the §6.4.1 / §6.4.2 / §6.4.4 setup-body decode
  procedures that subsequent rounds will plug into the entrypoint.
  17 new tests bring the total from 46 to 63.

  **Known spec gap — §6.4.1.** The numbered procedure steps for
  §6.4.1 (Loop Filter Limit Table Decode) are absent from the
  current spec PDF: page 50 ends with "It is decoded as follows:"
  and page 51 begins immediately with "VP3 Compatibility" / §6.4.2.
  Round 3 declines to guess and surfaces
  `SetupHeaderBodyNotImplemented` until the docs collaborator
  recovers the procedure body. Round 4 onward will populate the
  setup-header body fields once the gap is closed.

* **Comment-header parser (2026-05-21, round 2).**
  `parse_comment_header` returns a typed `TheoraCommentHeader`
  (`vendor: String`, `comments: Vec<(String, String)>`) per §6.3.1
  / §6.3.2 / §6.3.3 of the Xiph Theora I Specification. Handles the
  Vorbis-compatible 4-octet **little-endian** length encoding from
  §6.3.1 ("on platforms with 8-bit bytes, the memory organization
  of the comment header is identical with that of Vorbis I"), the
  `0x81`+"theora" common-header guard, UTF-8 decoding for both the
  vendor string and each user-comment vector, and `KEY=value`
  splitting per §6.3.3. A `lookup(&str) -> Option<&str>` helper
  performs the case-insensitive key lookup §6.3.3 mandates.
  Reject paths cover wrong header type (`0x80` / `0x82` /
  video-data), bad magic, declared length exceeds packet remaining,
  invalid UTF-8 on either the vendor or any individual comment
  vector (with per-comment index reporting via `CommentField`), and
  truncation at every prefix. Verified against the comment header
  carried by every fixture under `docs/video/theora/fixtures/`
  (vendor `"Lavf62.13.102"`, one comment `encoder=Lavc62.30.100
  libtheora`). 21 new tests bring the total from 25 to 46.

* **Identification-header parser (2026-05-21).**
  `decode_identification_header` returns a typed `TheoraIdentHeader`
  per §6.1 + §6.2 of the Xiph Theora I Specification: 7-byte
  `0x80`+"theora" sync, `VMAJ`/`VMIN`/`VREV` (forward-compatible on
  `VREV`), `FMBW`/`FMBH`, 24-bit `PICW`/`PICH` with `PICX`/`PICY`
  offsets (lower-left convention preserved), `FRN`/`FRD`,
  `PARN`/`PARD`, `CS` (typed `ColorSpace` enum with `Reserved` catch-
  all), `NOMBR`, and the packed
  `QUAL`/`KFGSHIFT`/`PF`/reserved trailer. `PF=1` and non-zero
  reserved bits are rejected. Derived counts `NSBS`/`NBS`/`NMBS`
  (Tables 6.5/6.6) plus `coded_width()`/`coded_height()` exposed on
  the typed header. 25 tests validate against the
  `tiny-i-only-16x16`, `picture-region-non-mb-aligned`, and
  `dimensions-1080p-very-short` fixtures, plus all spec reject
  paths.

### Changed

* **Orphan rebuild (2026-05-20).** The crate was reset to a clean-room
  scaffold. The prior implementation contained module-level docstrings
  and inline comments whose provenance could not be defended against
  the workspace clean-room rule. Orphan-master rebuild per workspace
  policy; no `old` branch retained.

  Every public API path returned `Error::NotImplemented` after the
  reset. The 2026-05-21 entry above is the first round of the
  clean-room rebuild.
