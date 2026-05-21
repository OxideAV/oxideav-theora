# Changelog

All notable changes to `oxideav-theora` are recorded here.

## [Unreleased]

### Added

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
