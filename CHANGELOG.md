# Changelog

All notable changes to `oxideav-theora` are recorded here.

## [Unreleased]

### Added

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
