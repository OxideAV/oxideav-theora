# oxideav-theora

Pure-Rust Theora video codec — clean-room implementation in progress.

## Status — 2026-05-21

**Identification + comment + setup-entrypoint parsers (rounds 1–3).**
§6.1, §6.2 (identification), §6.3 (comment), and §6.4.5 step 1
(setup-header common-header guard) of the Theora I Specification are
wired up. Three entry points plus an MSb-first bit reader:

* [`decode_identification_header`] — typed `TheoraIdentHeader` per
  Figure 6.2 (round 1).
* [`parse_comment_header`] — typed `TheoraCommentHeader` per
  §6.3.1 / §6.3.2 / §6.3.3 (round 2). 7-byte `0x81`+"theora" sync,
  4-octet **little-endian** vendor length (§6.3.1's
  Vorbis-compatible memory layout), UTF-8 vendor string, 4-octet LE
  `NCOMMENTS`, then a length-prefixed `KEY=value` vector per comment.
  Case-insensitive `lookup("encoder")` helper exposed per §6.3.3.
* [`parse_setup_header`] — round 3 entrypoint validating §6.4.5
  step 1 (the `0x82`+"theora" common-header guard). Returns
  `Error::SetupHeaderBodyNotImplemented` after the common header
  passes — see "Round 3 spec gap" below.

### Identification header (round 1)
The typed `TheoraIdentHeader` exposes every field from Figure 6.2:

* 7-byte sync (`0x80` + ASCII `"theora"`).
* `VMAJ` / `VMIN` / `VREV` — version triple. The parser enforces
  `VMAJ=3`, `VMIN=2` and accepts any `VREV` (forward-compatible
  per §6.2 step 4).
* `FMBW` / `FMBH` — coded frame size in macroblocks (× 16 → pixels).
* `PICW` / `PICH` / `PICX` / `PICY` — visible picture region.
  `PICY` is stored in the spec's lower-left convention; consumers
  that need top-left flip via `coded_h − pich − picy`.
* `FRN` / `FRD` — frame-rate fraction (both must be non-zero).
* `PARN` / `PARD` — pixel-aspect-ratio fraction (zero/zero = "not
  declared").
* `CS` — color space (Table 6.3): `Undefined`, `Rec470M`,
  `Rec470Bg`, or `Reserved(u8)` for values 3..=255.
* `NOMBR` — nominal bitrate hint (24-bit; saturated value
  `2^24-1` means "≥ 2^24-1 bps").
* `QUAL`, `KFGSHIFT`, `PF` — extracted from the packed final 16-bit
  field. `PF=1` (reserved) and non-zero reserved trailing bits are
  rejected per §6.2 steps 19–20.

The parser also derives `NSBS`, `NBS`, `NMBS` per Tables 6.5 / 6.6,
and `coded_width()` / `coded_height()` in pixels.

Verified against three fixtures from `docs/video/theora/fixtures/`:
* `tiny-i-only-16x16` — coded 32×32 with `nsbs=3`, `nbs=24`.
* `picture-region-non-mb-aligned` — visible 26×18 / coded 32×32
  with `PICY=14`.
* `dimensions-1080p-very-short` — coded 1920×1088 with
  `nsbs=3060`, `nbs=48960`.

### Comment header (round 2)
The typed `TheoraCommentHeader` carries:

* `vendor: String` — the muxer/encoder name from the first vector
  (e.g. `"Lavf62.13.102"` for libtheora-via-FFmpeg).
* `comments: Vec<(String, String)>` — parsed `KEY=value` user
  comments. A vector lacking `=` is preserved with an empty value
  rather than rejected; §6.3.3 mandates `=` but we stay tolerant of
  real-world streams.
* `lookup(name) -> Option<&str>` — case-insensitive search for the
  first matching key (§6.3.3).

Reject paths: wrong header-type byte (`0x80`/`0x82`/video-data),
bad magic, declared length exceeds packet remaining, invalid UTF-8
on either the vendor string or any individual comment vector.

Verified against the comment-header packet that every fixture
under `docs/video/theora/fixtures/` carries
(`vendor="Lavf62.13.102"`, `comments=[("encoder", "Lavc62.30.100
libtheora")]`).

### Setup header (round 3)
[`parse_setup_header`] implements only §6.4.5 step 1: it validates
the 7-byte common header (`0x82` + ASCII `"theora"`) and returns
[`Error::SetupHeaderBodyNotImplemented`] once the preamble checks
out. The body — `LFLIMS` (§6.4.1), `ACSCALE`/`DCSCALE`/`NBMS`/`BMS`
(§6.4.2), quant ranges (§6.4.2 step 7), and DCT-token Huffman
tables (§6.4.4) — is deferred to subsequent clean-room rounds.

A crate-private MSb-first [`BitReader`] implementing §5.2 is landed
in this round too. The bit reader is held private until the
setup-header body lands; it is the substrate the §6.4.1 / §6.4.2 /
§6.4.4 procedures will consume in round 4+.

### Round 3 spec gap

The spec PDF as published does not contain the numbered procedure
steps for §6.4.1 (Loop Filter Limit Table Decode). Page 50 ends
with "It is decoded as follows:" and page 51 begins immediately
with "VP3 Compatibility" / §6.4.2 — the steps that should bridge
the two pages are absent from the PDF stream we have. By
symmetry with §6.4.2 the procedure is presumably: read 3-bit
`NBITS`+1, then for each `qi` in 0..63 read `NBITS`-bit values into
`LFLIMS[qi]` — but the guardrails prohibit guessing where a spec
gap exists. Round 3 therefore lands only the §6.4.5 step 1 guard
and stops, surfacing `SetupHeaderBodyNotImplemented` to callers.

Resolution path: docs collaborator to recover the §6.4.1 procedure
steps (possibly via a different source revision of the spec) before
round 4. Until then, the entrypoint is callable and verifies the
packet type, but cannot consume the body bits.

63 unit tests cover happy-path parses on all three header types,
every spec-mandated reject path on each, the optional
revision-future-compatible path on the identification header,
truncated packets at every prefix length, UTF-8 multi-byte payloads,
empty vendor / value, zero comments, trailing bytes, per-comment
index error reporting, all six §6.4.5 step 1 outcomes, and the §5.2
`BitReader` (MSb-first byte reads, multi-byte spans, full 32-bit
reads, zero-width reads, mid-field truncation).

No video-data packet decode yet. [`register`] is still a no-op —
`RuntimeContext` integration arrives once the codec can actually
decode a frame.

## Clean-room sources

Only the Xiph Theora I Specification
(`docs/video/theora/Theora.pdf`) and the fixture corpus under
`docs/video/theora/fixtures/` are consulted. No libtheora, no
FFmpeg `vp3.c`, no theora-rs.

Black-box `ffmpeg` and `theoradec` binary invocations are allowed
as opaque validators.

## License

MIT. See `LICENSE`.
