# oxideav-theora

Pure-Rust Theora video codec — clean-room implementation in progress.

## Status — 2026-05-24

**Identification + comment + setup-entrypoint + §6.4.2 quant-params
decode + §6.4.3 quant-matrix compute (rounds 1–6).** §6.1, §6.2
(identification), §6.3 (comment), §6.4.5 step 1 (setup-header
common-header guard), §6.4.2 (Quantization Parameters Decode), and
§6.4.3 (Computing a Quantization Matrix) of the Theora I Specification
are wired up. Three byte-aligned entry points plus a public §6.4.2
bit-level decoder over an MSb-first bit reader; round 4 added the
Appendix B VP3 fallback tables, round 5 added the full §6.4.2 procedure
(ACSCALE / DCSCALE / NBMS / BMS / NQRS / QRSIZES / QRBMIS), round 6
adds §6.4.3 — interpolate a 64-entry quantization matrix for any
`(qti, pli, qi)` selector from those parameters.

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
  passes — see "§6.4.1 spec gap" below.
* `TheoraSetupHeader::vp3_defaults` — round 4 constructor returning
  the Appendix-B-typed `LFLIMS` / `ACSCALE` / `DCSCALE` fallback
  applicable to `version < 0x030200` streams.
* [`decode_quantization_parameters`] — round 5 public §6.4.2 decoder
  producing a typed `QuantizationParameters` from the §6.4.2
  setup-header payload bytes.
* [`compute_quantization_matrix`] — round 6 public §6.4.3 procedure
  interpolating a typed `QuantizationMatrix` (`[u16; 64]`, natural
  order) for a `(qti, pli, qi)` selector from a
  `QuantizationParameters`.

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

### Setup header (rounds 3 + 4 + 5)
[`parse_setup_header`] implements only §6.4.5 step 1: it validates
the 7-byte common header (`0x82` + ASCII `"theora"`) and returns
[`Error::SetupHeaderBodyNotImplemented`] once the preamble checks
out. The end-to-end body decode — which begins with `LFLIMS`
(§6.4.1) per §6.4.5 step 2 — stays blocked on the §6.4.1 spec gap.

Round 5 lands the §6.4.2 (Quantization Parameters Decode) procedure
as a standalone public entry point so it can be exercised
independently of the §6.4.1 gap (§6.4.5 step 3 follows step 2; once
§6.4.1 is recovered, `parse_setup_header` will chain the two on a
shared bit reader):

```rust
pub fn decode_quantization_parameters(bits: &[u8])
    -> Result<QuantizationParameters, Error>;

pub struct QuantizationParameters {
    pub ac_scale:          [u16; 64],       // §6.4.2 steps 1–2 ACSCALE
    pub dc_scale:          [u16; 64],       // §6.4.2 steps 3–4 DCSCALE
    pub num_base_matrices: u16,             // §6.4.2 step 5 NBMS (≤ 384)
    pub base_matrices:     Vec<[u8; 64]>,   // §6.4.2 step 6 BMS
    pub num_quant_ranges:  [[u8; 3]; 2],    // §6.4.2 step 7 NQRS
    pub quant_range_sizes: [[[u8; 63]; 3]; 2],          // QRSIZES
    pub quant_range_base_matrix_indices: [[[u16; 64]; 3]; 2], // QRBMIS
}
```

The procedure transcribes all eight numbered steps of §6.4.2: the
4-bit-NBITS-prefixed AC/DC scale tables, the 9-bit `NBMS` (validated
`≤ 384`), the `NBMS × 64` base matrices, and the per-`(qti, pli)`
quant-range tables. Step 7's copy logic (NEWQR / RPQR set selection)
and the `ilog(NBMS−1)` / `ilog(62−qi)` field widths are implemented
verbatim. Undecodable streams (`NBMS > 384`, a `QRBMIS ≥ NBMS` at
step 7(a)ivC, or quant-range sizes overshooting 63 at step 7(a)ivI)
return typed errors. The remaining setup-header body — `LFLIMS`
(§6.4.1) and DCT-token Huffman tables (§6.4.4) — is deferred.

Round 6 adds §6.4.3 (Computing a Quantization Matrix), which consumes
the §6.4.2 `QuantizationParameters` and does not touch the bitstream,
so it is unblocked by the §6.4.1 gap:

```rust
pub fn compute_quantization_matrix(
    params: &QuantizationParameters,
    qti: usize,  // quantization type (Table 3.1): 0=intra, 1=inter
    pli: usize,  // color plane (Table 2.1): 0=luma, 1=Cb, 2=Cr
    qi:  usize,  // quantization index 0..=63
) -> Result<QuantizationMatrix, Error>;

pub struct QuantizationMatrix {
    pub values: [u16; 64], // §6.4.3 QMAT, natural order, each 1..=4096
}
```

The procedure locates the quant range bracketing `qi` (steps 1–3),
picks the two end-point base matrices `QRBMIS[qri]` / `QRBMIS[qri+1]`
(steps 4–5), linearly interpolates `BM[ci]` with the spec's
`//`-rounded formula (step 6(a)), then applies the Table 6.18 `QMIN`
floor, the DC (`DCSCALE`) vs AC (`ACSCALE`) scale, the `//100` and
`*4` scaling, and the `max(QMIN, min(…, 4096))` clamp (steps 6(b)–6(e))
for each of the 64 coefficients. All operands are non-negative, so the
spec's `//` reduces to ordinary integer division. Out-of-range `qti` /
`pli` / `qi` selectors return typed errors. The remaining setup-header
body — `LFLIMS` (§6.4.1) and DCT-token Huffman tables (§6.4.4) — is
deferred.

`TheoraSetupHeader` exposes the round-4 contract directly:

```rust
pub struct TheoraSetupHeader {
    pub loop_filter_limits: [u8;  64], // §6.4.1 LFLIMS
    pub ac_scale:           [u16; 64], // §6.4.2 ACSCALE
    pub dc_scale:           [u16; 64], // §6.4.2 DCSCALE
}
```

Round 4 ships the VP3 hardcoded tables from Appendix B.2 + B.3 of
`Theora.pdf` as public constants:

* `LFLIMS_VP3: [u8; 64]` — Appendix B.2. Range 0..=30, monotonically
  non-increasing across `qi`.
* `ACSCALE_VP3: [u16; 64]` — Appendix B.3. Range 10..=500,
  monotonically non-increasing across `qi`.
* `DCSCALE_VP3: [u16; 64]` — Appendix B.3. Range 10..=220,
  monotonically non-increasing across `qi`.

`TheoraSetupHeader::vp3_defaults()` returns a `TheoraSetupHeader`
populated from these three constants. Per Appendix B.1 first bullet,
streams declaring `version < 0x030200` use this fallback directly
because they predate the per-stream setup-header overrides.

A crate-private MSb-first [`BitReader`] implementing §5.2 remains
held private (used by the §6.4.1 / §6.4.2 procedures once their
bodies land).

### §6.4.1 spec gap

The spec PDF as published does not contain the numbered procedure
steps for §6.4.1 (Loop Filter Limit Table Decode). Page 50 ends
with "It is decoded as follows:" and page 51 begins immediately
with "VP3 Compatibility" / §6.4.2 — the steps that should bridge
the two pages are absent from the PDF stream we have. The
guardrails prohibit guessing where a spec gap exists, so the
`parse_setup_header` body decode remains blocked.

Round 4 mitigates the blockage for VP3-compatible bitstreams via
`vp3_defaults()`. `version >= 0x030200` streams continue to require
the §6.4.1 / §6.4.2 procedures and remain blocked until the docs
collaborator recovers the missing procedure body.

88 unit tests cover happy-path parses on all three header types,
every spec-mandated reject path on each, the optional
revision-future-compatible path on the identification header,
truncated packets at every prefix length, UTF-8 multi-byte payloads,
empty vendor / value, zero comments, trailing bytes, per-comment
index error reporting, all six §6.4.5 step 1 outcomes, the §5.2
`BitReader` (MSb-first byte reads, multi-byte spans, full 32-bit
reads, zero-width reads, mid-field truncation), monotonicity + spot
values + row-sum re-tally on each of the three Appendix B tables,
and the `vp3_defaults()` constructor. Round 5 adds the §6.4.2 quant-
parameters tests: `ilog` against every spec worked example, a
round-trip of synthesized payloads (single-range and two-range size
sums, NEWQR/RPQR copy variants for both INTRA and INTER planes), the
`NBMS = 384` boundary, and the three undecodable-stream rejects
(`NBMS > 384`, `QRBMIS ≥ NBMS`, quant-range overflow) plus mid-field
truncation. A test-only MSb-first bit writer mirrors `BitReader` to
build the fixtures bit-exactly. Round 6 adds nine §6.4.3 quant-matrix
tests: corner-base-matrix selection at `qi=0` / `qi=63`, midpoint
interpolation within a single range, two-range interpolation plus a
boundary-`qi` consistency check, the Table 6.18 `QMIN` floors (intra
and inter), the 4096 ceiling clamp, direct `qmin_table` values,
out-of-range selector rejects, per-`(qti, pli)` selector-wiring
isolation, and an end-to-end chain that decodes a synthesized §6.4.2
payload and feeds it straight into `compute_quantization_matrix`.

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
