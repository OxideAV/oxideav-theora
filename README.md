# oxideav-theora

Pure-Rust Theora video codec — clean-room implementation in progress.

## Status — 2026-05-25

**Identification + comment + setup-entrypoint + §6.4.2 quant-params
decode + §6.4.3 quant-matrix compute + §6.4.4 DCT-token Huffman tables
+ §7.1 frame-header decode + §7.2 long-/short-run bit strings
(rounds 1–9).** §6.1, §6.2 (identification), §6.3 (comment), §6.4.5
step 1 (setup-header common-header guard), §6.4.2 (Quantization
Parameters Decode), §6.4.3 (Computing a Quantization Matrix), §6.4.4
(DCT Token Huffman Tables), §7.1 (Frame Header Decode), and §7.2
(Run-Length Encoded Bit Strings) of the Theora I Specification are
wired up. Three byte-aligned entry points plus six public bit-level
decoders over an MSb-first bit reader; round 4 added the Appendix B
VP3 fallback tables, round 5 added the full §6.4.2 procedure (ACSCALE
/ DCSCALE / NBMS / BMS / NQRS / QRSIZES / QRBMIS), round 6 added
§6.4.3 (64-entry interpolated quantization matrix per `(qti, pli, qi)`
selector), round 7 added §6.4.4 — decode the 80-element array of
binary-tree Huffman tables that §7.7 will use to decode DCT-residual
tokens, round 8 added §7.1 (the typed `TheoraFrameHeader` from the
start of a video-data packet), round 9 adds §7.2 — the long-run and
short-run bit-string decoders that §7.3 (coded-block flags) and §7.6
(block-level `qi` values) will consume against the §5.2 bit reader.

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
* [`decode_dct_token_huffman_tables`] — round 7 public §6.4.4 procedure
  returning a `Box<[HuffmanTable; 80]>` of typed
  `HuffmanTable { entries, .. }` decoded from the setup-header's
  binary-tree description. Each table carries up to 32
  `HuffmanEntry { code, len, token }` leaves; `HuffmanTable::lookup`
  resolves a code back to its token. Implementation is iterative
  (explicit DFS stack) per the spec's own §6.4.4 recursion-depth
  caveat.
* [`decode_frame_header`] — round 8 public §7.1 procedure returning a
  typed `TheoraFrameHeader { ftype: FrameType, qis: Vec<u8> }` from the
  start of any video-data packet. Validates the leading 0-bit data-
  packet marker (step 1), the `FTYPE` field (step 2; first-frame Intra
  check), unrolls the `MOREQIS` chain for the 1..=3 `QIS` slots (steps
  3–6), and enforces the 3-bit reserved trailer on intra frames
  (step 7) — the first standalone procedure of the §7 frame-decode
  pipeline.
* [`decode_long_run_bit_string`] / [`decode_short_run_bit_string`] —
  round 9 public §7.2.1 / §7.2.2 procedures returning a `Vec<u8>` of
  `0`/`1` values from a Table 7.7 / Table 7.11 Huffman-coded run-length
  stream. Long-run handles the VP3+ "read fresh BIT after RLEN=4129"
  exception (`LONG_RUN_MAX = 4129`); short-run unconditionally toggles
  between runs (cap `SHORT_RUN_MAX = 30`). Both surface a typed
  `Error::RunLengthOverrun { len, nbits }` when a decoded run advances
  past the caller-supplied `NBITS` bound (§7.2 step 10).

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

Round 7 adds §6.4.4 (DCT Token Huffman Tables), which decodes the 80
binary-tree Huffman tables the setup header carries for §7.7's DCT-
token decode. Like §6.4.3 the procedure is unblocked by the §6.4.1
spec gap because §6.4.4 follows §6.4.2 in §6.4.5 step 4:

```rust
pub fn decode_dct_token_huffman_tables(bits: &[u8])
    -> Result<Box<[HuffmanTable; NUM_HUFFMAN_TABLES]>, Error>;

pub struct HuffmanEntry { pub code: u32, pub len: u8, pub token: u8 }

pub struct HuffmanTable { pub entries: Vec<HuffmanEntry>, /* + tree */ }

impl HuffmanTable {
    pub fn len(&self) -> usize;
    pub fn lookup(&self, code: u32, len: u8) -> Option<u8>;
}

pub const NUM_HUFFMAN_TABLES: usize = 80;
pub const MAX_HUFFMAN_ENTRIES: usize = 32;
```

The procedure transcribes the §6.4.4 binary-tree recursion: at every
node read a 1-bit `ISLEAF`; if 1, read a 5-bit `TOKEN` and add the
`(HBITS, TOKEN)` pair as a leaf; otherwise recurse `0` then `1`. It is
implemented iteratively with an explicit DFS stack (the spec itself
warns against host recursion on adversarial inputs that grow the path
to 32 bits), with the same depth-first `0`-before-`1` ordering. Each
decoded table is materialised twice: as a flat `Vec<HuffmanEntry>` in
spec-visit order (for inspection and round-trip testing) and as a flat
binary-tree node array for the per-bit lookup §7.7.2 will perform.
Undecodable streams — `HBITS` longer than 32 bits (step 1(b)) or a
33rd entry in a single table (step 1(d)i) — return typed errors
(`HuffmanCodeTooLong { hti }` / `HuffmanTableFull { hti }`). The
degenerate single-leaf-at-root case (which the spec explicitly
tolerates — "multiple codes for the same token value in a single
table" is allowed) is preserved by storing the root leaf and routing
the empty-code lookup to it.

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
end-to-end `parse_setup_header` body decode (§6.4.5 step 2 onward,
on a shared bit reader) remains blocked. With §6.4.4 landed in round
7, every other section the setup-header body chains through (§6.4.2 /
§6.4.3 / §6.4.4) is implemented as a standalone entry point that can
be exercised independently; only §6.4.1 still blocks the chained
end-to-end body decode.

Round 4 mitigates the blockage for VP3-compatible bitstreams via
`vp3_defaults()`. `version >= 0x030200` streams continue to require
the §6.4.1 / §6.4.2 procedures and remain blocked until the docs
collaborator recovers the missing procedure body.

118 unit tests cover happy-path parses on all three header types,
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
Round 7 adds eleven §6.4.4 Huffman-table tests: trivial single-leaf
tables on every `hti` slot with empty-code lookup, the balanced 32-leaf
table with full lookup coverage and depth-first leaf ordering checks,
variable-length codes in spec-visit order, independent shapes across
all 80 slots, the four failure modes (truncated `ISLEAF`, hand-crafted
truncated `TOKEN`, depth-33 left-spine code-too-long, 33-entry table-
full), multi-table truncation reporting the correct field, `Error`
`Display` rendering, and `HuffmanTable::lookup` returning `None` for
codes at the wrong length. Round 8 adds nineteen §7.1 frame-header
tests: intra single/two/three-`qi` happy paths (with the 63 upper-
boundary on every slot), inter-frame happy paths (single and three
`qi`), `first_frame` Intra-only enforcement (and the legal inter-after-
keyframe path), header-packet rejection (high bit set), step 7 reserved-
bits rejection on every non-zero 3-bit pattern, truncation at the
packet-type bit / `MOREQIS[0]` / `QIS[2]` field boundaries, the
sentinel-byte "doesn't consume past header" check, error `Display`
rendering for all three new variants, the `FrameType` Table 7.3 numeric
mapping, the `MAX_FRAME_QIS = 3` constant, and a five-case
independent-slot round-trip across byte boundaries. Round 9 adds
twenty-four §7.2 run-length tests (total now 142): Table 7.7 / Table
7.11 transcription against every row plus the `LONG_RUN_MAX = 4129` /
`SHORT_RUN_MAX = 30` derivations, the `NBITS = 0` empty-string short-
circuit, single-record decode of every Table 7.7 / 7.11 entry at both
range endpoints, the toggle-between-runs invariant on both procedures,
the long-run "read fresh BIT after RLEN=4129" exception (both fresh-0
and fresh-1 paths) plus the symmetric "RLEN=4128 still toggles" check,
the short-run "RLEN=30 still toggles" check (no exception path),
truncation rejects at the initial BIT / mid-Huffman-walk / mid-ROFFS
field boundaries on both procedures, the §7.2 step 10 `RunLengthOverrun`
reject on both procedures with `Display` rendering, a long-run byte-
boundary-crossing decode, and a realistic 16-bit short-run super-block
decode covering four toggled runs.

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
