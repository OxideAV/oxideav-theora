# oxideav-theora

Pure-Rust Theora video codec — clean-room implementation in progress.

## Status — 2026-05-29

**Identification + comment + setup-entrypoint + §6.4.1 loop-filter
limits + §6.4.2 quant-params decode + §6.4.3 quant-matrix compute +
§6.4.4 DCT-token Huffman tables + §7.1 frame-header decode + §7.2
long-/short-run bit strings + §7.3 coded-block-flags decode + §7.4
macro-block coding modes + §7.5 motion-vector decode + §7.6 block-level
qi decode + §7.7.1 EOB token decode (rounds 1–15).** §6.1, §6.2
(identification), §6.3 (comment), §6.4.5 step 1 (setup-header
common-header guard), §6.4.1 (Loop Filter Limit Table Decode), §6.4.2
(Quantization Parameters Decode), §6.4.3 (Computing a Quantization
Matrix), §6.4.4 (DCT Token Huffman Tables), §7.1 (Frame Header Decode),
§7.2 (Run-Length Encoded Bit Strings), §7.3 (Coded Block Flags Decode),
§7.4 (Macro Block Coding Modes), §7.5.1 (single Motion Vector decode),
§7.5.2 (per-macro-block Motion Vector decode), §7.6 (Block-Level *qi*
Decode), and §7.7.1 (EOB Token Decode) of the Theora I Specification
are wired up. Three
byte-aligned entry points plus eight public bit-level decoders over
an MSb-first bit reader; round 4 added the Appendix B VP3 fallback
tables, round 5 added the full §6.4.2 procedure (ACSCALE / DCSCALE /
NBMS / BMS / NQRS / QRSIZES / QRBMIS), round 6 added §6.4.3 (64-entry
interpolated quantization matrix per `(qti, pli, qi)` selector), round
7 added §6.4.4 — decode the 80-element array of binary-tree Huffman
tables that §7.7 will use to decode DCT-residual tokens, round 8 added
§7.1 (the typed `TheoraFrameHeader` from the start of a video-data
packet), round 9 added §7.2 — the long-run and short-run bit-string
decoders that §7.3 (coded-block flags) and §7.6 (block-level `qi`
values) consume against the §5.2 bit reader, round 10 added §7.3 — the
per-block `BCODED` array decoder that chains a partially-coded
super-block §7.2.1 long-run map, a fully-coded super-block §7.2.1
long-run map (over the non-partially-coded subset), and a per-block
§7.2.2 short-run stream against a caller-supplied block-to-super-block
mapping, round 11 adds §7.4 — the per-macro-block `MBMODES` array
decoder consuming `BCODED` plus a caller-supplied
macro-block-to-luma-blocks mapping, demultiplexing all eight Table
7.18 modes through Schemes 0..=7 (Table 7.19's six fixed Huffman
alphabets, the on-wire MSCHEME=0 alphabet, and the MSCHEME=7 direct
3-bit encoding), round 13 adds §7.6 — the per-block `QIIS` array
decoder chaining `NQIS − 1` §7.2.1 long-run passes over the per-block
subset of still-`qii`-tied coded blocks (VP3-compat `NQIS == 1`
short-circuit consumes zero bits and returns all-zero `QIIS`), and
round 14 adds §7.7.1 — the EOB token applicator decoding one of the
Table 7.33 EOB tokens against per-block `TIS`/`NCOEFFS`/`COEFFS`
state arrays and returning the residual `EOBS` run length.

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
  passes — see "§6.4.1 recovered procedure body (round 15)" below
  for what now unblocks the §6.4.5 step 2 piece and what still
  blocks the chained end-to-end body decode.
* [`decode_loop_filter_limit_table`] — round 15 public §6.4.1
  procedure returning a 64-element `[u8; 64]` `LFLIMS` table.
  Reads a 3-bit `NBITS` followed by 64 `NBITS`-bit unsigned values
  (one per `qi` in 0..=63). The procedure body is recovered from
  `docs/video/theora/theora-6.4.1-lflims.md` (the published
  `Theora.pdf` omits the numbered steps). Closes the §6.4.5 step 2
  spec-gap; the chained `parse_setup_header` body decode still
  needs §6.4.1 → §6.4.2 → §6.4.4 wired on a shared bit reader.
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
* [`decode_coded_block_flags`] — round 10 public §7.3 procedure
  returning an `NBS`-element `Vec<u8>` of `0`/`1` `BCODED` flags
  marking which blocks are coded. Intra frames short-circuit step 1
  (every block coded; packet not consumed); inter frames execute the
  §7.3 step 2 chain: one §7.2.1 long-run pass for `SBPCODED`
  (`NBITS = NSBS`), one §7.2.1 long-run pass for `SBFCODED`
  (`NBITS = #{sbi: SBPCODED[sbi]=0}`), then one §7.2.2 short-run pass
  for the per-block bits inside partially-coded super blocks
  (`NBITS = sum of block counts where SBPCODED[sbi]=1`, edge super
  blocks contribute < 16 blocks). Caller supplies
  `block_to_super_block: &[u32]` of length `NBS`; two new typed errors
  (`BlockSuperBlockMapLenMismatch`, `BlockSuperBlockIndexOutOfRange`)
  reject malformed mappings. The shared-reader chaining contract is
  preserved via a crate-private `decode_coded_block_flags_inner` that
  drives an already-positioned `BitReader`.
* [`decode_macroblock_modes`] — round 11 public §7.4 procedure
  returning an `NMBS`-element `Vec<MacroBlockMode>` of per-macro-block
  coding modes (Table 7.18: `InterNoMv`, `Intra`, `InterMv`,
  `InterMvLast`, `InterMvLast2`, `InterGoldenNoMv`, `InterGoldenMv`,
  `InterMvFour`). Intra frames short-circuit step 1 (every mb =
  INTRA; packet not consumed). Inter frames read a 3-bit `MSCHEME`,
  build `MALPHABET` either from the bitstream (Scheme 0; eight 3-bit
  `mi` values defining the permutation `MALPHABET[mi] = MODE`) or
  from one of the six fixed columns of Table 7.19 (Schemes 1..=6), or
  bypass the alphabet altogether for Scheme 7's direct 3-bit mode
  encoding. For each macro block in coded order, if at least one
  luma block in the caller-supplied
  `macro_block_to_luma_blocks: &[[u32; 4]]` (length `NMBS`) has
  `BCODED[bi] == 1`, the procedure decodes a Huffman-coded mode
  (Schemes 0..=6, Table 7.19's unary-with-cap-at-7-bits prefix) or a
  3-bit direct value (Scheme 7); otherwise step 2(d)ii assigns
  `INTER_NOMV` and reads no bits. Three new typed errors
  (`MacroBlockLumaMapLenMismatch`, `MacroBlockLumaBlockIndexOutOfRange`,
  `UnknownMacroBlockModeCode`) reject malformed inputs. The shared-
  reader chaining contract is preserved via a crate-private
  `decode_macroblock_modes_inner`.
* [`decode_block_level_qi`] — round 13 public §7.6 procedure returning
  an `NBS`-element `Vec<u8>` of per-block `QIIS` values (each in
  `0..NQIS`). For `NQIS == 1` the procedure short-circuits the spec's
  empty main loop and returns all-zero `QIIS` without reading any bits
  (VP3-compatibility path; the spec note in §7.6 explicitly observes
  this). For `NQIS == 2` or `NQIS == 3` it executes `NQIS − 1`
  §7.2.1 long-run passes — each pass's length is the number of coded
  blocks (`BCODED[bi] != 0`) still tied at the current `qii`, each
  decoded bit is *added* to that block's `QIIS[bi]` (0 keeps the block,
  1 promotes it into the second group the next pass sees). Two new
  typed errors (`BlockLevelQiBcodedLenMismatch`,
  `BlockLevelQiNqisOutOfRange`) reject malformed inputs. The shared-
  reader chaining contract is preserved via a crate-private
  `decode_block_level_qi_inner`.
* [`decode_eob_token`] — round 14 public §7.7.1 procedure consuming
  one of the seven Table 7.33 EOB tokens (`token: u8` in `0..=6`),
  reading the matching 0 / 2 / 3 / 4 / 12-bit extra-bits payload, and
  applying the per-block state mutation: step 8 zero-fills
  `coeffs[bi][ti..=63]`, step 9 captures `ncoeffs[bi] = tis[bi]`, step
  10 pins `tis[bi] = 64`, and the return value is the post-step-11
  residual `EOBS` run length (i.e. the number of *additional* blocks
  the current EOB run will close at the start of subsequent §7.7
  passes). Token-6 zero-payload special case (§7.7.1 step 7(b)):
  `EOBS` becomes the count of blocks `bj` with `tis[bj] < 64`
  *including* the current block, matching the spec's "the size of the
  remaining coded blocks" wording. Four new typed errors
  (`EobTokenOutOfRange` for `token > 6`,
  `EobTokenBlockIndexOutOfRange` for `bi >= nbs`,
  `EobTokenIndexOutOfRange` for `ti > 63`, and
  `EobTokenStateLenMismatch` with an `EobTokenStateSlice` discriminant
  for `tis`/`ncoeffs`/`coeffs` length mismatch) reject malformed
  inputs. The shared-reader chaining contract is preserved via a
  crate-private `decode_eob_token_inner`.
* [`decode_single_motion_vector`] / [`decode_macroblock_motion_vectors`]
  — round 12 public §7.5.1 / §7.5.2 procedures returning a single
  [`MotionVector`] or an `NBS`-element `Vec<MotionVector>` of per-
  block motion vectors. §7.5.1 supports both MVMODE=0 (Table 7.23
  3..=8-bit Huffman codes for signed values `-31..=31`) and MVMODE=1
  (5-bit unsigned magnitude + 1-bit sign per component, with the
  sign bit always read even when magnitude is zero — VP3 compat).
  §7.5.2 short-circuits intra frames (no MVMODE bit consumed; every
  output (0, 0)); inter frames execute step 1 (`LAST1 = LAST2 =
  (0, 0)`), step 2 (1-bit MVMODE always read), and step 3 dispatch
  on `MBMODES[mbi]`: `INTER_MV_FOUR` (per-coded-luma MVs with
  uncoded-luma (0, 0) + PF=0/2/4 chroma averaging via the spec's
  `round()` ties-away-from-zero + `LAST1` update from last coded
  luma), `INTER_GOLDEN_MV` (decode without LAST update), `INTER_MV_
  LAST2` (rotate LAST1/LAST2), `INTER_MV_LAST` (reuse LAST1),
  `INTER_MV` (decode + LAST update), and the NOMV/INTRA fallback
  (emit zero). Caller supplies `luma_map: &[[u32; 4]]` of length
  `NMBS` (raster A, B, C, D per macroblock) and
  [`ChromaBlockLayout`] (per-plane outer length `NMBS`; inner length
  `1` for PF=0, `2` for PF=2, `4` for PF=3). Six new typed errors
  reject malformed inputs. The shared-reader chaining contract is
  preserved via a crate-private
  `decode_macroblock_motion_vectors_inner`.

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

### Setup header (rounds 3 + 4 + 5 + 15)
[`parse_setup_header`] implements only §6.4.5 step 1: it validates
the 7-byte common header (`0x82` + ASCII `"theora"`) and returns
[`Error::SetupHeaderBodyNotImplemented`] once the preamble checks
out. Round 15 added the §6.4.5 step 2 standalone entry point
[`decode_loop_filter_limit_table`] (see "§6.4.1 recovered procedure
body (round 15)" below); the chained end-to-end body decode that
drives §6.4.1 → §6.4.2 → §6.4.4 against a single shared `BitReader`
inside `parse_setup_header` is still pending.

Round 5 lands the §6.4.2 (Quantization Parameters Decode) procedure
as a standalone public entry point so it can be exercised
independently of the §6.4.1 spec rendering issue (§6.4.5 step 3
follows step 2 in the chained decode):

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

### §6.4.1 recovered procedure body (round 15)

The spec PDF as published does not contain the numbered procedure
steps for §6.4.1 (Loop Filter Limit Table Decode). Page 50 ends
with "It is decoded as follows:" and page 51 begins immediately
with "VP3 Compatibility" / §6.4.2 — the steps that should bridge
the two pages are absent from the PDF stream we have.

Round 15 closes that gap using
`docs/video/theora/theora-6.4.1-lflims.md`, which transcribes the
recovered two-step procedure from the spec's own LaTeX source. The
new public [`decode_loop_filter_limit_table`] entry point exposes
the §6.4.1 decode against a §5.2-style MSb-first `BitReader`:

```rust
pub fn decode_loop_filter_limit_table(bits: &[u8])
    -> Result<[u8; 64], Error>;
```

The recovered procedure is straightforward:

1. Read a 3-bit unsigned integer as `NBITS`.
2. For each consecutive `qi` from 0 to 63 inclusive, read an
   `NBITS`-bit unsigned integer as `LFLIMS[qi]`.

Total bits consumed: `3 + 64 * NBITS`. `NBITS` is shared across all
64 entries (read once before the loop). There is no per-value
clamping — the 7-bit output width matches the `NBITS ≤ 7` ceiling.
The §7.10 loop filter consumes `LFLIMS[qi0]` as the deblocking
limit value `L`.

What now works end-to-end via three standalone entry points:

* §6.4.5 step 1 — `parse_setup_header` common-header guard.
* §6.4.5 step 2 — [`decode_loop_filter_limit_table`] (round 15).
* §6.4.5 step 3 — [`decode_quantization_parameters`] (round 5).
* §6.4.5 step 4 — [`decode_dct_token_huffman_tables`] (round 7).

What remains: a chained `parse_setup_header` body decode that drives
§6.4.1 → §6.4.2 → §6.4.4 on a single shared `BitReader` so the
sub-byte continuation between sections is honoured (the standalone
entry points each take a byte-aligned slice). The crate-private
`decode_lflims_inner` already implements the shared-reader contract,
matching `decode_quant_params_inner` and `decode_huffman_tables_inner`.

Round 4 already mitigates the gap for VP3-compatible bitstreams via
`vp3_defaults()` — `version < 0x030200` streams do not carry a
transmitted `LFLIMS` table, per Appendix B.1.

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
twenty-four §7.2 run-length tests (total then 142): Table 7.7 / Table
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
decode covering four toggled runs. Round 10 adds seventeen §7.3
coded-block-flags tests (total now 159): long-run / short-run encoder
round-trip helpers (sanity-checking the test fixtures), intra short-
circuit (every block coded; packet not consumed), the two input-
validation rejects (`BlockSuperBlockMapLenMismatch`,
`BlockSuperBlockIndexOutOfRange`) with `Display` rendering, inter all-
super-blocks-not-coded, inter all-super-blocks-fully-coded, inter
mixed-super-block-states with both `SBFCODED` and per-block paths
exercised, the edge-super-block-with-fewer-than-16-blocks tally check
(§7.3 step 2(g) note), the empty NSBS=0 short-circuit, mid-`SBPCODED`
truncation rejection, intra-vs-inter arm independence, the shared-
`BitReader` chaining contract via `decode_coded_block_flags_inner`,
a single-partially-coded-super-block uncoded-block-subset case, and
an interleaved (non-monotone) `block_to_super_block` mapping case.
Round 11 adds twenty §7.4 macro-block-coding-modes tests (total now
179): the `MacroBlockMode::from_index` / `to_index` round-trip across
the eight Table 7.18 values plus the `None` rejection for `8`, the
intra short-circuit (every mb = `INTRA`; empty packet decodes), the
two input-validation rejects (`MacroBlockLumaMapLenMismatch`,
`MacroBlockLumaBlockIndexOutOfRange`), Scheme 7's direct 3-bit mode
read across four modes, every Scheme 1..=6 column of Table 7.19
walked `mi=0..=7`, Scheme 0's on-wire alphabet permutation with
inverse-recovery checks across all eight `mi` slots, the
partially-coded-luma trigger path (single luma block coded forces a
mode read), the all-luma-uncoded step 2(d)ii path with NO mode bits
read, MSCHEME / MALPHABET / Huffman-walk / direct-mode truncation
rejects, the `b1111110` / `b1111111` 7-bit-prefix disambiguation
(Scheme 1, `mi=6` vs `mi=7`), the shared-`BitReader` chaining contract
via `decode_macroblock_modes_inner`, error-`Display` rendering, an
explicit no-bits-consumed assertion for fully-uncoded inter frames,
the `nmbs=0` short-circuit, and a defensive coverage walk across all
eight MSCHEME values. Round 12 adds twenty-eight §7.5 motion-vector
tests (total now 207): Table 7.23 round-trip across every value in
`-31..=31` (both as a standalone MVX with MVY=0 and as paired (v, -v)
components), MVMODE=1 5+1-bit round-trip with the sign-on-magnitude-
zero invariant exercised explicitly, MV reader truncation rejects on
both MVMODE paths, intra-frame short-circuit (empty packet decodes;
no MV bits read), INTER_NOMV-only inter consumes only the MVMODE
bit, INTER_MV decode-and-propagate to every coded block, INTER_MV_
LAST chain reuses LAST1, INTER_MV_LAST2 3-way rotation through
LAST1/LAST2, INTER_GOLDEN_MV decodes one MV without touching the
LASTs, all three chroma-averaging formulas (PF=0 with non-divisible-
by-4 sums to exercise `round()` ties; PF=2 bottom/top halves with
`round()` ties on both signs; PF=4 direct copy), INTER_MV_FOUR with
one uncoded luma → chroma averaging uses (0, 0) for that slot,
INTER_MV_FOUR's `LAST1` update from the last *coded* luma block (D
in raster order) feeding a following INTER_MV_LAST, all six input-
validation rejects (`MotionVectorMbModesLenMismatch` /
`MotionVectorLumaMapLenMismatch` /
`MotionVectorLumaBlockIndexOutOfRange` /
`MotionVectorChromaMapLenMismatch` /
`MotionVectorChromaMacroBlockSlotLenMismatch` /
`MotionVectorChromaBlockIndexOutOfRange`) with `Display` rendering,
truncation at the MVMODE bit, the shared-`BitReader` chaining
contract via `decode_macroblock_motion_vectors_inner` (followed by a
sentinel byte read on the same reader), an explicit "intra consumes
no bits" packet-reread, an uncoded chroma block in step 3(g) keeps
its (0, 0) default while coded blocks get the MV, and a `round_div`-
matches-spec check (1.5/-1.5/2.5/-2.5 all tie away from zero per the
§ Notation `round(a)` definition). Round 13 adds fifteen §7.6
block-level-`qi` tests (total now 222): VP3-compat `NQIS=1`
short-circuit (no bits consumed; sentinel-byte reread verifies
position), `NQIS=2` happy path assigning per-coded-block bits in
ascending-`bi` order with uncoded blocks staying at 0, `NQIS=3`
two-pass chain where pass 1's NBITS is restricted to blocks promoted
out of pass 0, an interleaved `NQIS=3` case confirming coded-order
iteration, the `NQIS=3` second-pass-empty path (sentinel-byte
verifies no extra bits read), the `NBS=0` short-circuit on both
`NQIS=1` and `NQIS=3`, the all-blocks-uncoded path
(sentinel-byte verifies 16 bits intact), both input-validation
rejects (`BlockLevelQiBcodedLenMismatch`,
`BlockLevelQiNqisOutOfRange` for `nqis=0` and `nqis=4`) with
`Display` rendering, truncation mid-pass-0 and mid-pass-1 surfacing
as `TruncatedHeader` from the §7.2.1 layer, the shared-`BitReader`
chaining contract via `decode_block_level_qi_inner` (followed by
twelve sentinel bits read off the same reader), and a multi-run
long-run round-trip that crosses two RSTART=10 long-run records.
Round 14 adds sixteen §7.7.1 EOB-token-decode tests (total now 238):
the constant-run TOKEN=0/1/2 paths consume no bits (empty packet
decodes; step-8 zero-fill of `coeffs[bi][ti..=63]`, step-9 capture
of `ncoeffs[bi]`, step-10 pin of `tis[bi] = 64`, all verified
against sentinel values), the TOKEN=3 two-bit payload across all
four values (0..=3), TOKEN=4 three-bit payload across all eight
(0..=7), TOKEN=5 four-bit payload across all sixteen (0..=15),
TOKEN=6 with a non-zero 12-bit payload returning the literal,
TOKEN=6 with a zero payload falling into the "all remaining coded
blocks" sentinel (including the current block in the tally;
excluded pinned-block case verified separately), token-out-of-
range rejects across `7..=20`, `bi >= nbs` rejects, `ti > 63`
rejects across `64..=127`, all three state-slice length-mismatch
rejects (`Tis` / `Ncoeffs` / `Coeffs`), truncation rejection on
each of TOKEN=3..=6's extra-bits payload reads with no-partial-
state assertions, the ti=0 full-block zero-fill case, the
step-9-captures-pre-call-TIS invariant, the shared-`BitReader`
chaining contract via `decode_eob_token_inner` (six tail bits then
a sentinel byte read off the same reader), and `Display` rendering
for every new error variant carrying `§7.7.1` plus the offending
quantity.

Round 15 adds seven §6.4.1 loop-filter-limit tests (total now 245):
the `NBITS = 0` corner case (every entry forced to 0, only 3 bits
consumed), the Appendix B.2 VP3 table round-trip at `NBITS = 5`
(maximum entry 30 < 2^5 − 1), a synthesized payload at the
`NBITS = 7` ceiling that exercises the full 7-bit output width via
`qi`-derived expected values, every entry at the `127 = 2^7 − 1`
maximum value, a 24-byte truncated payload at `NBITS = 3` reporting
the `LFLIMS` field, a zero-byte payload reporting the `LFLIMS NBITS`
field, and the shared-`BitReader` chaining contract via
`decode_lflims_inner` (a 5-bit mid-byte tail then an 8-bit sentinel
byte read off the same reader after the 195-bit §6.4.1 payload).
A `pack_msb_first` test helper packs `(value, nbits)` slot lists
MSb-first to mirror the `BitReader` decoding without a real setup-
header fixture.

No video-data packet decode yet. [`register`] is still a no-op —
`RuntimeContext` integration arrives once the codec can actually
decode a frame.

## Clean-room sources

Only the Xiph Theora I Specification
(`docs/video/theora/Theora.pdf`), the staged §6.4.1 procedure body
at `docs/video/theora/theora-6.4.1-lflims.md` (transcribed from the
spec's own LaTeX source for the section the published PDF omits),
and the fixture corpus under `docs/video/theora/fixtures/` are
consulted. No libtheora, no FFmpeg `vp3.c`, no theora-rs.

Black-box `ffmpeg` and `theoradec` binary invocations are allowed
as opaque validators.

## License

MIT. See `LICENSE`.
