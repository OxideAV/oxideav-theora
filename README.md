# oxideav-theora

Pure-Rust Theora video codec — clean-room implementation in progress.

## Status — 2026-06-11 (round 278)

Round 278 extends the §7.11 step 1 chain a fourth link — **step
1(d)**, the §7.5.2 motion vectors — onto the same shared `BitReader`.
`decode_data_packet_header_and_blocks` now takes `pf` (the pixel
format, which §7.5.2 needs for `INTER_MV_FOUR` chroma-MV averaging)
and the per-macro-block `ChromaBlockLayout`, decodes the §7.1 header
(1(a)), §7.3 coded-block flags (1(b)), and §7.4 modes (1(c)) as
before, then threads §7.5.2 on the same cursor — the `MVMODE` bit and
per-macro-block MV stream resume immediately after the §7.4 mode
stream with no byte re-alignment, exactly as each earlier link
resumed after its predecessor. The typed `DataPacketHeaderAndBlocks`
gains an `mvects: Vec<MotionVector>` field (the `NBS`-element
`MVECTS` array §7.9.4 consumes at step 5). Step 1(d)'s "If FTYPE is
non-zero" gate is realised through §7.5.2's own intra short-circuit
(§7.5 opening sentence) — identical bit consumption (none) and output
(all-zero `MVECTS`) on intra frames, with the §7.5.2 shape validation
applied uniformly. All §7.5.2 reject paths propagate unchanged.
+2 net tests (484 → 486): a §7.5.2 chroma-map-length reject through
the driver and a `LAST1`/`LAST2` register-file walk (`INTER_MV` seed →
`INTER_MV_LAST` reuse → `INTER_MV_LAST2` swap → fresh `INTER_MV`)
under `MVMODE=0` Table 7.23 Huffman components; the prior step-1
tests were extended to assert the new `MVECTS` output, including the
4:2:0 `INTER_MV_FOUR` chroma average with a ties-away-from-zero
`round()` case. The step 1 chain now covers 1(a)+1(b)+1(c)+1(d);
steps 1(e) (§7.6 qi), 1(f) (§7.7.3 coeffs), 1(g) (§7.8.2 DC
inversion) and the step 5 / step 6 dispatch into `reconstruct_frame`
/ `loop_filter_frame` remain pending.

## Status — 2026-06-11 (round 274)

Round 274 extends the §7.11 step 1 chain a third link —
**step 1(c)**, the §7.4 macro-block coding modes — onto the same
shared `BitReader`. `decode_data_packet_header_and_blocks` now takes
`nmbs` and the `NMBS`-element `macro_block_to_luma_blocks` map (the
luma-block indices §7.4 step 2(d)i reads `BCODED` through), decodes
the §7.1 header (1(a)) and §7.3 coded-block flags (1(b)) as before,
then threads the §7.4 modes procedure on the same cursor — §7.4's
`MSCHEME` / `MALPHABET` / mode stream resumes immediately after the
§7.3 run-length streams with no byte re-alignment, exactly as §7.3
resumed after §7.1. The typed `DataPacketHeaderAndBlocks` gains an
`mbmodes: Vec<MacroBlockMode>` field and an `nmbs()` accessor. Intra
frames short-circuit to all-`Intra` with no mode bits (§7.4 step 1);
inter frames decode the on-wire modes, including step 2(d)ii's
no-bits `INTER_NOMV` for a wholly-uncoded macro block. All §7.4
reject paths propagate unchanged. +2 net tests (482 → 484): a §7.4
luma-map-length reject and an inter path where one macro block is
wholly uncoded (its mode is skipped yet the shared reader stays
aligned for the coded neighbours); the seven prior step-1(a)+(b)
tests were extended to assert the new `MBMODES` output. The step 1
chain now covers 1(a)+1(b)+1(c); steps 1(d) (§7.5.2 motion vectors),
1(e) (§7.6 qi), 1(f) (§7.7.3 coeffs), 1(g) (§7.8.2 DC inversion) and
the step 5 / step 6 dispatch into `reconstruct_frame` /
`loop_filter_frame` remain pending.

## Status — 2026-06-10 (round 267)

Round 267 lands the first composed link of the §7.11 step 1 chain —
the "If the size of the data packet is non-zero" branch — as
[`decode_data_packet_header_and_blocks`]. The driver decodes the §7.1
frame header (step 1(a)) followed by the §7.3 coded block flags (step
1(b)) against a **single shared** `BitReader`, which is the whole
point of the round: §7.3's run-length-coded `SBPCODED` / `SBFCODED`
streams begin at the bit position immediately after the §7.1 header's
last bit, mid-byte, with no re-alignment — so the two procedures must
run on one cursor, something the byte-aligned standalone entry points
[`decode_frame_header`] / [`decode_coded_block_flags`] cannot model.
The typed `DataPacketHeaderAndBlocks { header, bcoded }` carries
`ftype()` / `nqis()` / `nbs()` accessors for the downstream step
1(c)..1(g) links (§7.4 modes → §7.5.2 MVs → §7.6 qi → §7.7.3 coeffs →
§7.8.2 DC inversion), each of which resumes on the same reader in
later rounds. All §7.1 and §7.3 reject paths propagate unchanged.
+7 tests (475 → 482): intra all-coded short-circuit, two inter
shared-reader paths (fully-coded and a genuinely mixed `BCODED` that
exercises the per-block §7.2.2 short-run stream after the two §7.2.1
long-run passes), the header-packet / first-frame-must-be-intra /
block-map-length-mismatch rejects, and an accessor-consistency check.
The step 1 chain now covers steps 1(a)+1(b); steps 1(c)..1(g) and the
step 5 / step 6 dispatch into `reconstruct_frame` / `loop_filter_frame`
remain pending.

## Status — 2026-06-08 (round 260)

Round 260 lands §7.11 step 2 — the "Otherwise" / zero-byte packet
branch — as the standalone synthesiser
`synthesize_empty_packet_frame_state(nbs)`. Returns a typed
`EmptyPacketFrameState { ftype, nqis, qis, bcoded }` carrying the
four hard-coded outputs the spec mandates: step 2(a) `FTYPE := 1`
(`FrameType::Inter`), step 2(b) `NQIS := 1`, step 2(c) `QIS[0] :=
63`, and step 2(d) `BCODED[bi] := 0` for every block. `nbs == 0`
is a structural reject via the new `Error::EmptyPacketFrameStateZeroNbs`
variant (§6.2 step 23 derives `NBS >= 1` for any well-formed
stream). A `From<&EmptyPacketFrameState>` projection emits a
[`TheoraFrameHeader`] view for the §7.9.4 / §7.10.3 callers that
read only `FTYPE` / `QIS`. Two convenience accessors `qi0()` and
`nbs()` keep step-1 / step-2 call sites symmetric. +6 tests
(469 → 475). The §7.11 entry-shape now covers steps 2 / 3 / 4 /
7 / 8 end-to-end; the step 1 chain (§7.1 → §7.3 → §7.4 →
(§7.5.2) → §7.6 → §7.7.3 → §7.8.2 on a shared bit reader) and
the step 5 / step 6 dispatch into the existing `reconstruct_frame`
/ `loop_filter_frame` drivers remain pending.

## Status — 2026-06-08

Round 256 wires §7.11 step 7 + step 8 — the reference-frame
promotion that closes the §7 frame-decode loop. New owned
`ReferenceFrameStore` carries the six long-lived planes
(`GOLDREFY` / `GOLDREFCB` / `GOLDREFCR` + `PREVREFY` / `PREVREFCB`
/ `PREVREFCR`) across frames; `promote_from_reconstructed(rec,
ftype)` executes step 7 (`if FTYPE == 0 then GOLDREF* := REC*`)
and step 8 (`PREVREF* := REC*` always) via `copy_from_slice` with
shape and length validation against the store's declared `(RPYW,
RPYH)` / `(RPCW, RPCH)` geometry. Three constructors cover
fresh-allocation, ref-dims-bridge, and buffer-recycling decoders;
`as_reference_plane_set()` bridges the owned store back to a
borrowed `ReferencePlaneSet<'_>` for the next frame's §7.9.4 call.
Companion helpers `frame_type_from_ftype` / `frame_type_as_ftype`
decode the §7.1 / §7.11-step-2 raw 1-bit `FTYPE` into the existing
`FrameType` enum with a typed reject for out-of-range bytes.
+13 tests (456 → 469).

## Status — 2026-06-07

**Identification + comment + setup-entrypoint + §6.4.1 loop-filter
limits + §6.4.2 quant-params decode + §6.4.3 quant-matrix compute +
§6.4.4 DCT-token Huffman tables + §7.1 frame-header decode + §7.2
long-/short-run bit strings + §7.3 coded-block-flags decode + §7.4
macro-block coding modes + §7.5 motion-vector decode + §7.6 block-level
qi decode + §7.7.1 EOB token decode + §7.7.2 coefficient token decode +
§7.7.3 DCT coefficient decode driver + §7.8.1 DC predictor compute +
§7.8.2 DC prediction inversion driver + §7.9.1 predictors (intra /
whole-pixel / half-pixel) + §7.9.2 dequantization + §7.9.3 inverse DCT
(1D + 2D) + §7.9.4 per-block reconstruction + §7.9.4 frame-level
driver + §2.3 / §2.4 coded-order resolver (rounds 1–23, 233, 238).** §6.1, §6.2
(identification), §6.3 (comment), §6.4.5 step 1 (setup-header
common-header guard), §6.4.1 (Loop Filter Limit Table Decode), §6.4.2
(Quantization Parameters Decode), §6.4.3 (Computing a Quantization
Matrix), §6.4.4 (DCT Token Huffman Tables), §7.1 (Frame Header Decode),
§7.2 (Run-Length Encoded Bit Strings), §7.3 (Coded Block Flags Decode),
§7.4 (Macro Block Coding Modes), §7.5.1 (single Motion Vector decode),
§7.5.2 (per-macro-block Motion Vector decode), §7.6 (Block-Level *qi*
Decode), §7.7.1 (EOB Token Decode), §7.7.2 (Coefficient Token Decode),
and §7.7.3 (DCT Coefficient Decode) of the Theora I Specification are
wired up. Three byte-aligned entry points plus ten public bit-level
decoders over an MSb-first bit reader; round 4 added the Appendix B
VP3 fallback tables, round 5 added the full §6.4.2 procedure (ACSCALE
/ DCSCALE / NBMS / BMS / NQRS / QRSIZES / QRBMIS), round 6 added §6.4.3
(64-entry interpolated quantization matrix per `(qti, pli, qi)`
selector), round 7 added §6.4.4 — decode the 80-element array of
binary-tree Huffman tables that §7.7 will use to decode DCT-residual
tokens, round 8 added §7.1 (the typed `TheoraFrameHeader` from the
start of a video-data packet), round 9 added §7.2 — the long-run and
short-run bit-string decoders that §7.3 (coded-block flags) and §7.6
(block-level `qi` values) consume against the §5.2 bit reader, round
10 added §7.3 — the per-block `BCODED` array decoder that chains a
partially-coded super-block §7.2.1 long-run map, a fully-coded
super-block §7.2.1 long-run map (over the non-partially-coded subset),
and a per-block §7.2.2 short-run stream against a caller-supplied
block-to-super-block mapping, round 11 adds §7.4 — the per-macro-block
`MBMODES` array decoder consuming `BCODED` plus a caller-supplied
macro-block-to-luma-blocks mapping, demultiplexing all eight Table
7.18 modes through Schemes 0..=7 (Table 7.19's six fixed Huffman
alphabets, the on-wire MSCHEME=0 alphabet, and the MSCHEME=7 direct
3-bit encoding), round 13 adds §7.6 — the per-block `QIIS` array
decoder chaining `NQIS − 1` §7.2.1 long-run passes over the per-block
subset of still-`qii`-tied coded blocks (VP3-compat `NQIS == 1`
short-circuit consumes zero bits and returns all-zero `QIIS`), round
14 adds §7.7.1 — the EOB token applicator decoding one of the
Table 7.33 EOB tokens against per-block `TIS`/`NCOEFFS`/`COEFFS`
state arrays and returning the residual `EOBS` run length, round 16
adds §7.7.2 — the coefficient token applicator decoding one of the
25 non-EOB tokens (Table 7.38 values 7..=31), writing the implied
SIGN/MAG-derived coefficients to `COEFFS[bi]`, advancing `TIS[bi]`,
updating `NCOEFFS[bi]` (skipping the count update for pure zero-run
tokens 7 / 8 per the §7.7.2 introductory text), and returning a typed
`CoefficientTokenKind` discriminating zero-run / single / run-plus-one
classes for the §7.7.3 driver to branch on, and round 17 adds §7.7.3
— the per-frame `DCT Coefficient Decode` driver that runs the `ti`
0..=63 zig-zag outer loop, reads `htiL` / `htiC` at `ti ∈ {0, 1}`,
selects the Huffman table from the §6.4.4 80-table array via
Table 7.42's `(HG, htiL|htiC)` lookup (with the `bi < NLBS`
luma-vs-chroma split), walks the table bit-by-bit to recover `TOKEN`,
dispatches to §7.7.1 for `TOKEN < 7` or §7.7.2 for `TOKEN >= 7`, and
enforces the closing-paragraph contract (`EOBS = 0`, `TIS[bi] = 64`
for every coded block) via two typed reject variants.

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
* [`decode_coefficient_token`] — round 16 public §7.7.2 procedure
  consuming one of the 25 non-EOB DCT tokens (`token: u8` in `7..=31`,
  Table 7.38), reading the token-specific SIGN / MAG / RLEN
  extra-bits payload (0..=11 bits), writing the implied coefficient(s)
  to `coeffs[bi]`, advancing `tis[bi]`, and updating `ncoeffs[bi]`
  except for the pure zero-run tokens 7 / 8 (per §7.7.2's introductory
  text: "we do not update the coefficient count for the block if we
  decode a pure zero run"). Returns a typed `CoefficientTokenKind`
  (`ZeroRun` / `Single` / `RunPlusOne`) so the §7.7.3 driver can
  branch on the token's structural class without re-deriving
  Table 7.38. Multi-coefficient tokens fail closed with
  `Error::CoefficientTokenWouldOverflowBlock` when their implied
  coefficient count would push `TIS[bi]` past 64, surfacing §7.7.2's
  own MUST-NOT clause ("they MUST NOT bring the total number of
  coefficients in the block to more than 64"). Five new typed errors
  (`CoefficientTokenOutOfRange`, `CoefficientTokenBlockIndexOutOfRange`,
  `CoefficientTokenIndexOutOfRange`, `CoefficientTokenStateLenMismatch`
  with a `CoefficientTokenStateSlice` discriminant,
  `CoefficientTokenWouldOverflowBlock`) reject malformed inputs. The
  shared-reader chaining contract is preserved via a crate-private
  `decode_coefficient_token_inner`.
* [`invert_dc_prediction`] — round 19 public §7.8.2 driver chaining
  [`compute_dc_predictor`] over every coded block of the frame in
  plane-local raster order. Walks `pli` from 0 to 2, resets the
  `LASTDC` register file to `[0, 0, 0]` at the top of each plane
  (steps 1(a)–(c)), and for each coded block calls §7.8.1 to form
  the predictor against the freshly-updated DC values of earlier
  raster neighbours (step 1(d)i.A), adds the predictor to the
  residual `COEFFS[bi][0]` (step 1(d)i.B), truncates the sum to a
  16-bit two's-complement representation via an `i32 → i16 → i32`
  narrowing cast (step 1(d)i.C), writes the reconstructed DC back
  into `COEFFS[bi][0]` (step 1(d)i.D), then seeds `LASTDC[rfi]` with
  the new DC for the current macro block's reference frame
  (steps 1(d)i.E–G). AC coefficients (indices 1..=63) are never
  touched. The caller supplies `plane_raster_order: &[&[u32]]` (an
  exact-3 array of plane slices, each listing the coded-order block
  indices belonging to that plane in plane-local raster order); the
  driver enforces that no coded-order index appears in two plane
  slices via [`Error::DcInversionDuplicateBlockIndex`]. Four new
  typed errors (`DcInversionPlaneCount`, `DcInversionLenMismatch`
  with a [`DcInversionLenField`] discriminant for
  `BlockToMacroBlock` / `Neighbors` / `Coeffs`,
  `DcInversionBlockIndexOutOfRange`,
  `DcInversionDuplicateBlockIndex`) reject malformed inputs; any
  §7.8.1 error from the inner [`compute_dc_predictor`] propagates
  unchanged. Operates in place on `coeffs: &mut [[i16; 64]]` so the
  call site keeps the §7.7.3 output's allocation.
* [`compute_intra_predictor`] / [`compute_whole_pixel_predictor`] /
  [`compute_half_pixel_predictor`] — round 21 public §7.9.1.x
  procedures returning an `[[u8; 8]; 8]` predictor tile.
  `compute_intra_predictor` is the §7.9.1.1 constant-128 fill (the
  spec's centring rationale leaves DC residuals in the signed range
  `[-128, 127]`). `compute_whole_pixel_predictor(refp, BX, BY, MV)`
  transcribes §7.9.1.2 step 1: copy `REFP[ry][rx]` for `ry = BY +
  MVY + by` and `rx = BX + MVX + bx`, with both indices clamped into
  `[0, RPH-1]` and `[0, RPW-1]` so the predictor degrades gracefully
  at the reference plane's edges. `compute_half_pixel_predictor(refp,
  BX, BY, MV1, MV2)` transcribes §7.9.1.3 step 1: average two
  clamped reads with `(s1 + s2) >> 1` (truncation toward negative
  infinity per the spec's wording). Reference samples flow through a
  typed [`ReferencePlane { rpw, rph, samples }`] view; the
  `ReferencePlane::new` constructor validates `rpw * rph` against the
  buffer length and rejects zero-dimension / overflow inputs with
  three new typed errors (`ReferencePlaneLenMismatch`,
  `ReferencePlaneZeroDimension`, `ReferencePlaneDimensionsOverflow`).
  Companion helper [`split_half_pixel_motion_vector`] consumes a
  doubled MV and emits the (truncate-toward-zero, truncate-away-from-
  zero) pair §7.9.1.3 expects when the §7.9.4 chroma derivation
  halves an MV with odd integer-doubled components.
* [`dequantize_block`] / [`dequantize_block_from_params`] — round 20
  public §7.9.2 procedure. Takes a single block's zig-zag-order
  `[i16; 64]` (the §7.8.2 output) plus the precomputed DC and AC
  quantization matrices (§6.4.3 outputs for `(qti, pli, qi0)` and
  `(qti, pli, qi)` respectively) and returns the natural-order
  dequantized `[i16; 64]` (`DQC`). Step 2 sets `DQC[0] =
  trunc16(COEFFS[bi][0] * QMAT_DC[0])`; step 6 walks `ci` from 1 to 63,
  reads the zig-zag position `zzi` via the new public
  `ZIGZAG_NATURAL_TO_ZIGZAG: [u8; 64]` table (a row-by-row flattening
  of Figure 2.8), and sets `DQC[ci] = trunc16(COEFFS[bi][zzi] *
  QMAT_AC[ci])`. The §7.9.2 narrative explicitly anticipates the
  16-bit truncation ("If large coefficient values are decoded for
  coarsely quantized coefficients, the resulting dequantized value can
  be significantly larger than 16 bits"); Rust's `i32 → i16 → i32`
  narrowing performs the spec's "discarding the higher-order bits"
  under well-defined two's-complement semantics. The two-matrix split
  honours §7.6's per-block `qii` selector, which lets the per-block AC
  `qi` differ from the frame-wide DC `qi0`. The `_from_params`
  convenience form takes the spec's `(qti, pli, qi0, qi)` selector
  directly and propagates any §6.4.3 validation error
  (`qti > 1`, `pli > 2`, or `qi > 63`); production callers will
  amortise the §6.4.3 work over multiple blocks sharing the same
  `(qti, pli, qi)` by passing pre-built matrices into `dequantize_block`
  directly, per the §7.9.2 narrative's efficiency note.
* [`inverse_dct_1d`] / [`inverse_dct_2d`] — round 22 public §7.9.3
  procedures. `inverse_dct_1d(y: &[i16; 8]) -> [i16; 8]` walks the
  55 numbered steps of §7.9.3.1 verbatim — eight i32-typed `T[]`
  slots, three explicit "Truncate `T[i]` to a 16-bit signed
  representation" narrowing steps before each `C4 * T[i] >> 16`
  multiply, the eight rotation-pair multiplications using `C3` /
  `S3` / `C6` / `S6` / `C7` / `S7` from Table 7.65, the butterfly
  ladder in steps 13–31, and the eight `(T[a] ± T[b]) as i16`
  output truncations in steps 32–55. `inverse_dct_2d(dqc: &[i16;
  64]) -> [[i16; 8]; 8]` is the §7.9.3.2 two-pass driver: pass 1
  walks `ri ∈ 0..=7`, extracts each row of natural-order `DQC`,
  applies the 1D transform, and stores the result in `RES[ri][ci]`;
  pass 2 walks `ci ∈ 0..=7`, extracts each column of `RES`, applies
  the 1D transform again, and finalises each cell with
  `(X[ri] + 8) >> 4` to scale out the factor-of-four amplification
  introduced by the two 1D passes. The `(+8) >> 4` step implements
  the §7.9.3 final-rounding rule ("ties rounded towards positive
  infinity"): Rust's arithmetic `>>` on signed i32 truncates toward
  negative infinity, and the `+8` bias shifts the tie boundary so
  the resulting rounding direction matches the spec.
* [`compute_dc_predictor`] — round 18 public §7.8.1 procedure
  returning the typed `DCPRED: i32` for a single block from up to four
  already-decoded neighbour DC values (left, lower-left, lower,
  lower-right in plane-local raster geometry). Caller supplies `bi`,
  the per-block `bcoded` / `mbmodes` / `block_to_macro_block` /
  `neighbors: &[DcPredictorNeighbors]` arrays + the current `lastdc:
  &DcLastDc` register file (one slot per Table 7.46 reference frame:
  `None` / `Previous` / `Golden`) + the `coeffs: &[[i16; 64]]` per-
  block coefficient array. Each neighbour slot is gated by the §7.8.1
  steps 3..=10 chain: present-and-coded-and-same-rfi → `P[k]=1,
  PBI[k]=bj`; otherwise `P[k]=0`. When the mask `P[0..=3]` is zero
  (step 11), `LASTDC[rfi]` is returned directly. Otherwise step 12
  applies the Table 7.47 weights + divisor (15 rows, exposed as
  [`dc_predictor_weights`] for spot-check / round-trip use) with the
  spec's `//` (truncated-toward-zero) division, then the step 12(h)
  outranging guard pulls `DCPRED` back to `COEFFS[PBI[k]][0]` for
  `k ∈ {2, 0, 1}` (in that spec-mandated order) when the weighted
  estimate diverges by more than 128 — but only when `P[0]`, `P[1]`,
  `P[2]` are all set (the spec explicitly excludes `P[3]` from the
  guard predicate). The Table 7.46 `MBMODES → ReferenceFrame` mapping
  is exposed as the public helper [`reference_frame_for_mb_mode`].
  Four new typed errors (`DcPredictorBlockIndexOutOfRange`,
  `DcPredictorBcodedLenMismatch` with a `DcPredictorLenField`
  discriminant for `BlockToMacroBlock` / `Neighbors` / `Coeffs`,
  `DcPredictorMacroBlockIndexOutOfRange`,
  `DcPredictorNeighborIndexOutOfRange`) reject malformed inputs.
* [`decode_dct_coefficients`] — round 17 public §7.7.3 procedure
  driving the entire frame's DCT-coefficient stream end-to-end. The
  driver iterates `ti` from 0 to 63 (the zig-zag axis), reads
  `htiL` / `htiC` (4 bits each) at `ti ∈ {0, 1}`, and for every coded
  block whose `TIS[bi] == ti` either continues an in-flight EOB run
  (zero-fill `COEFFS[bi][ti..=63]`, pin `TIS[bi]=64`, decrement
  `EOBS`) or decodes a fresh `TOKEN` by selecting `hti = 16*HG +
  htiL/htiC` (Table 7.42 maps `ti → HG`, `bi < NLBS` chooses luma vs
  chroma), walking the §6.4.4 Huffman tree bit-by-bit, and dispatching
  to `decode_eob_token_inner` (`TOKEN < 7`) or
  `decode_coefficient_token_inner` (`TOKEN >= 7`). The closing-
  paragraph contract from §7.7.3 ("EOBS MUST be zero, and TIS[bi]
  MUST be 64 for every coded bi") is enforced fail-closed via two
  new reject variants (`DctCoefficientLeftoverEobs`,
  `DctCoefficientBlockNotClosed`). Five more typed errors cover
  input validation and tree corruption
  (`DctCoefficientBcodedLenMismatch`, `DctCoefficientNlbsExceedsNbs`,
  `DctCoefficientEmptyHuffmanTable`, `DctCoefficientHuffmanWalkOffTree`,
  plus the closing-paragraph pair). The shared-reader chaining
  contract is preserved via a crate-private
  `decode_dct_coefficients_inner`. Returns `(coeffs: Vec<[i16; 64]>,
  ncoeffs: Vec<u8>)` — the populated `NBS × 64` zig-zag-order
  quantized-DCT array and per-block coefficient count, both sized
  `nbs`.
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

Round 17 adds fifteen §7.7.3 DCT-coefficient-decode tests (total now
286): Table 7.42 row-by-row `huffman_table_group()` lookup across
every `ti` in 0..=63, the `bcoded.len() != nbs` reject with `Display`
rendering, the `NLBS = NMBS * 4 > NBS` reject, the all-uncoded
short-circuit confirming `htiL` / `htiC` are still read at `ti` 0
and 1 (16 bits total) but no per-block work runs, a single-block EOB
run via TOKEN=0 (block fully zero, `TIS=64`, `NCOEFFS=0`), a
single-block full coefficient fill via TOKEN=9 (every `ti` writes
`+1`, ending with `NCOEFFS=64` and `TIS=64`), truncation rejection
during a TOKEN=7 RLEN payload read, walk-off-tree rejection via a
hand-crafted corrupt table with out-of-range child indices, an empty-
table rejection (slot 0 contains zero entries), the `nbs=0` short-
circuit (still reads the table indices, vacuous closure check), a
custom-pack `htiL=3 / htiC=5` round-trip confirming step 4(a) fires
only at `ti ∈ {0, 1}`, the shared-`BitReader` chaining contract via
`decode_dct_coefficients_inner` followed by an 8-bit sentinel read
off the same reader, the `bi < NLBS` luma-vs-chroma split exercised
on a 5-block / 1-macro-block frame with bi=0 (luma) and bi=4 (chroma)
both decoded, the leftover-EOBS rejection via TOKEN=6 with a 12-bit
payload of 4095 producing an EOBS residue of 4094 the loop cannot
consume, and `Display` rendering for every new §7.7.3 error variant
carrying the section tag plus the offending quantity. A test-only
`make_uniform_hts(token)` helper builds an 80-element single-leaf
table set so the driver picks a deterministic table at every
`(ti, hg)` slot without forcing the test to lay out 80 distinct
trees.

Round 18 adds seventeen §7.8.1 DC-predictor tests (total now 303):
the Table 7.46 `MBMODES → ReferenceFrame` round-trip across every
coding mode (with the `ReferenceFrame::index()` numeric check), four
spot-checks of Table 7.47 rows including the signed-weight
`[29, -26, 29, 0]` PDIV=32 entry and the asymmetric `[0, 3, 10, 3]`
PDIV=16 entry, the step-11 `LASTDC[rfi]` fallback across all three
reference-frame slots (Intra → `LASTDC[0]`, InterMv → `LASTDC[1]`,
InterGoldenMv → `LASTDC[2]`), the single-left-neighbour P=0b0001
copy-DC0 path, an uncoded-neighbour case (`BCODED[bj] == 0`
masks `P[k]` to zero), a different-reference-frame case (Golden
neighbour to an Intra block masks `P[k]` even though
`BCODED[bj] == 1`), the three-neighbour P=0b0111 signed-weight
worked example (`(29*30 + -26*10 + 29*20) / 32 = 37`, guard within
range so no swap), the outranging-guard DC2-swap path with a
hand-crafted large divergence, the no-swap guard path
(`DCPRED == DC0 == DC1 == DC2` so every branch is within 128), a
negative-DC + truncated-division case, the `bi >= nbs` reject, all
three paired-length-mismatch rejects with the correct
`DcPredictorLenField` discriminator (`BlockToMacroBlock` /
`Neighbors` / `Coeffs`), both flavours of the macro-block-index OOR
reject (current block's `b2m[bi]` and a neighbour's `b2m[bj]`), the
neighbour-`bj >= nbs` reject, the `DcLastDc::get` / `set` round-
trip across all three slots, and `Display` rendering for every new
§7.8.1 error variant carrying the section tag.

### §7.8.1 Computing the DC Predictor (round 18)

```rust
pub fn compute_dc_predictor(
    bi: u32,
    bcoded: &[u8],
    mbmodes: &[MacroBlockMode],
    block_to_macro_block: &[u32],
    neighbors: &[DcPredictorNeighbors],
    lastdc: &DcLastDc,
    coeffs: &[[i16; 64]],
) -> Result<i32, Error>;

pub fn reference_frame_for_mb_mode(mode: MacroBlockMode) -> ReferenceFrame;
pub fn dc_predictor_weights(mask: u8) -> Option<DcPredictorWeights>;

pub struct DcPredictorNeighbors {
    pub left:        Option<u32>, // P[0] candidate (§7.8.1 step 3)
    pub lower_left:  Option<u32>, // P[1] candidate (§7.8.1 step 5)
    pub lower:       Option<u32>, // P[2] candidate (§7.8.1 step 7)
    pub lower_right: Option<u32>, // P[3] candidate (§7.8.1 step 9)
}

pub struct DcLastDc { pub intra: i32, pub previous: i32, pub golden: i32 }
pub enum ReferenceFrame { None, Previous, Golden }
pub struct DcPredictorWeights { pub w: [i32; 4], pub pdiv: i32 }
```

The procedure transcribes all twelve numbered steps of §7.8.1: the
`(mbi, rfi)` lookup via Table 7.46, the per-slot
present-and-coded-and-same-rfi gate that fills the four `P[k]` /
`PBI[k]` arrays (steps 3..=10), the `LASTDC[rfi]` step-11 fallback
when no neighbour qualifies, the Table 7.47-indexed weighted sum
(step 12(a)..=(f)), the spec's `//` (truncated-toward-zero) divide
(step 12(g) — Rust's `/` already truncates toward zero for signed
operands, matching `//`), and the step 12(h) outranging guard that
pulls `DCPRED` back to one of `COEFFS[PBI[k]][0]` for
`k ∈ {2, 0, 1}` (DC2 → DC0 → DC1, first match wins) when the
weighted estimate diverges by more than 128 — but only when `P[0]`,
`P[1]`, `P[2]` are all set (the spec writes "If P[0], P[1], and P[2]
are all non-zero"; `P[3]` is intentionally excluded from the guard
predicate). `DCPRED` and the per-slot DC values are widened to
`i32` so the worst-case `29 * 32767 ≈ 9.5e5` weighted-sum
intermediate (plus the `LASTDC` register file's signed `i32`
domain) fits without overflow.

The caller-supplied `DcPredictorNeighbors` table encodes the plane-
local raster geometry that the spec assumes but does not compute
(the procedure prose says "in the same row but one column to the
left" etc., but block ordering in Theora is plane-local raster — a
mapping that depends on §2.x geometry, the plane's subsampling,
and the coded-order Hilbert traversal within super-blocks). A
`None` slot represents BOTH "block `bi` is on the corresponding
edge of the coded frame" and "the neighbour `bj` is not coded";
both collapse to `P[k] = 0` in the spec, so the typed `Option`
captures both branches without a separate edge mask.

`LASTDC` is a 3-element register file (one slot per Table 7.46
reference frame: `None` = Intra, `Previous`, `Golden`) tracked
across blocks by §7.8.2. §7.8.1 itself only reads from it — the
step 11 fallback returns `LASTDC[rfi]` when no neighbour qualifies.
The `DcLastDc::get` / `set` accessors plus `DcLastDc::zero()`
(matching §7.8.2 step 1's initialisation to zero) give §7.8.2 a
typed surface to thread through the raster-order driver landing in
the next round.

No video-data packet decode yet. [`register`] is still a no-op —
`RuntimeContext` integration arrives once the codec can actually
decode a frame.

### §7.8.2 Inverting the DC Prediction Process (round 19)

```rust
pub fn invert_dc_prediction(
    bcoded: &[u8],
    mbmodes: &[MacroBlockMode],
    block_to_macro_block: &[u32],
    neighbors: &[DcPredictorNeighbors],
    plane_raster_order: &[&[u32]],
    coeffs: &mut [[i16; 64]],
) -> Result<(), Error>;
```

The driver walks the three colour planes (Y, Cb, Cr) in order. At
the start of each plane the `LASTDC` register file is reset to
`[0, 0, 0]` (steps 1(a)–(c)). For every block of the plane in the
caller-supplied raster ordering:

1. If `BCODED[bi] == 0` the block is skipped (step 1(d)i's
   non-zero gate is false).
2. Otherwise step 1(d)i.A recomputes the DC predictor via
   [`compute_dc_predictor`] against the freshly-updated DC values
   of earlier raster neighbours. The neighbour table is the same
   `&[DcPredictorNeighbors]` consumed by §7.8.1 and references
   coded-order indices of blocks already visited in this plane.
3. Step 1(d)i.B forms `DC = COEFFS[bi][0] + DCPRED` in `i32`.
4. Step 1(d)i.C truncates `DC` to a 16-bit two's-complement
   representation. The spec explicitly allows the raw sum to
   exceed 16 bits ("Because it is possible to add a value as large
   as 580 to the predicted DC coefficient value at every block …
   the reconstructed DC value could overflow a 16-bit integer.
   This is handled by truncating the result to a 16-bit signed
   representation, simply throwing away any higher bits"). An
   `i32 → i16 → i32` narrowing cast performs exactly this wrap
   under Rust's well-defined two's-complement narrowing rules.
5. Step 1(d)i.D writes the truncated DC back into
   `COEFFS[bi][0]`. AC coefficients (indices 1..=63) are never
   touched.
6. Steps 1(d)i.E–G compute `mbi = block_to_macro_block[bi]`, look
   up `rfi` via Table 7.46 ([`reference_frame_for_mb_mode`]), and
   record `LASTDC[rfi] = DC` so the next coded block on this plane
   can read it from the §7.8.1 step-11 fallback.

The caller supplies the plane raster ordering as
`plane_raster_order: &[&[u32]]` — a three-element slice (Y / Cb /
Cr) of coded-order indices in plane-local raster order. The driver
checks `plane_raster_order.len() == 3`
([`Error::DcInversionPlaneCount`]), `bi < bcoded.len()` for every
entry ([`Error::DcInversionBlockIndexOutOfRange`]), and (via a
crate-private visit bitmap) that no coded-order index appears in
more than one plane slice ([`Error::DcInversionDuplicateBlockIndex`]).
Any §7.8.1 error returned by the inner [`compute_dc_predictor`]
propagates unchanged so existing reject paths stay observable.

Round 19 adds 17 new tests (total now 320): the all-uncoded
short-circuit, the single-Intra-block reconstructed-DC path, the
two-block left-chain showing `LASTDC[None]` flows from block 0 to
block 1, the plane-boundary LASTDC reset (step 1(a)/(b)/(c) — second
plane's first block must NOT pick up the first plane's reconstructed
DC), the per-reference-frame `LASTDC[rfi]` index check (Intra →
Previous → Golden each fresh), the 16-bit two's-complement wrap
(`30_000 + 2_768 = 32_768 → -32_768`), the
AC-coefficients-untouched assertion, the wrong-plane-count reject
(2 and 4 plane slices), all three paired-length-mismatch rejects
with the correct `DcInversionLenField` discriminator, the
`bi >= nbs` plane-entry reject, the cross-plane duplicate-block
reject citing the second-visit plane, the §7.8.1 inner-error
propagation case (neighbour `bj >= nbs`), the three-planes-each-one-
block independence check, the `NBS = 0` short-circuit, the
§7.8.2-tag Display assertion for every new error variant, the
left-chain accumulation across four blocks
(`[+10, +20, +30, +40] → [10, 30, 60, 100]`), and the
uncoded-intermediate-doesn't-reset-LASTDC case (uncoded blocks are
visited for duplicate-tracking but leave `LASTDC[rfi]` unchanged).

### §7.9.1 Predictors (round 21)

```rust
pub fn compute_intra_predictor() -> [[u8; 8]; 8];

pub struct ReferencePlane<'a> {
    pub rpw: u32,
    pub rph: u32,
    pub samples: &'a [u8],
}

impl<'a> ReferencePlane<'a> {
    pub fn new(rpw: u32, rph: u32, samples: &'a [u8]) -> Result<Self, Error>;
}

pub fn compute_whole_pixel_predictor(
    refp: &ReferencePlane<'_>,
    bx_origin: u32,
    by_origin: u32,
    mv: MotionVector,
) -> [[u8; 8]; 8];

pub fn compute_half_pixel_predictor(
    refp: &ReferencePlane<'_>,
    bx_origin: u32,
    by_origin: u32,
    mv1: MotionVector,
    mv2: MotionVector,
) -> [[u8; 8]; 8];

pub fn split_half_pixel_motion_vector(
    double_mvx: i32,
    double_mvy: i32,
) -> Option<(MotionVector, MotionVector)>;
```

§7.9.1.1 (Intra) is the constant-128 fill. §7.9.1.2 (Whole-Pixel)
copies `REFP[ry][rx]` per `(by, bx)`, clamping `ry` into
`[0, RPH-1]` and `rx` into `[0, RPW-1]` so motion vectors that point
outside the reference plane fall back to the nearest edge sample.
§7.9.1.3 (Half-Pixel) averages two such clamped reads with the
spec's `>> 1` (truncation toward negative infinity for the always-
non-negative sum). The narrative anchors are §7.9.1.1's centring
rationale ("applied for the sole purpose of centering the range of
possible DC values for INTRA blocks around zero") and §7.9.1.3's
two-vectors-only invariant ("Only two samples from the reference
frame contribute to each predictor value, even if both components
of the motion vector have non-zero fractional components").

The §7.9.4 reconstruction driver decides at each block whether to
call the intra path, the whole-pixel path (fractional MV components
both zero), or the half-pixel path (any non-zero fractional
component). Theora MVs always come in as integer values; the
fractional shape arises only when §7.9.4 halves an MV for the
chroma planes — the `split_half_pixel_motion_vector` helper performs
that decomposition into the (round-toward-zero, round-away-from-
zero) pair the §7.9.1.3 procedure consumes.

The `ReferencePlane` wrapper carries the plane dimensions next to
the row-major flat byte slice; `ReferencePlane::new` validates the
`rpw * rph` invariant against the buffer length and rejects zero-
dimension / overflow inputs (`ReferencePlaneLenMismatch` /
`ReferencePlaneZeroDimension` / `ReferencePlaneDimensionsOverflow`).
The §7.9.1.2 / §7.9.1.3 procedures themselves do not return errors
— the `[[u8; 8]; 8]` output is fixed-size, the inputs are pre-
validated, and the edge clamping is total over the i32-widened
coordinate space.

Round 21 adds twenty-eight new tests (total now 366): intra
constant-128 invariants, the DC-centering math (`(s - 128)` lands
in `[-128, 127]`), `ReferencePlane::new` accept and three reject
paths plus their `Display` rendering, whole-pixel zero-MV
ramp-block copy, positive-MV offsetting, all four clamping edges
(`above_rph_minus_1`, `above_rpw_minus_1`, `below_zero`,
`all_four_corners_combined`), constant-plane round-trip, distinct-
origin isolation, half-pixel identical-vectors degenerate-to-whole-
pixel agreement, adjacent-column and adjacent-row averaging, the
truncation-toward-negative-infinity worked example
(`(2+3)>>1 = 2`), independent per-vector clamping, the
two-samples-only out-of-window invariant, even/odd/zero/mixed-
signs/out-of-range branches of `split_half_pixel_motion_vector`,
and a split-then-predict round-trip exercising the §7.9.1.3
narrative's "first toward zero, second away from zero" rule end-
to-end.

### §7.9.3 The Inverse DCT (round 22)

§7.9.3 of the Theora I Specification defines an integerised
8-point 1D inverse DCT (§7.9.3.1, 55 numbered steps) and a 2D
driver that applies the 1D transform once per row and once per
column of the resulting 8×8 grid (§7.9.3.2). The 1D transform
uses the Chen factorisation: each "rotation" pair `(Ci, Si)`
introduces a multiplication of the form `(Ci * Y[j]) >> 16` /
`(Si * Y[j]) >> 16` with the constants taken from Table 7.65.
The 1D transform scales the orthonormal result by a factor of
two; the 2D driver finalises with `(X[ri] + 8) >> 4` to scale
out the resulting factor-of-four and apply the spec's "ties
rounded towards positive infinity" rounding rule.

The two new public entry points are pure functions:

```text
pub fn inverse_dct_1d(y: &[i16; 8]) -> [i16; 8]
pub fn inverse_dct_2d(dqc: &[i16; 64]) -> [[i16; 8]; 8]
```

§7.9.3 prose: "A compliant decoder MUST use the exact
implementation of the inverse DCT defined in this specification.
Some operations may be re-ordered, but the result must be
precisely equivalent." The implementation does not reorder; each
of the 55 numbered steps appears as a single Rust statement,
preserving the spec's i16/i32 narrowing schedule verbatim. The
truncation steps are implemented as `(t[i] as i16) as i32`
narrow-then-widen so the subsequent `wrapping_mul(IDCT_C4)`
operand is the spec's mandated low-16-bit value. Output
truncation in steps 32–55 is `wrapping_add` / `wrapping_sub`
producing an i32 that is then narrowed via `as i16`.

The §7.9.3.2 driver respects the spec's exact ordering: the
row-pass loop completes (filling all of `RES`) before the column
pass starts. The `(X[ri] + 8) >> 4` rounding step happens only
on the column pass — the row pass stores the 1D output verbatim
into the i16 `RES` buffer. The `(+8) >> 4` step relies on Rust's
arithmetic right shift on signed i32 truncating toward negative
infinity; the `+8` bias shifts the tie boundary so that
`(0 + 8) >> 4 = 0` (no tie), `(8 + 8) >> 4 = 1` (positive tie
toward `+∞` = 1), and `(−8 + 8) >> 4 = 0` (negative tie toward
`+∞` = 0), matching the spec's three rounding modes.

The DC-only special case mentioned in §7.9.3 ("a special DC-only
case is used, which is described below in step 2(d)vii of
Section 7.9.4") is *not* part of the §7.9.3 procedures themselves
— it is a §7.9.4 decision. This round lands the full §7.9.3.1 +
§7.9.3.2 path; the §7.9.4 driver will own the per-block branch
between the full IDCT and the special case.

Round 22 adds thirteen new tests (total now 379): all-zero
input on both 1D and 2D paths, DC-only positive input on the 1D
path (pinned to `(C4 * 1024) >> 16 = 724` with literal
cross-check), DC-only negative input on the 1D path (pinned to
`−725` documenting the toward-negative-infinity floor of the
arithmetic right shift), `i16::MAX` / `i16::MIN` extreme input
exercising the wrapping arithmetic without panicking,
determinism of the 1D path under repeated calls with the same
non-trivial input, the §7.9.3.2 DC-only flat-block invariant
(`DQC[0] = 1024 → RES[ri][ci] = 32`), the negative-DC parallel
(`DQC[0] = −1024 → RES[ri][ci] = −32`), the small-DC rounds-to-
zero case (`DQC[0] = 16 → RES = 0` end-to-end), an AC-only
excitation test confirming the §7.9.3.1 transform is actually
active (output values are not all equal), a row-vs-column
transpose invariant confirming `DQC[1]` (`ri=0, ci=1`) and
`DQC[8]` (`ri=1, ci=0`) produce *transposed* `RES` blocks
(catching an accidental row/column swap), extreme i14 DC input
not panicking, and Table 7.65 constants pinned to their integer
values via a guard test.

### §7.9.4 The Complete Reconstruction Algorithm (round 23)

§7.9.4 of the Theora I Specification defines the per-block
reconstruction algorithm that selects the predictor (intra /
whole-pixel / half-pixel), computes the residual (DC-only
shortcut or full dequantize + inverse DCT), sums them, and clamps
to `0..=255`. Round 23 lands the **per-block body** of §7.9.4
step 2 in a single public entry point. The frame-level driver
(step 2 loop body framing — `bi in 0..NBS` raster walk with
per-plane geometry) is intentionally kept for a later round so
that the per-block procedure can be audited against §7.9.4 in
isolation.

```text
pub struct ReconstructedBlock { pub samples: [[u8; 8]; 8] }

pub struct ReconstructBlockInputs {
    pub bcoded: bool,
    pub mb_mode: MacroBlockMode,
    pub mvect: MotionVector,
    pub coeffs_zz: [i16; 64],
    pub ncoeffs: u32,
    pub qii: u8,
}

pub struct ReferencePlaneSet<'a> {
    pub previous_y: ReferencePlane<'a>,
    pub previous_cb: ReferencePlane<'a>,
    pub previous_cr: ReferencePlane<'a>,
    pub golden_y: ReferencePlane<'a>,
    pub golden_cb: ReferencePlane<'a>,
    pub golden_cr: ReferencePlane<'a>,
}

pub fn reconstruct_block(
    inputs: &ReconstructBlockInputs,
    pli: usize,
    bx_origin: u32,
    by_origin: u32,
    qis: &[u8],
    params: &QuantizationParameters,
    refs: &ReferencePlaneSet<'_>,
) -> Result<ReconstructedBlock, Error>;
```

The procedure walks each step of §7.9.4 verbatim:

* **Step 1.** `qi0 = QIS[0]`. Empty `qis` is rejected with
  `Error::ReconstructEmptyQis`.
* **Step 2(d) — coded block.** `qti` (step 2(d)ii / iii) = 0 for
  `Intra`, 1 otherwise. `rfi` (step 2(d)iv) comes from
  `reference_frame_for_mb_mode` (Table 7.46). `rfi == 0` (None)
  → §7.9.1.1 (constant 128) predictor; `rfi != 0` →
  `ReferencePlaneSet::pick` selects `REFP` / `RPW` / `RPH` per
  Table 7.75. Integer-valued `MVECTS[bi]` collapses MVX==MVX2 and
  MVY==MVY2, so step 2(d)vi.F always routes to §7.9.1.2; the
  §7.9.1.3 branch (step 2(d)vi.G) is reachable only via the
  `split_half_pixel_motion_vector` helper from round 21 once the
  §7.9.4 chroma-MV halving driver lands.
* **Step 2(d)vii — DC-only shortcut.** When `NCOEFFS[bi] < 2`,
  build the DC quant matrix from `(qti, pli, qi0)`, compute
  `DC = (COEFFS[bi][0] * QMAT[0] + 15) >> 5` truncated to a
  16-bit signed representation (Rust's `as i16` narrowing), and
  fill `RES[by][bx] = DC` for every cell. Both the `+15` bias and
  the `>> 5` shift match step 2(d)vii.B verbatim.
* **Step 2(d)viii — full DCT path.** Otherwise read
  `qi = QIS[QIIS[bi]]` (`Error::ReconstructQiiIndexOutOfRange`
  on a malformed `qii`), build the DC matrix from
  `(qti, pli, qi0)` and the AC matrix from `(qti, pli, qi)`,
  dequantize via §7.9.2, and `inverse_dct_2d` the result.
* **Step 2(e) — uncoded block.** `rfi = 1` (Previous), MV =
  zero, PRED via §7.9.1.2 (a direct co-located copy), `RES = 0`.
* **Step 2(f) — finalisation.** For each `(by, bx)`, sum PRED
  and RES, clamp to `0..=255`, store into
  `ReconstructedBlock.samples`.

The `ReferencePlaneSet` wrapper carries the six (rfi, pli)
reference planes Table 7.75 enumerates. `pick(rfi, pli)`
returns the right plane; an Intra-routed call into pick is the
`ReconstructIntraInterBranchMismatch` reject (an internal-
invariant guard — `reconstruct_block` itself routes Intra to the
§7.9.1.1 path before consulting `refs`).

Round 23 adds eighteen new tests (total now 397): the
Intra-DC-zero-residual all-128 invariant, an Intra-DC-positive
PRED-plus-DC offset, both clamp paths (high and low), the
uncoded `Previous-Y` co-located copy, plane-specific routing
(uncoded Cb → Previous-Cb, uncoded Cr → Previous-Cr), the
Inter-Golden Table 7.46 routing (`InterGoldenNoMv` →
Golden-Y), the Inter-Previous Table 7.46 routing
(`InterMv` → Previous-Cr at `pli = 2`), the whole-pixel-MV
positional shift, the all-zero-COEFFS full-DCT path collapsing
to PRED, the three reject paths (`ReconstructEmptyQis`,
`ReconstructQiiIndexOutOfRange`, `ReconstructPlaneIndexOutOfRange`),
the Table 7.75 `pick` lookup over all six (rfi, pli) pairs,
a pinned step 2(d)vii.B formula cross-check
(`DC = (COEFFS[0] * qmat_dc[0] + 15) >> 5` matched against a
hand-computed expectation), Display rendering for all four new
error variants, and a determinism guard for the per-block path.

### §7.9.4 frame-level driver (round 233)

Round 233 closes the framing layer above [`reconstruct_block`]:
the public [`reconstruct_frame`] entry point walks
`bi in 0..NBS`, dispatches each block to the round-23 per-block
body, and tiles the resulting 8×8 samples into three flat output
plane vectors per §7.9.4 step 2(f)iv-vi.

```text
pub struct PlaneDimensions { pub width: u32, pub height: u32 }

pub struct ReconstructedFrame {
    pub samples_y:  Vec<u8>,
    pub samples_cb: Vec<u8>,
    pub samples_cr: Vec<u8>,
    pub dims_y:  PlaneDimensions,
    pub dims_cb: PlaneDimensions,
    pub dims_cr: PlaneDimensions,
}

pub fn reconstruct_frame(
    bcoded:        &[bool],          // BCODED[bi]   — §7.3
    mb_modes:      &[MacroBlockMode],// MBMODES[mbi] — §7.4
    mvects:        &[MotionVector],  // MVECTS[bi]   — §7.5
    coeffs:        &[[i16; 64]],     // COEFFS[bi]   — §7.7
    ncoeffs:       &[u32],           // NCOEFFS[bi]  — §7.7
    qiis:          &[u8],            // QIIS[bi]     — §7.6
    pli_of_block:  &[u8],            // §2.3 colour-plane index per bi
    bx_of_block:   &[u32],           // §7.9.4 step 2(b)
    by_of_block:   &[u32],           // §7.9.4 step 2(c)
    mbi_of_block:  &[u32],           // §7.9.4 step 2(d)i
    qis:           &[u8],            // §7.1
    params:        &QuantizationParameters,
    refs:          &ReferencePlaneSet<'_>,
    dims_y:  PlaneDimensions,
    dims_cb: PlaneDimensions,
    dims_cr: PlaneDimensions,
) -> Result<ReconstructedFrame, Error>;
```

Per-plane (§2.3) raster geometry is **caller-supplied** rather
than re-derived inside the driver — matching the §7.8.2 convention
where the caller resolves the coded-order Hilbert mapping into a
per-block `(pli, BX, BY, mbi)` quadruple. This keeps the driver
focused on the §7.9.4 step 2 algebra (block-by-block dispatch +
output tiling) rather than on §2.3 / §2.4 coded-order mechanics.

The driver:

* **Length-checks** every per-block input slice against `nbs`
  before iterating. A mismatch surfaces as
  `ReconstructFrameBlockLenMismatch { which, got, nbs }` with a
  typed [`ReconstructFrameBlockSlice`] tag identifying the
  failing slice.
* **Allocates** three `Vec<u8>` planes of `width * height` zero
  bytes; `width * height` overflow surfaces as
  `ReconstructFramePlaneDimensionsOverflow`.
* **Resolves** `pli` (step 2(a)) from `pli_of_block[bi]`, with
  out-of-range values rejected as `ReconstructFramePliOutOfRange`.
* **Resolves** the destination plane + its dimensions, then rejects
  any `(BX, BY)` whose 8×8 footprint would escape the plane as
  `ReconstructFrameBlockOutOfPlane`.
* **For coded blocks** (step 2(d)) reads
  `mb_modes[mbi_of_block[bi]]`, with `mbi >= nmbs` rejected as
  `ReconstructFrameMbIndexOutOfRange`.
* **For uncoded blocks** (step 2(e)) skips the macro-block lookup
  entirely — step 2(d) is gated on `BCODED[bi] != 0`. The
  per-block call enters [`reconstruct_block`] with a placeholder
  `mb_mode` that the uncoded path ignores.
* **Tiles** the [`ReconstructedBlock`] return into the destination
  plane in lower-left row-major order per §2.1 ("origin in the
  lower-left corner"), writing `samples[BY + by][BX + bx]` for
  each of the 64 cells.

Round 233 adds eight new tests (total now 405): an all-intra
all-zero-coefficient 4:2:0 frame producing three planes of all-128
across the four-luma + one-Cb + one-Cr geometry; an uncoded luma
block at `(BX=0, BY=0)` confirmed to sample the `PREVREFY` ramp at
the same coordinates (with the other three luma blocks still
hitting the intra path); the four reject paths
(`ReconstructFramePliOutOfRange`,
`ReconstructFrameBlockLenMismatch` on `Mvects`,
`ReconstructFrameBlockOutOfPlane` for a `BX = 10` luma block on a
16-wide plane, and `ReconstructFrameMbIndexOutOfRange` for an
`mbi = 5` against `nmbs = 1`); Display tag rendering for all five
new error variants; and a determinism guard over a mixed
coded/uncoded input.

### §2.3 / §2.4 coded-order resolver (round 238)

`PlaneBlockCodedOrder` and `PlaneMacroBlockCodedOrder` are typed
iterators that walk one colour plane in the spec's coded order:
super-blocks in raster order (lower-left origin), Hilbert curve
inside each super-block per Figure 2.4 (16 blocks) and Figure 2.6
(4 macro-blocks). Each item carries the super-block index, the
slot inside the super-block, and the plane-local `(bx, by)` (or
`(mbx, mby)`) coordinates.

`PlaneBlockDims { mb_w, mb_h }` carries the plane geometry in
macroblocks. The two constructors derive the dimensions from a
parsed identification header:

* `PlaneBlockDims::luma_from_ident(&ident)` — luma plane:
  `(fmbw, fmbh)`.
* `PlaneBlockDims::chroma_from_ident(&ident)` — chroma plane:
  axes scaled per `PF` (Table 6.4): both halved for 4:2:0, only
  horizontal halved for 4:2:2, full size for 4:4:4. Halving
  rounds up so the chroma plane covers the entire luma area even
  when the luma macro-block count is odd.

Edge super-blocks (those that straddle the right or top of the
plane) follow §2.3's "the same ordering is still used, simply with
any blocks outside the frame boundary omitted" rule: the iterator
advances through all 16 (or 4) slots of the over-padded super-block
but filters out the slots whose `(bx, by)` lies outside the plane's
block extent. The 240×48 worked example of §2.3 page 8 / §2.4
page 11 is reproduced as two unit tests, plus a 4×4-block single
super-block test pinning the inner Hilbert sequence, a 6×6-block
partial-super-block test pinning the edge-filter behaviour, the
`PF` axis-by-axis subsampling rules of `chroma_from_ident`, and a
zero-dim degenerate edge case.

This is the standalone helper the roadmap below mentioned. It
unblocks any caller that needs to translate a coded-order block
index back into plane-local raster coordinates (or vice versa)
without redoing the Hilbert-curve geometry by hand — for example,
the upcoming §7.10 loop-filter driver and the §7.5 / §7.7 / §7.8
chains that currently take pre-resolved raster orderings as
caller-supplied arrays. Five new unit tests; whole-crate total now
411 (was 406).

### §7.10.1 / §7.10.2 loop-filter edge primitives + `lflim()` (round 241)

The §7.10 deblocking step is built around a piecewise non-linear
response function

```text
                  0,                  R <= -2 * L
                  -R - 2 * L,    -2L < R <= -L
lflim(R, L) =      R,              -L < R <  L
                  -R + 2 * L,       L <= R <  2 * L
                  0,                  R >=  2 * L
```

with peaks at `(±L, ±L)` and zero-crossings at `R = ±2 * L`. `L`
is the per-frame loop-filter limit value `LFLIMS[qi0]` — the entry
of the §6.4.1 `LFLIMS` array indexed by the frame's first
quantization index. The response declines to soften genuine edges
(`|R| >= 2L`) while attenuating small block-artifact differences
(`|R| < L` is passed through unchanged, `L <= |R| < 2L` is
attenuated linearly to zero).

[`lflim`] returns the signed delta as an `i32`. The two edge
primitives — [`horizontal_loop_filter_edge`] (§7.10.1) and
[`vertical_loop_filter_edge`] (§7.10.2) — apply this delta to a
single block edge of the reconstructed plane:

* The horizontal filter consumes a 4-wide × 8-tall footprint
  anchored at `(fx, fy)`. For each of 8 rows it computes
  `R = (recp[fy+by][fx+0] - 3*recp[fy+by][fx+1]
       + 3*recp[fy+by][fx+2] - recp[fy+by][fx+3] + 4) >> 3` and
  writes the two middle columns: `recp[fy+by][fx+1] +=
  lflim(R, L)` and `recp[fy+by][fx+2] -= lflim(R, L)`, each
  clamped to `0..=255` per §7.10.1 step 1(c-e) / step 1(g-i).
  The two outer reference columns `(fx, fx+3)` are untouched.
* The vertical filter is the same shape rotated 90°: an 8-wide
  × 4-tall footprint, walking 8 columns and writing rows
  `fy+1` / `fy+2`, with the two outer reference rows `(fy, fy+3)`
  left untouched per §7.10.2 step 1(c-e) / step 1(g-i).

Both primitives operate directly on the same lower-left
row-major plane buffer layout emitted by the round-233
[`reconstruct_frame`] driver, so the §7.10.3 raster-order ordering
driver (the natural follow-up) can dispatch into them without
re-shaping the per-plane storage.

Three new `Error` variants pin the rejects:
[`Error::LoopFilterHorizontalFootprintOutOfPlane`] /
[`Error::LoopFilterVerticalFootprintOutOfPlane`] reject any
`(fx, fy)` whose footprint would escape the plane; 
[`Error::LoopFilterPlaneBufferLenMismatch`] rejects a plane buffer
whose length does not match `plane_w * plane_h`. The §7.10 Display
strings carry the procedure tag so the source of a reject is
obvious at a `to_string()`.

19 new unit tests bring the whole-crate total to 430: `lflim`
piecewise breakpoints at `L = 5` covering every distinguished arm,
`lflim(R, 0)` collapses to identically zero, antisymmetry of
`lflim` in its first argument across `L ∈ {0, 1, 3, 7, 25, 127}`,
constant-input identity for both edge filters, step-edge softening
inside the central `lflim` segment for both filters (R = 5, delta
= 5), `LFLIMS = 0` disabling the filter on a strong step,
outside-of-band `|R| >= 2L` returning identity, plane-bounds clamp
at 255 with a contrived `(p2 - delta) = 275` setup, outside-
footprint pixels left untouched for both filters, the four
out-of-plane footprint rejects (right / top for each filter), the
plane-buffer-length-mismatch reject before either filter
dereferences the buffer, and `Display` rendering of the three new
error variants.

The §7.10.3 ordering driver — the raster-order walk that reads
`BCODED` and dispatches the left / bottom / right / top edge of
each coded block to one of these two primitives — is the natural
next round: these primitives are the per-edge body it dispatches
to.

### §7.10.3 Complete Loop Filter raster-order driver (round 244)

Round 244 lands the **§7.10.3 raster-order driver** as
[`loop_filter_frame`]. The procedure is the per-frame composition
of the §7.10.1 / §7.10.2 edge primitives that landed in round 241:

1. Step 1 reads `L = LFLIMS[QIS[0]]` once per frame — the per-frame
   loop-filter limit driving every `lflim()` call inside the walk.
2. Step 2 walks every block in raster order (left-to-right within
   a row, then bottom-to-top per the §2.1 lower-left convention)
   over each of the three planes in turn. For every block whose
   `BCODED[bi]` flag is set, the driver applies up to four edge
   filters:
   * step 2(a) v — left edge when `BX > 0` (the already-filtered
     right edge of the left neighbour).
   * step 2(a) vi — bottom edge when `BY > 0` (mirror of the above).
   * step 2(a) vii — right edge when `BX + 8 < RPW` **and** the
     right neighbour's `BCODED[bj]` is zero; the boundary needs
     filtering here because the right neighbour, being uncoded, will
     never visit it.
   * step 2(a) viii — top edge under the same neighbour-handover
     rule, rotated 90°.

The driver's API takes the raster→coded-order index map per plane
as `LoopFilterPlaneInput::grid_to_bi` (lower-left row-major,
`block_w * block_h` entries) and the three plane buffers + their
pixel dimensions as a single mutable `LoopFilterPlanes` struct.
Per-block arrays (`bcoded`, `pli_of_block`, `bx_of_block`,
`by_of_block`) are indexed by the coded-order `bi` the grid yields.

Errors raised (eight new typed variants):

* [`Error::LoopFilterFrameQisEmpty`] — `QIS` is empty so no `qi0`.
* [`Error::LoopFilterFrameQiOutOfRange`] — `QIS[0] > 63`.
* [`Error::LoopFilterFrameBlockLenMismatch`] — per-block slice
  length disagrees with `nbs`. The `which` field discriminates
  `BCODED` / `pli_of_block` / `bx_of_block` / `by_of_block`.
* [`Error::LoopFilterFrameRasterLenMismatch`] — `grid_to_bi.len()`
  disagrees with `block_w * block_h`.
* [`Error::LoopFilterFrameRasterEntryOutOfRange`] — a grid entry's
  `bi >= nbs`.
* [`Error::LoopFilterFramePliOutOfRange`] — `pli_of_block[bi] > 2`.
* [`Error::LoopFilterFrameBlockOutOfPlane`] — a coded block's
  `(bx, by) + 8` escapes the plane handed in for its `pli`, *or*
  the declared `block_w * 8 != plane_w` / `block_h * 8 != plane_h`.

Seventeen new tests cover the per-frame round-trip: single-block
identity (no neighbours, all four boundary tests short-circuit),
two-block shared-edge filtered-exactly-once (horizontal + vertical
mirror cases), step 2(a) vii / viii uncoded-neighbour boundary
filtering, plane independence (a Cb-only delta leaves Y and Cr
untouched), and all eight error variants exercised through their
typed `Error::*` pattern matches plus their `Display` rendering.
The driver is bit-exact against hand-computed `lflim()` deltas on
each fixture.

### §7.11 Complete Frame Decode entry-shape primitives (round 250)

Round 250 lands the first plumbing for the **§7.11 frame-decode
driver**: two public entry points implementing the geometry
derivation (steps 3-4) and the empty-packet branch (step 1 / step 2):

* [`reference_plane_dimensions_from_ident`] returns a typed
  [`ReferencePlaneDimensions { rpyw, rpyh, rpcw, rpch }`]. Step 3
  computes luma as `RPYW = 16 * FMBW`, `RPYH = 16 * FMBH`. Step 4
  picks chroma from Table 7.89:

  | PF | RPCW         | RPCH         | Format |
  |----|--------------|--------------|--------|
  | 0  | `8 * FMBW`   | `8 * FMBH`   | 4:2:0  |
  | 2  | `8 * FMBW`   | `16 * FMBH`  | 4:2:2  |
  | 3  | `16 * FMBW`  | `16 * FMBH`  | 4:4:4  |

  The reserved `PF = 1` row is unreachable: the identification
  header parser rejects it at construction time, so the
  `PixelFormat` variant set is `{Yuv420, Yuv422, Yuv444}` and the
  Table 7.89 match is total. Both dimensions are `u32`;
  `16 * 0xFFFF = 0x000F_FFF0` fits comfortably so no overflow check
  is needed.

* [`classify_frame_decode_packet`] returns
  [`FrameDecodePacket::Empty`] for a zero-byte packet (step 2
  special case: synthesise `FTYPE = 1`, `NQIS = 1`, `QIS[0] = 63`,
  all-zero `BCODED`, consume no bits) and
  [`FrameDecodePacket::Data`] otherwise (step 1 chain over the
  packet body). The enum exposes `is_empty()` and
  `data() -> Option<&[u8]>` inspectors. The classifier is a pure
  predicate — it does not read the packet body — so a follow-up
  round can chain it into the §7.1 → §7.3 → §7.4 → (§7.5.2) →
  §7.6 → §7.7.3 → §7.8.2 walk against the same shared bit reader
  the existing decoders already use.

Both entries are pure functions; nine new tests cover the per-PF
chroma rows, the `u16::MAX` boundary on `FMBW` / `FMBH`,
independence from the picture-region offsets (the reference plane
is always macroblock-aligned; cropping to `(PICW, PICH)` at
`(PICX, PICY)` is a §A.1 / display-time concern), the zero-byte
vs non-empty packet split, and the single-byte boundary case
(any `len >= 1` takes the step 1 path).

These primitives slot into the §7.11 driver alongside the
already-shipping §7.9.4 step-5 reconstruction
([`reconstruct_frame`]) and the §7.10.3 step-6 loop-filter pass
([`loop_filter_frame`]); step 1 (the per-procedure chaining) and
steps 7/8 (reference-frame carry-forward of `GOLDREF*` / `PREVREF*`)
remain for subsequent rounds.

## Roadmap

* Next: wire the §7.10.3 driver into the §7.9.4 frame reconstruction
  pipeline so [`reconstruct_frame`] returns the post-loop-filter
  buffer rather than the pre-loop-filter one. The blocker is the
  per-plane raster-order to coded-order index map, which a follow-up
  composition layer can build by iterating
  `PlaneBlockCodedOrder::new(dims)` and bucketing the
  `(bx, by)` → `bi` pairs into the raster grid.
* The §2.3 / §2.4 coded-order resolver landed in round 238 as
  `PlaneBlockCodedOrder` + `PlaneMacroBlockCodedOrder` over
  `PlaneBlockDims`. Follow-ups that compose it into per-block
  raster-order arrays (i.e. the `block_to_super_block`,
  `block_to_macro_block`, and three per-plane raster orderings the
  §7.3 / §7.5 / §7.8 drivers consume) can land alongside the
  §7.10 driver in a later round, on top of the iterators that now
  exist.
* §7.9.3.3 (the 1D forward DCT) is explicitly non-normative per
  the spec ("the version of the transform used by Xiph.Org's
  Theora encoder, which is the same as that used by VP3") and is
  deferred to a later encoder-side round.

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
