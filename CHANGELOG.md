# Changelog

All notable changes to `oxideav-theora` are recorded here.

## [Unreleased]

## [0.0.8](https://github.com/OxideAV/oxideav-theora/compare/v0.0.7...v0.0.8) - 2026-05-29

### Other

- decode an EOB token + apply per-block state (round 14)
- §7.6 Block-Level qi Decode (round 13)
- §7.5 Motion Vectors decode (round 12)
- §7.4 Macro Block Coding Modes decode (round 11)
- §7.3 Coded Block Flags Decode (round 10)
- §7.2 Long-/Short-Run Bit Strings decode (round 9)
- decode the frame header (round 8)
- decode the 80 DCT-token Huffman tables (round 7)
- §6.4.3 Computing a Quantization Matrix (round 6)
- §6.4.2 Quantization Parameters Decode (round 5)
- VP3 LFLIMS/ACSCALE/DCSCALE tables from Appendix B (round 4)
- Setup-header entrypoint (§6.4.5 step 1) + MSb-first BitReader (§5.2)
- Comment-header parser per §6.3 of the Theora I Specification
- Identification-header parser per §6.1 + §6.2 of the Theora I Specification
- orphan rebuild: clean-room scaffold post 2026-05-20 audit

### Added

* **§7.7.1 EOB Token Decode (2026-05-29, round 14).** New public
  `decode_eob_token(packet: &[u8], token: u8, nbs: u32, bi: u32,
  ti: u8, tis: &mut [u8], ncoeffs: &mut [u8],
  coeffs: &mut [[i16; 64]]) -> Result<u64, Error>` transcribing the
  full §7.7.1 procedure of the Xiph Theora I Specification ("EOB
  Token Decode"). Consumes one of the seven EOB-token values
  (Table 7.33 tokens 0..=6), reads the matching 0 / 2 / 3 / 4 / 12-
  bit extra-bits payload, ends the current block by zero-filling
  `COEFFS[bi][ti..=63]` (step 8), captures the block's pre-call
  coefficient count in `NCOEFFS[bi]` (step 9), pins `TIS[bi]` to
  `64` (step 10), and returns the residual `EOBS` run length after
  the step-11 decrement.

  Token-6 zero-payload sentinel (§7.7.1 step 7(b)): when the 12-bit
  payload reads zero, `EOBS` becomes the count of blocks `bj` such
  that `TIS[bj] < 64` *including* the current block (because step 10
  has not yet pinned it) — matching the spec's "the size of the
  remaining coded blocks" wording.

  Four new `Error` variants reject malformed inputs:
  `EobTokenOutOfRange { token }` for `token > 6` (tokens 7..=31 are
  §7.7.2 coefficient tokens, not handled here);
  `EobTokenBlockIndexOutOfRange { bi, nbs }` for `bi >= nbs`;
  `EobTokenIndexOutOfRange { ti }` for `ti > 63`; and
  `EobTokenStateLenMismatch { which, got, nbs }` (with a new
  `EobTokenStateSlice` discriminant) when any of `tis` / `ncoeffs` /
  `coeffs` has a length other than `nbs`. All four carry `Display`
  arms citing §7.7.1.

* **§7.6 Block-Level *qi* Decode (2026-05-27, round 13).** New public
  `decode_block_level_qi(packet: &[u8], nbs: u32, bcoded: &[u8],
  nqis: usize) -> Result<Vec<u8>, Error>` transcribing the full §7.6
  procedure of the Xiph Theora I Specification ("Block-Level *qi*
  Decode"). Returns the per-block `QIIS` array — `QIIS[bi]` indexes
  into the frame's `qi` value list (the 1..=3 values
  `decode_frame_header` returns in [`TheoraFrameHeader::qis`]) and
  drives AC dequantization of block `bi`.

  Per §7.6 the procedure assigns `QIIS[bi] = 0` for every block (step
  1) then makes `NQIS − 1` passes through the list of coded blocks
  (step 2; outer loop `qii` from 0 to `NQIS − 2`). Pass `qii` tallies
  `NBITS = #{bi : BCODED[bi] != 0 AND QIIS[bi] == qii}` (step 2(a)),
  decodes an `NBITS`-bit long-run bit string via the §7.2.1 procedure
  (step 2(b)), and for each matching block in coded order (ascending
  `bi`) adds the next consumed bit to `QIIS[bi]` (step 2(c)) — bit 0
  keeps the block at `qii`, bit 1 promotes it to `qii + 1` for the
  next pass's tally.

  VP3-compatibility short-circuit: `NQIS == 1` evaluates the outer
  loop as `0..=−1`, i.e. zero passes, so no bits are read and every
  returned `QIIS[bi]` is `0`. This is the only path Theora streams
  whose `decode_frame_header` returned `nqis() == 1` exercise — and is
  the only path `version < 0x030200` streams can take per §B.1 (VP3
  frame headers carry exactly one `qi` value).

  Two new `Error` variants reject malformed inputs:
  `BlockLevelQiBcodedLenMismatch { bcoded_len, nbs }` when the
  supplied `bcoded` slice length disagrees with `nbs`, and
  `BlockLevelQiNqisOutOfRange { nqis }` when `nqis` is outside the
  `1..=MAX_FRAME_QIS` (= 3) range mandated by §7.1 step 6. Both carry
  `Display` arms citing §7.6.

  The procedure is split into `decode_block_level_qi` (byte-aligned
  entry point) plus `decode_block_level_qi_inner(&mut BitReader<'_>,
  …)` (crate-private) so a future end-to-end frame decoder can chain
  §7.1 → §7.2 → §7.3 → §7.4 → §7.5 → §7.6 on a shared reader without
  re-aligning to a byte boundary. §7.6 is unblocked by the still-open
  §6.4.1 spec gap (docs-gap #944): §7.6 operates on a video-data
  packet's own payload and does not consume the setup-header body.

  Fifteen new tests bring the total from 207 to 222: the VP3-compat
  `NQIS=1` short-circuit (no bits consumed, verified via a sentinel
  byte read off the same reader), `NQIS=2` happy path assigning per-
  coded-block bits in ascending-`bi` order with uncoded blocks staying
  at 0, `NQIS=3` two-pass chain where pass 1's `NBITS` is restricted
  to blocks promoted out of pass 0, an interleaved `NQIS=3` case
  confirming coded-order iteration is by ascending `bi` (not by
  `bcoded` slice order), the `NQIS=3` second-pass-empty path (sentinel
  byte verifies no extra bits read), the `NBS=0` short-circuit on
  `NQIS=1` and `NQIS=3`, the all-blocks-uncoded path (sentinel byte
  verifies 16 bits intact), both input-validation rejects
  (`BlockLevelQiBcodedLenMismatch`, `BlockLevelQiNqisOutOfRange` for
  `nqis=0` and `nqis=4`) with `Display` rendering, truncation in
  pass 0 and pass 1 surfacing as `TruncatedHeader` from the §7.2.1
  layer, the shared-`BitReader` chaining contract via
  `decode_block_level_qi_inner` (followed by twelve sentinel bits
  read off the same reader), and a multi-run long-run round-trip that
  crosses two RSTART=10 long-run records.

* **§7.5 Motion Vectors (2026-05-27, round 12).** New public
  `decode_single_motion_vector(bits, mvmode) -> Result<MotionVector,
  Error>` transcribing §7.5.1 of the Xiph Theora I Specification
  ("Motion Vector Decode") and
  `decode_macroblock_motion_vectors(packet, ftype, pf, nbs, nmbs,
  bcoded, mbmodes, luma_map, chroma_map) ->
  Result<Vec<MotionVector>, Error>` transcribing §7.5.2 ("Macro Block
  Motion Vector Decode") on top of the §5.2 MSb-first `BitReader`.

  `decode_single_motion_vector` implements both MV decoding methods
  §7.5.1 allows: `MVMODE=0` resolves a 3..=8-bit Huffman code per
  Table 7.23 (signed values `-31..=31`); `MVMODE=1` reads a 5-bit
  unsigned magnitude + 1-bit sign per component, with the sign bit
  always read even when the magnitude is zero (the VP3-compat
  invariant called out in §7.5.1).

  `decode_macroblock_motion_vectors` implements §7.5.2 in full: intra
  short-circuit (every entry `MotionVector::ZERO`, no MVMODE bit
  consumed), `LAST1`/`LAST2` initialisation (step 1), MVMODE read
  (step 2), and step 3 dispatch on `MBMODES[mbi]` —
  `INTER_MV_FOUR` (four per-coded-luma MVs + uncoded-luma (0, 0) +
  PF=0/2/4 chroma averaging via spec `round(...)` with ties away
  from zero + `LAST1` update from last *coded* luma),
  `INTER_GOLDEN_MV` (decode-without-LAST-update), `INTER_MV_LAST2`
  (rotate `LAST1`/`LAST2`), `INTER_MV_LAST` (reuse `LAST1`),
  `INTER_MV` (decode + LAST update), and the NOMV/INTRA fallback
  (emit zero). Step 3(g) propagates the single MV to every coded
  block (luma + chroma) for non-`INTER_MV_FOUR` macroblocks.

  New public types: `MotionVector { x: i8, y: i8 }` (with
  `MotionVector::ZERO`) and `ChromaBlockLayout { cb, cr }`. Six new
  typed errors reject malformed inputs:
  `MotionVectorMbModesLenMismatch`, `MotionVectorLumaMapLenMismatch`,
  `MotionVectorLumaBlockIndexOutOfRange`,
  `MotionVectorChromaBlockIndexOutOfRange`,
  `MotionVectorChromaMapLenMismatch`,
  `MotionVectorChromaMacroBlockSlotLenMismatch`.

  Twenty-eight new tests cover: Table 7.23 round-trip across every
  value in `-31..=31`, MVMODE=1 round-trip + the read-sign-on-
  magnitude-zero invariant, MV reader truncation rejects for both
  modes, intra short-circuit (empty packet decodes), INTER_NOMV only
  consumes the MVMODE bit, INTER_MV decode + propagate, INTER_MV_LAST
  chain, INTER_MV_LAST2 3-way rotation, INTER_GOLDEN_MV does NOT
  update LASTs, all three chroma-averaging formulas (PF=0/2/4 — `round()`
  ties away from zero exercised explicitly), INTER_MV_FOUR with
  uncoded luma → chroma averaging uses (0, 0), INTER_MV_FOUR's `LAST1`
  update from the last coded luma block, all six input-validation
  rejects with `Display` rendering, truncation at MVMODE, the
  shared-`BitReader` chaining contract via
  `decode_macroblock_motion_vectors_inner`, intra-frame consumes-no-
  bits assertion, uncoded chroma block in step 3(g) keeps its
  (0, 0) default, and a `round_div`-against-spec check covering
  tie-away-from-zero. Brings the unit-test total to **207**
  (previously 179).

* **§7.4 Macro Block Coding Modes (2026-05-27, round 11).** New public
  `decode_macroblock_modes(packet, ftype, nmbs, nbs, bcoded,
  macro_block_to_luma_blocks) -> Result<Vec<MacroBlockMode>, Error>`
  transcribing the full §7.4 procedure of the Xiph Theora I
  Specification ("Macro Block Coding Modes"). Returns the per-macro-
  block `MBMODES` array as typed [`MacroBlockMode`] variants matching
  Table 7.18 (`InterNoMv`, `Intra`, `InterMv`, `InterMvLast`,
  `InterMvLast2`, `InterGoldenNoMv`, `InterGoldenMv`, `InterMvFour`).
  `MacroBlockMode::to_index` / `from_index` round-trip with the on-
  wire `Index` column.

  Intra frames short-circuit step 1: every macro block is `INTRA` and
  the packet payload is not consumed. Inter frames execute the step 2
  chain on the §5.2 MSb-first `BitReader`:

  * Step 2(a): read the 3-bit `MSCHEME`.
  * Step 2(b): if `MSCHEME=0`, read eight 3-bit `mi` values and
    populate `MALPHABET[mi] = MODE` in `MODE`-order (the on-wire
    permutation defines the alphabet).
  * Step 2(c): if `MSCHEME` is `1..=6`, set `MALPHABET` to the
    corresponding column of Table 7.19 (the six fixed alphabets
    encoded as `MALPHABETS_SCHEMES_1_TO_6`).
  * Step 2(d): for each macro block in coded order:
    * If at least one of its four luma blocks (caller-supplied
      `macro_block_to_luma_blocks[mbi]` of length 4) has
      `BCODED[bi]=1`, decode `MBMODES[mbi]` by either reading a
      Huffman-coded `mi` (Schemes 0..=6, see Table 7.19) and looking
      up `MALPHABET[mi]`, or by reading three bits directly (Scheme
      7).
    * Otherwise assign `MBMODES[mbi] = INTER_NOMV` and read no bits.

  Three new `Error` variants reject malformed inputs:
  `MacroBlockLumaMapLenMismatch { map_len, nmbs }` when the supplied
  mapping length disagrees with `nmbs`,
  `MacroBlockLumaBlockIndexOutOfRange { mbi, slot, bi, nbs }` when the
  mapping references a luma-block index `>= nbs`, and
  `UnknownMacroBlockModeCode` (defensive — unreachable for any
  conforming bitstream). All three carry `Display` arms describing
  the violating §7.4 step.

  The Huffman walk over Table 7.19 (`b0` / `b10` / … / `b1111111`) is
  decoded by `read_table_7_19_mi` as a unary-with-cap: up to six bits
  of unary, then a seventh disambiguation bit splitting `b1111110`
  (`mi=6`) and `b1111111` (`mi=7`). The seventh bit truncation is
  surfaced as `TruncatedHeader { field: "MBMODES_huffman_code" }`.

  As with §7.3, a crate-private `decode_macroblock_modes_inner` drives
  the procedure on an already-positioned `BitReader`, enabling an
  end-to-end frame decoder to chain §7.1 → §7.2 → §7.3 → §7.4 on a
  single reader without re-aligning to a byte boundary.

  Twenty new tests cover: the 0..=7 round-trip on `MacroBlockMode`,
  the intra short-circuit (no packet consumed), the two input-
  validation rejects, Scheme 7's 3-bit direct mode read for all four
  output modes, every Scheme 1..=6 column of Table 7.19 walked
  `mi=0..=7`, Scheme 0's on-wire alphabet permutation, the
  partially-coded luma macro block read path, the uncoded-luma
  `INTER_NOMV` step 2(d)ii path with zero bits read, MSCHEME / alphabet
  / Huffman-walk / direct-mode truncation rejects, the `b1111110` /
  `b1111111` disambiguation (Scheme 1 mi=6 vs mi=7), the shared-
  `BitReader` chaining contract, error-`Display` rendering, the no-
  bits-consumed path for fully-uncoded inter frames, and the
  `nmbs=0` degenerate edge case.

  §6.4.1 LFLIMS body is still blocked, but §7.4 — like every other
  procedure landed since round 5 — runs independently of the gap and
  can be exercised standalone.

* **§7.3 Coded Block Flags Decode (2026-05-25, round 10).**
  New public `decode_coded_block_flags(packet: &[u8], ftype: FrameType,
  nsbs: u32, nbs: u32, block_to_super_block: &[u32]) ->
  Result<Vec<u8>, Error>` transcribing the full §7.3 procedure of the
  Xiph Theora I Specification ("Coded Block Flags Decode"). Returns the
  per-block `BCODED` array of `0`/`1` flags marking which blocks are
  coded.

  Intra frames short-circuit step 1: every block is marked coded and
  the packet payload is not consumed. Inter frames execute the step 2
  chain on the §5.2 MSb-first `BitReader`:

  * Step 2(a)–(c): decode `SBPCODED` (length `NSBS`) via the §7.2.1
    long-run procedure — bits flag which super blocks are partially
    coded.
  * Step 2(d)–(f): decode `SBFCODED` via the §7.2.1 long-run procedure
    over the non-partially-coded subset (`NBITS = #{sbi:
    SBPCODED[sbi]=0}`), then distribute the bits into the
    `SBFCODED[sbi]` slots whose `SBPCODED[sbi]` is zero.
  * Step 2(g)–(h): tally the `block_to_super_block` mapping for blocks
    inside partially-coded super blocks (edge super blocks contribute
    fewer than 16 blocks, per the spec's own note) and decode the
    resulting `NBITS`-bit string via the §7.2.2 short-run procedure.
  * Step 2(i): for each block in coded order, inherit `BCODED[bi] =
    SBFCODED[sbi]` when `SBPCODED[sbi]=0`, or consume the next bit from
    the step 2(h) string when `SBPCODED[sbi]=1`.

  Two new `Error` variants reject malformed inputs:
  `BlockSuperBlockMapLenMismatch { map_len, nbs }` when the supplied
  mapping length disagrees with `nbs`, and
  `BlockSuperBlockIndexOutOfRange { bi, sbi, nsbs }` when the mapping
  references a super-block index `>= nsbs`. Both carry `Display` arms
  citing §7.3 for diagnostics.

  The procedure is split into `decode_coded_block_flags` (byte-aligned
  entry point) plus `decode_coded_block_flags_inner(&mut BitReader<'_>,
  …)` (crate-private) so a future end-to-end frame decoder can chain
  §7.1 → §7.2 → §7.3 on a shared reader without re-aligning to a byte
  boundary.

  17 new tests bring the total from 142 to 159: long-run / short-run
  encoder round-trip helpers (sanity-checking the test fixtures), intra
  short-circuit (every block coded; packet not consumed), the two input-
  validation rejects (`BlockSuperBlockMapLenMismatch`,
  `BlockSuperBlockIndexOutOfRange`) with `Display` rendering, inter
  all-super-blocks-not-coded, inter all-super-blocks-fully-coded, inter
  mixed-super-block-states with both `SBFCODED` and per-block paths
  exercised, the edge-super-block-with-fewer-than-16-blocks tally check
  (§7.3 step 2(g) note), the empty `NSBS = 0` short-circuit, mid-
  `SBPCODED` truncation rejection, intra-vs-inter arm independence, the
  shared-`BitReader` chaining contract via
  `decode_coded_block_flags_inner`, a single-partially-coded-super-block
  uncoded-block-subset case, and an interleaved (non-monotone)
  `block_to_super_block` mapping case.

  §7.3 is unblocked by the still-open §6.4.1 spec gap (docs-gap #944):
  §7.3 operates on a video-data packet's own payload and does not
  consume the setup-header body. The block-to-super-block mapping is
  taken as a caller-supplied argument because computing it requires the
  §2 super-block scan order; a later round will land the scan-order
  helper alongside §7.4 (Macro Block Coding Modes) and §7.6 (Block-
  Level `qi` Decode).

* **§7.2 Long-/Short-Run Bit Strings Decode (2026-05-25, round 9).**
  New public `decode_long_run_bit_string(bits: &[u8], nbits: u64) ->
  Result<Vec<u8>, Error>` and `decode_short_run_bit_string(bits: &[u8],
  nbits: u64) -> Result<Vec<u8>, Error>` transcribing §7.2.1 and §7.2.2
  of the Xiph Theora I Specification ("Run-Length Encoded Bit Strings").
  Each procedure decodes a sequence of `0`/`1` values whose run lengths
  are Huffman-coded against Table 7.7 (long-run, RSTART∈{1,2,4,6,10,18,
  34}, RBITS∈{0,1,1,2,3,4,12}, max RLEN = 4129) or Table 7.11 (short-
  run, RSTART∈{1,3,5,7,11,15}, RBITS∈{1,1,1,2,2,4}, max RLEN = 30) over
  the §5.2 MSb-first `BitReader`. The long-run procedure implements the
  VP3+ "read fresh BIT after RLEN=4129" exception (§7.2.1 step 12); the
  short-run procedure unconditionally toggles between runs (§7.2.2
  step 12 has no exception path).

  New public types/constants: `LONG_RUN_MAX = 4129`,
  `SHORT_RUN_MAX = 30`. One new `Error` variant
  (`RunLengthOverrun { len, nbits }`) surfaces the §7.2 step 10
  invariant ("LEN MUST be less than or equal to NBITS") with a
  `Display` arm.

  Implementation walks Table 7.7 / 7.11 one bit at a time against the
  `BitReader`; the in-tree tables are six-row / seven-row constants
  isolated from the decode logic so a transcription regression is
  caught by the dedicated `*_constants_match_table_*` tests. The
  procedures are split into `decode_*_bit_string` (byte-aligned entry
  point) plus `decode_*_bit_string_inner(&mut BitReader<'_>, u64)` for
  a future end-to-end frame decoder to chain §7.3 / §7.6 onto the same
  reader without re-aligning.

  24 new tests bring the total from 118 to 142: Table 7.7 / Table 7.11
  transcription against every row plus the `LONG_RUN_MAX` /
  `SHORT_RUN_MAX` derivations, the `NBITS = 0` empty-string short-
  circuit, single-record decode of every Table 7.7 / 7.11 entry at
  both range endpoints (13 long-run cases + 12 short-run cases), the
  toggle-between-runs invariant on both procedures, the long-run
  "read fresh BIT after RLEN=4129" exception (both fresh-0 and
  fresh-1 paths) plus the symmetric "RLEN=4128 still toggles" check,
  the short-run "RLEN=30 still toggles" check (no exception path),
  truncation rejects at the initial BIT / mid-Huffman-walk / mid-ROFFS
  field boundaries on both procedures, the `RunLengthOverrun` reject
  on both procedures with `Display` rendering, a long-run byte-
  boundary-crossing decode, and a realistic 16-bit short-run super-
  block decode covering four toggled runs.

  §7.2 is unblocked by the still-open §6.4.1 spec gap (docs-gap #944):
  §7.2 operates on a video-data packet's own payload and does not
  consume the setup-header body. §7.3 (Coded Block Flags Decode) is the
  natural next step: it calls `decode_long_run_bit_string` for the
  super-block partial-coding map and `decode_short_run_bit_string`
  inside each partially-coded super block for the per-block flags.

* **§7.1 Frame Header Decode (2026-05-25, round 8).**
  New public `decode_frame_header(packet: &[u8], first_frame: bool) ->
  Result<TheoraFrameHeader, Error>` transcribing the full §7.1
  procedure of the Xiph Theora I Specification ("Frame Header Decode").
  The procedure decodes the first bits of every video-data packet on
  the §5.2 MSb-first `BitReader`: step 1 reads the 1-bit data-packet
  flag (and surfaces `Error::NotDataPacket` if the high bit is set,
  identifying a §6-series header packet); step 2 reads the 1-bit
  `FTYPE` (`Intra=0` / `Inter=1` per Table 7.3) and enforces the first-
  frame Intra mandate (`Error::FirstFrameMustBeIntra` for the violation);
  steps 3–6 unroll the `MOREQIS` chain into a 1..=3-element `qis` list;
  and step 7 reads the 3-bit reserved trailer on intra frames
  (`Error::FrameReservedBitsNonZero` for any non-zero value).

  New public types: `FrameType` (Intra/Inter), `TheoraFrameHeader
  { ftype, qis }` with `nqis()` accessor, `MAX_FRAME_QIS = 3`.
  Three new `Error` variants (`NotDataPacket` /
  `FirstFrameMustBeIntra { ftype }` / `FrameReservedBitsNonZero { bits }`)
  with `Display` arms citing the §7.1 step they correspond to.

  19 new tests bring the total from 99 to 118: intra single/two/three-
  `qi` happy paths (with the 63 upper-boundary on every slot), inter-
  frame happy paths (single and three `qi`), `first_frame` enforcement
  + inter-after-keyframe allowed path, header-packet rejection,
  step 7 reserved-bits rejection on every non-zero 3-bit pattern,
  truncation at the packet-type bit / `MOREQIS[0]` / `QIS[2]` field
  boundaries, "doesn't consume past header" sentinel-byte check, error
  `Display` rendering for all three new variants, `FrameType` Table 7.3
  numeric mapping, the `MAX_FRAME_QIS = 3` constant, and a five-case
  independent-slot round-trip across byte boundaries.

  §7.1 is unblocked by the still-open §6.4.1 spec gap (docs-gap #944):
  §7.1 operates on a video-data packet's own payload and does not
  require the setup-header body decode to have completed. A future
  end-to-end frame decoder will chain §7.1 → §7.2 (run-length bit
  strings) → §7.3 (coded block flags) → … on a shared bit reader; the
  inner `decode_frame_header_inner` is split out for that purpose.

* **§6.4.4 DCT Token Huffman Tables (2026-05-24, round 7).**
  New public `decode_dct_token_huffman_tables(bits: &[u8]) ->
  Result<Box<[HuffmanTable; NUM_HUFFMAN_TABLES]>, Error>` transcribing
  the full §6.4.4 procedure of the Xiph Theora I Specification ("DCT
  Token Huffman Tables"). The procedure decodes the 80-table setup-
  header payload that drives §7.7's DCT-token decode: each table is
  described as a binary tree where every `1`-bit `ISLEAF` flag is
  followed by a 5-bit `TOKEN` value at every leaf, with up to 32
  entries per table and a 32-bit maximum code length. The
  implementation walks each table iteratively (explicit DFS stack, no
  host recursion — addresses the spec's own §6.4.4 caveat about
  recursion depth on adversarial inputs) and emits leaves in the
  spec's `0`-before-`1` order. Each table is materialised both as a
  `Vec<HuffmanEntry { code, len, token }>` (for inspection / round-
  trip testing) and as a flat binary-tree representation suitable for
  the per-bit lookup §7.7.2 will use. Two new `Error` variants
  (`HuffmanCodeTooLong` / `HuffmanTableFull`) reject the two
  step 1(b) / step 1(d)i undecodable-stream paths with the offending
  `hti` for diagnostics, each with a `Display` arm.

  New public types: `HuffmanEntry`, `HuffmanTable` (with
  `HuffmanTable::lookup(code, len) -> Option<u8>` for bit-string
  lookup including the degenerate single-leaf-at-root case),
  `NUM_HUFFMAN_TABLES = 80`, `MAX_HUFFMAN_ENTRIES = 32`.

  11 new tests bring the total from 88 to 99: trivial single-leaf
  tables on every slot, the balanced 32-leaf table with full lookup
  coverage, variable-length codes in the order the §6.4.4 recursion
  visits the leaves, independent tables across all 80 slots,
  truncated-ISLEAF rejection, hand-crafted truncated-TOKEN rejection
  (single-byte payload exercising the within-leaf truncation path),
  the step 1(b) code-too-long reject via a depth-33 left-spine
  construction, the step 1(d)i 33-entry reject via a balanced
  32-leaf left subtree plus a right-child 33rd leaf, multi-table
  truncation reporting the correct field, `Error` `Display` rendering
  for both new variants, and `HuffmanTable::lookup` returning `None`
  for codes at the wrong length.

* **§6.4.3 Computing a Quantization Matrix (2026-05-24, round 6).**
  New public `compute_quantization_matrix(params: &QuantizationParameters,
  qti: usize, pli: usize, qi: usize) -> Result<QuantizationMatrix, Error>`
  transcribing the full §6.4.3 procedure of the Xiph Theora I
  Specification ("Computing a Quantization Matrix"). It consumes the
  §6.4.2 `QuantizationParameters` and interpolates a 64-element
  natural-order quantization matrix for a `(qti, pli, qi)` selector:
  * Steps 1–3: locate the quant range bracketing `qi`, deriving
    `QISTART` / `QIEND` from the cumulative `QRSIZES` sums.
  * Steps 4–5: pick the two end-point base-matrix indices `bmi` /
    `bmj` from `QRBMIS`.
  * Step 6(a): linearly interpolate `BM[ci]` between the two base
    matrices using the spec's `//`-rounded formula.
  * Steps 6(b)–6(e): apply the Table 6.18 `QMIN` floor, the DC
    (`DCSCALE`) vs AC (`ACSCALE`) scale, the `//100` and `*4`
    scaling, and the `max(QMIN, min(..., 4096))` clamp.

  New typed `QuantizationMatrix { values: [u16; 64] }` output and a
  crate-private `qmin_table` helper for Table 6.18. Three new `Error`
  variants (`QuantTypeIndexOutOfRange` / `QuantPlaneIndexOutOfRange` /
  `QuantIndexOutOfRange`) reject out-of-range `qti` / `pli` / `qi`
  selectors, each with a `Display` arm.

  9 new tests bring the total from 79 to 88: corner-base-matrix
  selection at `qi=0` / `qi=63`, midpoint interpolation within a
  single range, two-range interpolation plus a boundary-qi
  consistency check, the Table 6.18 `QMIN` floors (intra + inter), the
  4096 ceiling clamp, direct `qmin_table` values, out-of-range
  selector rejects, per-`(qti, pli)` selector-wiring isolation, and an
  end-to-end chain that decodes a synthesized §6.4.2 payload and feeds
  it straight into `compute_quantization_matrix`.

  All §6.4.3 arithmetic uses non-negative operands, so the spec's
  `//` reduces to ordinary integer division. The §6.4.1 spec gap
  (still open, docs-gap #944) does not block this section: §6.4.3
  operates purely on the §6.4.2 outputs.

* **§6.4.2 Quantization Parameters Decode (2026-05-24, round 5).**
  New public `decode_quantization_parameters(bits: &[u8]) ->
  Result<QuantizationParameters, Error>` transcribing the full §6.4.2
  procedure of the Xiph Theora I Specification ("Quantization
  Parameters Decode") onto the §5.2 MSb-first `BitReader`:
  * Steps 1–4: the 4-bit-NBITS-prefixed `ACSCALE` / `DCSCALE` tables
    (64 entries each).
  * Step 5: the 9-bit `NBMS`, incremented per spec and validated
    `<= 384` (`Error::TooManyBaseMatrices` otherwise).
  * Step 6: the `NBMS × 64` base-matrix array `BMS`.
  * Step 7: the per-`(qti, pli)` quant-range tables `NQRS` /
    `QRSIZES` / `QRBMIS`, including the NEWQR/RPQR copy-set selection
    logic (both the "previous quantization type, same plane" and
    "most recent set" sources) and the `ilog(NBMS−1)` /
    `ilog(62−qi)` variable field widths. Undecodable streams return
    `Error::BaseMatrixIndexOutOfRange` (a `QRBMIS >= NBMS` at step
    7(a)ivC) or `Error::QuantRangeOverflow` (range sizes overshooting
    63 at step 7(a)ivI).

  New typed `QuantizationParameters { ac_scale, dc_scale,
  num_base_matrices, base_matrices, num_quant_ranges,
  quant_range_sizes, quant_range_base_matrix_indices }` and a free
  `ilog` helper matching the spec's Notation and Conventions
  definition. Three new `Error` variants
  (`TooManyBaseMatrices` / `BaseMatrixIndexOutOfRange` /
  `QuantRangeOverflow`) with `Display` arms.

  11 new tests bring the total from 68 to 79: `ilog` against every
  spec worked example, synthesized-payload round-trips (single-range,
  two-range size sums, NEWQR/RPQR copy variants on INTRA and INTER
  planes), the `NBMS = 384` boundary, the three undecodable-stream
  rejects, and mid-field truncation. A test-only MSb-first bit writer
  mirrors `BitReader` to build the fixtures bit-exactly.

  **§6.4.1 spec gap still blocks the end-to-end `parse_setup_header`
  body.** §6.4.5 step 2 (§6.4.1 LFLIMS) precedes step 3 (§6.4.2), so
  `parse_setup_header` continues to surface
  `Error::SetupHeaderBodyNotImplemented`; the §6.4.2 procedure is
  exposed as a standalone entry point and will be chained onto a
  shared bit reader once §6.4.1's procedure body is recovered.

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
