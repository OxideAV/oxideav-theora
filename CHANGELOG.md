# Changelog

All notable changes to `oxideav-theora` are recorded here.

## [Unreleased]

### Added

- §7.11 step 2 (empty / zero-byte packet) state synthesiser (round
  260). New public function
  `synthesize_empty_packet_frame_state(nbs: usize) ->
  Result<EmptyPacketFrameState, Error>` materialises the four
  deterministic outputs the spec mandates for the "Otherwise"
  branch — step 2(a) `FTYPE := 1` (`FrameType::Inter`), step 2(b)
  `NQIS := 1`, step 2(c) `QIS[0] := 63`, step 2(d) `BCODED[bi] := 0`
  for every block — without consuming any bitstream. The typed
  `EmptyPacketFrameState { ftype, nqis, qis, bcoded }` carries
  ownership of the synthesised arrays so the §7.11 driver's
  downstream steps can mutate `BCODED` through shared references
  the same way the step 1 chain's outputs do. Convenience methods
  `qi0()` (always `63`) and `nbs()` keep step 1 / step 2 call sites
  symmetric. `From<&EmptyPacketFrameState>` projects to the existing
  [`TheoraFrameHeader`] view (`FTYPE` + `QIS` slice) for callers
  that only need the §7.1 surface. `nbs == 0` is a structural reject
  via the new `Error::EmptyPacketFrameStateZeroNbs` variant (§6.2
  step 23 derives `NBS >= 1` for any well-formed Theora stream).
  6 new tests (total now 475): step-2 hard-coded values across
  multiple `nbs`; `BCODED` all-zero sized to `nbs`; zero-NBS reject;
  `TheoraFrameHeader` projection; classifier-then-synthesiser entry
  chain; `qi0()` tracks post-construction `qis` mutation. The §7.11
  entry-shape now covers steps 2 / 3 / 4 / 7 / 8 end-to-end; step 1
  chain wiring (§7.1 → §7.3 → §7.4 → (§7.5.2) → §7.6 → §7.7.3 →
  §7.8.2 on a shared bit reader) and step 5 / step 6 dispatch into
  the existing `reconstruct_frame` / `loop_filter_frame` drivers
  remain pending.
- §7.11 step 7 + step 8 reference-frame promotion (round 256).
  New owned `ReferenceFrameStore` wraps the six long-lived
  reference-frame planes (`GOLDREFY` / `GOLDREFCB` / `GOLDREFCR`
  + `PREVREFY` / `PREVREFCB` / `PREVREFCR`) the §7.11 driver hands
  forward to the next frame's §7.9.4 reconstruction call. Three
  constructors: `zeroed(dims_y, dims_c)` for fresh allocation,
  `from_reference_plane_dimensions(ref_dims)` for the convenience
  bridge from the round-250 `reference_plane_dimensions_from_ident`
  output, and `new(...)` for buffer-recycling decoders that bring
  their own allocations. The `promote_from_reconstructed(rec,
  ftype)` method executes step 7 (`if FTYPE == 0 then GOLDREF* :=
  REC*`) and step 8 (`PREVREF* := REC*` unconditionally) via
  `copy_from_slice`, with shape + length validation against the
  store's declared `(RPYW, RPYH)` / `(RPCW, RPCH)` geometry.
  `as_reference_plane_set()` returns a borrowed
  `ReferencePlaneSet<'_>` ready for the next frame's §7.9.4 call.
  Companion helpers `frame_type_from_ftype(ftype: u8) ->
  Result<FrameType, Error>` and `frame_type_as_ftype(ft: FrameType)
  -> u8` decode the §7.1 / §7.11-step-2 raw 1-bit `FTYPE` field
  into the existing `FrameType` enum with a typed
  `Error::FrameTypeOutOfRange` reject for out-of-range bytes. Four
  new `Error` variants: `FrameTypeOutOfRange`,
  `ReferenceFrameStoreDimensionsOverflow`,
  `ReferenceFrameStorePlaneLenMismatch`,
  `ReferenceFrameStoreDimensionMismatch`. 13 new tests (total now
  469): FrameType round-trip + 254 rejected raw values; zeroed +
  from-ref-dims allocation; `new` rejects on Y-plane and Cb-plane
  length mismatch; intra promotion writes both Golden and Previous;
  inter promotion writes only Previous (Golden preserved);
  intra-then-inter sequence matches spec semantics (Golden frozen
  at intra-frame pixels, Previous tracking latest); Y / Cr
  dimension mismatch errors carry the right `plane` tag;
  reconstructed-plane `len()` mismatch surfaces as `rec_y`;
  `as_reference_plane_set` round-trip exposes the promoted bytes
  to the next frame's §7.9.4 driver; promotion reuses existing
  buffers without reallocating (pointer equality before/after).
- §7.11 Complete Frame Decode entry-shape primitives (round 250).
  Two new public entry points wire the first plumbing for the
  §7.11 driver: `reference_plane_dimensions_from_ident(ident:
  &TheoraIdentHeader) -> ReferencePlaneDimensions` implements
  step 3 (`RPYW = 16 * FMBW`, `RPYH = 16 * FMBH`) and step 4 /
  Table 7.89 (chroma keyed by `PF`: `PF=0 → 8*FMBW × 8*FMBH`,
  `PF=2 → 8*FMBW × 16*FMBH`, `PF=3 → 16*FMBW × 16*FMBH`), and
  `classify_frame_decode_packet(packet: &[u8]) -> FrameDecodePacket`
  implements the step 1 / step 2 branch — a zero-byte packet
  becomes `FrameDecodePacket::Empty` (step 2: synthesise `FTYPE =
  1`, `NQIS = 1`, `QIS[0] = 63`, all-zero `BCODED`; consume no
  bits), a non-empty packet becomes `FrameDecodePacket::Data` for
  the §7.1 → §7.3 → §7.4 → (§7.5.2) → §7.6 → §7.7.3 → §7.8.2
  chain that subsequent rounds will wire together. Both helpers
  are pure functions with no per-frame state. New public types:
  `ReferencePlaneDimensions { rpyw, rpyh, rpcw, rpch }` carrying
  the spec mnemonics one-for-one, and `FrameDecodePacket<'a> {
  Empty, Data(&'a [u8]) }` with `is_empty()` / `data()`
  inspectors. 9 new tests (total now 456): luma extent across
  all three `PF` values, each Table 7.89 row by chroma shape
  (`PF=0` halved both axes, `PF=2` horizontal-only, `PF=3`
  unscaled), `u16::MAX` boundary on `FMBW` / `FMBH` (no `u32`
  overflow), independence from the picture-region offsets
  (`PICW` / `PICH` / `PICX` / `PICY` do not affect reference-plane
  geometry — those govern display-time cropping per §A.1, not
  reconstruction), zero-byte vs non-empty packet classification,
  and the single-byte boundary case (any `len >= 1` is the
  step 1 path).
- §7.10.3 Complete Loop Filter raster-order driver (round 244).
  New public entry point `loop_filter_frame(lflims, qis, bcoded,
  pli_of_block, bx_of_block, by_of_block, plane_y, plane_cb,
  plane_cr, planes)` walks every coded block in raster order over
  each plane and dispatches up to four edge filters per block via
  the §7.10.1 / §7.10.2 primitives that landed in round 241. The
  driver reads `L = LFLIMS[QIS[0]]` once per frame (step 1) and
  honours the spec's neighbour-handover rule on the right / top
  edges (step 2(a) vii / viii filter the boundary only when the
  right / above neighbour has `BCODED[bj] == 0`, since a coded
  neighbour would later filter the shared edge as its own left /
  bottom edge in step 2(a) v / vi). The API exposes two new public
  helpers: `LoopFilterPlaneInput` (per-plane raster→coded-order
  block index map + plane block extent) and `LoopFilterPlanes`
  (three plane buffers + `RPYW` / `RPYH` / `RPCW` / `RPCH`
  dimensions). Eight new typed error variants —
  `LoopFilterFrameQisEmpty`, `LoopFilterFrameQiOutOfRange`,
  `LoopFilterFrameBlockLenMismatch` (discriminating
  `BCODED` / `pli_of_block` / `bx_of_block` / `by_of_block` via
  `LoopFilterFrameBlockSlice`), `LoopFilterFrameRasterLenMismatch`,
  `LoopFilterFrameRasterEntryOutOfRange`,
  `LoopFilterFramePliOutOfRange`, and
  `LoopFilterFrameBlockOutOfPlane` — each with §7.10.3-tagged
  Display strings. 17 new tests (total now 447): per-frame
  identity on a single-block plane (all four boundary tests
  short-circuit), shared-edge filtered-exactly-once for two
  coded blocks (horizontal + vertical mirror cases, bit-exact
  pre/post pixel deltas verified against hand-computed `lflim`
  output), step 2(a) vii / viii triggered by an uncoded
  middle neighbour, plane independence (Cb-only edits leave Y and
  Cr untouched), and every error variant exercised through both
  pattern-matching and `Display` rendering. Bit-exact against
  hand-derived `lflim(R, L)` values for `R = 10, L = 30` ⇒
  `lflim = 10` (linear segment) and `R = 15, L = 30` ⇒
  `lflim = 15`.
- §7.10.1 / §7.10.2 loop-filter edge primitives + `lflim()` response
  (round 241). New public entry points
  `lflim(r: i32, l: u8) -> i32` (the §7.10 piecewise non-linear
  response function with peaks at `(±L, ±L)` and zero-crossings at
  `±2 * L`),
  `horizontal_loop_filter_edge(recp, plane_w, plane_h, fx, fy, l)`
  (§7.10.1 — applies the 4-tap horizontal edge filter to a 4×8
  footprint, writing the two middle columns), and
  `vertical_loop_filter_edge(recp, plane_w, plane_h, fx, fy, l)`
  (§7.10.2 — applies the 4-tap vertical edge filter to an 8×4
  footprint, writing the two middle rows). Both edge primitives
  operate on the same lower-left row-major plane buffer layout
  emitted by `reconstruct_frame` (§7.9.4 round-233 driver) and use
  the central `lflim` response so a `LFLIMS[qi0] = 0` limit value
  collapses the filter to identity (matching §B.2's VP3 high-`qi`
  trailing zeros). Three new `Error` variants — 
  `LoopFilterHorizontalFootprintOutOfPlane`,
  `LoopFilterVerticalFootprintOutOfPlane`, and
  `LoopFilterPlaneBufferLenMismatch` — pin the footprint /
  buffer-length guards with §7.10-tagged Display strings. 19 new
  tests (total now 430): `lflim` piecewise breakpoints at `L = 5`
  covering every distinguished arm of the function, `lflim` with
  `L = 0` collapses to identically zero, antisymmetry of `lflim` in
  its first argument across `L ∈ {0, 1, 3, 7, 25, 127}` and
  `R ∈ [-300, 300]`, constant-input identity for both edge filters,
  step-edge softening with `R = 5` / delta 5 inside the central
  segment (both horizontal and vertical), `LFLIMS = 0` disables the
  filter on a strong step, outside-of-band `|R| >= 2L` returns
  identity, plane-bounds clamp at 255 with `(p2 - delta) = 275`,
  outside-footprint pixels left untouched for both filters, all
  four out-of-plane footprint rejects (right / top for each filter),
  plane-buffer-length mismatch rejected before either filter
  dereferences the buffer, and `Display` rendering of the three new
  error variants. The §7.10.3 ordering driver (raster-order walk
  consuming `BCODED` + `(BX, BY)` per block) is the natural next
  round: these primitives are the per-edge body it dispatches to.

- §2.3 / §2.4 coded-order resolver (round 238). New public
  iterators `PlaneBlockCodedOrder` and `PlaneMacroBlockCodedOrder`
  walk one colour plane in spec-defined coded order: super-blocks
  in raster order (lower-left origin), Hilbert curve inside each
  super-block per Figure 2.4 (16 blocks) and Figure 2.6 (4 macro-
  blocks). Each iterator item carries the plane-local
  super-block index, the slot inside the super-block, and the
  plane-local `(bx, by)` or `(mbx, mby)` coordinates. Edge
  super-blocks (right / top of the plane) follow §2.3's
  "ordering still used, blocks outside the frame boundary
  omitted" rule: the iterator advances through every slot of the
  over-padded super-block and filters out the slots whose
  coordinates lie outside the plane's block extent.
  Supporting types: `PlaneBlockDims { mb_w, mb_h }` (plane
  geometry in macroblocks) with `luma_from_ident` /
  `chroma_from_ident` constructors that derive the dimensions
  from a parsed identification header per the `PF` rules of
  Table 6.4, plus `sb_w`, `sb_h`, `block_w`, `block_h`,
  `mb_count`, `block_count` accessors; `CodedBlockPosition`
  (one §2.3 walk step: `sb_index`, `block_in_sb`, `mb_in_sb`,
  `block_in_mb`, `bx`, `by`) and `CodedMacroBlockPosition` (one
  §2.4 walk step: `sb_index`, `mb_in_sb`, `mbx`, `mby`).
  Five new tests (total now 411): the 240×48 worked example of
  §2.3 page 8 pinning the block walk against the spec's table
  of plane-local coded indices (bottom row 0/1/14/15…112/113,
  SB-row 1's bottom row 120/121/126/127…176/177, and SB-row 1
  top corners 178/179); a 4×4-block single-super-block trace
  pinning the inner Figure 2.4 Hilbert sequence verbatim; a
  3×3-macroblock (6×6-block) plane exercising the edge-filter
  on three over-padded super-blocks (right, top, and the
  top-right corner SB that holds only 4 in-plane blocks); the
  240×48 worked example of §2.4 page 11 pinning the macro-block
  walk; the `PF` axis-by-axis halving rules of
  `chroma_from_ident` across 4:2:0 / 4:2:2 / 4:4:4; and a
  zero-dim degenerate-input guard.

- §7.9.4 frame-level driver (round 233). New public entry point
  `reconstruct_frame(bcoded, mb_modes, mvects, coeffs, ncoeffs,
  qiis, pli_of_block, bx_of_block, by_of_block, mbi_of_block, qis,
  params, refs, dims_y, dims_cb, dims_cr) -> Result<ReconstructedFrame,
  Error>` walks the §7.9.4 step 2 loop over `bi in 0..NBS`,
  dispatches each block to the round-23 `reconstruct_block` body,
  and tiles the resulting 8×8 samples into three flat output plane
  vectors per §7.9.4 step 2(f)iv-vi. Per-plane `(pli, BX, BY, mbi)`
  geometry is caller-supplied (matching the §7.8.2 convention)
  rather than re-derived inside the driver, so the §2.3 coded-order
  Hilbert mapping is the caller's responsibility. Output planes
  are returned in lower-left row-major order per the §2.1 origin
  convention.
  Supporting types: `PlaneDimensions { width: u32, height: u32 }`
  (per-plane output dimensions), `ReconstructedFrame { samples_y,
  samples_cb, samples_cr, dims_y, dims_cb, dims_cr }` (the three
  reconstructed planes plus their geometry), and
  `ReconstructFrameBlockSlice` (a typed tag identifying which
  per-block slice failed a length check).
  Five new error variants: `ReconstructFrameBlockLenMismatch`
  (per-block slice length != `nbs`),
  `ReconstructFramePliOutOfRange` (`pli_of_block[bi] > 2`),
  `ReconstructFrameMbIndexOutOfRange` (`mbi >= nmbs` on a coded
  block), `ReconstructFrameBlockOutOfPlane` (block 8×8 footprint
  escapes destination plane), and
  `ReconstructFramePlaneDimensionsOverflow` (plane `width * height`
  overflows `usize`).
  Eight new tests (total now 405): an all-intra all-zero-coefficient
  4:2:0 frame producing three planes of all-128 across a four-luma
  + one-Cb + one-Cr geometry; an uncoded luma block confirmed to
  sample `PREVREFY` at the same coordinates while the other three
  luma blocks still hit the intra path; the four reject paths and
  a five-variant Display tag check; and a determinism guard over
  a mixed coded/uncoded input. The frame is returned
  pre-loop-filter — §7.10 is the next clause.

- §7.9.4 per-block reconstruction (round 23). New public entry
  point `reconstruct_block(inputs, pli, bx_origin, by_origin,
  qis, params, refs) -> Result<ReconstructedBlock, Error>`
  transcribing step 1 plus steps 2(a)–2(f) of §7.9.4 of the Xiph
  Theora I Specification for a single block. The procedure:
  resolves `qi0` from `QIS[0]`; on the coded branch (step 2(d))
  picks `qti` from the mode (`Intra` = 0, else 1), looks up
  `rfi` via `reference_frame_for_mb_mode` (Table 7.46), routes
  intra to §7.9.1.1 (constant 128), routes inter to
  `ReferencePlaneSet::pick` (Table 7.75), splits the per-block
  MV into the §7.9.1.3 truncate-toward-zero / truncate-away-from-
  zero pair (the integer-MV case collapses MVX==MVX2 so step
  2(d)vi.F routes to §7.9.1.2 whole-pixel), takes the DC-only
  shortcut when `NCOEFFS[bi] < 2` (build the DC quant matrix,
  compute `DC = (COEFFS[0] * QMAT[0] + 15) >> 5` truncated to
  i16, fill RES uniformly) or the full path when `NCOEFFS[bi] >=
  2` (build both DC + AC matrices from §6.4.3, dequantize via
  §7.9.2, inverse DCT via §7.9.3.2); on the uncoded branch (step
  2(e)) the predictor is a zero-MV whole-pixel read of the
  Previous reference and the residual is zero. Step 2(f) sums
  PRED + RES, clamps to `0..=255`, and writes the result into
  `ReconstructedBlock.samples`.
  Supporting types: `ReconstructedBlock { samples: [[u8; 8]; 8] }`
  (the 8×8 output tile), `ReconstructBlockInputs` (the per-block
  view of the §7.9.4 input tables — `bcoded`, `mb_mode`,
  `mvect`, `coeffs_zz`, `ncoeffs`, `qii`), and `ReferencePlaneSet`
  (the six (rfi, pli) reference planes Table 7.75 enumerates,
  with a `pick` lookup that rejects Intra-routed calls as an
  internal-invariant guard).
  Four new error variants: `ReconstructPlaneIndexOutOfRange`
  (`pli > 2`), `ReconstructEmptyQis` (step 1 has nothing to
  read), `ReconstructQiiIndexOutOfRange` (step 2(d)viii.A's
  `QIS[QIIS[bi]]` overrun), and
  `ReconstructIntraInterBranchMismatch` (the Intra-routed
  `ReferencePlaneSet::pick` reject — `reconstruct_block` itself
  routes Intra before consulting `refs`, so this is reachable
  only from a direct `pick` call). Each variant carries a
  `§7.9.4`-tagged `Display`.
  Eighteen new tests (total now 397): the Intra-DC-zero-residual
  all-128 invariant, an Intra-DC-positive PRED+DC offset, both
  clamp paths (high → 255, low → 0), the uncoded `Previous-Y`
  co-located copy, plane-specific routing (uncoded Cb →
  Previous-Cb, uncoded Cr → Previous-Cr), the Inter-Golden
  Table 7.46 routing (`InterGoldenNoMv` → Golden-Y), the
  Inter-Previous Table 7.46 routing (`InterMv` → Previous-Cr at
  `pli = 2`), the whole-pixel-MV positional shift across the 8×8
  tile, the all-zero-COEFFS full-DCT path collapsing to PRED,
  the three reject paths, the Table 7.75 `pick` lookup over all
  six (rfi, pli) pairs, a pinned step 2(d)vii.B formula
  cross-check (`DC = (COEFFS[0] * qmat_dc[0] + 15) >> 5` matched
  against a hand-computed expectation), `Display` rendering for
  all four new error variants, and a determinism guard for the
  per-block path. The §7.9.4 frame-level driver (the `bi in
  0..NBS` raster walk that consults per-plane `(BX, BY)`
  geometry and the macro-block-of-block mapping) is intentionally
  deferred so the per-block body can be audited against §7.9.4
  steps 2(a)–2(f) in isolation.

- §7.9.3 The Inverse DCT (round 22). Two new public entry points
  transcribing the normative parts of §7.9.3 of the Xiph Theora I
  Specification:
  - `inverse_dct_1d(y: &[i16; 8]) -> [i16; 8]` walks the §7.9.3.1
    55-step procedure verbatim — eight i32-typed `T[]` slots, the
    `Truncate to i16` narrowing steps before each `C4 * T[i] >> 16`
    multiply, and the final `as i16` truncation on every output
    `X[i]`. The constants `C3`/`S3`, `C4`, `C6`/`S6`, `C7`/`S7` come
    from Table 7.65 (16-bit approximations of cosines and sines
    of `iπ/16`).
  - `inverse_dct_2d(dqc: &[i16; 64]) -> [[i16; 8]; 8]` applies the
    §7.9.3.2 two-pass driver: pass 1 walks `ri ∈ 0..=7`, extracts
    each row of natural-order `DQC`, applies `inverse_dct_1d`, and
    stores the result in `RES[ri][ci]`; pass 2 walks `ci ∈ 0..=7`,
    extracts each column of `RES`, applies the 1D transform again,
    and finalises with `(X[ri] + 8) >> 4` to scale out the two 1D
    passes' factor-of-four expansion. The `(+8) >> 4` step
    implements the §7.9.3 final-rounding rule ("division by 16,
    ties rounded toward positive infinity"): Rust's signed `>>`
    truncates toward negative infinity, and the `+8` bias shifts
    the tie boundary so the rule comes out correct.
  Thirteen new tests (total 379): both procedures on all-zero
  input, DC-only positive and negative input on the 1D path
  (pinned to `(C4 * Y[0]) >> 16 = 724` / `−725` with literal
  cross-checks), extreme `i16::MAX` / `i16::MIN` input
  determinism, the §7.9.3.2 DC-only flat-block invariant
  (`DQC[0] = 1024 → RES[ri][ci] = 32`), the negative-DC parallel
  (`DQC[0] = −1024 → RES[ri][ci] = −32`), the small-DC rounds-to-
  zero case (`DQC[0] = 16 → RES = 0`), the row-vs-column AC-only
  excitation transpose-invariant (proving `DQC[1]` and `DQC[8]`
  produce transposed `RES`), and the Table 7.65 constants
  pinned to their integer values. §7.9.3.3 (the 1D Forward DCT
  for encoder convenience) is explicitly non-normative per the
  spec and is deferred to a later encoder round; this commit
  lands only the §7.9.3.1 / §7.9.3.2 normative decoder path.

- §7.9.1 Predictors (round 21). Three new public entry points
  transcribing the three §7.9.1.x predictor procedures of the Xiph
  Theora I Specification:
  - `compute_intra_predictor() -> [[u8; 8]; 8]` returns the constant
    128 tile from §7.9.1.1 ("The Intra Predictor"). The Theora prose
    rationale is that 128 centres the range of legal 8-bit DC values
    around zero so INTRA blocks fit signed arithmetic with no bias.
  - `compute_whole_pixel_predictor(refp, BX, BY, MV) -> [[u8; 8]; 8]`
    transcribes §7.9.1.2 ("The Whole-Pixel Predictor"): for each
    `by`/`bx` in 0..=7 it computes `ry = BY + MVY + by`, `rx = BX +
    MVX + bx`, clamps both into the reference plane's `[0, RPH-1]` /
    `[0, RPW-1]` window, and assigns `PRED[by][bx] = REFP[ry][rx]`.
  - `compute_half_pixel_predictor(refp, BX, BY, MV1, MV2) -> [[u8; 8];
    8]` transcribes §7.9.1.3 ("The Half-Pixel Predictor"): two
    clamped reference samples per output, averaged with `(s1 + s2) >>
    1` (truncating toward negative infinity per the spec).
  Supporting types: `ReferencePlane { rpw, rph, samples }` carries a
  reference-plane view alongside the row-major flat byte slice;
  `ReferencePlane::new(rpw, rph, samples)` validates `rpw * rph` against
  the buffer length plus the non-zero-dimension and overflow rejects
  (three new error variants: `ReferencePlaneLenMismatch`,
  `ReferencePlaneZeroDimension`, `ReferencePlaneDimensionsOverflow`).
  Companion helper `split_half_pixel_motion_vector(double_mvx,
  double_mvy) -> Option<(MotionVector, MotionVector)>` produces the
  truncate-toward-zero / truncate-away-from-zero pair the §7.9.1.3
  driver expects when the §7.9.4 chroma derivation halves an MV with
  an odd integer-doubled value. Twenty-eight new tests (total 366):
  intra constant-128 invariants, DC-centering math, whole-pixel zero-
  MV ramp-block copy, positive-MV offsetting, all four clamping
  edges (bottom, right, left, top — covered via combined-corner case
  and all-negative case), constant-plane round-trip, distinct-origin
  isolation, half-pixel identical-vectors degeneration to whole-pixel,
  adjacent-column and adjacent-row averaging, truncation-toward-
  negative-infinity worked example (the (2+3)>>1 = 2 spec case),
  independent per-vector clamping, two-samples-only invariant,
  even/odd/zero/mixed-signs/out-of-range branches of the split
  helper, the split-then-predict round-trip, plus all three reject-
  path tests for `ReferencePlane::new` and their `Display` rendering.

- §7.9.2 Dequantization (round 20). New public `dequantize_block(
  coeffs_zz, qmat_dc, qmat_ac) -> [i16; 64]` transcribing steps 2–6
  of §7.9.2 of the Theora I Specification: scales `COEFFS[bi][0]` by
  `QMAT_DC[0]` for the DC term, then for each natural-order
  coefficient `ci in 1..=63` maps `ci` to its zig-zag index `zzi` via
  Figure 2.8 (the new `ZIGZAG_NATURAL_TO_ZIGZAG: [u8; 64]` constant)
  and scales `COEFFS[bi][zzi]` by `QMAT_AC[ci]`. Both products are
  truncated to a signed 16-bit two's-complement representation per
  the spec's "discarding the higher-order bits" rule via Rust's
  well-defined `i32 -> i16 -> i32` narrowing. Convenience helper
  `dequantize_block_from_params(coeffs_zz, params, qti, pli, qi0,
  qi)` builds both quantization matrices inline via §6.4.3 to match
  the §7.9.2 input list verbatim; production callers will typically
  pre-build the at-most-six per-plane matrices and pass them into
  `dequantize_block` directly, per the §7.9.2 narrative's efficiency
  note. Eighteen new tests (zig-zag permutation invariants, DC/AC
  multiplication, ci-vs-zzi addressing, signed coefficient handling,
  16-bit overflow truncation, params-driven matrix construction,
  error propagation for `qti > 1` / `pli > 2` / `qi > 63`,
  independent `qi0`/`qi` selection) — total 338.

- §7.8.2 Inverting the DC Prediction Process (round 19). New public
  `invert_dc_prediction(bcoded, mbmodes, block_to_macro_block,
  neighbors, plane_raster_order, coeffs) -> Result<(), Error>`
  transcribing the full §7.8.2 procedure of the Xiph Theora I
  Specification. Walks every plane (Y / Cb / Cr) and inside each
  plane every block in raster order: resets the `LASTDC[0..=2]`
  register file to zero at the start of each plane (step 1(a)–(c)),
  recomputes the DC predictor via §7.8.1 for each coded block
  (step 1(d)i.A), adds it to the residual `COEFFS[bi][0]`
  (step 1(d)i.B), truncates to a 16-bit two's-complement
  representation via `i32 -> i16 -> i32` narrowing (step 1(d)i.C),
  writes the reconstructed DC back into `COEFFS[bi][0]`
  (step 1(d)i.D), and seeds `LASTDC[rfi]` for the current macro
  block's reference frame (steps 1(d)i.E–G). AC coefficients are
  not touched. New `DcInversionLenField` discriminant covers the
  three paired-length checks; four new error variants
  (`DcInversionPlaneCount`, `DcInversionLenMismatch`,
  `DcInversionBlockIndexOutOfRange`, `DcInversionDuplicateBlockIndex`)
  reject malformed inputs. Inner §7.8.1 errors propagate unchanged.
  Seventeen new tests (total 320).

- §7.8.1 Computing the DC Predictor (round 18). New public
  `compute_dc_predictor(bi, bcoded, mbmodes, block_to_macro_block,
  neighbors, lastdc, coeffs) -> Result<i32, Error>` transcribing all
  twelve numbered steps of §7.8.1 of the Theora I Specification.
  Forms `DCPRED` for a single block from up to four already-decoded
  neighbour DC values (left, lower-left, lower, lower-right), with
  the per-slot present-and-coded-and-same-rfi gate (steps 3..=10),
  the `LASTDC[rfi]` step-11 fallback, the Table 7.47-indexed weighted
  sum + `//` (truncated-toward-zero) divide (step 12(a)..=(g)), and
  the step 12(h) outranging guard (DC2 → DC0 → DC1 swap order,
  gated on P[0] && P[1] && P[2]). Public helpers:
  `reference_frame_for_mb_mode` exposes the Table 7.46
  `MBMODES → ReferenceFrame` mapping, `dc_predictor_weights` exposes
  the 15 non-zero Table 7.47 rows for round-trip / spot-check use,
  `DcLastDc::zero` / `get` / `set` give §7.8.2 a typed surface for
  the LASTDC register file. New types: `ReferenceFrame`,
  `DcPredictorNeighbors`, `DcLastDc`, `DcPredictorWeights`,
  `DcPredictorLenField`. Four new error variants
  (`DcPredictorBlockIndexOutOfRange`, `DcPredictorBcodedLenMismatch`,
  `DcPredictorMacroBlockIndexOutOfRange`,
  `DcPredictorNeighborIndexOutOfRange`) reject malformed inputs.
  Seventeen new tests (total 303).

- §7.7.3 DCT Coefficient Decode driver (round 17). `decode_dct_coefficients`
  walks the `ti` 0..=63 zig-zag axis, reads `htiL` / `htiC` at `ti ∈ {0, 1}`,
  selects the §6.4.4 Huffman table per Table 7.42 + `bi < NLBS` luma/chroma
  split, walks the tree bit-by-bit, and dispatches to §7.7.1 / §7.7.2 per
  TOKEN. Enforces the closing-paragraph contract (`EOBS = 0`, `TIS[bi] = 64`
  for every coded block) via two new typed rejects. Seven new error variants
  cover input validation, tree corruption, and closing-paragraph violations.
  Fifteen new tests (total 286).

## [0.0.9](https://github.com/OxideAV/oxideav-theora/compare/v0.0.8...v0.0.9) - 2026-05-30

### Other

- wire §7.7.2 coefficient token decode (round 16)
- wire §6.4.1 LFLIMS decode (round 15)

### Added

* **§7.7.2 Coefficient Token Decode (2026-05-30, round 16).** New
  public `decode_coefficient_token(packet: &[u8], token: u8, nbs: u32,
  bi: u32, ti: u8, tis: &mut [u8], ncoeffs: &mut [u8],
  coeffs: &mut [[i16; 64]]) -> Result<CoefficientTokenKind, Error>`
  transcribing the full §7.7.2 procedure of the Xiph Theora I
  Specification ("Coefficient Token Decode"). Consumes one of the 25
  Table 7.38 token values (7..=31), reads the token-specific SIGN /
  MAG / RLEN extra-bits payload (`0..=11` bits depending on the
  token), writes one or more entries to `COEFFS[bi]`, advances
  `TIS[bi]`, and updates `NCOEFFS[bi]` — except for the pure zero-run
  tokens 7 / 8, which advance `TIS[bi]` but leave `NCOEFFS[bi]`
  untouched (per §7.7.2's introductory text: "we do not update the
  coefficient count for the block if we decode a pure zero run").
  Returns a typed `CoefficientTokenKind` discriminating the token's
  structural class — `ZeroRun` (tokens 7 / 8), `Single`
  (tokens 9..=22), or `RunPlusOne` (tokens 23..=31) — so the §7.7.3
  driver can branch on class without re-deriving Table 7.38.

  Five new `Error` variants reject malformed inputs:
  `CoefficientTokenOutOfRange { token }` for `token < 7` or
  `token > 31` (tokens 0..=6 are §7.7.1 EOB tokens, not handled here);
  `CoefficientTokenBlockIndexOutOfRange { bi, nbs }` for `bi >= nbs`;
  `CoefficientTokenIndexOutOfRange { ti }` for `ti > 63`;
  `CoefficientTokenStateLenMismatch { which, got, nbs }` (with a new
  `CoefficientTokenStateSlice` discriminant) when any of `tis` /
  `ncoeffs` / `coeffs` has a length other than `nbs`; and
  `CoefficientTokenWouldOverflowBlock { token, ti, new_tis }` when a
  multi-coefficient token's implied count would push `TIS[bi]` past
  64 — surfacing the §7.7.2 MUST-NOT clause as a fail-closed decode
  error. All five carry `Display` arms citing §7.7.2.

  Twenty-six new unit tests (total now 271) exercise every Table
  7.38 token (7..=31), the SIGN/MAG/RLEN bit-field ordering, the
  pure-zero-run `NCOEFFS[bi]` non-update rule, the legal `ti = 63`
  single-coefficient pin (which produces `TIS = 64`), the legal
  `ti = 56, RLEN = 8` zero-run terminal-position case, the
  `RLEN = 64` from `ti = 0` legal boundary, the overflow rejection
  paths (TOKEN=8 from `ti = 1` with RLEN=64; TOKEN=29 from `ti = 47`
  with RLEN=17), the truncation error carrying the right `§7.7.2`
  field-name attribution, the `Display` rendering for each of the
  five new error variants, the shared-`BitReader` chaining contract
  via `decode_coefficient_token_inner`, and an end-to-end bit-count
  audit confirming each token reads exactly as many bits as Table
  7.38's "Extra Bits" column declares.

  The crate-private `decode_coefficient_token_inner` operates on an
  already-positioned `BitReader` so that the §7.7.3 driver — once it
  lands — can chain §7.6 → §7.7 on the same reader without
  re-aligning at byte boundaries, matching the existing
  `decode_eob_token_inner` chaining contract.

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

* **§6.4.1 Loop Filter Limit Table Decode (2026-05-29, round 15).**
  New public
  `decode_loop_filter_limit_table(bits: &[u8]) -> Result<[u8; 64], Error>`
  transcribing the full §6.4.1 procedure of the Xiph Theora I
  Specification ("Loop Filter Limit Table Decode"). The published
  `Theora.pdf` omits the section's numbered decode steps — the text
  jumps from "It is decoded as follows:" directly to "VP3
  Compatibility" / §6.4.2. The procedure body is recovered from the
  spec's own LaTeX source, staged into the repo at
  `docs/video/theora/theora-6.4.1-lflims.md`.

  The procedure is two steps: (1) read a 3-bit unsigned integer as
  `NBITS`; (2) for each `qi` in 0..=63, read an `NBITS`-bit unsigned
  integer as `LFLIMS[qi]`. Total bits consumed: `3 + 64 * NBITS`.
  `NBITS` is shared across all 64 entries — read once before the
  loop. There is no per-value clamping — the 7-bit output width
  matches the `NBITS ≤ 7` ceiling declared by the variable table.

  The crate-private `decode_lflims_inner` operates on an
  already-positioned `BitReader` so that a future
  `parse_setup_header` can chain §6.4.1 → §6.4.2 → §6.4.4 on a
  single shared reader without re-aligning at byte boundaries —
  matching the existing pattern of `decode_quant_params_inner` and
  `decode_huffman_tables_inner`.

  Seven new unit tests (total now 245) cover the `NBITS = 0` corner
  case (all-zero output, 3 bits consumed), the Appendix B.2 VP3
  table round-trip at `NBITS = 5`, the `NBITS = 7` ceiling
  exercising the full 7-bit output width, every entry pinned at
  `127 = 2^7 − 1`, mid-payload truncation reporting `LFLIMS`, empty-
  payload truncation reporting `LFLIMS NBITS`, and the shared-
  `BitReader` chaining contract via `decode_lflims_inner`. A
  `pack_msb_first` test helper packs `(value, nbits)` slot lists
  MSb-first to mirror the `BitReader` decoding.

  `parse_setup_header` continues to surface
  `Error::SetupHeaderBodyNotImplemented` because the chained
  §6.4.1 → §6.4.2 → §6.4.4 body decode on a shared bit reader is
  still pending; each section is now available as a standalone
  entry point that can be exercised independently.

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
