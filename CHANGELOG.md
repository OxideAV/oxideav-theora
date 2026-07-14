# Changelog

All notable changes to `oxideav-theora` are recorded here.

## [Unreleased]

### Fixed

- **`TheoraIdentHeader::for_picture` emits a container-carriable
  `KFGSHIFT` (round 413)** — the constructor wrote `KFGSHIFT = 0`,
  which makes the §A.2.3 granule mapping's offset-since-keyframe half
  zero bits wide: a stream containing *any* inter frame cannot be
  assigned granule positions at all, so encoder output built on a
  `for_picture` header was unmuxable into Ogg (the first P frame has
  no representable granule). The default is now 6 — offsets up to 63
  cover a keyframe interval of 64 frames at the fewest granule bits —
  and the field stays public for callers planning longer GOPs or
  matching an existing stream. Surfaced by this round's external
  container-carriage validation.

### Added

- **Four-step whole-pixel motion search (round 406)** — the motion
  estimator's exhaustive ±3-pixel grid (49 SAD probes) is replaced by
  a four-step logarithmic descent (`whole_pixel_step_search`): from
  the zero vector, each pass probes the eight neighbours at step 8, 4,
  2, then 1 whole pixels around the running winner, reaching every
  displacement in ±15 pixels with 33 probes. Strict-improvement moves
  keep the zero-vector bias, the §7.5.1 ±31-component clamp is
  enforced, and the existing half-pixel refinement still probes the
  odd neighbours of the winner — both the uniform macro-block search
  (previous and golden references) and the per-luma-block
  `INTER_MV_FOUR` search use it. Measured: a (6, −5)-pixel translation
  — unreachable before — is recovered (motion-vector components ≥ 8
  half-pels on the wire, pinned in-test) and the motion packet drops
  to 16 B where the zero-MV spelling of the same frame needs 125 B at
  equal round-trip fidelity. The keyframe-rate-policy regression now
  pins the whole-stream **Lagrangian cost** (`SSD + λ·bits`) instead
  of raw SSD dominance — the property the policy actually optimizes —
  since the wider search legally re-balances SSD against bytes.

- **Rate-control target declared in the §6.2 `NOMBR` field
  (round 406)** — `TheoraEncoder::with_target_bitrate` (and the
  bounded variant) now writes the target into the identification
  header's nominal-bitrate field, saturating at the 24-bit maximum
  (§6.2 defines `2^24 - 1` as "that value or greater"). The
  already-queued ident header packet and the advertised
  `output_params.extradata` chain are re-serialized so both header
  carriage paths agree; a decoder built from the rewritten extradata
  still decodes the stream (pinned in-test, saturation included).

- **Encoder-output corruption + stress hardening (round 406)** — two
  new regression harnesses. A deterministic corruption storm (xorshift
  PRNG, reproducible) applies 2400 mutations — bit flips, byte
  rewrites, truncations to any length including empty, tail
  extensions — to real self-encoded intra and inter packets and
  decodes every mutant on a clone of a mid-stream `FrameDecoder`:
  the decoder must return `Ok` or a typed `Err`, never panic (all
  2400 pass, both first-frame and referenced-frame paths). A
  randomized multi-frame stress runs pseudo-random smoothly-evolving
  sources through the full `TheoraEncoder` → `TheoraDecoder` loop at
  every chroma format × qi ∈ {10, 40, 63} (I + 3 P each): every
  packet decodes, output geometry is pinned per format, and the
  weakest quantizer holds a fidelity bound — guarding the
  less-travelled format × quantizer corners against panics, drift,
  and geometry mix-ups.

- **Duplicate-frame detection — zero-byte packet emission
  (round 406)** — when the RD plan of a `TheoraEncoder` P-frame codes
  no block at all (every residual dropped by the per-block skip
  decision, e.g. a repeated source frame), the encoder emits a §7.11
  step-2 **zero-byte packet** instead of the ~5-byte header + all-zero
  §7.3 flags spelling. The two spellings reconstruct identically — an
  all-uncoded inter frame is a bit-exact zero-MV previous copy, the
  loop filter only touches coded-block edges, and no inter frame
  refreshes the golden reference — so this is a pure rate win with the
  spec's own dedicated syntax for it. The branch is unreachable on the
  stream's first frame (always an intra keyframe), so the decode-side
  `FirstFrameEmptyPacket` guard never trips. Pinned end-to-end: an
  I, dup, P sequence emits packet sizes (N, 0, M), the duplicate
  decodes bit-exactly equal to the keyframe's output through
  `TheoraDecoder`, and the changed third frame still codes normally.

- **Rate-distortion per-block skip on `INTER_NOMV` macro blocks
  (round 406)** — the inter encode body previously coded every block
  whose quantized residual had any non-zero coefficient; only an
  exactly-zero residual stayed uncoded. The decoder's uncoded path
  (§7.9.4 step 2(e)) reconstructs a block as the bare zero-MV
  colocated previous copy — for an `INTER_NOMV` macro block that is
  exactly the predictor the encoder built — so *skipping* a block with
  a surviving residual is a legal alternative spelling. The per-block
  loop now scores both by the frame's own Lagrangian — skip at
  `SSD(src, pred)` and zero bits, code at
  `SSD(src, clamp(pred + rec)) + λ·(measured §7.7 token bits)` — and
  keeps the cheaper one (ties keep the coded spelling, preserving
  prior loop-filter behaviour), in every inter path and at every
  `NQIS`. `block_rd_cost` applies the same min inside the RD mode
  decision so an `INTER_NOMV` candidate is priced on the plan the
  encode body will actually emit. Measured on a P-frame differing from
  its reference only by ±3 noise (residuals that *survive* the weakest
  quantizer — the premise is pinned in-test): every block skips, the
  packet drops from 140 B (the hand-built no-skip spelling of the same
  frame) to 5 B while the frame reconstructs as a bit-exact previous
  copy; a strongly changed block in a sea of noise still codes and
  tracks the source (both directions regression-pinned).

- **§2.2 picture-region (non-macro-block-aligned) encoding
  (round 406)** — the encoder now *produces* streams whose visible
  picture is not a multiple of sixteen, the encode-side counterpart of
  the long-validated decode + display-crop path.
  `TheoraIdentHeader::for_picture(picw, pich, pf, frn, frd)` builds a
  §6.2 header around arbitrary picture dimensions (smallest containing
  coded frame, region placed left- and top-aligned — `PICX = 0`,
  `PICY = 16·FMBH − PICH` in the spec's lower-left coordinates,
  matching the placement observed on the reference-captured fixtures),
  and `TheoraIdentHeader::picture_plane_dims` exposes the §4.4.4
  chroma round-up of the region per pixel format.
  `SourceFrame::from_picture` pads picture-shaped planes to the coded
  frame by **edge replication** (§2.2 leaves the outside-region samples
  to the encoder; replication keeps the padding blocks' residual energy
  near zero instead of cutting a hard synthetic edge into every border
  block). `TheoraEncoder::send_frame` accepts `VideoFrame`s at either
  the coded or the picture shape — the latter being exactly what
  `TheoraDecoder` emits — and `output_params` now advertises the
  picture dimensions (identical to the coded ones whenever the region
  covers the whole frame, so MB-aligned callers see no change).
  Validated end-to-end across all three chroma formats: a 26×18
  picture (coded 32×32, the decode fixture's geometry) intra- and
  inter(RD)-encodes, decodes through this crate's own `FrameDecoder`,
  and crops back to picture planes faithful to the source (max error
  ≤ 8 intra / ≤ 32 inter at `qi = 63`), including the §4.4.4 odd-edge
  chroma windows (13×9 at 4:2:0); the framework-trait I+P loop
  round-trips picture-shaped frames both ways. New error
  `EncodePictureDimensionOutOfRange` covers axes outside
  `1..=1048560`.

- **§7.5.1 half-pixel motion-vector refinement (round 398)** — the
  motion search now refines its whole-pixel winner to half-pixel
  accuracy. `search_macro_block_mv_ref` and `search_luma_block_mv`
  each end with `refine_half_pixel_mv`, which evaluates the eight
  `HALF_PIXEL_NEIGHBORS` one half-pixel away and keeps any strict SAD
  improvement (a tie holds the integer winner, so no motion-vector
  magnitude is spent without a fidelity gain; candidates escaping the
  §7.5.1 `-31..=31` component range are dropped). The integer search
  only ever emitted even components, so the decoder's §7.9.1.3 two-tap
  half-pixel predictor was previously unreachable from the encoder;
  odd (half-pixel) components are now produced and flow into the same
  `encode_inter_frame_rd` candidate set (uniform `INTER_MV` /
  `INTER_GOLDEN_MV` and each per-block `INTER_MV_FOUR` vector), where
  the true `D + λ·R` decision prices their motion-vector bits. Measured
  on a linear luma ramp sampled half a pixel off the reference grid:
  whole-plane best integer-grid luma SAD 1184 vs half-pixel-refined
  704 (−40.5 %); the refined encode still round-trips through
  `FrameDecoder` within the quantizer bound. No bitstream-syntax change
  — the same §7.5 writer serializes the (now possibly odd) components.

- **Measured-rate keyframe policy — golden-frame refresh on reference
  decay (round 387)** — `TheoraEncoder::with_keyframe_rate_policy(r)`
  watches two decay signals on every RD P-frame: coded size exceeding
  `r ×` the last keyframe's, or the mode decision coding a majority of
  transmitted macro blocks INTRA (a de-facto intra frame paying §7.3 /
  §7.4 inter syntax while refreshing nothing — this also catches new
  content inherently cheaper than the old keyframe, invisible to any
  size ratio). On either signal the frame is also encoded intra, both
  candidates' delivered SSD is measured by decoding each in a
  throwaway clone of the mirror decoder (`FrameDecoder` is now
  `Clone`; loop filter included), and the lower-Lagrangian-cost
  spelling is emitted — ties to intra, which re-seeds both references
  (only an intra frame ever refreshes the golden frame). Measured on a
  6-frame content-family-switch sequence at interval 30: 1 keyframe /
  651 B / SSD 47553 without the policy vs 3 keyframes / 646 B / the
  identical SSD 47553 with it — fewer bytes, fresh references, equal
  reconstruction error.

- **Adaptive quantization on framework-encoder P-frames (round 387)**
  — `TheoraEncoder::with_adaptive_quant(qis)` now drives every
  rate-distortion P-frame through
  `FrameEncoder::encode_inter_frame_rd_adaptive` in addition to the
  keyframes (non-RD inter strategies keep the single frame-level
  `qi`), so an I,P,I stream carries the §7.1 multi-`qi` header and the
  §7.6 per-block selector stream on all three frames; the trait-path
  test re-parses each frame's wire `QIS` list and round-trips the
  stream through `TheoraDecoder`.

- **Adaptive block-level quantization on the inter encoder (round
  387)** — `FrameEncoder::encode_inter_frame_rd_adaptive` is the first
  P-frame path to emit the §7.1 `MOREQIS` / `QIS` chain and the §7.6
  block-level qi stream: `qis[0]` drives every DC quantizer and the RD
  mode decision, while each coded block's AC quantizer is chosen from
  the list by a per-block `D + λ·R` decision (delivered SSD against
  the block's own mode-selected predictor — motion-compensated
  reference, or flat-128 for intra-coded blocks with their qti = 0
  matrices — plus the measured §7.7 token rate). The §7.6 promotion
  writer is factored into `encode_block_level_qi_stream`, shared with
  the intra path, and the λ ramp is unified as `inter_rd_lambda(qi0)`
  across all three RD choosers. `NQIS = 1` is byte-identical to the
  single-`qi` RD packet. Measured on a mixed flat/noise P-frame at
  equal reference state: all-qi0 178 B / SSD 157576, all-qi63 586 B /
  SSD 2336, adaptive `[0, 63]` 565 B / SSD 9767 — strictly inside the
  single-`qi` curve on both corners, with the wire `QIS` / `QIIS`
  re-parsed exactly by the production §7.11 step-1 driver.

- **INTRA as the eighth RD mode in P-frames (round 387)** — the joint
  rate-distortion mode decision now evaluates `INTRA` alongside the
  seven inter modes: `mb_intra_mode_cost` scores each macro block on
  the delivered fidelity of coding it from scratch (the §7.9.1.1
  flat-128 predictor and the qti = 0 intra matrices — §7.9.4 step
  2(d)ii resolves qti from the block's coding *mode*, not the frame
  type) plus its measured token rate, and the planning loop emits
  intra-coded blocks inside the inter frame (always coded, no MV bits,
  excluded from the §7.5.2 LAST history per spec). New-content macro
  blocks no longer pay for a reference that cannot predict them:
  on a checkerboard-quadrant → gradient scene the RD path codes the
  quadrant INTRA and measures 26 B / luma SSD 224 against the
  intra-less motion path's 149 B / SSD 4052 — a strict win on both
  axes, round-tripped through this crate's own decoder (exercising the
  per-mode qti reconstruction and mixed-reference-frame §7.8 DC
  prediction from a real bitstream).

- **Measured §7.7 token rate in the encoder RD decisions (round
  387)** — `TokenBitCosts` precomputes, per Huffman group and token,
  the minimum §6.4.4 code length over the stream's own 16 selector
  tables (tuned tables included), and `block_bits` prices a candidate
  block by replaying its exact token plan (combined run+value folding,
  `TIS` group routing, terminal EOB, exact extra-bits payloads). The
  inter mode decision (`block_rd_cost` behind every `D + λ·R`
  candidate) and the intra adaptive-quant chooser both replace their
  flat bits-per-coefficient proxies (`6·nz + 3` / `5 + extras` per
  token) with this measured rate; unencodable coefficients now make a
  candidate effectively unaffordable instead of erroring later. Probe
  measurements on the mixed flat/noise frame: the weak-quant candidate
  really costs 519 bits where the old proxy guessed ~200, so the
  adaptive split test moves to a spread that is decisive under
  truthful rate (`qis = [0, 63]`).

- **Frame-optimal §7.4 mode-coding scheme selection (round 387)** —
  the inter mode writer no longer hard-codes `MSCHEME = 7` (3 bits per
  mode): `choose_mode_scheme` tallies the modes a frame actually
  transmits and picks the cheapest of the direct scheme, the six fixed
  Table 7.19 alphabets (the dominant mode costing as little as 1 bit),
  and the custom frequency-sorted scheme-0 alphabet (24-bit header),
  with ties keeping the earlier candidate. `write_table_7_19_mi` is
  the exact inverse of the decoder's unary-with-cap code reader. A
  golden-`MV`-dominated 31-macro-block mode section measures 8 B under
  the chosen scheme 0 versus 12 B direct; the choice only re-spells
  the identical mode array (round-tripped through the production §7.4
  decoder for scheme 7, a fixed alphabet, and scheme 0).

- **Mixed I/P GOP two-pass Huffman tuning (round 387)** —
  `SetupHeaderTables::with_gop_tuned_huffman_tables` tunes **four**
  §6.4.4 codebooks per Huffman group (intra/inter × luma/chroma), each
  replacing the selector slot scoring worst on its own statistics
  column (all-zero columns skipped), so keyframes and P-frames each
  get tables specialized to their very different token mixes and the
  §7.7.3 per-frame selector optimization routes every frame to the
  cheapest.
  `TheoraEncoder::with_gop_tuned_setup_keyframe_interval` wraps the
  full two-pass flow: the first pass encodes the sample sequence as
  the exact I/P GOP the second pass will emit (RD-planned inter frames
  against a mirror decoder's reconstructed references), tallying intra
  and inter tokens separately. Measured on a 6-frame textured motion
  sequence at interval 3: 2248 B (VP3 defaults) → 2141 B (intra-only
  tuning) → 2072 B GOP-tuned (−7.8% / −3.2%), with all three streams
  reconstructing bit-identically.

- **Inter-frame token statistics (round 387)** —
  `FrameEncoder::inter_token_statistics` tallies the §7.7 tokens an
  inter (P-frame) encode would write — combined run+value tokens and
  coalesced cross-block EOB runs included — without producing a
  packet: the inter counterpart to `intra_token_statistics` and the
  previously missing half of the two-pass Huffman-tuning flow for
  mixed I/P streams. The mode decision, motion search, and forward
  quantization run exactly as `encode_inter_frame_rd` runs them; the
  shared inter encode body is refactored into a planning stage
  (`plan_inter_frame` → `InterFramePlan`: modes, per-block MVs, coded
  flags, quantized coefficients) and a serialization stage
  (`write_inter_packet`), so the tally and the packet writer read the
  identical per-block state. Tests pin that a pure-copy P-frame
  tallies zero tokens, that the tally is deterministic, and that it
  equals the token plan of the packet actually written.

## [0.0.11](https://github.com/OxideAV/oxideav-theora/compare/v0.0.10...v0.0.11) - 2026-07-03

### Other

- README — round 384 entropy-writer depth (full §7.7 alphabet, EOB runs, tuned codebooks, adaptive quant)
- wire adaptive quantization into TheoraEncoder keyframes
- adaptive block-level quantization on the intra encoder (§7.1 MOREQIS + §7.6)
- two-pass Huffman tuning through the framework Encoder
- content-tuned §6.4.4 Huffman codebooks (two-pass encoding)
- coalesce cross-block §7.7.1 EOB runs (tokens 1..=6)
- fold zero runs into §7.7.2 combined run+value tokens 23..=31, use token 7 for short runs
- optimize the ti=0 (DC) and ti=1 (AC) Huffman selector pairs independently
- choose the §7.7.3 Huffman codebook to minimize frame size
- make_encoder synthesizes setup when extradata omits it
- synthesize VP3-default setup header from Appendix B data
- scene-cut keyframe insertion
- refresh stale encoder doc comments
- keyframe-aware rate-control budgeting
- fold §7.5.2 LAST-mode rate discount into the RD mode decision
- expose GoldenMotion + FourMv P-frame InterModeStrategy variants
- reject a zero-byte first packet instead of decoding from zeros
- add a target-bitrate rate-control loop to TheoraEncoder
- fold INTER_MV_FOUR into the unified rate-distortion candidate set
- measured RD vs previous-only delta + README/CHANGELOG rollup
- TheoraEncoder P-frames default to the rate-distortion decision
- unified rate-distortion inter mode decision (D + λ·R)
- round-trip four-MV chroma averaging across 4:2:0/4:2:2/4:4:4
- README — golden + four-MV encode modes now emitted and round-tripped (round 364)
- four-MV inter encoder — INTER_MV_FOUR emit + per-block decode round-trip
- golden-reference inter encoder — INTER_GOLDEN_MV/NOMV emit + decode round-trip
- per-plane/per-frame HD digest localisation
- HD frame-1 §7.4 per-position macro-block trace pin (all 8160 MBs)
- unit-cover split_motion_vector_per_axis per-axis divisors
- §7.5.1 quarter-pixel chroma MC — HD 1080p fixture now pixel-SHA exact
- document HD 1920x1088 fixture coverage + open pixel-fidelity item in README
- HD 1920x1088 two-frame decode integration test (dimensions-1080p-very-short)
- inter encoder picks INTER_MV_LAST / INTER_MV_LAST2 predicted modes
- TheoraEncoder emits P-frames via keyframe interval + mirror decoder
- inter (P-frame) encoder — §7.2/§7.3/§7.4/§7.5 encode + motion estimation
- encoder chroma-format (420/422/444) + multi-frame coverage
- oxideav_core::Encoder trait integration (TheoraEncoder + make_encoder)
- §6 header-packet serialization (encode_{identification,comment,setup}_header)
- scrub encoder-implementation naming from forward-DCT doc (round 338)
- README — document the intra encoder (round 338)
- intra encoder — encode→decode self-roundtrip (round 338 milestone)
- forward DC prediction — inverse of §7.8.2 (round 338)
- forward quantization — inverse of §7.9.2 (round 338)
- §7.9.3.3 forward DCT (round 338, first encoder commit)
- neutralize decorative reference-encoder naming in r334 fixture-gap prose
- exercise golden + four-MV inter modes through the full §7.9.4 frame driver
- pin keyframe-interval-1 fixture end-to-end — all-keyframe -g 1 run
- pin monochrome-via-zero-chroma fixture end-to-end (round 325)
- pin picture-region-non-mb-aligned fixture end-to-end (§2.2/§4.4.4 crop)
- Wire oxideav_core::Decoder trait integration (round 317)
- pin bitstream-version-3.2.1 fixture end-to-end (round 313)
- refresh to current status, drop per-round changelog cruft

### Added

- **Adaptive quantization on the framework encoder (round 384)** —
  `TheoraEncoder::with_adaptive_quant(qis)` drives every keyframe
  through `FrameEncoder::encode_intra_frame_adaptive` with the supplied
  §7.1 `QIS` list (1..=3 entries), each block's AC quantizer picked by
  the per-block `D + λ·R` decision; inter (P) frames keep the single
  frame-level `qi` (and target-bitrate rate control keeps steering it
  when enabled). A new trait-path test round-trips an I,P,I stream:
  the keyframes carry the two-entry `QIS` list on the wire (re-parsed
  with `FrameDecoder`), the P frame stays single-`qi`, and the whole
  stream decodes through `TheoraDecoder` — the encoder's internal
  mirror decoder tracks the multi-`qi` keyframes in lock-step.

- **Adaptive block-level quantization on the intra encoder (round
  384)** — the first encoder path to emit a §7.1 multi-`qi` frame
  header (the `MOREQIS` / `QIS` chain, up to `NQIS = 3`) and the §7.6
  block-level qi stream (`NQIS − 1` §7.2.1 long-run promotion passes
  over the coded blocks, the exact inverse of
  `decode_block_level_qi`). `FrameEncoder::encode_intra_frame_with_qiis`
  takes an explicit frame `qis` list plus a per-block selector array:
  `qis[0]` drives every DC quantizer (the §7.6 preamble's rule, so DC
  prediction is untouched) and the loop-filter limit, while each
  block's AC quantizer is `qis[qiis[bi]]`.
  `FrameEncoder::encode_intra_frame_adaptive` layers a per-block
  rate-distortion chooser on top: each block is quantized once per
  candidate, reconstructed exactly as the decoder will (dequantize →
  inverse DCT → clamp) and scored `D + λ·R` (delivered SSD + a
  token-plan bit estimate under the same quadratic λ ramp the inter RD
  decision uses). New error variants `EncodeQisCountOutOfRange` /
  `EncodeQiisLenMismatch` / `EncodeQiiOutOfRange` fail malformed
  shapes closed. Tests re-parse the emitted packet through the
  production §7.11 step-1 driver and pin the wire `QIS` / `QIIS`
  arrays exactly (two-qi split, full three-qi chain, RD-chosen mixed
  split), verify the weak-quantizer region reconstructs strictly
  better than a uniform strong-quantizer encode, and pin the
  `NQIS = 1` degenerate case byte-identical to the historical
  single-qi packet.

- **Two-pass Huffman tuning on the framework encoder (round 384)** —
  `TheoraEncoder::with_tuned_setup_keyframe_interval` wraps the
  two-pass flow in one constructor: it analyzes caller-supplied sample
  frames (`FrameEncoder::intra_token_statistics`), folds the
  accumulated `TokenStatistics` into tuned §6.4.4 codebooks
  (`SetupHeaderTables::with_tuned_huffman_tables`), and builds the
  encoder on them — so the §6.4 setup header it queues (and the
  `extradata` chain it advertises) carries the tuned tables and the
  stream stays fully self-describing. A new end-to-end trait test
  encodes the same I,P,P sequence through a default-table and a tuned
  encoder: the tuned stream's data payload is strictly smaller and both
  streams decode — from nothing but their own packets — to
  bit-identical pixels.

- **Content-tuned §6.4.4 Huffman codebooks — two-pass encoding (round
  384)** — the encoder can now build its own optimal DCT-token
  codebooks from measured statistics instead of relying solely on the
  Appendix B defaults. `HuffmanTable::from_token_counts` constructs a
  minimum-redundancy codebook from per-token emission counts (all 32
  tokens leafed so later frames stay encodable; canonical code
  assignment satisfying the Kraft equality, i.e. a full §6.4.4 tree;
  deterministic tie-breaking; lengths capped at 16 bits by iterative
  weight dampening). `FrameEncoder::intra_token_statistics` is the
  first pass: it tallies the tokens an intra encode *would* write —
  combined run+value tokens and coalesced EOB runs included — into a
  new `TokenStatistics` accumulator without producing a packet.
  `SetupHeaderTables::with_tuned_huffman_tables` then replaces, per
  Huffman group, the selector slot scoring **worst** on the sampled
  luma statistics with the luma-tuned codebook and the worst remaining
  slot with the chroma-tuned one — sacrificing the least-useful
  fallbacks, so content resembling the sample keeps the default table
  it would actually have picked — and the per-frame selector
  optimization chooses the tuned tables only where they win. The tuned
  set serializes through `encode_setup_header` like any other, so the
  stream stays fully self-describing. Measured on the 64×64 intra
  harness: the textured frame shrinks a further 19–23% versus the
  VP3-default tables (e.g. 1898 B → 1467 B at qi = 63; 521 B → 420 B at
  qi = 16) at bit-identical reconstruction; the sparse frame — unlike
  the tuning-dominant content — stays within ±1 B of its default-table
  size.

### Changed

- **Cross-block §7.7.1 EOB runs (tokens 1..=6) in the frame token
  writer (round 384)** — the writer previously closed every coded block
  with its own single-block EOB token 0, so a stretch of blocks ending
  together (all-zero chroma planes, flat regions, sparse frames) paid
  one Huffman code per block. The frame writer now simulates the
  §7.7.3 decoder's visit order once (the outer `ti` / inner coded-`bi`
  interleave is fully determined by the per-block token plans), then
  coalesces every maximal run of consecutive block-end visits into a
  single §7.7.1 EOB-run token read at the run's first visit — the
  remaining ends consume no bits at all (the decoder's step 4(b)ii).
  Runs freely cross the `ti = 1` selector refresh and the luma / chroma
  boundary, and runs longer than token 6's 12-bit payload (4095) split
  greedily. The Huffman selector optimization now scores the tokens
  actually written (EOB runs included) rather than the per-block plans.
  Measured on the 64×64 intra harness: 5099 B → 4871 B overall (−4.5%),
  with the sparse frames shrinking 25–40%. Round-384 cumulative
  (selector split + combined tokens + EOB runs): 8006 B → 4871 B,
  −39.2% at identical reconstruction.
- **Combined run+value tokens 23..=31 and short zero-run token 7 in
  the token planner (round 384)** — the per-block token planner
  previously spelled every zero gap as a 6-bit zero-run token 8 and
  every non-zero coefficient as its own single-coefficient token. It
  now folds a gap and the coefficient that ends it into one §7.7.2
  combined run+value token whenever one covers the pair (tokens 23..=29
  for a trailing `±1` after 1..=17 zeros; tokens 30..=31 for a trailing
  `±2`/`±3` after a run of 1 / 2..=3) — one Huffman code where two were
  written — and uses the 3-bit zero-run token 7 for uncombinable gaps
  of 1..=8 (halving the run payload of token 8). Purely an encoder
  token-choice change: the decoder already accepted the full §7.7
  alphabet, and every new token round-trips bit-exactly through
  `decode_dct_coefficients`. Measured on 64×64 textured + sparse intra
  frames at qi ∈ {16, 32, 48, 63}: total coded size fell from 7924 B to
  5099 B (−35.6%) at identical reconstruction (the coefficient stream
  is unchanged, only its spelling).
- **Independent DC / AC §7.7.3 Huffman selector optimization (round
  384)** — the §7.7.3 decoder reads a *fresh* `htiL` / `htiC` pair at
  `ti = 1`, so the pair written at `ti = 0` only ever addresses Huffman
  group 0 (the DC group) and the `ti = 1` pair only groups 1..=4 (the
  AC groups). The frame token writer previously emitted one jointly
  optimized pair twice; it now optimizes the two pairs independently
  per plane (`best_huffman_selector` gains a group-range argument) and
  emits the DC-optimized pair at `ti = 0` and the AC-optimized pair at
  `ti = 1`. The split search space contains every joint choice, so this
  is never worse than the round-379 joint selector — a pure bit-rate
  win at zero distortion, with no new bitstream syntax (the same two
  §7.7.3 selector fields, now decoupled). A new round-trip test drives
  a frame whose DC tokens prefer a different table than its AC tokens
  and pins the coded size to the split-selector cost model exactly.
- **§7.7.3 Huffman codebook selection (round 379)** — the frame token
  writer no longer hard-codes the `0` table within each Huffman group.
  It now tallies, per Huffman group and token value, the luma and chroma
  tokens a frame emits, then picks the `htiL` / `htiC` selector (one of
  16 per group) that minimizes the frame's summed Huffman-code length
  (`best_huffman_selector`). This is a pure bit-rate reduction at zero
  distortion: the token *plan* is unchanged, only the codebook indices
  written into the frame header differ, and the decoder reconstructs
  identically by reading whichever selector was chosen. The optimizer is
  never worse than the previous fixed-`0` behaviour. No new
  bitstream-syntax — the same §7.7.3 selector field, now used.
- **Rate control is keyframe-aware (round 375)** — the target-bitrate
  leaky-bucket loop no longer charges a large keyframe against a single
  per-frame budget (which spiked the bucket and slammed the quantizer for
  the P-frames after every GOP boundary). A keyframe now drains a weighted
  budget (`keyframe_weight × bits_per_frame`) and the surplus is repaid in
  equal shares across the GOP's inter frames, so the long-run average
  target is preserved exactly while a keyframe's bulk no longer perturbs
  the frames that follow it. `RateControl` learns the GOP length from the
  encoder's keyframe interval; an all-keyframe stream (`interval = 1`)
  leaves the bonus path inert (flat per-frame budget). No bitstream-syntax
  change.
- **RD mode decision folds the §7.5.2 LAST-mode rate discount (round
  375)** — the unified `D + λ·R` inter mode decision now tracks the
  running `LAST1` / `LAST2` motion-vector predictor and charges an
  `INTER_MV` candidate zero explicit-MV bits when its vector matches the
  predictor (it will recode to a no-MV-bit `INTER_MV_LAST` /
  `INTER_MV_LAST2`), instead of a flat 12-bit charge for every explicit
  vector. A vector the predictor already supplies for free is no longer
  over-penalised, so on global / repeated motion the RD decision codes
  `INTER_MV` widely and the recode pass collapses the repeats to LAST
  modes. No bitstream-syntax change; the chosen modes emit identical §7
  bytes.

### Added

- **Synthesized VP3-default setup header (round 379)** —
  `SetupHeaderTables::vp3_defaults()` builds the complete §6.4.5 setup
  bundle — the §B.2 loop-filter limits, the §B.3 AC/DC scale tables, the
  §B.4 base matrices with their single-range quant assignment, and the
  §B.4 80 DCT-token Huffman codebooks — entirely from the published
  Appendix B data, with no setup packet to parse. A new
  `HuffmanTable::from_code_list` constructor materializes each codebook
  from its `(code, length)` pairs into the exact tree layout the §6.4.4
  decode would have built (byte-identical, so it re-serializes and
  decodes back to itself). `TheoraEncoder::with_default_setup` (and the
  `…_keyframe_interval` variant) are zero-setup constructors: given only
  the identification header and a quantizer, the encoder synthesizes its
  own conformant, self-describing setup header and emits a stream that
  decodes through this crate's own `TheoraDecoder` (a flat frame
  lossless; a textured I,P,P sequence within the quantizer bound). This
  removes the encoder's last simplification — it no longer requires the
  caller to supply pre-decoded setup tables. No bitstream-syntax change.
  The registry `make_encoder` factory follows suit: the §6.4 setup header
  in `extradata` is now **optional** — given only the §6.2
  identification header (which still carries the un-defaultable
  dimensions / pixel format), the factory synthesizes the VP3 default
  setup and emits a complete self-describing stream; a setup header in
  `extradata`, when present, is still honoured verbatim.
- **Scene-cut keyframe insertion (round 375)** —
  `TheoraEncoder::with_scene_cut_threshold` enables an opt-in detector:
  before coding a non-boundary frame as inter, the encoder measures the
  mean absolute luma difference between the incoming source and the
  previous reconstructed reference, and when it exceeds the threshold
  codes a fresh keyframe mid-GOP instead (resetting the keyframe-interval
  counter). A scene change makes inter prediction worthless, so an intra
  frame is both smaller and resets the references cleanly. Disabled by
  default (keyframes stay interval-driven); the decision only changes
  which frames are intra vs inter, both emitting conformant §7 streams.
- **Golden-aware and four-MV P-frame strategies on the high-level encoder
  (round 375)** — `InterModeStrategy` gains `GoldenMotion` and `FourMv`
  variants alongside `RateDistortion` (default) and `PreviousMotion`.
  `GoldenMotion` searches both the previous and golden references per
  macro block and codes the smaller-SAD reference (`INTER_GOLDEN_*` on a
  golden win); `FourMv` searches an independent previous-reference vector
  per luma block and emits `INTER_MV_FOUR` (with §7.5.2 chroma averaging)
  when the four vectors disagree. Both were already reachable on
  `FrameEncoder` but are now selectable end-to-end through
  `TheoraEncoder::with_inter_mode` and the framework `Encoder` trait; the
  inter-mode round-trip test exercises all four strategies.

### Fixed

- **Reject a zero-byte first packet (round 371)** — a §7.11 step-2
  duplicate-frame marker (empty packet) as the very first packet had no
  reference frame to duplicate, yet the decoder previously synthesized an
  all-uncoded inter frame and reconstructed it from the zero-initialized
  reference store, silently emitting a garbage frame. `FrameDecoder`
  now returns the new `Error::FirstFrameEmptyPacket` (surfaced as an
  `InvalidData` error through the framework `Decoder` trait) before any
  reconstruction, and the rejection leaves decode state untouched so a
  subsequent real keyframe still decodes. Tests cover both the direct and
  trait paths.

### Added

- **Target-bitrate rate-control loop (round 371)** —
  `TheoraEncoder::with_target_bitrate` (and `with_target_bitrate_bounded`
  for explicit `qi` bounds) enables a leaky-bucket rate-control loop that
  adapts the frame-level quantization index before each frame to steer
  the running output size toward a byte budget. Each frame is allotted
  `target_bits / fps` bits (frame rate from the §6.2 `FRN / FRD` header);
  the coded size feeds a signed fullness accumulator and a clamped
  proportional step moves `qi` to oppose the imbalance — over budget
  lowers `qi` (stronger quantization, smaller frames), under budget
  raises it. New `FrameEncoder::qi` / `set_qi` accessors let the loop
  re-quantize between frames without rebuilding geometry. The adaptation
  only changes which valid `qi` each frame is coded at, so it has no
  bitstream-syntax effect (the decoder reads `QIS[0]` exactly as for a
  fixed-`qi` stream). Tests pin the feedback direction + clamps, an
  end-to-end strict-vs-generous target size delta, and the disabled-loop
  no-op.

- **`INTER_MV_FOUR` folded into the rate-distortion candidate set
  (round 371)** — `encode_inter_frame_rd` now evaluates a per-luma-block
  four-MV candidate alongside the uniform inter modes. A new
  `FrameEncoder::mb_four_mv_cost` searches an independent previous-
  reference vector per luma block, scores each luma block with its own
  vector and each chroma block with the §7.5.2 averaged vector through
  the same forward-quantize → reconstruct → SSD path the uniform modes
  use, and folds in the §7.4 mode code plus 4×12 explicit-MV bits. The
  macro block codes `INTER_MV_FOUR` only when its delivered `D + λ·R`
  beats every uniform mode, so the four explicit vectors are paid for
  out of the rate term (an all-equal four-MV is strictly dominated by
  uniform `INTER_MV` and never wins). A new test drives the
  `TheoraEncoder` default RD path to select `INTER_MV_FOUR` on a frame
  whose four luma blocks each moved by a distinct whole-pixel vector,
  and round-trips the per-block-displaced source within the quantizer
  bound — so every §7.5.2 inter mode is now reachable from the single
  unified RD entry point, not just the standalone four-MV search.

- **Measured RD vs previous-only delta (round 368)** — a new test
  quantifies the rate-distortion decision's value on a scene the golden
  reference predicts perfectly: the RD path codes a distortion-free
  golden copy (SSD 0 in 25 B) where the previous-reference-only motion
  path must code a large residual against the unrelated previous
  reference (SSD 50264 in 252 B) — a strict win on both delivered
  distortion and packet size, asserted in-test.

- **`TheoraEncoder` P-frames default to the rate-distortion decision
  (round 368)** — the framework `Encoder` integration now drives its
  inter (P) frames through `encode_inter_frame_rd` by default, so a full
  encode → decode loop through the `oxideav_core` traits benefits from
  the joint `D + λ·R` mode decision. A new `InterModeStrategy` enum plus
  `TheoraEncoder::with_inter_mode` builder selects between the
  rate-distortion default and the historical previous-reference SAD
  motion path (`PreviousMotion`). A new I,P,P round-trip test exercises
  both strategies end-to-end through the `Encoder` + `Decoder` traits.

- **Unified rate-distortion inter mode decision (round 368)** — a new
  `FrameEncoder::encode_inter_frame_rd` entry point replaces the
  fixed-per-strategy SAD heuristic with a single joint per-macro-block
  Lagrangian choice. Each macro block evaluates every reachable uniform
  inter mode (`INTER_NOMV`, `INTER_MV`, `INTER_GOLDEN_NOMV`,
  `INTER_GOLDEN_MV`) by its true rate-distortion cost `D + λ·R` and codes
  the cheapest. The distortion `D` is the sum of squared errors between
  the source and the block the decoder will actually reconstruct — the
  encoder runs its own forward quantize → dequantize → inverse-DCT →
  clamp (`block_rd_cost`) to measure *delivered* fidelity rather than
  scoring on a pre-quantization SAD proxy; the rate `R` folds in the §7.4
  mode code and the §7.5 `MVMODE = 1` explicit-MV bits, weighted by a
  quantizer-derived Lagrange multiplier λ (`inter_rd_lambda`, monotone in
  `qi`). This subsumes the raw-SAD previous-vs-golden choice
  `encode_inter_frame_golden` makes into one decision on reconstructed
  distortion; `INTER_MV` winners still flow through the §7.5.2
  `INTER_MV_LAST` / `INTER_MV_LAST2` recode. New tests pin: a changed
  frame round-trips within the quantizer bound, an identical frame
  collapses to an all-`INTER_NOMV` pure copy (the λ·R rate term picking
  the no-MV mode over equally-distortion golden/MV candidates), and a
  frame the golden reference predicts perfectly is coded golden
  everywhere. The mode decision is an encoder engineering choice with no
  bitstream-syntax effect; every chosen mode emits the same §7 bytes the
  existing writers produce.

- **Four-MV chroma averaging round-tripped across 4:2:0 / 4:2:2 / 4:4:4
  (round 364)** — a new round-trip test drives `INTER_MV_FOUR` through
  `encode_inter_frame_four_mv` for all three pixel formats, validating
  the §7.5.2 step 3(a)x..xii chroma-MV averaging end-to-end for the
  4:2:2 (bottom/top half-average, two chroma blocks) and 4:4:4
  (per-block copy, four chroma blocks) layouts. The fixture corpus is
  4:2:0-only, so these chroma-averaging paths had never been exercised
  through a real bitstream before — only by synthetic
  `decode_macroblock_motion_vectors` unit tests. The encoder's
  `four_mv_chroma_average` and the decoder's chroma-MV assignment now
  agree for every layout under a full encode→decode reconstruction.

- **Four-MV inter encoder (round 364)** —
  `FrameEncoder::encode_inter_frame_four_mv` searches the previous
  reference for an independent vector per luma block and codes
  `INTER_MV_FOUR` when the four winning vectors disagree (collapsing to
  the cheaper uniform `INTER_MV` / `INTER_NOMV` when they agree). Each
  luma block carries its own vector; each chroma block carries the
  §7.5.2 step 3(a)x..xii average of the four luma vectors for the pixel
  format, computed by the new `four_mv_chroma_average` helper (the
  exact inverse of the decoder's chroma-MV assignment). The shared
  encode body now builds a per-block motion-vector table up front (luma
  per-block, chroma averaged) and the §7.5 LAST recode pass tracks the
  four-MV step 3(a)xiii/xiv predictor update (LAST1 = the last coded
  luma block's vector) so a following `INTER_MV` stays in sync with the
  decoder. This is the first encoder path that emits per-luma-block
  motion vectors, so the decoder's §7.5.2 four-MV decode + chroma
  averaging + §7.9.4 per-block reconstruction is now exercised
  top-to-bottom from a real self-encoded bitstream (a new round-trip
  test displaces each 8×8 luma block of the previous reference by a
  distinct whole-pixel vector, asserts `INTER_MV_FOUR` lands on the
  wire, and checks the reconstruction within the quantizer bound).

- **Golden-reference inter encoder (round 364)** —
  `FrameEncoder::encode_inter_frame_golden` is a new inter-frame entry
  point that runs the §7.5 motion search against **both** the previous
  and golden references per macro block and codes whichever predicts
  its four luma blocks with the smaller SAD: `INTER_GOLDEN_NOMV` /
  `INTER_GOLDEN_MV` for a golden win (reconstructing the residual
  against the golden motion-compensated predictor), `INTER_NOMV` /
  `INTER_MV` otherwise. The golden modes are correctly excluded from
  the §7.5.2 LAST1 / LAST2 predictor history. This is the first encoder
  path that emits the golden-reference inter modes, so the decoder's
  §7.9.4 Table 7.75 golden reconstruction is now exercised
  **top-to-bottom from a real (self-encoded) bitstream** — two new
  encode→decode round-trip tests assert the golden modes appear on the
  wire and that a golden copy reproduces the golden reference
  bit-exactly (NOMV) / within the quantizer bound (MV). Previously the
  golden reconstruction path was only validated by synthetic
  `reconstruct_frame` unit tests with hand-built mode arrays, never
  from a packet the production encoder assembled. The shared inter
  encode body now selects its per-block reference frame from the macro
  block's mode (`reference_frame_for_mb_mode`) rather than hard-coding
  the previous reference.

- **Per-plane / per-frame HD digest localisation (round 360)** — the HD
  `dimensions-1080p-very-short` decode is now pinned by six SHA-256
  sub-digests (frame 0 / frame 1 × Y / Cb / Cr) in addition to the
  single authoritative `c48344b1…` combined digest. The combined pin
  proves the bytes correct; the sub-digests localise any future
  regression to a specific frame and plane (intra vs inter, luma vs
  chroma) instead of only reporting "the digest changed". The
  sub-digests are derived from the same display bytes the combined pin
  hashes, so they are consistent by construction.
- **Per-position §7.4 macro-block trace pin on the HD fixture (round
  360)** — the `dimensions-1080p-very-short` decode test previously
  checked only the 5-bucket frame-1 mode histogram. It now pins **every
  one of the 8160 macro blocks** at its exact super-block coded-order
  position against the staged instrumented trace: both the §7.4 mode
  index and the §7.4 step 2(d)i MB-coded flag (derived from the §7.3
  `BCODED` luma blocks). The two trace arrays are staged run-length-
  encoded in `fixture_data.rs` (`HD1080_FRAME1_MODE_CODED_ORDER_RLE`,
  `HD1080_FRAME1_CODED_FLAG_ORDER_RLE`; 397 + 253 runs). This catches
  any raster / Hilbert coded-order or mode-decode-position regression a
  bucket-total histogram would silently pass.

### Fixed

- **§7.5.1 quarter-pixel chroma motion compensation (round 356)** — the
  §7.9.4 step 2(d)vi conversion of a per-block motion vector to its
  whole-pixel reference offsets now honours the *per-axis* fractional
  resolution the spec fixes in §7.5.1: a luma component (and any
  non-sub-sampled chroma axis) is half-pixel and divides by 2, but a
  **sub-sampled chroma axis** (horizontal in 4:2:2; both in 4:2:0) is
  **quarter-pixel** (−7.75…7.75 px) and divides by 4. The previous code
  divided every component by 2, so 4:2:0 chroma blocks with a non-zero
  motion vector fetched reference samples twice as far as they should,
  leaving the luma plane bit-exact while chroma drifted (±1 at block
  edges, larger inside) wherever a frame carried real motion. New
  `split_motion_vector_per_axis` performs the per-axis divide; the frame
  driver derives each block's `(hsub, vsub)` from `PF` and `pli` and
  threads them through `ReconstructBlockInputs`. This makes the
  `dimensions-1080p-very-short` HD fixture **pixel-exact**: the SHA-256
  of the two concatenated cropped display frames now equals the
  `c48344b1…` digest recorded in the corpus `notes.md`, closing the
  prior open HD-fidelity item. `reconstruct_frame` gains a trailing
  `pf: PixelFormat` argument. +1 helper-validation test (`sha256_hex`
  known-vector guard); the HD test now asserts the pixel SHA.

### Added

- **HD (1920×1088) two-frame decode integration test (round 350)**. The
  `dimensions-1080p-very-short` corpus fixture is now exercised
  end-to-end. Its Ogg-encapsulated Theora packets were demuxed offline
  (the corpus README's documented framing-strip step) into
  `HD1080_IDENT_PACKET` / `HD1080_SETUP_PACKET` / `HD1080_DATA_PACKET_0`
  (intra) / `HD1080_DATA_PACKET_1` (inter). The new test
  `decode_frame_hd1080_two_frame_trace_invariants` decodes both frames
  and pins every invariant the staged instrumented trace records
  bit-exactly: §6.2 geometry at scale (`NMBS = 8160`, `NSBS = 3060`,
  `NBS = 48960`), the §7.1 per-frame headers (I `qis = 31,20,41`; P
  `qis = 31,20,40`), the §6.4.1 loop-filter limit (`qi = 31 → 3`), the
  full §7.4 frame-1 macro-block mode histogram across all 8160 MBs
  (`6829/1/3/1255/72`, matching the trace `MB` events exactly), and the
  §2.2 visible crop dimensions (1920×1080, `6_220_800` bytes for two
  4:2:0 frames). This is the crate's first end-to-end coverage of the
  large-frame geometry and the complete inter-mode set
  (`INTER_NO_MV` / `INTER_MV` / `INTER_MV_LAST` / `INTER_MV_LAST2`).
  +1 test.

  **Update (round 356): this HD pixel-fidelity item is now resolved.**
  The discrepancy was the §7.5.1 quarter-pixel chroma motion-vector
  interpretation (see the round-356 *Fixed* entry above); the HD test
  now asserts the cropped-output SHA-256 equals the fixture's recorded
  `c48344b1…` digest.

- **`INTER_MV_LAST` / `INTER_MV_LAST2` mode selection (round 347, third
  commit)**. The inter encoder now runs the §7.5.2 LAST1 / LAST2 state
  machine forward over the coded macro-block order and recodes an
  `INTER_MV` macro block whose vector equals the running LAST1 / LAST2 as
  the predicted `INTER_MV_LAST` / `INTER_MV_LAST2` mode — identical
  reconstruction (the reference frame is Previous for all three modes,
  so the residual and DC predictor are unaffected) but no explicit MV
  bits. A uniformly translated frame drives many macro blocks to the same
  vector, so most reuse LAST1; the packet round-trips through the decoder
  and exercises the predicted-MV decode arms end-to-end. +1 test.

- **`oxideav_core::Encoder` P-frame wiring (round 347, second commit)**.
  `TheoraEncoder::with_keyframe_interval` adds a `-g`-style keyframe
  interval: the first frame and every interval-boundary frame is an
  intra keyframe, the frames between are inter (P) frames. The encoder
  mirrors its own output through an internal `FrameDecoder` so each P
  frame predicts from the byte-identical reconstructed reference a
  downstream decoder holds. `TheoraEncoder::new` keeps the historical
  all-keyframe behaviour (interval 1). A full `Encoder` → `Decoder`
  round-trip with interval 3 confirms the I,P,P keyframe-flag pattern
  and reconstructs every frame within the quantizer bound. +1 test.

- **inter (P-frame) encoder (round 347)**. `FrameEncoder` now encodes
  §7 *inter* video-data packets that round-trip through this crate's
  own decoder. New production §7.2 run-length bit-string encoders
  (`encode_long_run_bit_string` / `encode_short_run_bit_string`) are the
  exact inverses of the §7.2.1 / §7.2.2 decoders; `encode_coded_block_flags`
  derives the §7.3 `SBPCODED` / `SBFCODED` / per-block streams from a
  `bcoded` array; `encode_macroblock_modes` emits the §7.4 mode stream
  (`MSCHEME = 7` direct modes); and `encode_macroblock_motion_vectors`
  emits the §7.5 MV stream (`MVMODE = 1` fixed-length components).
  `FrameEncoder::encode_inter_frame` is the zero-MV (`INTER_NOMV`)
  baseline — each block is predicted from the **reconstructed** previous
  reference, so an unchanged frame reconstructs bit-exactly as an
  all-uncoded pure copy and a changed frame stays within the quantizer
  bound. `encode_inter_frame_motion` adds a per-macro-block whole-pixel
  SAD motion search, codes the winning vector with `INTER_MV`, and
  forces that macro block's blocks coded (an uncoded inter block always
  copies the zero-MV colocated reference, so a non-zero MV requires
  coded blocks); a horizontally translated frame round-trips and the
  search measurably reduces luma SAD versus the zero-MV baseline. +5
  tests (two run-length round-trips, identical-frame pure copy,
  changed-frame round-trip, motion round-trip + SAD reduction).

- **encoder chroma-format + multi-frame coverage (round 342, third
  encoder commit)**. The `oxideav_core::Encoder` path is now exercised
  across all three chroma-sampling formats (4:2:0 / 4:2:2 / 4:4:4): a
  flat frame reconstructs losslessly under each, with the per-format
  chroma plane geometry driven by `FrameGeometry` and the advertised
  `output_params.pixel_format` mapped per format — coverage the
  4:2:0-only fixture corpus cannot reach end-to-end. A multi-frame
  sequence test confirms the encoder is reusable across frames, emits
  exactly one keyframe data packet per `send_frame` after the three
  headers, and advances the auto-assigned PTS monotonically. +2 tests.

- **`oxideav_core::Encoder` trait integration (round 342, second
  encoder commit)**. New public `TheoraEncoder` implements
  `oxideav_core::Encoder`, and `register` now installs an encoder
  factory (`make_encoder`) alongside the decoder. On construction the
  encoder serializes the three §6 header packets (via the new
  `encode_*_header` functions) and queues them as the first three
  `receive_packet` outputs (each flagged `PacketFlags::header`); every
  `send_frame` converts a top-down `oxideav_core::VideoFrame` at the
  coded macro-block-aligned dimensions into one §7 intra video-data
  packet (flagged keyframe) via `FrameEncoder::encode_intra_frame`,
  reversing the §2.1 row order and stripping stride padding.
  `output_params` advertises the coded dimensions, the mapped pixel
  format, and a length-prefixed extradata chain carrying the three
  headers (the same format `make_decoder` consumes), so a muxer hands a
  self-describing setup downstream. `make_encoder` rebuilds an encoder
  from that extradata (ident + setup headers required) at the `qi`
  codec-option quantizer (default 32). A complete encode → decode loop
  now round-trips through both `oxideav_core` traits: a flat frame is
  lossless, a 32×32 gradient stays within the quantizer bound, and the
  full path also round-trips through the shared `CodecRegistry`
  (`first_encoder` → `first_decoder`). +6 tests; the registration test
  was updated to assert both a decoder and an encoder are installed.

- **§6 header-packet serialization (round 342, first encoder commit)**.
  New public `encode_identification_header` (§6.2), `encode_comment_header`
  (§6.3), and `encode_setup_header` (§6.4) — the exact inverses of
  `decode_identification_header`, `parse_comment_header`, and
  `decode_setup_header`. The identification serializer is *byte-exact*:
  re-encoding either fixture ident packet reproduces the original 42-byte
  payload, and a struct → bytes → struct round-trip preserves every
  field across all three pixel formats and a reserved color space. The
  setup serializer writes the §6.4.5 body (LFLIMS with minimal NBITS, the
  §6.4.2 quantization parameters with fresh `NEWQR=1` ranges, and all 80
  §6.4.4 Huffman tables emitted from a pre-order tree walk that
  reproduces the exact ISLEAF/TOKEN bits the decoder reads) onto one
  shared `BitWriter`, mirroring the decode-side shared-reader contract;
  it round-trips the full fixture setup tables and a VP3-default-scale
  variant back to an equal `SetupHeaderTables` (LFLIMS + every quant
  parameter + every Huffman table). New `ColorSpace::to_byte` inverse and
  `Error::EncodeHeaderFieldOutOfRange` for fields that overrun their
  on-wire width. This is the keystone unblocker for a full
  `oxideav_core::Encoder` integration that emits a complete decodable
  stream. +9 tests.

- **intra-frame encoder — encode → decode self-roundtrip (round 338
  milestone)**. New public `FrameEncoder` / `SourceFrame` turn a
  macro-block-aligned source frame into a §7 video-data packet that
  this crate's own `FrameDecoder` reconstructs faithfully (to within
  the quantizer step). The forward pipeline inverts the decoder's: per
  block, extract the 8×8 spatial samples, subtract the §7.9.1.1 intra
  predictor (constant 128), apply the §7.9.3.3 forward DCT, and
  `quantize_block` to zig-zag coefficients; run `forward_dc_prediction`
  so each DC slot carries the §7.7 residual; then write the §7.1 intra
  frame header (`0` data bit, `FTYPE=0`, 6-bit `QIS[0]`, `MOREQIS=0`,
  3 reserved zero bits) and the §7.7.3 token stream onto one shared
  `BitWriter`. An intra, all-coded, single-`qi` frame needs no §7.3 /
  §7.4 / §7.5 / §7.6 bits, matching the decoder's no-bit-read
  short-circuits. Supporting additions: the MSb-first `BitWriter` (the
  encoder-side inverse of the §5.2 `BitReader`); the §7.7 forward token
  writer (`encode_dct_coefficients_inner` + `plan_block_tokens`,
  emitting single-coefficient tokens 9..=22, zero-run token 8, and a
  self-terminating EOB so EOB runs never span blocks), which
  round-trips quantized coefficients bit-for-bit through
  `decode_dct_coefficients` across both luma and chroma table
  selectors. New error variants `EncodeCoefficientOutOfRange`,
  `EncodeHuffmanTokenMissing`, `EncodePlaneLenMismatch`,
  `EncodeBlockOutOfPlane`. End-to-end: a 32×32 synthetic gradient
  round-trips with max Y error 5 / mean 0.27 at qi=63 (chroma exact)
  and a flat-128 frame round-trips losslessly; +13 tests (forward
  token round-trip, planner shapes, bit-writer round-trip, three
  encode→decode self-roundtrips, and the geometry/qi rejects).

- forward DC prediction — the inverse of §7.8.2 (round 338, third
  encoder commit). New public `forward_dc_prediction(bcoded, mbmodes,
  block_to_macro_block, neighbors, plane_raster_order, coeffs)` replaces
  each coded block's reconstructed DC (held in `COEFFS[bi][0]` on entry)
  with the residual the §7.7 token stream carries: `residual =
  (reconstructed_dc − DCPRED) as i16`. The predictor is formed by the
  same §7.8.1 `compute_dc_predictor` the decoder uses, against a frozen
  copy of the reconstructed DC column (so the encoder predicts from the
  same values `invert_dc_prediction` rebuilds), advancing the per-plane
  `LASTDC` register file identically. Forward then inverse is the exact
  identity on the reconstructed DCs for any `i16` value. 4 new tests
  (two-block and 2×2-grid round-trips back through
  `invert_dc_prediction`, uncoded-block skip, and the plane-count
  reject).

- forward (encoder-side) quantization — the inverse of §7.9.2
  dequantization (round 338, second encoder commit). New public
  `quantize_block(dqc, qmat_dc, qmat_ac) -> [i16; 64]` divides each
  natural-order forward-DCT coefficient `DQC[ci]` by its quantizer and
  rounds to nearest (ties away from zero), writing the result into the
  zig-zag slot `ZIGZAG_NATURAL_TO_ZIGZAG[ci]` — exactly inverting
  `dequantize_block`'s `DQC[ci] = COEFFS[zzi] * QMAT[ci]` read
  addressing, with an `i16` clamp on the quotient. Round-to-nearest
  minimises the dequantization error so the forward → quantize →
  dequantize → inverse-DCT path is faithful to within the quantizer
  step. Companion `quantize_block_from_params(dqc, params, qti, pli,
  qi0, qi)` builds the two §6.4.3 matrices inline, mirroring
  `dequantize_block_from_params`. 7 new tests (round-trip inverse on
  exact multiples, ties-away rounding, zig-zag AC routing, all-zero,
  params-driven round-trip, and error propagation).

- §7.9.3.3 forward DCT — the non-normative encoder-side transform
  (round 338, first encoder commit). New public `forward_dct_1d(x:
  &[i16; 8]) -> [i16; 8]` transcribes the 31-step §7.9.3.3 signal-flow
  procedure (the reverse of §7.9.3.1 with the rotation-constant signs
  flipped and the C4 scale factors moved to the opposite butterfly
  side) using the same Table 7.65 constants as the inverse, and
  `forward_dct_2d(input: &[[i16; 8]; 8]) -> [i16; 64]` applies it
  row-then-column to produce natural-order coefficients in the exact
  scale the §7.9.2 dequantizer expects. The §7.9.3.3 `//2^16`
  floor-division-toward-zero is implemented by the `fdct_div_2p16`
  helper (the spec's 0xFFFF-negative-offset form). A flat block of
  value `v` forward-transforms to a DC-only `DQC[0] ≈ 32 * v` that the
  §7.9.3.2 inverse maps back to `v` exactly; textured blocks round-trip
  through forward → inverse with ≤ 2 levels of fixed-point error and no
  quantization. 4 new tests. This is the first piece of the deferred
  encoder; quantization, forward DC prediction, the token entropy
  writer, and the frame-level encode driver follow.

- exercise the golden-reference and four-MV inter modes through the
  full §7.9.4 `reconstruct_frame` driver (round 334). Two new
  frame-driver integration tests close the gap between the existing
  block-level (`reconstruct_block`) and motion-vector-level
  (`decode_macroblock_motion_vectors`) coverage of these modes and a
  full-frame reconstruction run: (1) an `INTER_GOLDEN_NOMV` frame whose
  every block samples the **golden** reference plane (Table 7.46
  `rfi = 2`) rather than the previous one, verified per-pixel across all
  three planes with distinct previous/golden ramp biases; (2) an
  `INTER_MV_FOUR` macroblock whose four luma blocks each carry an
  independent whole-pixel motion vector while the chroma blocks consume
  the averaged MV, verified against the previous reference (Table 7.46
  `rfi = 1`) including the §7.9.1.2 edge-coordinate clamp on the chroma
  block. No reference-captured corpus fixture covers these modes, since
  the reference encoder's `testsrc`-class encodes never emit a golden or four-MV
  macroblock; these tests pin the reconstruction behaviour directly.
- pin the `keyframe-interval-1` fixture end-to-end (round 331): decode
  the 32×32 four-packet stream in which *every* frame is a fresh INTRA
  keyframe (`-g 1`) and compare all four reconstructions against the
  corpus `expected.yuv` sample-exactly. This is the first multi-frame
  pin that is a run of back-to-back keyframes (the existing multi-frame
  pins are I + P sequences), and it asserts the keyframe reference-store
  invariant on every frame: a keyframe re-seeds **both** the golden and
  the previous reference with itself. Adds the `KI1_DATA_PACKET_0..3`
  and `KI1_EXPECTED_YUV` fixture constants (identification header reused
  from `KI30_IDENT_PACKET`, setup header from the shared
  `FIXTURE_SETUP_PACKET`) and one end-to-end test.
- pin the `monochrome-via-zero-chroma` fixture end-to-end (round 325):
  decode the 64×64 two-frame I+P grayscale-source stream and compare both
  frames against the corpus `expected.yuv` sample-exactly. This is the
  first fixture to assert the **flat-chroma invariant** — both chroma
  planes stay uniformly `0x80` across the inter reconstruction (covering
  both the top-row coded INTER_NO_MV residual branch and the lower-row
  pure-copy branch). Adds the `MONO_IDENT_PACKET` / `MONO_IFRAME_PACKET`
  / `MONO_PFRAME_PACKET` / `MONO_EXPECTED_YUV` fixture constants and one
  end-to-end test (setup header reused verbatim from the shared
  `FIXTURE_SETUP_PACKET`; SHA-256 of the YUV recorded in the constant's
  doc comment).
- pin the `picture-region-non-mb-aligned` fixture end-to-end (round 321):
  decode the single intra frame (coded 32×32) and crop to its 26×18
  visible region, comparing against the corpus `expected.yuv`. This is
  the only corpus dump captured *after* the §2.2 crop (702 bytes:
  26×18 Y + 13×9 Cb + 13×9 Cr), so it is the first fixture to validate
  the §4.4.4 chroma round-up (`ceil(26/2)=13` × `ceil(18/2)=9`, not
  13×8) and the non-identity `crop_for_display` path against reference
  pixels rather than at the macroblock-aligned coded dimensions. Adds
  the `PICREG_IDENT_PACKET` / `PICREG_DATA_PACKET` / `PICREG_EXPECTED_YUV`
  fixture constants and one end-to-end test (setup header reused verbatim
  from the shared `FIXTURE_SETUP_PACKET`).
- wire the `oxideav_core::Decoder` trait (round 317): `TheoraDecoder`
  implements the framework decoder interface and `register` installs it
  into a `RuntimeContext` (the `register!` entry point was previously a
  no-op). Packets are dispatched by their §6.1 type bit — header packets
  (`0x80`/`0x81`/`0x82`) advance header collection, video-data packets
  decode through `FrameDecoder`, get §2.2-cropped to the picture region,
  and are flipped from the §2.1 lower-left order to top-down raster as an
  `oxideav_core::VideoFrame`. Headers can arrive as leading stream
  packets, via `TheoraDecoder::with_headers`, or as a length-prefixed
  `CodecParameters::extradata` chain consumed by the `make_decoder`
  factory. Adds `THEORA_CODEC_ID`, `TheoraDecoder`, `make_decoder`,
  `register_codecs`, and a `FrameDecoder::setup_tables` accessor, plus
  six integration tests (the `tiny-i-only-16x16` fixture decodes
  sample-exactly through the trait path).
- pin the `bitstream-version-3.2.1` fixture end-to-end (round 313):
  the single-frame 32×32 stream whose identification header reports
  version `0x030201` (`VMAJ=3`, `VMIN=2`, `VREV=1`, i.e. the
  `>= 0x030200` alpha3+ feature set per §6.2) decodes sample-exactly.
  Adds `BV321_IDENT_PACKET` / `BV321_DATA_PACKET` / `BV321_EXPECTED_YUV`
  to the embedded fixture data (Ogg framing stripped offline; the setup
  header is byte-identical to the shared `FIXTURE_SETUP_PACKET`) and an
  end-to-end §7.11 assertion against the staged reference dump. This is
  the first pin to assert the exact version-field value the decoder
  branches on for the alpha3+ feature set.

## [0.0.10](https://github.com/OxideAV/oxideav-theora/compare/v0.0.9...v0.0.10) - 2026-06-15

### Other

- pin q-high weak-quant fixture end-to-end (round 309)
- §2.2/§4.4.4 picture-region display crop
- §7.11 full keyframe-interval-30 run — 8-packet sustained decode, sample-exact
- inter-frame end-to-end decode + §7.9.4 half-pixel MC fix
- state clean-room sources positively in README
- §7.11 complete frame decode — intra end-to-end, sample-exact against staged fixtures
- §7.11 steps 1(e)+(f)+(g) — complete the step-1 chain with §7.6 qi, §7.7.3 coefficients, and §7.8.2 DC inversion
- §7.11 step 1(d) — thread §7.5.2 motion vectors onto the shared step-1 reader
- §7.11 step 1(c) — thread §7.4 macro-block coding modes onto the shared step-1 reader
- scrub pre-existing decorative libtheora/FFmpeg attribution from doc comments
- §7.11 step 1(a)+1(b) header+coded-block-flags chain (round 267)
- §7.11 step 2 (empty-packet) state synthesiser (round 260)
- §7.11 step 7 + step 8 reference-frame promotion (round 256)
- reference-plane geometry + packet classifier (round 250)
- drop release-plz.toml — use release-plz defaults across the workspace
- §7.10.3 Complete Loop Filter raster-order driver (round 244)
- §7.10.1 / §7.10.2 loop-filter edge primitives + lflim() (round 241)
- §2.3 / §2.4 coded-order resolver (round 238)
- §7.9.4 frame-level driver (round 233)
- §7.9.4 The Complete Reconstruction Algorithm — per-block body (round 23)
- §7.9.3 Inverse DCT — 1D + 2D (round 22)
- §7.9.1 Predictors: intra / whole-pixel / half-pixel (round 21)
- §7.9.2 Dequantization (round 20)
- §7.8.2 Inverting the DC Prediction Process (round 19)
- §7.8.1 Computing the DC Predictor (round 18)
- §7.7.3 DCT Coefficient Decode driver (round 17)

### Fixed

- §7.9.4 step 2(d)vi inter motion-compensation: `MVECTS` components
  are in half-pixel units (luma) / quarter-pixel units (sub-sampled
  chroma axes), not whole-pixel. `coded_block_pred_res` had hard-coded
  `MVX2 = MVX`, forcing the whole-pixel predictor (§7.9.1.2) for every
  inter block; explicit non-zero motion vectors with an odd component
  produced visibly wrong predictors. It now halves each component via
  `split_half_pixel_motion_vector` (`MVX = ⌊|MVECTS|/2⌋·sign`,
  `MVX2 = ⌈|MVECTS|/2⌉·sign`) and dispatches to the §7.9.1.3 half-pixel
  predictor when the offsets differ (round 288). New
  `Error::ReconstructMotionVectorOutOfRange` guards the (unreachable
  for spec-conformant input) i8 overflow of the halved offset pair.

### Added

- `q-high` weak-quant end-to-end decode pin (round 309): the 64×64,
  2-frame `q-high` fixture (bitstream version 3.2.1, `-q:v 10` →
  qi=63, `ac_scale=10`) decodes sample-exactly across both frames.
  At the weakest quantiser far more DCT AC coefficients survive
  dequantisation than in the strong-quant pins, so this is the
  hardest exercise of the §7.7.3 / §7.9.2 / §7.9.3 coefficient path
  to date (the §7.9.3.2 two-pass inverse DCT runs on most blocks
  rather than the DC-only shortcut). The fixture also pins two frame
  configurations no prior end-to-end test reached: `nqps=2` (a
  two-`qi` frame header, so the §7.6 block-level `qi` decode runs
  exactly one `NQIS − 1` long-run pass), and `filter_limit=0` at
  qi=63 (this stream's transmitted §6.4.1 `LFLIMS[63]` is zero, so
  the §7.10.3 loop filter collapses to identity via the
  `lflim(R, 0) = 0` response, leaving the reconstruction untouched —
  the decode is sample-exact regardless). The keyframe codes all 16
  macro blocks INTRA; the inter frame codes the first macro-block row
  `INTER_NOMV` and copies the remaining three rows from the previous
  reference. The identification and two data packets were
  Ogg-de-framed offline (the crate never parses Ogg); the
  setup-header packet is byte-identical to the existing fixtures and
  is reused. +1 test (511 → 512).

- §2.2 / §4.4.4 picture-region crop (round 302):
  `crop_frame_to_picture_region` and the `FrameDecoder::crop_for_display`
  convenience wrapper return a `CroppedFrame` cropped to the display
  picture region — the step §7.11 defers to the caller ("the frame
  must be cropped to the picture region before display"). Luma crops to
  `PICW` × `PICH` at `(PICX, PICY)`; chroma planes follow §4.4.4 with a
  floor-low / ceil-high per-axis window so every chroma sample whose
  luma footprint touches the picture region is retained (correct for
  odd offsets and odd sizes). Buffers stay lower-left row-major; `PICY`
  is a lower-left offset so no vertical flip is applied. New
  `Error::CropPlaneLenMismatch` / `Error::CropRegionOutOfPlane` guard
  inconsistent header-vs-frame geometry. +10 tests.

- Sustained multi-frame end-to-end decode pin (round 295): the full
  8-packet `keyframe-interval-30` fixture (32×32, bitstream version
  3.2.1) decodes sample-exactly across its entire run. The stream
  exercises a keyframe seeding both references, two consecutive
  **zero-byte** packets that take the §7.11 step 2 "Otherwise" branch
  (each synthesises an `INTER` / `NQIS=1` / `QIS[0]=63` /
  all-uncoded frame that reproduces the carried previous reference,
  so both replay the keyframe), and five real inter frames decoded in
  sequence with the reference store chained forward. The corpus
  `expected.yuv` records the six displayed frames (the two
  duplicate-only zero-byte packets are collapsed by the dump tool, so
  the 8 packets map onto 6 reference frames); the empty packets are
  validated by asserting they reproduce frame 0. The data-packet and
  identification-header bytes were Ogg-de-framed offline (the crate
  never parses Ogg); the setup-header packet is byte-identical to the
  existing `tiny-i` / `quant-table-custom` fixture and is reused.

- Inter-frame end-to-end decode pins (round 288): the
  `i-frame-then-p-frame-64x64` fixture (keyframe + `INTER_NOMV` /
  uncoded MODE_COPY P frame) and the `q-low` fixture (P frame with an
  explicit `INTER_PLUS_MV` non-zero motion vector) both reconstruct
  their two frames sample-exactly against the staged `expected.yuv`,
  exercising the §7.9.4 whole- and half-pixel motion-compensated
  reconstruction path against the carried reference store.

- §7.11 Complete Frame Decode dispatch — an intra frame now decodes
  end-to-end from a real Theora packet (round 284). Three new public
  surfaces:
  - `decode_setup_header` + `SetupHeaderTables` — the full §6.4.5
    setup-header body decode (§6.4.1 `LFLIMS` → §6.4.2 quantization
    parameters → §6.4.4 80-entry Huffman-table array) on one shared
    bit reader. `parse_setup_header` now delegates to it and returns
    a populated `TheoraSetupHeader` instead of the
    body-not-implemented sentinel.
  - `build_frame_geometry` + `FrameGeometry` — the resolved §2.3 /
    §2.4 frame layout: continuous cross-plane coded-order block
    numbering, `block → super block` / `macro block → luma blocks` /
    per-macro-block chroma maps, per-block `(pli, BX, BY, mbi)`
    tables, the §7.8.1 raster-neighbour table, and the per-plane
    raster orderings shared by §7.8.2 / §7.9.4 / §7.10.3. Pinned
    against the §2.3 / §2.4 240×48 worked examples. A new
    `PlaneBlockCodedOrder::from_block_extent` constructor handles
    subsampled chroma planes whose block extent is not
    `2 * ceil(mb / 2)` (odd-`FMBW`/`FMBH` 4:2:0 and 4:2:2 streams).
  - `FrameDecoder` + `DecodedFrame` — the stateful §7.11 steps 1–8
    driver: step 1 chain / step 2 empty-packet synthesis, §7.9.4
    reconstruction against the carried reference planes (step 5),
    §7.10.3 loop filter (step 6), and golden / previous reference
    promotion (steps 7–8) across frames.
- End-to-end fixture pins: `tiny-i-only-16x16` (solid-colour intra,
  qi=44) and `quant-table-custom` (textured intra, qi=18, 3-qi
  header) decode sample-exact against their staged `expected.yuv`
  reference dumps (packet bytes embedded from the corpus with Ogg
  framing stripped offline); a zero-byte packet replays the previous
  reference bit-identically with `QIS = [63]`; the transmitted setup
  tables match every `LOOP_FILTER` / `QUANT` trace event the corpus
  records for those streams.

- §7.11 steps 1(e) + 1(f) + 1(g) — the step 1 chain is complete
  (round 281). `decode_data_packet_header_and_blocks` now threads the
  §7.6 block-level `qi` decode (step 1(e)) and the §7.7.3 DCT
  coefficient decode (step 1(f)) onto the same shared `BitReader` —
  §7.6's `NQIS − 1` long-run passes resume immediately after the
  §7.5.2 MV stream and §7.7.3's `htiL` / `htiC` / token stream
  immediately after §7.6's, with no byte re-alignment anywhere — then
  applies the §7.8.2 DC-prediction inversion (step 1(g), which reads
  no bits) to the decoded coefficients in place. Two new inputs:
  `hts` (the 80-element §6.4.4 Huffman table array §7.11 lists among
  its inputs) and `dc_geometry` (the new public
  `DcPredictionGeometry` bundling the §7.8.2 `bi → mbi` map,
  raster-neighbour table, and per-plane raster orderings, mirroring
  the `ChromaBlockLayout` precedent). The typed
  `DataPacketHeaderAndBlocks` gains `qiis: Vec<u8>`,
  `coeffs: Vec<[i16; 64]>` (DC already reconstructed per 1(g)), and
  `ncoeffs: Vec<u8>` fields. Step 1(f)'s spec sentence "decode the
  DCT coefficients into NCOEFFS and NCOEFFS" carries a documented
  typo note (§7.7.3's outputs are `COEFFS` and `NCOEFFS`). All §7.6 /
  §7.7.x / §7.8.x reject paths propagate unchanged through the
  driver.
- §7.11 step 1(d) — the §7.5.2 motion vectors now extend the composed
  step 1 chain a fourth link (round 278).
  `decode_data_packet_header_and_blocks` gains two inputs — `pf` (the
  pixel format §7.5.2 needs for the `INTER_MV_FOUR` chroma-MV
  averaging of steps 3(a)x..3(a)xii) and `chroma_map` (the
  per-macro-block `ChromaBlockLayout` §7.5.2 writes chroma MVs
  through) — and threads the §7.5.2 procedure onto the same shared
  `BitReader` immediately after the §7.4 mode stream, with no
  re-alignment. The typed `DataPacketHeaderAndBlocks` gains an
  `mvects: Vec<MotionVector>` field (the `NBS`-element `MVECTS` array
  the §7.11 step 5 / §7.9.4 reconstruction call consumes). Step
  1(d)'s spec wording gates the §7.5.2 call on `FTYPE` being non-zero
  (inter frame); the driver realises the gate through §7.5.2's own
  intra short-circuit (§7.5 opening sentence: intra frames carry no
  motion vectors and consume no bits), which yields identical bit
  consumption (none) and output (all-zero `MVECTS`) while keeping the
  §7.5.2 shape validation uniform across frame types — matching how
  the step 1(c) link validates the luma map on intra frames. All
  §7.5.2 reject paths (`MotionVectorMbModesLenMismatch`,
  `MotionVectorLumaMapLenMismatch`,
  `MotionVectorLumaBlockIndexOutOfRange`,
  `MotionVectorChromaMapLenMismatch`,
  `MotionVectorChromaMacroBlockSlotLenMismatch`,
  `MotionVectorChromaBlockIndexOutOfRange`, truncation) propagate
  unchanged. +2 net tests (484 → 486): a §7.5.2 chroma-map-length
  reject through the driver (fires before the `MVMODE` bit is read)
  and a `LAST1`/`LAST2` register-file walk (`INTER_MV` seed →
  `INTER_MV_LAST` reuse → `INTER_MV_LAST2` swap → fresh `INTER_MV`)
  under `MVMODE=0` Table 7.23 Huffman components. The prior step-1
  tests were extended to assert the new `MVECTS` output, including
  per-luma `INTER_MV_FOUR` writes, the 4:2:0 chroma average with a
  ties-away-from-zero `round()` case (`round(2/4) = 1`), and
  coded-blocks-only step 3(g) propagation. The step 1 chain now
  covers 1(a)+1(b)+1(c)+1(d); steps 1(e)..1(g) and the step 5 / step
  6 dispatch into `reconstruct_frame` / `loop_filter_frame` remain
  pending.

- §7.11 step 1(c) — the §7.4 macro-block coding modes now extend the
  composed step 1 chain a third link (round 274).
  `decode_data_packet_header_and_blocks` gains two inputs — `nmbs`
  (the macro-block count `NMBS`) and `macro_block_to_luma_blocks`
  (the `NMBS`-element luma-block map §7.4 step 2(d)i reads `BCODED`
  through) — and threads the §7.4 modes procedure onto the same
  shared `BitReader` immediately after the §7.3 streams, with no
  re-alignment. The typed `DataPacketHeaderAndBlocks` gains an
  `mbmodes: Vec<MacroBlockMode>` field (the `MBMODES` array) and an
  `nmbs()` accessor. On an intra frame every mode is `Intra` (§7.4
  step 1, no bits read); on an inter frame the `MSCHEME` / alphabet /
  mode stream is decoded on the shared cursor, including §7.4 step
  2(d)ii's no-bits `INTER_NOMV` for a wholly-uncoded macro block. All
  §7.4 reject paths (`MacroBlockLumaMapLenMismatch`,
  `MacroBlockLumaBlockIndexOutOfRange`, `UnknownMacroBlockModeCode`,
  truncation) propagate unchanged. +2 net tests (482 → 484): a §7.4
  luma-map-length reject and an inter path where one macro block is
  wholly uncoded (mode skipped, shared reader stays aligned for its
  coded neighbours). The seven existing step-1(a)+(b) tests were
  extended to assert the new `MBMODES` output. The step 1 chain now
  covers 1(a)+1(b)+1(c); steps 1(d)..1(g) and the step 5 / step 6
  dispatch into `reconstruct_frame` / `loop_filter_frame` remain
  pending.

- §7.11 step 1(a) + step 1(b) chain — the first composed link of the
  "size of the data packet is non-zero" branch (round 267). New
  public function `decode_data_packet_header_and_blocks(packet,
  first_frame, nsbs, nbs, block_to_super_block) ->
  Result<DataPacketHeaderAndBlocks, Error>` decodes the §7.1 frame
  header (step 1(a)) and the §7.3 coded block flags (step 1(b))
  against a **single shared** `BitReader`, honouring the
  shared-reader contract the standalone byte-aligned entry points
  cannot: §7.3's run-length streams resume at the bit position
  immediately after the §7.1 header with no re-alignment. The typed
  `DataPacketHeaderAndBlocks { header, bcoded }` exposes `ftype()`,
  `nqis()`, and `nbs()` accessors for the downstream step 1(c)..1(g)
  links. All §7.1 and §7.3 reject paths propagate unchanged. +7
  tests (475 → 482): intra all-coded, two inter shared-reader paths
  (all-coded and mixed `BCODED` via the per-block short-run stream),
  the header-packet / first-frame-must-be-intra / block-map-length
  rejects, and an accessor-consistency check.

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

### Fixed

- §7.9.3.1 inverse-DCT sine constants: Table 7.65 pairs `Ci` with
  `S(8 − i)` (`S3 = 36410 = C5`, `S6 = 60547 = C2`,
  `S7 = 64277 = C1`); the previous transcription aliased `Sj = Cj`,
  attenuating low-frequency AC energy. The slip is invisible on
  DC-only content (the DC path only uses `C4`) and was caught by the
  first fixture carrying real AC coefficients; it is now pinned by
  the sample-exact decode tests plus a float-identity check on each
  constant.

### Removed

- `Error::SetupHeaderBodyNotImplemented` — the §6.4.5 body decode it
  guarded is now implemented; no path constructs it.

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
