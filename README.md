# oxideav-theora

[![CI](https://github.com/OxideAV/oxideav-theora/actions/workflows/ci.yml/badge.svg)](https://github.com/OxideAV/oxideav-theora/actions/workflows/ci.yml) [![crates.io](https://img.shields.io/crates/v/oxideav-theora.svg)](https://crates.io/crates/oxideav-theora) [![docs.rs](https://docs.rs/oxideav-theora/badge.svg)](https://docs.rs/oxideav-theora) [![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Pure-Rust Theora video codec — clean-room implementation, in progress.
The crate works on Theora packets directly (it does not parse the Ogg
container); callers de-frame the bitstream packets and hand them in. The
decoder is sample-exact for its supported feature set (validated up to
HD against reference dumps). The encoder emits complete self-describing
streams — §6 headers plus intra and inter (P) frames through a unified
rate-distortion mode decision over all eight §7.5.2 coding modes, with
motion estimation to half-pixel accuracy, per-block skip, adaptive
quantization, content-tuned Huffman codebooks, non-macro-block-aligned
picture regions, duplicate-frame packets, and optional target-bitrate
rate control — and is **externally validated**: twelve scenario
families (up to 1920×1080, all three pixel formats, every encoder
feature above) were muxed into Ogg and black-box-decoded by an
independent reference decoder **pixel-exactly** against this crate's
own reconstruction, with the corpus pinned by SHA-256 under `tests/`.

## What works

The decode pipeline is wired end-to-end at the packet level. For the
supported feature set the decoder reconstructs frames **sample-exactly**
against reference output.

* **Headers** — identification (`decode_identification_header`), comment
  (`parse_comment_header`), and setup (`parse_setup_header` /
  `decode_setup_header`, including loop-filter limits, quantisation
  parameters, and the DCT-token Huffman tables).
* **Frame header & bitstream** — frame header decode, coded-block flags,
  macroblock modes, motion-vector decode (zero-MV and explicit per-block
  MV with the half-pixel / quarter-pixel split), block-level `qi`
  selection, EOB and coefficient tokens, and full DCT coefficient
  decode.
* **Reconstruction** — DC prediction and inversion, the inverse DCT
  (DC-only shortcut plus the full two-pass path), intra and
  motion-compensated inter reconstruction (with the §7.5.1 per-axis
  fractional-resolution motion vectors: half-pixel on luma /
  non-sub-sampled axes, quarter-pixel on each sub-sampled chroma axis,
  validated pixel-exact at HD), the complete frame-level reconstruction
  driver, and reference-frame promotion.
* **Loop filter** — the edge primitives, `lflim()` response, and the
  complete raster-order loop-filter driver (applied in place; collapses
  to identity at a zero filter limit).
* **Display crop** — `crop_frame_to_picture_region` (and the
  `FrameDecoder::crop_for_display` wrapper) crops the macroblock-aligned
  reconstruction down to the picture region, with the spec's chroma-axis
  rounding for all three pixel formats. A non-MB-aligned fixture
  (visible 26×18 inside coded 32×32) is validated end-to-end against a
  reference dump captured *after* the §2.2 crop, exercising the §4.4.4
  chroma round-up (13×9, not 13×8) on real reference pixels.

`FrameDecoder::decode_frame` chains the per-packet path
(header → block decode → reconstruction → in-place loop filter →
reference promotion) and is the high-level entry point. Empty (zero-byte)
packets are handled as duplicate-frame markers — except as the very first
packet, where there is no reference to duplicate: that is rejected with
`Error::FirstFrameEmptyPacket` rather than reconstructing a frame from the
zero-initialized reference store.

* **Intra encoder** — `FrameEncoder` turns a macro-block-aligned
  `SourceFrame` into a §7 video-data packet. The forward pipeline
  inverts the decoder's: the §7.9.3.3 forward DCT (`forward_dct_1d` /
  `forward_dct_2d`), forward quantization (`quantize_block`, the inverse
  of §7.9.2), forward §7.8 DC prediction (`forward_dc_prediction`), and
  the §7.7 token entropy writer over a new MSb-first `BitWriter`. The
  token writer uses the **full §7.7 token alphabet**: single-coefficient
  tokens 9–22, both pure zero-run tokens (3-bit token 7 for short gaps,
  6-bit token 8 for long ones), the combined run+value tokens 23–31
  (folding a zero gap and its terminating coefficient into one Huffman
  code), and **cross-block §7.7.1 EOB runs** (tokens 0–6): the writer
  simulates the §7.7.3 decoder's visit order once and coalesces every
  maximal run of consecutive block-end visits into a single EOB-run
  token — the rest of the run's blocks consume no bits at all, with
  runs freely crossing the `ti = 1` selector refresh and the
  luma/chroma boundary (runs past 4095 split greedily). It also
  **optimizes the §7.7.3 Huffman selectors independently per range**:
  the `ti = 0` pair only ever addresses group 0 (DC) and the re-read
  `ti = 1` pair groups 1–4 (AC), so the writer tallies the tokens
  actually written and picks each pair's cheapest table separately — a
  pure bit-rate win at zero distortion. Together these cut the intra
  measurement harness ~39% versus the round-379 writer at identical
  reconstruction. An intra, all-coded, single-`qi` frame emits no §7.3 /
  §7.4 / §7.5 / §7.6 bits, matching the decoder's intra short-circuits.
  An encoded INTRA frame decodes back through this crate's own
  `FrameDecoder` faithfully to within the quantizer step (a 32×32
  gradient round-trips with max luma error 5 / mean 0.27 at the weakest
  quantiser, chroma exact; a flat frame is lossless).

* **Adaptive block-level quantization** —
  `FrameEncoder::encode_intra_frame_with_qiis` is the first encoder
  path to emit a §7.1 multi-`qi` frame header (the `MOREQIS` / `QIS`
  chain, up to `NQIS = 3`) and the §7.6 block-level qi stream
  (`NQIS − 1` §7.2.1 long-run promotion passes, the exact inverse of
  the decoder's `decode_block_level_qi`): `qis[0]` drives every DC
  quantizer (the §7.6 preamble's rule, so DC prediction is untouched)
  and the loop-filter limit, while each block's AC quantizer is
  `qis[qiis[bi]]`. `encode_intra_frame_adaptive` layers a per-block
  rate-distortion chooser on top — each block is quantized once per
  candidate, reconstructed exactly as the decoder will, and scored
  `D + λ·R` — and `TheoraEncoder::with_adaptive_quant` drives every
  keyframe through it end-to-end via the framework `Encoder` trait
  (inter frames keep the single frame-level `qi`). Tests re-parse the
  emitted packets through the production §7.11 step-1 driver and pin
  the wire `QIS` / `QIIS` arrays exactly; the `NQIS = 1` case is
  byte-identical to the single-`qi` packet.

* **Content-tuned Huffman codebooks (two-pass encoding)** —
  `HuffmanTable::from_token_counts` builds a minimum-redundancy §6.4.4
  codebook from per-token emission counts (all 32 tokens leafed,
  canonical codes satisfying the Kraft equality — a full tree —
  deterministic ties, 16-bit length cap).
  `FrameEncoder::intra_token_statistics` tallies the tokens an intra
  encode would write (combined tokens and EOB runs included) into a
  `TokenStatistics` accumulator without producing a packet, and
  `SetupHeaderTables::with_tuned_huffman_tables` replaces, per group,
  the selector slot scoring worst on the sampled luma statistics with
  the luma-tuned codebook and the worst remaining slot with the
  chroma-tuned one — so content resembling the sample keeps the
  fallback it would actually pick. The tuned set serializes through
  `encode_setup_header`, keeping the stream self-describing;
  `TheoraEncoder::with_tuned_setup_keyframe_interval` wraps the whole
  two-pass flow in one constructor. Measured: a textured intra frame
  shrinks a further 19–23% versus the VP3-default tables at
  bit-identical reconstruction.

* **Round 387 — inter-side encoder quality.** Four additions drive the
  P-frame rate curves at equal reconstruction:

  * **Inter token statistics + mixed I/P GOP tuning** —
    `FrameEncoder::inter_token_statistics` tallies the §7.7 tokens an
    RD inter encode would write (EOB runs and combined tokens
    included) without producing a packet, and
    `SetupHeaderTables::with_gop_tuned_huffman_tables` tunes **four**
    codebooks per Huffman group (intra/inter × luma/chroma), each
    replacing the selector slot scoring worst on its own statistics
    column. `TheoraEncoder::with_gop_tuned_setup_keyframe_interval`
    runs the first pass as the exact I/P GOP the stream will be
    (inter frames planned by RD against a mirror decoder's
    references). Measured on a 6-frame textured motion sequence at
    interval 3: 2248 B (VP3 defaults) → 2141 B (intra-only tuning) →
    2072 B GOP-tuned, all three reconstructing bit-identically.
  * **Measured token rate everywhere** — `TokenBitCosts` prices every
    RD candidate's exact token plan against the stream's own codebooks
    (group-minimum code length + exact extra bits), replacing the flat
    bits-per-coefficient proxies in both the inter mode decision and
    the intra adaptive-quant chooser. The §7.4 mode writer likewise
    picks the cheapest **mode-coding scheme** per frame (direct
    scheme 7, the six fixed Table 7.19 alphabets, or a
    frequency-sorted scheme-0 custom alphabet — a golden-dominated
    31-MB mode section spells in 8 B vs 12 B direct).
  * **INTRA in the P-frame candidate set** — the RD decision weighs
    all **eight** §7.5.2 coding modes: an INTRA macro block codes from
    scratch (flat-128 predictor, qti = 0 matrices per §7.9.4 step
    2(d)ii) when neither reference predicts it. On a
    checkerboard→gradient quadrant switch: 26 B / luma SSD 224 vs the
    intra-less motion path's 149 B / SSD 4052.
  * **Inter adaptive quantization** —
    `FrameEncoder::encode_inter_frame_rd_adaptive` emits the §7.1
    `MOREQIS` chain + §7.6 block-level qi stream on P-frames, each
    coded block's AC quantizer chosen by `D + λ·R` on delivered SSD +
    measured rate (`TheoraEncoder::with_adaptive_quant` now covers
    both frame types); an adaptive `[0, 63]` spread lands strictly
    inside the single-`qi` rate/distortion corners (pinned in-test at
    equal reference state).
  * **Measured-rate keyframe policy (golden-frame refresh)** —
    `TheoraEncoder::with_keyframe_rate_policy(r)` converts a P-frame
    to a true keyframe when its references have decayed (size ratio
    past `r ×` the last keyframe, or a majority-INTRA mode plan) *and*
    the intra spelling wins a Lagrangian comparison on SSD measured
    through a throwaway mirror-decoder clone — re-seeding both
    references (only intra frames refresh the golden frame). A
    content-family switch at interval 30 converts the de-facto
    keyframes while staying within a pinned whole-stream Lagrangian
    bound of the interval-only spelling (a greedy per-frame policy
    buys its refreshed references at a small, bounded premium).

* **Inter (P-frame) encoder** — `FrameEncoder::encode_inter_frame`
  (zero-MV baseline) and `encode_inter_frame_motion` (with motion
  estimation) emit a §7 *inter* video-data packet that round-trips
  through this crate's own decoder. The forward pipeline drives the
  complete inter bit-stream chain: the §7.2.1 / §7.2.2 run-length
  bit-string encoders (`encode_long_run_bit_string` /
  `encode_short_run_bit_string`, exact inverses of the decoders), the
  §7.3 coded-block-flag encoder (`encode_coded_block_flags`, deriving
  the `SBPCODED` / `SBFCODED` / per-block streams from a `bcoded`
  array), the §7.4 macro-block-mode encoder (`MSCHEME = 7` direct
  modes), the §7.5 motion-vector encoder (`MVMODE = 1` fixed-length
  components), and the shared §7.7 token writer. Each block is
  predicted from the **reconstructed** previous reference (the bytes
  the decoder holds), so an unchanged second frame reconstructs
  bit-exactly as an all-uncoded `INTER_NOMV` pure copy; a changed
  frame stays within the quantizer bound. The motion path runs a
  **four-step whole-pixel SAD descent** per macro block (steps 8, 4,
  2, 1 around the running winner: every displacement in ±15 pixels
  reachable with 33 probes, zero-biased, §7.5.1 ±31-component clamp;
  a (6, −5) translation spells a 16 B motion packet where the zero-MV
  spelling needs 125 B) **then refines the winner to
  half-pixel accuracy** (§7.5.1): the integer search only spells even
  motion-vector components, so the eight half-pixel neighbours are
  probed and any strict SAD improvement kept, making the decoder's
  §7.9.1.3 two-tap half-pixel predictor reachable from the encoder (a
  linear ramp sampled half a pixel off the reference grid drops the
  whole-plane luma SAD from 1184 to 704, −40.5 %, at equal
  reconstruction). It codes the winning vector with
  `INTER_MV`, and forces its blocks coded (an uncoded inter block always
  copies the zero-MV colocated reference, so a non-zero MV requires
  coded blocks). A forward §7.5.2 LAST1 / LAST2 pass then recodes a
  macro block whose vector matches the running last vectors as the
  predicted `INTER_MV_LAST` / `INTER_MV_LAST2` mode (no explicit MV
  bits, identical reconstruction). A horizontally translated frame
  round-trips and the motion search measurably reduces the luma SAD
  versus the zero-MV baseline; a uniformly translated frame drives most
  macro blocks through the predicted-MV modes.

  Two further entry points emit the remaining inter modes:
  `encode_inter_frame_golden` searches **both** the previous and golden
  references per macro block and codes `INTER_GOLDEN_NOMV` /
  `INTER_GOLDEN_MV` when the golden reference predicts the block's four
  luma sub-blocks with the smaller SAD (reconstructing against the
  golden motion-compensated predictor), and `encode_inter_frame_four_mv`
  searches one vector **per luma block** and codes `INTER_MV_FOUR` when
  the four vectors disagree — each luma block carrying its own vector
  and each chroma block the §7.5.2 averaged vector for the pixel format.
  Both modes are correctly handled by the §7.5.2 LAST1 / LAST2 predictor
  history (golden excluded; four-MV updating LAST1 to its last coded
  luma vector). Each round-trips through this crate's own decoder: a
  golden copy of a re-shown reference reconstructs bit-exactly, and a
  per-luma-block-displaced frame reconstructs within the quantizer
  bound — so every §7.5.2 inter mode is now reachable from the encoder.

  A **unified rate-distortion mode decision**
  (`encode_inter_frame_rd`) replaces the fixed per-entry-point SAD
  heuristic with one joint per-macro-block Lagrangian choice. Each macro
  block evaluates every reachable uniform inter mode (`INTER_NOMV`,
  `INTER_MV`, `INTER_GOLDEN_NOMV`, `INTER_GOLDEN_MV`) by its true cost
  `D + λ·R` and codes the cheapest. The distortion `D` is the sum of
  squared errors between the source and the block the decoder will
  actually reconstruct — the encoder runs its own forward
  quantize → dequantize → inverse-DCT → clamp (`block_rd_cost`), so a
  candidate is scored on *delivered* fidelity rather than a
  pre-quantization SAD proxy; the rate `R` folds in the §7.4 mode code
  and §7.5 `MVMODE = 1` explicit-MV bits, weighted by a multiplier λ
  derived from the frame's **actual quantizer step** (λ ∝ step² on the
  §6.4.3 luma first-AC quantization value; multi-`qi` frames take the
  geometric mean of their candidate steps). External measurement —
  encode → Ogg mux → black-box reference decode → PSNR — calibrated
  the constant and caught the previous raw-`qi` ramp sloping the wrong
  way (a higher `qi` is a *smaller* step in Theora): the I+P
  rate/PSNR curve used to fold over above qi 52 (44.8 → 39.0 dB at
  qi 63 with rate still rising) and is now monotone on both axes
  (qi 63 at 51.4 dB for +4.5 % bytes).
  This subsumes the raw-SAD previous-vs-golden choice into one decision
  on reconstructed distortion; `INTER_MV` winners still flow through the
  §7.5.2 `INTER_MV_LAST` / `INTER_MV_LAST2` recode. The RD rate term is
  **predictor-aware**: the decision tracks the running `LAST1` / `LAST2`
  motion-vector predictor and charges an `INTER_MV` candidate zero
  explicit-MV bits when its vector matches the predictor (it will recode
  to a no-MV-bit LAST mode), so a vector the predictor already supplies
  for free is not over-penalised against `INTER_NOMV`. **`INTER_MV_FOUR` is
  part of the same candidate set**: each macro block also evaluates a
  per-luma-block search (four independent vectors, chroma averaged) by its
  true `D + λ·R` cost and codes four-MV when it beats every uniform mode —
  its four explicit vectors are paid for out of the rate term, so it is
  chosen only when the per-block fidelity gain justifies them (an
  all-equal four-MV is strictly dominated by uniform `INTER_MV` and never
  wins). On a scene the golden
  reference predicts perfectly the RD path codes a distortion-free golden
  copy (SSD 0, 25 B) where the previous-reference-only motion path must
  code a large residual against the unrelated previous reference
  (SSD 50264, 252 B) — a measured strict win on both distortion and
  rate. The mode decision has **no bitstream-syntax effect**: every
  chosen mode emits the same §7 bytes the existing writers produce.
  `TheoraEncoder` drives its P-frames through this RD path by default;
  `with_inter_mode` selects an alternative `InterModeStrategy` —
  `PreviousMotion` (previous-reference SAD), `GoldenMotion` (previous-vs-
  golden raw-SAD), or `FourMv` (per-luma-block four-MV search) — each
  reachable end-to-end through the framework `Encoder` trait.

  A **per-block rate-distortion skip decision** completes the coded /
  uncoded choice: in an `INTER_NOMV` macro block the decoder's uncoded
  path is a bit-exact zero-MV previous copy — exactly the predictor
  the encoder built — so a block whose quantized residual survives
  quantization may still legally *skip*. Every inter path scores both
  spellings by the frame's Lagrangian (skip: predictor SSD, zero bits;
  code: delivered SSD + λ·measured token bits) and keeps the cheaper
  one; `block_rd_cost` applies the same min inside the mode decision.
  A noise-only P-frame (±3, residuals surviving a mid-strength qi-40
  quantizer) drops from ~140 B to 5 B and reconstructs as a bit-exact
  reference copy, while a strongly changed block in the same conditions
  still codes and tracks the source (at the *weakest* quantizers the
  step-derived λ correctly prefers coding the noise — bits are cheap
  in that regime). When the skip decision empties a
  `TheoraEncoder` P-frame entirely (a **duplicate frame**), the
  encoder emits the spec's own syntax for that: a §7.11 step-2
  zero-byte packet, decoding bit-exactly as a previous-frame copy.

* **Framework `Decoder` integration** — `TheoraDecoder` implements
  `oxideav_core::Decoder`, and `register` installs it into a
  `RuntimeContext` (reachable through the shared `CodecRegistry`). Packets
  are dispatched by their §6.1 type bit: header packets (`0x80` / `0x81`
  / `0x82`) advance header collection, video-data packets are decoded,
  §2.2-cropped to the picture region, and flipped to top-down raster as
  an `oxideav_core::VideoFrame`. Headers may arrive inline as leading
  stream packets, via `TheoraDecoder::with_headers`, or as a
  length-prefixed `CodecParameters::extradata` chain parsed by the
  `make_decoder` factory.

* **§6 header-packet serialization** — `encode_identification_header`
  (§6.2), `encode_comment_header` (§6.3), and `encode_setup_header`
  (§6.4) are the exact inverses of the three header decoders. The
  identification serializer is *byte-exact* (re-encoding a decoded ident
  packet reproduces the original payload); the setup serializer writes
  the §6.4.5 body (LFLIMS with minimal `NBITS`, the §6.4.2 quantization
  parameters as fresh `NEWQR=1` ranges, and all 80 §6.4.4 Huffman tables
  via a pre-order tree walk) onto one shared `BitWriter` and round-trips
  the full setup tables back to an equal `SetupHeaderTables`. The full
  §6.4.5 bundle can also be **synthesized from the published Appendix B
  data** — `SetupHeaderTables::vp3_defaults()` assembles the §B.2 / §B.3
  scale tables, the §B.4 base matrices + single-range quant assignment,
  and the §B.4 80 DCT-token Huffman codebooks (`HuffmanTable::from_code_list`
  rebuilds each codebook into the byte-identical §6.4.4 tree), so the
  encoder can emit a conformant setup header with no caller-supplied
  tables.

* **Framework `Encoder` integration** — `TheoraEncoder` implements
  `oxideav_core::Encoder`, and `register` now installs it (alongside the
  decoder) via the `make_encoder` factory (which needs only the §6.2
  identification header in `extradata` — a §6.4 setup header there is
  optional, synthesized from the VP3 defaults when omitted). The encoder
  serializes the three §6 headers up front and emits them as the first
  three packets
  (flagged `header`), then turns each top-down `VideoFrame` — at the
  coded dimensions or at the §2.2 picture dimensions (see the
  picture-region bullet below) — into one §7 data packet. The keyframe
  interval (`-g`,
  `TheoraEncoder::with_keyframe_interval`, default 1) decides intra vs
  inter: interval-boundary frames are intra keyframes, the frames
  between are inter (P) frames predicted from the reconstructed previous
  reference. Optional **scene-cut detection**
  (`with_scene_cut_threshold`) inserts a fresh keyframe mid-GOP when the
  mean absolute luma difference between the incoming source and the
  previous reconstruction exceeds the threshold **and** more than
  doubles the GOP's running average difference — the relative gate
  separates a cut (a one-frame spike) from steadily fast-moving
  content, which an absolute threshold alone converted to a keyframe
  storm (externally measured: the gated spelling of a moving-texture
  stream is 33 % smaller *and* 1.4 dB better); a scene change makes
  inter prediction worthless, so an intra frame is both smaller and
  resets the references; the keyframe-interval counter restarts from the
  inserted keyframe. The encoder mirrors its own output through an internal
  `FrameDecoder` so the reference it predicts from is byte-identical to
  the downstream decoder's. `output_params` carries the picture
  dimensions (equal to the coded ones for a full-frame picture region),
  the mapped pixel format, and a length-prefixed extradata
  header chain, so a complete encode → decode loop round-trips through
  both `oxideav_core` traits and through the shared `CodecRegistry`
  (`first_encoder` → `first_decoder`): a flat frame is lossless, a
  gradient stays within the quantizer bound, and an I,P,P interval-3
  sequence reconstructs every frame within the quantizer bound.

* **Picture-region (non-MB-aligned) encoding** — the encode-side
  counterpart of the display crop. `TheoraIdentHeader::for_picture`
  builds a §6.2 header around arbitrary picture dimensions (smallest
  containing coded frame; region left- and top-aligned — `PICX = 0`,
  `PICY = 16·FMBH − PICH` in the spec's lower-left coordinates — the
  placement observed on the reference-captured fixtures), with
  `picture_plane_dims` exposing the §4.4.4 chroma round-up per pixel
  format, and a container-carriable `KFGSHIFT` default of 6 (a zero
  shift leaves inter frames with no representable §A.2.3 granule
  position — the field stays public for longer GOPs). `SourceFrame::from_picture` pads picture-shaped planes to
  the coded frame by edge replication (§2.2 leaves the outside-region
  samples to the encoder; replication keeps the padding blocks'
  residual energy near zero), and `TheoraEncoder::send_frame` accepts
  picture-shaped `VideoFrame`s directly — exactly the shape
  `TheoraDecoder` emits. A 26×18 picture (coded 32×32, the decode
  fixture's geometry) intra- and inter(RD)-encodes and crops back to
  picture planes faithful to the source across all three chroma
  formats, including the odd-edge 13×9 chroma window at 4:2:0.

* **Target-bitrate rate control** — `TheoraEncoder::with_target_bitrate`
  (and the `with_target_bitrate_bounded` variant for explicit `qi`
  bounds) enables a leaky-bucket rate-control loop that adapts the
  frame-level quantization index before each frame to steer the running
  output size toward a byte budget. Each frame is allotted
  `target_bits / fps` bits (the frame rate read from the §6.2 `FRN / FRD`
  header); the actual coded size feeds a signed fullness accumulator, and
  a proportional, clamped step moves `qi` to oppose the imbalance — a
  stream running ahead of budget *lowers* `qi` (stronger quantization,
  smaller frames; recall a lower `qi` is stronger in Theora), one running
  behind *raises* it (better fidelity). The chosen `qi` lands in the
  frame header `QIS[0]`, so the adaptation has **no bitstream-syntax
  effect**: a downstream decoder reads each frame exactly as it would a
  fixed-`qi` stream. The loop is **keyframe-aware**: because an intra frame
  is typically several times larger than the inter frames around it,
  charging it against a single per-frame budget would spike the bucket and
  slam the quantizer at every GOP boundary; instead a keyframe drains a
  weighted budget (`keyframe_weight × bits_per_frame`) and the surplus is
  repaid in equal shares across the GOP's inter frames, so the long-run
  average target is preserved exactly while a keyframe no longer perturbs
  the frames after it (the controller learns the GOP length from the
  encoder's keyframe interval). The loop is fully opt-in (disabled by
  default, a no-op). **Externally measured** (5-second 320×240 runs
  decoded black-box from the muxed `.ogv`): a 400 kb/s target lands
  +0.5 % and an 800 kb/s target −0.9 %, while a target below the
  content's `qi_min` rate floor saturates cleanly at the floor. Enabling the loop
  also **declares the target in the §6.2 `NOMBR`** nominal-bitrate
  header field (saturating at the 24-bit "`2^24 − 1` or greater"
  ceiling), rewriting the queued ident packet and the advertised
  extradata chain so both header carriage paths agree.

End-to-end fixtures decoded sample-exactly cover intra-only streams,
intra-then-inter sequences, explicit motion vectors, custom quantisation
tables, weakest- and strongest-quantiser streams, a non-MB-aligned
picture region cropped to its visible 26×18 window (compared against a
post-crop reference dump), a single-frame bitstream-version-3.2.1
(alpha3+) stream, a sustained multi-frame keyframe-interval run, an
all-keyframe four-packet `-g 1` run (every frame INTRA, each re-seeding
both references), and a
grayscale-source `monochrome-via-zero-chroma` I+P stream whose chroma
planes stay a flat `0x80` through both the coded-residual and pure-copy
inter branches (asserted in addition to the sample-exact pixel match).

The `dimensions-1080p-very-short` fixture (coded 1920×1088, visible
1920×1080, two frames I then P) is decoded end-to-end and validated
against every invariant the staged instrumented trace records
bit-exactly: the HD §6.2 geometry (`NMBS = 8160`, `NSBS = 3060`,
`NBS = 48960`), both §7.1 frame headers, the §6.4.1 loop-filter limit,
the §2.2 visible-crop dimensions, and the §7.4 frame-1 macro-block
modes — not just the 5-bucket histogram but **every one of the 8160
macro blocks** pinned at its exact super-block coded-order position
against the trace (mode index + derived MB-coded flag), so a raster /
Hilbert-order / mode-decode-position regression can no longer pass a
bucket-total check. The decoded mode set spans the full inter range
(`INTER_NO_MV`, `INTER_MV`, `INTER_MV_LAST`, `INTER_MV_LAST2`). It is
also pinned **pixel-exactly**: the SHA-256 of
the two concatenated cropped display frames equals the `c48344b1…`
digest recorded in the fixture's `notes.md`. Frame 0 is the all-intra
keyframe; frame 1 exercises the §7.9.4 inter path including the §7.5.1
quarter-pixel chroma motion compensation — the band of nonzero-MV
chroma blocks that earlier diverged is now sample-exact.

## Not yet supported

* **Ogg container parsing** — out of scope here; packets are supplied
  pre-de-framed by the caller.
* **Encoder mode coverage is complete** — every §7.5.2 inter mode
  (`INTER_NOMV`, `INTER_MV`, `INTER_MV_LAST`, `INTER_MV_LAST2`,
  `INTER_GOLDEN_NOMV`, `INTER_GOLDEN_MV`, `INTER_MV_FOUR`) is reachable
  from the encoder, and **all** of them — including `INTER_MV_FOUR` (a
  per-luma-block search, chroma averaged) — are chosen by the one joint
  rate-distortion decision (`encode_inter_frame_rd`, the `TheoraEncoder`
  P-frame default — see the inter-encoder bullet above). A
  **target-bitrate rate-control loop** is now wired in too (see the
  rate-control bullet above). The encoder also **synthesizes its own
  §6.4 setup header** from scratch (`SetupHeaderTables::vp3_defaults()`:
  the §B.2 loop-filter limits, §B.3 AC/DC scale tables, §B.4 base
  matrices with their single-range quant assignment, and the §B.4 80
  DCT-token Huffman codebooks — all from the published Appendix B data),
  so `TheoraEncoder::with_default_setup` needs only the identification
  header and a quantizer to emit a complete self-describing stream; a
  caller may still supply pre-decoded tables via `TheoraEncoder::new` /
  `extradata` when matching an existing setup.
* **Reference-captured golden / four-MV decode fixture** — the
  golden-reference and four-MV inter modes are exercised
  top-to-bottom through this crate's own encoder→decoder round trip,
  and self-encoded golden / four-MV streams are now also decoded
  **pixel-exactly by an external black-box decoder** (see below). A
  *third-party-encoded* `expected.yuv` fixture for these modes is
  still absent because the reference encoder's `testsrc`-class encodes
  never emit a golden or four-MV macroblock.

## External validation (round 413)

The encoder's output is validated by an **independent reference
decoder**, black-box, end-to-end: twelve scenario families — plain
I/P GOPs at 320×240 and 1920×1080, 4:2:2 and 4:4:4, a non-MB-aligned
130×98 picture region, multi-`qi` adaptive quantization, target-bitrate
rate control (with the `NOMBR` declaration), GOP-tuned custom Huffman
setup headers, scene cuts, the four-MV and golden inter strategies, and
zero-byte duplicate-frame packets — were muxed into `.ogv` through the
published `oxideav-ogg` crate (in a throwaway helper project; this
crate itself takes no container dependency), passed `oggz-validate`,
and decoded to raw video that matched this crate's own reconstruction
**byte-for-byte in all twelve cases**. The same scenario family is
pinned by SHA-256 (wire packets + decoded output) in
`tests/encoded_corpus.rs`, with the route, verdicts, and measured
operating points (externally-measured rate/PSNR curves and
rate-control accuracy within ±1 % of reachable targets) documented in
`tests/encoded-corpus-notes.md`. The external route surfaced and fixed
three real defects this round: `for_picture`'s `KFGSHIFT = 0` (inter
frames had no representable Ogg granule position), a wrong-slope RD λ
(the rate/PSNR curve folded over above qi 52), and a scene-cut
detector that converted steady motion into keyframe storms.

## Usage

```rust
use oxideav_theora::{decode_identification_header, decode_setup_header, FrameDecoder};

// `ident_packet`, `setup_packet`, and the data packets are de-framed
// from the container by the caller.
let ident = decode_identification_header(ident_packet)?;
let setup = decode_setup_header(setup_packet)?;
let mut dec = FrameDecoder::new(ident, setup)?;
let frame = dec.decode_frame(data_packet)?;
let display = dec.crop_for_display(&frame)?;
# Ok::<(), oxideav_theora::Error>(())
```

Alternatively, drive the decoder through the `oxideav_core::Decoder`
trait. `TheoraDecoder` accepts the three §6 header packets (inline, via
`with_headers`, or as a length-prefixed `extradata` chain) and then
emits a top-down `VideoFrame` per video-data packet:

```rust
use oxideav_core::{CodecId, Decoder, Packet, TimeBase};
use oxideav_theora::{TheoraDecoder, THEORA_CODEC_ID};

# fn run(ident: &[u8], setup: &[u8], data: &[u8]) -> oxideav_core::Result<()> {
let mut dec = TheoraDecoder::with_headers(
    CodecId::new(THEORA_CODEC_ID),
    &[ident, setup],
)
.map_err(|e| oxideav_core::Error::invalid(e.to_string()))?;
dec.send_packet(&Packet::new(0, TimeBase::from_rate(1), data.to_vec()))?;
let _frame = dec.receive_frame()?; // Frame::Video, top-down planes
# Ok(())
# }
```

To encode an intra frame, build a `FrameEncoder` from the same header
objects and a quantization index, then hand it a `SourceFrame` (three
planes, lower-left row-major, at the coded macro-block-aligned
dimensions). The produced packet decodes back through `FrameDecoder`:

```rust
use oxideav_theora::{FrameEncoder, FrameDecoder, SourceFrame};
# fn run(ident: oxideav_theora::TheoraIdentHeader,
#        setup: oxideav_theora::SetupHeaderTables,
#        frame: SourceFrame) -> Result<(), oxideav_theora::Error> {
let enc = FrameEncoder::new(ident.clone(), setup.clone(), /* qi */ 32)?;
let packet = enc.encode_intra_frame(&frame)?;
let mut dec = FrameDecoder::new(ident, setup)?;
let _decoded = dec.decode_frame(&packet)?; // faithful to within the quantizer
# Ok(())
# }
```

To emit a complete decodable stream, drive `TheoraEncoder` through the
`oxideav_core::Encoder` trait. It serializes the three §6 headers up
front (returned as the first `receive_packet` outputs, flagged
`header`), then encodes each top-down `VideoFrame` — supplied at the
coded macro-block-aligned dimensions — into one intra data packet:

```rust
use oxideav_core::{CodecId, Encoder, Frame};
use oxideav_theora::{TheoraEncoder, THEORA_CODEC_ID};
# fn run(ident: oxideav_theora::TheoraIdentHeader,
#        setup: oxideav_theora::SetupHeaderTables,
#        frame: Frame) -> oxideav_core::Result<()> {
let mut enc = TheoraEncoder::new(
    CodecId::new(THEORA_CODEC_ID), ident, setup, /* qi */ 32,
)
.map_err(|e| oxideav_core::Error::invalid(e.to_string()))?;
enc.send_frame(&frame)?;
while let Ok(pkt) = enc.receive_packet() {
    // pkt.flags.header marks the three §6 header packets; the rest are
    // §7 intra video-data packets (each a keyframe).
    let _ = pkt;
}
# Ok(())
# }
```

## Clean-room sources

Only the published Theora I Specification (`docs/video/theora/`), the
staged §6.4.1 procedure body transcribed from the specification's own
source for the section the PDF omits, and the fixture corpus under
`docs/video/theora/fixtures/` are consulted. Reference binaries are used
only as opaque black-box validators.

## License

MIT. See [LICENSE](LICENSE).
