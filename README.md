# oxideav-theora

Pure-Rust Theora video codec — clean-room implementation, in progress.
The crate works on Theora packets directly (it does not parse the Ogg
container); callers de-frame the bitstream packets and hand them in. The
decoder is sample-exact for its supported feature set; an intra-frame
encoder now round-trips through the crate's own decoder faithfully.

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
packets are handled as duplicate-frame markers.

* **Intra encoder** — `FrameEncoder` turns a macro-block-aligned
  `SourceFrame` into a §7 video-data packet. The forward pipeline
  inverts the decoder's: the §7.9.3.3 forward DCT (`forward_dct_1d` /
  `forward_dct_2d`), forward quantization (`quantize_block`, the inverse
  of §7.9.2), forward §7.8 DC prediction (`forward_dc_prediction`), and
  the §7.7 token entropy writer (single-coefficient tokens 9–22,
  zero-run token 8, self-terminating EOB) over a new MSb-first
  `BitWriter`. An intra, all-coded, single-`qi` frame emits no §7.3 /
  §7.4 / §7.5 / §7.6 bits, matching the decoder's intra short-circuits.
  An encoded INTRA frame decodes back through this crate's own
  `FrameDecoder` faithfully to within the quantizer step (a 32×32
  gradient round-trips with max luma error 5 / mean 0.27 at the weakest
  quantiser, chroma exact; a flat frame is lossless).

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
  frame stays within the quantizer bound. The motion path runs a small
  whole-pixel SAD search per macro block, codes the winning vector with
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
  and §7.5 `MVMODE = 1` explicit-MV bits, weighted by a
  quantizer-derived multiplier λ (`inter_rd_lambda`, monotone in `qi`).
  This subsumes the raw-SAD previous-vs-golden choice into one decision
  on reconstructed distortion; `INTER_MV` winners still flow through the
  §7.5.2 `INTER_MV_LAST` / `INTER_MV_LAST2` recode. On a scene the golden
  reference predicts perfectly the RD path codes a distortion-free golden
  copy (SSD 0, 25 B) where the previous-reference-only motion path must
  code a large residual against the unrelated previous reference
  (SSD 50264, 252 B) — a measured strict win on both distortion and
  rate. The mode decision has **no bitstream-syntax effect**: every
  chosen mode emits the same §7 bytes the existing writers produce.
  `TheoraEncoder` drives its P-frames through this RD path by default
  (selectable via `InterModeStrategy` / `with_inter_mode`). A
  target-bitrate rate-control loop remains future work.

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
  the full setup tables back to an equal `SetupHeaderTables`.

* **Framework `Encoder` integration** — `TheoraEncoder` implements
  `oxideav_core::Encoder`, and `register` now installs it (alongside the
  decoder) via the `make_encoder` factory. The encoder serializes the
  three §6 headers up front and emits them as the first three packets
  (flagged `header`), then turns each top-down `VideoFrame` at the coded
  dimensions into one §7 data packet. The keyframe interval (`-g`,
  `TheoraEncoder::with_keyframe_interval`, default 1) decides intra vs
  inter: interval-boundary frames are intra keyframes, the frames
  between are inter (P) frames predicted from the reconstructed previous
  reference. The encoder mirrors its own output through an internal
  `FrameDecoder` so the reference it predicts from is byte-identical to
  the downstream decoder's. `output_params` carries the coded
  dimensions, the mapped pixel format, and a length-prefixed extradata
  header chain, so a complete encode → decode loop round-trips through
  both `oxideav_core` traits and through the shared `CodecRegistry`
  (`first_encoder` → `first_decoder`): a flat frame is lossless, a
  gradient stays within the quantizer bound, and an I,P,P interval-3
  sequence reconstructs every frame within the quantizer bound.

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
* **Rate control + four-MV in the RD decision** — every §7.5.2 inter
  mode (`INTER_NOMV`, `INTER_MV`, `INTER_MV_LAST`, `INTER_MV_LAST2`,
  `INTER_GOLDEN_NOMV`, `INTER_GOLDEN_MV`, `INTER_MV_FOUR`) is reachable
  from the encoder, and the **uniform** inter modes are now chosen by a
  joint rate-distortion decision (`encode_inter_frame_rd`, the
  `TheoraEncoder` P-frame default — see the inter-encoder bullet above).
  Still future work: folding `INTER_MV_FOUR` into the same RD candidate
  set (its per-luma-block search remains a separate entry point), and a
  rate control / target-bitrate loop (the frame quantizer is fixed at
  construction). The setup header (§6.4 quantization parameters +
  Huffman tables) is supplied by the caller via `TheoraEncoder::new` /
  `extradata` rather than synthesized from scratch.
* **Reference-captured golden / four-MV corpus fixture** — the
  golden-reference and four-MV inter modes are exercised
  top-to-bottom through this crate's own encoder→decoder round trip
  (the §7.5.2 four-MV decode + chroma averaging and §7.9.4 golden /
  per-block reconstruction all run against a real, self-encoded
  bitstream). A *third-party*-captured `expected.yuv` fixture for these
  modes is still absent because the reference encoder's `testsrc`-class
  encodes never emit a golden or four-MV macroblock.

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
