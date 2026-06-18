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
  motion-compensated inter reconstruction, the complete frame-level
  reconstruction driver, and reference-frame promotion.
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

## Not yet supported

* **Ogg container parsing** — out of scope here; packets are supplied
  pre-de-framed by the caller.
* **Inter-frame encoding** — the encoder is intra (keyframe) only. Inter
  prediction, motion estimation, mode decision, and rate control are
  future work. Header-packet *serialization* (writing the §6
  identification / comment / setup headers) is also not yet
  implemented: the encoder reuses the same in-memory header objects the
  decoder consumes, so a full encode that emits a complete stream
  (headers + data) — and the `oxideav_core::Encoder` trait integration
  that depends on it — remain to be wired.
* **Golden-reference and four-MV inter modes** (`INTER_GOLDEN_MV` /
  `INTER_MV_FOUR`) — implemented in the reconstruction code paths and
  now exercised through the full `reconstruct_frame` driver
  (golden-reference plane selection and per-block four-MV luma motion
  with the averaged chroma MV), but not yet by a *reference-captured*
  corpus fixture: the reference encoder's `testsrc`-class encodes never emit a
  golden or four-MV macroblock, so no `expected.yuv` fixture covers
  these modes top-to-bottom from a real bitstream.
* **Framework `Encoder` trait integration** — the intra encoder is
  reachable through the direct `FrameEncoder` API, but there is no
  `oxideav_core::Encoder` impl yet (it needs the header-packet
  serialization noted above). The decoder side is fully wired (see
  above).

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

## Clean-room sources

Only the published Theora I Specification (`docs/video/theora/`), the
staged §6.4.1 procedure body transcribed from the specification's own
source for the section the PDF omits, and the fixture corpus under
`docs/video/theora/fixtures/` are consulted. Reference binaries are used
only as opaque black-box validators.

## License

MIT. See [LICENSE](LICENSE).
