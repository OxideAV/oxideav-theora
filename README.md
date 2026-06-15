# oxideav-theora

Pure-Rust Theora video decoder — clean-room implementation, in
progress. The crate decodes Theora packets directly (it does not parse
the Ogg container); callers de-frame the bitstream packets and hand them
in. There is no encoder yet.

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
  rounding for all three pixel formats.

`FrameDecoder::decode_frame` chains the per-packet path
(header → block decode → reconstruction → in-place loop filter →
reference promotion) and is the high-level entry point. Empty (zero-byte)
packets are handled as duplicate-frame markers.

End-to-end fixtures decoded sample-exactly cover intra-only streams,
intra-then-inter sequences, explicit motion vectors, custom quantisation
tables, weakest- and strongest-quantiser streams, non-MB-aligned picture
regions, and a sustained multi-frame keyframe-interval run.

## Not yet supported

* **Ogg container parsing** — out of scope here; packets are supplied
  pre-de-framed by the caller.
* **Encoder** — none. The forward DCT (non-normative in the spec) is
  deferred to a future encoder-side effort.
* **Golden-reference and four-MV inter modes** (`INTER_GOLDEN_MV` /
  `INTER_MV_FOUR`) — implemented in the reconstruction code paths but
  not yet exercised end-to-end by the fixture corpus.
* **Framework `Decoder`/`Encoder` trait integration** — the
  `oxideav_core::register!` entry point is currently a no-op; use the
  direct functions above.

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

## Clean-room sources

Only the published Theora I Specification (`docs/video/theora/`), the
staged §6.4.1 procedure body transcribed from the specification's own
source for the section the PDF omits, and the fixture corpus under
`docs/video/theora/fixtures/` are consulted. Reference binaries are used
only as opaque black-box validators.

## License

MIT. See [LICENSE](LICENSE).
