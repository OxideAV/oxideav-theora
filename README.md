# oxideav-theora

Pure-Rust **Theora** (Xiph's VP3 descendant) video codec — decoder and
encoder, implemented from the [spec](https://www.theora.org/doc/Theora.pdf)
with zero C dependencies. No `libtheora`, no FFI, no `*-sys` crates.

Part of the [oxideav](https://github.com/OxideAV/oxideav-workspace)
framework but usable standalone.

## Installation

```toml
[dependencies]
oxideav-core = "0.0"
oxideav-codec = "0.0"
oxideav-theora = "0.0"
```

## Decode support

- Three-header bootstrap (identification + comment + setup) from Xiph-laced
  extradata. In-band header parsing is **not** supported; pass headers
  through `CodecParameters::extradata`.
- Intra (key) frames: token decode, DC prediction, integer dequant, bit-exact
  IDCT, full loop filter.
- Inter (P) frames: macro-block modes (including 4-MV), motion vectors
  against the LAST / LAST2 predictors, `INTER_GOLDEN_*` references, half-pel
  motion compensation, per-RFI DC prediction.
- Reference management: golden frame is updated on keyframes; previous
  reconstructed frame drives motion compensation.
- Chroma formats: 4:2:0, 4:2:2, 4:4:4.
- Bit depth: 8-bit (the only depth Theora defines).
- Cropping: decoded output is cropped from the coded-frame size down to
  `(picw, pich)` at offset `(picx, picy)` as the ID header specifies.
- Interop verified against ffmpeg-generated Ogg Theora clips (I-only and GOP)
  in the test suite.

## Encode support

- Intra (key) frames: forward DCT, integer-domain quantisation, forward DC
  prediction, token RLE, Huffman encoding against the setup header's trees.
- Inter (P) frames: SAD-based mode decision per macro-block with full-pel
  motion search inside `me_range`, considering
  `{INTER_NOMV, INTRA, INTER_MV, INTER_MV_LAST, INTER_MV_LAST2,
  INTER_GOLDEN_NOMV, INTER_GOLDEN_MV}`.
- Chroma formats: 4:2:0, 4:2:2, 4:4:4.
- Reference tracking: encoder reconstructs each frame internally (same
  dequant/IDCT/loop-filter path as the decoder), keeping the golden / LAST
  references bit-exact for the decoder to follow.
- Configurable keyframe interval and motion-search range via
  `encoder::EncoderOptions` + `encoder::make_encoder_with_options`.
- The setup header (loop filter limits, AC/DC scale tables, quant base
  matrices, 80 Huffman trees) is shipped verbatim as a libtheora-compatible
  reference blob; the ID and comment headers are synthesized from
  `CodecParameters`.
- Verified with ffmpeg: Ogg-muxed output decodes cleanly in
  `ffmpeg`/`libtheora`, and the round-trip via our own decoder achieves more
  than 35 dB PSNR on synthetic inputs across all three chroma formats.

### Known encoder limitations

- Integer-pel motion estimation only (no sub-pel refinement).
- No rate control: the quantisation index is fixed
  (`EncoderOptions::qi`, default `DEFAULT_QI = 32`).
- `INTER_MV_FOUR` (4-MV) mode is decoded but not produced by the encoder.
- Motion search is a brute-force full-pel scan within `±me_range`.

## Quick use

### Decoder

```rust
use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, Packet, TimeBase};

let mut codecs = CodecRegistry::new();
oxideav_theora::register(&mut codecs);

let mut params = CodecParameters::video(CodecId::new("theora"));
params.extradata = xiph_laced_headers;
let mut dec = codecs.make_decoder(&params)?;

let pkt = Packet::new(0, TimeBase::new(1, 90_000), frame_packet_bytes);
dec.send_packet(&pkt)?;
while let Ok(Frame::Video(vf)) = dec.receive_frame() {
    // vf.format in { Yuv420P, Yuv422P, Yuv444P }
    // vf.planes[0/1/2] carry luma/cb/cr, already cropped to (picw, pich)
}
# Ok::<(), oxideav_core::Error>(())
```

### Encoder

```rust
use oxideav_codec::CodecRegistry;
use oxideav_core::{CodecId, CodecParameters, Frame, PixelFormat, Rational};

let mut codecs = CodecRegistry::new();
oxideav_theora::register(&mut codecs);

let mut params = CodecParameters::video(CodecId::new("theora"));
params.width = Some(w);
params.height = Some(h);
params.pixel_format = Some(PixelFormat::Yuv420P);
params.frame_rate = Some(Rational::new(24, 1));
let mut enc = codecs.make_encoder(&params)?;

enc.send_frame(&Frame::Video(frame))?;
while let Ok(pkt) = enc.receive_packet() {
    // First three packets carry `pkt.flags.header = true`
    // (identification / comment / setup) — wire them into your container.
    // Subsequent packets are frame data; keyframes carry
    // `pkt.flags.keyframe = true`.
}
# Ok::<(), oxideav_core::Error>(())
```

The three-header extradata is also available pre-laced on
`enc.output_params().extradata` for containers that prefer it there.

### Codec ID

- Codec: `"theora"`; accepted pixel formats `Yuv420P`, `Yuv422P`, `Yuv444P`.

## License

MIT — see [LICENSE](LICENSE).
