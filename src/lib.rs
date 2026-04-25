// Parallel-array index loops read more naturally for block/IDCT code.
#![allow(clippy::needless_range_loop)]

//! Theora video codec (Xiph On2-VP3 descendant).
//!
//! Status:
//! * Header parsing (identification + comment + setup) — done.
//! * I-frame decode — done; bit-exact IDCT, integer-domain dequant, full
//!   loop filter.
//! * P-frame (inter) decode — done; macroblock modes, motion vectors
//!   (LAST/LAST2/golden), 4-MV mode, half-pel motion compensation,
//!   per-RFI DC prediction, and golden-frame reference are all wired up.
//! * I-frame **encode** — done (intra-only). Header packets are constructed
//!   from input parameters; the setup table is shipped as embedded standard
//!   libtheora bytes; per-frame DCT + DC prediction + token RLE +
//!   Huffman encoding mirrors the decode path.
//! * P-frame **encode** — done. Full-pel motion estimation + half-pel
//!   refinement within `±DEFAULT_ME_RANGE`; mode decision picks between
//!   INTRA, INTER_NOMV, INTER_MV, INTER_MV_LAST, INTER_MV_LAST2,
//!   INTER_GOLDEN_NOMV, INTER_GOLDEN_MV, and INTER_MV_FOUR (per-8x8
//!   sub-block MVs, selected when the RD-biased 4-MV SAD beats the 1-MV
//!   candidate). LAST and GOLDEN reference frames are managed and updated
//!   per spec. Configurable keyframe interval, ME range, 4-MV toggle, and
//!   optional CBR rate-control loop via [`encoder::EncoderOptions`] (see
//!   [`encoder::make_encoder_with_options`] and
//!   [`encoder::RateControlOptions`]).
//!   Limitations: 4-MV sub-block search is a diamond pattern seeded from
//!   the 16x16 best MV rather than an independent full scan.

pub mod bitreader;
pub mod block;
pub mod coded_order;
pub mod dct;
pub mod decoder;
pub mod encoder;
pub mod encoder_huffman;
pub mod fdct;
pub mod headers;
pub mod huffman;
pub mod inter;
pub mod quant;

use oxideav_core::{
    CodecCapabilities, CodecId, CodecParameters, CodecTag, PixelFormat as CorePixelFormat, Result,
};
use oxideav_core::{CodecInfo, CodecRegistry, Decoder, Encoder};

pub const CODEC_ID_STR: &str = "theora";

pub fn register(reg: &mut CodecRegistry) {
    let cid = CodecId::new(CODEC_ID_STR);
    let caps = CodecCapabilities::video("theora_sw")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(16384, 16384)
        .with_pixel_formats(vec![
            CorePixelFormat::Yuv420P,
            CorePixelFormat::Yuv422P,
            CorePixelFormat::Yuv444P,
        ]);
    // AVI FourCC — `THEO` is the conventional marker when Theora is
    // wrapped in AVI (rare; Ogg is the canonical container).
    reg.register(
        CodecInfo::new(cid.clone())
            .capabilities(caps)
            .decoder(make_decoder)
            .tag(CodecTag::fourcc(b"THEO")),
    );
    let enc_caps = CodecCapabilities::video("theora_sw_enc")
        .with_lossy(true)
        .with_intra_only(false)
        .with_max_size(16384, 16384)
        .with_pixel_formats(vec![
            CorePixelFormat::Yuv420P,
            CorePixelFormat::Yuv422P,
            CorePixelFormat::Yuv444P,
        ]);
    reg.register(
        CodecInfo::new(cid)
            .capabilities(enc_caps)
            .encoder(make_encoder),
    );
}

fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    decoder::make_decoder(params)
}

fn make_encoder(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    encoder::make_encoder(params)
}

/// Public factory intended for tests / ad-hoc integration.
pub fn make_decoder_for_tests(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    decoder::make_decoder(params)
}

/// Public factory intended for tests / ad-hoc integration.
pub fn make_encoder_for_tests(params: &CodecParameters) -> Result<Box<dyn Encoder>> {
    encoder::make_encoder(params)
}

pub use decoder::{
    classify_packet, codec_parameters_from_identification, count_mb_modes_in_frame, FrameType,
    PacketKind,
};
pub use headers::{
    parse_comment_header, parse_headers_from_extradata, parse_identification_header,
    parse_setup_header, parse_xiph_extradata, Comment, Headers, Identification, PixelFormat, Setup,
};
