#![no_main]

//! Hostile §7 video-data packets against a live [`FrameDecoder`].
//!
//! Three config bytes pick a small geometry — picture dimensions
//! 1..=48 on each axis (non-macro-block-aligned included, so the
//! §2.2 / §4.4.4 display crop always runs) and any of the three
//! pixel formats — then the rest of the buffer is framed into up to
//! eight length-prefixed packets fed to `decode_frame` in order.
//! Feeding several packets through one decoder matters: a first
//! packet the decoder *accepts* seeds reference planes and frame
//! state that subsequent hostile inter packets then exercise
//! (motion vectors against real references, golden vs previous
//! promotion, zero-byte duplicate markers mid-stream).
//!
//! Contract under test: `decode_frame` must `return` a `Result` for
//! any packet — never panic, abort, overflow, or index out of
//! bounds — and any `Ok` frame must survive `crop_for_display`.
//!
//! The setup tables are a fully valid set — either the §B VP3
//! defaults ([`SetupHeaderTables::vp3_defaults`]) or the
//! reference-encoded tables decoded from the tracked
//! `seed_setup_reference_tiny` corpus entry (a different §6.4.1
//! NBITS spelling, quant-range layout, and 80-codebook Huffman
//! forest), chosen by a config bit — so the fuzz budget lands in
//! the §7 packet chain (frame header, §7.3 coded-block flags, §7.4
//! modes, §7.5 motion vectors, §7.6 block-level qi, §7.7 tokens,
//! §7.9 reconstruction, §7.10 loop filter) rather than being burned
//! on setup-header rejection.

use libfuzzer_sys::fuzz_target;
use oxideav_theora::{
    decode_setup_header, FrameDecoder, PixelFormat, SetupHeaderTables, TheoraIdentHeader,
};
use std::sync::OnceLock;

/// Cap on packets per input: each accepted packet costs a full
/// reconstruction + loop-filter pass, so keep the per-input work
/// bounded to preserve the fuzzer's iteration rate.
const MAX_PACKETS: usize = 8;

/// The reference-encoded setup packet tracked as a corpus seed —
/// decoded once; its token codebooks route the §7.7 reads through
/// tree shapes the VP3 defaults never produce.
fn reference_setup() -> &'static SetupHeaderTables {
    static TABLES: OnceLock<SetupHeaderTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        decode_setup_header(include_bytes!(
            "../corpus/decode_headers/seed_setup_reference_tiny"
        ))
        .expect("the tracked reference setup seed decodes")
    })
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 3 {
        return;
    }
    let picw = 1 + (data[0] % 48) as u32;
    let pich = 1 + (data[1] % 48) as u32;
    let pf = match data[2] % 3 {
        0 => PixelFormat::Yuv420,
        1 => PixelFormat::Yuv422,
        _ => PixelFormat::Yuv444,
    };
    let ident = TheoraIdentHeader::for_picture(picw, pich, pf, 30, 1)
        .expect("1..=48 picture dimensions are always encodable");
    let setup = if data[2] & 0x80 != 0 {
        reference_setup().clone()
    } else {
        SetupHeaderTables::vp3_defaults()
    };
    let mut dec =
        FrameDecoder::new(ident, setup).expect("small VP3-default geometry must build");

    let mut rest = &data[3..];
    let mut packets = 0usize;
    while rest.len() >= 2 && packets < MAX_PACKETS {
        let len = u16::from_be_bytes([rest[0], rest[1]]) as usize;
        rest = &rest[2..];
        let take = len.min(rest.len());
        let (pkt, tail) = rest.split_at(take);
        rest = tail;
        if let Ok(frame) = dec.decode_frame(pkt) {
            dec.crop_for_display(&frame)
                .expect("a decoded frame must survive the §2.2 display crop");
        }
        packets += 1;
    }
});
