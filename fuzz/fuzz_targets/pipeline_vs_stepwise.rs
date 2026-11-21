#![no_main]

//! Differential target: the decoder's throughput pipeline against the
//! step-by-step §7 procedure chain.
//!
//! `FrameDecoder::decode_frame` runs a reorganised pipeline (prefix
//! lookup tables for the §7.7 tokens, a shrinking open-block list, a
//! lane inverse DCT, band-parallel reconstruction and loop filtering);
//! `FrameDecoder::decode_frame_stepwise` chains the specification's
//! numbered procedures with every intermediate array materialised.
//! Both read the same hostile packets here — framed exactly as in
//! `decode_frame_chain`, so its seeds apply — through three decoders:
//! the step-by-step one, the pipeline at the serial budget, and the
//! pipeline at a fuzz-chosen budget of 2..=5 workers.
//!
//! Contract under test: for every packet all three agree on
//! acceptance, and on acceptance they produce the same frame and
//! leave the same golden / previous reference planes. Picture
//! dimensions reach 96 on each axis so the luma plane spans up to
//! three super-block rows — enough for interior band boundaries, the
//! place where the banded loop filter has to reproduce the serial
//! order.

use libfuzzer_sys::fuzz_target;
use oxideav_theora::{
    FrameDecoder, PixelFormat, SetupHeaderTables, TheoraIdentHeader, decode_setup_header,
};
use std::sync::OnceLock;

/// Cap on packets per input (each accepted packet is three full
/// frame decodes).
const MAX_PACKETS: usize = 8;

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
    let wide = if data[2] & 0x40 != 0 { 48 } else { 0 };
    let picw = 1 + (data[0] % 48) as u32 + wide;
    let pich = 1 + (data[1] % 48) as u32 + wide;
    let pf = match data[2] % 3 {
        0 => PixelFormat::Yuv420,
        1 => PixelFormat::Yuv422,
        _ => PixelFormat::Yuv444,
    };
    let workers = 2 + ((data[2] >> 3) & 3) as usize;
    let ident = TheoraIdentHeader::for_picture(picw, pich, pf, 30, 1)
        .expect("1..=96 picture dimensions are always encodable");
    let setup = if data[2] & 0x80 != 0 {
        reference_setup().clone()
    } else {
        SetupHeaderTables::vp3_defaults()
    };
    let mut stepwise =
        FrameDecoder::new(ident.clone(), setup.clone()).expect("small geometry must build");
    let mut serial =
        FrameDecoder::new(ident.clone(), setup.clone()).expect("small geometry must build");
    let mut banded = FrameDecoder::new(ident, setup).expect("small geometry must build");
    banded.set_threads(workers);

    let mut rest = &data[3..];
    let mut packets = 0usize;
    while rest.len() >= 2 && packets < MAX_PACKETS {
        let len = u16::from_be_bytes([rest[0], rest[1]]) as usize;
        rest = &rest[2..];
        let take = len.min(rest.len());
        let (pkt, tail) = rest.split_at(take);
        rest = tail;
        let want = stepwise.decode_frame_stepwise(pkt);
        for (name, dec) in [("serial", &mut serial), ("banded", &mut banded)] {
            let got = dec.decode_frame(pkt);
            match (&want, &got) {
                (Ok(want), Ok(got)) => {
                    assert_eq!(got, want, "{name}: frame differs from the step-by-step chain");
                }
                (Err(_), Err(_)) => {}
                (want, got) => panic!(
                    "{name}: acceptance differs — step-by-step {:?}, pipeline {:?}",
                    want.as_ref().err(),
                    got.as_ref().err()
                ),
            }
            assert_eq!(
                dec.reference_store(),
                stepwise.reference_store(),
                "{name}: reference planes differ from the step-by-step chain"
            );
        }
        packets += 1;
    }
});
