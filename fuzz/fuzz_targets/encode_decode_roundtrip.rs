#![no_main]

//! Encoder→decoder contract on fuzz-derived streams.
//!
//! Fuzz bytes choose a picture geometry (1..=48 per axis,
//! non-macro-block-aligned included), pixel format, quantizer,
//! keyframe interval, inter-mode strategy, optional adaptive
//! quantization / target-bitrate rate control, and every plane's
//! content. `TheoraEncoder` (zero-setup VP3-defaults entry point)
//! encodes the frames through the `oxideav_core::Encoder` trait;
//! the emitted packet chain is then decoded back through the
//! `make_decoder` factory fed the encoder's own advertised
//! `output_params` (the length-prefixed extradata header chain).
//!
//! Contract asserted — the same one the externally validated corpus
//! pins, here on arbitrary content:
//!
//! * every emitted packet classifies per §6.1 (headers flagged
//!   `header`, the rest video data);
//! * the decoder accepts **every** data packet (an encoder that
//!   emits a packet its own decoder rejects has broken the stream
//!   contract);
//! * exactly one frame comes back per data packet, at the §2.2
//!   picture dimensions, in the advertised pixel format's plane
//!   shape.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, Encoder, Frame, Packet, TimeBase};
use oxideav_theora::{
    classify_packet, make_decoder, InterModeStrategy, PixelFormat, TheoraEncoder,
    TheoraIdentHeader, TheoraPacketKind, THEORA_CODEC_ID,
};

/// Frame-count cap: each frame is a full RD encode + mirror decode.
const MAX_FRAMES: usize = 4;

fuzz_target!(|data: &[u8]| {
    if data.len() < 8 {
        return;
    }
    let picw = 1 + (data[0] % 48) as u32;
    let pich = 1 + (data[1] % 48) as u32;
    let pf = match data[2] % 3 {
        0 => PixelFormat::Yuv420,
        1 => PixelFormat::Yuv422,
        _ => PixelFormat::Yuv444,
    };
    let qi = data[3] % 64;
    let interval = 1 + (data[4] % 4) as u32;
    let nframes = 1 + (data[5] % MAX_FRAMES as u8) as usize;
    let opts = data[6];
    let content = &data[7..];

    let ident = TheoraIdentHeader::for_picture(picw, pich, pf, 30, 1)
        .expect("1..=48 picture dimensions are always encodable");
    let (dims_y, dims_c) = ident.picture_plane_dims();

    let codec_id = CodecId::new(THEORA_CODEC_ID);
    let mut enc = TheoraEncoder::with_default_setup_keyframe_interval(
        codec_id.clone(),
        ident,
        qi,
        interval,
    )
    .expect("VP3-default encoder must build at any qi");
    enc = enc.with_inter_mode(match (opts >> 1) & 3 {
        0 => InterModeStrategy::RateDistortion,
        1 => InterModeStrategy::PreviousMotion,
        2 => InterModeStrategy::GoldenMotion,
        _ => InterModeStrategy::FourMv,
    });
    if opts & 1 != 0 {
        enc = enc.with_adaptive_quant(vec![qi, (qi + 17) % 64, (qi + 39) % 64]);
    }
    if opts & 0x10 != 0 {
        enc = enc.with_target_bitrate(200_000);
    }

    // Plane fill: cycle the fuzz content bytes with a per-frame,
    // per-plane offset so successive frames differ (exercising the
    // inter paths) in a fuzz-controlled way.
    let fill = |len: usize, salt: usize| -> Vec<u8> {
        if content.is_empty() {
            return vec![128; len];
        }
        (0..len)
            .map(|i| content[(i.wrapping_mul(31).wrapping_add(salt)) % content.len()])
            .collect()
    };
    let mk_frame = |fi: usize| -> Frame {
        use oxideav_core::frame::VideoPlane;
        let plane = |w: u32, h: u32, salt: usize| VideoPlane {
            stride: w as usize,
            data: fill((w * h) as usize, salt),
        };
        Frame::Video(oxideav_core::VideoFrame {
            pts: Some(fi as i64),
            planes: vec![
                plane(dims_y.width, dims_y.height, fi * 3),
                plane(dims_c.width, dims_c.height, fi * 3 + 1),
                plane(dims_c.width, dims_c.height, fi * 3 + 2),
            ],
        })
    };

    let mut packets: Vec<Packet> = Vec::new();
    for fi in 0..nframes {
        enc.send_frame(&mk_frame(fi)).expect("picture-shaped frame must encode");
        while let Ok(pkt) = enc.receive_packet() {
            packets.push(pkt);
        }
    }

    // Every packet must classify; header flags must agree with §6.1.
    let mut n_data = 0usize;
    for pkt in &packets {
        let kind = classify_packet(&pkt.data).expect("encoder output must classify");
        match kind {
            TheoraPacketKind::VideoData => {
                assert!(!pkt.flags.header, "data packet flagged as header");
                n_data += 1;
            }
            _ => assert!(pkt.flags.header, "header packet not flagged"),
        }
    }
    assert_eq!(n_data, nframes, "one data packet per source frame");

    // Decode back through the factory using the encoder's own
    // advertised parameters.
    let mut dec = make_decoder(enc.output_params())
        .expect("encoder-advertised extradata must build a decoder");
    let tb = TimeBase::from_rate(30);
    let mut frames_out = 0usize;
    for pkt in packets.iter() {
        if pkt.flags.header {
            continue;
        }
        dec.send_packet(&Packet::new(0, tb, pkt.data.clone()))
            .expect("decoder must accept encoder packets");
        while let Ok(frame) = dec.receive_frame() {
            let Frame::Video(vf) = frame else {
                panic!("theora decoder must emit video frames");
            };
            assert_eq!(vf.planes.len(), 3, "planar YUV output");
            assert_eq!(
                vf.planes[0].data.len(),
                (dims_y.width * dims_y.height) as usize,
                "luma plane at §2.2 picture dimensions"
            );
            assert_eq!(
                vf.planes[1].data.len(),
                (dims_c.width * dims_c.height) as usize,
                "chroma plane at §4.4.4 picture dimensions"
            );
            frames_out += 1;
        }
    }
    assert_eq!(frames_out, nframes, "one decoded frame per data packet");
});
