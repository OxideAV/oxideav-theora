//! Lookahead planning (round 457): keyframe placement from the
//! two-sided scene-cut detector, interval-keyframe deferral onto a cut
//! the window can see, the packet contract of a delayed encoder, and
//! the VBV clamp under rate control.

#[path = "common/rd.rs"]
mod rd;

use oxideav_core::{CodecId, Encoder as _, Frame, Packet};
use oxideav_theora::{
    PixelFormat, SetupHeaderTables, TheoraEncoder, TheoraIdentHeader, THEORA_CODEC_ID,
};
use rd::*;

fn ident(seq: &Sequence) -> TheoraIdentHeader {
    TheoraIdentHeader::for_picture(seq.width, seq.height, PixelFormat::Yuv420, 30, 1).unwrap()
}

fn encoder(seq: &Sequence, qi: u8, interval: u32) -> TheoraEncoder {
    TheoraEncoder::with_keyframe_interval(
        CodecId::new(THEORA_CODEC_ID),
        ident(seq),
        SetupHeaderTables::vp3_defaults(),
        qi,
        interval,
    )
    .unwrap()
}

/// The `square0` content for `a` frames, then `blobs` content: a hard
/// cut at frame `a`.
fn cut_at(a: u32, b: u32) -> Sequence {
    let head = synth_square(176, 144, a, 0);
    let tail = synth_blobs(176, 144, b);
    let mut frames = head.frames;
    frames.extend(tail.frames);
    Sequence {
        name: format!("cut{a}"),
        width: 176,
        height: 144,
        frames,
    }
}

fn drive(seq: &Sequence, mut enc: TheoraEncoder) -> (Vec<Packet>, Option<f64>) {
    for t in 0..seq.frames.len() {
        enc.send_frame(&Frame::Video(video_frame(seq, t))).unwrap();
    }
    enc.flush().unwrap();
    let vbv = enc.vbv_min_level_bits();
    let mut pkts = Vec::new();
    while let Ok(p) = enc.receive_packet() {
        pkts.push(p);
    }
    (pkts, vbv)
}

fn keyframe_indices(pkts: &[Packet]) -> Vec<usize> {
    pkts.iter()
        .filter(|p| !p.flags.header)
        .enumerate()
        .filter(|(_, p)| p.flags.keyframe)
        .map(|(i, _)| i)
        .collect()
}

#[test]
fn lookahead_zero_is_the_code_on_arrival_encoder() {
    let seq = synth_square(176, 144, 12, 0);
    let (a, _) = drive(&seq, encoder(&seq, 32, 8));
    let (b, _) = drive(&seq, encoder(&seq, 32, 8).with_lookahead(0));
    let bytes = |p: &[Packet]| -> Vec<Vec<u8>> { p.iter().map(|p| p.data.clone()).collect() };
    assert_eq!(bytes(&a), bytes(&b));
}

#[test]
fn lookahead_delays_packets_and_flush_drains_them_in_order() {
    let seq = synth_blobs(176, 144, 10);
    let mut enc = encoder(&seq, 32, 8).with_lookahead(4);
    for t in 0..seq.frames.len() {
        enc.send_frame(&Frame::Video(video_frame(&seq, t))).unwrap();
        let data_out = (0..)
            .map_while(|_| enc.receive_packet().ok())
            .filter(|p| !p.flags.header)
            .count();
        // After frame t has been sent, at most t + 1 - 4 data packets
        // can have been emitted (the window holds four).
        assert!(
            data_out <= (t + 1).saturating_sub(4),
            "frame {t}: {data_out} data packets out with a 4-frame window"
        );
    }
    enc.flush().unwrap();
    let rest: Vec<Packet> = (0..).map_while(|_| enc.receive_packet().ok()).collect();
    assert_eq!(rest.len(), 4, "flush drains exactly the window");
    let pts: Vec<i64> = rest.iter().map(|p| p.pts.unwrap()).collect();
    assert_eq!(pts, vec![6, 7, 8, 9], "packets come out in source order");
}

#[test]
fn lookahead_places_the_keyframe_on_the_cut_and_defers_the_interval_one() {
    // Cut at frame 18 with interval 16: on arrival the encoder codes
    // an interval keyframe at 16 and, with the scene-cut detector, a
    // second one two frames later at 18. The window sees the cut
    // coming, holds the frame-16 keyframe, and lands it on 18.
    let seq = cut_at(18, 8);
    let (plain, _) = drive(&seq, encoder(&seq, 32, 16).with_scene_cut_threshold(24.0));
    assert_eq!(keyframe_indices(&plain), vec![0, 16, 18]);
    let (look, _) = drive(
        &seq,
        encoder(&seq, 32, 16)
            .with_scene_cut_threshold(24.0)
            .with_lookahead(8),
    );
    assert_eq!(keyframe_indices(&look), vec![0, 18]);
    let size = |p: &[Packet]| -> usize {
        p.iter()
            .filter(|p| !p.flags.header)
            .map(|p| p.data.len())
            .sum()
    };
    assert!(
        size(&look) < size(&plain),
        "one keyframe fewer must cost fewer bytes ({} vs {})",
        size(&look),
        size(&plain)
    );
    // Both streams decode to the same fidelity class: measure them.
    let a = measure(
        &seq,
        encoder(&seq, 32, 16).with_scene_cut_threshold(24.0),
        "plain",
        None,
    );
    let b = measure(
        &seq,
        encoder(&seq, 32, 16)
            .with_scene_cut_threshold(24.0)
            .with_lookahead(8),
        "lookahead",
        None,
    );
    println!(
        "cut18: plain {} B / {:.2} dB, lookahead {} B / {:.2} dB",
        a.bytes, a.psnr_y, b.bytes, b.psnr_y
    );
    assert!(b.bytes < a.bytes && b.psnr_y > a.psnr_y - 0.3);
}

#[test]
fn lookahead_detects_a_cut_without_the_in_loop_threshold() {
    // No scene-cut threshold set: the window's two-sided detector
    // still puts the keyframe on the cut (default threshold 24).
    let seq = cut_at(12, 12);
    let (look, _) = drive(&seq, encoder(&seq, 32, 30).with_lookahead(6));
    assert_eq!(keyframe_indices(&look), vec![0, 12]);
    // Steady motion is not a cut: the pan never triggers one.
    let pan = synth_pan(176, 144, 20);
    let (look, _) = drive(&pan, encoder(&pan, 32, 30).with_lookahead(6));
    assert_eq!(keyframe_indices(&look), vec![0]);
}

#[test]
fn lookahead_rate_control_holds_the_target_and_the_vbv_model_tracks() {
    // 48 frames at 150 kb/s → 30000 bytes. The window's budget shares
    // keep the bucket within the one-pass loop's accuracy, and a VBV
    // of one second at the target rate is never underflowed.
    for seq in [synth_square(176, 144, 48, 0), synth_pan(176, 144, 48)] {
        let enc = TheoraEncoder::with_default_setup_keyframe_interval(
            CodecId::new(THEORA_CODEC_ID),
            ident(&seq),
            32,
            16,
        )
        .unwrap()
        .with_target_bitrate(150_000)
        .with_lookahead(8)
        .with_vbv_buffer(150_000);
        let (pkts, vbv) = drive(&seq, enc);
        let bytes: usize = pkts
            .iter()
            .filter(|p| !p.flags.header)
            .map(|p| p.data.len())
            .sum();
        let target = 150_000.0 * 48.0 / 30.0 / 8.0;
        let err = (bytes as f64 / target - 1.0) * 100.0;
        let vbv = vbv.expect("a VBV model was set");
        println!(
            "{}: {bytes} B ({err:+.1} %), VBV minimum {vbv:.0} bits",
            seq.name
        );
        assert!(err.abs() <= 10.0, "{}: rate error {err:+.1} %", seq.name);
        assert!(vbv >= 0.0, "{}: VBV underflow ({vbv:.0} bits)", seq.name);
    }
}

#[test]
fn lookahead_vbv_enforcement_recodes_an_oversized_frame() {
    // A 40000-bit buffer (0.27 s at 150 kb/s) against a cut into
    // unseen content: the planner's calibrated prediction has no
    // precedent for the cut's keyframe, so the enforcement pass must
    // re-code it. The modelled level never drops below the 2 % guard
    // band.
    let seq = cut_at(24, 24);
    let build = |vbv: Option<u64>| {
        let enc = TheoraEncoder::with_default_setup_keyframe_interval(
            CodecId::new(THEORA_CODEC_ID),
            ident(&seq),
            32,
            16,
        )
        .unwrap()
        .with_target_bitrate(150_000)
        .with_lookahead(8);
        match vbv {
            Some(v) => enc.with_vbv_buffer(v),
            None => enc,
        }
    };
    let (_, without) = drive(&seq, build(None));
    assert!(without.is_none(), "no VBV model without with_vbv_buffer");
    let (pkts, with) = drive(&seq, build(Some(40_000)));
    let min = with.unwrap();
    let largest = pkts
        .iter()
        .filter(|p| !p.flags.header)
        .map(|p| p.data.len() * 8)
        .max()
        .unwrap();
    println!("cut24 @150k, VBV 40000: minimum level {min:.0} bits, largest frame {largest} bits");
    assert!(
        min >= -0.02 * 40_000.0,
        "VBV level {min:.0} below the guard band"
    );
    assert!(
        largest <= 40_000,
        "a {largest}-bit frame cannot pass a 40000-bit buffer"
    );
}
