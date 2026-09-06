//! Pinned Bjøntegaard rate-distortion battery (round 457).
//!
//! Four deterministic 176×144 4:2:0 scenes (`tests/common/rd.rs`:
//! `square0` gradient + moving square, `blobs` sub-pixel drifting blobs
//! over fixed-pattern noise, `pan` a half-pixel-per-frame textured pan,
//! `cut` a mid-stream scene change), 24 frames each, encoded with the
//! `TheoraEncoder` defaults at keyframe interval 16 across the
//! five-point quantizer ladder, decoded back through this crate's own
//! decoder (pixel-exact against the black-box reference decoder for
//! every externally validated family), and scored on bytes, luma PSNR,
//! chroma PSNR and luma SSIM.
//!
//! Two pins:
//!
//! * **chroma never regresses** against the round-453 encoder's
//!   curves (`BASELINE_R453`, the operating points the README's
//!   round-453 table records) beyond a 1 % tolerance on any scene;
//! * **campaign floor** — the per-scene luma BD-rate the round-457
//!   campaign measured against that reference must still be delivered
//!   (`CAMPAIGN_FLOOR_PCT`; a positive entry records a scene the
//!   campaign knowingly traded), so a later change that quietly gives
//!   a gain back — or widens a known trade — fails here.
//!
//! Run with `--nocapture` to see the full table and the per-scene BD
//! deltas; the same numbers come out of
//! `cargo run --release --example rd_ladder -- --ref <round-453 save>`.

#[path = "common/rd.rs"]
mod rd;

use oxideav_core::CodecId;
use oxideav_theora::{
    PixelFormat, SetupHeaderTables, TheoraEncoder, TheoraIdentHeader, THEORA_CODEC_ID,
};
use rd::*;

/// Round-453 reference operating points: (scene, qi, bytes, luma PSNR,
/// chroma PSNR, luma SSIM) at keyframe interval 16, 24 frames.
const BASELINE_R453: &[(&str, u8, usize, f64, f64, f64)] = &[
    ("square0", 8, 10358, 29.2327, 39.5638, 0.967044),
    ("square0", 20, 13168, 36.6328, 46.0696, 0.987425),
    ("square0", 32, 15044, 41.1073, 50.4465, 0.992932),
    ("square0", 44, 18046, 44.4494, 52.2045, 0.996666),
    ("square0", 56, 21822, 48.4649, 52.4988, 0.998442),
    ("blobs", 8, 2562, 34.7694, 34.1391, 0.879850),
    ("blobs", 20, 4675, 37.3300, 39.3944, 0.909540),
    ("blobs", 32, 7801, 39.0292, 44.1930, 0.928450),
    ("blobs", 44, 11938, 39.9416, 47.6857, 0.938459),
    ("blobs", 56, 18799, 40.9354, 49.0737, 0.951477),
    ("pan", 8, 9298, 30.6334, 30.3938, 0.902820),
    ("pan", 20, 17068, 33.6463, 32.9306, 0.945296),
    ("pan", 32, 29852, 35.5550, 35.3951, 0.963442),
    ("pan", 44, 64926, 36.7676, 37.8171, 0.971997),
    ("pan", 56, 139046, 39.0672, 40.2429, 0.983772),
    ("cut", 8, 6378, 31.4139, 37.2829, 0.923772),
    ("cut", 20, 8992, 37.2365, 42.1959, 0.949520),
    ("cut", 32, 11653, 40.0183, 46.7162, 0.960657),
    ("cut", 44, 15873, 41.6668, 49.9867, 0.967392),
    ("cut", 56, 22266, 43.2148, 50.6448, 0.974459),
];

const QI_LADDER: [u8; 5] = [8, 20, 32, 44, 56];

/// Per-scene luma BD-rate (percent, negative = fewer bytes at equal
/// PSNR) the round-457 campaign must keep delivering against the
/// round-453 reference. Set from the measured campaign result; tighten
/// when the encoder improves, never loosen without a README note.
const CAMPAIGN_FLOOR_PCT: &[(&str, f64)] = &[
    ("square0", -2.9),
    ("blobs", 1.2),
    ("pan", -7.4),
    ("cut", -1.4),
];

/// Slack on the campaign-floor comparison, in BD-rate percent: the
/// reference table carries 4-decimal PSNRs, so an unchanged encoder
/// lands within ±0.005 % of the pinned curve, not at exactly zero.
const EPS: f64 = 0.01;

fn scenes() -> Vec<Sequence> {
    vec![
        synth_square(176, 144, 24, 0),
        synth_blobs(176, 144, 24),
        synth_pan(176, 144, 24),
        synth_cut(176, 144, 24),
    ]
}

fn ladder(seq: &Sequence) -> Vec<Point> {
    let ident =
        TheoraIdentHeader::for_picture(seq.width, seq.height, PixelFormat::Yuv420, 30, 1).unwrap();
    let setup = SetupHeaderTables::vp3_defaults();
    QI_LADDER
        .iter()
        .map(|&qi| {
            let enc = TheoraEncoder::with_keyframe_interval(
                CodecId::new(THEORA_CODEC_ID),
                ident.clone(),
                setup.clone(),
                qi,
                16,
            )
            .unwrap();
            measure(seq, enc, &format!("qi{qi}"), None)
        })
        .collect()
}

fn scene(name: &str) -> Sequence {
    scenes()
        .into_iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("no scene {name}"))
}

/// Encode one scene's ladder, print its table, and hold both pins.
fn check_scene(name: &str) {
    let seq = scene(name);
    println!(
        "{:<10} {:<6} {:>8} {:>8} {:>8} {:>7} {:>3}",
        "scene", "point", "bytes", "Y-PSNR", "C-PSNR", "Y-SSIM", "kf"
    );
    let pts = ladder(&seq);
    for p in &pts {
        println!(
            "{:<10} {:<6} {:>8} {:>8.2} {:>8.2} {:>7.4} {:>3}",
            p.seq, p.label, p.bytes, p.psnr_y, p.psnr_c, p.ssim_y, p.keyframes
        );
    }
    let base: Vec<_> = BASELINE_R453.iter().filter(|b| b.0 == seq.name).collect();
    assert_eq!(
        base.len(),
        QI_LADDER.len(),
        "{}: reference ladder",
        seq.name
    );
    let by: Vec<(f64, f64)> = base.iter().map(|b| (b.2 as f64, b.3)).collect();
    let bc: Vec<(f64, f64)> = base.iter().map(|b| (b.2 as f64, b.4)).collect();
    let bs: Vec<(f64, f64)> = base.iter().map(|b| (b.2 as f64, b.5)).collect();
    let ty: Vec<(f64, f64)> = pts.iter().map(|p| (p.bytes as f64, p.psnr_y)).collect();
    let tc: Vec<(f64, f64)> = pts.iter().map(|p| (p.bytes as f64, p.psnr_c)).collect();
    let ts: Vec<(f64, f64)> = pts.iter().map(|p| (p.bytes as f64, p.ssim_y)).collect();
    let (dpy, dry) = bd_deltas(&by, &ty).expect("luma curves overlap");
    let (_, drc) = bd_deltas(&bc, &tc).expect("chroma curves overlap");
    let drs = bd_deltas(&bs, &ts).map_or(f64::NAN, |(_, r)| r);
    println!(
        "  {:<10} vs round 453: BD-PSNR {dpy:+.3} dB  BD-rate Y {dry:+.2} %  C {drc:+.2} %  SSIM {drs:+.2} %",
        seq.name
    );
    assert!(
        drc <= 1.0,
        "{}: chroma BD-rate regressed {drc:+.2} % against the round-453 reference",
        seq.name
    );
    let floor = CAMPAIGN_FLOOR_PCT
        .iter()
        .find(|f| f.0 == seq.name)
        .map(|f| f.1)
        .expect("campaign floor for every scene");
    assert!(
        dry <= floor + EPS,
        "{}: luma BD-rate {dry:+.2} % misses the campaign floor {floor:+.2} %",
        seq.name
    );
}

#[test]
fn bd_rate_square0() {
    check_scene("square0");
}

#[test]
fn bd_rate_blobs() {
    check_scene("blobs");
}

#[test]
fn bd_rate_pan() {
    check_scene("pan");
}

#[test]
fn bd_rate_cut() {
    check_scene("cut");
}
