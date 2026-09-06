//! Rate-distortion ladder for the Theora encoder.
//!
//! Encodes a battery of deterministic sequences across a quantizer
//! ladder (and, optionally, target-bitrate points), decodes every
//! stream back through this crate's own decoder (which is pixel-exact
//! against the black-box reference decoder for every validated
//! scenario), and prints bytes / luma PSNR / chroma PSNR / luma SSIM
//! per point plus a Bjøntegaard-style summary against a reference run.
//! The sequences, the measurement loop, and the BD arithmetic live in
//! `tests/common/rd.rs`, shared with the pinned `tests/bd_rate.rs`
//! battery.
//!
//! ```text
//! cargo run --release --example rd_ladder -- [options]
//!   --fixtures <dir>   docs/video/theora/fixtures — adds the fixture-derived
//!                      sequences (their expected.yuv as source)
//!   --out <dir>        dump each stream's length-prefixed packet chain
//!                      (u32 LE length + bytes) for external validation
//!   --ref <file>       a previous run's `--save` file: prints BD deltas
//!   --save <file>      save this run's points for a later `--ref`
//!   --profile <name>   encoder profile: see `profiles()` in the shared module
//!   --seq <name>       restrict to one sequence
//!   --interval <n>     keyframe interval (default 16)
//!   --qis a,b,c        the qi ladder (default 8,20,32,44,56)
//!   --bitrates a,b     target-bitrate points (bits per second)
//!   --twopass          drive the bitrate points through two-pass control
//!   --lfscale n/d      rescale the §6.4.1 loop-filter limit table
//! ```

#[path = "../tests/common/rd.rs"]
mod rd;

use oxideav_core::frame::VideoPlane;
use oxideav_core::CodecId;
use oxideav_theora::{PixelFormat, SourceFrame, TheoraEncoder, TheoraIdentHeader, THEORA_CODEC_ID};
use rd::*;
use std::collections::BTreeMap;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut fixtures: Option<std::path::PathBuf> = None;
    let mut out: Option<std::path::PathBuf> = None;
    let mut reference: Option<std::path::PathBuf> = None;
    let mut save: Option<std::path::PathBuf> = None;
    let mut profile = "default".to_string();
    let mut only_seq: Option<String> = None;
    let mut interval = 16u32;
    let mut twopass = false;
    let mut lfscale: Option<(u32, u32)> = None;
    let mut qis: Vec<u8> = vec![8, 20, 32, 44, 56];
    let mut bitrates: Vec<u64> = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = args[i].as_str();
        let val = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i).cloned().unwrap_or_default()
        };
        match a {
            "--fixtures" => fixtures = Some(val(&mut i).into()),
            "--out" => out = Some(val(&mut i).into()),
            "--ref" => reference = Some(val(&mut i).into()),
            "--save" => save = Some(val(&mut i).into()),
            "--profile" => profile = val(&mut i),
            "--seq" => only_seq = Some(val(&mut i)),
            "--interval" => interval = val(&mut i).parse().unwrap(),
            "--twopass" => twopass = true,
            "--lfscale" => {
                let v = val(&mut i);
                let (n, d) = v.split_once('/').unwrap_or((v.as_str(), "1"));
                lfscale = Some((n.parse().unwrap(), d.parse().unwrap()));
            }
            "--qis" => qis = val(&mut i).split(',').map(|s| s.parse().unwrap()).collect(),
            "--bitrates" => bitrates = val(&mut i).split(',').map(|s| s.parse().unwrap()).collect(),
            _ => panic!("unknown option {a}"),
        }
        i += 1;
    }
    if let Some(d) = &out {
        std::fs::create_dir_all(d).unwrap();
    }
    let prof: Profile = profiles()
        .into_iter()
        .find(|(n, _)| *n == profile)
        .map(|(_, p)| p)
        .unwrap_or_else(|| panic!("unknown profile {profile}"));

    let mut seqs = vec![
        synth_square(176, 144, 24, 0),
        synth_blobs(176, 144, 24),
        synth_pan(176, 144, 24),
        synth_cut(176, 144, 24),
    ];
    if let Some(dir) = &fixtures {
        if let Some(s) = fixture_sequence(dir, "all-mb-modes-64x64", 64, 64) {
            seqs.push(s);
        }
        if let Some(s) = fixture_sequence(dir, "keyframe-interval-30", 32, 32) {
            seqs.push(s);
        }
    }
    if let Some(name) = &only_seq {
        seqs.retain(|s| &s.name == name);
    }

    let cid = || CodecId::new(THEORA_CODEC_ID);
    let mut points: Vec<Point> = Vec::new();
    println!(
        "{:<22} {:<10} {:>8} {:>8} {:>8} {:>7} {:>4}",
        "sequence", "point", "bytes", "Y-PSNR", "C-PSNR", "Y-SSIM", "kf"
    );
    for seq in &seqs {
        let ident =
            TheoraIdentHeader::for_picture(seq.width, seq.height, PixelFormat::Yuv420, 30, 1)
                .unwrap();
        let mut setup = oxideav_theora::SetupHeaderTables::vp3_defaults();
        if let Some((n, d)) = lfscale {
            for v in setup.loop_filter_limits.iter_mut() {
                *v = ((*v as u32 * n / d).min(127)) as u8;
            }
        }
        for &qi in &qis {
            let enc = TheoraEncoder::with_keyframe_interval(
                cid(),
                ident.clone(),
                setup.clone(),
                qi,
                interval,
            )
            .unwrap();
            let enc = prof(enc);
            let p = measure(seq, enc, &format!("qi{qi}"), out.as_deref());
            println!(
                "{:<22} {:<10} {:>8} {:>8.2} {:>8.2} {:>7.4} {:>4}",
                p.seq, p.label, p.bytes, p.psnr_y, p.psnr_c, p.ssim_y, p.keyframes
            );
            points.push(p);
        }
        for &br in &bitrates {
            let enc = TheoraEncoder::with_default_setup_keyframe_interval(
                cid(),
                ident.clone(),
                32,
                interval,
            )
            .unwrap();
            let enc = if twopass {
                // First pass over the exact frames (flipped to
                // lower-left SourceFrames at the picture shape).
                let sources: Vec<SourceFrame> = (0..seq.frames.len())
                    .map(|t| {
                        let vf = video_frame(seq, t);
                        let (py, pc) = ident.picture_plane_dims();
                        let flip = |p: &VideoPlane, w: u32, h: u32| -> Vec<u8> {
                            let mut out = Vec::with_capacity((w * h) as usize);
                            for row in (0..h as usize).rev() {
                                out.extend_from_slice(
                                    &p.data[row * p.stride..row * p.stride + w as usize],
                                );
                            }
                            out
                        };
                        SourceFrame::from_picture(
                            &ident,
                            &flip(&vf.planes[0], py.width, py.height),
                            &flip(&vf.planes[1], pc.width, pc.height),
                            &flip(&vf.planes[2], pc.width, pc.height),
                        )
                        .unwrap()
                    })
                    .collect();
                let stats = TheoraEncoder::two_pass_stats(
                    &ident,
                    &oxideav_theora::SetupHeaderTables::vp3_defaults(),
                    32,
                    interval,
                    &sources,
                )
                .unwrap();
                enc.with_two_pass_rate_control(br, &stats)
            } else {
                enc.with_target_bitrate(br)
            };
            let enc = prof(enc);
            let p = measure(seq, enc, &format!("br{}k", br / 1000), out.as_deref());
            println!(
                "{:<22} {:<10} {:>8} {:>8.2} {:>8.2} {:>7.4} {:>4}",
                p.seq, p.label, p.bytes, p.psnr_y, p.psnr_c, p.ssim_y, p.keyframes
            );
            points.push(p);
        }
    }

    if let Some(path) = &save {
        let mut s = String::new();
        for p in &points {
            s.push_str(&format!(
                "{}\t{}\t{}\t{:.4}\t{:.4}\t{:.6}\t{}\n",
                p.seq, p.label, p.bytes, p.psnr_y, p.psnr_c, p.ssim_y, p.keyframes
            ));
        }
        std::fs::write(path, s).unwrap();
    }

    if let Some(path) = &reference {
        let text = std::fs::read_to_string(path).unwrap();
        // Per-sequence reference curves: (bytes, Y-PSNR), (bytes,
        // C-PSNR), (bytes, Y-SSIM). Older save files without the SSIM
        // column (round 453) still parse; their SSIM curve is empty.
        let mut base: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
        let mut base_c: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
        let mut base_s: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
        let mut base_pts: BTreeMap<(String, String), (usize, f64)> = BTreeMap::new();
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 5 || !f[1].starts_with("qi") {
                continue;
            }
            let bytes: f64 = f[2].parse().unwrap();
            let py: f64 = f[3].parse().unwrap();
            let pc: f64 = f[4].parse().unwrap();
            base.entry(f[0].to_string()).or_default().push((bytes, py));
            base_c
                .entry(f[0].to_string())
                .or_default()
                .push((bytes, pc));
            if f.len() >= 7 {
                let ss: f64 = f[5].parse().unwrap();
                base_s
                    .entry(f[0].to_string())
                    .or_default()
                    .push((bytes, ss));
            }
            base_pts.insert((f[0].to_string(), f[1].to_string()), (bytes as usize, py));
        }
        println!();
        println!("per-point vs reference (same qi): bytes Δ%, Y-PSNR Δ dB");
        for p in &points {
            if let Some((bb, bp)) = base_pts.get(&(p.seq.clone(), p.label.clone())) {
                println!(
                    "  {:<22} {:<8} {:>+7.2}%  {:>+6.2} dB",
                    p.seq,
                    p.label,
                    (p.bytes as f64 / *bb as f64 - 1.0) * 100.0,
                    p.psnr_y - bp
                );
            }
        }
        println!();
        println!("BD deltas vs reference (qi ladder points): luma | chroma | luma SSIM-rate");
        let (mut sum_p, mut sum_r, mut sum_rc, mut sum_rs) = (0.0, 0.0, 0.0, 0.0);
        let mut n = 0;
        for (seq, bcurve) in &base {
            let curve = |f: &dyn Fn(&Point) -> f64| -> Vec<(f64, f64)> {
                points
                    .iter()
                    .filter(|p| &p.seq == seq && p.label.starts_with("qi"))
                    .map(|p| (p.bytes as f64, f(p)))
                    .collect()
            };
            let ty = curve(&|p| p.psnr_y);
            if ty.len() < 2 {
                continue;
            }
            let Some((dp, dr)) = bd_deltas(bcurve, &ty) else {
                continue;
            };
            let drc = base_c
                .get(seq)
                .and_then(|bc| bd_deltas(bc, &curve(&|p| p.psnr_c)))
                .map_or(f64::NAN, |(_, r)| r);
            let drs = base_s
                .get(seq)
                .and_then(|bs| bd_deltas(bs, &curve(&|p| p.ssim_y)))
                .map_or(f64::NAN, |(_, r)| r);
            println!(
                "  {seq:<22} BD-PSNR {dp:>+6.3} dB   BD-rate {dr:>+7.2} % | C {drc:>+7.2} % | SSIM {drs:>+7.2} %"
            );
            sum_p += dp;
            sum_r += dr;
            sum_rc += drc;
            sum_rs += drs;
            n += 1;
        }
        if n > 0 {
            let nf = n as f64;
            println!(
                "  {:<22} BD-PSNR {:>+6.3} dB   BD-rate {:>+7.2} % | C {:>+7.2} % | SSIM {:>+7.2} %",
                "MEAN",
                sum_p / nf,
                sum_r / nf,
                sum_rc / nf,
                sum_rs / nf
            );
        }
    }
}
