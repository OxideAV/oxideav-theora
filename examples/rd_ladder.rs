//! Rate-distortion ladder for the Theora encoder.
//!
//! Encodes a battery of deterministic sequences across a quantizer
//! ladder (and, optionally, target-bitrate points), decodes every
//! stream back through this crate's own decoder (which is pixel-exact
//! against the black-box reference decoder for every validated
//! scenario), and prints bytes / luma PSNR / chroma PSNR per point plus
//! a Bjøntegaard-style summary against a reference run.
//!
//! ```text
//! cargo run --release --example rd_ladder -- [options]
//!   --fixtures <dir>   docs/video/theora/fixtures — adds the fixture-derived
//!                      sequences (their expected.yuv as source)
//!   --out <dir>        dump each stream's length-prefixed packet chain
//!                      (u32 LE length + bytes) for external validation
//!   --ref <file>       a previous run's `--save` file: prints BD deltas
//!   --save <file>      save this run's points for a later `--ref`
//!   --profile <name>   encoder profile: see `profiles()` below
//!   --seq <name>       restrict to one sequence
//!   --interval <n>     keyframe interval (default 16)
//!   --qis a,b,c        the qi ladder (default 8,20,32,44,56)
//! ```
//!
//! Everything here is deterministic (integer content generators, no
//! randomness), so two runs on the same code print the same table.

use oxideav_core::frame::VideoPlane;
use oxideav_core::{CodecId, Decoder as _, Encoder as _, Frame, Packet, VideoFrame};
use oxideav_theora::{
    PixelFormat, SourceFrame, TheoraDecoder, TheoraEncoder, TheoraIdentHeader, THEORA_CODEC_ID,
};
use std::collections::BTreeMap;

// ----------------------------------------------------------------------
// Sequences
// ----------------------------------------------------------------------

struct Sequence {
    name: String,
    width: u32,
    height: u32,
    /// Top-down planar 4:2:0 frames (Y, Cb, Cr).
    frames: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)>,
}

/// Deterministic pseudo-random generator (LCG) for reproducible noise.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// Fixed-point sine over a 4096-entry period (integer, deterministic).
fn isin(phase: i64) -> i64 {
    // 12-bit phase, output in [-4096, 4096] (ties to a float table are
    // avoided; polynomial approximation on the quarter wave).
    let p = phase.rem_euclid(4096);
    let (q, sign) = if p < 2048 { (p, 1) } else { (p - 2048, -1) };
    let x = if q < 1024 { q } else { 2048 - q }; // 0..=1024
                                                 // sin(x·π/2048) ≈ x·(3π/2 − ... ) — use Bhaskara-style rational.
                                                 // 16·x·(π−x)/(5π²−4·x·(π−x)) with x in [0, π] mapped from 0..2048.
    let xx = x * 2; // 0..=2048 → maps to 0..=π over 2048
    let t = xx * (2048 - xx); // ≤ 1024²
    let num = 16 * t;
    let den = 5 * 2048 * 2048 - 4 * t;
    sign * (num * 4096 / den)
}

fn synth_square(w: u32, h: u32, n: u32, family: u32) -> Sequence {
    // The pinned-corpus generator (gradient + sweeping bright square).
    let (cw, ch) = (w / 2, h / 2);
    let mut frames = Vec::new();
    for t in 0..n {
        let (wu, hu, cwu, chu, tu, fam) = (
            w as usize,
            h as usize,
            cw as usize,
            ch as usize,
            t as usize,
            family as usize,
        );
        let mut y = vec![0u8; wu * hu];
        for row in 0..hu {
            for col in 0..wu {
                let base = match fam {
                    0 => (col * 2 + row * 3 + tu * 5) % 256,
                    _ => 255 - ((col * 3).wrapping_add(row * 7).wrapping_add(tu * 11) % 256),
                };
                let sq_x = (tu * 7 + fam * 40) % (wu.max(33) - 32);
                let sq_y = (tu * 4) % (hu.max(33) - 32);
                let v = if col >= sq_x && col < sq_x + 32 && row >= sq_y && row < sq_y + 32 {
                    220
                } else {
                    base
                };
                y[row * wu + col] = v as u8;
            }
        }
        let mut cb = vec![0u8; cwu * chu];
        let mut cr = vec![0u8; cwu * chu];
        for row in 0..chu {
            for col in 0..cwu {
                cb[row * cwu + col] = ((col * 200 / cwu.max(1)) + tu * 2 + fam * 30) as u8;
                cr[row * cwu + col] = ((row * 200 / chu.max(1)) + tu * 3) as u8;
            }
        }
        frames.push((y, cb, cr));
    }
    Sequence {
        name: format!("square{family}"),
        width: w,
        height: h,
        frames,
    }
}

/// Smooth drifting blobs with sub-pixel motion plus mild noise —
/// natural-ish content where half-pixel prediction and coefficient
/// rounding decisions matter.
fn synth_blobs(w: u32, h: u32, n: u32) -> Sequence {
    let (cw, ch) = (w / 2, h / 2);
    let mut frames = Vec::new();
    let mut rng = Lcg(0x5eed_1234_abcd);
    // Static noise texture (frame-invariant, so it moves with content
    // only where the content moves — here it stays put, like sensor
    // fixed-pattern texture).
    let tex: Vec<i32> = (0..(w * h)).map(|_| (rng.next() % 9) as i32 - 4).collect();
    for t in 0..n {
        let tt = t as i64;
        let mut y = vec![0u8; (w * h) as usize];
        for row in 0..h as i64 {
            for col in 0..w as i64 {
                // Three blobs drifting at fractional speeds (units of
                // 1/8 pixel per frame).
                let mut v: i64 = 96 * 4096;
                let blobs = [
                    (40 * 8 + tt * 12, 40 * 8 + tt * 4, 28, 90),
                    (120 * 8 - tt * 6, 60 * 8 + tt * 10, 20, -60),
                    (80 * 8 + tt * 3, 100 * 8 - tt * 7, 36, 70),
                ];
                for (bx, by, r, amp) in blobs {
                    let dx = col * 8 - bx;
                    let dy = row * 8 - by;
                    let d2 = (dx * dx + dy * dy) / 64; // pixel² units
                    let r2 = r * r;
                    if d2 < r2 * 4 {
                        // Smooth bump: amp·(1 − d²/(4r²))² in fixed point.
                        let f = 4096 - d2 * 4096 / (r2 * 4);
                        v += amp * f * f / 4096;
                    }
                }
                // Gentle background waves (static).
                v += 10 * isin(col * 64 + row * 32) + 6 * isin(row * 96);
                let mut s = v / 4096 + tex[(row * w as i64 + col) as usize] as i64;
                s = s.clamp(0, 255);
                y[(row * w as i64 + col) as usize] = s as u8;
            }
        }
        let mut cb = vec![0u8; (cw * ch) as usize];
        let mut cr = vec![0u8; (cw * ch) as usize];
        for row in 0..ch as i64 {
            for col in 0..cw as i64 {
                let a = 128 + (isin(col * 48 + tt * 40) * 30) / 4096;
                let b = 128 + (isin(row * 40 - tt * 24) * 24) / 4096;
                cb[(row * cw as i64 + col) as usize] = a.clamp(0, 255) as u8;
                cr[(row * cw as i64 + col) as usize] = b.clamp(0, 255) as u8;
            }
        }
        frames.push((y, cb, cr));
    }
    Sequence {
        name: "blobs".into(),
        width: w,
        height: h,
        frames,
    }
}

/// A fixed textured scene panned at a fractional velocity (1.5, 0.5)
/// pixels per frame with wrap-around: every macro block has a non-zero,
/// half-pixel motion vector.
fn synth_pan(w: u32, h: u32, n: u32) -> Sequence {
    let (cw, ch) = (w / 2, h / 2);
    let mut frames = Vec::new();
    // Texture at 2× resolution so half-pixel pans are exact samples.
    let tw = (w * 2) as i64;
    let th = (h * 2) as i64;
    let mut rng = Lcg(0x9e37_79b9_7f4a_7c15);
    let mut grain: Vec<i64> = Vec::with_capacity((tw * th) as usize);
    for _ in 0..tw * th {
        grain.push((rng.next() % 7) as i64 - 3);
    }
    let sample = |x: i64, yy: i64| -> i64 {
        let x = x.rem_euclid(tw);
        let yy = yy.rem_euclid(th);
        let mut v = 128 * 4096;
        v += 40 * isin(x * 24) + 30 * isin(yy * 40) + 25 * isin((x + yy) * 12);
        // Sharp edges: a checker of 24-pixel squares.
        let cx = (x / 48) & 1;
        let cy = (yy / 48) & 1;
        if cx ^ cy == 1 {
            v += 35 * 4096;
        }
        v / 4096 + grain[(yy * tw + x) as usize] * 2
    };
    for t in 0..n as i64 {
        let ox = t * 3; // 1.5 px/frame in half-pixel units
        let oy = t; // 0.5 px/frame
        let mut y = vec![0u8; (w * h) as usize];
        for row in 0..h as i64 {
            for col in 0..w as i64 {
                let v = sample(col * 2 + ox, row * 2 + oy);
                y[(row * w as i64 + col) as usize] = v.clamp(0, 255) as u8;
            }
        }
        let mut cb = vec![0u8; (cw * ch) as usize];
        let mut cr = vec![0u8; (cw * ch) as usize];
        for row in 0..ch as i64 {
            for col in 0..cw as i64 {
                let v = sample(col * 4 + ox, row * 4 + oy);
                cb[(row * cw as i64 + col) as usize] = ((v * 3 + 128) / 4).clamp(0, 255) as u8;
                cr[(row * cw as i64 + col) as usize] = ((384 - v) / 2 + 64).clamp(0, 255) as u8;
            }
        }
        frames.push((y, cb, cr));
    }
    Sequence {
        name: "pan".into(),
        width: w,
        height: h,
        frames,
    }
}

/// Scene cut half-way: the `square0` family then the `blobs` content.
fn synth_cut(w: u32, h: u32, n: u32) -> Sequence {
    let a = synth_square(w, h, n / 2, 0);
    let b = synth_blobs(w, h, n - n / 2);
    let mut frames = a.frames;
    frames.extend(b.frames);
    Sequence {
        name: "cut".into(),
        width: w,
        height: h,
        frames,
    }
}

fn fixture_sequence(dir: &std::path::Path, name: &str, w: u32, h: u32) -> Option<Sequence> {
    let path = dir.join(name).join("expected.yuv");
    let data = std::fs::read(&path).ok()?;
    let ysz = (w * h) as usize;
    let csz = ((w / 2) * (h / 2)) as usize;
    let fsz = ysz + 2 * csz;
    if data.len() % fsz != 0 {
        eprintln!(
            "{}: size {} is not a multiple of {fsz}",
            path.display(),
            data.len()
        );
        return None;
    }
    let mut frames = Vec::new();
    for f in data.chunks_exact(fsz) {
        frames.push((
            f[..ysz].to_vec(),
            f[ysz..ysz + csz].to_vec(),
            f[ysz + csz..].to_vec(),
        ));
    }
    Some(Sequence {
        name: format!("fx-{name}"),
        width: w,
        height: h,
        frames,
    })
}

// ----------------------------------------------------------------------
// Encoding profiles
// ----------------------------------------------------------------------

type Profile = fn(TheoraEncoder) -> TheoraEncoder;

fn profiles() -> Vec<(&'static str, Profile)> {
    vec![
        ("default", |e| e),
        ("scenecut", |e| e.with_scene_cut_threshold(24.0)),
        ("kfpolicy", |e| e.with_keyframe_rate_policy(0.7)),
        ("adaptive", |e| e.with_adaptive_quant(vec![32, 16, 48])),
        ("adaptauto", |e| e.with_adaptive_quant_auto()),
    ]
}

// ----------------------------------------------------------------------
// Encode / decode / measure
// ----------------------------------------------------------------------

struct Point {
    seq: String,
    label: String,
    bytes: usize,
    psnr_y: f64,
    psnr_c: f64,
    keyframes: usize,
}

fn video_frame(seq: &Sequence, t: usize) -> VideoFrame {
    let (y, cb, cr) = &seq.frames[t];
    VideoFrame {
        pts: Some(t as i64),
        planes: vec![
            VideoPlane {
                stride: seq.width as usize,
                data: y.clone(),
            },
            VideoPlane {
                stride: (seq.width / 2) as usize,
                data: cb.clone(),
            },
            VideoPlane {
                stride: (seq.width / 2) as usize,
                data: cr.clone(),
            },
        ],
    }
}

fn psnr(sse: f64, n: f64) -> f64 {
    if sse == 0.0 {
        99.0
    } else {
        10.0 * (255.0f64 * 255.0 * n / sse).log10()
    }
}

fn measure(
    seq: &Sequence,
    mut enc: TheoraEncoder,
    label: &str,
    out: Option<&std::path::Path>,
) -> Point {
    for t in 0..seq.frames.len() {
        enc.send_frame(&Frame::Video(video_frame(seq, t))).unwrap();
    }
    let mut pkts: Vec<Packet> = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => pkts.push(p),
            Err(oxideav_core::Error::NeedMore) => break,
            Err(e) => panic!("encoder error {e}"),
        }
    }
    let bytes: usize = pkts
        .iter()
        .filter(|p| !p.flags.header)
        .map(|p| p.data.len())
        .sum();
    let keyframes = pkts
        .iter()
        .filter(|p| !p.flags.header && p.flags.keyframe)
        .count();
    if let Some(dir) = out {
        let mut chain = Vec::new();
        for p in &pkts {
            chain.extend_from_slice(&(p.data.len() as u32).to_le_bytes());
            chain.extend_from_slice(&p.data);
        }
        let fname = dir.join(format!("{}-{}.chain", seq.name, label.replace(' ', "_")));
        std::fs::write(fname, chain).unwrap();
    }

    let mut dec = TheoraDecoder::new(CodecId::new(THEORA_CODEC_ID));
    let mut sse_y = 0f64;
    let mut sse_c = 0f64;
    let mut t = 0usize;
    for p in &pkts {
        dec.send_packet(p).unwrap();
        if p.flags.header {
            continue;
        }
        let Frame::Video(vf) = dec.receive_frame().unwrap() else {
            panic!("non-video frame");
        };
        let (sy, scb, scr) = &seq.frames[t];
        let dims = [
            (seq.width as usize, seq.height as usize, sy),
            ((seq.width / 2) as usize, (seq.height / 2) as usize, scb),
            ((seq.width / 2) as usize, (seq.height / 2) as usize, scr),
        ];
        for (pi, (w, h, src)) in dims.iter().enumerate() {
            let pl = &vf.planes[pi];
            let mut sse = 0u64;
            for row in 0..*h {
                let got = &pl.data[row * pl.stride..row * pl.stride + w];
                let want = &src[row * w..row * w + w];
                for (a, b) in got.iter().zip(want) {
                    let d = *a as i64 - *b as i64;
                    sse += (d * d) as u64;
                }
            }
            if pi == 0 {
                sse_y += sse as f64;
            } else {
                sse_c += sse as f64;
            }
            if std::env::var_os("RD_VERBOSE").is_some() {
                let n = (w * h) as f64;
                eprint!(
                    "  f{t:<3} p{pi} {:>6.2} dB{}",
                    psnr(sse as f64, n),
                    if pi == 2 {
                        format!("  ({} B, kf={})\n", p.data.len(), p.flags.keyframe)
                    } else {
                        String::new()
                    }
                );
            }
        }
        t += 1;
    }
    let n = seq.frames.len() as f64;
    let ny = n * (seq.width * seq.height) as f64;
    let nc = 2.0 * n * ((seq.width / 2) * (seq.height / 2)) as f64;
    Point {
        seq: seq.name.clone(),
        label: label.to_string(),
        bytes,
        psnr_y: psnr(sse_y, ny),
        psnr_c: psnr(sse_c, nc),
        keyframes,
    }
}

// ----------------------------------------------------------------------
// Bjøntegaard-style deltas (piecewise-linear in log-rate)
// ----------------------------------------------------------------------

/// Average PSNR difference over the overlapping rate range and average
/// rate difference (%) over the overlapping PSNR range, `test` vs
/// `base`. Piecewise-linear interpolation on (log2 bytes, PSNR).
fn bd_deltas(base: &[(f64, f64)], test: &[(f64, f64)]) -> Option<(f64, f64)> {
    fn interp(curve: &[(f64, f64)], x: f64) -> f64 {
        for w in curve.windows(2) {
            let (x0, y0) = w[0];
            let (x1, y1) = w[1];
            if (x0..=x1).contains(&x) {
                return if x1 == x0 {
                    y0
                } else {
                    y0 + (y1 - y0) * (x - x0) / (x1 - x0)
                };
            }
        }
        f64::NAN
    }
    fn integrate(curve: &[(f64, f64)], lo: f64, hi: f64) -> f64 {
        let steps = 200;
        let mut acc = 0.0;
        for i in 0..steps {
            let x = lo + (hi - lo) * (i as f64 + 0.5) / steps as f64;
            acc += interp(curve, x);
        }
        acc / steps as f64
    }
    let mut b: Vec<(f64, f64)> = base.iter().map(|&(r, p)| (r.log2(), p)).collect();
    let mut t: Vec<(f64, f64)> = test.iter().map(|&(r, p)| (r.log2(), p)).collect();
    b.sort_by(|a, c| a.0.partial_cmp(&c.0).unwrap());
    t.sort_by(|a, c| a.0.partial_cmp(&c.0).unwrap());
    let lo = b[0].0.max(t[0].0);
    let hi = b[b.len() - 1].0.min(t[t.len() - 1].0);
    if hi <= lo {
        return None;
    }
    let d_psnr = integrate(&t, lo, hi) - integrate(&b, lo, hi);
    // Rate axis: invert curves to (PSNR → log-rate).
    let mut bi: Vec<(f64, f64)> = b.iter().map(|&(r, p)| (p, r)).collect();
    let mut ti: Vec<(f64, f64)> = t.iter().map(|&(r, p)| (p, r)).collect();
    bi.sort_by(|a, c| a.0.partial_cmp(&c.0).unwrap());
    ti.sort_by(|a, c| a.0.partial_cmp(&c.0).unwrap());
    let plo = bi[0].0.max(ti[0].0);
    let phi = bi[bi.len() - 1].0.min(ti[ti.len() - 1].0);
    if phi <= plo {
        return Some((d_psnr, f64::NAN));
    }
    let d_log_rate = integrate(&ti, plo, phi) - integrate(&bi, plo, phi);
    let d_rate_pct = (2f64.powf(d_log_rate) - 1.0) * 100.0;
    Some((d_psnr, d_rate_pct))
}

// ----------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------

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
        "{:<22} {:<10} {:>8} {:>8} {:>8} {:>4}",
        "sequence", "point", "bytes", "Y-PSNR", "C-PSNR", "kf"
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
                "{:<22} {:<10} {:>8} {:>8.2} {:>8.2} {:>4}",
                p.seq, p.label, p.bytes, p.psnr_y, p.psnr_c, p.keyframes
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
                "{:<22} {:<10} {:>8} {:>8.2} {:>8.2} {:>4}",
                p.seq, p.label, p.bytes, p.psnr_y, p.psnr_c, p.keyframes
            );
            points.push(p);
        }
    }

    if let Some(path) = &save {
        let mut s = String::new();
        for p in &points {
            s.push_str(&format!(
                "{}\t{}\t{}\t{:.4}\t{:.4}\t{}\n",
                p.seq, p.label, p.bytes, p.psnr_y, p.psnr_c, p.keyframes
            ));
        }
        std::fs::write(path, s).unwrap();
    }

    if let Some(path) = &reference {
        let text = std::fs::read_to_string(path).unwrap();
        let mut base: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
        let mut base_pts: BTreeMap<(String, String), (usize, f64)> = BTreeMap::new();
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 5 || !f[1].starts_with("qi") {
                continue;
            }
            let bytes: f64 = f[2].parse().unwrap();
            let py: f64 = f[3].parse().unwrap();
            base.entry(f[0].to_string()).or_default().push((bytes, py));
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
        println!("BD deltas vs reference (luma; qi ladder points):");
        let mut sum_p = 0.0;
        let mut sum_r = 0.0;
        let mut n = 0;
        for (seq, bcurve) in &base {
            let tcurve: Vec<(f64, f64)> = points
                .iter()
                .filter(|p| &p.seq == seq && p.label.starts_with("qi"))
                .map(|p| (p.bytes as f64, p.psnr_y))
                .collect();
            if tcurve.len() < 2 {
                continue;
            }
            if let Some((dp, dr)) = bd_deltas(bcurve, &tcurve) {
                println!("  {seq:<22} BD-PSNR {dp:>+6.3} dB   BD-rate {dr:>+7.2} %");
                sum_p += dp;
                sum_r += dr;
                n += 1;
            }
        }
        if n > 0 {
            println!(
                "  {:<22} BD-PSNR {:>+6.3} dB   BD-rate {:>+7.2} %",
                "MEAN",
                sum_p / n as f64,
                sum_r / n as f64
            );
        }
    }
}
