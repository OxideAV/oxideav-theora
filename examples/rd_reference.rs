//! Rate-distortion comparison against a black-box reference Theora
//! encoder, when one is installed.
//!
//! The comparison needs an `ffmpeg` binary whose build includes a
//! Theora *encoder* (`ffmpeg -encoders` lists one). Many builds ship
//! only the decoder; in that case this example prints why it cannot
//! compare and exits successfully — the same runtime skip the tests
//! use for absent tools. Nothing here reads how the reference encoder
//! works: it is run for its output files only.
//!
//! For each of the deterministic sequences of `tests/common/rd.rs`
//! (at `--size`, default 352x288, `--frames` frames):
//!
//! * this crate's encoder runs across `--qis` (default 8,20,32,44,56),
//!   keyframe interval `--interval` (default 64), and is scored through
//!   this crate's decoder (bytes of the data packets, luma / chroma
//!   PSNR, luma SSIM);
//! * the reference encoder runs across `--ref-q` (its 0..=10 quality
//!   scale, default 1,3,5,7,9) at the same keyframe interval on the
//!   same raw frames; its `.ogv` is decoded back to raw video by the
//!   same binary and scored with the same PSNR / SSIM code, its rate
//!   being the sum of the video packet sizes `ffprobe` reports.
//!
//! The Bjøntegaard deltas (this encoder against the reference, negative
//! rate = fewer bytes at equal quality) are printed per sequence and as
//! a mean. Scratch files go to `--work <dir>` (default: a directory
//! under the system temporary directory).
//!
//! ```text
//! cargo run --release --example rd_reference -- [--size WxH] [--frames n]
//!     [--interval n] [--qis a,b,..] [--ref-q a,b,..] [--work dir]
//! ```

#[path = "../tests/common/rd.rs"]
mod rd;

use oxideav_core::CodecId;
use oxideav_theora::{PixelFormat, THEORA_CODEC_ID, TheoraEncoder, TheoraIdentHeader};
use rd::*;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Name of the Theora encoder the local `ffmpeg` offers, if any.
fn reference_encoder() -> Option<String> {
    let out = Command::new("ffmpeg")
        .args(["-hide_banner", "-encoders"])
        .output()
        .ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    for line in text.lines() {
        // One encoder per line: capability flags, name, description.
        let mut cols = line.split_whitespace();
        let (Some(flags), Some(name)) = (cols.next(), cols.next()) else {
            continue;
        };
        if flags.starts_with('V') && name.to_ascii_lowercase().contains("theora") {
            return Some(name.to_string());
        }
    }
    None
}

fn write_raw(seq: &Sequence, path: &Path) {
    let mut raw = Vec::new();
    for (y, cb, cr) in &seq.frames {
        raw.extend_from_slice(y);
        raw.extend_from_slice(cb);
        raw.extend_from_slice(cr);
    }
    std::fs::write(path, raw).expect("write raw source");
}

/// Encode `raw` with the reference encoder at quality `q`; return
/// (video bytes, luma PSNR, chroma PSNR, luma SSIM).
fn reference_point(
    encoder: &str,
    seq: &Sequence,
    raw: &Path,
    work: &Path,
    q: u32,
    interval: u32,
) -> Option<(f64, f64, f64, f64)> {
    let ogv = work.join(format!("{}-q{q}.ogv", seq.name));
    let dec = work.join(format!("{}-q{q}.yuv", seq.name));
    let size = format!("{}x{}", seq.width, seq.height);
    let ok = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y"])
        .args([
            "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", &size, "-r", "30", "-i",
        ])
        .arg(raw)
        .args([
            "-c:v",
            encoder,
            "-q:v",
            &q.to_string(),
            "-g",
            &interval.to_string(),
        ])
        .arg(&ogv)
        .status()
        .ok()?
        .success();
    if !ok {
        return None;
    }
    let ok = Command::new("ffmpeg")
        .args(["-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(&ogv)
        .args([
            "-fps_mode",
            "passthrough",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&dec)
        .status()
        .ok()?
        .success();
    if !ok {
        return None;
    }
    // Rate: the video packets only, without the container framing.
    let probe = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0"])
        .args(["-show_entries", "packet=size", "-of", "csv=p=0"])
        .arg(&ogv)
        .output()
        .ok()?;
    let bytes: f64 = String::from_utf8_lossy(&probe.stdout)
        .lines()
        .filter_map(|l| l.trim().trim_end_matches(',').parse::<f64>().ok())
        .sum();
    let got = std::fs::read(&dec).ok()?;
    let (w, h) = (seq.width as usize, seq.height as usize);
    let (cw, ch) = (w / 2, h / 2);
    let frame_len = w * h + 2 * cw * ch;
    if bytes <= 0.0 || got.len() != frame_len * seq.frames.len() {
        return None;
    }
    let (mut sse_y, mut sse_c, mut ssim) = (0f64, 0f64, 0f64);
    for (f, (sy, scb, scr)) in got.chunks_exact(frame_len).zip(&seq.frames) {
        let (gy, rest) = f.split_at(w * h);
        let (gcb, gcr) = rest.split_at(cw * ch);
        let sse = |a: &[u8], b: &[u8]| -> f64 {
            a.iter()
                .zip(b)
                .map(|(&x, &y)| {
                    let d = f64::from(x) - f64::from(y);
                    d * d
                })
                .sum()
        };
        sse_y += sse(gy, sy);
        sse_c += sse(gcb, scb) + sse(gcr, scr);
        ssim += ssim_plane(gy, w, sy, w, h);
    }
    let n = seq.frames.len() as f64;
    Some((
        bytes,
        psnr(sse_y, n * (w * h) as f64),
        psnr(sse_c, 2.0 * n * (cw * ch) as f64),
        ssim / n,
    ))
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut w, mut h) = (352u32, 288u32);
    let mut frames = 24u32;
    let mut interval = 64u32;
    let mut qis: Vec<u8> = vec![8, 20, 32, 44, 56];
    let mut ref_q: Vec<u32> = vec![1, 3, 5, 7, 9];
    let mut work: Option<PathBuf> = None;
    let mut i = 0;
    while i < args.len() {
        let val = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i).cloned().unwrap_or_default()
        };
        match args[i].as_str() {
            "--size" => {
                let v = val(&mut i);
                let (a, b) = v.split_once('x').expect("--size WxH");
                w = a.parse().unwrap();
                h = b.parse().unwrap();
            }
            "--frames" => frames = val(&mut i).parse().unwrap(),
            "--interval" => interval = val(&mut i).parse().unwrap(),
            "--qis" => qis = val(&mut i).split(',').map(|q| q.parse().unwrap()).collect(),
            "--ref-q" => ref_q = val(&mut i).split(',').map(|q| q.parse().unwrap()).collect(),
            "--work" => work = Some(val(&mut i).into()),
            other => panic!("unknown option {other}"),
        }
        i += 1;
    }

    let Some(encoder) = reference_encoder() else {
        println!(
            "rd_reference: skipped — `ffmpeg -encoders` lists no Theora encoder on this \
             machine (or ffmpeg is not installed), so there is no black-box reference to \
             compare against. The encoder's own ladder is `--example rd_ladder`."
        );
        return;
    };
    let work = work.unwrap_or_else(|| std::env::temp_dir().join("oxideav-theora-rd-reference"));
    std::fs::create_dir_all(&work).expect("create work directory");
    println!("reference encoder: ffmpeg -c:v {encoder}");

    let seqs = [
        synth_square(w, h, frames, 0),
        synth_blobs(w, h, frames),
        synth_pan(w, h, frames),
        synth_cut(w, h, frames),
    ];
    let mut rate_deltas: Vec<[f64; 3]> = Vec::new();
    for seq in &seqs {
        let raw = work.join(format!("{}.yuv", seq.name));
        write_raw(seq, &raw);
        let ident =
            TheoraIdentHeader::for_picture(seq.width, seq.height, PixelFormat::Yuv420, 30, 1)
                .unwrap();
        let ours: Vec<Point> = qis
            .iter()
            .map(|&qi| {
                let enc = TheoraEncoder::with_default_setup_keyframe_interval(
                    CodecId::new(THEORA_CODEC_ID),
                    ident.clone(),
                    qi,
                    interval,
                )
                .unwrap();
                measure(seq, enc, &format!("qi{qi}"), None)
            })
            .collect();
        let theirs: Vec<(u32, (f64, f64, f64, f64))> = ref_q
            .iter()
            .filter_map(|&q| {
                reference_point(&encoder, seq, &raw, &work, q, interval).map(|p| (q, p))
            })
            .collect();
        println!(
            "\n{:<10} {:<8} {:>9} {:>8} {:>8} {:>7}",
            seq.name, "point", "bytes", "Y-PSNR", "C-PSNR", "Y-SSIM"
        );
        for p in &ours {
            println!(
                "{:<10} {:<8} {:>9} {:>8.2} {:>8.2} {:>7.4}",
                "this", p.label, p.bytes, p.psnr_y, p.psnr_c, p.ssim_y
            );
        }
        for (q, (bytes, py, pc, ss)) in &theirs {
            println!(
                "{:<10} {:<8} {:>9.0} {:>8.2} {:>8.2} {:>7.4}",
                "reference",
                format!("q{q}"),
                bytes,
                py,
                pc,
                ss
            );
        }
        if theirs.len() < 2 {
            println!("  (reference encoder produced fewer than two usable points; no BD figures)");
            continue;
        }
        let curve = |f: fn(&Point) -> f64| -> Vec<(f64, f64)> {
            ours.iter().map(|p| (p.bytes as f64, f(p))).collect()
        };
        let base = |k: usize| -> Vec<(f64, f64)> {
            theirs
                .iter()
                .map(|(_, p)| (p.0, [p.1, p.2, p.3][k]))
                .collect()
        };
        let dy = bd_deltas(&base(0), &curve(|p| p.psnr_y));
        let dc = bd_deltas(&base(1), &curve(|p| p.psnr_c));
        let ds = bd_deltas(&base(2), &curve(|p| p.ssim_y));
        let rate = |d: Option<(f64, f64)>| d.map_or(f64::NAN, |(_, r)| r);
        println!(
            "  this encoder vs reference: BD-PSNR {:+.3} dB  BD-rate Y {:+.2} %  C {:+.2} %  SSIM {:+.2} %",
            dy.map_or(f64::NAN, |(p, _)| p),
            rate(dy),
            rate(dc),
            rate(ds)
        );
        rate_deltas.push([rate(dy), rate(dc), rate(ds)]);
    }
    if !rate_deltas.is_empty() {
        let mean = |k: usize| -> f64 {
            let v: Vec<f64> = rate_deltas
                .iter()
                .map(|d| d[k])
                .filter(|x| x.is_finite())
                .collect();
            v.iter().sum::<f64>() / v.len().max(1) as f64
        };
        println!(
            "\nMEAN BD-rate, this encoder vs reference: Y {:+.2} %  C {:+.2} %  SSIM {:+.2} %",
            mean(0),
            mean(1),
            mean(2)
        );
    }
}
