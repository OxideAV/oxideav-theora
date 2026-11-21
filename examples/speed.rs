//! Encode / decode throughput probe for the Theora codec.
//!
//! Generates one of the deterministic synthetic sequences from the
//! shared rate-distortion harness (`tests/common/rd.rs`) at a chosen
//! picture size, encodes it through [`TheoraEncoder`], decodes the
//! resulting packet chain through [`TheoraDecoder`], and prints frames
//! per second for both directions (median of `--repeat` runs) together
//! with a 64-bit FNV-1a digest of the wire bytes and of the decoded
//! planes — the digests are what "output identical" claims in the
//! README are checked against.
//!
//! ```text
//! cargo run --release --example speed -- [options]
//!   --size WxH        picture size (default 1280x720)
//!   --frames n        sequence length (default 16)
//!   --seq name        pan | blobs | square | cut (default pan)
//!   --interval n      keyframe interval (default 64; 1 = intra only)
//!   --qi n            frame quantizer index (default 32)
//!   --threads n       worker budget handed to both directions (default 1)
//!   --repeat n        timing repetitions, median reported (default 3)
//!   --save file       write the length-prefixed packet chain
//!   --load file       decode a previously saved chain (skips the encode)
//!   --recon file      write the decoded top-down planes (for a
//!                     black-box decode comparison)
//!   --no-digest       skip the plane digest (leaves the process doing
//!                     nothing but decoding, for CPU-time measurements)
//! ```

#[path = "../tests/common/rd.rs"]
mod rd;

use oxideav_core::{
    CodecId, Decoder as _, Encoder as _, ExecutionContext, Frame, Packet, TimeBase,
};
use oxideav_theora::{
    PixelFormat, THEORA_CODEC_ID, TheoraDecoder, TheoraEncoder, TheoraIdentHeader,
};
use std::time::Instant;

fn fnv1a(seed: u64, data: &[u8]) -> u64 {
    let mut h = seed;
    for &b in data {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut w, mut h) = (1280u32, 720u32);
    let mut frames = 16u32;
    let mut seq_name = "pan".to_string();
    let mut interval = 64u32;
    let mut qi = 32u8;
    let mut threads = 1usize;
    let mut repeat = 3usize;
    let mut save: Option<String> = None;
    let mut load: Option<String> = None;
    let mut recon_out: Option<String> = None;
    let mut digest_on = true;
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
            "--seq" => seq_name = val(&mut i),
            "--interval" => interval = val(&mut i).parse().unwrap(),
            "--qi" => qi = val(&mut i).parse().unwrap(),
            "--threads" => threads = val(&mut i).parse().unwrap(),
            "--repeat" => repeat = val(&mut i).parse::<usize>().unwrap().max(1),
            "--save" => save = Some(val(&mut i)),
            "--load" => load = Some(val(&mut i)),
            "--recon" => recon_out = Some(val(&mut i)),
            "--no-digest" => digest_on = false,
            other => panic!("unknown option {other}"),
        }
        i += 1;
    }
    let ctx = ExecutionContext::with_threads(threads);

    let pkts: Vec<Packet> = if let Some(path) = &load {
        let data = std::fs::read(path).expect("read chain");
        let mut out = Vec::new();
        let mut off = 0usize;
        let mut n = 0usize;
        while off + 4 <= data.len() {
            let len = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
            off += 4;
            let mut p = Packet::new(0, TimeBase::new(1, 30), data[off..off + len].to_vec());
            p.flags.header = n < 3;
            out.push(p);
            off += len;
            n += 1;
        }
        out
    } else {
        let seq = match seq_name.as_str() {
            "pan" => rd::synth_pan(w, h, frames),
            "blobs" => rd::synth_blobs(w, h, frames),
            "square" => rd::synth_square(w, h, frames, 0),
            "cut" => rd::synth_cut(w, h, frames),
            other => panic!("unknown sequence {other}"),
        };
        let ident = TheoraIdentHeader::for_picture(w, h, PixelFormat::Yuv420, 30, 1).unwrap();
        let inputs: Vec<Frame> = (0..seq.frames.len())
            .map(|t| Frame::Video(rd::video_frame(&seq, t)))
            .collect();
        let mut times = Vec::new();
        let mut pkts = Vec::new();
        for _ in 0..repeat {
            let mut enc = TheoraEncoder::with_default_setup_keyframe_interval(
                CodecId::new(THEORA_CODEC_ID),
                ident.clone(),
                qi,
                interval,
            )
            .unwrap();
            enc.set_execution_context(&ctx);
            let t0 = Instant::now();
            for f in &inputs {
                enc.send_frame(f).unwrap();
            }
            enc.flush().unwrap();
            pkts.clear();
            while let Ok(p) = enc.receive_packet() {
                pkts.push(p);
            }
            times.push(t0.elapsed().as_secs_f64());
        }
        let t = median(times);
        let bytes: usize = pkts
            .iter()
            .filter(|p| !p.flags.header)
            .map(|p| p.data.len())
            .sum();
        println!(
            "encode  {w}x{h} {seq_name} g={interval} qi={qi} threads={threads}: {:>8.2} fps  ({} frames, {:.3} s, {} bytes)",
            inputs.len() as f64 / t,
            inputs.len(),
            t,
            bytes
        );
        pkts
    };

    let mut wire = 0xcbf2_9ce4_8422_2325u64;
    let mut chain = Vec::new();
    for p in &pkts {
        wire = fnv1a(wire, &(p.data.len() as u32).to_le_bytes());
        wire = fnv1a(wire, &p.data);
        if save.is_some() {
            chain.extend_from_slice(&(p.data.len() as u32).to_le_bytes());
            chain.extend_from_slice(&p.data);
        }
    }
    if let Some(path) = &save {
        std::fs::write(path, &chain).expect("write chain");
    }

    let mut times = Vec::new();
    let mut digest = 0u64;
    let mut nframes = 0usize;
    let mut recon: Vec<u8> = Vec::new();
    for rep in 0..repeat {
        let mut dec = TheoraDecoder::new(CodecId::new(THEORA_CODEC_ID));
        dec.set_execution_context(&ctx);
        let mut d = 0xcbf2_9ce4_8422_2325u64;
        nframes = 0;
        let mut hash_time = 0f64;
        let t0 = Instant::now();
        for p in &pkts {
            dec.send_packet(p).unwrap();
            if p.flags.header {
                continue;
            }
            let Frame::Video(vf) = dec.receive_frame().unwrap() else {
                panic!("non-video frame");
            };
            let th = Instant::now();
            for pl in &vf.planes {
                if digest_on {
                    d = fnv1a(d, &pl.data);
                }
                if rep == 0 && recon_out.is_some() {
                    recon.extend_from_slice(&pl.data);
                }
            }
            hash_time += th.elapsed().as_secs_f64();
            nframes += 1;
        }
        times.push(t0.elapsed().as_secs_f64() - hash_time);
        digest = d;
    }
    if let Some(path) = &recon_out {
        std::fs::write(path, &recon).expect("write recon");
    }
    let t = median(times);
    println!(
        "decode  threads={threads}: {:>8.2} fps  ({} frames, {:.3} s)  wire={wire:016x} recon={digest:016x}",
        nframes as f64 / t,
        nframes,
        t
    );
}
