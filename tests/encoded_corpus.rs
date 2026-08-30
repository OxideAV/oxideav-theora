//! Pinned self-encoded corpus (round 413).
//!
//! Twelve deterministic encoder scenarios — the same family that was
//! validated externally in round 413 (muxed into Ogg via the published
//! container crate and black-box-decoded pixel-exactly; see
//! `tests/encoded-corpus-notes.md`) — are re-encoded here on every run
//! and pinned by SHA-256, both at the wire (the length-prefixed packet
//! chain: three §6 headers then every §7 data packet) and at the
//! output of this crate's own decoder (concatenated §2.2-cropped
//! top-down planar frames).
//!
//! A digest change means the encoder's output moved. That is allowed —
//! RD tuning legitimately re-spells streams — but it must be
//! *intentional*: update the digests in the same commit and re-run the
//! external validation route from the notes before doing so.

use oxideav_core::{CodecId, Decoder as _, Encoder as _, Packet};
use oxideav_theora::{
    InterModeStrategy, PixelFormat, SetupHeaderTables, SourceFrame, TheoraDecoder, TheoraEncoder,
    TheoraIdentHeader, THEORA_CODEC_ID,
};

// ----------------------------------------------------------------------
// Minimal SHA-256 (FIPS 180-4), test-only — no external dependencies.
// ----------------------------------------------------------------------

fn sha256_hex(data: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bitlen = (data.len() as u64) * 8;
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bitlen.to_be_bytes());
    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes(word.try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    h.iter().map(|v| format!("{v:08x}")).collect()
}

// ----------------------------------------------------------------------
// Deterministic content generator (identical to the round-413 external
// validation harness, scaled to CI-friendly sizes).
// ----------------------------------------------------------------------

fn gen_planes(
    w: u32,
    h: u32,
    cw: u32,
    ch: u32,
    t: u32,
    family: u32,
) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let (w, h, cw, ch) = (w as usize, h as usize, cw as usize, ch as usize);
    let t = t as usize;
    let fam = family as usize;
    let mut y = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..w {
            let base = match fam {
                0 => (col * 2 + row * 3 + t * 5) % 256,
                _ => 255 - ((col * 3).wrapping_add(row * 7).wrapping_add(t * 11) % 256),
            };
            // Moving 32×32 bright square sweeping diagonally.
            let sq_x = (t * 7 + fam * 40) % (w.max(33) - 32);
            let sq_y = (t * 4) % (h.max(33) - 32);
            let v = if col >= sq_x && col < sq_x + 32 && row >= sq_y && row < sq_y + 32 {
                220
            } else {
                base
            };
            y[row * w + col] = v as u8;
        }
    }
    let mut cb = vec![0u8; cw * ch];
    let mut cr = vec![0u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            cb[row * cw + col] = ((col * 200 / cw.max(1)) + t * 2 + fam * 30) as u8;
            cr[row * cw + col] = ((row * 200 / ch.max(1)) + t * 3) as u8;
        }
    }
    (y, cb, cr)
}

fn video_frame(
    ident: &TheoraIdentHeader,
    t: u32,
    family: u32,
    still: bool,
) -> oxideav_core::VideoFrame {
    use oxideav_core::frame::VideoPlane;
    let (py, pc) = ident.picture_plane_dims();
    let tt = if still { 0 } else { t };
    let (y, cb, cr) = gen_planes(py.width, py.height, pc.width, pc.height, tt, family);
    oxideav_core::VideoFrame {
        pts: Some(t as i64),
        planes: vec![
            VideoPlane {
                stride: py.width as usize,
                data: y,
            },
            VideoPlane {
                stride: pc.width as usize,
                data: cb,
            },
            VideoPlane {
                stride: pc.width as usize,
                data: cr,
            },
        ],
    }
}

fn ident(w: u32, h: u32, pf: PixelFormat) -> TheoraIdentHeader {
    // 30 fps; `for_picture` supplies the container-carriable KFGSHIFT.
    TheoraIdentHeader::for_picture(w, h, pf, 30, 1).unwrap()
}

/// Drive `enc` over `nframes` generated frames; return the emitted
/// packet chain (headers first) and the number of header packets.
fn drive(
    mut enc: TheoraEncoder,
    id: &TheoraIdentHeader,
    nframes: u32,
    family: fn(u32) -> u32,
    still: bool,
) -> Vec<Packet> {
    for t in 0..nframes {
        let vf = video_frame(id, t, family(t), still);
        enc.send_frame(&oxideav_core::Frame::Video(vf)).unwrap();
    }
    let mut pkts = Vec::new();
    loop {
        match enc.receive_packet() {
            Ok(p) => pkts.push(p),
            Err(oxideav_core::Error::NeedMore) => break,
            Err(e) => panic!("encoder error {e}"),
        }
    }
    pkts
}

/// Length-prefixed concatenation of the packet chain — the wire pin.
fn packet_chain_bytes(pkts: &[Packet]) -> Vec<u8> {
    let mut out = Vec::new();
    for p in pkts {
        out.extend_from_slice(&(p.data.len() as u32).to_le_bytes());
        out.extend_from_slice(&p.data);
    }
    out
}

/// Decode the chain through `TheoraDecoder` and concatenate the
/// §2.2-cropped top-down planes — the reconstruction pin.
fn reconstruction_bytes(id: &TheoraIdentHeader, pkts: &[Packet]) -> Vec<u8> {
    let mut dec = TheoraDecoder::new(CodecId::new(THEORA_CODEC_ID));
    let (py, pc) = id.picture_plane_dims();
    let dims = [
        (py.width as usize, py.height as usize),
        (pc.width as usize, pc.height as usize),
        (pc.width as usize, pc.height as usize),
    ];
    let mut out = Vec::new();
    for p in pkts {
        dec.send_packet(p).unwrap();
        if p.flags.header {
            continue;
        }
        let oxideav_core::Frame::Video(vf) = dec.receive_frame().unwrap() else {
            panic!("non-video frame");
        };
        for (plane, (w, h)) in vf.planes.iter().zip(dims) {
            for row in 0..h {
                out.extend_from_slice(&plane.data[row * plane.stride..row * plane.stride + w]);
            }
        }
    }
    out
}

fn fam0(_t: u32) -> u32 {
    0
}

fn fam_cut(t: u32) -> u32 {
    if t < 4 {
        0
    } else {
        1
    }
}

struct Pin {
    name: &'static str,
    wire_sha256: &'static str,
    recon_sha256: &'static str,
}

fn check(pin: &Pin, id: &TheoraIdentHeader, pkts: &[Packet]) {
    assert_eq!(
        pkts.iter().filter(|p| p.flags.header).count(),
        3,
        "{}: three §6 header packets",
        pin.name
    );
    let wire = sha256_hex(&packet_chain_bytes(pkts));
    let recon = sha256_hex(&reconstruction_bytes(id, pkts));
    // `CORPUS_DUMP=<dir>` writes each scenario's packet chain (the
    // wire-pinned bytes) to `<dir>/<name>.chain` for the external
    // validation route in `encoded-corpus-notes.md`.
    if let Some(dir) = std::env::var_os("CORPUS_DUMP") {
        let path = std::path::Path::new(&dir).join(format!("{}.chain", pin.name));
        std::fs::write(path, packet_chain_bytes(pkts)).expect("CORPUS_DUMP write");
    }
    if std::env::var_os("CORPUS_PRINT").is_some() {
        println!(
            "        Pin {{ name: \"{}\", wire_sha256: \"{wire}\", recon_sha256: \"{recon}\" }},",
            pin.name
        );
        return;
    }
    assert_eq!(
        wire, pin.wire_sha256,
        "{}: wire digest moved — if intentional, re-run the external validation in \
         tests/encoded-corpus-notes.md and update both digests",
        pin.name
    );
    assert_eq!(
        recon, pin.recon_sha256,
        "{}: reconstruction digest moved — if intentional, re-run the external validation \
         in tests/encoded-corpus-notes.md and update both digests",
        pin.name
    );
}

/// The twelve-pin corpus. `CORPUS_PRINT=1 cargo test --test
/// encoded_corpus -- --nocapture` prints the current `Pin` lines for a
/// deliberate re-pin.
#[test]
fn encoded_corpus_digests_are_stable() {
    let cid = || CodecId::new(THEORA_CODEC_ID);

    const PINS: [Pin; 12] = [
        Pin {
            name: "basic420",
            wire_sha256: "a4930c9932033ebc6390a05e7d62de76384207bea459163cf749a86e3d20d330",
            recon_sha256: "8d5acc6e5bda4dd08e0ed40d556e886e184c77ba02add1ec9f5300986d41f4cc",
        },
        Pin {
            name: "fmt422",
            wire_sha256: "0ec0e600e41e06663baf55d31d5b4d2db11000e12c7afdd6673dbbdf0caa2231",
            recon_sha256: "94ab87eb835f85852dcd9106708fb4d6ae9f274f8c96ad6b753cd6f8c358a76f",
        },
        Pin {
            name: "fmt444",
            wire_sha256: "3f32c3ace7ec11926cbe7bf66a6d9d2df238f5eb9eb66e17873f819b3dd86bc6",
            recon_sha256: "e5cf415015a5bdac7cf59d54abdef53578b32ab538c1c877fab012c52ef6b0e2",
        },
        Pin {
            name: "piccrop",
            wire_sha256: "642b6366452c6efba82a5858ed087c6c719264909bf1559d7d46eea5a4c2df24",
            recon_sha256: "475fea4f579584c8da6eeb0e82e45e3ced5b2e5d45d840b93eb70a57f2fd3b35",
        },
        Pin {
            name: "adaptiveq",
            wire_sha256: "ac0dcae5eeb80faa5bd5cf484f9b14032f1f286aa2d7ea7dbe09a33621601eb5",
            recon_sha256: "6d0d31be7e8f4bc3428407b62014839fe20ec0df00c23cecae49ea4747e90e5a",
        },
        Pin {
            name: "ratecontrol",
            wire_sha256: "81ed0ff6e246b6e23efeaa80df6427126bf209838e4d26c72f06d8e4a0db0edb",
            recon_sha256: "ea5deedd785234e9287d13de8417ea0cfc2d13db0ee1f8448bbdd7461a3bf422",
        },
        Pin {
            name: "dupframes",
            wire_sha256: "e47b138ce1c1bda82eed258c93c0a390fcc5f3f282c6d1e8ebd876e14c45a145",
            recon_sha256: "980a6e4e7f780cc1b96e51e292f82b77054cce325c2f1d32061fc8a91d9abf1f",
        },
        Pin {
            name: "scenecut",
            wire_sha256: "fd1f14e09b30ebf994d18b69082b14ae6cbcd8ef3f2f4bb187250b664292f0e9",
            recon_sha256: "c4931331d0da68d2ea50b23342e27d85b22ed90ff5c0b601934dfb2ea32c1d14",
        },
        Pin {
            name: "goptuned",
            wire_sha256: "17180cca82bc5ad24c9c4c7ea5cd6c5932d81deb6aaca7b05f8e9cbf0c81aa2a",
            recon_sha256: "046cc7098881165cbd618e8d488b66b4eafac717486f657626e7513900a13545",
        },
        Pin {
            name: "fourmv",
            wire_sha256: "b0c29423c813373f216a49978a95c47e35499f0479bb20b039413e8cf4974e79",
            recon_sha256: "8b9daf4407cbe09c5b4428c0fa6e67224942a02838cd57ae542b1efec21164c1",
        },
        Pin {
            name: "golden",
            wire_sha256: "a21553cb1c7939b8b2c5c40759cc121bb2758227d75e53dfbf98ea756f707966",
            recon_sha256: "60bb57f5c12da9d9906df2910c3d12e438e1dc5eda7395fb39181980280ca992",
        },
        Pin {
            name: "rcadaptive",
            wire_sha256: "fc00546de2cec463b49d1b6d0fbd6ddba01218c3a9f171fd9e33095a00d72f88",
            recon_sha256: "f3f817f87ef527b74b190fcc341fa49af39bd7df88ab39116012093661374db4",
        },
    ];

    // 1. Plain 4:2:0 I/P GOPs on the synthesized VP3-default setup.
    let id = ident(176, 144, PixelFormat::Yuv420);
    let pkts = drive(
        TheoraEncoder::with_default_setup_keyframe_interval(cid(), id.clone(), 40, 8).unwrap(),
        &id,
        16,
        fam0,
        false,
    );
    check(&PINS[0], &id, &pkts);

    // 2. / 3. The other two pixel formats.
    let id = ident(96, 80, PixelFormat::Yuv422);
    let pkts = drive(
        TheoraEncoder::with_default_setup_keyframe_interval(cid(), id.clone(), 44, 6).unwrap(),
        &id,
        8,
        fam0,
        false,
    );
    check(&PINS[1], &id, &pkts);

    let id = ident(96, 80, PixelFormat::Yuv444);
    let pkts = drive(
        TheoraEncoder::with_default_setup_keyframe_interval(cid(), id.clone(), 44, 6).unwrap(),
        &id,
        8,
        fam0,
        false,
    );
    check(&PINS[2], &id, &pkts);

    // 4. Non-MB-aligned §2.2 picture region (odd chroma window).
    let id = ident(130, 98, PixelFormat::Yuv420);
    let pkts = drive(
        TheoraEncoder::with_default_setup_keyframe_interval(cid(), id.clone(), 40, 5).unwrap(),
        &id,
        8,
        fam0,
        false,
    );
    check(&PINS[3], &id, &pkts);

    // 5. Adaptive quantization (§7.1 MOREQIS + §7.6 block-level qi).
    let id = ident(176, 144, PixelFormat::Yuv420);
    let pkts = drive(
        TheoraEncoder::with_default_setup_keyframe_interval(cid(), id.clone(), 40, 6)
            .unwrap()
            .with_adaptive_quant(vec![40, 24, 56]),
        &id,
        8,
        fam0,
        false,
    );
    check(&PINS[4], &id, &pkts);

    // 6. Target-bitrate rate control (NOMBR declared in the ident).
    let id = ident(176, 144, PixelFormat::Yuv420);
    let pkts = drive(
        TheoraEncoder::with_default_setup_keyframe_interval(cid(), id.clone(), 40, 8)
            .unwrap()
            .with_target_bitrate(150_000),
        &id,
        16,
        fam0,
        false,
    );
    check(&PINS[5], &id, &pkts);

    // 7. Still content: §7.11 step-2 zero-byte duplicate-frame packets.
    let id = ident(128, 96, PixelFormat::Yuv420);
    let pkts = drive(
        TheoraEncoder::with_default_setup_keyframe_interval(cid(), id.clone(), 40, 10).unwrap(),
        &id,
        12,
        fam0,
        true,
    );
    assert!(
        pkts.iter().any(|p| !p.flags.header && p.data.is_empty()),
        "dupframes: still content must emit zero-byte duplicate packets"
    );
    check(&PINS[6], &id, &pkts);

    // 8. Scene-cut detection (content family switch at frame 4).
    let id = ident(176, 144, PixelFormat::Yuv420);
    let pkts = drive(
        TheoraEncoder::with_default_setup_keyframe_interval(cid(), id.clone(), 40, 15)
            .unwrap()
            .with_scene_cut_threshold(20.0),
        &id,
        8,
        fam_cut,
        false,
    );
    check(&PINS[7], &id, &pkts);

    // 9. Two-pass GOP-tuned custom Huffman codebooks in the setup header.
    let id = ident(176, 144, PixelFormat::Yuv420);
    {
        let (py, pc) = id.picture_plane_dims();
        let flip = |p: &[u8], w: u32, h: u32| -> Vec<u8> {
            let (w, h) = (w as usize, h as usize);
            let mut o = Vec::with_capacity(w * h);
            for row in (0..h).rev() {
                o.extend_from_slice(&p[row * w..(row + 1) * w]);
            }
            o
        };
        let samples: Vec<SourceFrame> = (0..4)
            .map(|t| {
                let (y, cb, cr) = gen_planes(py.width, py.height, pc.width, pc.height, t, 0);
                SourceFrame::from_picture(
                    &id,
                    &flip(&y, py.width, py.height),
                    &flip(&cb, pc.width, pc.height),
                    &flip(&cr, pc.width, pc.height),
                )
                .unwrap()
            })
            .collect();
        let pkts = drive(
            TheoraEncoder::with_gop_tuned_setup_keyframe_interval(
                cid(),
                id.clone(),
                SetupHeaderTables::vp3_defaults(),
                40,
                5,
                &samples,
            )
            .unwrap(),
            &id,
            8,
            fam0,
            false,
        );
        check(&PINS[8], &id, &pkts);
    }

    // 10. / 11. The alternative inter-mode strategies.
    let id = ident(176, 144, PixelFormat::Yuv420);
    let pkts = drive(
        TheoraEncoder::with_default_setup_keyframe_interval(cid(), id.clone(), 40, 6)
            .unwrap()
            .with_inter_mode(InterModeStrategy::FourMv),
        &id,
        8,
        fam0,
        false,
    );
    check(&PINS[9], &id, &pkts);

    let id = ident(176, 144, PixelFormat::Yuv420);
    let pkts = drive(
        TheoraEncoder::with_default_setup_keyframe_interval(cid(), id.clone(), 40, 6)
            .unwrap()
            .with_inter_mode(InterModeStrategy::GoldenMotion),
        &id,
        8,
        fam0,
        false,
    );
    check(&PINS[10], &id, &pkts);

    // 12. Rate control **composed** with adaptive quantization (round
    // 444): the leaky bucket owns each frame's QIS[0] while the
    // caller's candidates ride as the per-block AC alternatives —
    // scenarios 5 and 6 pin each feature alone; this pins them
    // together (the combination was previously unreachable: the loop
    // observed every adaptive frame but steered none).
    let id = ident(176, 144, PixelFormat::Yuv420);
    let pkts = drive(
        TheoraEncoder::with_default_setup_keyframe_interval(cid(), id.clone(), 40, 8)
            .unwrap()
            .with_adaptive_quant(vec![40, 24, 56])
            .with_target_bitrate(150_000),
        &id,
        16,
        fam0,
        false,
    );
    // Composition evidence on the wire, independent of the digests:
    // some frame's QIS[0] left the seed (the bucket steered), and
    // every data frame still carries a multi-entry QIS list (the
    // caller's candidates ride).
    {
        let data: Vec<&Packet> = pkts.iter().filter(|p| !p.flags.header).collect();
        let mut moved = false;
        for (i, p) in data.iter().enumerate() {
            let hdr = oxideav_theora::decode_frame_header(&p.data, i == 0).unwrap();
            assert!(
                hdr.qis.len() >= 2,
                "rcadaptive frame {i}: AC candidates must ride behind the RC head"
            );
            moved |= hdr.qis[0] != 40;
        }
        assert!(
            moved,
            "rcadaptive: rate control must steer QIS[0] off the seed"
        );
    }
    check(&PINS[11], &id, &pkts);
}

// ----------------------------------------------------------------------
// Round 437 — decode-corner scenarios. Same generator, same external
// validation route (see tests/encoded-corpus-notes.md), aimed at wire
// states the staged fixture corpus never reaches on the decode side.
// ----------------------------------------------------------------------

/// VP3-default tables with every §6.4.1 loop-filter limit forced to
/// `limit`. The serializer picks the minimal `NBITS` for the table, so
/// `limit = 127` puts a 7-bit-wide LFLIMS on the wire and `limit = 0`
/// a zero-bit one (§5.2.5 zero-bit integer reads).
fn lflims_setup(limit: u8) -> SetupHeaderTables {
    let mut s = SetupHeaderTables::vp3_defaults();
    s.loop_filter_limits = [limit; 64];
    s
}

/// VP3-default tables rebuilt around **three quant ranges** per
/// `(qti, pli)` set (sizes 21 + 21 + 21, alternating between the set's
/// own VP3 base matrix and an extra flat matrix), so §6.4.3 must
/// interpolate across interior range boundaries the single-range VP3
/// assignment never has.
fn multiqrange_setup() -> SetupHeaderTables {
    let mut s = SetupHeaderTables::vp3_defaults();
    let qp = &mut s.quantization_parameters;
    qp.base_matrices.push([24u8; 64]);
    qp.num_base_matrices = qp.base_matrices.len() as u16;
    for qti in 0..2 {
        for pli in 0..3 {
            let orig = qp.quant_range_base_matrix_indices[qti][pli][0];
            qp.num_quant_ranges[qti][pli] = 3;
            qp.quant_range_sizes[qti][pli][..3].copy_from_slice(&[21, 21, 21]);
            let b = &mut qp.quant_range_base_matrix_indices[qti][pli];
            b[0] = orig;
            b[1] = 3;
            b[2] = orig;
            b[3] = 3;
        }
    }
    s
}

/// The §6.4.1 `NBITS` field of a serialized setup packet: the three
/// bits immediately after the 7-byte common header.
fn setup_lflims_nbits(setup_packet: &[u8]) -> u8 {
    setup_packet[7] >> 5
}

/// Round-437 decode-corner pins. Externally validated through the
/// round-413 route (Ogg mux → `oggz-validate` → black-box reference
/// decode, byte-compared against this crate's own reconstruction) at
/// these exact geometries; see tests/encoded-corpus-notes.md.
#[test]
fn encoded_corpus_decode_corner_digests_are_stable() {
    let cid = || CodecId::new(THEORA_CODEC_ID);

    const PINS: [Pin; 3] = [
        Pin {
            name: "lflims127",
            wire_sha256: "021c5499c04b44f29e866f7e7c64fd30bc72da61070c91f4ff11488b9547ecf0",
            recon_sha256: "69b474033201969bcdeff81c13adaacc9960e7f244d4aa84b15764101404bfc7",
        },
        Pin {
            name: "lflims0",
            wire_sha256: "2f9f367ba1405b9ebe7919410dfc79b99c4e25c2194f59632b023a1f8d7f110a",
            recon_sha256: "b93c9a0bd164079a0c4e39aa374c566d70142cc748fd3ffbc33f561e1b5abea0",
        },
        Pin {
            name: "multiqrange",
            wire_sha256: "3b78209cbd44dad89aaf332fd18ec8718ecff990b27406d61f71ba061b472592",
            recon_sha256: "13cfdaa95fd93a52f19c77af8740da521ff5627699fe192b0eff240fa4a67e6a",
        },
    ];

    // 12. LFLIMS at the 7-bit ceiling: `lflim()` runs with `L = 127`
    // on every edge of an I+P GOP — the staged fixtures only exercise
    // limits 0 and 15, so the wide half of the §7.10 response ramp
    // never ran on a real stream before.
    let id = ident(176, 144, PixelFormat::Yuv420);
    let pkts = drive(
        TheoraEncoder::with_keyframe_interval(cid(), id.clone(), lflims_setup(127), 40, 6).unwrap(),
        &id,
        8,
        fam0,
        false,
    );
    let setup_pkt = &pkts[2].data;
    assert_eq!(
        setup_lflims_nbits(setup_pkt),
        7,
        "lflims127: the serialized §6.4.1 table must be 7 bits wide"
    );
    check(&PINS[0], &id, &pkts);

    // 13. LFLIMS all-zero: the serializer picks NBITS = 0, so the
    // §6.4.1 table is sixty-four §5.2.5 zero-bit reads on the wire and
    // the §7.10 loop filter is skipped at every qi (the staged corpus
    // reaches the skip only through the reference table's qi-63 zero).
    let id = ident(176, 144, PixelFormat::Yuv420);
    let pkts = drive(
        TheoraEncoder::with_keyframe_interval(cid(), id.clone(), lflims_setup(0), 40, 6).unwrap(),
        &id,
        8,
        fam0,
        false,
    );
    let setup_pkt = &pkts[2].data;
    assert_eq!(
        setup_lflims_nbits(setup_pkt),
        0,
        "lflims0: the serialized §6.4.1 table must be zero bits wide"
    );
    check(&PINS[1], &id, &pkts);

    // 14. Three custom quant ranges per set + adaptive quantization
    // whose candidate qis (40 / 10 / 60) land in different ranges, so
    // the decoder's §6.4.3 interpolation crosses interior boundaries
    // of a *transmitted* (non-VP3) range layout on both frame types.
    let id = ident(176, 144, PixelFormat::Yuv420);
    let pkts = drive(
        TheoraEncoder::with_keyframe_interval(cid(), id.clone(), multiqrange_setup(), 40, 6)
            .unwrap()
            .with_adaptive_quant(vec![40, 10, 60]),
        &id,
        8,
        fam0,
        false,
    );
    check(&PINS[2], &id, &pkts);
}

/// The test-local SHA-256 must match FIPS 180-4 vectors (so the pins
/// above mean what they claim).
#[test]
fn corpus_sha256_matches_known_vectors() {
    assert_eq!(
        sha256_hex(b""),
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    );
    assert_eq!(
        sha256_hex(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}
