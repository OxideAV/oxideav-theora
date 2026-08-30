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
            wire_sha256: "2d6c30450cac948624a8fbd183f60842431724a7b4959dd7724157b5f971f9bc",
            recon_sha256: "ecd60900b75d4c963f6384147ed1094420b05f127bc274ec806f0a3ec3254184",
        },
        Pin {
            name: "fmt422",
            wire_sha256: "70d646e951aa7629642e5c032eb9e2f8ffa84612d646c93bd92d4525268f5e04",
            recon_sha256: "1b4be661332f360b8ca988f68b30ae546143142068053f926c98f2391802f887",
        },
        Pin {
            name: "fmt444",
            wire_sha256: "92cd4ef1c4be0deab9d96fc1ca67809cc95a113db1c85d437983a7d25d3e1828",
            recon_sha256: "2af8a81448a4b2d0898535828cfd741bc99b526c48ffe88f87d39e831914b860",
        },
        Pin {
            name: "piccrop",
            wire_sha256: "9c2fafff19c91ae18abdacae48d1c11f55caf56ea90dc1752bb5b3ed179dac70",
            recon_sha256: "8d0e380cf8b403422ad278c492cdc068b0369b5a0bc5671b811e09693475b30d",
        },
        Pin {
            name: "adaptiveq",
            wire_sha256: "1a597e6859d22d4dc2380fa42ba7884ecec32c91655ce452866345620d8d25a3",
            recon_sha256: "fdca1a271d61b6ba564ef8e4806deee97c949e159ee631a5baebc07b9463a5fe",
        },
        Pin {
            name: "ratecontrol",
            wire_sha256: "88fb6e139e93f7c33b9e51015ebdf3773cb7736b8350be1742b3a3ba3e1e64d0",
            recon_sha256: "02cfff982e12628d28801dfb664dfd5b55075d09b58d194ed87214e6489c4433",
        },
        Pin {
            name: "dupframes",
            wire_sha256: "633065f337f5b3b10490596e629773c2a210c0fe2f3602a71271876eb72169de",
            recon_sha256: "5b028a045aae96cf0ec0de40ade6a39c1669717f637af10aa6b199d519c9002b",
        },
        Pin {
            name: "scenecut",
            wire_sha256: "4b10501b203fa2b96cbd316734c82d5862a49e909934f16c1061bee26525058b",
            recon_sha256: "f232dab8b3d960c525e63314ffcfed129d3cdc12c2d137e6d1296932b68dfbc3",
        },
        Pin {
            name: "goptuned",
            wire_sha256: "74e58d0079731e1290f454155416dc6b34609966149ba0f89db1985cd9ccaa13",
            recon_sha256: "7ef7b0412e29a850339b2402a92d059f5e537056a31504a43316194cf1ec56a5",
        },
        Pin {
            name: "fourmv",
            wire_sha256: "7d85c9ecceaaafd858780d096a8bf28e18e8ddb2f770b3022e177ff977c7bfd1",
            recon_sha256: "279e7ba7aa7cbcccbaf104053139fcad7857c4bbed8ff84c1c6a3890ad432508",
        },
        Pin {
            name: "golden",
            wire_sha256: "e88deb93ccd45f8f46e82b8de257469321c1e162060c48518253edb4923a6a9d",
            recon_sha256: "9602989d73056a34df268bd684d1d514e2eeb274d29a7ce96ff5f1ddb196e21b",
        },
        Pin {
            name: "rcadaptive",
            wire_sha256: "9682c4a124f0b47a32905af791bf4317d4d647ae6171744b744431961dc6728d",
            recon_sha256: "1b290a09ca262aed038878d92da7c7ed37435a066b0da0965ce58dcd7af56db8",
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
            wire_sha256: "89062d65ed876f077f0627fd83140b999ae69be206a45472774caa5a70a184b1",
            recon_sha256: "296b66de497e29ec93d321efeacbb4f8a41b4fa370354c3fb41fc03cd9d5358f",
        },
        Pin {
            name: "lflims0",
            wire_sha256: "62af6a99e486bd2a01ad5984465d7b6deeedf313dd32fe98df8031efc0b1850f",
            recon_sha256: "9c2e7bbb17b018e26716848f6c66b10b860222a174c05345d7313e006955dfdb",
        },
        Pin {
            name: "multiqrange",
            wire_sha256: "73343767adfe4100e9b0c6d90338a0a0eb47e7f76c1ee97f7e51684d5550a00e",
            recon_sha256: "c1999b8602c94e813492708dca1bb379925bdae6ed2aa71704835b26611916d2",
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
