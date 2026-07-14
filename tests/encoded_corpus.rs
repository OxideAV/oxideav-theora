//! Pinned self-encoded corpus (round 413).
//!
//! Eleven deterministic encoder scenarios — the same family that was
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

/// The eleven-pin corpus. `CORPUS_PRINT=1 cargo test --test
/// encoded_corpus -- --nocapture` prints the current `Pin` lines for a
/// deliberate re-pin.
#[test]
fn encoded_corpus_digests_are_stable() {
    let cid = || CodecId::new(THEORA_CODEC_ID);

    const PINS: [Pin; 11] = [
        Pin {
            name: "basic420",
            wire_sha256: "dc711a7e953898769035d18ebcd2cb565facc19b39fc446191e414b00b4c4d19",
            recon_sha256: "c1816e32c5e1e1d3c07bae535f67793026ea3e15d584f352dd7c4526fcafac88",
        },
        Pin {
            name: "fmt422",
            wire_sha256: "8a88f6f48e4a63f5876bff3a88406b7e9e8699f76068044ad1cfb4fe39f1d2e1",
            recon_sha256: "6505afaf546466c3c2af32982b4faffb52d9099856a5c0c553b09dfad6db48d5",
        },
        Pin {
            name: "fmt444",
            wire_sha256: "3f32c3ace7ec11926cbe7bf66a6d9d2df238f5eb9eb66e17873f819b3dd86bc6",
            recon_sha256: "e5cf415015a5bdac7cf59d54abdef53578b32ab538c1c877fab012c52ef6b0e2",
        },
        Pin {
            name: "piccrop",
            wire_sha256: "70d09f102d8651711a3c9eca6901850547d4837c60eab1d15262bb5202243bee",
            recon_sha256: "a2990ed8a3eda9722ffa8a365f97ac914e968df92d3eb465057141ea5744d78e",
        },
        Pin {
            name: "adaptiveq",
            wire_sha256: "abe8e0c4d72eaa8b24386c8e43ed0b0c35bdd9ef0b1a74be7629f7c32bc93878",
            recon_sha256: "19edf8f0011dca44d6dea825d814be7f93f0225b8fb85abbe9b061f91d0dc9cb",
        },
        Pin {
            name: "ratecontrol",
            wire_sha256: "6361d1ef26954d6111d77d1e0d0391cb168dab1b518e9d55bd9430418ad88c86",
            recon_sha256: "eddc6c3aa6a85a8a4e81b826f4d76eaed131c687bab150ddaa15d951a1e8cb0e",
        },
        Pin {
            name: "dupframes",
            wire_sha256: "e47b138ce1c1bda82eed258c93c0a390fcc5f3f282c6d1e8ebd876e14c45a145",
            recon_sha256: "980a6e4e7f780cc1b96e51e292f82b77054cce325c2f1d32061fc8a91d9abf1f",
        },
        Pin {
            name: "scenecut",
            wire_sha256: "76781d2717018fbedc3771433d74a8911725181bcec45d58a1d8837864a1bda1",
            recon_sha256: "42d02b500787046e6ed8a3c100c96d980cbfb8deba0551e3ae0e5f9ab3c361a9",
        },
        Pin {
            name: "goptuned",
            wire_sha256: "9bae7964b412a9278416dd0981fd2db0010d1d81ebfe656ff5c2b6b3c6417211",
            recon_sha256: "b180da8390be222a7dd755c686b81970d93a0d83ee9c6df7df85ca16f26ad231",
        },
        Pin {
            name: "fourmv",
            wire_sha256: "68af6993968be068245e39db18e805d0e9f929ac7a0d0de7ea0d4a2e828b8eab",
            recon_sha256: "d9554244df6ac57fcdb990010f4cdacc73b1134952deda67bd090615f5056b2e",
        },
        Pin {
            name: "golden",
            wire_sha256: "92c6a89a9d3e0b721236649ad01090a6d88091494b98fde3fb9cd12c75607d4b",
            recon_sha256: "18ee6baa0788b715b7a216d4b6fa0c95e2c9baafcbd50506f11effbd593caade",
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
