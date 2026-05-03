//! Integration tests against the docs/video/theora/ fixture corpus.
//!
//! Each fixture under `../../docs/video/theora/fixtures/<name>/` ships
//! an `input.ogv` carrying one or more Theora frames in an Ogg
//! container (3 header packets + N data packets), an `expected.yuv`
//! byte-for-byte ground truth at the *visible* picture dimensions
//! (FFmpeg's `-f rawvideo` applies the picture-region crop), a
//! `notes.md` describing the bitstream feature focus, and a
//! `trace.txt` (sometimes gzipped) capturing THEORA_TRACE events from
//! an instrumented vp3.c decode pass.
//!
//! This driver:
//! 1. Reads `input.ogv` and walks its Ogg pages, reassembling Xiph
//!    lacing into one logical packet per `lacing_value < 255`. This
//!    avoids pulling `oxideav-ogg` into the dev-deps for a pure
//!    test-integration crate; the reassembly is byte-trivial.
//! 2. Treats packets[0..3] as the identification + comment + setup
//!    headers, repackaged into Xiph extradata for the decoder's
//!    `CodecParameters::extradata`.
//! 3. Feeds packets[3..] one at a time to the in-tree
//!    [`oxideav_theora::decoder`] via the public
//!    [`make_decoder_for_tests`] factory, draining `receive_frame`
//!    after each packet.
//! 4. Compares each decoded frame's planes against the per-frame slice
//!    of `expected.yuv` (per-plane match-pct + max diff + PSNR).
//!
//! Acceptance:
//! * `Tier::BitExact` — must round-trip exactly. Failure = CI red.
//! * `Tier::ReportOnly` — divergence is logged but the test does NOT
//!   fail. The in-tree decoder uses an integer IDCT but per-MB
//!   prediction + dequantisation paths can drift ±1 against
//!   libtheora at the strong-quant extreme; every fixture starts as
//!   ReportOnly until the maintainer confirms bit-exact behaviour
//!   against the trace.
//! * `Tier::Ignored` — disabled with #[ignore]; for fixtures that
//!   require infrastructure that does not yet exist (e.g. a real
//!   YUV422 Theora encoder, or a VP3 input.ogv).
//!
//! All fixtures start as ReportOnly per the workspace policy in
//! `feedback_no_external_libs.md`: NO external decoder source
//! (libtheora, libavcodec/vp3.c) was consulted while writing this
//! driver — the fixtures are data, the Theora I Specification PDF is
//! the authority.

use std::fs;
use std::path::PathBuf;

use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, TimeBase};
use oxideav_theora::make_decoder_for_tests;

/// Locate `docs/video/theora/fixtures/<name>/`. Tests run with CWD
/// set to the crate root; we walk two levels up to reach the
/// workspace root and then into `docs/`.
fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from("../../docs/video/theora/fixtures").join(name)
}

/// Walk a raw Ogg file byte-by-byte, emitting complete logical
/// packets (after Xiph-style lacing reassembly). Same shape as
/// `collect_packets` in tests/reference_clips.rs — duplicated here
/// because Cargo integration tests are independent compilation units.
/// Good enough for the fixture corpus (single video stream, no
/// multi-stream demultiplexing required).
fn collect_packets(data: &[u8]) -> Vec<Vec<u8>> {
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut buf: Vec<u8> = Vec::new();
    let mut i = 0usize;
    while i + 27 <= data.len() {
        if &data[i..i + 4] != b"OggS" {
            break;
        }
        let n_segs = data[i + 26] as usize;
        if i + 27 + n_segs > data.len() {
            break;
        }
        let lacing = &data[i + 27..i + 27 + n_segs];
        let mut off = i + 27 + n_segs;
        for &lv in lacing {
            if off + lv as usize > data.len() {
                break;
            }
            buf.extend_from_slice(&data[off..off + lv as usize]);
            off += lv as usize;
            if lv < 255 {
                out.push(std::mem::take(&mut buf));
            }
        }
        i = off;
    }
    if !buf.is_empty() {
        out.push(buf);
    }
    out
}

/// Re-pack three Theora header packets as a Xiph-laced extradata
/// blob suitable for `CodecParameters::extradata`.
fn xiph_lace(packets: &[&[u8]]) -> Vec<u8> {
    let n = packets.len();
    let mut out = Vec::new();
    out.push((n - 1) as u8);
    for p in &packets[..n - 1] {
        let mut sz = p.len();
        while sz >= 255 {
            out.push(255);
            sz -= 255;
        }
        out.push(sz as u8);
    }
    for p in packets {
        out.extend_from_slice(p);
    }
    out
}

/// Per-frame, per-plane comparison summary. All counters are in
/// SAMPLES (= bytes for the 8-bit Theora baseline).
#[derive(Default)]
struct FrameDiff {
    y_total: usize,
    y_exact: usize,
    y_max: i32,
    y_sse: f64,
    uv_total: usize,
    uv_exact: usize,
    uv_max: i32,
    uv_sse: f64,
}

impl FrameDiff {
    fn pct(&self) -> f64 {
        let exact = self.y_exact + self.uv_exact;
        let total = self.y_total + self.uv_total;
        if total == 0 {
            0.0
        } else {
            exact as f64 / total as f64 * 100.0
        }
    }

    fn merge(&mut self, other: &FrameDiff) {
        self.y_total += other.y_total;
        self.y_exact += other.y_exact;
        self.y_max = self.y_max.max(other.y_max);
        self.y_sse += other.y_sse;
        self.uv_total += other.uv_total;
        self.uv_exact += other.uv_exact;
        self.uv_max = self.uv_max.max(other.uv_max);
        self.uv_sse += other.uv_sse;
    }
}

fn diff_plane(our: &[u8], refp: &[u8]) -> (usize, usize, i32, f64) {
    let n = our.len().min(refp.len());
    let mut ex = 0usize;
    let mut max = 0i32;
    let mut sse = 0.0f64;
    for i in 0..n {
        let d = (our[i] as i32 - refp[i] as i32).abs();
        if d == 0 {
            ex += 1;
        }
        if d > max {
            max = d;
        }
        sse += (d as f64) * (d as f64);
    }
    (n, ex, max, sse)
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)] // BitExact reserved for promotion as fixtures are confirmed
enum Tier {
    /// Must decode bit-exactly. Test fails on any divergence.
    BitExact,
    /// Decode is permitted to diverge from the reference; per-fixture
    /// stats are logged but the test does not fail.
    ReportOnly,
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
enum ChromaFormat {
    /// 4:2:0 — chroma planes half width × half height of luma.
    Yuv420,
    /// 4:2:2 — chroma planes half width, full height of luma.
    Yuv422,
    /// 4:4:4 — chroma planes full width × full height of luma.
    Yuv444,
}

struct CorpusCase {
    name: &'static str,
    /// Visible width (after picture-region crop). For most fixtures
    /// visible == coded; for picture-region-non-mb-aligned the visible
    /// dims are smaller than the coded 16-multiple dims.
    visible_w: usize,
    visible_h: usize,
    n_frames: usize,
    chroma: ChromaFormat,
    tier: Tier,
}

impl CorpusCase {
    fn frame_bytes(&self) -> usize {
        let y = self.visible_w * self.visible_h;
        match self.chroma {
            ChromaFormat::Yuv420 => {
                y + 2 * (self.visible_w.div_ceil(2) * self.visible_h.div_ceil(2))
            }
            ChromaFormat::Yuv422 => y + 2 * (self.visible_w.div_ceil(2) * self.visible_h),
            ChromaFormat::Yuv444 => 3 * y,
        }
    }

    fn plane_dims(&self, p: usize) -> (usize, usize) {
        if p == 0 {
            return (self.visible_w, self.visible_h);
        }
        match self.chroma {
            ChromaFormat::Yuv420 => (self.visible_w.div_ceil(2), self.visible_h.div_ceil(2)),
            ChromaFormat::Yuv422 => (self.visible_w.div_ceil(2), self.visible_h),
            ChromaFormat::Yuv444 => (self.visible_w, self.visible_h),
        }
    }
}

struct DecodeReport {
    per_frame: Vec<Result<FrameDiff, String>>,
    /// First fatal error (if any) — recorded for the report banner.
    fatal: Option<String>,
    /// Total number of visible frames the decoder produced (may differ
    /// from `case.n_frames` if the stream stops early or a packet errors).
    visible_produced: usize,
}

fn decode_fixture(case: &CorpusCase) -> Option<DecodeReport> {
    let dir = fixture_dir(case.name);
    let in_path = dir.join("input.ogv");
    let yuv_path = dir.join("expected.yuv");
    let trace_path = dir.join("trace.txt");
    let trace_gz_path = dir.join("trace.txt.gz");
    let ogv = match fs::read(&in_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "skip {}: missing {} ({e}). docs/ corpus is in the workspace \
                 umbrella repo; the standalone crate checkout has no fixtures.",
                case.name,
                in_path.display()
            );
            return None;
        }
    };
    let yuv_ref = match fs::read(&yuv_path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skip {}: missing {} ({e})", case.name, yuv_path.display());
            return None;
        }
    };
    eprintln!(
        "fixture {}: ogv={} bytes, expected.yuv={} bytes, trace={}",
        case.name,
        ogv.len(),
        yuv_ref.len(),
        if trace_path.exists() {
            trace_path.display().to_string()
        } else {
            trace_gz_path.display().to_string()
        }
    );

    let pkts = collect_packets(&ogv);
    if pkts.len() < 4 {
        return Some(DecodeReport {
            per_frame: vec![Err(format!(
                "{}: insufficient packets after Ogg de-lacing ({} < 4 = 3 headers + 1 frame)",
                case.name,
                pkts.len()
            ))],
            fatal: Some("insufficient packets".to_string()),
            visible_produced: 0,
        });
    }

    let frame_size = case.frame_bytes();
    assert_eq!(
        yuv_ref.len(),
        case.n_frames * frame_size,
        "fixture {} expected.yuv size mismatch (have {} bytes, expected {} = {} frames * {} \
         [{}x{} {:?}])",
        case.name,
        yuv_ref.len(),
        case.n_frames * frame_size,
        case.n_frames,
        frame_size,
        case.visible_w,
        case.visible_h,
        case.chroma,
    );

    // Build extradata from the three Theora header packets.
    let hdrs = [&pkts[0][..], &pkts[1][..], &pkts[2][..]];
    let extradata = xiph_lace(&hdrs);
    let mut params = CodecParameters::video(CodecId::new("theora"));
    params.extradata = extradata;
    let mut dec = match make_decoder_for_tests(&params) {
        Ok(d) => d,
        Err(e) => {
            return Some(DecodeReport {
                per_frame: vec![Err(format!(
                    "{}: make_decoder_for_tests failed: {e:?}",
                    case.name
                ))],
                fatal: Some(format!("make_decoder failed: {e:?}")),
                visible_produced: 0,
            });
        }
    };

    let mut per_frame: Vec<Result<FrameDiff, String>> = Vec::with_capacity(case.n_frames);
    let mut fatal: Option<String> = None;
    let mut visible_idx = 0usize;

    for (pkt_idx, pkt) in pkts.iter().enumerate().skip(3) {
        let p = Packet::new(0u32, TimeBase::new(1, 25), pkt.clone());
        match dec.send_packet(&p) {
            Ok(()) => {}
            Err(e) => {
                let msg = format!("packet {pkt_idx}: send_packet: {e:?}");
                if fatal.is_none() {
                    fatal = Some(msg.clone());
                }
                per_frame.push(Err(msg));
                continue;
            }
        }
        loop {
            match dec.receive_frame() {
                Ok(Frame::Video(vf)) => {
                    if visible_idx >= case.n_frames {
                        // Decoder produced more visible frames than the
                        // reference. Record but do not compare.
                        visible_idx += 1;
                        continue;
                    }
                    let ref_off = visible_idx * frame_size;
                    let ref_slice = &yuv_ref[ref_off..ref_off + frame_size];
                    // Slice the reference into planes by walking
                    // plane_dims() in order.
                    let mut ref_off_within = 0usize;
                    let mut diff = FrameDiff::default();
                    let mut size_mismatch: Option<String> = None;
                    for p_idx in 0..3 {
                        let (pw, ph) = case.plane_dims(p_idx);
                        let plane_bytes = pw * ph;
                        let ref_plane = &ref_slice[ref_off_within..ref_off_within + plane_bytes];
                        ref_off_within += plane_bytes;
                        let our_plane = match vf.planes.get(p_idx) {
                            Some(pl) => pl.data.as_slice(),
                            None => {
                                size_mismatch = Some(format!(
                                    "visible {visible_idx}: decoder produced {} planes, \
                                     reference expects 3",
                                    vf.planes.len(),
                                ));
                                break;
                            }
                        };
                        if our_plane.len() != plane_bytes {
                            size_mismatch = Some(format!(
                                "visible {visible_idx} plane {p_idx}: our len {}, \
                                 expected {plane_bytes} (= {pw}x{ph} samples)",
                                our_plane.len(),
                            ));
                            break;
                        }
                        let (n, ex, mx, sse) = diff_plane(our_plane, ref_plane);
                        if p_idx == 0 {
                            diff.y_total += n;
                            diff.y_exact += ex;
                            diff.y_max = diff.y_max.max(mx);
                            diff.y_sse += sse;
                        } else {
                            diff.uv_total += n;
                            diff.uv_exact += ex;
                            diff.uv_max = diff.uv_max.max(mx);
                            diff.uv_sse += sse;
                        }
                    }
                    if let Some(msg) = size_mismatch {
                        per_frame.push(Err(msg));
                    } else {
                        per_frame.push(Ok(diff));
                    }
                    visible_idx += 1;
                }
                Ok(_) => continue,
                Err(Error::NeedMore) => break,
                Err(e) => {
                    let msg = format!("visible {visible_idx}: receive_frame: {e:?}");
                    if fatal.is_none() {
                        fatal = Some(msg.clone());
                    }
                    per_frame.push(Err(msg));
                    break;
                }
            }
        }
    }

    Some(DecodeReport {
        per_frame,
        fatal,
        visible_produced: visible_idx,
    })
}

/// PSNR in dB given a sum-of-squared-errors and a sample count
/// (8-bit Theora — peak = 255).
fn psnr(sse: f64, n: usize) -> f64 {
    if n == 0 || sse == 0.0 {
        return 120.0;
    }
    let mse = sse / n as f64;
    10.0 * (255.0_f64 * 255.0 / mse).log10()
}

fn evaluate(case: &CorpusCase) {
    let report = match decode_fixture(case) {
        Some(r) => r,
        None => return, // missing fixture — already logged
    };

    let mut agg = FrameDiff::default();
    let mut errors: Vec<String> = Vec::new();
    for (i, r) in report.per_frame.iter().enumerate() {
        match r {
            Ok(d) => {
                eprintln!(
                    "  frame {i}: Y {}/{} exact (max diff {}, PSNR {:.2} dB), \
                     UV {}/{} exact (max diff {}, PSNR {:.2} dB), pct={:.2}%",
                    d.y_exact,
                    d.y_total,
                    d.y_max,
                    psnr(d.y_sse, d.y_total),
                    d.uv_exact,
                    d.uv_total,
                    d.uv_max,
                    psnr(d.uv_sse, d.uv_total),
                    d.pct()
                );
                agg.merge(d);
            }
            Err(e) => {
                eprintln!("  frame {i}: ERROR {e}");
                errors.push(format!("frame {i}: {e}"));
            }
        }
    }

    let pct = agg.pct();
    eprintln!(
        "[{:?}] {}: aggregate {}/{} exact ({pct:.2}%), \
         Y max diff {} (PSNR {:.2} dB), \
         UV max diff {} (PSNR {:.2} dB), \
         visible_produced={}/{}{}",
        case.tier,
        case.name,
        agg.y_exact + agg.uv_exact,
        agg.y_total + agg.uv_total,
        agg.y_max,
        psnr(agg.y_sse, agg.y_total),
        agg.uv_max,
        psnr(agg.uv_sse, agg.uv_total),
        report.visible_produced,
        case.n_frames,
        match &report.fatal {
            Some(f) => format!(", first_fatal=\"{f}\""),
            None => String::new(),
        }
    );

    match case.tier {
        Tier::BitExact => {
            assert!(
                errors.is_empty(),
                "{}: {} frame errors prevented bit-exact comparison: {:?}",
                case.name,
                errors.len(),
                errors
            );
            assert_eq!(
                agg.y_exact + agg.uv_exact,
                agg.y_total + agg.uv_total,
                "{}: not bit-exact (Y max diff {}, UV max diff {}; {:.4}% match)",
                case.name,
                agg.y_max,
                agg.uv_max,
                pct
            );
        }
        Tier::ReportOnly => {
            // Don't fail. The eprintln output above is the report.
            // TODO(theora-corpus): tighten to BitExact once the
            // underlying decoder gap for this fixture is closed.
            let _ = pct;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-fixture tests
// ---------------------------------------------------------------------------
//
// All fixtures start as ReportOnly. As the in-tree Theora decoder
// closes the relevant gap, individual cases should be promoted to
// BitExact.
//
// Trace files (referenced in `decode_fixture`'s eprintln! header)
// live alongside each fixture and capture the THEORA_TRACE event
// sequence emitted by an instrumented vp3.c decode pass — useful for
// diffing against our own decoder trace if/when divergence
// localisation is needed.

/// Smallest fixture: single 32x32 keyframe, all four MBs INTRA.
/// (Named tiny-i-only-16x16 to mirror the VP8 corpus, but vp3.c
/// enforces visible_width >= 18 so the smallest practical Theora
/// fixture is 32x32.)
#[test]
fn corpus_tiny_i_only_16x16() {
    evaluate(&CorpusCase {
        name: "tiny-i-only-16x16",
        visible_w: 32,
        visible_h: 32,
        n_frames: 1,
        chroma: ChromaFormat::Yuv420,
        tier: Tier::ReportOnly,
    });
}

/// Bitstream version 0x030201 (Theora 3.2.1) — confirms the decoder
/// reads the version field and enables picture-region offsets,
/// per-stream loop-filter limits, per-stream quant tables, and up to
/// 3 QPIs per frame.
#[test]
fn corpus_bitstream_version_3_2_1() {
    evaluate(&CorpusCase {
        name: "bitstream-version-3.2.1",
        visible_w: 32,
        visible_h: 32,
        n_frames: 1,
        chroma: ChromaFormat::Yuv420,
        tier: Tier::ReportOnly,
    });
}

/// One KEY frame followed by one INTER. Single-reference inter is
/// the canonical motion-compensation smoke test (LAST reference,
/// half-pel MC, golden-frame update path).
#[test]
fn corpus_i_frame_then_p_frame_64x64() {
    evaluate(&CorpusCase {
        name: "i-frame-then-p-frame-64x64",
        visible_w: 64,
        visible_h: 64,
        n_frames: 2,
        chroma: ChromaFormat::Yuv420,
        tier: Tier::ReportOnly,
    });
}

/// `-g 1` = every frame is a keyframe. All four 32x32 frames decode
/// independently, frame_type=I throughout.
#[test]
fn corpus_keyframe_interval_1() {
    evaluate(&CorpusCase {
        name: "keyframe-interval-1",
        visible_w: 32,
        visible_h: 32,
        n_frames: 4,
        chroma: ChromaFormat::Yuv420,
        tier: Tier::ReportOnly,
    });
}

/// `-g 30` = typical GOP layout. The 6-frame stream has a single
/// keyframe at idx=0 followed by 5 inter frames — exercises
/// LAST/LAST2 management across multiple inter frames in a row.
#[test]
fn corpus_keyframe_interval_30() {
    evaluate(&CorpusCase {
        name: "keyframe-interval-30",
        visible_w: 32,
        visible_h: 32,
        n_frames: 6,
        chroma: ChromaFormat::Yuv420,
        tier: Tier::ReportOnly,
    });
}

/// Pseudo-monochrome via gray->yuv420p conversion. Theora has no
/// dedicated monochrome bit; chroma planes are flat 0x80. Useful as
/// a sanity check that the chroma decoder doesn't introduce drift on
/// uniform input.
#[test]
fn corpus_monochrome_via_zero_chroma() {
    evaluate(&CorpusCase {
        name: "monochrome-via-zero-chroma",
        visible_w: 64,
        visible_h: 64,
        n_frames: 2,
        chroma: ChromaFormat::Yuv420,
        tier: Tier::ReportOnly,
    });
}

/// Visible 26x18 over coded 32x32 — Theora's picture-region offsets
/// (PICX=0, PICY=14) carry a non-MB-aligned crop. expected.yuv is at
/// VISIBLE dimensions (the in-tree decoder applies the crop in
/// `crop_to_picture`); 702 bytes total = 26*18 + 13*9*2.
#[test]
fn corpus_picture_region_non_mb_aligned() {
    evaluate(&CorpusCase {
        name: "picture-region-non-mb-aligned",
        visible_w: 26,
        visible_h: 18,
        n_frames: 1,
        chroma: ChromaFormat::Yuv420,
        tier: Tier::ReportOnly,
    });
}

/// `-q:v 10` → libtheora QI=63, the *weakest* quantisation index
/// (`ac_scale=10`, `filter_limit=0`). The decoder's loop filter is
/// bypassed (vp3.c sets skip_loop_filter=1) — useful to isolate
/// IDCT + dequant paths from deblocking.
#[test]
fn corpus_q_high() {
    evaluate(&CorpusCase {
        name: "q-high",
        visible_w: 64,
        visible_h: 64,
        n_frames: 2,
        chroma: ChromaFormat::Yuv420,
        tier: Tier::ReportOnly,
    });
}

/// `-q:v 0` → libtheora QI=0, the *strongest* quantisation index
/// (`ac_scale=365`). Verifies decoder behaviour at the high-quant
/// extreme.
#[test]
fn corpus_q_low() {
    evaluate(&CorpusCase {
        name: "q-low",
        visible_w: 64,
        visible_h: 64,
        n_frames: 2,
        chroma: ChromaFormat::Yuv420,
        tier: Tier::ReportOnly,
    });
}

/// Mid-range quant (`-q:v 3` → qi=18, ac_scale=160) with three QPIs
/// (18 / 8 / 27) — exercises the `nqps=3` per-block QPI selection
/// path introduced in Theora 3.2.0.
#[test]
fn corpus_quant_table_custom() {
    evaluate(&CorpusCase {
        name: "quant-table-custom",
        visible_w: 32,
        visible_h: 32,
        n_frames: 1,
        chroma: ChromaFormat::Yuv420,
        tier: Tier::ReportOnly,
    });
}

// --- Large-frame / negative / out-of-scope ---

/// 1920x1080 visible (coded 1920x1088), 2 frames at HD resolution.
/// expected.yuv is NOT shipped (would be ~6 MB); only its SHA-256 is
/// recorded in notes.md. We can still drive the decoder against
/// input.ogv to surface large-frame errors / timeouts even without
/// the reference bytes — but this driver requires expected.yuv to
/// score, so the test is currently `#[ignore]`. Promote when a
/// SHA-only mode is added (see TODO in decode_fixture).
#[test]
#[ignore = "expected.yuv not stored (~6 MB, sha-only); driver needs a sha-only mode \
            before this can run. See docs/video/theora/fixtures/dimensions-1080p-very-short/"]
fn corpus_dimensions_1080p_very_short() {
    evaluate(&CorpusCase {
        name: "dimensions-1080p-very-short",
        visible_w: 1920,
        visible_h: 1080,
        n_frames: 2,
        chroma: ChromaFormat::Yuv420,
        tier: Tier::ReportOnly,
    });
}

/// libtheora 1.x can't actually emit yuv422p (it auto-converts or
/// errors out). The fixture ships only `attempted_input.ogv` +
/// `encoder_log.txt` — a real 4:2:2 Theora encoder is out of scope
/// for the corpus. The fixture exists to document the negative; this
/// test is `#[ignore]` so it doesn't pollute the matrix.
#[test]
#[ignore = "libtheora 1.x can't emit yuv422p; no expected.yuv. \
            See docs/video/theora/fixtures/multiple-coded-frames-yuv422/notes.md"]
fn corpus_multiple_coded_frames_yuv422() {
    evaluate(&CorpusCase {
        name: "multiple-coded-frames-yuv422",
        visible_w: 64,
        visible_h: 64,
        n_frames: 2,
        chroma: ChromaFormat::Yuv422,
        tier: Tier::ReportOnly,
    });
}

/// The Theora and VP3 bitstreams share one decoder file (vp3.c with
/// `s->theora` switch) but VP3 has no actively-maintained CLI
/// encoder, so the fixture intentionally ships only `notes.md` (no
/// input). Documented here so the matrix stays complete.
#[test]
#[ignore = "VP3 has no actively-maintained encoder; fixture is documentation-only. \
            See docs/video/theora/fixtures/vp3-compat-decode/notes.md"]
fn corpus_vp3_compat_decode() {
    let dir = fixture_dir("vp3-compat-decode");
    let _ = fs::read(dir.join("notes.md"));
}
