//! # oxideav-theora
//!
//! Pure-Rust Theora video codec — clean-room implementation.
//!
//! ## Status — 2026-05-24 (round 6 of clean-room rebuild)
//!
//! Round 1 landed the **identification header** parser per Theora I
//! Specification §6.1 (common header) + §6.2 (identification header).
//! Round 2 added the **comment header** parser per §6.3.
//! Round 3 landed the **setup header packet entrypoint** per §6.4.5
//! step 1 plus the **MSb-first bit reader** mandated by §5.2.
//! Round 4 landed the **VP3 hardcoded scale / limit tables** from
//! Appendix B.2 + B.3 of the spec — `LFLIMS_VP3` (`[u8; 64]`,
//! Appendix B.2), `ACSCALE_VP3` and `DCSCALE_VP3` (`[u16; 64]` each,
//! Appendix B.3) — and reshaped [`TheoraSetupHeader`] to expose
//! [`TheoraSetupHeader::loop_filter_limits`] / `ac_scale` / `dc_scale`
//! as plain `[u8; 64]` / `[u16; 64]` fields. The VP3 defaults
//! constructor [`TheoraSetupHeader::vp3_defaults`] returns the
//! Appendix-B-typed tables — usable directly to decode the
//! `vp3-compat-decode` fixture's pre-3.2.0 bitstreams once the frame
//! pipeline lands. Round 5 landed the **§6.4.2 Quantization Parameters
//! Decode** procedure as the standalone
//! [`decode_quantization_parameters`] entry point. Round 6 lands the
//! **§6.4.3 Computing a Quantization Matrix** procedure as
//! [`compute_quantization_matrix`], which interpolates a 64-entry
//! [`QuantizationMatrix`] for a `(qti, pli, qi)` selector from a
//! [`QuantizationParameters`].
//!
//! Bitstream-version-3.2.0+ streams override the Appendix-B defaults
//! via the §6.4.1 / §6.4.2 procedures the setup-header body decode
//! consumes — that body is still blocked by the §6.4.1 spec gap (see
//! below) and [`parse_setup_header`] continues to surface
//! [`Error::SetupHeaderBodyNotImplemented`]. §6.4.3 is unblocked by
//! that gap because it operates purely on the §6.4.2 outputs.
//!
//! * [`decode_identification_header`] returns a typed
//!   [`TheoraIdentHeader`] describing every field declared in
//!   Figure 6.2.
//! * [`parse_comment_header`] returns a typed
//!   [`TheoraCommentHeader`] with the decoded vendor string and the
//!   list of `KEY=value` user comments.
//! * [`parse_setup_header`] validates §6.4.5 step 1 (the common
//!   header carrying the `0x82`+"theora" identifier) and returns
//!   [`Error::SetupHeaderBodyNotImplemented`] for the body decode.
//! * [`TheoraSetupHeader::vp3_defaults`] constructs the Appendix B
//!   VP3 fallback for streams that declare `version < 0x030200`
//!   (alpha2 / VP3-compatibility decode, §B.1 first bullet).
//! * [`decode_quantization_parameters`] returns a typed
//!   [`QuantizationParameters`] from a §6.4.2 setup-header payload.
//! * [`compute_quantization_matrix`] interpolates a typed
//!   [`QuantizationMatrix`] per §6.4.3 from those parameters.
//!
//! ## Clean-room provenance
//!
//! Source material limited to `docs/video/theora/Theora.pdf` (Xiph
//! Theora I Specification) and the fixture corpus under
//! `docs/video/theora/fixtures/`. No libtheora, no FFmpeg vp3.c, no
//! theora-rs.
//!
//! ## Known spec gap — §6.4.1 procedure body
//!
//! The §6.4.1 Loop Filter Limit Table Decode section in
//! `Theora.pdf` declares its inputs/outputs (`LFLIMS`: 64-element 7-bit
//! array; `NBITS`: 3-bit) and ends with the sentence "It is decoded as
//! follows:" — but the numbered procedure steps that should follow are
//! **absent** from the PDF as published. The next page begins
//! immediately with "VP3 Compatibility" / §6.4.2. Round 4 therefore
//! continues to defer the bitstream `LFLIMS` decode (the
//! `parse_setup_header` body path) and ships the Appendix-B-typed VP3
//! fallback table as a workaround for `version < 0x030200` streams.
//! For `version >= 0x030200` streams a per-stream LFLIMS is mandated
//! by §6.4.1; until the docs collaborator recovers the numbered
//! procedure body, the per-stream decode remains blocked.
//! See [`Error::SetupHeaderBodyNotImplemented`] for the entrypoint's
//! delegation behaviour.

#![warn(missing_debug_implementations)]
#![deny(unsafe_code)]

use oxideav_core::RuntimeContext;

/// Crate-local error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Packet was shorter than the field being read.
    TruncatedHeader {
        /// Field name being read when the buffer ran out.
        field: &'static str,
    },
    /// Common header type byte (§6.1 step 1) did not match the expected
    /// `0x80`. Returned when [`decode_identification_header`] is called
    /// on a comment (`0x81`) or setup (`0x82`) packet, or on a video
    /// data packet (high bit clear).
    BadHeaderType {
        /// The header-type byte actually present.
        got: u8,
    },
    /// The six bytes following the header-type byte were not the ASCII
    /// `"theora"` sync token mandated by §6.1 step 2.
    BadMagic,
    /// `VMAJ` was not 3, so the stream is not decodable per §6.2
    /// step 2.
    UnsupportedMajorVersion {
        /// The major version reported by the bitstream.
        vmaj: u8,
    },
    /// `VMIN` was not 2, so the stream is not decodable per §6.2
    /// step 3.
    UnsupportedMinorVersion {
        /// The minor version reported by the bitstream.
        vmin: u8,
    },
    /// `FMBW` or `FMBH` was zero, violating §6.2 steps 5–6 which
    /// require both to be greater than zero.
    ZeroMacroblockDimension {
        /// Which of the two dimensions failed the check.
        which: MacroblockDimension,
    },
    /// `PICW` exceeded `FMBW * 16`, violating §6.2 step 7.
    PictureWidthOutOfRange {
        /// Reported picture width.
        picw: u32,
        /// `FMBW * 16` (frame width in pixels).
        coded_w: u32,
    },
    /// `PICH` exceeded `FMBH * 16`, violating §6.2 step 8.
    PictureHeightOutOfRange {
        /// Reported picture height.
        pich: u32,
        /// `FMBH * 16` (frame height in pixels).
        coded_h: u32,
    },
    /// `PICX + PICW` would extend past the coded frame width.
    PictureXOutOfRange {
        /// `PICX`.
        picx: u8,
        /// `PICW`.
        picw: u32,
        /// `FMBW * 16`.
        coded_w: u32,
    },
    /// `PICY + PICH` would extend past the coded frame height.
    PictureYOutOfRange {
        /// `PICY`.
        picy: u8,
        /// `PICH`.
        pich: u32,
        /// `FMBH * 16`.
        coded_h: u32,
    },
    /// `FRN` or `FRD` was zero, violating §6.2 steps 11–12.
    ZeroFrameRate {
        /// Which of the two frame-rate fields was zero.
        which: FrameRateField,
    },
    /// `PF` (pixel format) was the reserved value 1, so the stream is
    /// not decodable per §6.2 step 19.
    ReservedPixelFormat,
    /// The 3 reserved bits at the end of the identification header
    /// were not all zero, violating §6.2 step 20.
    NonZeroReservedBits {
        /// The reserved bit pattern.
        bits: u8,
    },
    /// A length field inside the comment header (vendor length,
    /// comment count, or per-comment length) declared more octets than
    /// the remaining packet contained — see §6.3.1 / §6.3.2.
    CommentLengthOverflow {
        /// Field name being read (e.g. `"vendor"`, `"comment"`).
        field: &'static str,
        /// Length value pulled from the wire.
        len: u32,
        /// Number of octets still available in the packet body.
        remaining: usize,
    },
    /// The vendor string or a `COMMENTS[i]` payload was not valid
    /// UTF-8. §6.3.3 says the comment value is "encoded as a UTF-8
    /// string"; the vendor string is described as a "vector" but is
    /// also UTF-8 in every libtheora-emitted stream we observe and the
    /// reference implementations expose it as a C string.
    CommentNotUtf8 {
        /// Which vector failed UTF-8 decoding.
        field: CommentField,
    },
    /// The setup-header body (§6.4.1 LFLIMS through §6.4.4 Huffman
    /// tables) is not yet implemented. Returned by
    /// [`parse_setup_header`] after the common-header check
    /// succeeds. See the crate-level "Known spec gap" notice for
    /// the underlying reason: §6.4.1's numbered procedure steps are
    /// absent from the spec PDF.
    SetupHeaderBodyNotImplemented,
    /// `NBMS` (number of base matrices, §6.4.2 step 5) exceeded the
    /// spec maximum of 384. The spec says "NBMS MUST be no greater
    /// than 384".
    TooManyBaseMatrices {
        /// The decoded `NBMS` value (already incremented by one per
        /// step 5).
        nbms: u32,
    },
    /// A `QRBMIS` base-matrix index read in §6.4.2 step 7(a)ivC was
    /// greater than or equal to `NBMS`, which the spec declares
    /// makes the stream undecodable ("If this is greater than or
    /// equal to NBMS, stop. The stream is undecodable.").
    BaseMatrixIndexOutOfRange {
        /// The offending `QRBMIS` value.
        bmi: u32,
        /// `NBMS` (number of base matrices).
        nbms: u32,
    },
    /// The accumulated quant-range size `qi` overshot 63 in §6.4.2
    /// step 7(a)ivI ("If qi is greater than 63, stop. The stream is
    /// undecodable."). The sum of all range sizes for a (qti, pli)
    /// pair MUST land exactly on 63.
    QuantRangeOverflow {
        /// The accumulated `qi` value that overshot 63.
        qi: u32,
    },
    /// A `qti` (quantization type index) argument to
    /// [`compute_quantization_matrix`] was outside the `0..=1` range
    /// mandated by Table 3.1 (§6.4.3 declares `qti` as a 1-bit field).
    QuantTypeIndexOutOfRange {
        /// The offending `qti` value.
        qti: usize,
    },
    /// A `pli` (color-plane index) argument to
    /// [`compute_quantization_matrix`] was outside the `0..=2` range
    /// mandated by Table 2.1 (§6.4.3 declares `pli` as a 2-bit field).
    QuantPlaneIndexOutOfRange {
        /// The offending `pli` value.
        pli: usize,
    },
    /// A `qi` (quantization index) argument to
    /// [`compute_quantization_matrix`] was outside the `0..=63` range
    /// (§6.4.3 declares `qi` as a 6-bit field).
    QuantIndexOutOfRange {
        /// The offending `qi` value.
        qi: usize,
    },
    /// The crate has not yet implemented this surface (planned for a
    /// later round).
    NotImplemented,
}

/// Identifies which UTF-8 vector in the comment header triggered an
/// [`Error::CommentNotUtf8`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommentField {
    /// The single vendor string (Figure 6.4, first vector).
    Vendor,
    /// One of the per-comment `KEY=value` vectors. `index` matches the
    /// `ci` loop variable in §6.3.2.
    Comment {
        /// Zero-based index of the offending comment.
        index: u32,
    },
}

impl core::fmt::Display for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Error::TruncatedHeader { field } => {
                write!(
                    f,
                    "oxideav-theora: identification header truncated while reading {field}"
                )
            }
            Error::BadHeaderType { got } => write!(
                f,
                "oxideav-theora: expected identification header type 0x80, got 0x{got:02x}"
            ),
            Error::BadMagic => write!(
                f,
                "oxideav-theora: missing 'theora' magic after header-type byte"
            ),
            Error::UnsupportedMajorVersion { vmaj } => {
                write!(f, "oxideav-theora: VMAJ={vmaj}, expected 3")
            }
            Error::UnsupportedMinorVersion { vmin } => {
                write!(f, "oxideav-theora: VMIN={vmin}, expected 2")
            }
            Error::ZeroMacroblockDimension { which } => write!(
                f,
                "oxideav-theora: {which:?} macroblock dimension was zero (§6.2 forbids)"
            ),
            Error::PictureWidthOutOfRange { picw, coded_w } => {
                write!(f, "oxideav-theora: PICW={picw} exceeds FMBW*16={coded_w}")
            }
            Error::PictureHeightOutOfRange { pich, coded_h } => {
                write!(f, "oxideav-theora: PICH={pich} exceeds FMBH*16={coded_h}")
            }
            Error::PictureXOutOfRange {
                picx,
                picw,
                coded_w,
            } => write!(
                f,
                "oxideav-theora: PICX={picx} + PICW={picw} exceeds FMBW*16={coded_w}"
            ),
            Error::PictureYOutOfRange {
                picy,
                pich,
                coded_h,
            } => write!(
                f,
                "oxideav-theora: PICY={picy} + PICH={pich} exceeds FMBH*16={coded_h}"
            ),
            Error::ZeroFrameRate { which } => write!(
                f,
                "oxideav-theora: frame-rate {which:?} was zero (§6.2 forbids)"
            ),
            Error::ReservedPixelFormat => write!(
                f,
                "oxideav-theora: PF=1 is reserved and not decodable (§6.2 step 19)"
            ),
            Error::NonZeroReservedBits { bits } => write!(
                f,
                "oxideav-theora: trailing reserved bits = 0b{bits:03b}, expected 0"
            ),
            Error::CommentLengthOverflow {
                field,
                len,
                remaining,
            } => write!(
                f,
                "oxideav-theora: comment header {field} length {len} exceeds remaining {remaining} octets"
            ),
            Error::CommentNotUtf8 { field } => write!(
                f,
                "oxideav-theora: comment header {field:?} is not valid UTF-8"
            ),
            Error::SetupHeaderBodyNotImplemented => write!(
                f,
                "oxideav-theora: setup-header common header validated; body (§6.4.1–§6.4.4) deferred to later clean-room round"
            ),
            Error::TooManyBaseMatrices { nbms } => write!(
                f,
                "oxideav-theora: NBMS={nbms} exceeds the §6.4.2 maximum of 384"
            ),
            Error::BaseMatrixIndexOutOfRange { bmi, nbms } => write!(
                f,
                "oxideav-theora: QRBMIS base-matrix index {bmi} >= NBMS={nbms} (§6.4.2 step 7(a)ivC: undecodable)"
            ),
            Error::QuantRangeOverflow { qi } => write!(
                f,
                "oxideav-theora: quant-range sizes summed to qi={qi} > 63 (§6.4.2 step 7(a)ivI: undecodable)"
            ),
            Error::QuantTypeIndexOutOfRange { qti } => write!(
                f,
                "oxideav-theora: qti={qti} out of range 0..=1 (§6.4.3 / Table 3.1)"
            ),
            Error::QuantPlaneIndexOutOfRange { pli } => write!(
                f,
                "oxideav-theora: pli={pli} out of range 0..=2 (§6.4.3 / Table 2.1)"
            ),
            Error::QuantIndexOutOfRange { qi } => write!(
                f,
                "oxideav-theora: qi={qi} out of range 0..=63 (§6.4.3)"
            ),
            Error::NotImplemented => write!(
                f,
                "oxideav-theora: surface not implemented in current clean-room round"
            ),
        }
    }
}

impl std::error::Error for Error {}

/// Identifies which of `FMBW` / `FMBH` triggered a zero-dimension
/// error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroblockDimension {
    /// `FMBW` — frame width in macroblocks.
    Width,
    /// `FMBH` — frame height in macroblocks.
    Height,
}

/// Identifies which of `FRN` / `FRD` triggered a zero-rate error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameRateField {
    /// `FRN` — frame-rate numerator.
    Numerator,
    /// `FRD` — frame-rate denominator.
    Denominator,
}

/// Pixel format from §6.2 step 19 / Table 6.4.
///
/// Value 1 is reserved and rejected at parse time, so it is not
/// representable here.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// `PF=0`. 4:2:0 chroma subsampling (one chroma sample per 2×2
    /// luma block). The only format libtheora 1.x will emit.
    Yuv420 = 0,
    /// `PF=2`. 4:2:2 chroma subsampling (one chroma sample per 1×2
    /// luma column).
    Yuv422 = 2,
    /// `PF=3`. 4:4:4 — chroma sampled at full resolution.
    Yuv444 = 3,
}

/// Color space from §6.2 step 15 / Table 6.3.
///
/// Values 3..=255 are reserved; we accept them as
/// [`ColorSpace::Reserved`] rather than failing, because the spec says
/// a decoder *MAY* refuse such a stream — it does not require it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    /// `CS=0`. Color space was not available to the encoder; the
    /// application may supply one externally.
    Undefined,
    /// `CS=1`. ITU-R Rec. 470M (NTSC primaries).
    Rec470M,
    /// `CS=2`. ITU-R Rec. 470BG (PAL/SECAM primaries).
    Rec470Bg,
    /// `CS` was in the reserved range 3..=255. Carried through so
    /// applications can decide whether to honour or reject it.
    Reserved(u8),
}

impl ColorSpace {
    fn from_byte(b: u8) -> Self {
        match b {
            0 => ColorSpace::Undefined,
            1 => ColorSpace::Rec470M,
            2 => ColorSpace::Rec470Bg,
            other => ColorSpace::Reserved(other),
        }
    }
}

/// Parsed Theora identification header, per Figure 6.2.
///
/// Field names match the spec mnemonics one-for-one. All multi-byte
/// integers are stored in their natural width (not padded out to 24 or
/// 32 bits in cases where the spec reserves extra bits for octet
/// alignment) — for example `picw` is a `u32` even though only 20 bits
/// of its 24-bit on-wire field are meaningful.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TheoraIdentHeader {
    /// `VMAJ` — major version (always 3 for Theora I).
    pub vmaj: u8,
    /// `VMIN` — minor version (always 2 for Theora I).
    pub vmin: u8,
    /// `VREV` — revision; libtheora 1.x emits 1 (i.e. Theora 3.2.1).
    pub vrev: u8,
    /// `FMBW` — coded frame width in macroblocks (always > 0).
    pub fmbw: u16,
    /// `FMBH` — coded frame height in macroblocks (always > 0).
    pub fmbh: u16,
    /// `PICW` — visible (picture) region width in pixels.
    pub picw: u32,
    /// `PICH` — visible (picture) region height in pixels.
    pub pich: u32,
    /// `PICX` — picture region X offset within the coded frame.
    pub picx: u8,
    /// `PICY` — picture region Y offset within the coded frame. Per
    /// §6.2 step 10 this measures the lower-left corner; consumers that
    /// work in a top-left coordinate system must flip via
    /// `coded_h - pich - picy`.
    pub picy: u8,
    /// `FRN` — frame-rate numerator (always > 0).
    pub frn: u32,
    /// `FRD` — frame-rate denominator (always > 0).
    pub frd: u32,
    /// `PARN` — pixel-aspect-ratio numerator. Zero (with `pard=0`)
    /// means aspect ratio was not declared.
    pub parn: u32,
    /// `PARD` — pixel-aspect-ratio denominator.
    pub pard: u32,
    /// `CS` — color space (Table 6.3).
    pub cs: ColorSpace,
    /// `NOMBR` — nominal bitrate in bits per second; the saturation
    /// value `2^24 - 1` represents "≥ 2^24-1". Zero means the encoder
    /// chose not to declare a rate.
    pub nombr: u32,
    /// `QUAL` — 6-bit relative quality hint.
    pub qual: u8,
    /// `KFGSHIFT` — 5-bit shift used to split the Ogg granule position
    /// into key-frame index and offset.
    pub kfgshift: u8,
    /// `PF` — pixel format (Table 6.4).
    pub pf: PixelFormat,
}

impl TheoraIdentHeader {
    /// Combined 24-bit version field `(VMAJ << 16) | (VMIN << 8) | VREV`.
    ///
    /// libtheora 1.x always emits `0x030201`; values `>= 0x030200` are
    /// the alpha3+ feature set required by §6.2.
    pub fn version(&self) -> u32 {
        ((self.vmaj as u32) << 16) | ((self.vmin as u32) << 8) | (self.vrev as u32)
    }

    /// Coded frame width in pixels (`FMBW * 16`).
    pub fn coded_width(&self) -> u32 {
        (self.fmbw as u32) * 16
    }

    /// Coded frame height in pixels (`FMBH * 16`).
    pub fn coded_height(&self) -> u32 {
        (self.fmbh as u32) * 16
    }

    /// Total macroblock count (`NMBS` per §6.2 step 23).
    pub fn nmbs(&self) -> u32 {
        (self.fmbw as u32) * (self.fmbh as u32)
    }

    /// Total super-block count (`NSBS` per Table 6.5), as a function
    /// of `PF`.
    pub fn nsbs(&self) -> u32 {
        let w = self.fmbw as u32;
        let h = self.fmbh as u32;
        let half_w = w.div_ceil(2);
        let half_h = h.div_ceil(2);
        let quarter_w = w.div_ceil(4);
        let quarter_h = h.div_ceil(4);
        match self.pf {
            // 4:2:0: one luma SB plane + two chroma SB planes at 1/4 area.
            PixelFormat::Yuv420 => half_w * half_h + 2 * quarter_w * quarter_h,
            // 4:2:2: luma SB plane + two chroma SB planes at 1/2 area
            // (chroma keeps full vertical resolution).
            PixelFormat::Yuv422 => half_w * half_h + 2 * quarter_w * half_h,
            // 4:4:4: three SB planes of equal size.
            PixelFormat::Yuv444 => 3 * half_w * half_h,
        }
    }

    /// Total block count (`NBS` per Table 6.6).
    pub fn nbs(&self) -> u64 {
        let mb = (self.fmbw as u64) * (self.fmbh as u64);
        match self.pf {
            PixelFormat::Yuv420 => 6 * mb,
            PixelFormat::Yuv422 => 8 * mb,
            PixelFormat::Yuv444 => 12 * mb,
        }
    }
}

/// Decode a Theora identification header from `packet`.
///
/// Implements the procedure in §6.1 (common header) + §6.2
/// (identification header) verbatim. `packet` must contain the whole
/// header packet starting from the 0x80 header-type byte — i.e. the
/// payload of the first Ogg packet on a Theora stream, with Ogg
/// framing already stripped.
///
/// Returns [`Error::BadHeaderType`] if called on a comment (`0x81`) or
/// setup (`0x82`) packet, and [`Error::TruncatedHeader`] if the
/// payload is shorter than the 42 bytes Figure 6.2 mandates.
pub fn decode_identification_header(packet: &[u8]) -> Result<TheoraIdentHeader, Error> {
    let mut r = Reader::new(packet);

    // --- §6.1: common header.
    let header_type = r.read_u8("header_type")?;
    if header_type & 0x80 == 0 {
        // High bit clear → video data packet, not a header.
        return Err(Error::BadHeaderType { got: header_type });
    }
    if header_type != 0x80 {
        // High bit set but wrong type code → this is a header, but a
        // different one (comment / setup / reserved).
        return Err(Error::BadHeaderType { got: header_type });
    }
    // 't','h','e','o','r','a'
    const MAGIC: [u8; 6] = [0x74, 0x68, 0x65, 0x6f, 0x72, 0x61];
    for expected in MAGIC {
        let got = r.read_u8("magic")?;
        if got != expected {
            return Err(Error::BadMagic);
        }
    }

    // --- §6.2 step 2.
    let vmaj = r.read_u8("VMAJ")?;
    if vmaj != 3 {
        return Err(Error::UnsupportedMajorVersion { vmaj });
    }
    // step 3.
    let vmin = r.read_u8("VMIN")?;
    if vmin != 2 {
        return Err(Error::UnsupportedMinorVersion { vmin });
    }
    // step 4 — VREV > 1 is forward-compatible per spec; do not reject.
    let vrev = r.read_u8("VREV")?;

    // step 5.
    let fmbw = r.read_u16_be("FMBW")?;
    if fmbw == 0 {
        return Err(Error::ZeroMacroblockDimension {
            which: MacroblockDimension::Width,
        });
    }
    // step 6.
    let fmbh = r.read_u16_be("FMBH")?;
    if fmbh == 0 {
        return Err(Error::ZeroMacroblockDimension {
            which: MacroblockDimension::Height,
        });
    }

    // steps 7–8 — 24-bit reads even though only 20 bits are meaningful.
    let picw = r.read_u24_be("PICW")?;
    let coded_w = (fmbw as u32) * 16;
    if picw > coded_w {
        return Err(Error::PictureWidthOutOfRange { picw, coded_w });
    }
    let pich = r.read_u24_be("PICH")?;
    let coded_h = (fmbh as u32) * 16;
    if pich > coded_h {
        return Err(Error::PictureHeightOutOfRange { pich, coded_h });
    }

    // steps 9–10.
    let picx = r.read_u8("PICX")?;
    if (picx as u32) + picw > coded_w {
        return Err(Error::PictureXOutOfRange {
            picx,
            picw,
            coded_w,
        });
    }
    let picy = r.read_u8("PICY")?;
    if (picy as u32) + pich > coded_h {
        return Err(Error::PictureYOutOfRange {
            picy,
            pich,
            coded_h,
        });
    }

    // steps 11–12.
    let frn = r.read_u32_be("FRN")?;
    if frn == 0 {
        return Err(Error::ZeroFrameRate {
            which: FrameRateField::Numerator,
        });
    }
    let frd = r.read_u32_be("FRD")?;
    if frd == 0 {
        return Err(Error::ZeroFrameRate {
            which: FrameRateField::Denominator,
        });
    }

    // steps 13–14 — PARN/PARD may both be zero (== "not declared").
    let parn = r.read_u24_be("PARN")?;
    let pard = r.read_u24_be("PARD")?;

    // step 15.
    let cs = ColorSpace::from_byte(r.read_u8("CS")?);

    // step 16 — NOMBR is informational, no validation.
    let nombr = r.read_u24_be("NOMBR")?;

    // steps 17–20 — packed bitfield (QUAL:6, KFGSHIFT:5, PF:2, Res:3).
    // 6 + 5 + 2 + 3 = 16 bits = 2 octets, read big-endian.
    let packed = r.read_u16_be("QUAL/KFGSHIFT/PF/Res")?;
    let qual = ((packed >> 10) & 0x3f) as u8;
    let kfgshift = ((packed >> 5) & 0x1f) as u8;
    let pf_raw = ((packed >> 3) & 0x03) as u8;
    let reserved = (packed & 0x07) as u8;
    let pf = match pf_raw {
        0 => PixelFormat::Yuv420,
        1 => return Err(Error::ReservedPixelFormat),
        2 => PixelFormat::Yuv422,
        3 => PixelFormat::Yuv444,
        _ => unreachable!("pf_raw masked to 2 bits"),
    };
    if reserved != 0 {
        return Err(Error::NonZeroReservedBits { bits: reserved });
    }

    Ok(TheoraIdentHeader {
        vmaj,
        vmin,
        vrev,
        fmbw,
        fmbh,
        picw,
        pich,
        picx,
        picy,
        frn,
        frd,
        parn,
        pard,
        cs,
        nombr,
        qual,
        kfgshift,
        pf,
    })
}

/// Parsed Theora comment header, per Figure 6.4 / §6.3.2.
///
/// The header carries a single vendor string and a list of user
/// comments. Each user comment is a `KEY=value` vector; §6.3.3
/// constrains the key (case-insensitive ASCII, no `=` byte) and lets
/// the value be any UTF-8 string up to the per-vector length limit.
///
/// We surface the comments split into `(key, value)` tuples for
/// convenience. A vector that does **not** contain an `=` byte is
/// preserved with an empty value (per §6.3.3 the field name "is
/// immediately followed by ASCII 0x3D ('=')"; we keep malformed
/// vectors in the parsed output rather than reject because real-world
/// files occasionally carry vendor-format strings without `=`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TheoraCommentHeader {
    /// Vendor string from Figure 6.4 (the first vector of the comment
    /// header). libtheora-via-FFmpeg writes the muxer name here
    /// (e.g. `"Lavf62.13.102"`).
    pub vendor: String,
    /// User comments from §6.3.3. Each entry is a parsed
    /// `(key, value)` pair. Keys are returned in the case they
    /// appeared on the wire; per §6.3.3 they must be compared
    /// case-insensitively.
    pub comments: Vec<(String, String)>,
}

impl TheoraCommentHeader {
    /// Look up the first comment whose key matches `name`
    /// case-insensitively, per §6.3.3 ("the field name is case-
    /// insensitive").
    ///
    /// Returns the value of the first matching entry, or `None` if no
    /// comment has that key. Comments may legally repeat; callers
    /// that need every value should iterate
    /// [`TheoraCommentHeader::comments`] directly.
    pub fn lookup(&self, name: &str) -> Option<&str> {
        self.comments
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }
}

/// Decode a Theora comment header from `packet`.
///
/// Implements §6.3.1 (length decode) and §6.3.2 (comment header
/// decode) verbatim. The packet body uses the Vorbis-compatible
/// memory layout from §6.3.1: a 7-byte common header (`0x81` +
/// `"theora"`), followed by the vendor-string length as 4 LE octets,
/// the vendor string itself, a 4-octet LE comment count, and then
/// each comment as a 4-octet LE length plus the comment vector.
///
/// `packet` must contain the whole header packet starting from the
/// `0x81` header-type byte — i.e. the payload of the second Ogg
/// packet on a Theora stream, with Ogg framing already stripped.
///
/// The comment header packet is allowed to contain trailing bytes
/// after the last comment vector per §6.3.2 ("the comment header
/// comprises the entirety of the second header packet"); we accept
/// trailing bytes silently to keep the parser robust against
/// real-world streams.
pub fn parse_comment_header(packet: &[u8]) -> Result<TheoraCommentHeader, Error> {
    let mut r = Reader::new(packet);

    // --- §6.1 common header (called out by §6.3.2 step 1).
    let header_type = r.read_u8("header_type")?;
    if header_type & 0x80 == 0 {
        return Err(Error::BadHeaderType { got: header_type });
    }
    if header_type != 0x81 {
        return Err(Error::BadHeaderType { got: header_type });
    }
    const MAGIC: [u8; 6] = [0x74, 0x68, 0x65, 0x6f, 0x72, 0x61];
    for expected in MAGIC {
        let got = r.read_u8("magic")?;
        if got != expected {
            return Err(Error::BadMagic);
        }
    }

    // --- §6.3.2 step 2: vendor-string length (§6.3.1 little-endian
    // decode).
    let vendor_len = r.read_u32_le("vendor_len")?;
    let vendor_bytes = r.read_octets("vendor", vendor_len)?;
    let vendor = match core::str::from_utf8(vendor_bytes) {
        Ok(s) => s.to_owned(),
        Err(_) => {
            return Err(Error::CommentNotUtf8 {
                field: CommentField::Vendor,
            });
        }
    };

    // --- §6.3.2 step 5: NCOMMENTS.
    let ncomments = r.read_u32_le("ncomments")?;
    let mut comments = Vec::with_capacity(ncomments.min(64) as usize);

    // --- §6.3.2 step 7: per-comment loop.
    for ci in 0..ncomments {
        let comment_bytes = {
            let len = r.read_u32_le("comment_len")?;
            r.read_octets("comment", len)?
        };
        let text = match core::str::from_utf8(comment_bytes) {
            Ok(s) => s,
            Err(_) => {
                return Err(Error::CommentNotUtf8 {
                    field: CommentField::Comment { index: ci },
                });
            }
        };
        // §6.3.3: split on the first `=`. A vector without `=` is
        // unusual but preserved with an empty value.
        let (key, value) = match text.split_once('=') {
            Some((k, v)) => (k.to_owned(), v.to_owned()),
            None => (text.to_owned(), String::new()),
        };
        comments.push((key, value));
    }

    Ok(TheoraCommentHeader { vendor, comments })
}

/// `LFLIMS` — VP3 hardcoded loop-filter limit values, transcribed
/// from `Theora.pdf` Appendix B.2 ("Loop Filter Limit Values"). The
/// 64 entries are indexed by `qi` (the quantization index, range
/// 0..=63) and each value is the deblocking-filter strength `L`
/// described in §7.10 ("Loop Filtering").
///
/// Theora bitstreams with `VMAJ.VMIN.VREV < 3.2.0` ("VP3 compatible",
/// §B.1) MUST use this table directly because they predate the
/// per-stream LFLIMS carried in the setup header. Streams with
/// `>= 3.2.0` override it via §6.4.1 (procedure body currently absent
/// from the spec — see the crate-level "Known spec gap" notice).
///
/// Source: `Theora.pdf` Appendix B.2.
pub const LFLIMS_VP3: [u8; 64] = [
    30, 25, 20, 20, 15, 15, 14, 14, //
    13, 13, 12, 12, 11, 11, 10, 10, //
    9, 9, 8, 8, 7, 7, 7, 7, //
    6, 6, 6, 6, 5, 5, 5, 5, //
    4, 4, 4, 4, 3, 3, 3, 3, //
    2, 2, 2, 2, 2, 2, 2, 2, //
    0, 0, 0, 0, 0, 0, 0, 0, //
    0, 0, 0, 0, 0, 0, 0, 0,
];

/// `ACSCALE` — VP3 hardcoded AC dequantization scale values,
/// transcribed from `Theora.pdf` Appendix B.3 ("Quantization
/// Parameters"). The 64 entries are indexed by `qi` and multiplied
/// against the base matrix per §6.4.3 to compute the actual
/// per-coefficient quantization step.
///
/// Theora bitstreams with `version < 0x030200` use this table
/// directly; later streams override it via §6.4.2 steps 1–2.
///
/// Source: `Theora.pdf` Appendix B.3.
pub const ACSCALE_VP3: [u16; 64] = [
    500, 450, 400, 370, 340, 310, 285, 265, //
    245, 225, 210, 195, 185, 180, 170, 160, //
    150, 145, 135, 130, 125, 115, 110, 107, //
    100, 96, 93, 89, 85, 82, 75, 74, //
    70, 68, 64, 60, 57, 56, 52, 50, //
    49, 45, 44, 43, 40, 38, 37, 35, //
    33, 32, 30, 29, 28, 25, 24, 22, //
    21, 19, 18, 17, 15, 13, 12, 10,
];

/// `DCSCALE` — VP3 hardcoded DC dequantization scale values,
/// transcribed from `Theora.pdf` Appendix B.3 ("Quantization
/// Parameters"). Same indexing and override rules as
/// [`ACSCALE_VP3`]; later streams override via §6.4.2 steps 3–4.
///
/// Source: `Theora.pdf` Appendix B.3.
pub const DCSCALE_VP3: [u16; 64] = [
    220, 200, 190, 180, 170, 170, 160, 160, //
    150, 150, 140, 140, 130, 130, 120, 120, //
    110, 110, 100, 100, 90, 90, 90, 80, //
    80, 80, 70, 70, 70, 60, 60, 60, //
    60, 50, 50, 50, 50, 40, 40, 40, //
    40, 40, 30, 30, 30, 30, 30, 30, //
    30, 20, 20, 20, 20, 20, 20, 20, //
    20, 10, 10, 10, 10, 10, 10, 10,
];

/// Parsed Theora setup header, per §6.4.5.
///
/// The Theora setup header (`HEADERTYPE=0x82`) carries five logical
/// payloads:
///
/// 1. The loop-filter limit table (`LFLIMS`, §6.4.1).
/// 2. The AC/DC scale tables (`ACSCALE`, `DCSCALE`, §6.4.2 steps 1–4).
/// 3. The base matrices (`NBMS`, `BMS`, §6.4.2 steps 5–6).
/// 4. The quant-range index tables (`NQRS`, `QRSIZES`, `QRBMIS`,
///    §6.4.2 step 7).
/// 5. The DCT-token Huffman tables (`HTS`, §6.4.4 — an 80-element
///    array of Huffman tables with up to 32 entries each).
///
/// **Round 4 carries the Appendix B fallback tables only.**
/// [`parse_setup_header`] still surfaces
/// [`Error::SetupHeaderBodyNotImplemented`] (§6.4.1 procedure body
/// gap), but the value layer is now in place: callers handling
/// `vp3-compat-decode` style streams (`version < 0x030200`) can
/// build a usable [`TheoraSetupHeader`] via
/// [`TheoraSetupHeader::vp3_defaults`]. Base matrices, NQRS /
/// QRSIZES / QRBMIS, and the Huffman tables are deferred to round 5.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TheoraSetupHeader {
    /// `LFLIMS` (§6.4.1) — 64-element array of 7-bit loop-filter
    /// limit values indexed by `qi`. Each entry is in the range
    /// 0..=127 (7-bit unsigned per the §6.4.1 output-parameters
    /// table). VP3-compatible streams populate this from
    /// [`LFLIMS_VP3`]; later streams override per §6.4.1's
    /// procedure (currently blocked by spec gap).
    pub loop_filter_limits: [u8; 64],
    /// `ACSCALE` (§6.4.2 steps 1–2) — 64-element array of 16-bit AC
    /// dequantization scale values, indexed by `qi`. VP3-compatible
    /// streams populate this from [`ACSCALE_VP3`]; later streams
    /// override per §6.4.2.
    pub ac_scale: [u16; 64],
    /// `DCSCALE` (§6.4.2 steps 3–4) — 64-element array of 16-bit DC
    /// dequantization scale values, indexed by `qi`. VP3-compatible
    /// streams populate this from [`DCSCALE_VP3`]; later streams
    /// override per §6.4.2.
    pub dc_scale: [u16; 64],
}

impl TheoraSetupHeader {
    /// Construct a [`TheoraSetupHeader`] populated with the VP3
    /// hardcoded tables from `Theora.pdf` Appendix B.2 + B.3. Use
    /// this for streams whose identification header declares a
    /// `version < 0x030200` (alpha2 / VP3-compatible bitstreams):
    /// per §B.1, those streams do not carry a setup-header LFLIMS
    /// or ACSCALE/DCSCALE override.
    ///
    /// For `version >= 0x030200` streams, the spec requires the
    /// setup-header §6.4.1 / §6.4.2 procedures to be applied; until
    /// the §6.4.1 procedure-body spec gap is closed,
    /// [`parse_setup_header`] returns
    /// [`Error::SetupHeaderBodyNotImplemented`] rather than these
    /// defaults.
    pub fn vp3_defaults() -> Self {
        Self {
            loop_filter_limits: LFLIMS_VP3,
            ac_scale: ACSCALE_VP3,
            dc_scale: DCSCALE_VP3,
        }
    }
}

/// The quantization parameters decoded from the setup header per
/// §6.4.2 ("Quantization Parameters Decode").
///
/// These five logical payloads together drive dequantization
/// (§7.9.2): the AC/DC scale tables select a per-`qi` step multiplier,
/// while the base matrices, referenced through the quant-range tables,
/// supply the per-coefficient shape that §6.4.3 interpolates into a
/// full quantization matrix.
///
/// Field semantics follow the §6.4.2 output-parameters table verbatim:
///
/// * `qti` — the quantization type index (`0` = INTRA DC/AC matrices,
///   `1` = INTER), per Table 3.1. There are two types.
/// * `pli` — the color-plane index (`0` = luma, `1` = Cb, `2` = Cr),
///   per Table 2.1. There are three planes.
///
/// Both indices appear in the 2×3 / 2×3×N quant-range arrays.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QuantizationParameters {
    /// `ACSCALE` (§6.4.2 steps 1–2) — 64-element array of AC scale
    /// values, one per `qi`. Each is a 16-bit unsigned integer.
    pub ac_scale: [u16; 64],
    /// `DCSCALE` (§6.4.2 steps 3–4) — 64-element array of DC scale
    /// values, one per `qi`. Each is a 16-bit unsigned integer.
    pub dc_scale: [u16; 64],
    /// `NBMS` (§6.4.2 step 5) — the number of base matrices. Already
    /// incremented per the spec ("NBMS the value decoded, plus one")
    /// and validated `<= 384`.
    pub num_base_matrices: u16,
    /// `BMS` (§6.4.2 step 6) — an `NBMS × 64` array of base matrices.
    /// Only the first `num_base_matrices` rows are meaningful; the
    /// remaining rows (up to the 384-row capacity) are left zeroed.
    /// Each entry is an 8-bit unsigned integer.
    pub base_matrices: Vec<[u8; 64]>,
    /// `NQRS` (§6.4.2 step 7) — a `2 × 3` array (indexed
    /// `[qti][pli]`) giving the number of quant ranges defined for
    /// each quantization type and color plane. At most 63.
    pub num_quant_ranges: [[u8; 3]; 2],
    /// `QRSIZES` (§6.4.2 step 7) — a `2 × 3 × 63` array of quant-range
    /// sizes. Only the first `num_quant_ranges[qti][pli]` entries of
    /// each `[qti][pli]` row are used.
    pub quant_range_sizes: [[[u8; 63]; 3]; 2],
    /// `QRBMIS` (§6.4.2 step 7) — a `2 × 3 × 64` array of base-matrix
    /// indices, one for each end of each quant range. Only the first
    /// `num_quant_ranges[qti][pli] + 1` entries of each `[qti][pli]`
    /// row are used.
    pub quant_range_base_matrix_indices: [[[u16; 64]; 3]; 2],
}

/// Decode a Theora setup header from `packet`.
///
/// Rounds 3+4 implement only §6.4.5 step 1: the common-header check
/// (the `0x82` header-type byte followed by the ASCII `"theora"`
/// sync token mandated by §6.1).
///
/// `packet` must contain the whole header packet starting from the
/// `0x82` header-type byte — i.e. the payload of the third Ogg
/// packet on a Theora stream, with Ogg framing already stripped.
///
/// Returns:
///
/// * [`Error::BadHeaderType`] if called on an identification (`0x80`)
///   or comment (`0x81`) packet, or on a video data packet (high
///   bit clear).
/// * [`Error::BadMagic`] if the six bytes following the header-type
///   byte are not `"theora"`.
/// * [`Error::TruncatedHeader`] if the packet is shorter than the
///   7-byte common header.
/// * [`Error::SetupHeaderBodyNotImplemented`] after the common
///   header has been validated, while the spec gap on §6.4.1
///   blocks further body decode. The error is "soft" in the sense
///   that the caller has at least verified that the packet looks
///   like a setup header — it just can't be decoded yet. Callers
///   handling `version < 0x030200` (VP3-compatible) streams may
///   substitute [`TheoraSetupHeader::vp3_defaults`] for the missing
///   body; the Appendix B tables apply directly there.
///
/// A later round will replace the `SetupHeaderBodyNotImplemented`
/// return with a fully-populated [`TheoraSetupHeader`] once §6.4.1's
/// procedure body is recovered.
pub fn parse_setup_header(packet: &[u8]) -> Result<TheoraSetupHeader, Error> {
    let mut r = Reader::new(packet);

    // --- §6.1 common header (called out by §6.4.5 step 1).
    let header_type = r.read_u8("header_type")?;
    if header_type & 0x80 == 0 {
        return Err(Error::BadHeaderType { got: header_type });
    }
    if header_type != 0x82 {
        return Err(Error::BadHeaderType { got: header_type });
    }
    const MAGIC: [u8; 6] = [0x74, 0x68, 0x65, 0x6f, 0x72, 0x61];
    for expected in MAGIC {
        let got = r.read_u8("magic")?;
        if got != expected {
            return Err(Error::BadMagic);
        }
    }

    // --- §6.4.5 steps 2–4 would follow here, consuming the body
    // through a `BitReader` (§5.2). Step 2 (§6.4.1 LFLIMS) is
    // currently blocked by a spec gap — see the crate-level "Known
    // spec gap" notice and the comment on
    // `Error::SetupHeaderBodyNotImplemented`. Round 4 ships the
    // Appendix B fallback tables via `TheoraSetupHeader::vp3_defaults`
    // for `version < 0x030200` callers, but
    // `parse_setup_header` itself still requires the §6.4.1 body to
    // be present in the spec; until then the soft sentinel lets
    // callers route on the unparsed / fallback / parsed distinction
    // once a later round closes the gap.
    Err(Error::SetupHeaderBodyNotImplemented)
}

/// `ilog(a)` per the spec's Notation and Conventions section: the
/// minimum number of bits required to store a positive integer `a`
/// in two's-complement notation, or `0` for a non-positive integer.
///
/// `ilog(a) = 0` for `a <= 0`, and `floor(log2(a)) + 1` for `a > 0`.
/// Worked examples from the spec: `ilog(0) = 0`, `ilog(1) = 1`,
/// `ilog(2) = 2`, `ilog(3) = 2`, `ilog(4) = 3`, `ilog(7) = 3`.
///
/// The argument is `i64` so callers can pass `62 - qi` (which can go
/// to `-1` at `qi = 63` in §6.4.2 step 7) without an underflow.
fn ilog(a: i64) -> u32 {
    if a <= 0 {
        0
    } else {
        // floor(log2(a)) + 1; for a u64 that is `64 - leading_zeros`.
        64 - (a as u64).leading_zeros()
    }
}

/// Decode the quantization parameters from the §6.4.2 setup-header
/// payload carried by `bits`.
///
/// `bits` must start at the first bit of the §6.4.2 payload — i.e.
/// immediately after the §6.4.1 LFLIMS table within the setup-header
/// body (§6.4.5 step 3 runs after step 2). This entry point decodes
/// the payload in isolation so it can be exercised independently of
/// the §6.4.1 procedure-body spec gap; once that gap closes,
/// `parse_setup_header` will chain §6.4.1 then call this on the same
/// underlying bit reader.
///
/// Because the §6.4.2 payload is bit-packed and not byte-aligned in a
/// real setup header, this helper is most useful in two situations:
/// (1) a synthesized payload that is byte-aligned at offset 0, and
/// (2) a future caller that hands over a slice starting on the
/// §6.4.2 byte boundary. Sub-byte continuation from §6.4.1 is handled
/// inside `parse_setup_header` once that section lands.
///
/// Returns:
///
/// * [`Error::TruncatedHeader`] if the bitstream runs out before a
///   field is fully read.
/// * [`Error::TooManyBaseMatrices`] if `NBMS > 384` (step 5).
/// * [`Error::BaseMatrixIndexOutOfRange`] if a `QRBMIS` value read in
///   step 7(a)ivC is `>= NBMS`.
/// * [`Error::QuantRangeOverflow`] if the accumulated range sizes
///   overshoot 63 (step 7(a)ivI).
///
/// The procedure transcribes the numbered steps of §6.4.2 of the
/// Xiph Theora I Specification directly.
pub fn decode_quantization_parameters(bits: &[u8]) -> Result<QuantizationParameters, Error> {
    let mut r = BitReader::new(bits);
    decode_quant_params_inner(&mut r)
}

/// Inner §6.4.2 procedure operating on an already-positioned
/// [`BitReader`]. Split out so a future `parse_setup_header` can chain
/// it onto the same reader after §6.4.1 without re-aligning.
fn decode_quant_params_inner(r: &mut BitReader<'_>) -> Result<QuantizationParameters, Error> {
    // Step 1: read 4-bit NBITS, then add one.
    let mut nbits = r.read_bits(4, "ACSCALE NBITS")? + 1;

    // Step 2: read ACSCALE[qi] for qi = 0..=63 as NBITS-bit values.
    let mut ac_scale = [0u16; 64];
    for slot in ac_scale.iter_mut() {
        *slot = r.read_bits(nbits, "ACSCALE")? as u16;
    }

    // Step 3: read 4-bit NBITS, then add one.
    nbits = r.read_bits(4, "DCSCALE NBITS")? + 1;

    // Step 4: read DCSCALE[qi] for qi = 0..=63 as NBITS-bit values.
    let mut dc_scale = [0u16; 64];
    for slot in dc_scale.iter_mut() {
        *slot = r.read_bits(nbits, "DCSCALE")? as u16;
    }

    // Step 5: read 9-bit NBMS, then add one. NBMS MUST be <= 384.
    let nbms = r.read_bits(9, "NBMS")? + 1;
    if nbms > 384 {
        return Err(Error::TooManyBaseMatrices { nbms });
    }

    // Step 6: read BMS[bmi][ci] (8-bit) for bmi = 0..NBMS, ci = 0..=63.
    let mut base_matrices: Vec<[u8; 64]> = Vec::with_capacity(nbms as usize);
    for _bmi in 0..nbms {
        let mut matrix = [0u8; 64];
        for slot in matrix.iter_mut() {
            *slot = r.read_bits(8, "BMS")? as u8;
        }
        base_matrices.push(matrix);
    }

    // Step 7: quant-range tables for each (qti, pli).
    let mut num_quant_ranges = [[0u8; 3]; 2];
    let mut quant_range_sizes = [[[0u8; 63]; 3]; 2];
    let mut quant_range_base_matrix_indices = [[[0u16; 64]; 3]; 2];

    // ilog(NBMS - 1) is the field width of each QRBMIS read.
    let bmi_bits = ilog(nbms as i64 - 1);

    for qti in 0usize..2 {
        for pli in 0usize..3 {
            // Step 7(a)i / 7(a)ii: NEWQR flag.
            let newqr = if qti > 0 || pli > 0 {
                r.read_bits(1, "NEWQR")?
            } else {
                1
            };

            if newqr == 0 {
                // Step 7(a)iii: copy a previously defined set.
                // A / B: RPQR flag.
                let rpqr = if qti > 0 { r.read_bits(1, "RPQR")? } else { 0 };

                // C / D: select source (qtj, plj).
                let (qtj, plj) = if rpqr == 1 {
                    // Same color plane, previous quantization type.
                    (qti - 1, pli)
                } else {
                    // Most recent set defined.
                    ((3 * qti + pli - 1) / 3, (pli + 2) % 3)
                };

                // E / F / G: copy NQRS / QRSIZES / QRBMIS.
                num_quant_ranges[qti][pli] = num_quant_ranges[qtj][plj];
                quant_range_sizes[qti][pli] = quant_range_sizes[qtj][plj];
                quant_range_base_matrix_indices[qti][pli] =
                    quant_range_base_matrix_indices[qtj][plj];
            } else {
                // Step 7(a)iv: define a new set of quant ranges.
                let mut qri = 0usize; // A
                let mut qi = 0i64; // B

                // C: first QRBMIS, range-checked against NBMS.
                let bmi0 = r.read_bits(bmi_bits, "QRBMIS")?;
                if bmi0 >= nbms {
                    return Err(Error::BaseMatrixIndexOutOfRange { bmi: bmi0, nbms });
                }
                quant_range_base_matrix_indices[qti][pli][qri] = bmi0 as u16;

                loop {
                    // D: QRSIZES[qri] = ilog(62 - qi)-bit value + 1.
                    let size = r.read_bits(ilog(62 - qi), "QRSIZES")? + 1;
                    quant_range_sizes[qti][pli][qri] = size as u8;

                    // E: qi += QRSIZES[qri].
                    qi += size as i64;

                    // F: qri += 1.
                    qri += 1;

                    // G: QRBMIS[qri] (NOT range-checked here, per spec).
                    let bmi = r.read_bits(bmi_bits, "QRBMIS")?;
                    quant_range_base_matrix_indices[qti][pli][qri] = bmi as u16;

                    // H: if qi < 63, loop back to D.
                    if qi < 63 {
                        continue;
                    }
                    // I: if qi > 63, undecodable.
                    if qi > 63 {
                        return Err(Error::QuantRangeOverflow { qi: qi as u32 });
                    }
                    // qi == 63: fall through to J.
                    break;
                }

                // J: NQRS[qti][pli] = qri.
                num_quant_ranges[qti][pli] = qri as u8;
            }
        }
    }

    Ok(QuantizationParameters {
        ac_scale,
        dc_scale,
        num_base_matrices: nbms as u16,
        base_matrices,
        num_quant_ranges,
        quant_range_sizes,
        quant_range_base_matrix_indices,
    })
}

/// Minimum quantization value (`QMIN`) for a DCT coefficient, per
/// Table 6.18 of the Theora I Specification (§6.4.3 step 6(b)).
///
/// The minimum depends on the quantization type (`qti`: `0` = intra,
/// `1` = inter) and whether the coefficient is the DC term (`ci == 0`)
/// or an AC term (`ci > 0`):
///
/// | `ci`  | `qti`     | `QMIN` |
/// | ----- | --------- | ------ |
/// | `0`   | 0 (Intra) | 16     |
/// | `> 0` | 0 (Intra) | 8      |
/// | `0`   | 1 (Inter) | 32     |
/// | `> 0` | 1 (Inter) | 16     |
#[inline]
fn qmin_table(qti: usize, ci: usize) -> u32 {
    match (qti, ci) {
        (0, 0) => 16,
        (0, _) => 8,
        (1, 0) => 32,
        (_, _) => 16,
    }
}

/// A single quantization matrix in natural coefficient order, produced
/// by [`compute_quantization_matrix`] (§6.4.3 output `QMAT`).
///
/// Each of the 64 entries is the quantizer for the DCT coefficient at
/// the matching natural-order index (`ci = 0` is the DC term). Values
/// are in the range `1..=4096`: §6.4.3 step 6(e) clamps every entry to
/// `max(QMIN, min(..., 4096))`, and the per-coefficient `QMIN` (Table
/// 6.18) is always `>= 8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantizationMatrix {
    /// The 64 quantizer values, indexed by natural-order coefficient
    /// index `ci` (`0` = DC). Each is in `1..=4096`.
    pub values: [u16; 64],
}

/// Compute a single quantization matrix for a given quantization type,
/// color plane, and quantization index, per §6.4.3 ("Computing a
/// Quantization Matrix") of the Theora I Specification.
///
/// This consumes the [`QuantizationParameters`] produced by
/// [`decode_quantization_parameters`] (§6.4.2) and interpolates a
/// 64-element quantization matrix for the `(qti, pli, qi)` selector.
///
/// * `qti` — quantization type index (Table 3.1): `0` = intra (DC- and
///   AC-quantised keyframe blocks), `1` = inter. Must be `0..=1`.
/// * `pli` — color-plane index (Table 2.1): `0` = luma, `1` = Cb,
///   `2` = Cr. Must be `0..=2`. Selects which `(qti, pli)` quant-range
///   row of the parameters is consulted.
/// * `qi` — the quantization index `0..=63`. Selects the scale entry
///   and locates the quant range that brackets it.
///
/// The procedure (§6.4.3 steps 1–6):
///
/// 1. Locate the quant range `qri` whose cumulative size bounds bracket
///    `qi` (steps 1–3), giving the `[QISTART, QIEND]` end-points.
/// 2. Linearly interpolate between the two base matrices at the range
///    end-points (`bmi = QRBMIS[qri]`, `bmj = QRBMIS[qri + 1]`) for
///    each coefficient (steps 4–6(a)). The interpolation rounds via
///    the `//` operator with `QRSIZES[qri]` added to the numerator.
/// 3. Scale each interpolated base value by `DCSCALE[qi]` (for the DC
///    term) or `ACSCALE[qi]` (for AC terms), divide by 100, multiply by
///    4 to match the DCT output scaling, and clamp to
///    `max(QMIN, min(..., 4096))` (steps 6(b)–6(e)). `QMIN` comes from
///    Table 6.18 ([`qmin_table`]).
///
/// All arithmetic uses non-negative operands, so the spec's `//`
/// (integer division, rounding toward negative infinity for `a >= 0`)
/// reduces to ordinary integer division.
///
/// Returns:
///
/// * [`Error::QuantTypeIndexOutOfRange`] if `qti > 1`.
/// * [`Error::QuantPlaneIndexOutOfRange`] if `pli > 2`.
/// * [`Error::QuantIndexOutOfRange`] if `qi > 63`.
pub fn compute_quantization_matrix(
    params: &QuantizationParameters,
    qti: usize,
    pli: usize,
    qi: usize,
) -> Result<QuantizationMatrix, Error> {
    if qti > 1 {
        return Err(Error::QuantTypeIndexOutOfRange { qti });
    }
    if pli > 2 {
        return Err(Error::QuantPlaneIndexOutOfRange { pli });
    }
    if qi > 63 {
        return Err(Error::QuantIndexOutOfRange { qi });
    }

    let sizes = &params.quant_range_sizes[qti][pli];
    let bmis = &params.quant_range_base_matrix_indices[qti][pli];
    let nqrs = params.num_quant_ranges[qti][pli] as usize;

    // Steps 1–3: find the quant range qri whose cumulative size bounds
    // bracket qi, accumulating QISTART (the cumulative sum up to qri)
    // and QIEND (the cumulative sum through qri). The sum from 0 to -1
    // is defined to be zero.
    //
    // The defined quant ranges partition [0, 63]; their sizes sum to
    // exactly 63 (§6.4.2 step 7(a)ivH/I), so a qi in 0..=63 always
    // lands in some range. We pick the first qri whose right end-point
    // (QIEND) is >= qi; per step 1, when qi lies on a boundary either
    // choice yields the same QMAT.
    let qi = qi as u32;
    let mut qri = 0usize;
    let mut qistart = 0u32;
    let mut qiend = sizes[0] as u32;
    while qri + 1 < nqrs && qiend < qi {
        qistart = qiend;
        qiend += sizes[qri + 1] as u32;
        qri += 1;
    }

    // Step 4 / 5: the base-matrix indices at the two range end-points.
    let bmi = bmis[qri] as usize;
    let bmj = bmis[qri + 1] as usize;
    let bm_lo = &params.base_matrices[bmi];
    let bm_hi = &params.base_matrices[bmj];

    // Denominator for the step 6(a) interpolation: 2 * QRSIZES[qri].
    // QRSIZES values are always >= 1 (§6.4.2 encodes them as value + 1),
    // so this is always >= 2 — no division by zero.
    let qrsize = sizes[qri] as u32;
    let denom = 2 * qrsize;

    let mut values = [0u16; 64];
    // Step 6: for each coefficient ci in 0..=63.
    for (ci, out) in values.iter_mut().enumerate() {
        // 6(a): interpolate the base matrix value BM[ci].
        let bm =
            (2 * (qiend - qi) * bm_lo[ci] as u32 + 2 * (qi - qistart) * bm_hi[ci] as u32 + qrsize)
                / denom;

        // 6(b): minimum quantization value from Table 6.18.
        let qmin = qmin_table(qti, ci);

        // 6(c) / 6(d): DC vs AC scale.
        let qscale = if ci == 0 {
            params.dc_scale[qi as usize] as u32
        } else {
            params.ac_scale[qi as usize] as u32
        };

        // 6(e): QMAT[ci] = max(QMIN, min((QSCALE * BM // 100) * 4, 4096)).
        let scaled = (qscale * bm / 100) * 4;
        *out = qmin.max(scaled.min(4096)) as u16;
    }

    Ok(QuantizationMatrix { values })
}

/// MSb-first bit reader implementing §5.2 of the Theora I
/// Specification.
///
/// Theora differs from Vorbis in bit packing: per §5.2 "the decoder
/// logically unpacks integers by first reading the MSb of a binary
/// integer from the logical bitstream, followed by the next most
/// significant bit, etc." — i.e. each output integer is built MSb
/// first, and within each source byte the MSb is consumed first.
///
/// This is the bit reader that §6.4.1 / §6.4.2 / §6.4.4 will consume
/// once their full decoding procedures are available. It is held
/// crate-private until then; the public API only exposes the byte-
/// aligned parsers ([`decode_identification_header`],
/// [`parse_comment_header`], [`parse_setup_header`] step 1).
///
/// The reader is non-panicking: every read returns `Err` rather than
/// indexing past end-of-buffer. The byte position is advanced
/// automatically as bits are consumed.
#[derive(Debug)]
struct BitReader<'a> {
    buf: &'a [u8],
    /// Byte index of the byte currently being consumed. `bit_pos`
    /// counts down from 7 to 0 within this byte.
    byte_pos: usize,
    /// Bit index within `buf[byte_pos]`. Values 7..=0 correspond to
    /// the MSb..LSb of the current byte; -1 means "advance to the
    /// next byte before the next read". Stored as `i8` (signed)
    /// rather than `u8` so the underflow is cheap to test.
    bit_pos: i8,
}

impl<'a> BitReader<'a> {
    /// Create a reader positioned at the MSb of `buf[0]`.
    fn new(buf: &'a [u8]) -> Self {
        Self {
            buf,
            byte_pos: 0,
            bit_pos: 7,
        }
    }

    /// Read `n` bits (0 ≤ n ≤ 32) as an unsigned integer, MSb first.
    /// Returns [`Error::TruncatedHeader`] if the stream is exhausted
    /// before `n` bits have been read.
    fn read_bits(&mut self, n: u32, field: &'static str) -> Result<u32, Error> {
        debug_assert!(n <= 32, "read_bits called with n > 32");
        let mut value: u32 = 0;
        for _ in 0..n {
            if self.byte_pos >= self.buf.len() {
                return Err(Error::TruncatedHeader { field });
            }
            let bit = (self.buf[self.byte_pos] >> (self.bit_pos as u8)) & 1;
            value = (value << 1) | (bit as u32);
            self.bit_pos -= 1;
            if self.bit_pos < 0 {
                self.bit_pos = 7;
                self.byte_pos += 1;
            }
        }
        Ok(value)
    }
}

/// Big-endian byte-stream cursor. Crate-private; consumers should call
/// [`decode_identification_header`] instead.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Self { buf, pos: 0 }
    }

    fn read_u8(&mut self, field: &'static str) -> Result<u8, Error> {
        if self.pos >= self.buf.len() {
            return Err(Error::TruncatedHeader { field });
        }
        let b = self.buf[self.pos];
        self.pos += 1;
        Ok(b)
    }

    fn read_u16_be(&mut self, field: &'static str) -> Result<u16, Error> {
        if self.pos + 2 > self.buf.len() {
            return Err(Error::TruncatedHeader { field });
        }
        let v = u16::from_be_bytes([self.buf[self.pos], self.buf[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn read_u24_be(&mut self, field: &'static str) -> Result<u32, Error> {
        if self.pos + 3 > self.buf.len() {
            return Err(Error::TruncatedHeader { field });
        }
        let v = ((self.buf[self.pos] as u32) << 16)
            | ((self.buf[self.pos + 1] as u32) << 8)
            | (self.buf[self.pos + 2] as u32);
        self.pos += 3;
        Ok(v)
    }

    fn read_u32_be(&mut self, field: &'static str) -> Result<u32, Error> {
        if self.pos + 4 > self.buf.len() {
            return Err(Error::TruncatedHeader { field });
        }
        let v = u32::from_be_bytes([
            self.buf[self.pos],
            self.buf[self.pos + 1],
            self.buf[self.pos + 2],
            self.buf[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    /// Little-endian u32 reader used by the comment-header parser.
    /// §6.3.1 step 5 explicitly defines the construction as
    /// `LEN0 + (LEN1 << 8) + (LEN2 << 16) + (LEN3 << 24)`, i.e. LE
    /// across 4 octets read in order.
    fn read_u32_le(&mut self, field: &'static str) -> Result<u32, Error> {
        if self.pos + 4 > self.buf.len() {
            return Err(Error::TruncatedHeader { field });
        }
        let v = u32::from_le_bytes([
            self.buf[self.pos],
            self.buf[self.pos + 1],
            self.buf[self.pos + 2],
            self.buf[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    /// Borrow the next `len` octets as a slice. Returns
    /// [`Error::CommentLengthOverflow`] when the wire length exceeds
    /// the packet's remaining bytes.
    fn read_octets(&mut self, field: &'static str, len: u32) -> Result<&'a [u8], Error> {
        let remaining = self.buf.len() - self.pos;
        let len_usize = len as usize;
        if len_usize > remaining {
            return Err(Error::CommentLengthOverflow {
                field,
                len,
                remaining,
            });
        }
        let s = &self.buf[self.pos..self.pos + len_usize];
        self.pos += len_usize;
        Ok(s)
    }
}

/// No-op codec registration. Decoder/encoder wiring will land in a
/// later clean-room round; the identification-header parser is a
/// standalone helper that does not need to be wired into a
/// [`RuntimeContext`] to be useful.
pub fn register(_ctx: &mut RuntimeContext) {}

oxideav_core::register!("theora", register);

#[cfg(test)]
mod tests {
    use super::*;

    /// Bytes extracted from
    /// `docs/video/theora/fixtures/tiny-i-only-16x16/input.ogv`: first
    /// Ogg packet payload, framing already stripped. Coded 32×32,
    /// visible 32×32, libtheora 1.x defaults.
    const TINY_HEADER: [u8; 42] = [
        0x80, 0x74, 0x68, 0x65, 0x6f, 0x72, 0x61, // 0x80 + "theora"
        0x03, 0x02, 0x01, // VMAJ=3 VMIN=2 VREV=1
        0x00, 0x02, // FMBW=2
        0x00, 0x02, // FMBH=2
        0x00, 0x00, 0x20, // PICW=32
        0x00, 0x00, 0x20, // PICH=32
        0x00, // PICX=0
        0x00, // PICY=0
        0x00, 0x00, 0x00, 0x19, // FRN=25
        0x00, 0x00, 0x00, 0x01, // FRD=1
        0x00, 0x00, 0x01, // PARN=1
        0x00, 0x00, 0x01, // PARD=1
        0x00, // CS=0
        0x00, 0x00, 0x00, // NOMBR=0
        0xb0, 0xc0, // QUAL=44 KFGSHIFT=6 PF=0 Res=0
    ];

    /// Bytes from
    /// `docs/video/theora/fixtures/picture-region-non-mb-aligned/input.ogv`.
    /// Coded 32×32, visible 26×18, PICY=14 (lower-left convention).
    const PIC_REGION_HEADER: [u8; 42] = [
        0x80, 0x74, 0x68, 0x65, 0x6f, 0x72, 0x61, 0x03, 0x02, 0x01, 0x00, 0x02, 0x00, 0x02, 0x00,
        0x00, 0x1a, 0x00, 0x00, 0x12, 0x00, 0x0e, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0xb0, 0xc0,
    ];

    /// Bytes from
    /// `docs/video/theora/fixtures/dimensions-1080p-very-short/input.ogv`.
    /// Coded 1920×1088, visible 1920×1080, mb_w=120 / mb_h=68.
    const HD_HEADER: [u8; 42] = [
        0x80, 0x74, 0x68, 0x65, 0x6f, 0x72, 0x61, 0x03, 0x02, 0x01, 0x00, 0x78, 0x00, 0x44, 0x00,
        0x07, 0x80, 0x00, 0x04, 0x38, 0x00, 0x08, 0x00, 0x00, 0x00, 0x19, 0x00, 0x00, 0x00, 0x01,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x7c, 0xc0,
    ];

    #[test]
    fn parses_tiny_fixture_header() {
        let h = decode_identification_header(&TINY_HEADER).expect("tiny header should decode");
        assert_eq!(h.vmaj, 3);
        assert_eq!(h.vmin, 2);
        assert_eq!(h.vrev, 1);
        assert_eq!(h.version(), 0x030201);
        assert_eq!(h.fmbw, 2);
        assert_eq!(h.fmbh, 2);
        assert_eq!(h.coded_width(), 32);
        assert_eq!(h.coded_height(), 32);
        assert_eq!(h.picw, 32);
        assert_eq!(h.pich, 32);
        assert_eq!(h.picx, 0);
        assert_eq!(h.picy, 0);
        assert_eq!(h.frn, 25);
        assert_eq!(h.frd, 1);
        assert_eq!(h.parn, 1);
        assert_eq!(h.pard, 1);
        assert_eq!(h.cs, ColorSpace::Undefined);
        assert_eq!(h.nombr, 0);
        assert_eq!(h.qual, 44);
        assert_eq!(h.kfgshift, 6);
        assert_eq!(h.pf, PixelFormat::Yuv420);
        // Table 6.5 for PF=0: ((2+1)/2)*((2+1)/2) + 2*((2+3)/4)*((2+3)/4)
        // = 1*1 + 2*1*1 = 3 super-blocks.
        assert_eq!(h.nsbs(), 3);
        // Table 6.6: 6 * 2 * 2 = 24 blocks.
        assert_eq!(h.nbs(), 24);
        // NMBS = 2 * 2 = 4.
        assert_eq!(h.nmbs(), 4);
    }

    #[test]
    fn parses_picture_region_offsets() {
        let h = decode_identification_header(&PIC_REGION_HEADER)
            .expect("picture-region header should decode");
        assert_eq!(h.coded_width(), 32);
        assert_eq!(h.coded_height(), 32);
        assert_eq!(h.picw, 26);
        assert_eq!(h.pich, 18);
        assert_eq!(h.picx, 0);
        // Spec stores PICY as a lower-left offset; the fixture's
        // trace records the top-flipped value (14) directly.
        assert_eq!(h.picy, 14);
        // Top-left convention: 32 - 18 - 14 = 0 (visible region sits
        // flush against the top of the coded frame).
        assert_eq!(h.coded_height() - h.pich - h.picy as u32, 0);
    }

    #[test]
    fn parses_hd_fixture_dimensions() {
        let h = decode_identification_header(&HD_HEADER).expect("HD header should decode");
        assert_eq!(h.fmbw, 120);
        assert_eq!(h.fmbh, 68);
        assert_eq!(h.coded_width(), 1920);
        assert_eq!(h.coded_height(), 1088);
        assert_eq!(h.picw, 1920);
        assert_eq!(h.pich, 1080);
        assert_eq!(h.picx, 0);
        assert_eq!(h.picy, 8); // 1088 - 1080 - 8 = 0 top.
        assert_eq!(h.frn, 25);
        assert_eq!(h.frd, 1);
        // PF=0 ⇒ NSBS = ((120+1)/2)*((68+1)/2) + 2*((120+3)/4)*((68+3)/4)
        //              = 60*34 + 2*30*17 = 2040 + 1020 = 3060.
        assert_eq!(h.nsbs(), 3060);
        // NBS = 6 * 120 * 68 = 48960.
        assert_eq!(h.nbs(), 48960);
        assert_eq!(h.nmbs(), 8160);
        assert_eq!(h.qual, 31); // 0x7cc0 >> 10 = 0b011111 = 31.
        assert_eq!(h.kfgshift, 6); // (0x7cc0 >> 5) & 0x1f = 0b00110 = 6.
        assert_eq!(h.pf, PixelFormat::Yuv420);
    }

    #[test]
    fn rejects_comment_header_type() {
        let mut bad = TINY_HEADER;
        bad[0] = 0x81;
        match decode_identification_header(&bad) {
            Err(Error::BadHeaderType { got: 0x81 }) => {}
            other => panic!("expected BadHeaderType(0x81), got {other:?}"),
        }
    }

    #[test]
    fn rejects_video_data_packet() {
        let mut bad = TINY_HEADER;
        bad[0] = 0x00;
        match decode_identification_header(&bad) {
            Err(Error::BadHeaderType { got: 0x00 }) => {}
            other => panic!("expected BadHeaderType(0x00), got {other:?}"),
        }
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bad = TINY_HEADER;
        bad[1] = b'T';
        assert_eq!(decode_identification_header(&bad), Err(Error::BadMagic));
    }

    #[test]
    fn rejects_wrong_major_version() {
        let mut bad = TINY_HEADER;
        bad[7] = 4;
        assert_eq!(
            decode_identification_header(&bad),
            Err(Error::UnsupportedMajorVersion { vmaj: 4 })
        );
    }

    #[test]
    fn rejects_wrong_minor_version() {
        let mut bad = TINY_HEADER;
        bad[8] = 3;
        assert_eq!(
            decode_identification_header(&bad),
            Err(Error::UnsupportedMinorVersion { vmin: 3 })
        );
    }

    #[test]
    fn accepts_future_revision() {
        // §6.2 step 4: VREV > 1 may indicate forward-compatible
        // features but the stream is still decodable.
        let mut h = TINY_HEADER;
        h[9] = 42;
        let parsed = decode_identification_header(&h).expect("VREV>1 must decode");
        assert_eq!(parsed.vrev, 42);
    }

    #[test]
    fn rejects_zero_fmbw() {
        let mut bad = TINY_HEADER;
        bad[10] = 0;
        bad[11] = 0;
        assert_eq!(
            decode_identification_header(&bad),
            Err(Error::ZeroMacroblockDimension {
                which: MacroblockDimension::Width
            })
        );
    }

    #[test]
    fn rejects_zero_fmbh() {
        let mut bad = TINY_HEADER;
        bad[12] = 0;
        bad[13] = 0;
        assert_eq!(
            decode_identification_header(&bad),
            Err(Error::ZeroMacroblockDimension {
                which: MacroblockDimension::Height
            })
        );
    }

    #[test]
    fn rejects_picw_over_coded_width() {
        let mut bad = TINY_HEADER;
        // FMBW=2 -> coded_w=32; set PICW=33.
        bad[14] = 0;
        bad[15] = 0;
        bad[16] = 33;
        assert_eq!(
            decode_identification_header(&bad),
            Err(Error::PictureWidthOutOfRange {
                picw: 33,
                coded_w: 32
            })
        );
    }

    #[test]
    fn rejects_pich_over_coded_height() {
        let mut bad = TINY_HEADER;
        // PICH bytes are at [17..20].
        bad[17] = 0;
        bad[18] = 0;
        bad[19] = 33;
        assert_eq!(
            decode_identification_header(&bad),
            Err(Error::PictureHeightOutOfRange {
                pich: 33,
                coded_h: 32
            })
        );
    }

    #[test]
    fn rejects_picx_plus_picw_overflow() {
        let mut bad = PIC_REGION_HEADER;
        // PICW=26, coded_w=32; PICX is at byte 20. PICX=7 -> 26+7=33 > 32.
        bad[20] = 7;
        // Reset PICY to a value that won't trip its own check.
        bad[21] = 0;
        assert_eq!(
            decode_identification_header(&bad),
            Err(Error::PictureXOutOfRange {
                picx: 7,
                picw: 26,
                coded_w: 32
            })
        );
    }

    #[test]
    fn rejects_picy_plus_pich_overflow() {
        let mut bad = PIC_REGION_HEADER;
        bad[20] = 0;
        bad[21] = 15; // 18 + 15 = 33 > 32
        assert_eq!(
            decode_identification_header(&bad),
            Err(Error::PictureYOutOfRange {
                picy: 15,
                pich: 18,
                coded_h: 32
            })
        );
    }

    #[test]
    fn rejects_zero_frn() {
        let mut bad = TINY_HEADER;
        // FRN bytes at [22..26].
        bad[22] = 0;
        bad[23] = 0;
        bad[24] = 0;
        bad[25] = 0;
        assert_eq!(
            decode_identification_header(&bad),
            Err(Error::ZeroFrameRate {
                which: FrameRateField::Numerator
            })
        );
    }

    #[test]
    fn rejects_zero_frd() {
        let mut bad = TINY_HEADER;
        // FRD bytes at [26..30].
        bad[26..30].fill(0);
        assert_eq!(
            decode_identification_header(&bad),
            Err(Error::ZeroFrameRate {
                which: FrameRateField::Denominator
            })
        );
    }

    #[test]
    fn accepts_zero_pixel_aspect_ratio() {
        // PARN/PARD = 0 ⇒ "not declared", per §6.2 step 14.
        let mut h = TINY_HEADER;
        h[30..36].fill(0);
        let parsed = decode_identification_header(&h).expect("PARN/PARD=0 must decode");
        assert_eq!(parsed.parn, 0);
        assert_eq!(parsed.pard, 0);
    }

    #[test]
    fn parses_color_spaces() {
        let mut h = TINY_HEADER;
        h[36] = 1; // CS=Rec470M
        assert_eq!(
            decode_identification_header(&h).unwrap().cs,
            ColorSpace::Rec470M
        );
        h[36] = 2;
        assert_eq!(
            decode_identification_header(&h).unwrap().cs,
            ColorSpace::Rec470Bg
        );
        h[36] = 99;
        assert_eq!(
            decode_identification_header(&h).unwrap().cs,
            ColorSpace::Reserved(99)
        );
    }

    #[test]
    fn rejects_reserved_pixel_format() {
        let mut bad = TINY_HEADER;
        // Packed bits: QUAL(6)=44 KFGSHIFT(5)=6 PF(2)=1 Res(3)=0.
        // (44<<10)|(6<<5)|(1<<3)|0 = 0xb0c8.
        bad[40] = 0xb0;
        bad[41] = 0xc8;
        assert_eq!(
            decode_identification_header(&bad),
            Err(Error::ReservedPixelFormat)
        );
    }

    #[test]
    fn rejects_nonzero_reserved_bits() {
        let mut bad = TINY_HEADER;
        // Set Res to 0b001.
        bad[41] = 0xc1;
        match decode_identification_header(&bad) {
            Err(Error::NonZeroReservedBits { bits: 1 }) => {}
            other => panic!("expected NonZeroReservedBits, got {other:?}"),
        }
    }

    #[test]
    fn parses_pf_422_and_444() {
        let mut h = TINY_HEADER;
        // PF=2 ⇒ (44<<10)|(6<<5)|(2<<3) = 0xb0d0.
        h[40] = 0xb0;
        h[41] = 0xd0;
        let parsed = decode_identification_header(&h).unwrap();
        assert_eq!(parsed.pf, PixelFormat::Yuv422);
        // Table 6.6: NBS = 8 * 2 * 2 = 32.
        assert_eq!(parsed.nbs(), 32);

        // PF=3 ⇒ (44<<10)|(6<<5)|(3<<3) = 0xb0d8.
        h[41] = 0xd8;
        let parsed = decode_identification_header(&h).unwrap();
        assert_eq!(parsed.pf, PixelFormat::Yuv444);
        assert_eq!(parsed.nbs(), 48);
    }

    #[test]
    fn rejects_truncated_packet() {
        for len in 0..TINY_HEADER.len() {
            match decode_identification_header(&TINY_HEADER[..len]) {
                Err(Error::TruncatedHeader { .. }) => {}
                // BadHeaderType is acceptable for len==0 since the
                // first byte read already fails. Actually no — len==0
                // hits TruncatedHeader on the first read_u8 call.
                Err(Error::BadHeaderType { .. }) => {
                    panic!("len={len} should hit TruncatedHeader before BadHeaderType")
                }
                Err(Error::BadMagic) => {
                    panic!("len={len} should hit TruncatedHeader before BadMagic")
                }
                Ok(_) => panic!("len={len} ought not parse"),
                Err(other) => panic!("len={len} unexpected {other:?}"),
            }
        }
        // Full length must succeed.
        assert!(decode_identification_header(&TINY_HEADER).is_ok());
    }

    #[test]
    fn parses_qual_kfgshift_pf_packed_field_correctly() {
        // QUAL=63, KFGSHIFT=31, PF=3, Res=0:
        //   (63<<10)|(31<<5)|(3<<3) = 0xfc00|0x03e0|0x0018 = 0xfff8.
        let mut h = TINY_HEADER;
        h[40] = 0xff;
        h[41] = 0xf8;
        let parsed = decode_identification_header(&h).unwrap();
        assert_eq!(parsed.qual, 63);
        assert_eq!(parsed.kfgshift, 31);
        assert_eq!(parsed.pf, PixelFormat::Yuv444);

        // QUAL=0, KFGSHIFT=0, PF=0, Res=0 ⇒ 0x0000.
        h[40] = 0x00;
        h[41] = 0x00;
        let parsed = decode_identification_header(&h).unwrap();
        assert_eq!(parsed.qual, 0);
        assert_eq!(parsed.kfgshift, 0);
        assert_eq!(parsed.pf, PixelFormat::Yuv420);
    }

    #[test]
    fn display_error_does_not_panic() {
        // Smoke-test all error variants format without panic.
        let _ = format!("{}", Error::TruncatedHeader { field: "x" });
        let _ = format!("{}", Error::BadHeaderType { got: 0x42 });
        let _ = format!("{}", Error::BadMagic);
        let _ = format!("{}", Error::UnsupportedMajorVersion { vmaj: 4 });
        let _ = format!("{}", Error::UnsupportedMinorVersion { vmin: 3 });
        let _ = format!(
            "{}",
            Error::ZeroMacroblockDimension {
                which: MacroblockDimension::Width
            }
        );
        let _ = format!(
            "{}",
            Error::PictureWidthOutOfRange {
                picw: 33,
                coded_w: 32
            }
        );
        let _ = format!(
            "{}",
            Error::PictureHeightOutOfRange {
                pich: 33,
                coded_h: 32
            }
        );
        let _ = format!(
            "{}",
            Error::PictureXOutOfRange {
                picx: 7,
                picw: 26,
                coded_w: 32
            }
        );
        let _ = format!(
            "{}",
            Error::PictureYOutOfRange {
                picy: 15,
                pich: 18,
                coded_h: 32
            }
        );
        let _ = format!(
            "{}",
            Error::ZeroFrameRate {
                which: FrameRateField::Numerator
            }
        );
        let _ = format!("{}", Error::ReservedPixelFormat);
        let _ = format!("{}", Error::NonZeroReservedBits { bits: 1 });
        let _ = format!(
            "{}",
            Error::CommentLengthOverflow {
                field: "vendor",
                len: 99,
                remaining: 5,
            }
        );
        let _ = format!(
            "{}",
            Error::CommentNotUtf8 {
                field: CommentField::Vendor
            }
        );
        let _ = format!(
            "{}",
            Error::CommentNotUtf8 {
                field: CommentField::Comment { index: 7 }
            }
        );
        let _ = format!("{}", Error::NotImplemented);
    }

    // ------- §6.3 comment header tests -------

    /// Bytes extracted from
    /// `docs/video/theora/fixtures/tiny-i-only-16x16/input.ogv`:
    /// second Ogg packet payload, framing already stripped. Encoder
    /// emitted vendor `"Lavf62.13.102"` and one comment
    /// `"encoder=Lavc62.30.100 libtheora"`. Every fixture currently
    /// in the corpus carries the same comment header byte-for-byte
    /// because all of them came out of the same FFmpeg / libtheora
    /// build.
    const TINY_COMMENT: [u8; 63] = [
        0x81, 0x74, 0x68, 0x65, 0x6f, 0x72, 0x61, // 0x81 + "theora"
        0x0d, 0x00, 0x00, 0x00, // vendor_len = 13 (LE)
        b'L', b'a', b'v', b'f', b'6', b'2', b'.', b'1', b'3', b'.', b'1', b'0', b'2', 0x01, 0x00,
        0x00, 0x00, // ncomments = 1 (LE)
        0x1f, 0x00, 0x00, 0x00, // comment[0] length = 31 (LE)
        b'e', b'n', b'c', b'o', b'd', b'e', b'r', b'=', b'L', b'a', b'v', b'c', b'6', b'2', b'.',
        b'3', b'0', b'.', b'1', b'0', b'0', b' ', b'l', b'i', b'b', b't', b'h', b'e', b'o', b'r',
        b'a',
    ];

    #[test]
    fn parses_tiny_fixture_comment_header() {
        let c = parse_comment_header(&TINY_COMMENT).expect("tiny comment header should decode");
        assert_eq!(c.vendor, "Lavf62.13.102");
        assert_eq!(c.comments.len(), 1);
        assert_eq!(c.comments[0].0, "encoder");
        assert_eq!(c.comments[0].1, "Lavc62.30.100 libtheora");
    }

    #[test]
    fn lookup_is_case_insensitive() {
        let c = parse_comment_header(&TINY_COMMENT).unwrap();
        assert_eq!(c.lookup("encoder"), Some("Lavc62.30.100 libtheora"));
        assert_eq!(c.lookup("ENCODER"), Some("Lavc62.30.100 libtheora"));
        assert_eq!(c.lookup("Encoder"), Some("Lavc62.30.100 libtheora"));
        assert_eq!(c.lookup("missing"), None);
    }

    /// Build a comment-header packet inline; used by negative-path
    /// tests so they can pin down exact bytes.
    fn synth_comment(vendor: &[u8], comments: &[&[u8]]) -> Vec<u8> {
        let mut out = vec![0x81, b't', b'h', b'e', b'o', b'r', b'a'];
        out.extend_from_slice(&(vendor.len() as u32).to_le_bytes());
        out.extend_from_slice(vendor);
        out.extend_from_slice(&(comments.len() as u32).to_le_bytes());
        for c in comments {
            out.extend_from_slice(&(c.len() as u32).to_le_bytes());
            out.extend_from_slice(c);
        }
        out
    }

    #[test]
    fn handles_zero_comments() {
        let pkt = synth_comment(b"Xiph.Org libtheora 1.2", &[]);
        let c = parse_comment_header(&pkt).unwrap();
        assert_eq!(c.vendor, "Xiph.Org libtheora 1.2");
        assert!(c.comments.is_empty());
    }

    #[test]
    fn handles_multiple_comments() {
        let pkt = synth_comment(
            b"libfaux 0.1",
            &[
                b"TITLE=the look of Theora",
                b"DIRECTOR=me",
                b"DATE=2026-05-21",
            ],
        );
        let c = parse_comment_header(&pkt).unwrap();
        assert_eq!(c.vendor, "libfaux 0.1");
        assert_eq!(c.comments.len(), 3);
        assert_eq!(c.comments[0], ("TITLE".into(), "the look of Theora".into()));
        assert_eq!(c.comments[1], ("DIRECTOR".into(), "me".into()));
        assert_eq!(c.comments[2], ("DATE".into(), "2026-05-21".into()));
        assert_eq!(c.lookup("title"), Some("the look of Theora"));
    }

    #[test]
    fn handles_utf8_value_payload() {
        // §6.3.3 explicitly permits UTF-8 in values; key is ASCII only.
        // The Yen-sign + Japanese character pair is 4 UTF-8 bytes.
        let comment = "TITLE=¥日".as_bytes();
        let pkt = synth_comment(b"v", &[comment]);
        let c = parse_comment_header(&pkt).unwrap();
        assert_eq!(c.comments[0].0, "TITLE");
        assert_eq!(c.comments[0].1, "¥日");
    }

    #[test]
    fn handles_empty_vendor() {
        let pkt = synth_comment(b"", &[b"X=y"]);
        let c = parse_comment_header(&pkt).unwrap();
        assert!(c.vendor.is_empty());
        assert_eq!(c.comments[0], ("X".into(), "y".into()));
    }

    #[test]
    fn handles_empty_value_after_equals() {
        let pkt = synth_comment(b"v", &[b"NULL="]);
        let c = parse_comment_header(&pkt).unwrap();
        assert_eq!(c.comments[0], ("NULL".into(), "".into()));
    }

    #[test]
    fn handles_comment_without_equals() {
        // §6.3.3 says the field name MUST be followed by '='; if a
        // vector lacks one we keep the whole text as the key with an
        // empty value rather than reject, to stay tolerant of
        // real-world streams.
        let pkt = synth_comment(b"v", &[b"NOEQUALSHERE"]);
        let c = parse_comment_header(&pkt).unwrap();
        assert_eq!(c.comments[0], ("NOEQUALSHERE".into(), "".into()));
    }

    #[test]
    fn allows_trailing_bytes_after_last_comment() {
        let mut pkt = synth_comment(b"v", &[b"K=v"]);
        pkt.extend_from_slice(b"trailing-garbage-that-should-be-ignored");
        let c = parse_comment_header(&pkt).expect("trailing bytes must not fail");
        assert_eq!(c.vendor, "v");
        assert_eq!(c.comments[0], ("K".into(), "v".into()));
    }

    #[test]
    fn rejects_identification_header_type() {
        let mut bad = TINY_COMMENT;
        bad[0] = 0x80;
        match parse_comment_header(&bad) {
            Err(Error::BadHeaderType { got: 0x80 }) => {}
            other => panic!("expected BadHeaderType(0x80), got {other:?}"),
        }
    }

    #[test]
    fn rejects_setup_header_type() {
        let mut bad = TINY_COMMENT;
        bad[0] = 0x82;
        match parse_comment_header(&bad) {
            Err(Error::BadHeaderType { got: 0x82 }) => {}
            other => panic!("expected BadHeaderType(0x82), got {other:?}"),
        }
    }

    #[test]
    fn rejects_video_data_packet_in_comment_path() {
        let mut bad = TINY_COMMENT;
        bad[0] = 0x40;
        match parse_comment_header(&bad) {
            Err(Error::BadHeaderType { got: 0x40 }) => {}
            other => panic!("expected BadHeaderType(0x40), got {other:?}"),
        }
    }

    #[test]
    fn rejects_bad_magic_in_comment_path() {
        let mut bad = TINY_COMMENT;
        bad[2] = b'X';
        assert_eq!(parse_comment_header(&bad), Err(Error::BadMagic));
    }

    #[test]
    fn rejects_vendor_length_overflow() {
        // Declare a 99-byte vendor in a packet that only has 0
        // payload octets after the length field.
        let pkt = [
            0x81, b't', b'h', b'e', b'o', b'r', b'a', // header
            0x63, 0x00, 0x00, 0x00, // vendor_len = 99
        ];
        match parse_comment_header(&pkt) {
            Err(Error::CommentLengthOverflow {
                field: "vendor",
                len: 99,
                remaining: 0,
            }) => {}
            other => panic!("expected CommentLengthOverflow(vendor), got {other:?}"),
        }
    }

    #[test]
    fn rejects_comment_length_overflow() {
        // Declare 1 comment of length 99 with only a few payload
        // bytes available.
        let pkt = synth_comment(b"v", &[b"K=v"])
            .into_iter()
            .chain([].iter().copied())
            .collect::<Vec<u8>>();
        // Take the prefix up through ncomments, replace the comment
        // length with 99, drop the rest.
        // Layout: 7-byte header + 4 (vendor_len) + 1 (vendor) + 4
        // (ncomments) + 4 (comment_len) + 3 (K=v) = 23 bytes.
        let mut bad = pkt[..16].to_vec(); // header + vendor_len + vendor + ncomments
        bad.extend_from_slice(&99u32.to_le_bytes());
        bad.extend_from_slice(b"K=v"); // only 3 bytes available
        match parse_comment_header(&bad) {
            Err(Error::CommentLengthOverflow {
                field: "comment",
                len: 99,
                remaining: 3,
            }) => {}
            other => panic!("expected CommentLengthOverflow(comment), got {other:?}"),
        }
    }

    #[test]
    fn rejects_ncomments_with_no_payload() {
        // Claim ncomments=1 but provide no comment-length octets.
        let mut bad = vec![0x81, b't', b'h', b'e', b'o', b'r', b'a'];
        bad.extend_from_slice(&0u32.to_le_bytes()); // vendor_len = 0
        bad.extend_from_slice(&1u32.to_le_bytes()); // ncomments = 1
                                                    // (no comment_len follows)
        match parse_comment_header(&bad) {
            Err(Error::TruncatedHeader {
                field: "comment_len",
            }) => {}
            other => panic!("expected TruncatedHeader(comment_len), got {other:?}"),
        }
    }

    #[test]
    fn rejects_truncated_vendor_length() {
        // Drop everything after the 7-byte header.
        match parse_comment_header(&TINY_COMMENT[..7]) {
            Err(Error::TruncatedHeader {
                field: "vendor_len",
            }) => {}
            other => panic!("expected TruncatedHeader(vendor_len), got {other:?}"),
        }
        // Partial vendor_len (3 of 4 octets).
        match parse_comment_header(&TINY_COMMENT[..10]) {
            Err(Error::TruncatedHeader {
                field: "vendor_len",
            }) => {}
            other => panic!("expected TruncatedHeader(vendor_len), got {other:?}"),
        }
    }

    #[test]
    fn rejects_invalid_utf8_vendor() {
        // Vendor length = 2 with bytes 0xff 0xfe (not valid UTF-8).
        let mut bad = vec![0x81, b't', b'h', b'e', b'o', b'r', b'a'];
        bad.extend_from_slice(&2u32.to_le_bytes());
        bad.extend_from_slice(&[0xff, 0xfe]);
        bad.extend_from_slice(&0u32.to_le_bytes());
        match parse_comment_header(&bad) {
            Err(Error::CommentNotUtf8 {
                field: CommentField::Vendor,
            }) => {}
            other => panic!("expected CommentNotUtf8(Vendor), got {other:?}"),
        }
    }

    #[test]
    fn rejects_invalid_utf8_comment() {
        // Valid vendor, comment[0] contains a lone 0xff byte.
        let mut bad = vec![0x81, b't', b'h', b'e', b'o', b'r', b'a'];
        bad.extend_from_slice(&1u32.to_le_bytes());
        bad.extend_from_slice(b"v");
        bad.extend_from_slice(&1u32.to_le_bytes()); // ncomments = 1
        bad.extend_from_slice(&1u32.to_le_bytes()); // comment_len = 1
        bad.push(0xff);
        match parse_comment_header(&bad) {
            Err(Error::CommentNotUtf8 {
                field: CommentField::Comment { index: 0 },
            }) => {}
            other => panic!("expected CommentNotUtf8(Comment{{0}}), got {other:?}"),
        }
    }

    #[test]
    fn ncomments_loop_records_comment_index_on_utf8_error() {
        // Two valid comments then a third with invalid UTF-8 — make
        // sure the reported index is the offending one.
        let mut pkt = vec![0x81, b't', b'h', b'e', b'o', b'r', b'a'];
        pkt.extend_from_slice(&0u32.to_le_bytes()); // vendor_len = 0
        pkt.extend_from_slice(&3u32.to_le_bytes()); // ncomments = 3
        for body in [b"A=1" as &[u8], b"B=2"] {
            pkt.extend_from_slice(&(body.len() as u32).to_le_bytes());
            pkt.extend_from_slice(body);
        }
        pkt.extend_from_slice(&1u32.to_le_bytes());
        pkt.push(0xff);
        match parse_comment_header(&pkt) {
            Err(Error::CommentNotUtf8 {
                field: CommentField::Comment { index: 2 },
            }) => {}
            other => panic!("expected CommentNotUtf8(Comment{{2}}), got {other:?}"),
        }
    }

    #[test]
    fn rejects_truncated_at_each_prefix() {
        // Every strict prefix of TINY_COMMENT must error; the full
        // packet must succeed. Allowed terminal error variants:
        // TruncatedHeader / BadMagic (for the very short ones).
        for len in 0..TINY_COMMENT.len() {
            match parse_comment_header(&TINY_COMMENT[..len]) {
                Err(Error::TruncatedHeader { .. }) | Err(Error::CommentLengthOverflow { .. }) => {}
                Ok(_) => panic!("len={len} should not parse"),
                Err(other) => panic!("len={len} unexpected {other:?}"),
            }
        }
        assert!(parse_comment_header(&TINY_COMMENT).is_ok());
    }

    // ------- §6.4.5 setup header entrypoint tests -------

    /// Minimal setup-header common header: `0x82` + "theora". Round
    /// 3 doesn't decode the body, so the test fixtures only need to
    /// carry the 7-byte preamble.
    const SETUP_PREAMBLE: [u8; 7] = [0x82, b't', b'h', b'e', b'o', b'r', b'a'];

    #[test]
    fn parse_setup_header_returns_body_not_implemented_on_valid_preamble() {
        // Round 3: any packet whose first 7 bytes are 0x82 + "theora"
        // should pass the §6.4.5 step 1 guard and surface the
        // SetupHeaderBodyNotImplemented sentinel.
        match parse_setup_header(&SETUP_PREAMBLE) {
            Err(Error::SetupHeaderBodyNotImplemented) => {}
            other => panic!("expected SetupHeaderBodyNotImplemented, got {other:?}"),
        }
    }

    #[test]
    fn parse_setup_header_accepts_trailing_body_bytes() {
        // The setup-header body lives after the 7-byte common header;
        // round 3's entrypoint must not care about its content. Drop
        // some arbitrary trailing bytes onto the preamble and confirm
        // we still get the body-not-implemented sentinel rather than
        // any parsing-derived error.
        let mut pkt = SETUP_PREAMBLE.to_vec();
        pkt.extend_from_slice(&[0xff, 0x00, 0xaa, 0x55, 0x12, 0x34]);
        match parse_setup_header(&pkt) {
            Err(Error::SetupHeaderBodyNotImplemented) => {}
            other => panic!("expected SetupHeaderBodyNotImplemented, got {other:?}"),
        }
    }

    #[test]
    fn parse_setup_header_rejects_identification_type() {
        let mut bad = SETUP_PREAMBLE;
        bad[0] = 0x80;
        match parse_setup_header(&bad) {
            Err(Error::BadHeaderType { got: 0x80 }) => {}
            other => panic!("expected BadHeaderType(0x80), got {other:?}"),
        }
    }

    #[test]
    fn parse_setup_header_rejects_comment_type() {
        let mut bad = SETUP_PREAMBLE;
        bad[0] = 0x81;
        match parse_setup_header(&bad) {
            Err(Error::BadHeaderType { got: 0x81 }) => {}
            other => panic!("expected BadHeaderType(0x81), got {other:?}"),
        }
    }

    #[test]
    fn parse_setup_header_rejects_video_data_packet() {
        let mut bad = SETUP_PREAMBLE;
        bad[0] = 0x42;
        match parse_setup_header(&bad) {
            Err(Error::BadHeaderType { got: 0x42 }) => {}
            other => panic!("expected BadHeaderType(0x42), got {other:?}"),
        }
    }

    #[test]
    fn parse_setup_header_rejects_bad_magic() {
        let mut bad = SETUP_PREAMBLE;
        bad[3] = b'X';
        assert_eq!(parse_setup_header(&bad), Err(Error::BadMagic));
    }

    #[test]
    fn parse_setup_header_rejects_truncation() {
        for len in 0..SETUP_PREAMBLE.len() {
            match parse_setup_header(&SETUP_PREAMBLE[..len]) {
                Err(Error::TruncatedHeader { .. }) => {}
                Err(Error::BadHeaderType { .. }) if len == 0 => {
                    panic!("len={len} should hit TruncatedHeader before BadHeaderType")
                }
                Ok(_) => panic!("len={len} should not parse"),
                Err(other) => panic!("len={len} unexpected {other:?}"),
            }
        }
    }

    #[test]
    fn setup_header_vp3_defaults_use_appendix_b_tables() {
        // The vp3_defaults constructor must wire each field to the
        // Appendix-B-derived constants. Spot-check the headline
        // values from Appendix B.2 / B.3.
        let h = TheoraSetupHeader::vp3_defaults();
        assert_eq!(h.loop_filter_limits, LFLIMS_VP3);
        assert_eq!(h.ac_scale, ACSCALE_VP3);
        assert_eq!(h.dc_scale, DCSCALE_VP3);
        // Endpoints called out in the trace fixtures.
        assert_eq!(h.loop_filter_limits[0], 30);
        assert_eq!(h.loop_filter_limits[63], 0);
        assert_eq!(h.ac_scale[0], 500);
        assert_eq!(h.ac_scale[63], 10);
        assert_eq!(h.dc_scale[0], 220);
        assert_eq!(h.dc_scale[63], 10);
    }

    #[test]
    fn setup_header_body_not_implemented_error_renders() {
        let s = format!("{}", Error::SetupHeaderBodyNotImplemented);
        assert!(s.contains("setup-header"));
        assert!(s.contains("§6.4.1") || s.contains("body"));
    }

    // ------- Appendix B.2 / B.3 VP3 table transcription tests -------

    /// `LFLIMS_VP3` is 64 entries. Appendix B.2 lists them in eight
    /// rows of eight; every entry is 7-bit (0..=127), and the spec
    /// table monotonically non-increases across `qi` (later `qi` =
    /// higher quantization = lower loop-filter limit until 0).
    #[test]
    fn lflims_vp3_table_shape() {
        assert_eq!(LFLIMS_VP3.len(), 64);
        for &v in &LFLIMS_VP3 {
            assert!(v <= 127, "LFLIMS entries are 7-bit; got {v}");
        }
        // Monotonic non-increasing per Appendix B.2 layout.
        for i in 1..64 {
            assert!(
                LFLIMS_VP3[i] <= LFLIMS_VP3[i - 1],
                "LFLIMS_VP3[{i}]={} > LFLIMS_VP3[{}]={}",
                LFLIMS_VP3[i],
                i - 1,
                LFLIMS_VP3[i - 1]
            );
        }
        // Spot values along the rows of Appendix B.2.
        assert_eq!(LFLIMS_VP3[0], 30);
        assert_eq!(LFLIMS_VP3[7], 14);
        assert_eq!(LFLIMS_VP3[8], 13);
        assert_eq!(LFLIMS_VP3[15], 10);
        assert_eq!(LFLIMS_VP3[16], 9);
        assert_eq!(LFLIMS_VP3[23], 7);
        assert_eq!(LFLIMS_VP3[24], 6);
        assert_eq!(LFLIMS_VP3[31], 5);
        assert_eq!(LFLIMS_VP3[32], 4);
        assert_eq!(LFLIMS_VP3[39], 3);
        assert_eq!(LFLIMS_VP3[40], 2);
        assert_eq!(LFLIMS_VP3[47], 2);
        assert_eq!(LFLIMS_VP3[48], 0);
        assert_eq!(LFLIMS_VP3[63], 0);
    }

    /// `ACSCALE_VP3` is 64 entries of 16-bit values, monotonically
    /// non-increasing per Appendix B.3.
    #[test]
    fn acscale_vp3_table_shape() {
        assert_eq!(ACSCALE_VP3.len(), 64);
        for i in 1..64 {
            assert!(
                ACSCALE_VP3[i] <= ACSCALE_VP3[i - 1],
                "ACSCALE_VP3[{i}]={} > ACSCALE_VP3[{}]={}",
                ACSCALE_VP3[i],
                i - 1,
                ACSCALE_VP3[i - 1]
            );
        }
        // Spot values along the rows of Appendix B.3.
        assert_eq!(ACSCALE_VP3[0], 500);
        assert_eq!(ACSCALE_VP3[7], 265);
        assert_eq!(ACSCALE_VP3[8], 245);
        assert_eq!(ACSCALE_VP3[15], 160);
        assert_eq!(ACSCALE_VP3[16], 150);
        assert_eq!(ACSCALE_VP3[23], 107);
        assert_eq!(ACSCALE_VP3[24], 100);
        assert_eq!(ACSCALE_VP3[31], 74);
        assert_eq!(ACSCALE_VP3[32], 70);
        assert_eq!(ACSCALE_VP3[39], 50);
        assert_eq!(ACSCALE_VP3[40], 49);
        assert_eq!(ACSCALE_VP3[47], 35);
        assert_eq!(ACSCALE_VP3[48], 33);
        assert_eq!(ACSCALE_VP3[55], 22);
        assert_eq!(ACSCALE_VP3[56], 21);
        assert_eq!(ACSCALE_VP3[63], 10);
    }

    /// `DCSCALE_VP3` is 64 entries of 16-bit values, monotonically
    /// non-increasing per Appendix B.3.
    #[test]
    fn dcscale_vp3_table_shape() {
        assert_eq!(DCSCALE_VP3.len(), 64);
        for i in 1..64 {
            assert!(
                DCSCALE_VP3[i] <= DCSCALE_VP3[i - 1],
                "DCSCALE_VP3[{i}]={} > DCSCALE_VP3[{}]={}",
                DCSCALE_VP3[i],
                i - 1,
                DCSCALE_VP3[i - 1]
            );
        }
        // Spot values along the rows of Appendix B.3.
        assert_eq!(DCSCALE_VP3[0], 220);
        assert_eq!(DCSCALE_VP3[7], 160);
        assert_eq!(DCSCALE_VP3[8], 150);
        assert_eq!(DCSCALE_VP3[15], 120);
        assert_eq!(DCSCALE_VP3[16], 110);
        assert_eq!(DCSCALE_VP3[23], 80);
        assert_eq!(DCSCALE_VP3[24], 80);
        assert_eq!(DCSCALE_VP3[31], 60);
        assert_eq!(DCSCALE_VP3[32], 60);
        assert_eq!(DCSCALE_VP3[39], 40);
        assert_eq!(DCSCALE_VP3[40], 40);
        assert_eq!(DCSCALE_VP3[47], 30);
        assert_eq!(DCSCALE_VP3[48], 30);
        assert_eq!(DCSCALE_VP3[55], 20);
        assert_eq!(DCSCALE_VP3[56], 20);
        assert_eq!(DCSCALE_VP3[63], 10);
    }

    /// Sanity-check the row sums against an independent re-tally of
    /// Appendix B.2 / B.3, in case a transcription typo only shows
    /// up as a swapped pair within the same row.
    #[test]
    fn appendix_b_row_sums_match_spec() {
        // Each row is 8 consecutive entries. Sums independently
        // re-derived from Appendix B's printed table layout.
        const LFLIMS_ROW_SUMS: [u32; 8] = [
            30 + 25 + 20 + 20 + 15 + 15 + 14 + 14, // row 0
            13 + 13 + 12 + 12 + 11 + 11 + 10 + 10, // row 1
            9 + 9 + 8 + 8 + 7 + 7 + 7 + 7,         // row 2
            6 + 6 + 6 + 6 + 5 + 5 + 5 + 5,         // row 3
            4 + 4 + 4 + 4 + 3 + 3 + 3 + 3,         // row 4
            2 + 2 + 2 + 2 + 2 + 2 + 2 + 2,         // row 5
            0,
            0, // rows 6, 7
        ];
        for (row, &expected) in LFLIMS_ROW_SUMS.iter().enumerate() {
            let got: u32 = LFLIMS_VP3[row * 8..row * 8 + 8]
                .iter()
                .map(|&v| v as u32)
                .sum();
            assert_eq!(got, expected, "LFLIMS row {row} sum mismatch");
        }

        const ACSCALE_ROW_SUMS: [u32; 8] = [
            500 + 450 + 400 + 370 + 340 + 310 + 285 + 265,
            245 + 225 + 210 + 195 + 185 + 180 + 170 + 160,
            150 + 145 + 135 + 130 + 125 + 115 + 110 + 107,
            100 + 96 + 93 + 89 + 85 + 82 + 75 + 74,
            70 + 68 + 64 + 60 + 57 + 56 + 52 + 50,
            49 + 45 + 44 + 43 + 40 + 38 + 37 + 35,
            33 + 32 + 30 + 29 + 28 + 25 + 24 + 22,
            21 + 19 + 18 + 17 + 15 + 13 + 12 + 10,
        ];
        for (row, &expected) in ACSCALE_ROW_SUMS.iter().enumerate() {
            let got: u32 = ACSCALE_VP3[row * 8..row * 8 + 8]
                .iter()
                .map(|&v| v as u32)
                .sum();
            assert_eq!(got, expected, "ACSCALE row {row} sum mismatch");
        }

        const DCSCALE_ROW_SUMS: [u32; 8] = [
            220 + 200 + 190 + 180 + 170 + 170 + 160 + 160,
            150 + 150 + 140 + 140 + 130 + 130 + 120 + 120,
            110 + 110 + 100 + 100 + 90 + 90 + 90 + 80,
            80 + 80 + 70 + 70 + 70 + 60 + 60 + 60,
            60 + 50 + 50 + 50 + 50 + 40 + 40 + 40,
            40 + 40 + 30 + 30 + 30 + 30 + 30 + 30,
            30 + 20 + 20 + 20 + 20 + 20 + 20 + 20,
            20 + 10 + 10 + 10 + 10 + 10 + 10 + 10,
        ];
        for (row, &expected) in DCSCALE_ROW_SUMS.iter().enumerate() {
            let got: u32 = DCSCALE_VP3[row * 8..row * 8 + 8]
                .iter()
                .map(|&v| v as u32)
                .sum();
            assert_eq!(got, expected, "DCSCALE row {row} sum mismatch");
        }
    }

    /// VP3 default constructor exposes the same `[u8; 64]` and
    /// `[u16; 64]` field layout the round-4 prompt scopes.
    #[test]
    fn vp3_defaults_layout_matches_round4_contract() {
        let h = TheoraSetupHeader::vp3_defaults();
        let _: [u8; 64] = h.loop_filter_limits;
        let _: [u16; 64] = h.ac_scale;
        let _: [u16; 64] = h.dc_scale;
    }

    // ------- §5.2 BitReader tests -------

    #[test]
    fn bitreader_reads_msb_first_within_byte() {
        // §5.2 example: byte 0b1100_0000 produces 3 (b11) for the
        // first 2-bit read, then 0 (b00) for the next 2-bit read.
        // (Adapted from the spec's §5.2.3 decoding example.)
        let mut r = BitReader::new(&[0b1100_1110, 0b0100_0111, 0b0110_0111, 0b0010_0000]);
        assert_eq!(r.read_bits(2, "a").unwrap(), 3);
        assert_eq!(r.read_bits(2, "b").unwrap(), 0);
    }

    #[test]
    fn bitreader_reads_4_bit_value() {
        // The spec's §5.2.2 encoding example writes 12 (b1100) as
        // the first 4-bit field. Reading 4 bits from byte 0
        // 0b1100_0000 must return 12.
        let mut r = BitReader::new(&[0b1100_0000]);
        assert_eq!(r.read_bits(4, "x").unwrap(), 12);
    }

    #[test]
    fn bitreader_spans_byte_boundary() {
        // Read 12 bits across two bytes: byte0=0xAB, byte1=0xCD
        // 0xAB = 1010_1011, 0xCD = 1100_1101
        // First 12 MSb-first bits = 1010_1011_1100 = 0xABC.
        let mut r = BitReader::new(&[0xAB, 0xCD]);
        assert_eq!(r.read_bits(12, "x").unwrap(), 0xABC);
    }

    #[test]
    fn bitreader_reads_full_32_bits() {
        // 32 bits MSb-first equals u32::from_be_bytes.
        let bytes = [0xDE, 0xAD, 0xBE, 0xEF];
        let mut r = BitReader::new(&bytes);
        assert_eq!(r.read_bits(32, "x").unwrap(), 0xDEAD_BEEF);
    }

    #[test]
    fn bitreader_zero_width_read_does_nothing() {
        let mut r = BitReader::new(&[0xFF]);
        assert_eq!(r.read_bits(0, "x").unwrap(), 0);
        // Subsequent 8-bit read must still see the original byte.
        assert_eq!(r.read_bits(8, "x").unwrap(), 0xFF);
    }

    #[test]
    fn bitreader_returns_truncated_when_exhausted() {
        let mut r = BitReader::new(&[0xAB]);
        // Consume the byte, then attempt one more bit.
        assert_eq!(r.read_bits(8, "byte").unwrap(), 0xAB);
        match r.read_bits(1, "overflow") {
            Err(Error::TruncatedHeader { field: "overflow" }) => {}
            other => panic!("expected TruncatedHeader, got {other:?}"),
        }
    }

    #[test]
    fn bitreader_returns_truncated_mid_field() {
        // Buffer has 12 bits available; requesting 16 fails at the
        // 13th bit.
        let mut r = BitReader::new(&[0xAB, 0xC0]);
        // First 12 bits succeed.
        assert_eq!(r.read_bits(12, "first").unwrap(), 0xABC);
        // Next 16 bits fail because only 4 remain.
        match r.read_bits(16, "tail") {
            Err(Error::TruncatedHeader { field: "tail" }) => {}
            other => panic!("expected TruncatedHeader(tail), got {other:?}"),
        }
    }

    #[test]
    fn bitreader_3_then_4_then_5_bit_sequence() {
        // §6.4.5 step 2 will read a 3-bit NBITS (§6.4.1) immediately
        // after the 7-byte common header. Smoke-test mixed bit
        // widths against a hand-traced value.
        //
        // Bits in order MSb-first: a=010 b=1010 c=11100
        //   (3-bit value 2 + 4-bit value 10 + 5-bit value 28 = 12 bits).
        //
        // Packed 12-bit value = 0101_0101_1100 = 0x55C.
        // As MSb-first bytes: byte0 = 0101_0101 = 0x55,
        //                     byte1 = 1100_xxxx (low 4 bits unused).
        let mut r = BitReader::new(&[0x55, 0xC0]);
        assert_eq!(r.read_bits(3, "a").unwrap(), 0b010);
        assert_eq!(r.read_bits(4, "b").unwrap(), 0b1010);
        assert_eq!(r.read_bits(5, "c").unwrap(), 0b11100);
    }

    // ------- §6.4.2 Quantization Parameters tests -------

    /// MSb-first bit writer mirroring [`BitReader`]. Used to synthesize
    /// §6.4.2 payloads bit-exactly so the decode can be round-tripped
    /// against a known encoding. (Test-only — the crate ships no
    /// encoder yet.)
    struct BitWriter {
        bytes: Vec<u8>,
        /// Number of bits already filled in the current (last) byte,
        /// from the MSb down. 0 means "start a new byte".
        used: u8,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                used: 0,
            }
        }

        /// Append the low `n` bits of `value`, MSb first.
        fn put(&mut self, value: u32, n: u32) {
            for i in (0..n).rev() {
                let bit = ((value >> i) & 1) as u8;
                if self.used == 0 {
                    self.bytes.push(0);
                }
                let last = self.bytes.len() - 1;
                self.bytes[last] |= bit << (7 - self.used);
                self.used = (self.used + 1) % 8;
            }
        }

        fn finish(self) -> Vec<u8> {
            self.bytes
        }
    }

    /// `ilog` helper mirrored in the test to compute field widths for
    /// the synthesizer, independent of the impl's `ilog`.
    fn test_ilog(a: i64) -> u32 {
        if a <= 0 {
            0
        } else {
            let mut v = a as u64;
            let mut bits = 0;
            while v > 0 {
                v >>= 1;
                bits += 1;
            }
            bits
        }
    }

    #[test]
    fn ilog_matches_spec_examples() {
        // Worked examples from the spec's Notation and Conventions
        // section.
        assert_eq!(ilog(-1), 0);
        assert_eq!(ilog(0), 0);
        assert_eq!(ilog(1), 1);
        assert_eq!(ilog(2), 2);
        assert_eq!(ilog(3), 2);
        assert_eq!(ilog(4), 3);
        assert_eq!(ilog(7), 3);
        // A few more boundary values.
        assert_eq!(ilog(8), 4);
        assert_eq!(ilog(62), 6);
        assert_eq!(ilog(63), 6);
        assert_eq!(ilog(383), 9); // ilog(NBMS-1) for NBMS=384
    }

    /// Encode the AC + DC scale tables (§6.4.2 steps 1–4) into `w`,
    /// using a fixed NBITS of 16 (4-bit field value 15, plus one) so
    /// the full 16-bit scale range is representable.
    fn encode_scales(w: &mut BitWriter, ac: &[u16; 64], dc: &[u16; 64]) {
        w.put(15, 4); // ACSCALE NBITS = 16
        for &v in ac.iter() {
            w.put(v as u32, 16);
        }
        w.put(15, 4); // DCSCALE NBITS = 16
        for &v in dc.iter() {
            w.put(v as u32, 16);
        }
    }

    #[test]
    fn decode_quant_params_minimal_single_range() {
        // Synthesize a minimal but spec-valid §6.4.2 payload:
        //   * ACSCALE / DCSCALE = a recognizable ramp.
        //   * NBMS = 2 (two base matrices).
        //   * Each (qti, pli) defines ONE quant range of size 63 with
        //     endpoints {bmi 0, bmi 1}.
        let mut ac = [0u16; 64];
        let mut dc = [0u16; 64];
        for i in 0..64 {
            ac[i] = (500 - i * 7) as u16;
            dc[i] = (220 - i * 3) as u16;
        }

        let mut w = BitWriter::new();
        encode_scales(&mut w, &ac, &dc);

        let nbms = 2u32;
        w.put(nbms - 1, 9); // NBMS field
        let bmi_bits = test_ilog(nbms as i64 - 1); // ilog(1) = 1
                                                   // Base matrices: matrix 0 = all 16, matrix 1 = all 100.
        for _ci in 0..64 {
            w.put(16, 8);
        }
        for _ci in 0..64 {
            w.put(100, 8);
        }

        // Step 7: for each (qti, pli), define a single 63-wide range.
        for qti in 0..2 {
            for pli in 0..3 {
                if qti > 0 || pli > 0 {
                    w.put(1, 1); // NEWQR = 1 (define new)
                }
                // 7(a)iv: qri=0, qi=0
                w.put(0, bmi_bits); // QRBMIS[0] = 0
                                    // D: ilog(62 - 0) = 6 bits; size = read + 1 = 63 -> read 62.
                w.put(62, test_ilog(62));
                // G: QRBMIS[1] = 1
                w.put(1, bmi_bits);
                // qi == 63 -> stop; NQRS = 1.
            }
        }

        let payload = w.finish();
        let qp = decode_quantization_parameters(&payload).expect("valid payload decodes");

        assert_eq!(qp.ac_scale, ac);
        assert_eq!(qp.dc_scale, dc);
        assert_eq!(qp.num_base_matrices, 2);
        assert_eq!(qp.base_matrices.len(), 2);
        assert_eq!(qp.base_matrices[0], [16u8; 64]);
        assert_eq!(qp.base_matrices[1], [100u8; 64]);
        for qti in 0..2 {
            for pli in 0..3 {
                assert_eq!(qp.num_quant_ranges[qti][pli], 1, "NQRS[{qti}][{pli}]");
                assert_eq!(qp.quant_range_sizes[qti][pli][0], 63);
                assert_eq!(qp.quant_range_base_matrix_indices[qti][pli][0], 0);
                assert_eq!(qp.quant_range_base_matrix_indices[qti][pli][1], 1);
            }
        }
    }

    #[test]
    fn decode_quant_params_two_ranges_sum_to_63() {
        // A (qti=0, pli=0) range list with two ranges 20 + 43 = 63,
        // exercising the inner loop's continuation (step 7(a)ivH).
        // Remaining (qti, pli) copy the first via NEWQR=0.
        let ac = [400u16; 64];
        let dc = [200u16; 64];

        let mut w = BitWriter::new();
        encode_scales(&mut w, &ac, &dc);

        let nbms = 4u32;
        w.put(nbms - 1, 9);
        let bmi_bits = test_ilog(nbms as i64 - 1); // ilog(3) = 2
        for bmi in 0..nbms {
            for _ci in 0..64 {
                w.put(bmi * 10, 8);
            }
        }

        // (0,0): NEWQR forced 1. Define two ranges.
        // qri=0, qi=0:  QRBMIS[0]=0
        w.put(0, bmi_bits);
        //   D: ilog(62-0)=6 bits, size=20 -> read 19
        w.put(19, test_ilog(62));
        //   qi=20; G: QRBMIS[1]=1
        w.put(1, bmi_bits);
        //   qi<63 -> back to D: ilog(62-20)=ilog(42)=6 bits, size=43 -> read 42
        w.put(42, test_ilog(62 - 20));
        //   qi=63; G: QRBMIS[2]=2
        w.put(2, bmi_bits);
        //   qi==63 -> stop. NQRS[0][0]=2.

        // (0,1): NEWQR=0 -> copy. qti==0 so RPQR=0 forced (not read).
        //   selects (qtj,plj) = ((3*0+1-1)/3, (1+2)%3) = (0, 0).
        w.put(0, 1);
        // (0,2): NEWQR=0 -> copy from (0,1).
        w.put(0, 1);
        // (1,0): NEWQR=0 -> qti>0 so read RPQR. RPQR=1 -> copy (0,0).
        w.put(0, 1); // NEWQR
        w.put(1, 1); // RPQR
                     // (1,1): NEWQR=0 -> RPQR=1 -> copy (0,1).
        w.put(0, 1);
        w.put(1, 1);
        // (1,2): NEWQR=0 -> RPQR=1 -> copy (0,2).
        w.put(0, 1);
        w.put(1, 1);

        let payload = w.finish();
        let qp = decode_quantization_parameters(&payload).expect("valid payload decodes");

        assert_eq!(qp.num_quant_ranges[0][0], 2);
        assert_eq!(qp.quant_range_sizes[0][0][0], 20);
        assert_eq!(qp.quant_range_sizes[0][0][1], 43);
        assert_eq!(qp.quant_range_base_matrix_indices[0][0][0], 0);
        assert_eq!(qp.quant_range_base_matrix_indices[0][0][1], 1);
        assert_eq!(qp.quant_range_base_matrix_indices[0][0][2], 2);
        // Copied sets must match the source set exactly.
        for (qti, pli) in [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)] {
            assert_eq!(qp.num_quant_ranges[qti][pli], 2, "NQRS[{qti}][{pli}] copy");
            assert_eq!(qp.quant_range_sizes[qti][pli][0], 20);
            assert_eq!(qp.quant_range_sizes[qti][pli][1], 43);
        }
    }

    #[test]
    fn decode_quant_params_rejects_nbms_over_384() {
        let ac = [10u16; 64];
        let dc = [10u16; 64];
        let mut w = BitWriter::new();
        encode_scales(&mut w, &ac, &dc);
        // NBMS field of 384 -> decoded NBMS = 385 > 384.
        w.put(384, 9);
        let payload = w.finish();
        match decode_quantization_parameters(&payload) {
            Err(Error::TooManyBaseMatrices { nbms: 385 }) => {}
            other => panic!("expected TooManyBaseMatrices(385), got {other:?}"),
        }
    }

    #[test]
    fn decode_quant_params_accepts_nbms_exactly_384() {
        // Boundary: NBMS field 383 -> decoded NBMS = 384, the maximum.
        let ac = [10u16; 64];
        let dc = [10u16; 64];
        let mut w = BitWriter::new();
        encode_scales(&mut w, &ac, &dc);
        let nbms = 384u32;
        w.put(nbms - 1, 9);
        let bmi_bits = test_ilog(nbms as i64 - 1); // ilog(383) = 9
        for _bmi in 0..nbms {
            for _ci in 0..64 {
                w.put(0, 8);
            }
        }
        // One 63-wide range per (qti, pli), endpoints {0, 1}.
        for qti in 0..2 {
            for pli in 0..3 {
                if qti > 0 || pli > 0 {
                    w.put(1, 1);
                }
                w.put(0, bmi_bits);
                w.put(62, test_ilog(62));
                w.put(1, bmi_bits);
            }
        }
        let payload = w.finish();
        let qp = decode_quantization_parameters(&payload).expect("NBMS=384 is valid");
        assert_eq!(qp.num_base_matrices, 384);
        assert_eq!(qp.base_matrices.len(), 384);
    }

    #[test]
    fn decode_quant_params_rejects_qrbmis_ge_nbms() {
        // First QRBMIS read (step 7(a)ivC) >= NBMS is undecodable.
        let ac = [10u16; 64];
        let dc = [10u16; 64];
        let mut w = BitWriter::new();
        encode_scales(&mut w, &ac, &dc);
        let nbms = 2u32; // bmi_bits = ilog(1) = 1, range 0..=1
        w.put(nbms - 1, 9);
        for _bmi in 0..nbms {
            for _ci in 0..64 {
                w.put(0, 8);
            }
        }
        // (0,0) NEWQR forced 1. QRBMIS[0] = ... but ilog(1)=1 can only
        // encode 0 or 1, both < 2. To force an out-of-range index we
        // need NBMS where the field can encode >= NBMS. Use NBMS=3:
        // ilog(2)=2 bits encode 0..=3; value 3 >= 3 is undecodable.
        // Rebuild with NBMS=3.
        let mut w = BitWriter::new();
        encode_scales(&mut w, &ac, &dc);
        let nbms = 3u32;
        w.put(nbms - 1, 9);
        let bmi_bits = test_ilog(nbms as i64 - 1); // ilog(2) = 2
        for _bmi in 0..nbms {
            for _ci in 0..64 {
                w.put(0, 8);
            }
        }
        // (0,0): QRBMIS[0] = 3 >= NBMS=3 -> undecodable.
        w.put(3, bmi_bits);
        let payload = w.finish();
        match decode_quantization_parameters(&payload) {
            Err(Error::BaseMatrixIndexOutOfRange { bmi: 3, nbms: 3 }) => {}
            other => panic!("expected BaseMatrixIndexOutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn decode_quant_params_rejects_qi_overflow() {
        // A range whose accumulated qi overshoots 63 is undecodable
        // (step 7(a)ivI).
        let ac = [10u16; 64];
        let dc = [10u16; 64];
        let mut w = BitWriter::new();
        encode_scales(&mut w, &ac, &dc);
        let nbms = 2u32;
        w.put(nbms - 1, 9);
        let bmi_bits = test_ilog(nbms as i64 - 1);
        for _bmi in 0..nbms {
            for _ci in 0..64 {
                w.put(0, 8);
            }
        }
        // (0,0): one range, size = 64 (read 63 over ilog(62)=6 bits).
        // qi becomes 64 > 63 -> undecodable.
        w.put(0, bmi_bits);
        w.put(63, test_ilog(62)); // size = 64
        w.put(1, bmi_bits); // QRBMIS[1]
        let payload = w.finish();
        match decode_quantization_parameters(&payload) {
            Err(Error::QuantRangeOverflow { qi: 64 }) => {}
            other => panic!("expected QuantRangeOverflow(64), got {other:?}"),
        }
    }

    #[test]
    fn decode_quant_params_truncated_in_scales() {
        // A payload that ends mid-ACSCALE returns TruncatedHeader.
        let mut w = BitWriter::new();
        w.put(7, 4); // ACSCALE NBITS = 8
                     // Only 3 ACSCALE values, then EOF.
        w.put(100, 8);
        w.put(99, 8);
        w.put(98, 8);
        let payload = w.finish();
        match decode_quantization_parameters(&payload) {
            Err(Error::TruncatedHeader { field: "ACSCALE" }) => {}
            other => panic!("expected TruncatedHeader(ACSCALE), got {other:?}"),
        }
    }

    #[test]
    fn decode_quant_params_truncated_at_nbms() {
        let ac = [10u16; 64];
        let dc = [10u16; 64];
        let mut w = BitWriter::new();
        encode_scales(&mut w, &ac, &dc);
        // Stop before the 9-bit NBMS field.
        let payload = w.finish();
        match decode_quantization_parameters(&payload) {
            Err(Error::TruncatedHeader { field: "NBMS" }) => {}
            other => panic!("expected TruncatedHeader(NBMS), got {other:?}"),
        }
    }

    #[test]
    fn decode_quant_params_chroma_copy_via_rpqr_zero() {
        // Exercise the "most recent set" copy branch (RPQR=0 / step
        // 7(a)iiiD) on an INTER plane. (qti=1, pli=0) with NEWQR=0,
        // RPQR=0 selects (qtj,plj) = ((3*1+0-1)/3, (0+2)%3) = (0, 2).
        let ac = [300u16; 64];
        let dc = [150u16; 64];
        let mut w = BitWriter::new();
        encode_scales(&mut w, &ac, &dc);
        let nbms = 2u32;
        w.put(nbms - 1, 9);
        let bmi_bits = test_ilog(nbms as i64 - 1);
        for _bmi in 0..nbms {
            for _ci in 0..64 {
                w.put(0, 8);
            }
        }
        // Define new ranges for all three INTRA planes with distinct
        // single-range sizes so we can tell which one is copied.
        // (0,0): size 63, QRBMIS {0,1}
        w.put(0, bmi_bits);
        w.put(62, test_ilog(62));
        w.put(1, bmi_bits);
        // (0,1): NEWQR=1, size 63, QRBMIS {1,0}
        w.put(1, 1);
        w.put(1, bmi_bits);
        w.put(62, test_ilog(62));
        w.put(0, bmi_bits);
        // (0,2): NEWQR=1, size 63, QRBMIS {0,0}
        w.put(1, 1);
        w.put(0, bmi_bits);
        w.put(62, test_ilog(62));
        w.put(0, bmi_bits);
        // (1,0): NEWQR=0, RPQR=0 -> copy (0,2): QRBMIS {0,0}
        w.put(0, 1); // NEWQR
        w.put(0, 1); // RPQR
                     // (1,1): NEWQR=1, size 63, QRBMIS {1,1}
        w.put(1, 1);
        w.put(1, bmi_bits);
        w.put(62, test_ilog(62));
        w.put(1, bmi_bits);
        // (1,2): NEWQR=1, size 63, QRBMIS {0,1}
        w.put(1, 1);
        w.put(0, bmi_bits);
        w.put(62, test_ilog(62));
        w.put(1, bmi_bits);

        let payload = w.finish();
        let qp = decode_quantization_parameters(&payload).expect("valid payload decodes");

        // (1,0) copied (0,2): endpoints {0, 0}.
        assert_eq!(qp.num_quant_ranges[1][0], 1);
        assert_eq!(qp.quant_range_base_matrix_indices[1][0][0], 0);
        assert_eq!(qp.quant_range_base_matrix_indices[1][0][1], 0);
        // (0,2) source endpoints {0, 0} confirm it really matched.
        assert_eq!(qp.quant_range_base_matrix_indices[0][2][0], 0);
        assert_eq!(qp.quant_range_base_matrix_indices[0][2][1], 0);
    }

    #[test]
    fn quant_params_error_displays_render() {
        // Smoke-test the three new error Display arms don't panic.
        let e1 = Error::TooManyBaseMatrices { nbms: 385 };
        let e2 = Error::BaseMatrixIndexOutOfRange { bmi: 3, nbms: 3 };
        let e3 = Error::QuantRangeOverflow { qi: 64 };
        assert!(format!("{e1}").contains("NBMS=385"));
        assert!(format!("{e2}").contains("QRBMIS"));
        assert!(format!("{e3}").contains("qi=64"));
    }

    // ---- §6.4.3 Computing a Quantization Matrix ----------------------

    /// Build a [`QuantizationParameters`] populated with a single
    /// 63-wide quant range for every `(qti, pli)`, with base-matrix
    /// endpoints `{bmi0, bmi1}`. The base-matrix table is supplied
    /// directly. AC/DC scales are constants `ac`/`dc` at every qi.
    ///
    /// This bypasses the §6.4.2 bit decode entirely so the §6.4.3
    /// procedure can be exercised against hand-computed expectations.
    fn single_range_params(
        base_matrices: Vec<[u8; 64]>,
        bmi0: u16,
        bmi1: u16,
        ac: u16,
        dc: u16,
    ) -> QuantizationParameters {
        let mut qrsizes = [[[0u8; 63]; 3]; 2];
        let mut qrbmis = [[[0u16; 64]; 3]; 2];
        for qti in 0..2 {
            for pli in 0..3 {
                qrsizes[qti][pli][0] = 63;
                qrbmis[qti][pli][0] = bmi0;
                qrbmis[qti][pli][1] = bmi1;
            }
        }
        QuantizationParameters {
            ac_scale: [ac; 64],
            dc_scale: [dc; 64],
            num_base_matrices: base_matrices.len() as u16,
            base_matrices,
            num_quant_ranges: [[1u8; 3]; 2],
            quant_range_sizes: qrsizes,
            quant_range_base_matrix_indices: qrbmis,
        }
    }

    #[test]
    fn compute_qmat_endpoints_pick_corner_base_matrices() {
        // Single 63-wide range, endpoints bm0 (all 16) .. bm1 (all 100).
        // qi=0 selects QISTART end -> BM == bm0; qi=63 selects QIEND end
        // -> BM == bm1 (§6.4.3 steps 1-3 / 6(a)).
        let params = single_range_params(vec![[16u8; 64], [100u8; 64]], 0, 1, 400, 200);

        // qi=0, intra (qti=0), luma (pli=0): BM[ci]=16 everywhere.
        //   DC (ci=0): (200*16//100)*4 = (3200//100)*4 = 32*4 = 128.
        //   AC (ci>0): (400*16//100)*4 = (6400//100)*4 = 64*4 = 256.
        let q0 = compute_quantization_matrix(&params, 0, 0, 0).unwrap();
        assert_eq!(q0.values[0], 128, "qi=0 DC");
        for &v in &q0.values[1..] {
            assert_eq!(v, 256, "qi=0 AC");
        }

        // qi=63: BM[ci]=100 everywhere.
        //   DC: (200*100//100)*4 = 200*4 = 800.
        //   AC: (400*100//100)*4 = 400*4 = 1600.
        let q63 = compute_quantization_matrix(&params, 0, 0, 63).unwrap();
        assert_eq!(q63.values[0], 800, "qi=63 DC");
        for &v in &q63.values[1..] {
            assert_eq!(v, 1600, "qi=63 AC");
        }
    }

    #[test]
    fn compute_qmat_interpolates_within_a_range() {
        // qi at the midpoint of a 63-wide range interpolates halfway
        // between the two endpoint base matrices.
        //   QISTART=0, QIEND=63, qrsize=63, denom=126.
        //   At qi=31 with bm0=0, bm1=126:
        //     BM = (2*(63-31)*0 + 2*(31-0)*126 + 63)//126
        //        = (0 + 7812 + 63)//126 = 7875//126 = 62.
        let params = single_range_params(vec![[0u8; 64], [126u8; 64]], 0, 1, 100, 100);
        let q = compute_quantization_matrix(&params, 0, 0, 31).unwrap();
        // BM = 62 everywhere.
        //   DC (ci=0): (100*62//100)*4 = 62*4 = 248.
        //   AC: same scale here -> also 248.
        assert_eq!(q.values[0], 248);
        for &v in &q.values[1..] {
            assert_eq!(v, 248);
        }
    }

    #[test]
    fn compute_qmat_two_range_interpolation_and_boundary() {
        // Two ranges: size 32 then 31 (sum 63). Base-matrix endpoints
        // QRBMIS = [0, 1, 2]; bm0=0, bm1=64, bm2=128. Scale = 100 so
        // QMAT[ci] == BM[ci]*4.
        let base = vec![[0u8; 64], [64u8; 64], [128u8; 64]];
        let mut qrsizes = [[[0u8; 63]; 3]; 2];
        let mut qrbmis = [[[0u16; 64]; 3]; 2];
        for qti in 0..2 {
            for pli in 0..3 {
                qrsizes[qti][pli][0] = 32;
                qrsizes[qti][pli][1] = 31;
                qrbmis[qti][pli][0] = 0;
                qrbmis[qti][pli][1] = 1;
                qrbmis[qti][pli][2] = 2;
            }
        }
        let params = QuantizationParameters {
            ac_scale: [100u16; 64],
            dc_scale: [100u16; 64],
            num_base_matrices: 3,
            base_matrices: base,
            num_quant_ranges: [[2u8; 3]; 2],
            quant_range_sizes: qrsizes,
            quant_range_base_matrix_indices: qrbmis,
        };

        // qi=16 in range 0 [0,32]: QISTART=0, QIEND=32, denom=64.
        //   BM = (2*(32-16)*0 + 2*(16-0)*64 + 32)//64
        //      = (0 + 2048 + 32)//64 = 2080//64 = 32.
        //   QMAT = 32*4 = 128 (DC and AC, scale 100).
        let q16 = compute_quantization_matrix(&params, 0, 0, 16).unwrap();
        for &v in q16.values.iter() {
            assert_eq!(v, 128, "qi=16 interpolation");
        }

        // qi=48 in range 1 [32,63]: QISTART=32, QIEND=63, qrsize=31,
        // denom=62.
        //   BM = (2*(63-48)*64 + 2*(48-32)*128 + 31)//62
        //      = (1920 + 4096 + 31)//62 = 6047//62 = 97.
        //   QMAT = 97*4 = 388.
        let q48 = compute_quantization_matrix(&params, 0, 0, 48).unwrap();
        for &v in q48.values.iter() {
            assert_eq!(v, 388, "qi=48 interpolation");
        }

        // qi=32 on the boundary must agree no matter which range is
        // chosen (§6.4.3 step 1 note). Our impl picks range 0:
        //   BM = (2*(32-32)*0 + 2*(32-0)*64 + 32)//64
        //      = (0 + 4096 + 32)//64 = 4128//64 = 64.
        //   QMAT = 64*4 = 256.
        // Manually computing via range 1 gives the same:
        //   BM = (2*(63-32)*64 + 2*(32-32)*128 + 31)//62
        //      = (3968 + 0 + 31)//62 = 3999//62 = 64.
        let q32 = compute_quantization_matrix(&params, 0, 0, 32).unwrap();
        for &v in q32.values.iter() {
            assert_eq!(v, 256, "qi=32 boundary");
        }
    }

    #[test]
    fn compute_qmat_applies_qmin_floor_per_table_6_18() {
        // Tiny scales drive the scaled value below QMIN so the floor
        // (Table 6.18) wins. bm=16 everywhere; dc_scale=1, ac_scale=1.
        //   scaled = (1*16//100)*4 = (0)*4 = 0.
        let params = single_range_params(vec![[16u8; 64], [16u8; 64]], 0, 1, 1, 1);

        // Intra (qti=0): DC floor=16, AC floor=8.
        let intra = compute_quantization_matrix(&params, 0, 0, 0).unwrap();
        assert_eq!(intra.values[0], 16, "intra DC floor");
        for &v in &intra.values[1..] {
            assert_eq!(v, 8, "intra AC floor");
        }

        // Inter (qti=1): DC floor=32, AC floor=16.
        let inter = compute_quantization_matrix(&params, 1, 0, 0).unwrap();
        assert_eq!(inter.values[0], 32, "inter DC floor");
        for &v in &inter.values[1..] {
            assert_eq!(v, 16, "inter AC floor");
        }
    }

    #[test]
    fn compute_qmat_clamps_at_4096_ceiling() {
        // Large scale drives the product past 4096; step 6(e) clamps.
        // bm=100 everywhere, scale=65535 (max).
        //   scaled = (65535*100//100)*4 = 65535*4 = 262140 -> min 4096.
        let params = single_range_params(vec![[100u8; 64], [100u8; 64]], 0, 1, 65535, 65535);
        let q = compute_quantization_matrix(&params, 0, 0, 0).unwrap();
        for &v in q.values.iter() {
            assert_eq!(v, 4096, "4096 ceiling clamp");
        }
    }

    #[test]
    fn compute_qmat_qmin_table_matches_spec() {
        // Direct check of Table 6.18 values for the four cells.
        assert_eq!(qmin_table(0, 0), 16); // intra DC
        assert_eq!(qmin_table(0, 1), 8); // intra AC
        assert_eq!(qmin_table(0, 63), 8); // intra AC (last)
        assert_eq!(qmin_table(1, 0), 32); // inter DC
        assert_eq!(qmin_table(1, 1), 16); // inter AC
        assert_eq!(qmin_table(1, 63), 16); // inter AC (last)
    }

    #[test]
    fn compute_qmat_rejects_out_of_range_selectors() {
        let params = single_range_params(vec![[16u8; 64], [16u8; 64]], 0, 1, 100, 100);
        match compute_quantization_matrix(&params, 2, 0, 0) {
            Err(Error::QuantTypeIndexOutOfRange { qti: 2 }) => {}
            other => panic!("expected QuantTypeIndexOutOfRange(2), got {other:?}"),
        }
        match compute_quantization_matrix(&params, 0, 3, 0) {
            Err(Error::QuantPlaneIndexOutOfRange { pli: 3 }) => {}
            other => panic!("expected QuantPlaneIndexOutOfRange(3), got {other:?}"),
        }
        match compute_quantization_matrix(&params, 0, 0, 64) {
            Err(Error::QuantIndexOutOfRange { qi: 64 }) => {}
            other => panic!("expected QuantIndexOutOfRange(64), got {other:?}"),
        }
    }

    #[test]
    fn compute_qmat_all_planes_and_types_are_independent() {
        // Distinct base matrices per (qti, pli) confirm the selector
        // wiring reaches the right quant-range row.
        let base = vec![[10u8; 64], [20u8; 64], [30u8; 64], [40u8; 64]];
        let mut qrsizes = [[[0u8; 63]; 3]; 2];
        let mut qrbmis = [[[0u16; 64]; 3]; 2];
        // (0,0): flat bm index 0; (0,1): bm 1; (0,2): bm 2; (1,*): bm 3.
        let pick = [[0u16, 1, 2], [3u16, 3, 3]];
        for qti in 0..2 {
            for pli in 0..3 {
                qrsizes[qti][pli][0] = 63;
                qrbmis[qti][pli][0] = pick[qti][pli];
                qrbmis[qti][pli][1] = pick[qti][pli];
            }
        }
        let params = QuantizationParameters {
            ac_scale: [100u16; 64],
            dc_scale: [100u16; 64],
            num_base_matrices: 4,
            base_matrices: base,
            num_quant_ranges: [[1u8; 3]; 2],
            quant_range_sizes: qrsizes,
            quant_range_base_matrix_indices: qrbmis,
        };
        // BM is flat at the picked matrix value; QMAT = bm*4 (scale 100),
        // floored by Table 6.18.
        let expect = |bm: u16| -> u16 { (bm * 4).max(8) };
        assert_eq!(
            compute_quantization_matrix(&params, 0, 0, 30)
                .unwrap()
                .values[1],
            expect(10)
        );
        assert_eq!(
            compute_quantization_matrix(&params, 0, 1, 30)
                .unwrap()
                .values[1],
            expect(20)
        );
        assert_eq!(
            compute_quantization_matrix(&params, 0, 2, 30)
                .unwrap()
                .values[1],
            expect(30)
        );
        assert_eq!(
            compute_quantization_matrix(&params, 1, 0, 30)
                .unwrap()
                .values[1],
            expect(40)
        );
    }

    #[test]
    fn compute_qmat_chains_from_decoded_quant_params() {
        // End-to-end: decode a synthesized §6.4.2 payload, then compute
        // a §6.4.3 matrix from it. Uses the single-range payload shape
        // from `decode_quant_params_minimal_single_range` (bm0=16,
        // bm1=100, one 63-wide range with endpoints {0,1}).
        let mut ac = [0u16; 64];
        let mut dc = [0u16; 64];
        for i in 0..64 {
            ac[i] = (500 - i * 7) as u16;
            dc[i] = (220 - i * 3) as u16;
        }
        let mut w = BitWriter::new();
        encode_scales(&mut w, &ac, &dc);
        let nbms = 2u32;
        w.put(nbms - 1, 9);
        let bmi_bits = test_ilog(nbms as i64 - 1);
        for _ci in 0..64 {
            w.put(16, 8);
        }
        for _ci in 0..64 {
            w.put(100, 8);
        }
        for qti in 0..2 {
            for pli in 0..3 {
                if qti > 0 || pli > 0 {
                    w.put(1, 1);
                }
                w.put(0, bmi_bits);
                w.put(62, test_ilog(62));
                w.put(1, bmi_bits);
            }
        }
        let payload = w.finish();
        let qp = decode_quantization_parameters(&payload).unwrap();

        // qi=0 -> BM=16; qi=63 -> BM=100 (endpoint selection).
        // Intra DC at qi=0: (dc[0]=220 * 16 // 100)*4 = (3520//100)*4
        //   = 35*4 = 140.
        let q0 = compute_quantization_matrix(&qp, 0, 0, 0).unwrap();
        assert_eq!(q0.values[0], 140);
        // Intra AC at qi=0: (ac[0]=500 * 16 // 100)*4 = (8000//100)*4
        //   = 80*4 = 320.
        assert_eq!(q0.values[1], 320);
    }
}
