//! # oxideav-theora
//!
//! Pure-Rust Theora video codec — clean-room implementation.
//!
//! ## Status — 2026-05-21 (round 2 of clean-room rebuild)
//!
//! Round 1 landed the **identification header** parser per Theora I
//! Specification §6.1 (common header) + §6.2 (identification header).
//! Round 2 adds the **comment header** parser per §6.3 (comment
//! length decode, comment header decode, user comment format).
//!
//! * [`decode_identification_header`] returns a typed
//!   [`TheoraIdentHeader`] describing every field declared in
//!   Figure 6.2.
//! * [`parse_comment_header`] returns a typed
//!   [`TheoraCommentHeader`] with the decoded vendor string and the
//!   list of `KEY=value` user comments.
//!
//! All other public entry points still return [`Error::NotImplemented`].
//!
//! ## Clean-room provenance
//!
//! Source material limited to `docs/video/theora/Theora.pdf` (Xiph
//! Theora I Specification) and the fixture corpus under
//! `docs/video/theora/fixtures/`. No libtheora, no FFmpeg vp3.c, no
//! theora-rs.

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
}
