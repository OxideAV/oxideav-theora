//! # oxideav-theora
//!
//! Pure-Rust Theora video codec — clean-room implementation.
//!
//! ## Status — 2026-05-21 (round 1 of clean-room rebuild)
//!
//! This round lands the **identification header** parser per Theora I
//! Specification §6.1 (common header) + §6.2 (identification header).
//! No setup header, comment header, or video-data packet decode yet —
//! [`decode_identification_header`] returns a typed
//! [`TheoraIdentHeader`] describing every field declared in Figure 6.2.
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
    /// The crate has not yet implemented this surface (planned for a
    /// later round).
    NotImplemented,
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
        let _ = format!("{}", Error::NotImplemented);
    }
}
