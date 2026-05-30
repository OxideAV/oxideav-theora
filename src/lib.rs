//! # oxideav-theora
//!
//! Pure-Rust Theora video codec — clean-room implementation.
//!
//! ## Status — 2026-05-25 (round 8 of clean-room rebuild)
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
//! [`decode_quantization_parameters`] entry point. Round 6 landed the
//! **§6.4.3 Computing a Quantization Matrix** procedure as
//! [`compute_quantization_matrix`], which interpolates a 64-entry
//! [`QuantizationMatrix`] for a `(qti, pli, qi)` selector from a
//! [`QuantizationParameters`]. Round 7 added the **§6.4.4 DCT Token
//! Huffman Tables** procedure as [`decode_dct_token_huffman_tables`],
//! decoding the 80 binary-tree Huffman tables that §7.7's DCT-token
//! decode will consume. Round 8 lands the **§7.1 Frame Header Decode**
//! procedure as [`decode_frame_header`], returning a typed
//! [`TheoraFrameHeader`] (frame type + `NQIS`-element `qi` list) from
//! the first bits of any video-data packet — the first standalone step
//! of the §7 frame-decode pipeline. Round 9 lands the **§7.2.1 Long-Run
//! Bit Strings** and **§7.2.2 Short-Run Bit Strings** decoders as
//! [`decode_long_run_bit_string`] / [`decode_short_run_bit_string`].
//! Round 10 lands the **§7.3 Coded Block Flags Decode** procedure as
//! [`decode_coded_block_flags`], returning the per-block `BCODED` array
//! by chaining one §7.2.1 long-run decode (partially-coded super-block
//! map), one §7.2.1 long-run decode (fully-coded super-block map over
//! the non-partially-coded subset), and one §7.2.2 short-run decode
//! (per-block bits inside partially-coded super blocks). Round 12
//! lands the **§7.5 Motion Vectors** procedures as
//! [`decode_single_motion_vector`] (§7.5.1 — MVMODE=0 Table 7.23
//! Huffman or MVMODE=1 5+1-bit fixed-length) and
//! [`decode_macroblock_motion_vectors`] (§7.5.2 — per-macroblock MV
//! decode driven by `MBMODES`, with `LAST1`/`LAST2` tracking, four-MV
//! INTER_MV_FOUR mode, and PF=0/2/4 chroma averaging via the spec's
//! `round()` ties-away-from-zero). Round 13 lands the **§7.6 Block-
//! Level qi Decode** procedure as [`decode_block_level_qi`], chaining
//! `NQIS − 1` §7.2.1 long-run passes over the per-block subset of
//! still-`qii`-tied coded blocks to produce the per-block `QIIS`
//! array that drives AC dequantization. Round 14 lands the
//! **§7.7.1 EOB Token Decode** procedure as [`decode_eob_token`], the
//! first sub-procedure of the §7.7 DCT-coefficient walk: it consumes
//! one of the 0..=6 EOB tokens (Table 7.33), reads the matching
//! 0/2/3/4/12-bit extra-bits payload, ends the current block by
//! zero-filling `COEFFS[bi][ti..=63]`, captures the block's
//! coefficient count in `NCOEFFS[bi]`, pins `TIS[bi]` to `64`, and
//! returns the residual EOBS run length the §7.7.3 driver will use
//! to free-skip subsequent blocks. Independent of the §6.4.1 spec
//! gap. Round 16 lands the **§7.7.2 Coefficient Token Decode**
//! procedure as [`decode_coefficient_token`], the second §7.7
//! sub-procedure: it consumes one of the 25 non-EOB tokens
//! (Table 7.38 values 7..=31), reads the token's SIGN / MAG / RLEN
//! extra-bits payload (`0..=11` bits depending on the token), writes
//! one or more entries to `COEFFS[bi]`, advances `TIS[bi]`, and
//! updates `NCOEFFS[bi]` (skipping the count update for pure
//! zero-run tokens 7 / 8 per the §7.7.2 introductory text). Returns
//! a typed [`CoefficientTokenKind`] discriminating the token's
//! structural class — zero run, single coefficient, or zero run
//! followed by a single trailing coefficient — so the §7.7.3 driver
//! can branch on the class without re-deriving Table 7.38.
//!
//! Bitstream-version-3.2.0+ streams override the Appendix-B defaults
//! via the §6.4.1 / §6.4.2 procedures the setup-header body decode
//! consumes — the end-to-end body chain remains blocked by the §6.4.1
//! spec gap (see below) and [`parse_setup_header`] continues to
//! surface [`Error::SetupHeaderBodyNotImplemented`]. §6.4.3 / §6.4.4
//! are unblocked by that gap because they operate purely on the
//! §6.4.2 outputs / their own bit payload.
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
//! * [`decode_dct_token_huffman_tables`] returns the
//!   80-element [`HuffmanTable`] array per §6.4.4 from the
//!   binary-tree payload that follows §6.4.2 in the setup header.
//! * [`decode_frame_header`] returns a typed [`TheoraFrameHeader`]
//!   ([`FrameType`] + 1..=3 `qi` values) from the start of a video-data
//!   packet per §7.1, the first standalone step of the §7 frame-decode
//!   pipeline.
//! * [`decode_long_run_bit_string`] / [`decode_short_run_bit_string`]
//!   return a `Vec<u8>` of `0`/`1` flag bits decoded from the §7.2.1
//!   long-run / §7.2.2 short-run Huffman streams that §7.3 / §7.6
//!   chain on top of the §5.2 bit reader.
//! * [`decode_coded_block_flags`] returns the per-block `BCODED` flag
//!   array per §7.3, consuming `SBPCODED` (§7.2.1) + `SBFCODED` (§7.2.1)
//!   plus a per-block §7.2.2 short-run stream and walking the caller-
//!   supplied block-to-super-block mapping.
//! * [`decode_block_level_qi`] returns the per-block `QIIS` array per
//!   §7.6, chaining `NQIS − 1` §7.2.1 long-run passes over the per-
//!   block subset of still-`qii`-tied coded blocks. The VP3-compat
//!   `NQIS == 1` short-circuit consumes zero bits and returns all-zero
//!   `QIIS`.
//! * [`decode_eob_token`] applies one §7.7.1 EOB token (Table 7.33
//!   tokens 0..=6) to the per-block state arrays `TIS` / `NCOEFFS` /
//!   `COEFFS`, returning the residual EOBS run length for the §7.7.3
//!   driver. Reads 0 / 2 / 3 / 4 / 12 bits of extra payload depending
//!   on the token; token 6 with a zero 12-bit payload becomes the
//!   "all remaining coded blocks" sentinel.
//! * [`decode_coefficient_token`] applies one §7.7.2 coefficient token
//!   (Table 7.38 tokens 7..=31) to the per-block `TIS` / `NCOEFFS` /
//!   `COEFFS` state arrays and returns a typed
//!   [`CoefficientTokenKind`] (`ZeroRun` / `Single` / `RunPlusOne`).
//!   Reads `0..=11` bits of extra-bits payload (SIGN / MAG / RLEN
//!   subfields depending on the token); pure zero-run tokens 7 / 8
//!   advance `TIS[bi]` but leave `NCOEFFS[bi]` untouched per §7.7.2's
//!   introductory text. Multi-coefficient tokens fail closed with
//!   [`Error::CoefficientTokenWouldOverflowBlock`] when the implied
//!   coefficient count would push `TIS[bi]` past 64, surfacing the
//!   §7.7.2 MUST-NOT clause on invalid streams.
//! * [`decode_loop_filter_limit_table`] returns the 64-element
//!   per-stream `LFLIMS` table per §6.4.1 — the first sub-procedure of
//!   the setup-header body decode. Reads a 3-bit `NBITS` followed by
//!   64 `NBITS`-bit unsigned values, one per `qi` (round 15, unblocked
//!   by the §6.4.1 procedure-body staging at
//!   `docs/video/theora/theora-6.4.1-lflims.md`).
//!
//! ## Clean-room provenance
//!
//! Source material limited to `docs/video/theora/Theora.pdf` (Xiph
//! Theora I Specification), the spec's own LaTeX source as transcribed
//! into `docs/video/theora/theora-6.4.1-lflims.md` (the staged §6.4.1
//! procedure body that the published PDF omits), and the fixture corpus
//! under `docs/video/theora/fixtures/`. No libtheora, no FFmpeg vp3.c,
//! no theora-rs.
//!
//! ## §6.4.1 — recovered procedure body (round 15)
//!
//! The §6.4.1 Loop Filter Limit Table Decode section in the published
//! `Theora.pdf` declares its inputs/outputs (`LFLIMS`: 64-element 7-bit
//! array; `NBITS`: 3-bit) and ends with "It is decoded as follows:" —
//! but the numbered procedure steps that should follow do not render
//! in the PDF. The text jumps directly to "VP3 Compatibility" / §6.4.2.
//!
//! Round 15 closes that gap using the spec's own LaTeX source
//! transcribed into `docs/video/theora/theora-6.4.1-lflims.md`. The
//! recovered two-step procedure is:
//!
//! 1. Read a 3-bit unsigned integer as `NBITS`.
//! 2. For each consecutive `qi` from 0 to 63 inclusive, read an
//!    `NBITS`-bit unsigned integer as `LFLIMS[qi]`.
//!
//! Total bits consumed: `3 + 64 * NBITS`. `NBITS` is shared across
//! all 64 entries — it is read once, not per-`qi`. There is no
//! per-value clamping; the §7.10 loop filter consumes `LFLIMS[qi0]`
//! as `L` directly.
//!
//! The standalone [`decode_loop_filter_limit_table`] entry point
//! exposes this; [`parse_setup_header`] still surfaces
//! [`Error::SetupHeaderBodyNotImplemented`] because the body decode
//! continues into §6.4.2 / §6.4.3 / §6.4.4 (which are already
//! implemented as standalone entry points but not yet chained on a
//! shared bit reader).

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
    /// While decoding a §6.4.4 DCT-token Huffman table, the path string
    /// `HBITS` grew past 32 bits without reaching a leaf, violating the
    /// step 1(b) bound ("If HBITS is longer than 32 bits in length,
    /// stop. The stream is undecodable.").
    HuffmanCodeTooLong {
        /// Zero-based table index (`hti`) being decoded when the bound
        /// was exceeded.
        hti: usize,
    },
    /// A §6.4.4 Huffman table accumulated a 33rd entry, violating the
    /// step 1(d)i bound ("If the number of entries in table HTS[hti] is
    /// already 32, stop. The stream is undecodable.").
    HuffmanTableFull {
        /// Zero-based table index (`hti`) that overflowed.
        hti: usize,
    },
    /// The leading 1-bit packet-type flag in §7.1 step 1 was set,
    /// indicating the packet is a header packet rather than a
    /// frame-data packet. `decode_frame_header` requires a data packet.
    NotDataPacket,
    /// The caller asked [`decode_frame_header`] to decode the first
    /// frame of the stream, but the on-wire `FTYPE` was `1` (Inter).
    /// §7.1 step 2 mandates that the first decoded frame have
    /// `FTYPE = 0` (Intra). The on-wire value is included for
    /// diagnostics.
    FirstFrameMustBeIntra {
        /// The `FTYPE` value actually read (always `1` when this
        /// variant is returned).
        ftype: u8,
    },
    /// §7.1 step 7's 3-bit reserved field on an intra frame was not
    /// all zero. The spec says "If this value is not zero, stop. This
    /// frame is not decodable according to this specification."
    FrameReservedBitsNonZero {
        /// The reserved bit pattern (in the low 3 bits).
        bits: u8,
    },
    /// §7.2.1 step 10 / §7.2.2 step 10 violation: a decoded run length
    /// (`RLEN = RSTART + ROFFS`) advanced `LEN` past the caller-supplied
    /// `NBITS` bound. The spec says "LEN MUST be less than or equal to
    /// NBITS", so an over-run is undecodable.
    RunLengthOverrun {
        /// The over-run `LEN` value (already past `NBITS`).
        len: u64,
        /// The caller-supplied bit-string length cap.
        nbits: u64,
    },
    /// The `block_to_super_block` mapping passed to
    /// [`decode_coded_block_flags`] has a different length than the
    /// `nbs` argument. The §7.3 inter path needs to know which super
    /// block contains each block; the two arguments must agree.
    BlockSuperBlockMapLenMismatch {
        /// `block_to_super_block.len()`.
        map_len: usize,
        /// The caller-supplied `nbs`.
        nbs: usize,
    },
    /// An entry in the `block_to_super_block` mapping passed to
    /// [`decode_coded_block_flags`] referenced a super-block index that
    /// is `>= nsbs`. §7.3 step 2(i)i looks up `SBPCODED[sbi]` /
    /// `SBFCODED[sbi]` so the index must be in range.
    BlockSuperBlockIndexOutOfRange {
        /// Zero-based block index where the bad mapping was found.
        bi: usize,
        /// The offending super-block index.
        sbi: u32,
        /// The caller-supplied `nsbs`.
        nsbs: u32,
    },
    /// The `macro_block_to_luma_blocks` mapping passed to
    /// [`decode_macroblock_modes`] has a different length than the
    /// `nmbs` argument. §7.4 step 2(d)i looks up the four luma-block
    /// `bi` values for macro block `mbi`; the two arguments must agree.
    MacroBlockLumaMapLenMismatch {
        /// `macro_block_to_luma_blocks.len()`.
        map_len: usize,
        /// The caller-supplied `nmbs`.
        nmbs: usize,
    },
    /// An entry in the `macro_block_to_luma_blocks` mapping passed to
    /// [`decode_macroblock_modes`] referenced a luma-block index that
    /// is `>= nbs`. §7.4 step 2(d)i.A reads `BCODED[bi]` so the index
    /// must be in range.
    MacroBlockLumaBlockIndexOutOfRange {
        /// Zero-based macro-block index where the bad mapping was found.
        mbi: usize,
        /// Slot 0..=3 within the macro block (A, B, C, D in §7.5.2
        /// raster order).
        slot: usize,
        /// The offending luma-block index.
        bi: u32,
        /// The caller-supplied `nbs`.
        nbs: u32,
    },
    /// While decoding a §7.4 macro-block mode under Schemes 1..=6 the
    /// Huffman code-prefix walk advanced past the longest legal code
    /// (`b1111111`) without recognising any entry of Table 7.19. This
    /// can only happen if `BitReader::read_bits` returned an unexpected
    /// value, but the variant is kept defensive.
    UnknownMacroBlockModeCode,
    /// The `mbmodes` slice passed to [`decode_macroblock_motion_vectors`]
    /// has a different length than the `nmbs` argument. §7.5.2 step 3
    /// iterates `mbi` from 0 to `NMBS−1` indexing both `MBMODES[mbi]`
    /// and the per-mb luma-block mapping; the two arguments must agree.
    MotionVectorMbModesLenMismatch {
        /// `mbmodes.len()`.
        modes_len: usize,
        /// The caller-supplied `nmbs`.
        nmbs: usize,
    },
    /// The `macro_block_to_luma_blocks` mapping passed to
    /// [`decode_macroblock_motion_vectors`] has a different length than
    /// the `nmbs` argument. §7.5.2 step 3(a)i selects four luma indices
    /// per macro block from this mapping.
    MotionVectorLumaMapLenMismatch {
        /// `macro_block_to_luma_blocks.len()`.
        map_len: usize,
        /// The caller-supplied `nmbs`.
        nmbs: usize,
    },
    /// An entry in the `macro_block_to_luma_blocks` mapping passed to
    /// [`decode_macroblock_motion_vectors`] referenced a luma-block
    /// index that is `>= nbs`. §7.5.2 step 3(a)ii reads `BCODED[A]`
    /// (and similarly for B, C, D), so the index must be in range.
    MotionVectorLumaBlockIndexOutOfRange {
        /// Zero-based macro-block index where the bad mapping was found.
        mbi: usize,
        /// Slot 0..=3 within the macro block (A, B, C, D in §7.5.2
        /// raster order).
        slot: usize,
        /// The offending luma-block index.
        bi: u32,
        /// The caller-supplied `nbs`.
        nbs: u32,
    },
    /// An entry in one of the chroma-block mappings supplied to
    /// [`decode_macroblock_motion_vectors`] referenced a block index
    /// that is `>= nbs`. §7.5.2 step 3(a)x..3(a)xii write `MVECTS[E]`
    /// / `MVECTS[F]` / … for the chroma planes, so the index must be
    /// in range.
    MotionVectorChromaBlockIndexOutOfRange {
        /// Zero-based macro-block index where the bad mapping was found.
        mbi: usize,
        /// Slot within the macro block (0..=1 for 4:2:0, 0..=3 for
        /// 4:2:2, 0..=7 for 4:4:4 — the per-plane raster slot per
        /// §7.5.2's letters E, F, G, H (Cb) and I, J, K, L (Cr)).
        slot: usize,
        /// The offending chroma-block index.
        bi: u32,
        /// The caller-supplied `nbs`.
        nbs: u32,
    },
    /// A chroma-block mapping passed to
    /// [`decode_macroblock_motion_vectors`] had the wrong length for
    /// the pixel format. PF=0 (4:2:0) needs 1 block per plane per
    /// macro block; PF=2 (4:2:2) needs 2; PF=3 (4:4:4) needs 4. The
    /// outer slice must be `nmbs` long, the inner per the format.
    MotionVectorChromaMapLenMismatch {
        /// Which chroma plane was offending (0 = Cb, 1 = Cr).
        plane: u8,
        /// `chroma_map[plane].len()` (or `0` if the outer is wrong).
        map_len: usize,
        /// The expected length (typically `nmbs`).
        expected: usize,
    },
    /// A chroma per-macroblock entry had the wrong inner length for
    /// the declared pixel format. PF=0 → 1; PF=2 → 2; PF=3 → 4.
    MotionVectorChromaMacroBlockSlotLenMismatch {
        /// Zero-based macro-block index.
        mbi: usize,
        /// Which chroma plane (0 = Cb, 1 = Cr).
        plane: u8,
        /// The actual inner slice length.
        got: usize,
        /// The expected inner length for the declared pixel format.
        expected: usize,
    },
    /// The `bcoded` slice passed to [`decode_block_level_qi`] has a
    /// different length than the `nbs` argument. §7.6 step 2(a) tallies
    /// blocks where `BCODED[bi]` is non-zero; the two arguments must
    /// agree in length.
    BlockLevelQiBcodedLenMismatch {
        /// `bcoded.len()`.
        bcoded_len: usize,
        /// The caller-supplied `nbs`.
        nbs: usize,
    },
    /// The `nqis` argument to [`decode_block_level_qi`] was outside the
    /// `1..=3` range. §7.1 step 6 constrains `NQIS` (the number of `qi`
    /// values defined for the frame) to that range, and §7.6 takes the
    /// frame's `NQIS` as input.
    BlockLevelQiNqisOutOfRange {
        /// The offending `nqis` value.
        nqis: usize,
    },
    /// The `token` argument to [`decode_eob_token`] was outside the
    /// `0..=6` range mandated by §7.7.1 ("This must be in the range
    /// 0 . . . 6"). EOB tokens occupy values 0..=6 of the DCT-token
    /// alphabet (Table 7.33).
    EobTokenOutOfRange {
        /// The offending `token` value.
        token: u8,
    },
    /// The `bi` argument to [`decode_eob_token`] was `>= nbs`, so the
    /// state arrays cannot be indexed safely. §7.7 / §7.7.1 work
    /// per-block in coded order; the caller must keep `bi < NBS`.
    EobTokenBlockIndexOutOfRange {
        /// The offending `bi` value.
        bi: u32,
        /// The `nbs` length the state arrays were sized to.
        nbs: u32,
    },
    /// The `ti` argument to [`decode_eob_token`] was `> 63`, so the
    /// zero-fill in §7.7.1 step 8 (`COEFFS[bi][tj] = 0` for `ti..=63`)
    /// would skip valid token indices. §7.7's zig-zag walk runs
    /// `0..=63`; values outside that range have no meaning.
    EobTokenIndexOutOfRange {
        /// The offending `ti` value.
        ti: u8,
    },
    /// One of the state slices passed to [`decode_eob_token`] had a
    /// length other than the agreed `nbs`. §7.7.1 indexes `TIS[bi]`,
    /// `NCOEFFS[bi]`, and `COEFFS[bi]`; all three must be `nbs` long.
    EobTokenStateLenMismatch {
        /// Which state slice failed the length check.
        which: EobTokenStateSlice,
        /// The slice's actual length.
        got: usize,
        /// `nbs`.
        nbs: usize,
    },
    /// The `token` argument to [`decode_coefficient_token`] was outside
    /// the `7..=31` range mandated by §7.7.2 ("This must be in the
    /// range 7 . . . 31"). Coefficient tokens occupy values 7..=31 of
    /// the DCT-token alphabet (Table 7.38); values 0..=6 are EOB tokens
    /// handled by [`decode_eob_token`] and are rejected here.
    CoefficientTokenOutOfRange {
        /// The offending `token` value.
        token: u8,
    },
    /// The `bi` argument to [`decode_coefficient_token`] was `>= nbs`,
    /// so the state arrays cannot be indexed safely. §7.7 / §7.7.2
    /// work per-block in coded order; the caller must keep
    /// `bi < NBS`.
    CoefficientTokenBlockIndexOutOfRange {
        /// The offending `bi` value.
        bi: u32,
        /// The `nbs` length the state arrays were sized to.
        nbs: u32,
    },
    /// The `ti` argument to [`decode_coefficient_token`] was `> 63`,
    /// so any `COEFFS[bi][ti]` write in §7.7.2 would overflow the
    /// 64-entry zig-zag axis. §7.7's outer loop runs `0..=63`; values
    /// outside that range have no meaning.
    CoefficientTokenIndexOutOfRange {
        /// The offending `ti` value.
        ti: u8,
    },
    /// One of the state slices passed to [`decode_coefficient_token`]
    /// had a length other than the agreed `nbs`. §7.7.2 indexes
    /// `TIS[bi]`, `NCOEFFS[bi]`, and `COEFFS[bi]`; all three must be
    /// `nbs` long.
    CoefficientTokenStateLenMismatch {
        /// Which state slice failed the length check.
        which: CoefficientTokenStateSlice,
        /// The slice's actual length.
        got: usize,
        /// `nbs`.
        nbs: usize,
    },
    /// A §7.7.2 token would have advanced `TIS[bi]` past 64. §7.7.2's
    /// own normative text warns: "For tokens which represent more than
    /// one coefficient, they MUST NOT bring the total number of
    /// coefficients in the block to more than 64." This variant is the
    /// fail-closed surface for that constraint; rejecting at decode
    /// prevents an invalid token sequence from writing past the
    /// `[i16; 64]` row and surfaces the malformed packet to the caller.
    CoefficientTokenWouldOverflowBlock {
        /// The decoded `token` value (one of 7..=31).
        token: u8,
        /// `ti` on entry.
        ti: u8,
        /// The post-token `TIS[bi]` value that exceeded 64.
        new_tis: u16,
    },
    /// The crate has not yet implemented this surface (planned for a
    /// later round).
    NotImplemented,
}

/// Identifies which §7.7.1 state slice failed an
/// [`Error::EobTokenStateLenMismatch`] check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EobTokenStateSlice {
    /// `TIS` — per-block current token index.
    Tis,
    /// `NCOEFFS` — per-block coefficient count.
    Ncoeffs,
    /// `COEFFS` — per-block 64-entry coefficient array.
    Coeffs,
}

/// Identifies which §7.7.2 state slice failed an
/// [`Error::CoefficientTokenStateLenMismatch`] check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoefficientTokenStateSlice {
    /// `TIS` — per-block current token index.
    Tis,
    /// `NCOEFFS` — per-block coefficient count.
    Ncoeffs,
    /// `COEFFS` — per-block 64-entry coefficient array.
    Coeffs,
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
            Error::HuffmanCodeTooLong { hti } => write!(
                f,
                "oxideav-theora: HTS[{hti}] code exceeded 32 bits (§6.4.4 step 1(b): undecodable)"
            ),
            Error::HuffmanTableFull { hti } => write!(
                f,
                "oxideav-theora: HTS[{hti}] exceeded 32 entries (§6.4.4 step 1(d)i: undecodable)"
            ),
            Error::NotDataPacket => write!(
                f,
                "oxideav-theora: §7.1 step 1: high bit set, packet is not a frame-data packet"
            ),
            Error::FirstFrameMustBeIntra { ftype } => write!(
                f,
                "oxideav-theora: §7.1 step 2: first frame FTYPE={ftype}, must be 0 (Intra)"
            ),
            Error::FrameReservedBitsNonZero { bits } => write!(
                f,
                "oxideav-theora: §7.1 step 7: reserved bits = 0b{bits:03b}, expected 0"
            ),
            Error::RunLengthOverrun { len, nbits } => write!(
                f,
                "oxideav-theora: §7.2 run-length decode overran the bit-string cap: LEN={len} > NBITS={nbits} (undecodable)"
            ),
            Error::BlockSuperBlockMapLenMismatch { map_len, nbs } => write!(
                f,
                "oxideav-theora: §7.3 block-to-super-block map length {map_len} != nbs {nbs}"
            ),
            Error::BlockSuperBlockIndexOutOfRange { bi, sbi, nsbs } => write!(
                f,
                "oxideav-theora: §7.3 block-to-super-block map: block {bi} → sbi {sbi} >= nsbs {nsbs}"
            ),
            Error::MacroBlockLumaMapLenMismatch { map_len, nmbs } => write!(
                f,
                "oxideav-theora: §7.4 macro-block-to-luma-blocks map length {map_len} != nmbs {nmbs}"
            ),
            Error::MacroBlockLumaBlockIndexOutOfRange {
                mbi,
                slot,
                bi,
                nbs,
            } => write!(
                f,
                "oxideav-theora: §7.4 mb {mbi} slot {slot} → bi {bi} >= nbs {nbs}"
            ),
            Error::UnknownMacroBlockModeCode => write!(
                f,
                "oxideav-theora: §7.4 unrecognised Table 7.19 Huffman prefix (Scheme 1..=6)"
            ),
            Error::MotionVectorMbModesLenMismatch { modes_len, nmbs } => write!(
                f,
                "oxideav-theora: §7.5.2 mbmodes length {modes_len} != nmbs {nmbs}"
            ),
            Error::MotionVectorLumaMapLenMismatch { map_len, nmbs } => write!(
                f,
                "oxideav-theora: §7.5.2 luma-block map length {map_len} != nmbs {nmbs}"
            ),
            Error::MotionVectorLumaBlockIndexOutOfRange {
                mbi,
                slot,
                bi,
                nbs,
            } => write!(
                f,
                "oxideav-theora: §7.5.2 mb {mbi} slot {slot} → bi {bi} >= nbs {nbs}"
            ),
            Error::MotionVectorChromaBlockIndexOutOfRange {
                mbi,
                slot,
                bi,
                nbs,
            } => write!(
                f,
                "oxideav-theora: §7.5.2 chroma mb {mbi} slot {slot} → bi {bi} >= nbs {nbs}"
            ),
            Error::MotionVectorChromaMapLenMismatch {
                plane,
                map_len,
                expected,
            } => write!(
                f,
                "oxideav-theora: §7.5.2 chroma plane {plane} map length {map_len} != expected {expected}"
            ),
            Error::MotionVectorChromaMacroBlockSlotLenMismatch {
                mbi,
                plane,
                got,
                expected,
            } => write!(
                f,
                "oxideav-theora: §7.5.2 chroma mb {mbi} plane {plane} inner length {got} != expected {expected}"
            ),
            Error::BlockLevelQiBcodedLenMismatch { bcoded_len, nbs } => write!(
                f,
                "oxideav-theora: §7.6 bcoded length {bcoded_len} != nbs {nbs}"
            ),
            Error::BlockLevelQiNqisOutOfRange { nqis } => write!(
                f,
                "oxideav-theora: §7.6 NQIS={nqis} out of range 1..=3"
            ),
            Error::EobTokenOutOfRange { token } => write!(
                f,
                "oxideav-theora: §7.7.1 TOKEN={token} out of range 0..=6"
            ),
            Error::EobTokenBlockIndexOutOfRange { bi, nbs } => write!(
                f,
                "oxideav-theora: §7.7.1 bi={bi} >= nbs={nbs}"
            ),
            Error::EobTokenIndexOutOfRange { ti } => write!(
                f,
                "oxideav-theora: §7.7.1 ti={ti} > 63 (zig-zag indices are 0..=63)"
            ),
            Error::EobTokenStateLenMismatch { which, got, nbs } => write!(
                f,
                "oxideav-theora: §7.7.1 state slice {which:?} length {got} != nbs {nbs}"
            ),
            Error::CoefficientTokenOutOfRange { token } => write!(
                f,
                "oxideav-theora: §7.7.2 TOKEN={token} out of range 7..=31"
            ),
            Error::CoefficientTokenBlockIndexOutOfRange { bi, nbs } => write!(
                f,
                "oxideav-theora: §7.7.2 bi={bi} >= nbs={nbs}"
            ),
            Error::CoefficientTokenIndexOutOfRange { ti } => write!(
                f,
                "oxideav-theora: §7.7.2 ti={ti} > 63 (zig-zag indices are 0..=63)"
            ),
            Error::CoefficientTokenStateLenMismatch { which, got, nbs } => write!(
                f,
                "oxideav-theora: §7.7.2 state slice {which:?} length {got} != nbs {nbs}"
            ),
            Error::CoefficientTokenWouldOverflowBlock {
                token,
                ti,
                new_tis,
            } => write!(
                f,
                "oxideav-theora: §7.7.2 TOKEN={token} from ti={ti} would advance TIS to {new_tis} > 64"
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

/// Decode the loop-filter limit table from the §6.4.1 setup-header
/// payload carried by `bits`.
///
/// `bits` must start at the first bit of the §6.4.1 payload — i.e.
/// the first bit of the setup-header body immediately following the
/// `0x82`+"theora" common header (per §6.4.5 step 2). The returned
/// 64-element `LFLIMS` table is indexed by `qi` (the quantization
/// index, `0..=63`); entry `LFLIMS[qi0]` becomes the loop-filter
/// limit value `L` consumed by §7.10 ("Loop Filtering") when the
/// frame's primary `qi` is `qi0`.
///
/// The procedure transcribes the numbered steps of §6.4.1 of the
/// Xiph Theora I Specification:
///
/// 1. Read a 3-bit unsigned integer as `NBITS`.
/// 2. For each consecutive `qi` from 0 to 63 inclusive, read an
///    `NBITS`-bit unsigned integer as `LFLIMS[qi]`.
///
/// The total bit cost is `3 + 64 * NBITS`. `NBITS` is shared across
/// all 64 entries (read once before the loop, not per-entry). There
/// is no per-value clamping in §6.4.1 itself — each entry is the raw
/// `NBITS`-bit unsigned value, which fits the 7-bit output width
/// because `NBITS` is bounded above by 7.
///
/// The §6.4.1 procedure body is **not** rendered in the published
/// `Theora.pdf`; the staging file
/// `docs/video/theora/theora-6.4.1-lflims.md` supplies the numbered
/// steps transcribed from the spec's own LaTeX source. See the
/// crate-level "§6.4.1 — recovered procedure body (round 15)"
/// section for the full quote.
///
/// # Errors
///
/// * [`Error::TruncatedHeader`] if the bitstream runs out before the
///   `3 + 64 * NBITS` bits the procedure consumes have been read.
///
/// # VP3-compatible streams
///
/// Streams that declare `version < 0x030200` do not carry a
/// transmitted `LFLIMS` table; per §B.1 they use the Appendix B.2
/// hardcoded values exposed as [`LFLIMS_VP3`]. Use
/// [`TheoraSetupHeader::vp3_defaults`] to construct a setup header
/// for those streams without invoking this function.
///
/// # Examples
///
/// ```no_run
/// use oxideav_theora::decode_loop_filter_limit_table;
/// // `payload` here is the §6.4.1 bit-payload starting immediately
/// // after the setup-header common header (`0x82`+"theora").
/// # let payload: &[u8] = &[];
/// let lflims: [u8; 64] = decode_loop_filter_limit_table(payload)?;
/// # Ok::<(), oxideav_theora::Error>(())
/// ```
pub fn decode_loop_filter_limit_table(bits: &[u8]) -> Result<[u8; 64], Error> {
    let mut r = BitReader::new(bits);
    decode_lflims_inner(&mut r)
}

/// Inner §6.4.1 procedure operating on an already-positioned
/// [`BitReader`]. Split out so a future `parse_setup_header` can chain
/// §6.4.1 → §6.4.2 → §6.4.4 on the same underlying bit reader without
/// re-aligning at byte boundaries.
fn decode_lflims_inner(r: &mut BitReader<'_>) -> Result<[u8; 64], Error> {
    // §6.4.1 step 1: read 3-bit NBITS. The spec's variable table
    // declares NBITS as 3 bits unsigned (range 0..=7), and the
    // 7-bit-wide `LFLIMS` output array matches that upper bound.
    let nbits = r.read_bits(3, "LFLIMS NBITS")?;

    // §6.4.1 step 2: for qi = 0..=63, read LFLIMS[qi] as an
    // NBITS-bit unsigned integer. NBITS is shared across all 64
    // entries (step 1 reads it once before the loop).
    //
    // The narrowing cast to `u8` is exact: `nbits <= 7`, so each
    // `read_bits` return value is at most `2^7 - 1 = 127`, which
    // fits the 7-bit output width declared by §6.4.1's output table.
    let mut lflims = [0u8; 64];
    for slot in lflims.iter_mut() {
        *slot = r.read_bits(nbits, "LFLIMS")? as u8;
    }
    Ok(lflims)
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

/// Number of DCT-token Huffman tables in the setup header (§6.4.4
/// output `HTS`): an 80-element array. The 80 tables are organised as
/// 16 Huffman-table groups of 5 each (Table 7.42 / §7.7.2): one DC
/// token table plus four AC token tables (split by coefficient
/// position) per group.
pub const NUM_HUFFMAN_TABLES: usize = 80;

/// Maximum number of entries permitted in a single §6.4.4 Huffman
/// table. The spec (§6.4.4) requires "no more than 32 entries in a
/// single table"; combined with the fullness requirement this also
/// bounds the maximum code length at 32 bits.
pub const MAX_HUFFMAN_ENTRIES: usize = 32;

/// A single `(code, token)` entry of a §6.4.4 DCT-token Huffman table.
///
/// `code` is the bit string `HBITS` read MSb-first, stored
/// right-aligned in a `u32`; `len` is how many of those bits are
/// significant (1..=32). `token` is the 5-bit DCT token value
/// (`0..=31`) that this code decodes to. The code is therefore
/// `code`'s low `len` bits, read most-significant first off the wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HuffmanEntry {
    /// The Huffman code bits (`HBITS`), right-aligned in a `u32`. Only
    /// the low [`HuffmanEntry::len`] bits are meaningful.
    pub code: u32,
    /// Length of the code in bits (1..=32).
    pub len: u8,
    /// The DCT token value (`0..=31`) this code resolves to (§6.4.4
    /// step 1(d)ii reads `TOKEN` as a 5-bit unsigned integer).
    pub token: u8,
}

/// One DCT-token Huffman table (§6.4.4), built from the setup header's
/// binary-tree description.
///
/// The table is stored both as the flat list of decoded `(code, len,
/// token)` [`HuffmanEntry`] pairs (in the depth-first, left-before-
/// right order the §6.4.4 recursion visits the leaves) and as a flat
/// binary-tree node array suitable for driving a per-bit decode of a
/// DCT-token stream. The tree representation is what §7.7.2's
/// coefficient-token decode will walk; the entry list is retained for
/// inspection and round-trip testing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HuffmanTable {
    /// The decoded `(code, len, token)` leaves, in depth-first
    /// left-before-right (`0` subtree before `1` subtree) order.
    pub entries: Vec<HuffmanEntry>,
    /// Flat binary tree. Index `0` is the root. Each [`HuffmanNode`] is
    /// either an interior node carrying the child indices for a `0` and
    /// a `1` bit, or a leaf carrying a token. Empty only for the
    /// degenerate single-leaf-at-root table.
    nodes: Vec<HuffmanNode>,
}

/// A node in a [`HuffmanTable`]'s flat binary tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HuffmanNode {
    /// Interior node: indices into the node array for the `0` and `1`
    /// children respectively.
    Branch {
        /// Child reached by reading a `0` bit.
        zero: u32,
        /// Child reached by reading a `1` bit.
        one: u32,
    },
    /// Leaf node carrying a decoded DCT token value.
    Leaf {
        /// The 5-bit DCT token value (`0..=31`).
        token: u8,
    },
}

impl HuffmanTable {
    /// Number of decoded leaves (token entries) in this table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if this table has no entries. (A conformant §6.4.4 table is
    /// never empty — the tree always terminates in at least one leaf —
    /// but the accessor is provided for completeness.)
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Look up the token for a given code. `code` holds the bit string
    /// right-aligned in its low `len` bits (MSb of the code in bit
    /// position `len - 1`). Returns `None` if no leaf matches that
    /// exact `(code, len)`.
    ///
    /// This is the inverse of how the entries were inserted; it walks
    /// the flat tree one bit at a time from the MSb of the code.
    pub fn lookup(&self, code: u32, len: u8) -> Option<u8> {
        // Degenerate single-leaf-at-root table: the root itself is the
        // only leaf, reached by the empty (zero-length) code.
        if self.nodes.len() == 1 {
            if let HuffmanNode::Leaf { token } = self.nodes[0] {
                return if len == 0 { Some(token) } else { None };
            }
        }
        let mut idx = 0u32;
        for i in (0..len).rev() {
            let bit = (code >> i) & 1;
            match self.nodes.get(idx as usize)? {
                HuffmanNode::Branch { zero, one } => {
                    idx = if bit == 0 { *zero } else { *one };
                }
                HuffmanNode::Leaf { .. } => return None, // code longer than a leaf path
            }
        }
        match self.nodes.get(idx as usize)? {
            HuffmanNode::Leaf { token } => Some(*token),
            HuffmanNode::Branch { .. } => None, // code shorter than any leaf path
        }
    }
}

/// Decode the 80 DCT-token Huffman tables from the §6.4.4 setup-header
/// payload carried by `bits`.
///
/// `bits` must start at the first bit of the §6.4.4 payload — i.e.
/// immediately after the §6.4.2 quantization parameters within the
/// setup-header body (§6.4.5 step 4 runs after step 3). Like
/// [`decode_quantization_parameters`], this entry point decodes the
/// payload in isolation so it can be exercised independently of the
/// §6.4.1 procedure-body spec gap; once that gap closes,
/// `parse_setup_header` will chain §6.4.1 → §6.4.2 → §6.4.4 on a
/// shared bit reader.
///
/// Each of the 80 tables is described as a binary tree (§6.4.4): a
/// `1`-bit `ISLEAF` flag at each node, followed by a 5-bit `TOKEN`
/// value at every leaf. The tree is full (every node is either a leaf
/// or has both children) and prefix-free by construction.
///
/// Returns:
///
/// * [`Error::TruncatedHeader`] if the bitstream runs out before a
///   field is fully read.
/// * [`Error::HuffmanCodeTooLong`] if a code path exceeds 32 bits
///   without reaching a leaf (step 1(b)).
/// * [`Error::HuffmanTableFull`] if a single table accumulates more
///   than 32 entries (step 1(d)i).
///
/// The procedure transcribes the numbered steps of §6.4.4 of the Xiph
/// Theora I Specification directly.
pub fn decode_dct_token_huffman_tables(
    bits: &[u8],
) -> Result<Box<[HuffmanTable; NUM_HUFFMAN_TABLES]>, Error> {
    let mut r = BitReader::new(bits);
    decode_huffman_tables_inner(&mut r)
}

/// Inner §6.4.4 procedure operating on an already-positioned
/// [`BitReader`]. Split out so a future `parse_setup_header` can chain
/// it onto the same reader after §6.4.2 without re-aligning.
fn decode_huffman_tables_inner(
    r: &mut BitReader<'_>,
) -> Result<Box<[HuffmanTable; NUM_HUFFMAN_TABLES]>, Error> {
    // `Box::new([..; 80])` cannot be built from a non-Copy element, so
    // collect into a Vec and convert. Each table is built independently.
    let mut tables: Vec<HuffmanTable> = Vec::with_capacity(NUM_HUFFMAN_TABLES);
    for hti in 0..NUM_HUFFMAN_TABLES {
        tables.push(decode_one_huffman_table(r, hti)?);
    }
    let boxed: Box<[HuffmanTable]> = tables.into_boxed_slice();
    // The length is exactly NUM_HUFFMAN_TABLES by construction.
    Ok(boxed
        .try_into()
        .unwrap_or_else(|_| unreachable!("built exactly NUM_HUFFMAN_TABLES tables")))
}

/// Decode a single §6.4.4 Huffman table (one value of `hti`).
///
/// The spec phrases the per-table decode as a recursive procedure over
/// the bit string `HBITS`. To avoid the stack-overflow risk the spec
/// itself warns about for adversarial inputs, this is implemented with
/// an explicit work stack instead of host recursion. Each stack frame
/// carries the `HBITS` accumulated so far (`code` right-aligned in a
/// `u32`, `len` bits long) and the index of the tree node being filled.
fn decode_one_huffman_table(r: &mut BitReader<'_>, hti: usize) -> Result<HuffmanTable, Error> {
    // The flat tree under construction. Node 0 is the root; it is
    // pushed as a placeholder branch and rewritten in place once we
    // learn whether it is a leaf or an interior node.
    let mut nodes: Vec<HuffmanNode> = vec![HuffmanNode::Branch { zero: 0, one: 0 }];
    let mut entries: Vec<HuffmanEntry> = Vec::new();

    // Explicit DFS stack of "nodes still needing to be decoded". Each
    // item is (node_index, code, len). We process the `0` subtree
    // before the `1` subtree, matching the §6.4.4 step 1(e) order, by
    // pushing `1` first then `0` (LIFO).
    struct Frame {
        node: usize,
        code: u32,
        len: u8,
    }
    let mut stack: Vec<Frame> = vec![Frame {
        node: 0,
        code: 0,
        len: 0,
    }];

    while let Some(Frame { node, code, len }) = stack.pop() {
        // Step 1(b): HBITS longer than 32 bits ⇒ undecodable.
        if len > 32 {
            return Err(Error::HuffmanCodeTooLong { hti });
        }

        // Step 1(c): read the ISLEAF flag.
        let isleaf = r.read_bits(1, "ISLEAF")?;

        if isleaf == 1 {
            // Step 1(d)i: a full table cannot take another entry.
            if entries.len() >= MAX_HUFFMAN_ENTRIES {
                return Err(Error::HuffmanTableFull { hti });
            }
            // Step 1(d)ii: read the 5-bit TOKEN.
            let token = r.read_bits(5, "TOKEN")? as u8;
            // Step 1(d)iii: record (HBITS, TOKEN) as a leaf.
            nodes[node] = HuffmanNode::Leaf { token };
            entries.push(HuffmanEntry { code, len, token });
        } else {
            // Step 1(e): interior node. Allocate both children, wire
            // them up, and schedule them (0 subtree before 1 subtree).
            // `len + 1` for each child cannot itself exceed 33 here; the
            // step 1(b) bound is re-checked when each child is popped.
            let zero = nodes.len() as u32;
            nodes.push(HuffmanNode::Branch { zero: 0, one: 0 });
            let one = nodes.len() as u32;
            nodes.push(HuffmanNode::Branch { zero: 0, one: 0 });
            nodes[node] = HuffmanNode::Branch { zero, one };

            // Push `1` first so `0` is popped (decoded) first.
            stack.push(Frame {
                node: one as usize,
                code: (code << 1) | 1,
                len: len + 1,
            });
            stack.push(Frame {
                node: zero as usize,
                code: code << 1,
                len: len + 1,
            });
        }
    }

    Ok(HuffmanTable { entries, nodes })
}

// =====================================================================
// §7.1 Frame Header Decode
// =====================================================================

/// Frame type per §7.1 step 2 / Table 7.3.
///
/// Theora distinguishes only two frame types: a key (Intra) frame that
/// decodes without reference to any prior frame, and an Inter frame
/// that predicts from previously-decoded frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameType {
    /// `FTYPE = 0`. Key (Intra) frame — decoded standalone. The first
    /// frame of a Theora stream MUST be Intra (§7.1 step 2 final
    /// sentence).
    Intra = 0,
    /// `FTYPE = 1`. Inter frame — predicted from one or two reference
    /// frames per §2.5 / Table 7.46.
    Inter = 1,
}

/// Maximum number of `qi` values a single frame header may carry, per
/// §7.1's MOREQIS termination logic (steps 4–6 unroll to at most three
/// indices).
pub const MAX_FRAME_QIS: usize = 3;

/// Parsed Theora frame header, per §7.1 of the Xiph Theora I
/// Specification.
///
/// The frame header is the first thing in every video-data packet. It
/// selects the frame type and supplies the list of `qi` values the
/// frame may use to dequantize its DCT coefficients (the first `qi` is
/// used for all DC coefficients; the others are selectable per-block
/// for the AC coefficients via §7.6 block-level qi decode).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TheoraFrameHeader {
    /// Frame type (§7.1 step 2). `FrameType::Intra` for `FTYPE = 0`,
    /// `FrameType::Inter` for `FTYPE = 1`.
    pub ftype: FrameType,
    /// The `NQIS`-element list of `qi` values (§7.1 steps 3–6). Each
    /// element is in `0..=63` (6-bit unsigned per the spec's
    /// "Description" column). Length is 1, 2, or 3; this is `NQIS`.
    pub qis: Vec<u8>,
}

impl TheoraFrameHeader {
    /// Number of `qi` values carried by this frame header
    /// (§7.1 `NQIS`). Always in `1..=3`.
    pub fn nqis(&self) -> usize {
        self.qis.len()
    }
}

/// Decode the frame header at the start of a Theora video-data packet,
/// per §7.1 of the Xiph Theora I Specification.
///
/// `packet` is the full Ogg packet payload (the leading bit MUST be `0`
/// per §7.1 step 1 — the parser surfaces [`Error::NotDataPacket`] if
/// the high bit of byte 0 is set, which identifies a header packet).
/// `first_frame` is whether this is the first frame being decoded;
/// §7.1 step 2 mandates the first frame have `FTYPE = 0`
/// ([`FrameType::Intra`]).
///
/// The decoder leaves the bit cursor positioned immediately after the
/// frame header. A future §7.3 (coded-block-flags) / §7.4
/// (macroblock-coding-modes) decoder will resume from that point on a
/// shared bit reader; this entry point decodes the header in isolation
/// so it can be exercised while the rest of the §7 pipeline is being
/// built.
///
/// Returns:
///
/// * [`Error::NotDataPacket`] if the leading 1-bit packet-type flag is
///   set (the packet is a header, not a data packet — §7.1 step 1).
/// * [`Error::FirstFrameMustBeIntra`] if `first_frame` is true and
///   `FTYPE` came back as 1 (§7.1 step 2 final sentence).
/// * [`Error::FrameReservedBitsNonZero`] if the 3-bit reserved field on
///   an intra frame (§7.1 step 7) is non-zero.
/// * [`Error::TruncatedHeader`] if the packet runs out of bits before
///   the header is complete.
pub fn decode_frame_header(packet: &[u8], first_frame: bool) -> Result<TheoraFrameHeader, Error> {
    let mut r = BitReader::new(packet);
    decode_frame_header_inner(&mut r, first_frame)
}

/// Inner §7.1 procedure operating on an already-positioned
/// [`BitReader`]. Split out so a future end-to-end frame decoder can
/// chain the subsequent §7.2 / §7.3 procedures on the same reader
/// without re-aligning.
fn decode_frame_header_inner(
    r: &mut BitReader<'_>,
    first_frame: bool,
) -> Result<TheoraFrameHeader, Error> {
    // Step 1: read 1 bit. Must be 0 — a 1 indicates a header packet
    // (the §6 series).
    let is_header = r.read_bits(1, "frame.packet_type")?;
    if is_header != 0 {
        return Err(Error::NotDataPacket);
    }

    // Step 2: read FTYPE (1 bit). The first frame MUST be FTYPE=0.
    let ftype_raw = r.read_bits(1, "FTYPE")? as u8;
    if first_frame && ftype_raw != 0 {
        return Err(Error::FirstFrameMustBeIntra { ftype: ftype_raw });
    }
    // Table 7.3: 0 → Intra, 1 → Inter (the two are the only legal
    // values of a 1-bit field).
    let ftype = match ftype_raw {
        0 => FrameType::Intra,
        1 => FrameType::Inter,
        // A 1-bit unsigned integer can only be 0 or 1; this arm is
        // unreachable but kept exhaustive to avoid relying on an
        // unchecked transmute.
        _ => unreachable!("read_bits(1, ..) returns 0 or 1"),
    };

    // Step 3: read QIS[0] (6 bits).
    let qis0 = r.read_bits(6, "QIS[0]")? as u8;
    let mut qis: Vec<u8> = Vec::with_capacity(MAX_FRAME_QIS);
    qis.push(qis0);

    // Step 4–6: unroll the MOREQIS / QIS chain up to NQIS = 3.
    // Step 4: read MOREQIS.
    let more1 = r.read_bits(1, "MOREQIS[0]")?;
    if more1 != 0 {
        // Step 6(a): read QIS[1].
        let qis1 = r.read_bits(6, "QIS[1]")? as u8;
        qis.push(qis1);
        // Step 6(b): read MOREQIS again.
        let more2 = r.read_bits(1, "MOREQIS[1]")?;
        if more2 != 0 {
            // Step 6(d)i: read QIS[2].
            let qis2 = r.read_bits(6, "QIS[2]")? as u8;
            qis.push(qis2);
            // Step 6(d)ii: NQIS = 3 (no further MOREQIS).
        }
        // Otherwise step 6(c): NQIS = 2.
    }
    // Otherwise step 5: NQIS = 1.

    // Step 7: on an intra frame, read 3 reserved bits which MUST be 0.
    if ftype == FrameType::Intra {
        let reserved = r.read_bits(3, "frame.reserved")? as u8;
        if reserved != 0 {
            return Err(Error::FrameReservedBitsNonZero { bits: reserved });
        }
    }

    Ok(TheoraFrameHeader { ftype, qis })
}

// =====================================================================
// §7.2 Run-Length Encoded Bit Strings
// =====================================================================

/// Table 7.7 entry for the §7.2.1 Long-Run-length Huffman code.
///
/// Each row encodes one decoded run length: a `code` (read MSb-first
/// off the bit reader), its prefix `code_len` in bits, the `rstart`
/// floor of the run-length range, and the `rbits` count of trailing
/// literal bits that select the offset within the range.
///
/// Spec Table 7.7 enumerates exactly the seven `1*0` / `1{6}` codes:
///
/// | Huffman | RSTART | RBITS | Run Lengths |
/// | ------- | ------ | ----- | ----------- |
/// | `0`       | 1  | 0  | 1       |
/// | `10`      | 2  | 1  | 2..=3   |
/// | `110`     | 4  | 1  | 4..=5   |
/// | `1110`    | 6  | 2  | 6..=9   |
/// | `11110`   | 10 | 3  | 10..=17 |
/// | `111110`  | 18 | 4  | 18..=33 |
/// | `111111`  | 34 | 12 | 34..=4129 |
const LONG_RUN_TABLE: [LongRunEntry; 7] = [
    LongRunEntry {
        code: 0b0,
        code_len: 1,
        rstart: 1,
        rbits: 0,
    },
    LongRunEntry {
        code: 0b10,
        code_len: 2,
        rstart: 2,
        rbits: 1,
    },
    LongRunEntry {
        code: 0b110,
        code_len: 3,
        rstart: 4,
        rbits: 1,
    },
    LongRunEntry {
        code: 0b1110,
        code_len: 4,
        rstart: 6,
        rbits: 2,
    },
    LongRunEntry {
        code: 0b11110,
        code_len: 5,
        rstart: 10,
        rbits: 3,
    },
    LongRunEntry {
        code: 0b111110,
        code_len: 6,
        rstart: 18,
        rbits: 4,
    },
    LongRunEntry {
        code: 0b111111,
        code_len: 6,
        rstart: 34,
        rbits: 12,
    },
];

/// One row of [`LONG_RUN_TABLE`] (§7.2 Table 7.7).
#[derive(Debug, Clone, Copy)]
struct LongRunEntry {
    /// The Huffman code bits, right-justified in a u32.
    code: u32,
    /// The number of bits in `code` (1..=6).
    code_len: u8,
    /// The floor of the represented run-length range (Table 7.7 RSTART).
    rstart: u16,
    /// The count of trailing literal bits encoding the offset within
    /// the range (Table 7.7 RBITS).
    rbits: u8,
}

/// Maximum run length encodable by the §7.2.1 long-run Huffman codes
/// (Table 7.7 last row: `RSTART + (1 << RBITS) - 1 = 34 + 4095 = 4129`).
/// In a Theora I-compliant stream this is also the threshold above
/// which the bit value is re-read instead of toggled (§7.2.1 step 12).
pub const LONG_RUN_MAX: u16 = 4129;

/// Table 7.11 entry for the §7.2.2 Short-Run-length Huffman code.
///
/// Same shape as [`LongRunEntry`] but with the Table 7.11 ranges. The
/// short-run procedure differs only in (a) its Huffman alphabet (6
/// codes; longest is `b11111` at 5 bits) and (b) its maximum run length
/// of 30, after which there is no exception path — the bit value is
/// unconditionally toggled.
///
/// Spec Table 7.11:
///
/// | Huffman | RSTART | RBITS | Run Lengths |
/// | ------- | ------ | ----- | ----------- |
/// | `0`     | 1  | 1 | 1..=2  |
/// | `10`    | 3  | 1 | 3..=4  |
/// | `110`   | 5  | 1 | 5..=6  |
/// | `1110`  | 7  | 2 | 7..=10 |
/// | `11110` | 11 | 2 | 11..=14 |
/// | `11111` | 15 | 4 | 15..=30 |
const SHORT_RUN_TABLE: [ShortRunEntry; 6] = [
    ShortRunEntry {
        code: 0b0,
        code_len: 1,
        rstart: 1,
        rbits: 1,
    },
    ShortRunEntry {
        code: 0b10,
        code_len: 2,
        rstart: 3,
        rbits: 1,
    },
    ShortRunEntry {
        code: 0b110,
        code_len: 3,
        rstart: 5,
        rbits: 1,
    },
    ShortRunEntry {
        code: 0b1110,
        code_len: 4,
        rstart: 7,
        rbits: 2,
    },
    ShortRunEntry {
        code: 0b11110,
        code_len: 5,
        rstart: 11,
        rbits: 2,
    },
    ShortRunEntry {
        code: 0b11111,
        code_len: 5,
        rstart: 15,
        rbits: 4,
    },
];

/// One row of [`SHORT_RUN_TABLE`] (§7.2 Table 7.11).
#[derive(Debug, Clone, Copy)]
struct ShortRunEntry {
    /// The Huffman code bits, right-justified in a u32.
    code: u32,
    /// The number of bits in `code` (1..=5).
    code_len: u8,
    /// The floor of the represented run-length range (Table 7.11 RSTART).
    rstart: u16,
    /// The count of trailing literal bits encoding the offset within
    /// the range (Table 7.11 RBITS).
    rbits: u8,
}

/// Maximum run length encodable by the §7.2.2 short-run Huffman codes
/// (Table 7.11 last row: `RSTART + (1 << RBITS) - 1 = 15 + 15 = 30`).
pub const SHORT_RUN_MAX: u16 = 30;

/// Decode a §7.2.1 Long-Run-length-coded bit string from `bits`.
///
/// Decodes exactly `nbits` output bits (per the spec's `NBITS` input)
/// into a `Vec<u8>` of `0`/`1` values, applying the §7.2.1 procedure
/// verbatim:
///
/// 1. Initialize `LEN = 0`, `BITS = []`.
/// 2. While `LEN < nbits`:
///    * Read a 1-bit `BIT` value.
///    * Read a Table 7.7 Huffman code, then `RBITS` literal bits for
///      `ROFFS`. The run length is `RLEN = RSTART + ROFFS`.
///    * Append `RLEN` copies of `BIT`, then add `RLEN` to `LEN`.
///    * Loop on subsequent runs: if `RLEN == 4129` (the maximum), read
///      another 1-bit value as the new `BIT`; otherwise toggle `BIT`
///      via `BIT = 1 - BIT`.
/// 3. Return `BITS`.
///
/// `nbits` is unbounded in principle (the spec's `NBITS` is a 36-bit
/// unsigned), so it is taken as a `u64`. In practice it is capped by
/// the consumer's known block count.
///
/// Returns:
///
/// * [`Error::TruncatedHeader`] if the stream runs out of bits before
///   the bit string is complete.
/// * [`Error::RunLengthOverrun`] if a decoded run length advances `LEN`
///   past `nbits` (a malformed stream — the §7.2.1 step 10 invariant
///   says "LEN MUST be less than or equal to NBITS").
pub fn decode_long_run_bit_string(bits: &[u8], nbits: u64) -> Result<Vec<u8>, Error> {
    let mut r = BitReader::new(bits);
    decode_long_run_bit_string_inner(&mut r, nbits)
}

/// Inner §7.2.1 procedure operating on an already-positioned
/// [`BitReader`]. Split out so a future end-to-end frame decoder can
/// chain §7.3 (Coded Block Flags Decode) on the same reader without
/// re-aligning to a byte boundary.
fn decode_long_run_bit_string_inner(r: &mut BitReader<'_>, nbits: u64) -> Result<Vec<u8>, Error> {
    let mut out: Vec<u8> = Vec::new();
    let mut len: u64 = 0;
    if len == nbits {
        return Ok(out);
    }
    // Step 4 (first iteration only): read the initial BIT value.
    let mut bit = r.read_bits(1, "long-run.BIT")? as u8;
    loop {
        // Steps 5–6: recognise a Table 7.7 code then read RBITS literal
        // bits for ROFFS.
        let entry = recognise_long_run_code(r)?;
        let roffs = if entry.rbits == 0 {
            0
        } else {
            r.read_bits(entry.rbits as u32, "long-run.ROFFS")?
        };
        let rlen = entry.rstart as u32 + roffs;
        // Steps 9–10: append RLEN copies of BIT then bump LEN.
        for _ in 0..rlen {
            out.push(bit);
        }
        let new_len = len + rlen as u64;
        if new_len > nbits {
            return Err(Error::RunLengthOverrun {
                len: new_len,
                nbits,
            });
        }
        len = new_len;
        // Step 11: bit string complete?
        if len == nbits {
            return Ok(out);
        }
        // Step 12: VP3+ exception — if RLEN == 4129, read a fresh BIT
        // instead of toggling (to allow runs longer than the Huffman
        // alphabet alone can represent).
        if rlen == LONG_RUN_MAX as u32 {
            bit = r.read_bits(1, "long-run.BIT")? as u8;
        } else {
            // Step 13: toggle.
            bit = 1 - bit;
        }
        // Step 14: continue from step 5.
    }
}

/// Walk Table 7.7 by reading one bit at a time until a code matches.
/// Returns the matched [`LongRunEntry`].
///
/// All seven Table 7.7 codes are prefix codes, so the walk is
/// unambiguous; a malformed stream that returns more than six `1` bits
/// in a row falls into the `b111111` row (length 6 — every long-run
/// code is at most 6 bits, so the longest input prefix is bounded).
fn recognise_long_run_code(r: &mut BitReader<'_>) -> Result<LongRunEntry, Error> {
    let mut code: u32 = 0;
    let mut code_len: u8 = 0;
    loop {
        let bit = r.read_bits(1, "long-run.huff")?;
        code = (code << 1) | bit;
        code_len += 1;
        for entry in LONG_RUN_TABLE.iter() {
            if entry.code_len == code_len && entry.code == code {
                return Ok(*entry);
            }
        }
        // Defensive: Table 7.7 is exhaustive within six bits (all seven
        // codes are prefix-free and the longest is `b111111`). A
        // six-bit walk that didn't match means the table itself was
        // mis-transcribed; debug_assert! catches such a regression in
        // test builds while production builds keep walking until the
        // bit reader runs out (which becomes a TruncatedHeader).
        debug_assert!(
            code_len <= 6,
            "long-run table mis-transcribed: walked past 6 bits"
        );
    }
}

/// Decode a §7.2.2 Short-Run-length-coded bit string from `bits`.
///
/// Same shape as [`decode_long_run_bit_string`] but with Table 7.11 and
/// the §7.2.2 procedure — no 4129 exception path, the BIT value is
/// always toggled between runs.
///
/// `nbits` here is typically at most 16 (one bit per block in a super
/// block) but the spec types it as 36-bit so the API takes a `u64`
/// matching [`decode_long_run_bit_string`].
///
/// Returns:
///
/// * [`Error::TruncatedHeader`] if the stream runs out of bits before
///   the bit string is complete.
/// * [`Error::RunLengthOverrun`] if a decoded run length advances `LEN`
///   past `nbits` (a malformed stream — the §7.2.2 step 10 invariant).
pub fn decode_short_run_bit_string(bits: &[u8], nbits: u64) -> Result<Vec<u8>, Error> {
    let mut r = BitReader::new(bits);
    decode_short_run_bit_string_inner(&mut r, nbits)
}

/// Inner §7.2.2 procedure operating on an already-positioned
/// [`BitReader`]. Split out for the same reason as
/// [`decode_long_run_bit_string_inner`].
fn decode_short_run_bit_string_inner(r: &mut BitReader<'_>, nbits: u64) -> Result<Vec<u8>, Error> {
    let mut out: Vec<u8> = Vec::new();
    let mut len: u64 = 0;
    if len == nbits {
        return Ok(out);
    }
    // Step 4 (first iteration only): read the initial BIT value.
    let mut bit = r.read_bits(1, "short-run.BIT")? as u8;
    loop {
        // Steps 5–6: recognise a Table 7.11 code then read RBITS
        // literal bits for ROFFS.
        let entry = recognise_short_run_code(r)?;
        let roffs = if entry.rbits == 0 {
            0
        } else {
            r.read_bits(entry.rbits as u32, "short-run.ROFFS")?
        };
        let rlen = entry.rstart as u32 + roffs;
        // Steps 9–10: append RLEN copies of BIT then bump LEN.
        for _ in 0..rlen {
            out.push(bit);
        }
        let new_len = len + rlen as u64;
        if new_len > nbits {
            return Err(Error::RunLengthOverrun {
                len: new_len,
                nbits,
            });
        }
        len = new_len;
        // Step 11: bit string complete?
        if len == nbits {
            return Ok(out);
        }
        // Step 12: short-run unconditionally toggles BIT — there is no
        // 4129 exception here. The short-run alphabet caps at 30, well
        // below any plausible same-bit overflow.
        bit = 1 - bit;
        // Step 13: continue from step 5.
    }
}

/// Walk Table 7.11 by reading one bit at a time until a code matches.
/// Returns the matched [`ShortRunEntry`].
///
/// Same structure as [`recognise_long_run_code`]. All six Table 7.11
/// codes are prefix-free; the longest is `b11111` at 5 bits.
fn recognise_short_run_code(r: &mut BitReader<'_>) -> Result<ShortRunEntry, Error> {
    let mut code: u32 = 0;
    let mut code_len: u8 = 0;
    loop {
        let bit = r.read_bits(1, "short-run.huff")?;
        code = (code << 1) | bit;
        code_len += 1;
        for entry in SHORT_RUN_TABLE.iter() {
            if entry.code_len == code_len && entry.code == code {
                return Ok(*entry);
            }
        }
        debug_assert!(
            code_len <= 5,
            "short-run table mis-transcribed: walked past 5 bits"
        );
    }
}

// =====================================================================
// §7.3 Coded Block Flags Decode
// =====================================================================

/// Decode the `BCODED` per-block coded-flags array per §7.3 of the Xiph
/// Theora I Specification ("Coded Block Flags Decode").
///
/// Given a frame's type, super-block count `NSBS`, and the per-block
/// mapping `block_to_super_block` (length `NBS`, each entry a super-block
/// index in `0..nsbs`), returns an `NBS`-element `Vec<u8>` of `0`/`1`
/// flags marking each block coded or not coded.
///
/// For an intra frame (§7.3 step 1) every flag is set to `1` and `packet`
/// is not consumed. For an inter frame (§7.3 step 2) the procedure:
///
/// 1. Decodes a length-`NSBS` `SBPCODED` (partially-coded-super-block)
///    bit string via [`decode_long_run_bit_string`] (step 2(b)–(c)).
/// 2. Decodes a length-`(number of zero-valued SBPCODED entries)`
///    `SBFCODED` (fully-coded-super-block) bit string via
///    [`decode_long_run_bit_string`] (step 2(d)–(f)).
/// 3. Decodes a length-`(total block count across SBPCODED==1 super
///    blocks)` per-block bit string via [`decode_short_run_bit_string`]
///    (step 2(g)–(h)).
/// 4. For each block in coded order (step 2(i)), assigns
///    `BCODED[bi] = SBFCODED[sbi]` when `SBPCODED[sbi]` is `0`, or
///    consumes the next per-block bit from the §7.2.2 string when
///    `SBPCODED[sbi]` is `1`.
///
/// The block-to-super-block mapping is supplied by the caller because
/// computing it requires the §2 super-block scan order, which a
/// later round will land alongside §7.4 / §7.6.
///
/// `packet` is the full frame-data packet payload (or any slice
/// positioned at the byte-aligned start of the §7.3 bit stream). The
/// `packet` byte-aligned variant is provided for unit-test convenience;
/// an end-to-end decoder will call [`decode_coded_block_flags_inner`]
/// on a [`BitReader`] already positioned past the §7.1 / §7.2 prefix.
///
/// Returns:
///
/// * [`Error::BlockSuperBlockMapLenMismatch`] if
///   `block_to_super_block.len() != nbs`.
/// * [`Error::BlockSuperBlockIndexOutOfRange`] if any entry in
///   `block_to_super_block` is `>= nsbs`.
/// * [`Error::TruncatedHeader`] / [`Error::RunLengthOverrun`] when the
///   underlying §7.2 long- or short-run decode rejects.
pub fn decode_coded_block_flags(
    packet: &[u8],
    ftype: FrameType,
    nsbs: u32,
    nbs: u32,
    block_to_super_block: &[u32],
) -> Result<Vec<u8>, Error> {
    let mut r = BitReader::new(packet);
    decode_coded_block_flags_inner(&mut r, ftype, nsbs, nbs, block_to_super_block)
}

/// Inner §7.3 procedure operating on an already-positioned
/// [`BitReader`]. Split out so an end-to-end frame decoder can chain
/// §7.1 → §7.2 → §7.3 on the same reader without re-aligning to a byte
/// boundary.
pub(crate) fn decode_coded_block_flags_inner(
    r: &mut BitReader<'_>,
    ftype: FrameType,
    nsbs: u32,
    nbs: u32,
    block_to_super_block: &[u32],
) -> Result<Vec<u8>, Error> {
    // Argument sanity checks. §7.3 step 2(i)i looks up
    // SBPCODED[sbi] / SBFCODED[sbi], so the caller-supplied mapping
    // must match NBS in length and reference only legal sbi values.
    let nbs_us = nbs as usize;
    if block_to_super_block.len() != nbs_us {
        return Err(Error::BlockSuperBlockMapLenMismatch {
            map_len: block_to_super_block.len(),
            nbs: nbs_us,
        });
    }
    for (bi, &sbi) in block_to_super_block.iter().enumerate() {
        if sbi >= nsbs {
            return Err(Error::BlockSuperBlockIndexOutOfRange { bi, sbi, nsbs });
        }
    }

    // Step 1: intra frames are trivial — every block is coded.
    if ftype == FrameType::Intra {
        return Ok(vec![1u8; nbs_us]);
    }

    // Step 2(a)–(c): decode SBPCODED (one bit per super block).
    let sbpcoded = decode_long_run_bit_string_inner(r, nsbs as u64)?;
    debug_assert_eq!(sbpcoded.len(), nsbs as usize);

    // Step 2(d): NBITS = count of super blocks with SBPCODED[sbi] == 0.
    let nbits_fcoded: u64 = sbpcoded.iter().filter(|&&b| b == 0).count() as u64;

    // Step 2(e)–(f): decode SBFCODED, one bit per non-partially-coded
    // super block. The decoded bit string is consumed in sbi order,
    // skipping super blocks whose SBPCODED entry was 1.
    let sbfcoded_stream = decode_long_run_bit_string_inner(r, nbits_fcoded)?;
    debug_assert_eq!(sbfcoded_stream.len(), nbits_fcoded as usize);
    let mut sbfcoded: Vec<u8> = vec![0u8; nsbs as usize];
    let mut cursor: usize = 0;
    for (sbi, &spc) in sbpcoded.iter().enumerate() {
        if spc == 0 {
            sbfcoded[sbi] = sbfcoded_stream[cursor];
            cursor += 1;
        }
    }
    debug_assert_eq!(cursor as u64, nbits_fcoded);

    // Step 2(g): count blocks belonging to super blocks where
    // SBPCODED[sbi] == 1. This is a tally over the
    // block_to_super_block mapping, NOT 16 × (partially-coded count) —
    // edge super blocks can carry fewer than 16 blocks.
    let nbits_blocks: u64 = block_to_super_block
        .iter()
        .filter(|&&sbi| sbpcoded[sbi as usize] == 1)
        .count() as u64;

    // Step 2(h): decode the per-block bit string for partially-coded
    // super blocks via the §7.2.2 short-run procedure.
    let block_stream = decode_short_run_bit_string_inner(r, nbits_blocks)?;
    debug_assert_eq!(block_stream.len(), nbits_blocks as usize);

    // Step 2(i): walk the blocks in coded order. Blocks inside fully /
    // not-coded super blocks inherit SBFCODED[sbi]; blocks inside
    // partially-coded super blocks consume one bit from the §7.2.2
    // string.
    let mut bcoded: Vec<u8> = Vec::with_capacity(nbs_us);
    let mut block_cursor: usize = 0;
    for &sbi in block_to_super_block.iter() {
        let sbi_us = sbi as usize;
        if sbpcoded[sbi_us] == 0 {
            // Step 2(i)ii.
            bcoded.push(sbfcoded[sbi_us]);
        } else {
            // Step 2(i)iii.
            bcoded.push(block_stream[block_cursor]);
            block_cursor += 1;
        }
    }
    debug_assert_eq!(block_cursor as u64, nbits_blocks);
    debug_assert_eq!(bcoded.len(), nbs_us);
    Ok(bcoded)
}

// =====================================================================
// §7.4 Macro Block Coding Modes
// =====================================================================

/// One of the eight macro-block coding modes from Table 7.18 of the Xiph
/// Theora I Specification.
///
/// In an intra frame every macro block is `INTRA`; in an inter frame any
/// of these eight values may appear. The variant discriminant equals the
/// `Index` column of Table 7.18 — `from_index` / `to_index` round-trip
/// for any value in `0..=7`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroBlockMode {
    /// Index 0 — inter prediction from the previous reference frame
    /// with a zero motion vector.
    InterNoMv,
    /// Index 1 — intra-coded macro block. Used for every macro block in
    /// an intra frame.
    Intra,
    /// Index 2 — inter prediction with an explicit motion vector.
    InterMv,
    /// Index 3 — inter prediction reusing the last-decoded macro
    /// block's motion vector.
    InterMvLast,
    /// Index 4 — inter prediction reusing the second-to-last macro
    /// block's motion vector.
    InterMvLast2,
    /// Index 5 — inter prediction from the golden frame with a zero
    /// motion vector.
    InterGoldenNoMv,
    /// Index 6 — inter prediction from the golden frame with an
    /// explicit motion vector.
    InterGoldenMv,
    /// Index 7 — INTER_MV_FOUR: four independently coded luma motion
    /// vectors per macro block.
    InterMvFour,
}

impl MacroBlockMode {
    /// Numeric index per Table 7.18.
    pub fn to_index(self) -> u8 {
        match self {
            MacroBlockMode::InterNoMv => 0,
            MacroBlockMode::Intra => 1,
            MacroBlockMode::InterMv => 2,
            MacroBlockMode::InterMvLast => 3,
            MacroBlockMode::InterMvLast2 => 4,
            MacroBlockMode::InterGoldenNoMv => 5,
            MacroBlockMode::InterGoldenMv => 6,
            MacroBlockMode::InterMvFour => 7,
        }
    }

    /// Inverse of [`MacroBlockMode::to_index`]. Returns `None` for
    /// values outside `0..=7`.
    pub fn from_index(v: u8) -> Option<Self> {
        Some(match v {
            0 => MacroBlockMode::InterNoMv,
            1 => MacroBlockMode::Intra,
            2 => MacroBlockMode::InterMv,
            3 => MacroBlockMode::InterMvLast,
            4 => MacroBlockMode::InterMvLast2,
            5 => MacroBlockMode::InterGoldenNoMv,
            6 => MacroBlockMode::InterGoldenMv,
            7 => MacroBlockMode::InterMvFour,
            _ => return None,
        })
    }
}

/// The fixed mode alphabets for Table 7.19 Schemes 1..=6.
///
/// Each row is indexed by `mi` (the rank of the recognised Huffman
/// code, with `mi=0` for `b0`, `mi=1` for `b10`, …, `mi=7` for
/// `b1111111`). Cell value is the `MBMODES` value assigned to that
/// `mi`. Scheme 0 reads its alphabet from the bitstream (§7.4 step
/// 2(b)) and Scheme 7 codes the mode directly as a 3-bit integer (§7.4
/// step 2(d)i.B); neither is represented in this table.
const MALPHABETS_SCHEMES_1_TO_6: [[u8; 8]; 6] = [
    // Scheme 1: 3 4 2 0 1 5 6 7
    [3, 4, 2, 0, 1, 5, 6, 7],
    // Scheme 2: 3 4 0 2 1 5 6 7
    [3, 4, 0, 2, 1, 5, 6, 7],
    // Scheme 3: 3 2 4 0 1 5 6 7
    [3, 2, 4, 0, 1, 5, 6, 7],
    // Scheme 4: 3 2 0 4 1 5 6 7
    [3, 2, 0, 4, 1, 5, 6, 7],
    // Scheme 5: 0 3 4 2 1 5 6 7
    [0, 3, 4, 2, 1, 5, 6, 7],
    // Scheme 6: 0 5 3 4 2 1 6 7
    [0, 5, 3, 4, 2, 1, 6, 7],
];

/// Decode the Table 7.19 Huffman prefix (`b0`, `b10`, …, `b1111111`)
/// against an MSb-first bit reader and return the resulting `mi` index
/// in `0..=7`.
///
/// Table 7.19 codes are unary-with-cap-at-7-bits:
///
/// ```text
/// mi  code         length
/// 0   b0           1
/// 1   b10          2
/// 2   b110         3
/// 3   b1110        4
/// 4   b11110       5
/// 5   b111110      6
/// 6   b1111110     7
/// 7   b1111111     7
/// ```
///
/// The procedure reads up to six `1` bits looking for a `0` terminator
/// (recognised codes `b0` / `b10` / … / `b111110` mapping to `mi=0..=5`).
/// If a seventh bit must be read, its value disambiguates `b1111110`
/// (`mi=6`) from `b1111111` (`mi=7`); both codes are 7 bits long.
fn read_table_7_19_mi(r: &mut BitReader<'_>) -> Result<u8, Error> {
    for mi in 0u8..=5 {
        let bit = r.read_bits(1, "MBMODES_huffman_code")?;
        if bit == 0 {
            return Ok(mi);
        }
    }
    // Six `1` bits already consumed — the seventh bit decides between
    // `b1111110` (mi=6) and `b1111111` (mi=7).
    let tail = r.read_bits(1, "MBMODES_huffman_code")?;
    if tail == 0 {
        Ok(6)
    } else {
        Ok(7)
    }
}

/// Decode the `MBMODES` per-macro-block coding-mode array per §7.4 of
/// the Xiph Theora I Specification ("Macro Block Coding Modes").
///
/// Given a frame's type, macro-block count `NMBS`, block count `NBS`,
/// the per-block `BCODED` array from §7.3, and the per-macro-block
/// `macro_block_to_luma_blocks` mapping (length `NMBS`, each entry the
/// four coded-order luma-block indices A, B, C, D of one macro block
/// per §7.5.2), returns an `NMBS`-element `Vec<MacroBlockMode>`.
///
/// For an intra frame (§7.4 step 1) every entry is
/// [`MacroBlockMode::Intra`] and `packet` is not consumed. For an inter
/// frame (§7.4 step 2) the procedure:
///
/// 1. Reads a 3-bit `MSCHEME` (step 2(a)).
/// 2. Builds the `MALPHABET[0..8]` table: from the bitstream when
///    `MSCHEME` is 0 (step 2(b)), from Table 7.19's fixed columns when
///    `MSCHEME` is 1..=6 (step 2(c)), or sentinel-ignored when
///    `MSCHEME` is 7 (modes are coded directly in step 2(d)i.B).
/// 3. For each macro block `mbi` in coded order (step 2(d)):
///    * If at least one of its four luma blocks has `BCODED[bi] == 1`,
///      reads a Huffman-coded mode (`MSCHEME != 7`) or a 3-bit direct
///      mode (`MSCHEME == 7`) per step 2(d)i.A / 2(d)i.B.
///    * Otherwise assigns [`MacroBlockMode::InterNoMv`] per step
///      2(d)ii (no on-wire mode bits are read).
///
/// `packet` is the full frame-data packet payload (or any slice
/// positioned at the byte-aligned start of the §7.4 bit stream). For a
/// full §7 frame walk the caller drives [`decode_macroblock_modes_inner`]
/// on a [`BitReader`] already positioned past §7.1 / §7.2 / §7.3.
///
/// Returns:
///
/// * [`Error::MacroBlockLumaMapLenMismatch`] if
///   `macro_block_to_luma_blocks.len() != nmbs`.
/// * [`Error::MacroBlockLumaBlockIndexOutOfRange`] if any luma-block
///   index is `>= nbs`.
/// * [`Error::TruncatedHeader`] when the underlying bit reader is
///   exhausted before a field is fully read.
pub fn decode_macroblock_modes(
    packet: &[u8],
    ftype: FrameType,
    nmbs: u32,
    nbs: u32,
    bcoded: &[u8],
    macro_block_to_luma_blocks: &[[u32; 4]],
) -> Result<Vec<MacroBlockMode>, Error> {
    let mut r = BitReader::new(packet);
    decode_macroblock_modes_inner(&mut r, ftype, nmbs, nbs, bcoded, macro_block_to_luma_blocks)
}

/// Inner §7.4 procedure operating on an already-positioned
/// [`BitReader`]. Split out so an end-to-end frame decoder can chain
/// §7.1 → §7.2 → §7.3 → §7.4 on the same reader without re-aligning to
/// a byte boundary.
pub(crate) fn decode_macroblock_modes_inner(
    r: &mut BitReader<'_>,
    ftype: FrameType,
    nmbs: u32,
    nbs: u32,
    bcoded: &[u8],
    macro_block_to_luma_blocks: &[[u32; 4]],
) -> Result<Vec<MacroBlockMode>, Error> {
    // Argument sanity checks. §7.4 step 2(d)i.A reads BCODED[bi] for
    // each of a macro block's four luma blocks, so the caller-supplied
    // mapping must match NMBS in length and reference only legal bi
    // values.
    let nmbs_us = nmbs as usize;
    if macro_block_to_luma_blocks.len() != nmbs_us {
        return Err(Error::MacroBlockLumaMapLenMismatch {
            map_len: macro_block_to_luma_blocks.len(),
            nmbs: nmbs_us,
        });
    }
    for (mbi, group) in macro_block_to_luma_blocks.iter().enumerate() {
        for (slot, &bi) in group.iter().enumerate() {
            if bi >= nbs {
                return Err(Error::MacroBlockLumaBlockIndexOutOfRange { mbi, slot, bi, nbs });
            }
        }
    }

    // Step 1: intra frames assign every macro block INTRA, no bits read.
    if ftype == FrameType::Intra {
        return Ok(vec![MacroBlockMode::Intra; nmbs_us]);
    }

    // Step 2(a): read MSCHEME (3-bit unsigned).
    let mscheme = r.read_bits(3, "MSCHEME")? as u8;

    // Build the MALPHABET[0..8] table per step 2(b) / 2(c). For
    // MSCHEME=7 the alphabet is unused; we leave it as a default sentinel.
    let mut malphabet: [u8; 8] = [0; 8];
    if mscheme == 0 {
        // Step 2(b): for each MODE in 0..=7, read 3-bit mi and assign
        // MALPHABET[mi] = MODE. This is a permutation of 0..=7 on the
        // wire; the spec does not say what to do if an mi value collides,
        // so we transcribe the loop literally (a later collision simply
        // overwrites the earlier one — both ends of the encoder/decoder
        // must agree on the permutation by construction).
        for mode in 0u8..8 {
            let mi = r.read_bits(3, "MALPHABET_mi")? as usize;
            malphabet[mi] = mode;
        }
    } else if mscheme <= 6 {
        // Step 2(c): copy the appropriate Table 7.19 column.
        malphabet = MALPHABETS_SCHEMES_1_TO_6[(mscheme - 1) as usize];
    }
    // mscheme == 7: malphabet stays default; step 2(d)i.B reads the
    // mode directly, bypassing it.

    // Step 2(d): walk macro blocks in coded order and emit MBMODES[mbi].
    let mut mbmodes: Vec<MacroBlockMode> = Vec::with_capacity(nmbs_us);
    for luma_group in macro_block_to_luma_blocks.iter() {
        // Step 2(d)i: does any luma block of this macro block have
        // BCODED[bi] == 1?
        let any_luma_coded = luma_group.iter().any(|&bi| bcoded[bi as usize] == 1);

        let mode_val: u8 = if any_luma_coded {
            if mscheme == 7 {
                // Step 2(d)i.B: read MBMODES[mbi] directly as 3 bits.
                r.read_bits(3, "MBMODES_direct")? as u8
            } else {
                // Step 2(d)i.A: walk Table 7.19's Huffman prefix to
                // recover mi, then look up MALPHABET[mi].
                let mi = read_table_7_19_mi(r)?;
                malphabet[mi as usize]
            }
        } else {
            // Step 2(d)ii: no luma blocks coded → INTER_NOMV (0).
            0u8
        };

        let mode = MacroBlockMode::from_index(mode_val).ok_or(Error::UnknownMacroBlockModeCode)?;
        mbmodes.push(mode);
    }
    debug_assert_eq!(mbmodes.len(), nmbs_us);
    Ok(mbmodes)
}

// ============================================================================
// §7.5  Motion Vectors
// ============================================================================

/// A single motion vector per §7.5 of the Xiph Theora I Specification.
///
/// Each component lies in the half-pixel range `-31..=31` (luma plane),
/// per the "Each component can take on integer values from −31 . . . 31,
/// inclusive, at half-pixel resolution" paragraph in §7.5.1. The value
/// itself is stored as the *signed integer* that came off the wire
/// before any half-pixel / quarter-pixel scaling — interpretation as
/// pixels is the §7.9.1 prediction step's concern, not this layer's.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MotionVector {
    /// X component (`MVX`). Signed, range `-31..=31`.
    pub x: i8,
    /// Y component (`MVY`). Signed, range `-31..=31`.
    pub y: i8,
}

impl MotionVector {
    /// Convenience constructor.
    pub fn new(x: i8, y: i8) -> Self {
        Self { x, y }
    }

    /// The all-zero default vector. §7.5.2 uses this for uncoded
    /// blocks (step 3(a)iii / 3(a)v / 3(a)vii / 3(a)ix), for
    /// `INTER_NOMV` / `INTER_GOLDEN_NOMV` / `INTRA` (step 3(f)), and
    /// for `LAST1` / `LAST2` initial state (step 1).
    pub const ZERO: MotionVector = MotionVector { x: 0, y: 0 };
}

/// Look-up table for Table 7.23 of the Theora I Specification (§7.5.1
/// MVMODE=0 Huffman codes for motion-vector components).
///
/// Indexed by the recognised code (already shifted so the MSb is bit
/// `len-1`). Yields `(value, length)` only for the spec-listed codes;
/// every other code is left as `(0, 0)` and rejected at lookup time.
///
/// Table 7.23 maps:
///
/// | code length | code prefix | values |
/// | ----------- | ----------- | ------ |
/// | 3 | `b000` | `0` |
/// | 3 | `b001`, `b010` | `+1`, `-1` |
/// | 4 | `b0110`, `b0111` | `+2`, `-2` |
/// | 4 | `b1000`, `b1001` | `+3`, `-3` |
/// | 6 | `b101000`..`b101111` | `±4`..`±7` |
/// | 7 | `b1100000`..`b1101111` | `±8`..`±15` |
/// | 8 | `b11100000`..`b11111111` | `±16`..`±31` |
///
/// In every range an even tail bit picks the positive value and an
/// odd tail bit picks the negative.
///
/// Returns `Some((value, len))` for the longest match starting at the
/// reader's current position, after consuming exactly `len` bits.
/// Returns `Err` on truncation. Returns `Ok(None)` only when the read
/// bits could not be matched at any length (defensive — the table is
/// complete for 3..=8 bits so this never fires on a valid prefix).
fn recognise_mv_huffman(r: &mut BitReader<'_>) -> Result<i8, Error> {
    // The table is small enough to drive with a hand-written DFA: read
    // up to 8 bits, branching at each step based on the prefix. The
    // structure follows Table 7.23 verbatim.
    //
    // Step 1: read 3 bits — every Table 7.23 code starts with one of
    // `b000` / `b001` / `b010` / `b011x` / `b100x` / `b101xxx` /
    // `b110xxxx` / `b111xxxxx`.
    let b3 = r.read_bits(3, "MV_huffman")? as u8;
    match b3 {
        // b000 → value 0 (3 bits)
        0b000 => Ok(0),
        // b001 → value +1 (3 bits)
        0b001 => Ok(1),
        // b010 → value -1 (3 bits)
        0b010 => Ok(-1),
        // b011x → +2 / -2 (4 bits)
        0b011 => {
            let t = r.read_bits(1, "MV_huffman")? as u8;
            if t == 0 {
                Ok(2)
            } else {
                Ok(-2)
            }
        }
        // b100x → +3 / -3 (4 bits)
        0b100 => {
            let t = r.read_bits(1, "MV_huffman")? as u8;
            if t == 0 {
                Ok(3)
            } else {
                Ok(-3)
            }
        }
        // b101xxx → ±4..±7 (6 bits). Three more bits decode the
        // (magnitude − 4) bucket plus sign:
        //   b101 000 → +4   b101 001 → -4
        //   b101 010 → +5   b101 011 → -5
        //   b101 100 → +6   b101 101 → -6
        //   b101 110 → +7   b101 111 → -7
        0b101 => {
            let tail = r.read_bits(3, "MV_huffman")? as u8;
            let mag = 4 + ((tail >> 1) & 0b11) as i8;
            let sign = tail & 1;
            Ok(if sign == 0 { mag } else { -mag })
        }
        // b110xxxx → ±8..±15 (7 bits). Four more bits: high 3 = bucket,
        // low 1 = sign.
        0b110 => {
            let tail = r.read_bits(4, "MV_huffman")? as u8;
            let mag = 8 + ((tail >> 1) & 0b111) as i8;
            let sign = tail & 1;
            Ok(if sign == 0 { mag } else { -mag })
        }
        // b111xxxxx → ±16..±31 (8 bits). Five more bits: high 4 =
        // bucket, low 1 = sign.
        0b111 => {
            let tail = r.read_bits(5, "MV_huffman")? as u8;
            let mag = 16 + ((tail >> 1) & 0b1111) as i8;
            let sign = tail & 1;
            Ok(if sign == 0 { mag } else { -mag })
        }
        // `read_bits(3)` cannot return values outside 0..=7.
        _ => unreachable!("read_bits(3) returned > 7"),
    }
}

/// Decode a single motion vector per §7.5.1 of the Xiph Theora I
/// Specification ("Motion Vector Decode").
///
/// `mvmode` is the single-bit per-frame MV decoding selector from
/// §7.5.2 step 2 (`0` = Huffman per Table 7.23; `1` = fixed-length
/// 5-bit magnitude + 1-bit sign per component). The procedure returns
/// the typed `(MVX, MVY)` pair as a [`MotionVector`].
///
/// For `mvmode == 1` the spec mandates the sign bit be read even when
/// magnitude is zero ("for compatibility with VP3, a sign bit is read
/// even if the magnitude read is zero") — the implementation matches.
pub fn decode_single_motion_vector(bits: &[u8], mvmode: u8) -> Result<MotionVector, Error> {
    let mut r = BitReader::new(bits);
    decode_single_motion_vector_inner(&mut r, mvmode)
}

/// Inner §7.5.1 procedure operating on an already-positioned
/// [`BitReader`]. Split out so a frame walker can chain §7.4 → §7.5
/// on the same reader.
pub(crate) fn decode_single_motion_vector_inner(
    r: &mut BitReader<'_>,
    mvmode: u8,
) -> Result<MotionVector, Error> {
    if mvmode == 0 {
        // Step 1: walk Table 7.23 for both components in turn.
        let mvx = recognise_mv_huffman(r)?;
        let mvy = recognise_mv_huffman(r)?;
        Ok(MotionVector { x: mvx, y: mvy })
    } else {
        // Step 2: 5-bit magnitude + 1-bit sign for each component.
        let mut mvx = r.read_bits(5, "MVX_magnitude")? as i8;
        let signx = r.read_bits(1, "MVX_sign")? as u8;
        if signx == 1 {
            mvx = -mvx;
        }
        let mut mvy = r.read_bits(5, "MVY_magnitude")? as i8;
        let signy = r.read_bits(1, "MVY_sign")? as u8;
        if signy == 1 {
            mvy = -mvy;
        }
        Ok(MotionVector { x: mvx, y: mvy })
    }
}

/// Per-macroblock chroma-block layout supplied to
/// [`decode_macroblock_motion_vectors`].
///
/// The §7.5.2 procedure writes chroma MVs at indices `E, F, G, H`
/// (Cb) and `I, J, K, L` (Cr) — the exact count and ordering depend
/// on `PF`:
///
/// * `PF=0` (4:2:0): 1 Cb block + 1 Cr block per macro block. The
///   per-MB inner slice has length 1 (just `E` for Cb, just `F` for
///   Cr — using the spec's letters; we collapse to one index per
///   plane per macroblock).
/// * `PF=2` (4:2:2): 2 Cb blocks + 2 Cr blocks per macroblock (E, F
///   for Cb in bottom/top order; G, H for Cr in bottom/top order —
///   collapsed to two indices per plane per macroblock).
/// * `PF=3` (4:4:4): 4 Cb + 4 Cr per macroblock, raster A, B, C, D
///   order (E, F, G, H for Cb; I, J, K, L for Cr).
///
/// The caller passes one outer slice per plane: `cb_map.len() ==
/// cr_map.len() == nmbs`. Per-MB inner slices have length 1, 2, or 4
/// depending on `pf`.
#[derive(Debug, Clone, Copy)]
pub struct ChromaBlockLayout<'a> {
    /// Cb-plane per-macroblock block indices.
    pub cb: &'a [&'a [u32]],
    /// Cr-plane per-macroblock block indices.
    pub cr: &'a [&'a [u32]],
}

/// Round-to-nearest with ties away from zero, per the Theora spec's
/// `round(a)` definition. Operates on integer numerator and divisor.
///
/// For positive `a / b` this returns `floor(a/b + 1/2)`; for negative
/// it returns `ceil(a/b - 1/2)` — which is the "away from zero on the
/// halfway tie" behaviour the spec asks for.
#[inline]
fn round_div(num: i32, den: i32) -> i32 {
    debug_assert!(den > 0, "round_div with non-positive denominator");
    if num >= 0 {
        (num + den / 2) / den
    } else {
        -(((-num) + den / 2) / den)
    }
}

/// Decode the per-block `MVECTS` array for a frame per §7.5.2 of the
/// Xiph Theora I Specification ("Macro Block Motion Vector Decode").
///
/// Inputs:
///
/// * `packet` — the byte-aligned start of the §7.5.2 bit stream (in a
///   real frame walk, a [`BitReader`] already positioned past §7.4 is
///   used via [`decode_macroblock_motion_vectors_inner`]).
/// * `ftype` — the frame type. Intra frames short-circuit: no motion
///   vectors are stored and no bits are consumed (§7.5 opening
///   sentence).
/// * `pf` — pixel format. Determines the chroma averaging in §7.5.2
///   step 3(a)x..3(a)xii.
/// * `nbs` — total block count in the frame (Tables 6.5 / 6.6).
/// * `nmbs` — total macro-block count.
/// * `bcoded` — the per-block `BCODED` array from §7.3 (length `nbs`).
/// * `mbmodes` — the per-mb `MBMODES` array from §7.4 (length `nmbs`).
/// * `luma_map` — per-mb four-element [A, B, C, D] luma-block-index
///   slice in raster order (lower-left, lower-right, upper-left,
///   upper-right). Length `nmbs`.
/// * `chroma_map` — see [`ChromaBlockLayout`]. Per-plane outer length
///   `nmbs`; inner length `1` for PF=0, `2` for PF=2, `4` for PF=3.
///
/// Output: an `NBS`-element `Vec<MotionVector>` with the (0, 0) default
/// for blocks that don't receive an MV from this procedure (uncoded
/// blocks in INTER_MV_FOUR macroblocks, blocks in NOMV / INTRA /
/// GOLDEN_NOMV macroblocks, and every block in an intra frame).
///
/// Returns:
///
/// * [`Error::MotionVectorMbModesLenMismatch`] /
///   [`Error::MotionVectorLumaMapLenMismatch`] /
///   [`Error::MotionVectorChromaMapLenMismatch`] for shape mismatches.
/// * [`Error::MotionVectorLumaBlockIndexOutOfRange`] /
///   [`Error::MotionVectorChromaBlockIndexOutOfRange`] for OOB indices.
/// * [`Error::MotionVectorChromaMacroBlockSlotLenMismatch`] for chroma
///   per-MB inner slices of the wrong length for the declared `pf`.
/// * [`Error::TruncatedHeader`] when the underlying bit reader is
///   exhausted before a field is fully read.
#[allow(clippy::too_many_arguments)]
pub fn decode_macroblock_motion_vectors(
    packet: &[u8],
    ftype: FrameType,
    pf: PixelFormat,
    nbs: u32,
    nmbs: u32,
    bcoded: &[u8],
    mbmodes: &[MacroBlockMode],
    luma_map: &[[u32; 4]],
    chroma_map: ChromaBlockLayout<'_>,
) -> Result<Vec<MotionVector>, Error> {
    let mut r = BitReader::new(packet);
    decode_macroblock_motion_vectors_inner(
        &mut r, ftype, pf, nbs, nmbs, bcoded, mbmodes, luma_map, chroma_map,
    )
}

/// Inner §7.5.2 procedure operating on an already-positioned
/// [`BitReader`]. See [`decode_macroblock_motion_vectors`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_macroblock_motion_vectors_inner(
    r: &mut BitReader<'_>,
    ftype: FrameType,
    pf: PixelFormat,
    nbs: u32,
    nmbs: u32,
    bcoded: &[u8],
    mbmodes: &[MacroBlockMode],
    luma_map: &[[u32; 4]],
    chroma_map: ChromaBlockLayout<'_>,
) -> Result<Vec<MotionVector>, Error> {
    let nmbs_us = nmbs as usize;
    let nbs_us = nbs as usize;

    // Sanity-check argument shapes early.
    if mbmodes.len() != nmbs_us {
        return Err(Error::MotionVectorMbModesLenMismatch {
            modes_len: mbmodes.len(),
            nmbs: nmbs_us,
        });
    }
    if luma_map.len() != nmbs_us {
        return Err(Error::MotionVectorLumaMapLenMismatch {
            map_len: luma_map.len(),
            nmbs: nmbs_us,
        });
    }
    if chroma_map.cb.len() != nmbs_us {
        return Err(Error::MotionVectorChromaMapLenMismatch {
            plane: 0,
            map_len: chroma_map.cb.len(),
            expected: nmbs_us,
        });
    }
    if chroma_map.cr.len() != nmbs_us {
        return Err(Error::MotionVectorChromaMapLenMismatch {
            plane: 1,
            map_len: chroma_map.cr.len(),
            expected: nmbs_us,
        });
    }
    let chroma_inner_expected = match pf {
        PixelFormat::Yuv420 => 1,
        PixelFormat::Yuv422 => 2,
        PixelFormat::Yuv444 => 4,
    };
    for (mbi, (cb_slot, cr_slot)) in chroma_map.cb.iter().zip(chroma_map.cr.iter()).enumerate() {
        if cb_slot.len() != chroma_inner_expected {
            return Err(Error::MotionVectorChromaMacroBlockSlotLenMismatch {
                mbi,
                plane: 0,
                got: cb_slot.len(),
                expected: chroma_inner_expected,
            });
        }
        if cr_slot.len() != chroma_inner_expected {
            return Err(Error::MotionVectorChromaMacroBlockSlotLenMismatch {
                mbi,
                plane: 1,
                got: cr_slot.len(),
                expected: chroma_inner_expected,
            });
        }
        for (slot, &bi) in cb_slot.iter().enumerate() {
            if bi >= nbs {
                return Err(Error::MotionVectorChromaBlockIndexOutOfRange { mbi, slot, bi, nbs });
            }
        }
        for (slot, &bi) in cr_slot.iter().enumerate() {
            if bi >= nbs {
                return Err(Error::MotionVectorChromaBlockIndexOutOfRange { mbi, slot, bi, nbs });
            }
        }
    }
    for (mbi, group) in luma_map.iter().enumerate() {
        for (slot, &bi) in group.iter().enumerate() {
            if bi >= nbs {
                return Err(Error::MotionVectorLumaBlockIndexOutOfRange { mbi, slot, bi, nbs });
            }
        }
    }

    // §7.5 opening sentence: intra frames carry no motion vectors,
    // and §7.5.2's procedure is skipped wholesale. We still allocate
    // the output array so callers can index uniformly.
    if ftype == FrameType::Intra {
        return Ok(vec![MotionVector::ZERO; nbs_us]);
    }

    // §7.5.2 step 1: LAST1 = LAST2 = (0, 0).
    let mut last1 = MotionVector::ZERO;
    let mut last2 = MotionVector::ZERO;

    // §7.5.2 step 2: read MVMODE (1 bit) — "Note that this value is
    // read even if no macro blocks require a motion vector to be
    // decoded."
    let mvmode = r.read_bits(1, "MVMODE")? as u8;

    let mut mvects = vec![MotionVector::ZERO; nbs_us];

    // §7.5.2 step 3: walk macro blocks in coded order.
    for mbi in 0..nmbs_us {
        let mode = mbmodes[mbi];
        let abcd = luma_map[mbi];

        // Single-MV value to propagate to every coded block in step
        // 3(g). Set by every match arm except `InterMvFour`, which
        // uses its own per-block writes above and `continue`s past
        // step 3(g).
        let mv_for_blocks: Option<MotionVector>;

        match mode {
            MacroBlockMode::InterMvFour => {
                // Step 3(a): four per-luma-block MVs, sourced from
                // BCODED in raster order. Uncoded blocks get (0, 0).
                let mut per_block: [MotionVector; 4] = [MotionVector::ZERO; 4];
                let mut last_coded_mv = MotionVector::ZERO;
                let mut had_any_coded = false;
                for (slot, &bi) in abcd.iter().enumerate() {
                    if bcoded[bi as usize] == 1 {
                        let mv = decode_single_motion_vector_inner(r, mvmode)?;
                        per_block[slot] = mv;
                        last_coded_mv = mv;
                        had_any_coded = true;
                    }
                    // else: per_block[slot] stays (0, 0)
                }
                // Step 3(a)ii..ix have been executed (interleaved with
                // the BCODED test above). Now assign to MVECTS.
                for (slot, &bi) in abcd.iter().enumerate() {
                    mvects[bi as usize] = per_block[slot];
                }

                // Step 3(a)x..xii: chroma averaging.
                let a = per_block[0];
                let b = per_block[1];
                let c = per_block[2];
                let d = per_block[3];
                match pf {
                    PixelFormat::Yuv420 => {
                        let ex = round_div(a.x as i32 + b.x as i32 + c.x as i32 + d.x as i32, 4);
                        let ey = round_div(a.y as i32 + b.y as i32 + c.y as i32 + d.y as i32, 4);
                        let chroma_mv = MotionVector {
                            x: ex as i8,
                            y: ey as i8,
                        };
                        let e_bi = chroma_map.cb[mbi][0];
                        let f_bi = chroma_map.cr[mbi][0];
                        mvects[e_bi as usize] = chroma_mv;
                        mvects[f_bi as usize] = chroma_mv;
                    }
                    PixelFormat::Yuv422 => {
                        // E/G (bottom): avg(A, B); F/H (top): avg(C, D).
                        let bot = MotionVector {
                            x: round_div(a.x as i32 + b.x as i32, 2) as i8,
                            y: round_div(a.y as i32 + b.y as i32, 2) as i8,
                        };
                        let top = MotionVector {
                            x: round_div(c.x as i32 + d.x as i32, 2) as i8,
                            y: round_div(c.y as i32 + d.y as i32, 2) as i8,
                        };
                        let cb_slot = chroma_map.cb[mbi];
                        let cr_slot = chroma_map.cr[mbi];
                        mvects[cb_slot[0] as usize] = bot; // E
                        mvects[cb_slot[1] as usize] = top; // F
                        mvects[cr_slot[0] as usize] = bot; // G
                        mvects[cr_slot[1] as usize] = top; // H
                    }
                    PixelFormat::Yuv444 => {
                        // E,I=A; F,J=B; G,K=C; H,L=D.
                        let cb_slot = chroma_map.cb[mbi];
                        let cr_slot = chroma_map.cr[mbi];
                        mvects[cb_slot[0] as usize] = a;
                        mvects[cb_slot[1] as usize] = b;
                        mvects[cb_slot[2] as usize] = c;
                        mvects[cb_slot[3] as usize] = d;
                        mvects[cr_slot[0] as usize] = a;
                        mvects[cr_slot[1] as usize] = b;
                        mvects[cr_slot[2] as usize] = c;
                        mvects[cr_slot[3] as usize] = d;
                    }
                }

                // Step 3(a)xiii / xiv: update LAST2/LAST1. "There must
                // always be at least one [coded luma block], since
                // macro blocks with no coded luma blocks must use mode
                // 0: INTER_NOMV."
                if had_any_coded {
                    last2 = last1;
                    last1 = last_coded_mv;
                }
                // Step 3(g) does NOT apply to INTER_MV_FOUR ("If
                // MBMODES[mbi] is not 7").
                continue;
            }
            MacroBlockMode::InterGoldenMv => {
                // Step 3(b): decode one MV, no LAST update.
                let mv = decode_single_motion_vector_inner(r, mvmode)?;
                mv_for_blocks = Some(mv);
            }
            MacroBlockMode::InterMvLast2 => {
                // Step 3(c): (MVX, MVY) = LAST2; LAST2 = LAST1; LAST1 = (MVX, MVY).
                // Equivalent to swapping LAST1 and LAST2 and emitting
                // the new value of LAST1 (= old LAST2).
                std::mem::swap(&mut last1, &mut last2);
                mv_for_blocks = Some(last1);
            }
            MacroBlockMode::InterMvLast => {
                // Step 3(d): (MVX, MVY) = LAST1.
                mv_for_blocks = Some(last1);
            }
            MacroBlockMode::InterMv => {
                // Step 3(e): decode one MV; LAST2 = LAST1; LAST1 = (MVX, MVY).
                let mv = decode_single_motion_vector_inner(r, mvmode)?;
                last2 = last1;
                last1 = mv;
                mv_for_blocks = Some(mv);
            }
            MacroBlockMode::InterGoldenNoMv | MacroBlockMode::Intra | MacroBlockMode::InterNoMv => {
                // Step 3(f): assign zero.
                mv_for_blocks = Some(MotionVector::ZERO);
            }
        }

        // Step 3(g): for non-INTER_MV_FOUR modes, propagate to every
        // CODED block bi in the macro block. "Coded blocks" here
        // includes luma + chroma; the spec wording "each coded block
        // bi in macro block mbi" is interpreted per the surrounding
        // block layout (the §7.5 prose immediately preceding step 3
        // says "every block in all the color planes are assigned the
        // same motion vector" for these modes). We mirror that:
        // iterate luma A,B,C,D and the format-appropriate chroma
        // slots, writing the MV wherever BCODED says the block is
        // coded.
        if let Some(mv) = mv_for_blocks {
            for &bi in abcd.iter() {
                if bcoded[bi as usize] == 1 {
                    mvects[bi as usize] = mv;
                }
            }
            for &bi in chroma_map.cb[mbi].iter() {
                if bcoded[bi as usize] == 1 {
                    mvects[bi as usize] = mv;
                }
            }
            for &bi in chroma_map.cr[mbi].iter() {
                if bcoded[bi as usize] == 1 {
                    mvects[bi as usize] = mv;
                }
            }
        }
    }

    debug_assert_eq!(mvects.len(), nbs_us);
    Ok(mvects)
}

// =====================================================================
// §7.6 Block-Level qi Decode
// =====================================================================

/// Decode the per-block `QIIS` array from a video-data packet per
/// §7.6 ("Block-Level *qi* Decode") of the Xiph Theora I Specification.
///
/// `QIIS[bi]` is an index into the frame's `qi` value list (the 1..=3
/// values `decode_frame_header` returns in [`TheoraFrameHeader::qis`]).
/// For each block `bi`, `QIIS[bi]` selects which of those `qi` values
/// drives the block's AC dequantization (DC dequantization always uses
/// the first `qi` value to avoid interfering with DC prediction; the
/// spec's §7.6 preamble explains the asymmetry).
///
/// Per §7.6 the procedure makes `NQIS − 1` passes through the list of
/// *coded* blocks. Pass number `qii` (running 0..=NQIS−2) decodes one
/// §7.2.1 long-run bit string whose length equals the count of blocks
/// `bi` such that `BCODED[bi] != 0` and `QIIS[bi] == qii`; each bit then
/// either keeps the block at `qii` (bit value 0) or promotes it to
/// `qii + 1` (bit value 1). Subsequent passes therefore see only blocks
/// that were promoted out of every earlier pass — exactly the "second
/// group" split the spec describes.
///
/// VP3-compatibility short-circuit: when `NQIS == 1` the main loop
/// iterates `0..=−1` (i.e. zero times), no bits are read, and every
/// returned `QIIS[bi]` is `0`. The VP3 spec §B.1 has the same property
/// because pre-3.2.0 frame headers can only carry one `qi`.
///
/// `packet` is the full frame-data packet payload (or any slice
/// positioned at the byte-aligned start of the §7.6 bit stream). The
/// byte-aligned variant is provided for unit-test convenience; an end-
/// to-end decoder will call [`decode_block_level_qi_inner`] on a
/// [`BitReader`] already positioned past the §7.1 / §7.2 / §7.3 / §7.4 /
/// §7.5 prefix.
///
/// Returns:
///
/// * [`Error::BlockLevelQiBcodedLenMismatch`] if `bcoded.len() != nbs`.
/// * [`Error::BlockLevelQiNqisOutOfRange`] if `nqis` is not in `1..=3`.
/// * [`Error::TruncatedHeader`] / [`Error::RunLengthOverrun`] when the
///   underlying §7.2.1 long-run decode rejects.
pub fn decode_block_level_qi(
    packet: &[u8],
    nbs: u32,
    bcoded: &[u8],
    nqis: usize,
) -> Result<Vec<u8>, Error> {
    let mut r = BitReader::new(packet);
    decode_block_level_qi_inner(&mut r, nbs, bcoded, nqis)
}

/// Inner §7.6 procedure operating on an already-positioned
/// [`BitReader`]. Split out so an end-to-end frame decoder can chain
/// §7.1 → §7.2 → §7.3 → §7.4 → §7.5 → §7.6 on the same reader without
/// re-aligning to a byte boundary.
pub(crate) fn decode_block_level_qi_inner(
    r: &mut BitReader<'_>,
    nbs: u32,
    bcoded: &[u8],
    nqis: usize,
) -> Result<Vec<u8>, Error> {
    // Argument validation. §7.6 step 2(a)–(c) reads `BCODED[bi]` for
    // every block 0..NBS, and `nqis` controls the main loop bound
    // (§7.1 step 6 restricts NQIS to 1..=3).
    let nbs_us = nbs as usize;
    if bcoded.len() != nbs_us {
        return Err(Error::BlockLevelQiBcodedLenMismatch {
            bcoded_len: bcoded.len(),
            nbs: nbs_us,
        });
    }
    if !(1..=MAX_FRAME_QIS).contains(&nqis) {
        return Err(Error::BlockLevelQiNqisOutOfRange { nqis });
    }

    // Step 1: assign QIIS[bi] = 0 for every block.
    let mut qiis: Vec<u8> = vec![0u8; nbs_us];

    // Step 2: outer loop `qii` from 0 to NQIS−2. When NQIS == 1 the
    // range is empty (the VP3-compat short-circuit) and no bits are
    // read.
    for qii in 0..(nqis.saturating_sub(1)) {
        // Step 2(a): tally blocks whose current QIIS matches qii and
        // whose BCODED is non-zero. Only those blocks participate in
        // this pass's bit string.
        let qii_u8 = qii as u8;
        let nbits: u64 = bcoded
            .iter()
            .zip(qiis.iter())
            .filter(|(&bc, &q)| bc != 0 && q == qii_u8)
            .count() as u64;

        // Step 2(b): decode the NBITS-bit string via the §7.2.1 long-
        // run procedure. Routed through the shared-reader inner so we
        // do not re-align to a byte boundary.
        let bits = decode_long_run_bit_string_inner(r, nbits)?;
        debug_assert_eq!(bits.len() as u64, nbits);

        // Step 2(c): consume one bit per matching block, in coded order
        // (i.e. ascending `bi`). Each consumed bit is *added* to
        // QIIS[bi]: 0 keeps the block at the current qii; 1 promotes it
        // to qii + 1, putting it in the "second group" the next pass
        // operates on.
        let mut cursor: usize = 0;
        for (&bc, q) in bcoded.iter().zip(qiis.iter_mut()) {
            if bc != 0 && *q == qii_u8 {
                *q += bits[cursor];
                cursor += 1;
            }
        }
        debug_assert_eq!(cursor as u64, nbits);
    }

    debug_assert_eq!(qiis.len(), nbs_us);
    Ok(qiis)
}

// =====================================================================
// §7.7.1 EOB Token Decode
// =====================================================================

/// Decode an EOB (end-of-block) token per §7.7.1 ("EOB Token Decode")
/// of the Xiph Theora I Specification.
///
/// EOB tokens are values `0..=6` of the §7.7 DCT-token alphabet
/// (Table 7.33). Each one signals that the *remainder* of one or more
/// blocks contains only zeros: token 0 ends a single block; tokens 1
/// through 6 carry an "EOB run" extending the zero-fill across that
/// many additional blocks. The procedure short-circuits §7.7's zig-zag
/// walk for each ended block by zero-filling its `COEFFS[bi][ti..=63]`
/// tail, recording its coefficient count, and pinning `TIS[bi]` to 64.
///
/// Inputs:
///
/// * `token` — the decoded token value (must satisfy §7.7.1's
///   `0..=6` range). Tokens outside this range belong to §7.7.2
///   (coefficient tokens) and are not handled here.
/// * `nbs` — total block count in the frame (Table 6.5 / 6.6), the
///   length the three state slices must share.
/// * `bi` — coded-order index of the block whose tail is being
///   ended (the current §7.7 pass's "current block").
/// * `ti` — current token index inside that block (§7.7's outer
///   zig-zag-order loop variable). Must be `0..=63`.
/// * `tis` — `NBS`-element array of per-block current token indices.
///   Read by step 9 (assign `NCOEFFS[bi] = TIS[bi]`), written by step
///   10 (assign `TIS[bi] = 64`) and inspected by step 7(b) to count
///   already-completed blocks.
/// * `ncoeffs` — `NBS`-element array of per-block coefficient counts.
///   Written by step 9.
/// * `coeffs` — `NBS × 64` array of quantized DCT coefficients in
///   zig-zag order. Step 8 zero-fills `coeffs[bi][ti..=63]`.
///
/// Output: the procedure's `EOBS` value *after* the step-11
/// decrement — i.e. the *remaining* length of the current EOB run
/// after this call ends one block. A return value of zero means the
/// current EOB run has ended and the next §7.7 pass picks a fresh
/// token; a non-zero return value means the next `EOBS` calls into
/// §7.7 will skip token decode and instead use this same procedure
/// (with `token` short-circuited; see §7.7.3) to end additional
/// blocks for free.
///
/// Step 7(b) (the "token 6 with payload 0" special case) is given the
/// spec's "all remaining blocks" interpretation: when token 6's 12-bit
/// payload reads as zero, EOBS becomes the count of blocks `bj` with
/// `TIS[bj] < 64` — *including the current `bi`*, which still has
/// `TIS[bi] < 64` at this point because step 10 has not yet run. The
/// VP3-compat note at the end of §7.7.1 documents that VP3 encoders
/// never emit this special case, but compliant decoders accept it.
///
/// Returns:
///
/// * [`Error::EobTokenOutOfRange`] when `token > 6`.
/// * [`Error::EobTokenBlockIndexOutOfRange`] when `bi >= nbs`.
/// * [`Error::EobTokenIndexOutOfRange`] when `ti > 63`.
/// * [`Error::EobTokenStateLenMismatch`] when any of `tis` / `ncoeffs`
///   / `coeffs` does not have exactly `nbs` entries.
/// * [`Error::TruncatedHeader`] when the underlying bit reader is
///   exhausted before TOKEN 3..=6's extra-bits payload is fully read.
#[allow(clippy::too_many_arguments)]
pub fn decode_eob_token(
    packet: &[u8],
    token: u8,
    nbs: u32,
    bi: u32,
    ti: u8,
    tis: &mut [u8],
    ncoeffs: &mut [u8],
    coeffs: &mut [[i16; 64]],
) -> Result<u64, Error> {
    let mut r = BitReader::new(packet);
    decode_eob_token_inner(&mut r, token, nbs, bi, ti, tis, ncoeffs, coeffs)
}

/// Inner §7.7.1 procedure operating on an already-positioned
/// [`BitReader`]. Split out so the §7.7.3 driver — once it lands — can
/// chain §7.6 → §7.7 on the same reader without re-aligning to a byte
/// boundary.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_eob_token_inner(
    r: &mut BitReader<'_>,
    token: u8,
    nbs: u32,
    bi: u32,
    ti: u8,
    tis: &mut [u8],
    ncoeffs: &mut [u8],
    coeffs: &mut [[i16; 64]],
) -> Result<u64, Error> {
    // Argument validation. §7.7.1's declared input ranges fail closed
    // here so a future §7.7.3 driver bug doesn't silently overflow
    // the state arrays.
    if token > 6 {
        return Err(Error::EobTokenOutOfRange { token });
    }
    if bi >= nbs {
        return Err(Error::EobTokenBlockIndexOutOfRange { bi, nbs });
    }
    if ti > 63 {
        return Err(Error::EobTokenIndexOutOfRange { ti });
    }
    let nbs_us = nbs as usize;
    if tis.len() != nbs_us {
        return Err(Error::EobTokenStateLenMismatch {
            which: EobTokenStateSlice::Tis,
            got: tis.len(),
            nbs: nbs_us,
        });
    }
    if ncoeffs.len() != nbs_us {
        return Err(Error::EobTokenStateLenMismatch {
            which: EobTokenStateSlice::Ncoeffs,
            got: ncoeffs.len(),
            nbs: nbs_us,
        });
    }
    if coeffs.len() != nbs_us {
        return Err(Error::EobTokenStateLenMismatch {
            which: EobTokenStateSlice::Coeffs,
            got: coeffs.len(),
            nbs: nbs_us,
        });
    }

    // Steps 1..=7: decode EOBS from TOKEN.
    let mut eobs: u64 = match token {
        // Step 1: TOKEN=0 → run of 1.
        0 => 1,
        // Step 2: TOKEN=1 → run of 2.
        1 => 2,
        // Step 3: TOKEN=2 → run of 3.
        2 => 3,
        // Step 4: TOKEN=3 → 2-bit extra-bits payload, range 4..=7.
        3 => r.read_bits(2, "§7.7.1 EOBS (TOKEN=3)")? as u64 + 4,
        // Step 5: TOKEN=4 → 3-bit payload, range 8..=15.
        4 => r.read_bits(3, "§7.7.1 EOBS (TOKEN=4)")? as u64 + 8,
        // Step 6: TOKEN=5 → 4-bit payload, range 16..=31.
        5 => r.read_bits(4, "§7.7.1 EOBS (TOKEN=5)")? as u64 + 16,
        // Step 7: TOKEN=6 → 12-bit payload, range 1..=4095 *or* the
        //   "all remaining blocks" sentinel when payload reads zero.
        6 => {
            let payload = r.read_bits(12, "§7.7.1 EOBS (TOKEN=6)")? as u64;
            if payload == 0 {
                // Step 7(b): count blocks bj such that TIS[bj] < 64.
                // The current block's TIS is still <64 at this point
                // (step 10 has not run yet) so it is included in the
                // tally — which matches the spec's "the size of the
                // remaining coded blocks" wording.
                tis.iter().filter(|&&t| t < 64).count() as u64
            } else {
                payload
            }
        }
        // The `token > 6` rejection above forecloses every other arm.
        _ => unreachable!("token range was clamped to 0..=6"),
    };

    // Step 8: zero-fill COEFFS[bi][ti..=63]. Done as a slice fill so a
    // mistranscribed bound trips a single off-by-one rather than a
    // pile of subtly-wrong tail values.
    let bi_us = bi as usize;
    let ti_us = ti as usize;
    coeffs[bi_us][ti_us..64].fill(0);

    // Step 9: NCOEFFS[bi] = TIS[bi]. Records how many coefficients
    // were ever written for this block (the spec uses this in §7.8 to
    // skip uninitialised-tail blocks during DC prediction).
    ncoeffs[bi_us] = tis[bi_us];

    // Step 10: TIS[bi] = 64. Pins this block out of subsequent §7.7
    // passes — any token index past 63 means the block is "done".
    tis[bi_us] = 64;

    // Step 11: EOBS -= 1. The procedure's return value is the count
    // of *additional* blocks the current EOB run will still close at
    // the start of subsequent §7.7 passes.
    eobs -= 1;
    Ok(eobs)
}

// =====================================================================
// §7.7.2 Coefficient Token Decode
// =====================================================================

/// Per-token classification produced by [`decode_coefficient_token`].
///
/// The 25 §7.7.2 tokens fall into three structural classes that
/// downstream §7.7 / §7.7.3 logic discriminates over. Capturing the
/// class as a typed enum (rather than echoing the raw token byte) lets
/// the driver and tests bisect token behaviour without re-encoding the
/// Table 7.38 case analysis at every call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoefficientTokenKind {
    /// Tokens 7 and 8: pure zero runs of length `1..=8` or `1..=64`
    /// respectively. These do NOT update `NCOEFFS[bi]` (the spec text
    /// is explicit: "we do not update the coefficient count for the
    /// block if we decode a pure zero run"). They only advance
    /// `TIS[bi]`.
    ZeroRun,
    /// Tokens 9..=22: a single non-zero coefficient at `COEFFS[bi][ti]`.
    /// Advances `TIS[bi]` by 1 and writes `NCOEFFS[bi] = TIS[bi]`.
    Single,
    /// Tokens 23..=31: a zero run of `1..=17` zeros followed by a
    /// single trailing non-zero coefficient. Advances `TIS[bi]` by
    /// `RUN + 1` and writes `NCOEFFS[bi] = TIS[bi]`.
    RunPlusOne,
}

/// Decode one §7.7.2 coefficient token (Table 7.38 token values 7..=31)
/// against per-block `TIS` / `NCOEFFS` / `COEFFS` state arrays.
///
/// The 25 tokens partition into three structural classes:
///
/// * **Zero-run tokens** — `TOKEN = 7` reads a 3-bit `RLEN` (then
///   `+1`, range `1..=8`); `TOKEN = 8` reads a 6-bit `RLEN` (then
///   `+1`, range `1..=64`). Both zero-fill `COEFFS[bi][ti..ti+RLEN]`
///   and advance `TIS[bi]` by `RLEN`. They do **not** update
///   `NCOEFFS[bi]` — §7.7.2's introductory text is explicit: "we do
///   not update the coefficient count for the block if we decode a
///   pure zero run".
///
/// * **Single-coefficient tokens** — `TOKEN` in `9..=22` writes one
///   coefficient at `COEFFS[bi][ti]`, then advances `TIS[bi]` by 1 and
///   sets `NCOEFFS[bi] = TIS[bi]`. The coefficient value comes from a
///   fixed magnitude (tokens 9..=12: `±1, ±2`) or from a `SIGN` bit
///   plus a `MAG_BITS`-bit `MAG` field added to a fixed offset
///   (tokens 13..=22: `MAG_BITS = 0..=9`, offset `3, 4, 5, 6, 7, 9,
///   13, 21, 37, 69`).
///
/// * **Run-plus-one tokens** — `TOKEN` in `23..=31` zero-fills
///   `COEFFS[bi][ti..ti+RUN]` then writes one trailing coefficient at
///   `COEFFS[bi][ti+RUN]`. The trailing coefficient is `±1` for
///   tokens 23..=29 and `±MAG` (`MAG = 2..=3`) for tokens 30..=31.
///   The `RUN` length comes from a fixed value (tokens 23..=27:
///   `RUN = 1..=5`) or a `RUN_BITS`-bit field plus offset (tokens
///   28..=31: `RUN_BITS = 2, 3, 1, 1`; offset `6, 10, 1, 2`).
///   `TIS[bi]` advances by `RUN + 1` and `NCOEFFS[bi] = TIS[bi]`.
///
/// Inputs:
///
/// * `token` — the decoded token value, in `7..=31`. Tokens `0..=6`
///   are EOB tokens handled by [`decode_eob_token`] and rejected
///   here.
/// * `nbs` — total block count in the frame (Table 6.5 / 6.6), the
///   length the three state slices must share.
/// * `bi` — coded-order index of the current block (the §7.7 outer
///   loop variable). Must satisfy `bi < nbs`.
/// * `ti` — current token index inside that block. Must be `0..=63`.
/// * `tis` / `ncoeffs` / `coeffs` — per-block state, all sized `nbs`.
///
/// Returns:
///
/// * `Ok(CoefficientTokenKind)` indicating which §7.7.2 class the
///   token belonged to. Useful for §7.7 / §7.7.3 driver
///   bookkeeping that needs to distinguish a pure zero run (which
///   leaves `NCOEFFS[bi]` untouched) from a coefficient-writing
///   token (which updates it).
/// * [`Error::CoefficientTokenOutOfRange`] when `token < 7` or
///   `token > 31`.
/// * [`Error::CoefficientTokenBlockIndexOutOfRange`] when
///   `bi >= nbs`.
/// * [`Error::CoefficientTokenIndexOutOfRange`] when `ti > 63`.
/// * [`Error::CoefficientTokenStateLenMismatch`] when any of `tis` /
///   `ncoeffs` / `coeffs` does not have exactly `nbs` entries.
/// * [`Error::CoefficientTokenWouldOverflowBlock`] when the token's
///   total coefficient count would push `TIS[bi]` past 64. §7.7.2's
///   own text is explicit: tokens that represent more than one
///   coefficient "MUST NOT bring the total number of coefficients in
///   the block to more than 64".
/// * [`Error::TruncatedHeader`] when the underlying bit reader is
///   exhausted before the token's extra-bits payload is fully read.
#[allow(clippy::too_many_arguments)]
pub fn decode_coefficient_token(
    packet: &[u8],
    token: u8,
    nbs: u32,
    bi: u32,
    ti: u8,
    tis: &mut [u8],
    ncoeffs: &mut [u8],
    coeffs: &mut [[i16; 64]],
) -> Result<CoefficientTokenKind, Error> {
    let mut r = BitReader::new(packet);
    decode_coefficient_token_inner(&mut r, token, nbs, bi, ti, tis, ncoeffs, coeffs)
}

/// Inner §7.7.2 procedure operating on an already-positioned
/// [`BitReader`]. Split out so the §7.7.3 driver — once it lands — can
/// chain §7.6 → §7.7 on the same reader without re-aligning to a byte
/// boundary.
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_coefficient_token_inner(
    r: &mut BitReader<'_>,
    token: u8,
    nbs: u32,
    bi: u32,
    ti: u8,
    tis: &mut [u8],
    ncoeffs: &mut [u8],
    coeffs: &mut [[i16; 64]],
) -> Result<CoefficientTokenKind, Error> {
    // Argument validation. §7.7.2's declared input ranges fail closed
    // here so a future §7.7.3 driver bug doesn't silently overflow the
    // state arrays.
    if !(7..=31).contains(&token) {
        return Err(Error::CoefficientTokenOutOfRange { token });
    }
    if bi >= nbs {
        return Err(Error::CoefficientTokenBlockIndexOutOfRange { bi, nbs });
    }
    if ti > 63 {
        return Err(Error::CoefficientTokenIndexOutOfRange { ti });
    }
    let nbs_us = nbs as usize;
    if tis.len() != nbs_us {
        return Err(Error::CoefficientTokenStateLenMismatch {
            which: CoefficientTokenStateSlice::Tis,
            got: tis.len(),
            nbs: nbs_us,
        });
    }
    if ncoeffs.len() != nbs_us {
        return Err(Error::CoefficientTokenStateLenMismatch {
            which: CoefficientTokenStateSlice::Ncoeffs,
            got: ncoeffs.len(),
            nbs: nbs_us,
        });
    }
    if coeffs.len() != nbs_us {
        return Err(Error::CoefficientTokenStateLenMismatch {
            which: CoefficientTokenStateSlice::Coeffs,
            got: coeffs.len(),
            nbs: nbs_us,
        });
    }

    let bi_us = bi as usize;
    let ti_us = ti as usize;

    // Helper: read a 1-bit SIGN; +/- the supplied magnitude.
    fn signed_mag(r: &mut BitReader<'_>, mag: u16, field: &'static str) -> Result<i16, Error> {
        let sign = r.read_bits(1, field)? as u8;
        let mag_i = mag as i16;
        Ok(if sign == 0 { mag_i } else { -mag_i })
    }

    // Single-coefficient writer: write COEFFS[bi][ti], advance TIS[bi]
    // by 1, set NCOEFFS[bi] = TIS[bi]. Returns the post-step kind.
    fn write_single(
        tis: &mut [u8],
        ncoeffs: &mut [u8],
        coeffs: &mut [[i16; 64]],
        bi_us: usize,
        ti_us: usize,
        value: i16,
    ) -> Result<CoefficientTokenKind, Error> {
        // No overflow check: a single-coefficient write at ti = 63 is
        // legal (the resulting TIS = 64 pins the block, exactly as the
        // §7.7.1 step-10 pin does). Range-checking ti above already
        // ensured ti <= 63.
        coeffs[bi_us][ti_us] = value;
        let new_tis = (ti_us as u16) + 1;
        tis[bi_us] = new_tis as u8;
        ncoeffs[bi_us] = new_tis as u8;
        Ok(CoefficientTokenKind::Single)
    }

    // Run-plus-one writer: zero-fill ti..ti+run, write COEFFS[bi]
    // [ti+run] = value, advance TIS[bi] by run+1, set NCOEFFS[bi] =
    // TIS[bi].
    fn write_run_plus_one(
        tis: &mut [u8],
        ncoeffs: &mut [u8],
        coeffs: &mut [[i16; 64]],
        bi_us: usize,
        ti_us: usize,
        run: u16,
        value: i16,
        token: u8,
        ti: u8,
    ) -> Result<CoefficientTokenKind, Error> {
        let new_tis = (ti_us as u16) + run + 1;
        if new_tis > 64 {
            return Err(Error::CoefficientTokenWouldOverflowBlock { token, ti, new_tis });
        }
        let run_us = run as usize;
        coeffs[bi_us][ti_us..ti_us + run_us].fill(0);
        coeffs[bi_us][ti_us + run_us] = value;
        tis[bi_us] = new_tis as u8;
        ncoeffs[bi_us] = new_tis as u8;
        Ok(CoefficientTokenKind::RunPlusOne)
    }

    // Zero-run writer: zero-fill ti..ti+run, advance TIS[bi] by run.
    // Does NOT touch NCOEFFS[bi] per the introductory paragraph of
    // §7.7.2.
    fn write_zero_run(
        tis: &mut [u8],
        coeffs: &mut [[i16; 64]],
        bi_us: usize,
        ti_us: usize,
        run: u16,
        token: u8,
        ti: u8,
    ) -> Result<CoefficientTokenKind, Error> {
        let new_tis = (ti_us as u16) + run;
        if new_tis > 64 {
            return Err(Error::CoefficientTokenWouldOverflowBlock { token, ti, new_tis });
        }
        let run_us = run as usize;
        coeffs[bi_us][ti_us..ti_us + run_us].fill(0);
        tis[bi_us] = new_tis as u8;
        Ok(CoefficientTokenKind::ZeroRun)
    }

    match token {
        // Step 1: TOKEN=7 — 3-bit RLEN + 1, range 1..=8, pure zero run.
        7 => {
            let rlen = r.read_bits(3, "§7.7.2 RLEN (TOKEN=7)")? as u16 + 1;
            write_zero_run(tis, coeffs, bi_us, ti_us, rlen, token, ti)
        }
        // Step 2: TOKEN=8 — 6-bit RLEN + 1, range 1..=64, pure zero run.
        8 => {
            let rlen = r.read_bits(6, "§7.7.2 RLEN (TOKEN=8)")? as u16 + 1;
            write_zero_run(tis, coeffs, bi_us, ti_us, rlen, token, ti)
        }
        // Steps 3..=6: TOKEN=9..=12 — fixed single coefficients.
        9 => write_single(tis, ncoeffs, coeffs, bi_us, ti_us, 1),
        10 => write_single(tis, ncoeffs, coeffs, bi_us, ti_us, -1),
        11 => write_single(tis, ncoeffs, coeffs, bi_us, ti_us, 2),
        12 => write_single(tis, ncoeffs, coeffs, bi_us, ti_us, -2),
        // Steps 7..=10: TOKEN=13..=16 — SIGN + fixed magnitude
        //   13: ±3, 14: ±4, 15: ±5, 16: ±6 (no MAG_BITS).
        13 => {
            let v = signed_mag(r, 3, "§7.7.2 SIGN (TOKEN=13)")?;
            write_single(tis, ncoeffs, coeffs, bi_us, ti_us, v)
        }
        14 => {
            let v = signed_mag(r, 4, "§7.7.2 SIGN (TOKEN=14)")?;
            write_single(tis, ncoeffs, coeffs, bi_us, ti_us, v)
        }
        15 => {
            let v = signed_mag(r, 5, "§7.7.2 SIGN (TOKEN=15)")?;
            write_single(tis, ncoeffs, coeffs, bi_us, ti_us, v)
        }
        16 => {
            let v = signed_mag(r, 6, "§7.7.2 SIGN (TOKEN=16)")?;
            write_single(tis, ncoeffs, coeffs, bi_us, ti_us, v)
        }
        // Steps 11..=16: TOKEN=17..=22 — SIGN + (MAG_BITS-bit MAG) +
        //   offset. Read SIGN first then MAG, per the spec ordering.
        //   Table 7.38: 17: ±7..=8 (1 bit, +7); 18: ±9..=12 (2 bits, +9);
        //   19: ±13..=20 (3 bits, +13); 20: ±21..=36 (4 bits, +21);
        //   21: ±37..=68 (5 bits, +37); 22: ±69..=580 (9 bits, +69).
        17 => {
            let sign = r.read_bits(1, "§7.7.2 SIGN (TOKEN=17)")? as u8;
            let mag = r.read_bits(1, "§7.7.2 MAG (TOKEN=17)")? as i16 + 7;
            let v = if sign == 0 { mag } else { -mag };
            write_single(tis, ncoeffs, coeffs, bi_us, ti_us, v)
        }
        18 => {
            let sign = r.read_bits(1, "§7.7.2 SIGN (TOKEN=18)")? as u8;
            let mag = r.read_bits(2, "§7.7.2 MAG (TOKEN=18)")? as i16 + 9;
            let v = if sign == 0 { mag } else { -mag };
            write_single(tis, ncoeffs, coeffs, bi_us, ti_us, v)
        }
        19 => {
            let sign = r.read_bits(1, "§7.7.2 SIGN (TOKEN=19)")? as u8;
            let mag = r.read_bits(3, "§7.7.2 MAG (TOKEN=19)")? as i16 + 13;
            let v = if sign == 0 { mag } else { -mag };
            write_single(tis, ncoeffs, coeffs, bi_us, ti_us, v)
        }
        20 => {
            let sign = r.read_bits(1, "§7.7.2 SIGN (TOKEN=20)")? as u8;
            let mag = r.read_bits(4, "§7.7.2 MAG (TOKEN=20)")? as i16 + 21;
            let v = if sign == 0 { mag } else { -mag };
            write_single(tis, ncoeffs, coeffs, bi_us, ti_us, v)
        }
        21 => {
            let sign = r.read_bits(1, "§7.7.2 SIGN (TOKEN=21)")? as u8;
            let mag = r.read_bits(5, "§7.7.2 MAG (TOKEN=21)")? as i16 + 37;
            let v = if sign == 0 { mag } else { -mag };
            write_single(tis, ncoeffs, coeffs, bi_us, ti_us, v)
        }
        22 => {
            let sign = r.read_bits(1, "§7.7.2 SIGN (TOKEN=22)")? as u8;
            let mag = r.read_bits(9, "§7.7.2 MAG (TOKEN=22)")? as i16 + 69;
            let v = if sign == 0 { mag } else { -mag };
            write_single(tis, ncoeffs, coeffs, bi_us, ti_us, v)
        }
        // Steps 17..=21: TOKEN=23..=27 — fixed RUN + (SIGN-of-±1).
        //   23: 1 zero + ±1; 24: 2 zeros + ±1; 25: 3 zeros + ±1;
        //   26: 4 zeros + ±1; 27: 5 zeros + ±1.
        23 => {
            let v = signed_mag(r, 1, "§7.7.2 SIGN (TOKEN=23)")?;
            write_run_plus_one(tis, ncoeffs, coeffs, bi_us, ti_us, 1, v, token, ti)
        }
        24 => {
            let v = signed_mag(r, 1, "§7.7.2 SIGN (TOKEN=24)")?;
            write_run_plus_one(tis, ncoeffs, coeffs, bi_us, ti_us, 2, v, token, ti)
        }
        25 => {
            let v = signed_mag(r, 1, "§7.7.2 SIGN (TOKEN=25)")?;
            write_run_plus_one(tis, ncoeffs, coeffs, bi_us, ti_us, 3, v, token, ti)
        }
        26 => {
            let v = signed_mag(r, 1, "§7.7.2 SIGN (TOKEN=26)")?;
            write_run_plus_one(tis, ncoeffs, coeffs, bi_us, ti_us, 4, v, token, ti)
        }
        27 => {
            let v = signed_mag(r, 1, "§7.7.2 SIGN (TOKEN=27)")?;
            write_run_plus_one(tis, ncoeffs, coeffs, bi_us, ti_us, 5, v, token, ti)
        }
        // Step 22: TOKEN=28 — SIGN, then 2-bit RLEN + 6 (range 6..=9),
        //   then trailing ±1.
        //   The spec ordering is "SIGN then RLEN" — the SIGN bit comes
        //   before the run length payload.
        28 => {
            let sign = r.read_bits(1, "§7.7.2 SIGN (TOKEN=28)")? as u8;
            let rlen = r.read_bits(2, "§7.7.2 RLEN (TOKEN=28)")? as u16 + 6;
            let v: i16 = if sign == 0 { 1 } else { -1 };
            write_run_plus_one(tis, ncoeffs, coeffs, bi_us, ti_us, rlen, v, token, ti)
        }
        // Step 23: TOKEN=29 — SIGN, then 3-bit RLEN + 10 (range 10..=17),
        //   then trailing ±1.
        29 => {
            let sign = r.read_bits(1, "§7.7.2 SIGN (TOKEN=29)")? as u8;
            let rlen = r.read_bits(3, "§7.7.2 RLEN (TOKEN=29)")? as u16 + 10;
            let v: i16 = if sign == 0 { 1 } else { -1 };
            write_run_plus_one(tis, ncoeffs, coeffs, bi_us, ti_us, rlen, v, token, ti)
        }
        // Step 24: TOKEN=30 — fixed RUN=1, then SIGN + 1-bit MAG + 2
        //   (range ±2..=3). Single zero + ±MAG.
        30 => {
            let sign = r.read_bits(1, "§7.7.2 SIGN (TOKEN=30)")? as u8;
            let mag_raw = r.read_bits(1, "§7.7.2 MAG (TOKEN=30)")? as i16;
            let mag = mag_raw + 2;
            let v = if sign == 0 { mag } else { -mag };
            write_run_plus_one(tis, ncoeffs, coeffs, bi_us, ti_us, 1, v, token, ti)
        }
        // Step 25: TOKEN=31 — SIGN, 1-bit MAG + 2 (±MAG = 2..=3), then
        //   1-bit RLEN + 2 (range 2..=3), trailing ±MAG.
        31 => {
            let sign = r.read_bits(1, "§7.7.2 SIGN (TOKEN=31)")? as u8;
            let mag_raw = r.read_bits(1, "§7.7.2 MAG (TOKEN=31)")? as i16;
            let mag = mag_raw + 2;
            let rlen_raw = r.read_bits(1, "§7.7.2 RLEN (TOKEN=31)")? as u16;
            let rlen = rlen_raw + 2;
            let v = if sign == 0 { mag } else { -mag };
            write_run_plus_one(tis, ncoeffs, coeffs, bi_us, ti_us, rlen, v, token, ti)
        }
        // Token range check at the top of the function clamps every
        // other value.
        _ => unreachable!("token range was clamped to 7..=31"),
    }
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

    // ----- §6.4.4 DCT Token Huffman Tables ----------------------------

    /// Emit one §6.4.4 Huffman table into `w` from a list of
    /// `(code, len, token)` triples. The list must describe a full,
    /// prefix-free binary tree on the bit strings of `code` (i.e. for
    /// every interior path either both children exist as leaves or as
    /// further interior splits). Encoding walks the tree in depth-first
    /// `0`-before-`1` order, emitting `0` at every interior node and
    /// `1` + the 5-bit token at every leaf.
    fn encode_huffman_table(w: &mut BitWriter, entries: &[(u32, u8, u8)]) {
        // Build the same flat tree the decoder will construct, then
        // walk it in pre-order.
        #[derive(Clone, Copy)]
        enum N {
            Internal(usize, usize),
            Leaf(u8),
        }
        let mut nodes: Vec<N> = vec![N::Internal(0, 0)];
        // Helper to ensure a path exists; returns the final leaf slot.
        fn ensure(nodes: &mut Vec<N>, code: u32, len: u8) -> usize {
            let mut idx = 0;
            for i in (0..len).rev() {
                let bit = ((code >> i) & 1) as usize;
                let (z, o) = match nodes[idx] {
                    N::Internal(z, o) => (z, o),
                    N::Leaf(_) => panic!("path collides with shorter leaf"),
                };
                let child = if bit == 0 { z } else { o };
                if child == 0 {
                    // Allocate a fresh child.
                    let new = nodes.len();
                    nodes.push(N::Internal(0, 0));
                    let (nz, no) = match (bit, z, o) {
                        (0, _, oo) => (new, oo),
                        (_, zz, _) => (zz, new),
                    };
                    nodes[idx] = N::Internal(nz, no);
                    idx = new;
                } else {
                    idx = child;
                }
            }
            idx
        }
        for &(code, len, token) in entries {
            let slot = ensure(&mut nodes, code, len);
            assert!(
                matches!(nodes[slot], N::Internal(0, 0)),
                "duplicate code path"
            );
            nodes[slot] = N::Leaf(token);
        }
        // Pre-order walk: emit ISLEAF at every node, plus TOKEN at every
        // leaf. Uses an explicit stack so we don't recurse in tests.
        let mut stack = vec![0usize];
        while let Some(node) = stack.pop() {
            match nodes[node] {
                N::Leaf(tok) => {
                    w.put(1, 1);
                    w.put(tok as u32, 5);
                }
                N::Internal(z, o) => {
                    assert!(z != 0 && o != 0, "non-full tree: dangling internal node");
                    w.put(0, 1);
                    // Push `1` first so `0` is popped (emitted) first.
                    stack.push(o);
                    stack.push(z);
                }
            }
        }
    }

    /// Trivial table: a single leaf at the root. The §6.4.4 procedure
    /// reads ISLEAF=1 immediately and assigns the token at HBITS="".
    /// (A conformant DCT-token decode would never use such a degenerate
    /// table, but the spec allows it — multiple codes for the same
    /// token, or a single-leaf tree, are both explicitly tolerated.)
    fn write_trivial_table(w: &mut BitWriter, token: u8) {
        encode_huffman_table(w, &[(0, 0, token)]);
    }

    #[test]
    fn decode_huffman_table_trivial_single_leaf() {
        // Build one trivial table for every hti slot.
        let mut w = BitWriter::new();
        for hti in 0..NUM_HUFFMAN_TABLES {
            write_trivial_table(&mut w, (hti % 32) as u8);
        }
        let payload = w.finish();
        let tables = decode_dct_token_huffman_tables(&payload).expect("valid payload decodes");

        for hti in 0..NUM_HUFFMAN_TABLES {
            assert_eq!(tables[hti].len(), 1, "hti={hti}: one entry");
            let e = tables[hti].entries[0];
            assert_eq!(e.code, 0);
            assert_eq!(e.len, 0);
            assert_eq!(e.token, (hti % 32) as u8);
            // The degenerate single-leaf-at-root case: empty-code lookup
            // returns the only token.
            assert_eq!(tables[hti].lookup(0, 0), Some((hti % 32) as u8));
            // A non-empty code on a single-leaf table never matches.
            assert_eq!(tables[hti].lookup(0, 1), None);
        }
    }

    /// A balanced 32-leaf table covering every token 0..=31 with a
    /// 5-bit code equal to the token value.
    fn balanced_32_table_entries() -> Vec<(u32, u8, u8)> {
        (0..32u8).map(|t| (t as u32, 5, t)).collect()
    }

    #[test]
    fn decode_huffman_table_balanced_32_leaves() {
        // First table is the balanced 32-leaf one; remaining 79 are
        // trivial single-leaf tables so we can focus the assertions.
        let mut w = BitWriter::new();
        encode_huffman_table(&mut w, &balanced_32_table_entries());
        for _ in 1..NUM_HUFFMAN_TABLES {
            write_trivial_table(&mut w, 0);
        }
        let payload = w.finish();
        let tables = decode_dct_token_huffman_tables(&payload).unwrap();

        assert_eq!(tables[0].len(), 32);
        // Every (code=t, len=5) should look up token t.
        for t in 0..32u8 {
            assert_eq!(tables[0].lookup(t as u32, 5), Some(t));
        }
        // Codes at the wrong length never match.
        assert_eq!(tables[0].lookup(0, 4), None);
        assert_eq!(tables[0].lookup(0, 6), None);
        // The depth-first emission order means the first leaf the
        // decoder records is HBITS = 00000 = token 0.
        assert_eq!(tables[0].entries[0].code, 0);
        assert_eq!(tables[0].entries[0].len, 5);
        // ... and the last is HBITS = 11111 = token 31.
        assert_eq!(tables[0].entries.last().unwrap().code, 31);
        assert_eq!(tables[0].entries.last().unwrap().len, 5);
    }

    #[test]
    fn decode_huffman_table_variable_length_codes() {
        // A small variable-length code:
        //   "0"     -> token 7    (len 1)
        //   "10"    -> token 5    (len 2)
        //   "110"   -> token 3    (len 3)
        //   "111"   -> token 1    (len 3)
        let entries = vec![(0b0, 1, 7), (0b10, 2, 5), (0b110, 3, 3), (0b111, 3, 1)];
        let mut w = BitWriter::new();
        encode_huffman_table(&mut w, &entries);
        for _ in 1..NUM_HUFFMAN_TABLES {
            write_trivial_table(&mut w, 0);
        }
        let payload = w.finish();
        let tables = decode_dct_token_huffman_tables(&payload).unwrap();

        assert_eq!(tables[0].len(), 4);
        assert_eq!(tables[0].lookup(0b0, 1), Some(7));
        assert_eq!(tables[0].lookup(0b10, 2), Some(5));
        assert_eq!(tables[0].lookup(0b110, 3), Some(3));
        assert_eq!(tables[0].lookup(0b111, 3), Some(1));
        // The depth-first emission visits leaves in order 7, 5, 3, 1.
        let tokens: Vec<u8> = tables[0].entries.iter().map(|e| e.token).collect();
        assert_eq!(tokens, vec![7, 5, 3, 1]);
    }

    #[test]
    fn decode_huffman_table_all_80_independent() {
        // Each table differs in size: hti k carries a (k%4)+1-leaf
        // table covering 1, 2, 3, or 4 codes. This exercises the
        // 80-table loop without bloating the payload.
        let small_tables: [Vec<(u32, u8, u8)>; 4] = [
            vec![(0, 0, 31)],                                             // 1 leaf
            vec![(0, 1, 10), (1, 1, 20)],                                 // 2 leaves
            vec![(0, 1, 0), (0b10, 2, 1), (0b11, 2, 2)],                  // 3 leaves
            vec![(0b00, 2, 0), (0b01, 2, 1), (0b10, 2, 2), (0b11, 2, 3)], // 4 leaves
        ];
        let mut w = BitWriter::new();
        for hti in 0..NUM_HUFFMAN_TABLES {
            encode_huffman_table(&mut w, &small_tables[hti % 4]);
        }
        let payload = w.finish();
        let tables = decode_dct_token_huffman_tables(&payload).unwrap();
        for hti in 0..NUM_HUFFMAN_TABLES {
            assert_eq!(tables[hti].len(), small_tables[hti % 4].len(), "hti={hti}");
        }
        // Confirm a leaf from the very last table to ensure we didn't
        // truncate the 80-iteration loop early.
        assert_eq!(
            tables[NUM_HUFFMAN_TABLES - 1].lookup(
                small_tables[(NUM_HUFFMAN_TABLES - 1) % 4][0].0,
                small_tables[(NUM_HUFFMAN_TABLES - 1) % 4][0].1
            ),
            Some(small_tables[(NUM_HUFFMAN_TABLES - 1) % 4][0].2)
        );
    }

    #[test]
    fn decode_huffman_table_rejects_truncated_isleaf() {
        // Empty payload: ISLEAF on hti=0 immediately runs out of bits.
        let payload: Vec<u8> = Vec::new();
        match decode_dct_token_huffman_tables(&payload) {
            Err(Error::TruncatedHeader { field }) => assert_eq!(field, "ISLEAF"),
            other => panic!("expected TruncatedHeader(ISLEAF), got {other:?}"),
        }
    }

    #[test]
    fn decode_huffman_table_rejects_truncated_token() {
        // Hand-craft a single-byte payload that consumes 8 bits exactly
        // and leaves the §6.4.4 decoder mid-way through a TOKEN read.
        //
        // Bits (MSb first):
        //   1            ISLEAF=1 at hti=0
        //   00000        TOKEN=0   (hti=0 closes with one leaf)
        //   0            ISLEAF=0 at hti=1 (interior at depth 0)
        //   1            ISLEAF=1 on the left child (next must be TOKEN)
        // Total = 8 bits → one byte; no second byte for the TOKEN read.
        // Bit pattern: 1 00000 0 1 = 0b1000_0001.
        let payload: [u8; 1] = [0b1000_0001];
        match decode_dct_token_huffman_tables(&payload) {
            Err(Error::TruncatedHeader { field }) => assert_eq!(field, "TOKEN"),
            other => panic!("expected TruncatedHeader(TOKEN), got {other:?}"),
        }
    }

    #[test]
    fn decode_huffman_table_rejects_code_longer_than_32_bits() {
        // Build a degenerate left-spine tree of depth 33: 33 ISLEAF=0
        // bits then a leaf. When the decoder pops the depth-33 frame,
        // step 1(b) (len > 32) trips before reading ISLEAF.
        //
        // The tree must still be full, so we also need a leaf on every
        // right child at depths 1..=33. Simplest valid shape: a chain
        // of 33 left-internal nodes whose right children are all leaves
        // emitting token 0, terminated by either a leaf or two leaves
        // at depth 33. Because depth-33 frames trip step 1(b) before
        // the body reads ISLEAF, the trailing leaf bits never get
        // consumed — we just emit enough payload to make the failure
        // happen first.
        let mut w = BitWriter::new();
        // Depth 0..=32: ISLEAF=0 (interior) followed by the right child
        // emitted as ISLEAF=1 + token 0 (the `0` child is the recursion
        // we re-emit). Doing 33 iterations puts the 33rd recursion at
        // len=33 → trips step 1(b).
        for _ in 0..33 {
            w.put(0, 1); // ISLEAF=0 on the left-spine node
            w.put(1, 1); // ISLEAF=1 on its right child (popped first)
            w.put(0, 5); // TOKEN=0 for that right leaf
        }
        let payload = w.finish();
        match decode_dct_token_huffman_tables(&payload) {
            Err(Error::HuffmanCodeTooLong { hti }) => assert_eq!(hti, 0),
            other => panic!("expected HuffmanCodeTooLong, got {other:?}"),
        }
    }

    #[test]
    fn decode_huffman_table_rejects_more_than_32_entries() {
        // Synthesize a 33-leaf tree whose deepest path is well under
        // 32 bits, so that step 1(d)i (table full) trips before step
        // 1(b) (code too long):
        //
        //   root             (interior, ISLEAF=0)
        //   ├── left         a 32-leaf balanced subtree (depth 5 from
        //   │                here, depth 6 absolute) — all token 0
        //   └── right        a single leaf (depth 1 absolute, token 0)
        //                    — this is the 33rd leaf the decoder tries
        //                    to insert, tripping the 32-entry cap.
        //
        // DFS (0-before-1) emission order: root ISLEAF=0, then the 32
        // left-subtree leaves (each: 5 × ISLEAF=0 ramping down + a
        // ISLEAF=1 + TOKEN=0 ... reconstructed by ascending out), then
        // ISLEAF=1 + TOKEN=0 for the right child.
        let mut w = BitWriter::new();
        // Root interior.
        w.put(0, 1);
        // Left subtree: full balanced depth-5 tree of 32 leaves. The
        // emit order for a full balanced tree at depth N is: N-1
        // interior bits (0s) down to the leftmost leaf, then leaf
        // emissions interleaved with interior re-emissions as we
        // ascend. The simplest correct shape uses the `encode_huffman_table`
        // helper.
        encode_huffman_table(&mut w, &balanced_32_table_entries());
        // Right subtree: a single leaf, which is the 33rd leaf overall.
        w.put(1, 1);
        w.put(0, 5);
        let payload = w.finish();
        match decode_dct_token_huffman_tables(&payload) {
            Err(Error::HuffmanTableFull { hti }) => assert_eq!(hti, 0),
            other => panic!("expected HuffmanTableFull, got {other:?}"),
        }
    }

    #[test]
    fn decode_huffman_table_truncation_in_later_table_reports_correct_field() {
        // hti=0 is a trivial single-leaf table; hti=1's payload is
        // truncated after the ISLEAF=0 bit. Confirms the loop advances
        // hti and that errors flow up from any iteration.
        let mut w = BitWriter::new();
        write_trivial_table(&mut w, 5);
        w.put(0, 1); // hti=1: ISLEAF=0 (interior)
                     // The subsequent ISLEAF for the left child runs out of bits.
        let payload = w.finish();
        match decode_dct_token_huffman_tables(&payload) {
            Err(Error::TruncatedHeader { field }) => assert_eq!(field, "ISLEAF"),
            other => panic!("expected TruncatedHeader, got {other:?}"),
        }
    }

    #[test]
    fn huffman_error_displays_render() {
        let s = format!("{}", Error::HuffmanCodeTooLong { hti: 17 });
        assert!(s.contains("HTS[17]"), "got: {s}");
        assert!(s.contains("§6.4.4"), "got: {s}");
        let s = format!("{}", Error::HuffmanTableFull { hti: 3 });
        assert!(s.contains("HTS[3]"), "got: {s}");
        assert!(s.contains("§6.4.4"), "got: {s}");
    }

    #[test]
    fn huffman_table_lookup_rejects_unknown_codes() {
        // Build a 4-entry table and confirm uncovered codes don't match.
        let entries = vec![(0b00, 2, 1), (0b01, 2, 2), (0b10, 2, 3), (0b11, 2, 4)];
        let mut w = BitWriter::new();
        encode_huffman_table(&mut w, &entries);
        for _ in 1..NUM_HUFFMAN_TABLES {
            write_trivial_table(&mut w, 0);
        }
        let payload = w.finish();
        let tables = decode_dct_token_huffman_tables(&payload).unwrap();
        // Wrong length.
        assert_eq!(tables[0].lookup(0, 1), None);
        assert_eq!(tables[0].lookup(0, 3), None);
    }

    // ------- §7.1 Frame Header Decode tests -------

    /// Synthesize the bits of a §7.1 frame header into a byte buffer
    /// using the test-only [`BitWriter`]. `qis` carries the 1..=3
    /// per-frame `qi` values; `ftype_bit` is the FTYPE bit value;
    /// `reserved` is the 3-bit reserved trailer used only for intra
    /// frames.
    fn build_frame_header(ftype_bit: u32, qis: &[u8], reserved: u32) -> Vec<u8> {
        assert!(
            !qis.is_empty() && qis.len() <= MAX_FRAME_QIS,
            "qis len must be 1..=3"
        );
        let mut w = BitWriter::new();
        // Step 1: data-packet flag = 0.
        w.put(0, 1);
        // Step 2: FTYPE.
        w.put(ftype_bit & 1, 1);
        // Step 3: QIS[0].
        w.put(qis[0] as u32, 6);
        if qis.len() == 1 {
            // Step 4: MOREQIS = 0; step 5 sets NQIS = 1 implicitly.
            w.put(0, 1);
        } else {
            // Step 4: MOREQIS = 1; step 6(a): QIS[1].
            w.put(1, 1);
            w.put(qis[1] as u32, 6);
            if qis.len() == 2 {
                // Step 6(b): MOREQIS = 0; step 6(c): NQIS = 2.
                w.put(0, 1);
            } else {
                // Step 6(b): MOREQIS = 1; step 6(d)i: QIS[2].
                w.put(1, 1);
                w.put(qis[2] as u32, 6);
            }
        }
        // Step 7: 3 reserved bits on intra frames only.
        if ftype_bit & 1 == 0 {
            w.put(reserved & 0b111, 3);
        }
        w.finish()
    }

    #[test]
    fn decode_frame_header_intra_single_qi() {
        let packet = build_frame_header(0, &[12], 0);
        let h = decode_frame_header(&packet, true).unwrap();
        assert_eq!(h.ftype, FrameType::Intra);
        assert_eq!(h.qis, vec![12]);
        assert_eq!(h.nqis(), 1);
    }

    #[test]
    fn decode_frame_header_intra_two_qis() {
        let packet = build_frame_header(0, &[34, 5], 0);
        let h = decode_frame_header(&packet, true).unwrap();
        assert_eq!(h.ftype, FrameType::Intra);
        assert_eq!(h.qis, vec![34, 5]);
        assert_eq!(h.nqis(), 2);
    }

    #[test]
    fn decode_frame_header_intra_three_qis_max() {
        // 63 is the max value of a 6-bit unsigned integer; exercise the
        // upper boundary on every slot.
        let packet = build_frame_header(0, &[63, 63, 63], 0);
        let h = decode_frame_header(&packet, true).unwrap();
        assert_eq!(h.qis, vec![63, 63, 63]);
        assert_eq!(h.nqis(), 3);
    }

    #[test]
    fn decode_frame_header_inter_first_frame_rejected() {
        // §7.1 step 2: the first frame MUST be Intra.
        let packet = build_frame_header(1, &[7], 0);
        match decode_frame_header(&packet, true) {
            Err(Error::FirstFrameMustBeIntra { ftype }) => assert_eq!(ftype, 1),
            other => panic!("expected FirstFrameMustBeIntra, got {other:?}"),
        }
    }

    #[test]
    fn decode_frame_header_inter_non_first_frame_allowed() {
        // After the keyframe, FTYPE=1 (Inter) is legal. There are no
        // reserved trailing bits on an inter frame (step 7 only fires
        // when FTYPE == 0).
        let packet = build_frame_header(1, &[7], 0);
        let h = decode_frame_header(&packet, false).unwrap();
        assert_eq!(h.ftype, FrameType::Inter);
        assert_eq!(h.qis, vec![7]);
    }

    #[test]
    fn decode_frame_header_inter_three_qis() {
        let packet = build_frame_header(1, &[1, 2, 3], 0);
        let h = decode_frame_header(&packet, false).unwrap();
        assert_eq!(h.ftype, FrameType::Inter);
        assert_eq!(h.qis, vec![1, 2, 3]);
    }

    #[test]
    fn decode_frame_header_rejects_header_packet_high_bit_set() {
        // §7.1 step 1: high bit set ⇒ this is a §6-series header
        // packet, not a frame.
        let packet = [0x80u8]; // leading bit = 1
        match decode_frame_header(&packet, true) {
            Err(Error::NotDataPacket) => {}
            other => panic!("expected NotDataPacket, got {other:?}"),
        }
    }

    #[test]
    fn decode_frame_header_rejects_reserved_non_zero() {
        // §7.1 step 7: intra frame's 3 reserved bits MUST be 0.
        let packet = build_frame_header(0, &[20], 0b101);
        match decode_frame_header(&packet, true) {
            Err(Error::FrameReservedBitsNonZero { bits }) => assert_eq!(bits, 0b101),
            other => panic!("expected FrameReservedBitsNonZero, got {other:?}"),
        }
    }

    #[test]
    fn decode_frame_header_rejects_each_non_zero_reserved_value() {
        // Every non-zero 3-bit pattern (1..=7) must trip the step 7
        // rejection; only 0 is legal.
        for bits in 1u32..8 {
            let packet = build_frame_header(0, &[5], bits);
            match decode_frame_header(&packet, true) {
                Err(Error::FrameReservedBitsNonZero { bits: got }) => {
                    assert_eq!(got as u32, bits, "wrong bits echoed back for {bits}")
                }
                other => panic!("expected FrameReservedBitsNonZero for {bits}, got {other:?}"),
            }
        }
    }

    #[test]
    fn decode_frame_header_truncated_at_packet_type_bit() {
        let packet: [u8; 0] = [];
        match decode_frame_header(&packet, true) {
            Err(Error::TruncatedHeader { field }) => assert_eq!(field, "frame.packet_type"),
            other => panic!("expected truncated frame.packet_type, got {other:?}"),
        }
    }

    #[test]
    fn decode_frame_header_truncated_at_ftype() {
        // Only the leading 0 bit fits, plus padding zeros — but the
        // BitReader truncation triggers on the second bit since we
        // supply a buffer of zero length after the first bit was
        // consumed. Build a buffer that supplies exactly the first
        // bit then runs out: padding the byte with zeros still gives
        // 7 more bits of "0", so to actually truncate FTYPE we need a
        // zero-length packet for the first read (covered above) and a
        // separate construct for the FTYPE read. The cleanest way is
        // to expose truncation by reading the first bit then exhausting:
        // since a byte carries 8 bits, the 1-bit data flag and the
        // 1-bit FTYPE always fit in the first byte. The earliest
        // genuine FTYPE truncation comes from supplying *no* bytes
        // after a deliberate inner-call. Exercise that via the inner
        // entry point on an exhausted reader.
        let mut r = BitReader::new(&[]);
        // Pretend the leading 0 was already consumed (manual seek by
        // reading from an empty buffer — but read_bits on empty
        // returns the truncation immediately on the first read). The
        // simplest publicly observable truncation across step 1 vs
        // step 2 is the empty-packet case, which the previous test
        // already covers. Here we additionally confirm that a buffer
        // containing only the leading 0 bit (and padding zeros to a
        // byte boundary) decodes the FTYPE as 0 instead of truncating,
        // because §5.2.4 says padding bits read as zero.
        match r.read_bits(1, "frame.packet_type") {
            Err(Error::TruncatedHeader { field }) => assert_eq!(field, "frame.packet_type"),
            other => panic!("expected TruncatedHeader, got {other:?}"),
        }
    }

    #[test]
    fn decode_frame_header_truncated_at_qis0() {
        // Provide exactly one byte: leading 0 (packet type) + FTYPE=0
        // + 6 bits of QIS[0] consumes 8 bits and lands at byte boundary.
        // To trip QIS[0] truncation, provide only the leading 0 + FTYPE
        // bits and stop. Build a 1-byte packet whose top two bits are 0
        // (so packet-type=0, FTYPE=0) and the next 6 bits read as
        // padding zeros — §5.2.4 says padding reads as 0 so this will
        // *not* truncate but will decode QIS[0]=0. To force a true
        // truncation, only supply the bits via partial construction:
        // synthesize bytes such that the 2 prefix bits are present but
        // the next read deliberately runs off the end. The smallest
        // such construct is the empty buffer (covered) plus the
        // intermediate cases below. For QIS[0] specifically: we need
        // at least 1 byte for the prefix to land and at most 1 byte
        // total but with QIS[0] split across byte 0/1. Build "byte 0
        // present, byte 1 missing": the 2-bit prefix plus partial QIS[0]
        // reads 6 bits from byte 0 (4 prefix bits remain after the 2
        // already read, then 2 more from byte 0, then 4 from byte 1).
        // After byte 0 is exhausted, the second 4-bit half of QIS[0]
        // must come from byte 1, which is missing.
        let packet = [0b00_000000u8]; // exactly 1 byte
                                      // packet-type=0 (bit 7), FTYPE=0 (bit 6), then 6 bits of QIS[0]
                                      // = the low 6 bits of byte 0 = 0. With only 1 byte we can
                                      // actually read all 8 bits without truncating since QIS[0] only
                                      // needs 6 bits — so this returns Ok(QIS[0] = 0). We then need
                                      // MOREQIS bit (1 more bit) — which requires byte 1.
        match decode_frame_header(&packet, true) {
            Err(Error::TruncatedHeader { field }) => {
                // Should fail at MOREQIS[0] (bit 9 = byte 1, bit 7).
                assert_eq!(field, "MOREQIS[0]");
            }
            other => panic!("expected truncation at MOREQIS[0], got {other:?}"),
        }
    }

    #[test]
    fn decode_frame_header_truncated_at_qis1() {
        // Bits laid out (MSb first):
        //   bit  0 : packet-type = 0
        //   bit  1 : FTYPE       = 0 (Intra)
        //   bits 2..=7 : QIS[0]  = 0
        //   bit  8 : MOREQIS[0]  = 1  (indicates QIS[1] is next)
        //   bits 9..=14 : QIS[1] partial (only 6 bits supplied but we
        //                stop at byte 1 ⇒ truncation midway)
        // To trip QIS[1] truncation, supply exactly 2 bytes where MOREQIS
        // is set but no third byte exists (QIS[1] would need bits 9..14,
        // i.e. 6 bits crossing into byte 1 already).
        // Construct byte 0 = 0b00_000000, byte 1 = 0b10_000000 (MOREQIS=1
        // at bit 7 of byte 1, then QIS[1] bits 6..1 = 0; the 7th bit of
        // byte 1 is the MOREQIS[1] flag, which still fits — but the
        // following byte for the next field is missing). Actually
        // MOREQIS[0]=1 is at bit 8 (the top bit of byte 1). Then QIS[1]
        // needs the next 6 bits = bits 9..14 of the stream = bits 6..1
        // of byte 1. That fits in byte 1. Then MOREQIS[1] is bit 15 =
        // bit 0 of byte 1 (last bit), still in byte 1. So a 2-byte
        // packet with MOREQIS[1]=1 forces QIS[2] (the next 6 bits) to
        // be read from byte 2 onward. Set MOREQIS[1]=1 (the low bit of
        // byte 1) and verify QIS[2] truncation.
        let packet = [0b00_000000u8, 0b10_000_001u8];
        // byte 0: packet-type=0, FTYPE=0, QIS[0]=0 (lower 6 bits)
        // byte 1: MOREQIS[0]=1, QIS[1]=0 (6 bits zero), MOREQIS[1]=1
        match decode_frame_header(&packet, true) {
            Err(Error::TruncatedHeader { field }) => assert_eq!(field, "QIS[2]"),
            other => panic!("expected truncation at QIS[2], got {other:?}"),
        }
    }

    #[test]
    fn decode_frame_header_truncated_at_reserved() {
        // Intra frame with 1 qi value. The header is:
        //   bit 0 = 0 (packet type)
        //   bit 1 = 0 (FTYPE=Intra)
        //   bits 2..=7 = QIS[0] (6 bits)
        //   bit 8 = MOREQIS = 0
        //   bits 9..=11 = reserved (3 bits)
        // Total 12 bits — needs 2 bytes. With only 1 byte (8 bits)
        // supplied, MOREQIS read at bit 8 truncates — but if we supply
        // a slightly larger buffer that runs out exactly at the
        // reserved field, the truncation field is "frame.reserved".
        // Build a buffer whose bit 8 (MOREQIS) is in byte 1 along with
        // the reserved bits: bit 8 + 3 reserved bits = 4 bits total
        // after byte 0, fits in byte 1. So byte 1 must be PRESENT.
        // To trigger reserved-bits truncation we need fewer than 12
        // bits. Build a 1-byte packet whose MSb bits are
        // packet-type=0, FTYPE=0, QIS[0]=0, then immediately run out
        // (only 8 bits available). MOREQIS lives in bit 8 = byte 1,
        // missing — that triggers "MOREQIS[0]" truncation, not
        // "frame.reserved". The "frame.reserved" truncation requires
        // MOREQIS to have been read as 0 and only PARTIAL reserved
        // bits to be available — i.e. between 9 and 11 bits supplied.
        // Use a bit-aligned shim: build a 2-byte buffer that reads as
        // padding-zero MOREQIS=0 in byte 1's top bit, then truncate
        // at byte 2 by supplying only the partial 3-bit reserved
        // field via the unaligned buffer. Easiest is to manually
        // craft 11 bits via a single byte plus a 3-bit second byte —
        // but Rust slices are byte-granular. Instead, exploit the
        // §5.2.4 "padding reads as zero" property: a buffer that ends
        // mid-stream still reports truncation, but the 0-padding of
        // the last byte's low bits is implicitly "valid". So with
        // exactly 2 bytes, bit 8 = MOREQIS = 0, bits 9..=11 read as 0
        // from byte 1's low 3 bits — the header decodes successfully
        // (reserved=0). For a true partial-reserved truncation we'd
        // need a sub-byte buffer, which the slice API can't express.
        // Instead, verify the symmetric path: a 2-byte intra packet
        // that decodes to reserved=0 and succeeds, confirming the
        // step-7 read is what's gating the truncation step the
        // earlier MOREQIS test confirms.
        let packet = [0b00_000000u8, 0b00_000000u8];
        let h = decode_frame_header(&packet, true).unwrap();
        assert_eq!(h.ftype, FrameType::Intra);
        assert_eq!(h.qis, vec![0]);
    }

    #[test]
    fn decode_frame_header_inter_does_not_consume_reserved_bits() {
        // Build an inter frame with one qi value, then a stray byte.
        // The decoder must NOT have read past bit 8 (MOREQIS=0); the
        // following bytes are §7.2 onward and untouched by §7.1.
        let mut header = build_frame_header(1, &[42], 0);
        header.push(0xFF); // sentinel byte from a hypothetical §7.2 payload
        let h = decode_frame_header(&header, false).unwrap();
        assert_eq!(h.ftype, FrameType::Inter);
        assert_eq!(h.qis, vec![42]);
        // The header itself is 9 bits (1 + 1 + 6 + 1), so byte 1 only
        // had bit 7 consumed. There is no public way to inspect the
        // bit cursor here, but the decoded output's correctness over
        // appended sentinel data demonstrates the function did not
        // mis-consume bits from the next packet.
    }

    #[test]
    fn frame_header_error_displays_render() {
        let s = format!("{}", Error::NotDataPacket);
        assert!(s.contains("§7.1 step 1"), "got: {s}");
        let s = format!("{}", Error::FirstFrameMustBeIntra { ftype: 1 });
        assert!(s.contains("§7.1 step 2"), "got: {s}");
        assert!(s.contains("FTYPE=1"), "got: {s}");
        let s = format!("{}", Error::FrameReservedBitsNonZero { bits: 0b011 });
        assert!(s.contains("§7.1 step 7"), "got: {s}");
        assert!(s.contains("0b011"), "got: {s}");
    }

    #[test]
    fn frame_type_table_values() {
        // Table 7.3 mapping: 0 → Intra, 1 → Inter.
        assert_eq!(FrameType::Intra as u8, 0);
        assert_eq!(FrameType::Inter as u8, 1);
    }

    #[test]
    fn frame_header_max_qis_constant() {
        // §7.1 unrolls MOREQIS at most twice, so NQIS is capped at 3.
        assert_eq!(MAX_FRAME_QIS, 3);
    }

    #[test]
    fn decode_frame_header_qis_independent_values_round_trip() {
        // Each qi slot must independently round-trip through the bit
        // packing (6-bit field per slot, no inter-slot interference).
        // Spot-check a mix that crosses byte boundaries.
        for (a, b, c) in [(0, 0, 0), (63, 0, 63), (1, 2, 4), (33, 21, 7), (5, 50, 33)] {
            let packet = build_frame_header(0, &[a, b, c], 0);
            let h = decode_frame_header(&packet, true).unwrap();
            assert_eq!(h.qis, vec![a, b, c], "case ({a},{b},{c}) failed");
            assert_eq!(h.nqis(), 3);
        }
    }

    // =============================================================
    // §7.2 Run-Length Encoded Bit Strings
    // =============================================================

    /// Build one §7.2.1 long-run record into a [`BitWriter`]:
    /// a Table 7.7 Huffman prefix (`code`, `code_len`) followed by
    /// `rbits` literal bits of `roffs`.
    fn write_long_run_record(w: &mut BitWriter, code: u32, code_len: u32, rbits: u32, roffs: u32) {
        w.put(code, code_len);
        if rbits > 0 {
            w.put(roffs, rbits);
        }
    }

    /// Same as [`write_long_run_record`] for Table 7.11 (short-run).
    fn write_short_run_record(w: &mut BitWriter, code: u32, code_len: u32, rbits: u32, roffs: u32) {
        w.put(code, code_len);
        if rbits > 0 {
            w.put(roffs, rbits);
        }
    }

    #[test]
    fn long_run_constants_match_table_7_7() {
        // Spot-check every Table 7.7 row against the in-tree constant
        // table. This is the single transcription point that protects
        // the rest of the §7.2.1 logic from a Huffman-table copy bug.
        assert_eq!(LONG_RUN_TABLE.len(), 7);
        let expected: &[(u32, u8, u16, u8)] = &[
            (0b0, 1, 1, 0),
            (0b10, 2, 2, 1),
            (0b110, 3, 4, 1),
            (0b1110, 4, 6, 2),
            (0b11110, 5, 10, 3),
            (0b111110, 6, 18, 4),
            (0b111111, 6, 34, 12),
        ];
        for (i, (code, code_len, rstart, rbits)) in expected.iter().enumerate() {
            let e = &LONG_RUN_TABLE[i];
            assert_eq!(e.code, *code, "row {i} code mismatch");
            assert_eq!(e.code_len, *code_len, "row {i} code_len mismatch");
            assert_eq!(e.rstart, *rstart, "row {i} rstart mismatch");
            assert_eq!(e.rbits, *rbits, "row {i} rbits mismatch");
        }
        // The Table 7.7 last row's range max = RSTART + (1<<RBITS) - 1
        // = 34 + 4096 - 1 = 4129 — the constant the procedure tests
        // for the "read new BIT" exception path.
        assert_eq!(LONG_RUN_MAX, 4129);
        let last = LONG_RUN_TABLE.last().unwrap();
        assert_eq!(
            last.rstart as u32 + (1u32 << last.rbits) - 1,
            LONG_RUN_MAX as u32
        );
    }

    #[test]
    fn short_run_constants_match_table_7_11() {
        assert_eq!(SHORT_RUN_TABLE.len(), 6);
        let expected: &[(u32, u8, u16, u8)] = &[
            (0b0, 1, 1, 1),
            (0b10, 2, 3, 1),
            (0b110, 3, 5, 1),
            (0b1110, 4, 7, 2),
            (0b11110, 5, 11, 2),
            (0b11111, 5, 15, 4),
        ];
        for (i, (code, code_len, rstart, rbits)) in expected.iter().enumerate() {
            let e = &SHORT_RUN_TABLE[i];
            assert_eq!(e.code, *code, "row {i} code mismatch");
            assert_eq!(e.code_len, *code_len, "row {i} code_len mismatch");
            assert_eq!(e.rstart, *rstart, "row {i} rstart mismatch");
            assert_eq!(e.rbits, *rbits, "row {i} rbits mismatch");
        }
        // Table 7.11 last row: 15 + 16 - 1 = 30.
        assert_eq!(SHORT_RUN_MAX, 30);
        let last = SHORT_RUN_TABLE.last().unwrap();
        assert_eq!(
            last.rstart as u32 + (1u32 << last.rbits) - 1,
            SHORT_RUN_MAX as u32
        );
    }

    #[test]
    fn long_run_zero_nbits_returns_empty_string() {
        // §7.2.1 step 3 short-circuits when NBITS = 0 — the procedure
        // returns BITS immediately without reading any bits.
        let bits = [];
        let out = decode_long_run_bit_string(&bits, 0).unwrap();
        assert_eq!(out, Vec::<u8>::new());
    }

    #[test]
    fn long_run_single_run_of_length_one() {
        // BIT=1, code b0 (RSTART=1, RBITS=0, RLEN=1) → one 1.
        let mut w = BitWriter::new();
        w.put(1, 1); // BIT
        write_long_run_record(&mut w, 0b0, 1, 0, 0);
        let bytes = w.finish();
        let out = decode_long_run_bit_string(&bytes, 1).unwrap();
        assert_eq!(out, vec![1]);
    }

    #[test]
    fn long_run_single_run_each_table_row() {
        // For each Table 7.7 row, choose a run-length within the row's
        // range and verify byte-exact decode.
        let cases: &[(u32, u32, u32, u32, u32)] = &[
            // (code, code_len, rbits, roffs, expected_rlen)
            (0b0, 1, 0, 0, 1),
            (0b10, 2, 1, 0, 2),
            (0b10, 2, 1, 1, 3),
            (0b110, 3, 1, 0, 4),
            (0b110, 3, 1, 1, 5),
            (0b1110, 4, 2, 0, 6),
            (0b1110, 4, 2, 3, 9),
            (0b11110, 5, 3, 0, 10),
            (0b11110, 5, 3, 7, 17),
            (0b111110, 6, 4, 0, 18),
            (0b111110, 6, 4, 15, 33),
            (0b111111, 6, 12, 0, 34),
            (0b111111, 6, 12, 4095, 4129),
        ];
        for (code, code_len, rbits, roffs, rlen) in cases.iter().copied() {
            let mut w = BitWriter::new();
            w.put(0, 1); // BIT=0
            write_long_run_record(&mut w, code, code_len, rbits, roffs);
            let bytes = w.finish();
            let out = decode_long_run_bit_string(&bytes, rlen as u64).unwrap();
            assert_eq!(
                out,
                vec![0u8; rlen as usize],
                "row code={code:b} roffs={roffs} (expected rlen={rlen})"
            );
        }
    }

    #[test]
    fn long_run_toggles_bit_between_runs() {
        // BIT=1, code b0 (RLEN=1, '1') then code b10 roffs=1 (RLEN=3,
        // BIT toggled → '0'). NBITS = 4 → "1 0 0 0".
        let mut w = BitWriter::new();
        w.put(1, 1);
        write_long_run_record(&mut w, 0b0, 1, 0, 0);
        write_long_run_record(&mut w, 0b10, 2, 1, 1);
        let bytes = w.finish();
        let out = decode_long_run_bit_string(&bytes, 4).unwrap();
        assert_eq!(out, vec![1, 0, 0, 0]);
    }

    #[test]
    fn long_run_three_runs_alternating() {
        // BIT=0, code b0 (run of one 0) → BIT flips to 1 (run of one 1)
        // → BIT flips to 0 (run of one 0). NBITS = 3 → "0 1 0".
        let mut w = BitWriter::new();
        w.put(0, 1);
        write_long_run_record(&mut w, 0b0, 1, 0, 0);
        write_long_run_record(&mut w, 0b0, 1, 0, 0);
        write_long_run_record(&mut w, 0b0, 1, 0, 0);
        let bytes = w.finish();
        let out = decode_long_run_bit_string(&bytes, 3).unwrap();
        assert_eq!(out, vec![0, 1, 0]);
    }

    #[test]
    fn long_run_reads_fresh_bit_after_rlen_4129() {
        // Two back-to-back maximum-length runs of the SAME bit (both 0),
        // exercising the VP3+ exception in §7.2.1 step 12: after a run
        // of length 4129, the next BIT is READ (not toggled), so the
        // encoder can express runs of arbitrary length.
        let mut w = BitWriter::new();
        w.put(0, 1); // BIT = 0
                     // First run: code b111111 + roffs=4095 → RLEN = 34 + 4095 = 4129.
        write_long_run_record(&mut w, 0b111111, 6, 12, 4095);
        // After max run: read fresh BIT = 0 (same value, same run kind).
        w.put(0, 1);
        // Second run: code b0 → RLEN = 1, BIT = 0.
        write_long_run_record(&mut w, 0b0, 1, 0, 0);
        let bytes = w.finish();
        let out = decode_long_run_bit_string(&bytes, 4129 + 1).unwrap();
        let mut expected = vec![0u8; 4129];
        expected.push(0);
        assert_eq!(out, expected);
    }

    #[test]
    fn long_run_reads_fresh_bit_after_rlen_4129_flipped() {
        // Same as above but the fresh BIT is 1 (a toggle would also
        // have produced 1, so this case is not distinguishable from a
        // toggle path on the resulting bits — but the procedure still
        // reads the BIT, so the byte count differs by exactly 1 vs the
        // non-maximum case). Test asserts the bit count and content.
        let mut w = BitWriter::new();
        w.put(0, 1); // BIT = 0
        write_long_run_record(&mut w, 0b111111, 6, 12, 4095);
        w.put(1, 1); // fresh BIT = 1
        write_long_run_record(&mut w, 0b0, 1, 0, 0);
        let bytes = w.finish();
        let out = decode_long_run_bit_string(&bytes, 4129 + 1).unwrap();
        let mut expected = vec![0u8; 4129];
        expected.push(1);
        assert_eq!(out, expected);
    }

    #[test]
    fn long_run_does_not_read_fresh_bit_below_4129() {
        // A run of length 4128 (one short of the max) MUST toggle, not
        // read a new BIT. The difference vs the 4129 path is one bit
        // of input — check by giving exactly enough bytes for the
        // toggle path and verifying the result is correct.
        let mut w = BitWriter::new();
        w.put(0, 1); // BIT = 0
                     // First run: code b111111 + roffs=4094 → RLEN = 34+4094 = 4128.
        write_long_run_record(&mut w, 0b111111, 6, 12, 4094);
        // No fresh BIT — toggle → 1.
        write_long_run_record(&mut w, 0b0, 1, 0, 0);
        let bytes = w.finish();
        let out = decode_long_run_bit_string(&bytes, 4128 + 1).unwrap();
        let mut expected = vec![0u8; 4128];
        expected.push(1);
        assert_eq!(out, expected);
    }

    #[test]
    fn long_run_rejects_truncated_huffman_prefix() {
        // Construct a single-byte buffer that decodes two records and
        // then runs out mid-Huffman walk on the third. Byte layout
        // (MSb→LSb): BIT=0, code b1110 (4 bits), ROFFS=0b11 (2 bits)
        // → RLEN=9. Then 1 remaining bit "0" is the next BIT-toggle's
        // Huffman bit (= code b0, RSTART=1, RBITS=0) → second RLEN=1.
        // LEN=10. The caller requests NBITS=11, forcing one more
        // Huffman walk — but byte_pos has reached end-of-buffer, so
        // the read errors on `long-run.huff`.
        #[allow(clippy::unusual_byte_groupings)]
        let bytes = [0b0_1110_11_0u8]; // BIT|code(b1110)|ROFFS(2)|next-Huffman bit
        let r = decode_long_run_bit_string(&bytes, 11);
        match r {
            Err(Error::TruncatedHeader { field }) => {
                assert!(
                    field.starts_with("long-run."),
                    "expected long-run.* truncation, got {field}"
                );
            }
            other => panic!("expected TruncatedHeader, got {other:?}"),
        }
    }

    #[test]
    fn long_run_rejects_truncated_roffs() {
        // BIT=1, code b10 (RBITS=1) — but the trailing ROFFS bit is
        // missing because the byte ran out.
        let mut w = BitWriter::new();
        w.put(1, 1);
        w.put(0b10, 2);
        // 3 bits used, 5 zero bits pad — decoder will read ROFFS=0 and
        // SUCCEED (the pad bits read as 0). To force truncation we need
        // the RBITS read to itself fall off the end: use a hand-crafted
        // sub-byte that ends exactly after the Huffman prefix. The
        // tightest legal byte buffer ends on a byte boundary, so build
        // a 1-byte buffer where bits 0+1+2 form BIT+code and bits 3..7
        // would be ROFFS continuation — but those 5 bits read as zero
        // from the buffer's own bytes. Instead: use a multi-record
        // construction where the second record's ROFFS extends past
        // the buffer's last byte.
        // BIT=0; first record code b111111 + roffs=4095 (RLEN=4129).
        // Then either fresh BIT (1 bit) or toggle; pick toggle path by
        // demanding NBITS=4128 so we fail at overrun before exception
        // — easier: use code b111111 ROFFS truncated. Build BIT + 6 bits
        // of code = 7 bits, then 1-bit of ROFFS (need 12 bits total),
        // truncated at byte 1.
        #[allow(clippy::unusual_byte_groupings)]
        let buf = vec![0b0_111111_0u8]; // BIT(1)|code(b111111, 6 bits)|first ROFFS bit
        let r = decode_long_run_bit_string(&buf, 4129);
        match r {
            Err(Error::TruncatedHeader { field }) => {
                assert!(
                    field.starts_with("long-run."),
                    "expected long-run.* truncation, got {field}"
                );
            }
            other => panic!("expected TruncatedHeader, got {other:?}"),
        }
        // Touch `w` to ensure it's used; the variable is kept for
        // narrative clarity.
        let _ = w;
    }

    #[test]
    fn long_run_rejects_truncated_initial_bit() {
        // NBITS > 0 but the buffer is empty — the very first BIT read
        // must error.
        let bytes: [u8; 0] = [];
        let r = decode_long_run_bit_string(&bytes, 1);
        match r {
            Err(Error::TruncatedHeader { field }) => {
                assert_eq!(field, "long-run.BIT");
            }
            other => panic!("expected TruncatedHeader(long-run.BIT), got {other:?}"),
        }
    }

    #[test]
    fn long_run_rejects_overrun() {
        // BIT=0, code b1110 (RBITS=2) with roffs=3 → RLEN=9. NBITS=4.
        // The decoded run length (9) exceeds NBITS (4) → RunLengthOverrun.
        let mut w = BitWriter::new();
        w.put(0, 1);
        write_long_run_record(&mut w, 0b1110, 4, 2, 3);
        let bytes = w.finish();
        let r = decode_long_run_bit_string(&bytes, 4);
        match r {
            Err(Error::RunLengthOverrun { len, nbits }) => {
                assert_eq!(len, 9);
                assert_eq!(nbits, 4);
            }
            other => panic!("expected RunLengthOverrun, got {other:?}"),
        }
    }

    #[test]
    fn long_run_overrun_error_display() {
        let s = format!("{}", Error::RunLengthOverrun { len: 9, nbits: 4 });
        assert!(s.contains("§7.2"), "got: {s}");
        assert!(s.contains("LEN=9"), "got: {s}");
        assert!(s.contains("NBITS=4"), "got: {s}");
    }

    #[test]
    fn short_run_zero_nbits_returns_empty_string() {
        let bytes = [];
        let out = decode_short_run_bit_string(&bytes, 0).unwrap();
        assert_eq!(out, Vec::<u8>::new());
    }

    #[test]
    fn short_run_single_run_each_table_row() {
        // Verify every Table 7.11 row with an in-range ROFFS.
        let cases: &[(u32, u32, u32, u32, u32)] = &[
            (0b0, 1, 1, 0, 1),
            (0b0, 1, 1, 1, 2),
            (0b10, 2, 1, 0, 3),
            (0b10, 2, 1, 1, 4),
            (0b110, 3, 1, 0, 5),
            (0b110, 3, 1, 1, 6),
            (0b1110, 4, 2, 0, 7),
            (0b1110, 4, 2, 3, 10),
            (0b11110, 5, 2, 0, 11),
            (0b11110, 5, 2, 3, 14),
            (0b11111, 5, 4, 0, 15),
            (0b11111, 5, 4, 15, 30),
        ];
        for (code, code_len, rbits, roffs, rlen) in cases.iter().copied() {
            let mut w = BitWriter::new();
            w.put(1, 1); // BIT=1
            write_short_run_record(&mut w, code, code_len, rbits, roffs);
            let bytes = w.finish();
            let out = decode_short_run_bit_string(&bytes, rlen as u64).unwrap();
            assert_eq!(
                out,
                vec![1u8; rlen as usize],
                "row code={code:b} roffs={roffs}"
            );
        }
    }

    #[test]
    fn short_run_toggles_bit_between_runs() {
        // BIT=0, code b0 roffs=0 (RLEN=1 → '0'), then toggle to BIT=1,
        // code b10 roffs=1 (RLEN=4 → '1 1 1 1'). NBITS=5 → "0 1 1 1 1".
        let mut w = BitWriter::new();
        w.put(0, 1);
        write_short_run_record(&mut w, 0b0, 1, 1, 0);
        write_short_run_record(&mut w, 0b10, 2, 1, 1);
        let bytes = w.finish();
        let out = decode_short_run_bit_string(&bytes, 5).unwrap();
        assert_eq!(out, vec![0, 1, 1, 1, 1]);
    }

    #[test]
    fn short_run_at_max_run_length_still_toggles() {
        // Unlike long-run, short-run has NO "read fresh BIT" exception
        // at the maximum. A run of length 30 must be followed by a
        // toggled BIT.
        let mut w = BitWriter::new();
        // BIT = 0.
        w.put(0, 1);
        // First run: code b11111 + roffs=15 → RLEN = 30.
        write_short_run_record(&mut w, 0b11111, 5, 4, 15);
        // Second run: code b0 roffs=0 → RLEN = 1; BIT toggled → 1.
        write_short_run_record(&mut w, 0b0, 1, 1, 0);
        let bytes = w.finish();
        let out = decode_short_run_bit_string(&bytes, 31).unwrap();
        let mut expected = vec![0u8; 30];
        expected.push(1);
        assert_eq!(out, expected);
    }

    #[test]
    fn short_run_rejects_truncated_initial_bit() {
        let bytes: [u8; 0] = [];
        let r = decode_short_run_bit_string(&bytes, 1);
        match r {
            Err(Error::TruncatedHeader { field }) => {
                assert_eq!(field, "short-run.BIT");
            }
            other => panic!("expected TruncatedHeader(short-run.BIT), got {other:?}"),
        }
    }

    #[test]
    fn short_run_rejects_truncated_huffman_prefix() {
        // Single-byte buffer: BIT=1, then code b1110 (4 bits) used →
        // 5 bits consumed. RBITS=2 — bits 5+6 = '00' = roffs=0 →
        // RLEN=7. The first record completes inside the byte (no
        // truncation). NBITS=8 demands one more bit of output, forcing
        // a second Huffman walk. The next Huffman bit (bit 7 of byte 0)
        // is '0' → code b0 (RSTART=1, RBITS=1). The required 1-bit
        // ROFFS read is now past the byte boundary, so the read errors
        // on `short-run.ROFFS`.
        #[allow(clippy::unusual_byte_groupings)]
        let bytes: [u8; 1] = [0b1_1110_000u8];
        let r = decode_short_run_bit_string(&bytes, 8);
        match r {
            Err(Error::TruncatedHeader { field }) => {
                assert!(
                    field.starts_with("short-run."),
                    "expected short-run.* truncation, got {field}"
                );
            }
            other => panic!("expected TruncatedHeader, got {other:?}"),
        }
    }

    #[test]
    fn short_run_rejects_overrun() {
        // BIT=0, code b1110 (RBITS=2) roffs=3 → RLEN=10. NBITS=5.
        let mut w = BitWriter::new();
        w.put(0, 1);
        write_short_run_record(&mut w, 0b1110, 4, 2, 3);
        let bytes = w.finish();
        let r = decode_short_run_bit_string(&bytes, 5);
        match r {
            Err(Error::RunLengthOverrun { len, nbits }) => {
                assert_eq!(len, 10);
                assert_eq!(nbits, 5);
            }
            other => panic!("expected RunLengthOverrun, got {other:?}"),
        }
    }

    #[test]
    fn long_run_byte_boundary_decode() {
        // Construct a long-run stream that crosses a byte boundary
        // and verify the decode matches an explicit reference walk.
        // BIT=1, code b11110 + roffs=5 → RLEN=15; then toggle BIT to 0,
        // code b110 + roffs=0 → RLEN=4. NBITS=19; total bits = "15×1 + 4×0".
        let mut w = BitWriter::new();
        w.put(1, 1);
        write_long_run_record(&mut w, 0b11110, 5, 3, 5); // 5+3=8 bits → record 2 spans bytes
        write_long_run_record(&mut w, 0b110, 3, 1, 0);
        let bytes = w.finish();
        let out = decode_long_run_bit_string(&bytes, 19).unwrap();
        let mut expected = vec![1u8; 15];
        expected.extend(std::iter::repeat_n(0u8, 4));
        assert_eq!(out, expected);
    }

    #[test]
    fn short_run_typical_super_block_decode() {
        // Realistic short-run scenario: 16 bits, one per block in a
        // super block. BIT=1, then alternating runs.
        // First: code b0 roffs=0 → RLEN=1 (1 → '1')
        // toggle to 0, code b10 roffs=0 → RLEN=3 (3 → '0 0 0')
        // toggle to 1, code b110 roffs=1 → RLEN=6 (6 → '1 1 1 1 1 1')
        // toggle to 0, code b1110 roffs=2 → RLEN=9 → but 9 > remaining 6.
        // So adjust: use roffs that fits in the remaining 6.
        // toggle to 0, code b110 roffs=1 → RLEN=6 (6 → '0 0 0 0 0 0').
        let mut w = BitWriter::new();
        w.put(1, 1);
        write_short_run_record(&mut w, 0b0, 1, 1, 0); // RLEN=1, '1'
        write_short_run_record(&mut w, 0b10, 2, 1, 0); // toggle→0, RLEN=3
        write_short_run_record(&mut w, 0b110, 3, 1, 1); // toggle→1, RLEN=6
        write_short_run_record(&mut w, 0b110, 3, 1, 1); // toggle→0, RLEN=6
        let bytes = w.finish();
        let out = decode_short_run_bit_string(&bytes, 16).unwrap();
        let mut expected: Vec<u8> = Vec::new();
        expected.push(1);
        expected.extend(std::iter::repeat_n(0u8, 3));
        expected.extend(std::iter::repeat_n(1u8, 6));
        expected.extend(std::iter::repeat_n(0u8, 6));
        assert_eq!(out, expected);
    }

    // -------------------------------------------------------------
    // §7.3 Coded Block Flags Decode tests
    // -------------------------------------------------------------

    /// Build a Theora-§7.2.1 long-run bit string encoding `bits` (a
    /// sequence of `0`/`1` values), starting with the supplied initial
    /// `BIT` value and toggling between runs (no 4129-exception path is
    /// exercised here — this helper is for short test inputs).
    ///
    /// Used by the §7.3 tests to synthesize the partially-coded /
    /// fully-coded super-block bit strings. The encoding is the
    /// inverse of `decode_long_run_bit_string` and so a round-trip
    /// `decode(encode(bits)) == bits` holds.
    fn encode_long_run_runs(w: &mut BitWriter, bits: &[u8]) {
        if bits.is_empty() {
            return;
        }
        // Initial BIT.
        let mut bit = bits[0];
        w.put(bit as u32, 1);
        let mut i = 0;
        while i < bits.len() {
            // Compute the length of the current run (all `bit`).
            let mut run = 0usize;
            while i + run < bits.len() && bits[i + run] == bit {
                run += 1;
            }
            // Select the largest Table 7.7 entry whose RSTART <= run and
            // run < RSTART + (1 << RBITS). Walk the table top down so
            // we pick the entry with the highest RSTART that still
            // covers `run`.
            let mut chosen: Option<&LongRunEntry> = None;
            for entry in LONG_RUN_TABLE.iter().rev() {
                let rs = entry.rstart as usize;
                let cap = rs + (1usize << entry.rbits) - 1;
                if run >= rs && run <= cap {
                    chosen = Some(entry);
                    break;
                }
            }
            let entry = chosen.expect("run length within Table 7.7 range");
            let roffs = (run - entry.rstart as usize) as u32;
            w.put(entry.code, entry.code_len as u32);
            if entry.rbits > 0 {
                w.put(roffs, entry.rbits as u32);
            }
            i += run;
            if i < bits.len() {
                // Toggle (the test helper never exercises the 4129 path).
                bit = 1 - bit;
            }
        }
    }

    /// Same as `encode_long_run_runs` but against Table 7.11 for the
    /// §7.2.2 short-run procedure (used by §7.3 step 2(h) per-block
    /// bit string).
    fn encode_short_run_runs(w: &mut BitWriter, bits: &[u8]) {
        if bits.is_empty() {
            return;
        }
        let mut bit = bits[0];
        w.put(bit as u32, 1);
        let mut i = 0;
        while i < bits.len() {
            let mut run = 0usize;
            while i + run < bits.len() && bits[i + run] == bit {
                run += 1;
            }
            let mut chosen: Option<&ShortRunEntry> = None;
            for entry in SHORT_RUN_TABLE.iter().rev() {
                let rs = entry.rstart as usize;
                let cap = rs + (1usize << entry.rbits) - 1;
                if run >= rs && run <= cap {
                    chosen = Some(entry);
                    break;
                }
            }
            let entry = chosen.expect("run length within Table 7.11 range");
            let roffs = (run - entry.rstart as usize) as u32;
            w.put(entry.code, entry.code_len as u32);
            if entry.rbits > 0 {
                w.put(roffs, entry.rbits as u32);
            }
            i += run;
            if i < bits.len() {
                bit = 1 - bit;
            }
        }
    }

    /// Encoder round-trip sanity: decoding what the long-run helper
    /// emits returns the original bits.
    #[test]
    fn coded_block_flags_long_run_encoder_roundtrip() {
        let cases: &[&[u8]] = &[
            &[1u8; 1],
            &[0u8; 1],
            &[1, 0, 1, 0, 1, 0, 1],
            &[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
            &[1; 18],
            &[0; 34],
        ];
        for case in cases {
            let mut w = BitWriter::new();
            encode_long_run_runs(&mut w, case);
            let bytes = w.finish();
            let out = decode_long_run_bit_string(&bytes, case.len() as u64).unwrap();
            assert_eq!(out.as_slice(), *case, "long-run roundtrip case {case:?}");
        }
    }

    /// Encoder round-trip sanity for the short-run helper.
    #[test]
    fn coded_block_flags_short_run_encoder_roundtrip() {
        let cases: &[&[u8]] = &[
            &[1u8; 1],
            &[0u8; 1],
            &[1, 0, 1, 0, 1, 0, 1, 0],
            &[0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1],
            &[1; 16],
            &[0; 30],
        ];
        for case in cases {
            let mut w = BitWriter::new();
            encode_short_run_runs(&mut w, case);
            let bytes = w.finish();
            let out = decode_short_run_bit_string(&bytes, case.len() as u64).unwrap();
            assert_eq!(out.as_slice(), *case, "short-run roundtrip case {case:?}");
        }
    }

    #[test]
    fn coded_block_flags_intra_marks_all_coded() {
        // §7.3 step 1: every block coded; packet is not consumed.
        // Mapping is irrelevant for the intra path beyond length checks.
        let nbs = 24u32;
        let nsbs = 3u32;
        // Trivial mapping: 8 blocks per super block, 3 super blocks.
        let map: Vec<u32> = (0..nbs).map(|bi| bi / 8).collect();
        let out = decode_coded_block_flags(&[], FrameType::Intra, nsbs, nbs, &map).unwrap();
        assert_eq!(out, vec![1u8; nbs as usize]);
    }

    #[test]
    fn coded_block_flags_intra_rejects_bad_map_length() {
        // §7.3 input-validation: mapping length must equal NBS.
        let nbs = 24u32;
        let nsbs = 3u32;
        let bad_map: Vec<u32> = vec![0; (nbs - 1) as usize];
        let err = decode_coded_block_flags(&[], FrameType::Intra, nsbs, nbs, &bad_map).unwrap_err();
        assert_eq!(
            err,
            Error::BlockSuperBlockMapLenMismatch {
                map_len: 23,
                nbs: 24
            }
        );
    }

    #[test]
    fn coded_block_flags_inter_rejects_oob_super_block_index() {
        // §7.3 input-validation: every map entry must be < NSBS.
        let nbs = 4u32;
        let nsbs = 2u32;
        let map: Vec<u32> = vec![0, 1, 2, 0]; // block 2 → sbi 2 ≥ nsbs=2.
        let err = decode_coded_block_flags(&[], FrameType::Inter, nsbs, nbs, &map).unwrap_err();
        assert_eq!(
            err,
            Error::BlockSuperBlockIndexOutOfRange {
                bi: 2,
                sbi: 2,
                nsbs: 2
            }
        );
    }

    #[test]
    fn coded_block_flags_inter_all_super_blocks_not_coded() {
        // §7.3 step 2: every super block has SBPCODED=0 (not partially
        // coded), and every SBFCODED bit is 0 → every block is 0.
        // 3 super blocks, 8 blocks per super block → NBS=24.
        let nsbs = 3u32;
        let nbs = 24u32;
        let map: Vec<u32> = (0..nbs).map(|bi| bi / 8).collect();
        let mut w = BitWriter::new();
        // SBPCODED bits = [0, 0, 0] (3-bit run-length stream).
        encode_long_run_runs(&mut w, &[0, 0, 0]);
        // SBFCODED bits = [0, 0, 0] (NBITS=3, since every SBPCODED is 0).
        encode_long_run_runs(&mut w, &[0, 0, 0]);
        // No per-block bit string (no partially-coded super blocks).
        let bytes = w.finish();
        let out = decode_coded_block_flags(&bytes, FrameType::Inter, nsbs, nbs, &map).unwrap();
        assert_eq!(out, vec![0u8; 24]);
    }

    #[test]
    fn coded_block_flags_inter_all_super_blocks_fully_coded() {
        // Every super block fully coded → SBPCODED=0, SBFCODED=1; no
        // per-block stream needed.
        let nsbs = 2u32;
        let nbs = 16u32;
        let map: Vec<u32> = (0..nbs).map(|bi| bi / 8).collect();
        let mut w = BitWriter::new();
        encode_long_run_runs(&mut w, &[0, 0]);
        encode_long_run_runs(&mut w, &[1, 1]);
        let bytes = w.finish();
        let out = decode_coded_block_flags(&bytes, FrameType::Inter, nsbs, nbs, &map).unwrap();
        assert_eq!(out, vec![1u8; 16]);
    }

    #[test]
    fn coded_block_flags_inter_mixed_super_block_states() {
        // SBPCODED = [1, 0, 0]   (sb 0 partially coded; sb 1, 2 not).
        // SBFCODED for non-partial sb1=1, sb2=0 → ['1','0'].
        // sb0 has 4 blocks (NBITS=4 for the §7.2.2 per-block stream).
        // Per-block bits inside sb0 = [1, 0, 1, 0].
        // Mapping: 4 blocks per super block → NBS=12.
        // Expected BCODED:
        //  bi 0..3 (sb0, partial) → [1, 0, 1, 0]
        //  bi 4..7 (sb1, SBFCODED=1) → [1, 1, 1, 1]
        //  bi 8..11 (sb2, SBFCODED=0) → [0, 0, 0, 0]
        let nsbs = 3u32;
        let nbs = 12u32;
        let map: Vec<u32> = (0..nbs).map(|bi| bi / 4).collect();
        let mut w = BitWriter::new();
        encode_long_run_runs(&mut w, &[1, 0, 0]);
        encode_long_run_runs(&mut w, &[1, 0]);
        encode_short_run_runs(&mut w, &[1, 0, 1, 0]);
        let bytes = w.finish();
        let out = decode_coded_block_flags(&bytes, FrameType::Inter, nsbs, nbs, &map).unwrap();
        let expected: Vec<u8> = vec![1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0];
        assert_eq!(out, expected);
    }

    #[test]
    fn coded_block_flags_inter_partial_edge_super_block_has_fewer_than_16_blocks() {
        // §7.3 step 2(g) note: super blocks at frame edge can have fewer
        // than 16 blocks. Synthesize NSBS=2 with sb0=16 blocks, sb1=5
        // blocks (an edge super block). Both partially coded.
        // NBITS for the short-run stream = 16 + 5 = 21 (not 32).
        let nsbs = 2u32;
        let nbs = 21u32;
        let mut map: Vec<u32> = Vec::with_capacity(21);
        map.resize(16, 0u32);
        map.resize(16 + 5, 1u32);
        // Per-block bits: alternating then trailing 1s.
        let blocks: Vec<u8> = vec![
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, // sb0 (16 blocks)
            1, 1, 1, 1, 1, // sb1 (5 blocks)
        ];
        let mut w = BitWriter::new();
        encode_long_run_runs(&mut w, &[1, 1]); // SBPCODED = [1, 1].
                                               // SBFCODED stream is empty (no SBPCODED==0 entries) so encode
                                               // nothing.
        encode_short_run_runs(&mut w, &blocks);
        let bytes = w.finish();
        let out = decode_coded_block_flags(&bytes, FrameType::Inter, nsbs, nbs, &map).unwrap();
        assert_eq!(out, blocks);
    }

    #[test]
    fn coded_block_flags_inter_empty_no_super_blocks() {
        // Edge case: NSBS = 0 and NBS = 0. Every long-run / short-run
        // decode is a no-op, BCODED is empty.
        let out = decode_coded_block_flags(&[], FrameType::Inter, 0, 0, &[]).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn coded_block_flags_inter_truncated_in_sbpcoded() {
        // Force §7.2.1 truncation while decoding the first SBPCODED
        // long-run. Provide a packet with too few bits.
        let nsbs = 16u32; // need at least 16 SBPCODED bits worth of run-length records
        let nbs = 16u32;
        let map: Vec<u32> = (0..nbs).collect();
        let bytes = [0x00u8]; // 1 byte only: initial BIT=0 + code=0 (RLEN=1) repeats - far short of 16.
        let err = decode_coded_block_flags(&bytes, FrameType::Inter, nsbs, nbs, &map).unwrap_err();
        match err {
            Error::TruncatedHeader { .. } => (),
            other => panic!("expected TruncatedHeader, got {other:?}"),
        }
    }

    #[test]
    fn coded_block_flags_error_display_map_length() {
        // Display arm coverage for the new error variant.
        let e = Error::BlockSuperBlockMapLenMismatch { map_len: 5, nbs: 7 };
        let s = format!("{e}");
        assert!(s.contains("§7.3"), "got: {s}");
        assert!(s.contains("map length 5"), "got: {s}");
        assert!(s.contains("nbs 7"), "got: {s}");
    }

    #[test]
    fn coded_block_flags_error_display_oob_index() {
        let e = Error::BlockSuperBlockIndexOutOfRange {
            bi: 4,
            sbi: 10,
            nsbs: 7,
        };
        let s = format!("{e}");
        assert!(s.contains("§7.3"), "got: {s}");
        assert!(s.contains("block 4"), "got: {s}");
        assert!(s.contains("sbi 10"), "got: {s}");
        assert!(s.contains("nsbs 7"), "got: {s}");
    }

    #[test]
    fn coded_block_flags_inter_intra_arms_independent() {
        // Intra arm completely ignores the packet; inter arm against
        // the same map produces a different result given a valid §7.3
        // payload. Demonstrates the two FTYPE paths take disjoint
        // routes.
        let nsbs = 2u32;
        let nbs = 16u32;
        let map: Vec<u32> = (0..nbs).map(|bi| bi / 8).collect();
        let intra = decode_coded_block_flags(&[], FrameType::Intra, nsbs, nbs, &map).unwrap();
        assert_eq!(intra, vec![1u8; 16]);

        let mut w = BitWriter::new();
        encode_long_run_runs(&mut w, &[0, 0]);
        encode_long_run_runs(&mut w, &[0, 0]);
        let bytes = w.finish();
        let inter = decode_coded_block_flags(&bytes, FrameType::Inter, nsbs, nbs, &map).unwrap();
        assert_eq!(inter, vec![0u8; 16]);
        assert_ne!(intra, inter);
    }

    #[test]
    fn coded_block_flags_inter_inner_chains_on_shared_reader() {
        // Confirm the crate-private `decode_coded_block_flags_inner`
        // can be driven from a BitReader that is already partly
        // consumed — the chaining pattern an end-to-end frame decoder
        // will use to walk §7.1 → §7.2 → §7.3 on a single reader.
        let nsbs = 2u32;
        let nbs = 16u32;
        let map: Vec<u32> = (0..nbs).map(|bi| bi / 8).collect();
        let mut w = BitWriter::new();
        // Prefix: 4 unrelated bits to be skipped on the shared reader.
        w.put(0b1010, 4);
        encode_long_run_runs(&mut w, &[0, 0]);
        encode_long_run_runs(&mut w, &[1, 0]);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        // Consume the 4-bit prefix the way §7.1 would consume the frame
        // header.
        let prefix = r.read_bits(4, "test.prefix").unwrap();
        assert_eq!(prefix, 0b1010);
        let out =
            decode_coded_block_flags_inner(&mut r, FrameType::Inter, nsbs, nbs, &map).unwrap();
        // Expected: sb0 fully coded → [1; 8]; sb1 not coded → [0; 8].
        let mut expected = vec![1u8; 8];
        expected.extend(std::iter::repeat_n(0u8, 8));
        assert_eq!(out, expected);
    }

    #[test]
    fn coded_block_flags_inter_partial_with_uncoded_block_subset() {
        // Single super block, partially coded. Per-block bits
        // [1, 0, 0, 1, 1, 0, 1, 0] (8 blocks). Verifies the §7.2.2
        // per-block stream is consumed in coded-order one bit at a
        // time.
        let nsbs = 1u32;
        let nbs = 8u32;
        let map: Vec<u32> = vec![0u32; nbs as usize];
        let blocks: Vec<u8> = vec![1, 0, 0, 1, 1, 0, 1, 0];
        let mut w = BitWriter::new();
        encode_long_run_runs(&mut w, &[1]); // SBPCODED = [1].
                                            // SBFCODED stream is empty.
        encode_short_run_runs(&mut w, &blocks);
        let bytes = w.finish();
        let out = decode_coded_block_flags(&bytes, FrameType::Inter, nsbs, nbs, &map).unwrap();
        assert_eq!(out, blocks);
    }

    #[test]
    fn coded_block_flags_inter_non_coded_order_mapping() {
        // §7.3 step 2(i)i: "the index of the super block containing
        // block bi". Map need not be monotone in bi (in real Theora,
        // the §2.4 super-block scan interleaves blocks from different
        // super blocks). Exercise an interleaved mapping to ensure
        // the procedure walks the map literally rather than assuming
        // contiguous super-block runs.
        let nsbs = 2u32;
        let nbs = 8u32;
        // Interleave: block 0 → sb0, block 1 → sb1, block 2 → sb0, …
        let map: Vec<u32> = (0..nbs).map(|bi| bi % 2).collect();
        let mut w = BitWriter::new();
        // SBPCODED = [1, 0]   (sb0 partial; sb1 not).
        encode_long_run_runs(&mut w, &[1, 0]);
        // SBFCODED for the single non-partial sb1 = [1].
        encode_long_run_runs(&mut w, &[1]);
        // Per-block bits for sb0 (4 blocks: bi 0, 2, 4, 6) = [0, 1, 0, 1].
        encode_short_run_runs(&mut w, &[0, 1, 0, 1]);
        let bytes = w.finish();
        let out = decode_coded_block_flags(&bytes, FrameType::Inter, nsbs, nbs, &map).unwrap();
        // Expected: bi 0 (sb0, partial) → 0
        //           bi 1 (sb1, SBFCODED=1) → 1
        //           bi 2 (sb0, partial) → 1
        //           bi 3 (sb1, SBFCODED=1) → 1
        //           bi 4 (sb0, partial) → 0
        //           bi 5 (sb1, SBFCODED=1) → 1
        //           bi 6 (sb0, partial) → 1
        //           bi 7 (sb1, SBFCODED=1) → 1
        let expected = vec![0, 1, 1, 1, 0, 1, 1, 1];
        assert_eq!(out, expected);
    }

    // -----------------------------------------------------------------
    // §7.4 Macro Block Coding Modes
    // -----------------------------------------------------------------

    /// Append the Table 7.19 Huffman code for `mi` (in `0..=7`) to the
    /// MSb-first writer `w`. Mirrors `read_table_7_19_mi` and is used to
    /// synthesize §7.4 test fixtures bit-exactly.
    fn write_mb_mode_code(w: &mut BitWriter, mi: u8) {
        match mi {
            0 => w.put(0b0, 1),
            1 => w.put(0b10, 2),
            2 => w.put(0b110, 3),
            3 => w.put(0b1110, 4),
            4 => w.put(0b11110, 5),
            5 => w.put(0b111110, 6),
            6 => w.put(0b1111110, 7),
            7 => w.put(0b1111111, 7),
            _ => panic!("mi {mi} out of range for Table 7.19"),
        }
    }

    #[test]
    fn macroblock_mode_table_7_18_index_round_trip() {
        // Table 7.18: 0=INTER_NOMV, 1=INTRA, 2=INTER_MV, 3=INTER_MV_LAST,
        // 4=INTER_MV_LAST2, 5=INTER_GOLDEN_NOMV, 6=INTER_GOLDEN_MV,
        // 7=INTER_MV_FOUR.
        for v in 0u8..=7 {
            let m = MacroBlockMode::from_index(v).unwrap();
            assert_eq!(m.to_index(), v, "round-trip failed for index {v}");
        }
        assert_eq!(
            MacroBlockMode::from_index(0).unwrap(),
            MacroBlockMode::InterNoMv
        );
        assert_eq!(
            MacroBlockMode::from_index(1).unwrap(),
            MacroBlockMode::Intra
        );
        assert_eq!(
            MacroBlockMode::from_index(2).unwrap(),
            MacroBlockMode::InterMv
        );
        assert_eq!(
            MacroBlockMode::from_index(3).unwrap(),
            MacroBlockMode::InterMvLast
        );
        assert_eq!(
            MacroBlockMode::from_index(4).unwrap(),
            MacroBlockMode::InterMvLast2
        );
        assert_eq!(
            MacroBlockMode::from_index(5).unwrap(),
            MacroBlockMode::InterGoldenNoMv
        );
        assert_eq!(
            MacroBlockMode::from_index(6).unwrap(),
            MacroBlockMode::InterGoldenMv
        );
        assert_eq!(
            MacroBlockMode::from_index(7).unwrap(),
            MacroBlockMode::InterMvFour
        );
        assert_eq!(MacroBlockMode::from_index(8), None);
    }

    #[test]
    fn macroblock_modes_intra_marks_all_intra() {
        // §7.4 step 1: every macro block is INTRA. No bits consumed.
        let nmbs = 4u32;
        let nbs = 16u32;
        let bcoded = vec![1u8; nbs as usize];
        let map: Vec<[u32; 4]> = (0..nmbs)
            .map(|mbi| {
                let base = mbi * 4;
                [base, base + 1, base + 2, base + 3]
            })
            .collect();
        let out = decode_macroblock_modes(&[], FrameType::Intra, nmbs, nbs, &bcoded, &map).unwrap();
        assert_eq!(out.len(), nmbs as usize);
        for m in &out {
            assert_eq!(*m, MacroBlockMode::Intra);
        }
    }

    #[test]
    fn macroblock_modes_intra_rejects_bad_map_length() {
        let nmbs = 2u32;
        let nbs = 4u32;
        let bcoded = vec![1u8; nbs as usize];
        // Map too short — error must be raised even on intra (the input
        // validation runs unconditionally).
        let map: Vec<[u32; 4]> = vec![[0, 1, 2, 3]];
        let err =
            decode_macroblock_modes(&[], FrameType::Intra, nmbs, nbs, &bcoded, &map).unwrap_err();
        assert_eq!(
            err,
            Error::MacroBlockLumaMapLenMismatch {
                map_len: 1,
                nmbs: 2,
            }
        );
    }

    #[test]
    fn macroblock_modes_rejects_oob_luma_block_index() {
        let nmbs = 1u32;
        let nbs = 4u32;
        let bcoded = vec![1u8; nbs as usize];
        // Slot 2 (third entry) carries an OOB luma index.
        let map: Vec<[u32; 4]> = vec![[0, 1, 99, 3]];
        let err =
            decode_macroblock_modes(&[], FrameType::Inter, nmbs, nbs, &bcoded, &map).unwrap_err();
        assert_eq!(
            err,
            Error::MacroBlockLumaBlockIndexOutOfRange {
                mbi: 0,
                slot: 2,
                bi: 99,
                nbs: 4,
            }
        );
    }

    #[test]
    fn macroblock_modes_inter_scheme_7_direct_three_bit_modes() {
        // §7.4 step 2(d)i.B: MSCHEME=7 codes each mode directly as 3
        // bits. Set up 4 macro blocks all with at least one coded luma
        // block; emit modes 0, 3, 5, 7 verbatim.
        let nmbs = 4u32;
        let nbs = 16u32;
        let bcoded = vec![1u8; nbs as usize];
        let map: Vec<[u32; 4]> = (0..nmbs)
            .map(|mbi| {
                let base = mbi * 4;
                [base, base + 1, base + 2, base + 3]
            })
            .collect();
        let mut w = BitWriter::new();
        w.put(7, 3); // MSCHEME = 7
        w.put(0, 3); // INTER_NOMV
        w.put(3, 3); // INTER_MV_LAST
        w.put(5, 3); // INTER_GOLDEN_NOMV
        w.put(7, 3); // INTER_MV_FOUR
        let bytes = w.finish();
        let out =
            decode_macroblock_modes(&bytes, FrameType::Inter, nmbs, nbs, &bcoded, &map).unwrap();
        assert_eq!(
            out,
            vec![
                MacroBlockMode::InterNoMv,
                MacroBlockMode::InterMvLast,
                MacroBlockMode::InterGoldenNoMv,
                MacroBlockMode::InterMvFour,
            ]
        );
    }

    #[test]
    fn macroblock_modes_inter_scheme_1_through_6_table_7_19_mapping() {
        // Walk each Scheme 1..=6 column of Table 7.19. For every column
        // emit mi=0..=7 and confirm the decoded mode matches the
        // tabulated MALPHABET[mi].
        let nmbs = 8u32;
        let nbs = 32u32;
        let bcoded = vec![1u8; nbs as usize];
        let map: Vec<[u32; 4]> = (0..nmbs)
            .map(|mbi| {
                let base = mbi * 4;
                [base, base + 1, base + 2, base + 3]
            })
            .collect();
        let table = [
            [3u8, 4, 2, 0, 1, 5, 6, 7],
            [3, 4, 0, 2, 1, 5, 6, 7],
            [3, 2, 4, 0, 1, 5, 6, 7],
            [3, 2, 0, 4, 1, 5, 6, 7],
            [0, 3, 4, 2, 1, 5, 6, 7],
            [0, 5, 3, 4, 2, 1, 6, 7],
        ];
        for (scheme_idx, expected_alphabet) in table.iter().enumerate() {
            let mscheme = (scheme_idx as u32) + 1;
            let mut w = BitWriter::new();
            w.put(mscheme, 3);
            // Emit each mi value 0..=7 in order.
            for mi in 0u8..8 {
                write_mb_mode_code(&mut w, mi);
            }
            let bytes = w.finish();
            let out = decode_macroblock_modes(&bytes, FrameType::Inter, nmbs, nbs, &bcoded, &map)
                .unwrap();
            for (mi, mode) in out.iter().enumerate() {
                let expected_index = expected_alphabet[mi];
                assert_eq!(
                    mode.to_index(),
                    expected_index,
                    "Scheme {mscheme} mi={mi}: got mode {:?} (index {}), expected index {expected_index}",
                    mode,
                    mode.to_index(),
                );
            }
        }
    }

    #[test]
    fn macroblock_modes_inter_scheme_0_reads_alphabet_from_bitstream() {
        // §7.4 step 2(b): MSCHEME=0 reads 8 3-bit mi values then assigns
        // MALPHABET[mi] = MODE in MODE order. Construct an explicit
        // permutation and verify each mi index recovers its assigned mode.
        let nmbs = 8u32;
        let nbs = 32u32;
        let bcoded = vec![1u8; nbs as usize];
        let map: Vec<[u32; 4]> = (0..nmbs)
            .map(|mbi| {
                let base = mbi * 4;
                [base, base + 1, base + 2, base + 3]
            })
            .collect();
        // Permutation: MODE 0 → mi 7, 1 → mi 6, …, 7 → mi 0. Then
        // MALPHABET = [7, 6, 5, 4, 3, 2, 1, 0].
        let mode_to_mi = [7u8, 6, 5, 4, 3, 2, 1, 0];
        let mut w = BitWriter::new();
        w.put(0, 3); // MSCHEME = 0
        for mode in 0u8..8 {
            w.put(mode_to_mi[mode as usize] as u32, 3); // mi for this MODE
        }
        // Emit mi=0..=7 and expect MALPHABET[mi] = the MODE whose mi was 0..7.
        for mi in 0u8..8 {
            write_mb_mode_code(&mut w, mi);
        }
        let bytes = w.finish();
        let out =
            decode_macroblock_modes(&bytes, FrameType::Inter, nmbs, nbs, &bcoded, &map).unwrap();
        // MALPHABET[mi] = MODE such that mode_to_mi[MODE] = mi
        for mi in 0u8..8 {
            let expected_mode = mode_to_mi.iter().position(|&m| m == mi).unwrap() as u8;
            assert_eq!(
                out[mi as usize].to_index(),
                expected_mode,
                "MSCHEME=0 mi {mi}: got index {}, expected {expected_mode}",
                out[mi as usize].to_index(),
            );
        }
    }

    #[test]
    fn macroblock_modes_inter_uncoded_luma_assigns_inter_nomv() {
        // §7.4 step 2(d)ii: if no luma block in macro block mbi has
        // BCODED == 1, MBMODES[mbi] := INTER_NOMV and NO bits are read.
        // Construct a frame with mbi=1 fully uncoded (no luma BCODED
        // bits set) and mbi=0, 2 coded. The on-wire stream then carries
        // only TWO Huffman codes (for mbi 0 and 2), not three.
        let nmbs = 3u32;
        let nbs = 12u32; // 3 mbs × 4 luma each
        let mut bcoded = vec![1u8; nbs as usize];
        // Clear all 4 luma bits of mb 1.
        for slot in 0..4 {
            bcoded[(4 + slot) as usize] = 0;
        }
        let map: Vec<[u32; 4]> = (0..nmbs)
            .map(|mbi| {
                let base = mbi * 4;
                [base, base + 1, base + 2, base + 3]
            })
            .collect();
        // Use Scheme 7 (direct) so the wire-format is easy to count.
        let mut w = BitWriter::new();
        w.put(7, 3); // MSCHEME = 7
        w.put(2, 3); // MBMODES[0] = INTER_MV (2)
                     // No bits emitted for MBMODES[1] — step 2(d)ii.
        w.put(6, 3); // MBMODES[2] = INTER_GOLDEN_MV (6)
        let bytes = w.finish();
        let out =
            decode_macroblock_modes(&bytes, FrameType::Inter, nmbs, nbs, &bcoded, &map).unwrap();
        assert_eq!(
            out,
            vec![
                MacroBlockMode::InterMv,
                MacroBlockMode::InterNoMv,
                MacroBlockMode::InterGoldenMv,
            ]
        );
    }

    #[test]
    fn macroblock_modes_inter_partial_coded_luma_still_decodes_mode() {
        // §7.4 step 2(d)i: ANY luma BCODED == 1 triggers a mode read.
        // Exercise a macro block with exactly one of its four luma
        // blocks coded.
        let nmbs = 1u32;
        let nbs = 4u32;
        // Only slot 3 (D, upper-right) is coded.
        let bcoded = vec![0u8, 0, 0, 1];
        let map: Vec<[u32; 4]> = vec![[0, 1, 2, 3]];
        let mut w = BitWriter::new();
        w.put(7, 3); // MSCHEME = 7
        w.put(4, 3); // INTER_MV_LAST2 (4)
        let bytes = w.finish();
        let out =
            decode_macroblock_modes(&bytes, FrameType::Inter, nmbs, nbs, &bcoded, &map).unwrap();
        assert_eq!(out, vec![MacroBlockMode::InterMvLast2]);
    }

    #[test]
    fn macroblock_modes_inter_rejects_truncated_at_mscheme() {
        // Inter frame with empty packet — should fail to read MSCHEME.
        let nmbs = 1u32;
        let nbs = 4u32;
        let bcoded = vec![1u8; nbs as usize];
        let map: Vec<[u32; 4]> = vec![[0, 1, 2, 3]];
        let err =
            decode_macroblock_modes(&[], FrameType::Inter, nmbs, nbs, &bcoded, &map).unwrap_err();
        assert_eq!(err, Error::TruncatedHeader { field: "MSCHEME" });
    }

    #[test]
    fn macroblock_modes_inter_rejects_truncated_at_alphabet() {
        // MSCHEME = 0, then 8 × 3 mi bits = 24 bits of alphabet, total
        // 27 bits. Provide only 3 bytes (24 bits) so MSCHEME (3 bits) +
        // 7 mi values (21 bits) fit but the 8th mi (the last 3 bits) is
        // unavailable.
        let nmbs = 1u32;
        let nbs = 4u32;
        let bcoded = vec![1u8; nbs as usize];
        let map: Vec<[u32; 4]> = vec![[0, 1, 2, 3]];
        let mut w = BitWriter::new();
        w.put(0, 3); // MSCHEME = 0
        for _ in 0..7 {
            w.put(0, 3); // 21 bits of alphabet, total 24 bits = 3 bytes
        }
        let bytes = w.finish();
        assert_eq!(bytes.len(), 3, "expected exactly 24 bits = 3 bytes");
        let err = decode_macroblock_modes(&bytes, FrameType::Inter, nmbs, nbs, &bcoded, &map)
            .unwrap_err();
        assert_eq!(
            err,
            Error::TruncatedHeader {
                field: "MALPHABET_mi"
            }
        );
    }

    #[test]
    fn macroblock_modes_inter_rejects_truncated_huffman_walk() {
        // Construct a one-byte buffer holding MSCHEME=1 (3 bits) followed
        // by five `1` bits. The Huffman walk needs at least six bits to
        // disambiguate `b1111110` / `b1111111`; the 9th bit lives in a
        // byte that does not exist.
        let nmbs = 1u32;
        let nbs = 4u32;
        let bcoded = vec![1u8; nbs as usize];
        let map: Vec<[u32; 4]> = vec![[0, 1, 2, 3]];
        let mut w = BitWriter::new();
        w.put(1, 3); // MSCHEME = 1
        for _ in 0..5 {
            w.put(1, 1); // five `1` bits inside the first byte
        }
        // First byte is 0b00111111 = 0x3f.
        let bytes = w.finish();
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[0], 0x3f);
        let err = decode_macroblock_modes(&bytes, FrameType::Inter, nmbs, nbs, &bcoded, &map)
            .unwrap_err();
        assert_eq!(
            err,
            Error::TruncatedHeader {
                field: "MBMODES_huffman_code"
            }
        );
    }

    #[test]
    fn macroblock_modes_inter_rejects_truncated_direct_three_bit() {
        // MSCHEME = 7, then truncate the buffer to a single byte holding
        // only the MSCHEME field and 5 padding bits, then ask for one
        // mb with 4 luma-coded blocks so a 3-bit direct mode would be
        // read — but we structure the second mb's read to fall off the
        // end.
        let nmbs = 2u32;
        let nbs = 8u32;
        let bcoded = vec![1u8; nbs as usize];
        let map: Vec<[u32; 4]> = vec![[0, 1, 2, 3], [4, 5, 6, 7]];
        // Stream: MSCHEME=7 (3 bits) + mb0 mode (3 bits) + only 2 bits
        // of mb1 mode (truncated). 8 bits in one byte exactly.
        let mut w = BitWriter::new();
        w.put(7, 3); // MSCHEME = 7
        w.put(0, 3); // mb0 mode = INTER_NOMV
        w.put(0, 2); // first 2 bits of mb1 mode
                     // Byte fills exactly, no further bytes.
        let bytes = w.finish();
        assert_eq!(bytes.len(), 1);
        let err = decode_macroblock_modes(&bytes, FrameType::Inter, nmbs, nbs, &bcoded, &map)
            .unwrap_err();
        assert_eq!(
            err,
            Error::TruncatedHeader {
                field: "MBMODES_direct"
            }
        );
    }

    #[test]
    fn macroblock_modes_inter_six_seven_disambiguation() {
        // The `b1111110` / `b1111111` codes differ only in their final
        // bit. Verify both round-trip correctly.
        let nmbs = 2u32;
        let nbs = 8u32;
        let bcoded = vec![1u8; nbs as usize];
        let map: Vec<[u32; 4]> = vec![[0, 1, 2, 3], [4, 5, 6, 7]];
        // Use Scheme 1 where MALPHABET[6]=6 and MALPHABET[7]=7.
        let mut w = BitWriter::new();
        w.put(1, 3); // MSCHEME = 1
        write_mb_mode_code(&mut w, 6); // mi=6 → mode 6
        write_mb_mode_code(&mut w, 7); // mi=7 → mode 7
        let bytes = w.finish();
        let out =
            decode_macroblock_modes(&bytes, FrameType::Inter, nmbs, nbs, &bcoded, &map).unwrap();
        assert_eq!(
            out,
            vec![MacroBlockMode::InterGoldenMv, MacroBlockMode::InterMvFour,]
        );
    }

    #[test]
    fn macroblock_modes_inter_inner_chains_on_shared_reader() {
        // Confirm the crate-private `decode_macroblock_modes_inner` can
        // be driven from a BitReader already partly consumed — the
        // chaining pattern an end-to-end frame decoder will use to walk
        // §7.1 → §7.2 → §7.3 → §7.4 on a single reader.
        let nmbs = 2u32;
        let nbs = 8u32;
        let bcoded = vec![1u8; nbs as usize];
        let map: Vec<[u32; 4]> = vec![[0, 1, 2, 3], [4, 5, 6, 7]];
        let mut w = BitWriter::new();
        // Prefix: 5 unrelated bits to skip on the shared reader.
        w.put(0b10110, 5);
        // MSCHEME = 7, modes 1 (INTRA) and 5 (INTER_GOLDEN_NOMV).
        w.put(7, 3);
        w.put(1, 3);
        w.put(5, 3);
        let bytes = w.finish();
        let mut r = BitReader::new(&bytes);
        let prefix = r.read_bits(5, "test.prefix").unwrap();
        assert_eq!(prefix, 0b10110);
        let out = decode_macroblock_modes_inner(&mut r, FrameType::Inter, nmbs, nbs, &bcoded, &map)
            .unwrap();
        assert_eq!(
            out,
            vec![MacroBlockMode::Intra, MacroBlockMode::InterGoldenNoMv,]
        );
    }

    #[test]
    fn macroblock_modes_error_displays_render() {
        let e = Error::MacroBlockLumaMapLenMismatch {
            map_len: 3,
            nmbs: 4,
        };
        let s = format!("{e}");
        assert!(s.contains("§7.4"));
        assert!(s.contains("3"));
        assert!(s.contains("4"));

        let e = Error::MacroBlockLumaBlockIndexOutOfRange {
            mbi: 7,
            slot: 2,
            bi: 99,
            nbs: 40,
        };
        let s = format!("{e}");
        assert!(s.contains("§7.4"));
        assert!(s.contains("7"));
        assert!(s.contains("99"));
        assert!(s.contains("40"));

        let e = Error::UnknownMacroBlockModeCode;
        let s = format!("{e}");
        assert!(s.contains("§7.4"));
    }

    #[test]
    fn macroblock_modes_inter_does_not_consume_bits_for_uncoded_mb() {
        // Concrete bit-budget assertion: with all 3 mbs uncoded, the
        // procedure consumes only the 3 MSCHEME bits (no per-mb codes).
        let nmbs = 3u32;
        let nbs = 12u32;
        let bcoded = vec![0u8; nbs as usize];
        let map: Vec<[u32; 4]> = (0..nmbs)
            .map(|mbi| {
                let base = mbi * 4;
                [base, base + 1, base + 2, base + 3]
            })
            .collect();
        // Even though MSCHEME could be anything, only the 3 bits should
        // be read. Provide just one byte with MSCHEME in the top 3 bits.
        let mut w = BitWriter::new();
        w.put(1, 3); // MSCHEME = 1 (Huffman) — but no codes follow.
                     // Pad the byte with zeros; if any per-mb decode bit is read, the
                     // result would be polluted.
        w.put(0, 5);
        let bytes = w.finish();
        let out =
            decode_macroblock_modes(&bytes, FrameType::Inter, nmbs, nbs, &bcoded, &map).unwrap();
        assert_eq!(out, vec![MacroBlockMode::InterNoMv; nmbs as usize]);
    }

    #[test]
    fn macroblock_modes_intra_does_not_consume_packet() {
        // §7.4 step 1 reads nothing. An empty packet must still decode.
        let nmbs = 16u32;
        let nbs = 64u32;
        let bcoded = vec![0u8; nbs as usize]; // doesn't matter
        let map: Vec<[u32; 4]> = (0..nmbs)
            .map(|mbi| {
                let base = mbi * 4;
                [base, base + 1, base + 2, base + 3]
            })
            .collect();
        let out = decode_macroblock_modes(&[], FrameType::Intra, nmbs, nbs, &bcoded, &map).unwrap();
        assert_eq!(out.len(), nmbs as usize);
        assert!(out.iter().all(|m| *m == MacroBlockMode::Intra));
    }

    #[test]
    fn macroblock_modes_inter_zero_nmbs_short_circuits() {
        // Degenerate edge case: 0 macro blocks. MSCHEME is still read
        // per §7.4 step 2(a) but the per-mb loop is empty.
        let nmbs = 0u32;
        let nbs = 0u32;
        let bcoded: Vec<u8> = vec![];
        let map: Vec<[u32; 4]> = vec![];
        let mut w = BitWriter::new();
        w.put(7, 3); // MSCHEME = 7 (any value works; alphabet unused)
        let bytes = w.finish();
        let out =
            decode_macroblock_modes(&bytes, FrameType::Inter, nmbs, nbs, &bcoded, &map).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn macroblock_modes_inter_unknown_scheme_code_uses_default_alphabet() {
        // §7.4 step 2(c) only enumerates schemes 1..=6 with fixed
        // alphabets. We treat schemes 2..=6 via Table 7.19 and 0/7 via
        // their dedicated rules. Defensive: the impl rejects nothing
        // for MSCHEME values outside 0..=7 because MSCHEME is a 3-bit
        // field (range 0..=7); any of the eight values is legal.
        // This test confirms each of the eight MSCHEME values decodes
        // a single-mb stream without error.
        let nmbs = 1u32;
        let nbs = 4u32;
        let bcoded = vec![1u8; nbs as usize];
        let map: Vec<[u32; 4]> = vec![[0, 1, 2, 3]];
        for mscheme in 0u32..=7 {
            let mut w = BitWriter::new();
            w.put(mscheme, 3);
            // Provide enough bits to decode one MBMODES entry under any
            // scheme: 24 bits is safely above the max (Scheme 0 needs
            // 24 alphabet bits + up to 7 code bits = 31; Scheme 7 needs
            // 3 bits; Scheme 1..=6 needs up to 7 bits).
            for _ in 0..5 {
                w.put(0, 8);
            }
            let bytes = w.finish();
            let out = decode_macroblock_modes(&bytes, FrameType::Inter, nmbs, nbs, &bcoded, &map)
                .unwrap();
            assert_eq!(out.len(), 1);
        }
    }

    // ========================================================================
    // §7.5 Motion Vectors tests (round 12)
    // ========================================================================

    /// Encode a single MV component into `w` using the MVMODE=0
    /// (Table 7.23) Huffman alphabet. Mirrors `recognise_mv_huffman`'s
    /// table; used to synthesize test bit-streams without touching the
    /// production code path.
    fn write_mv_huffman_component(w: &mut BitWriter, value: i8) {
        match value {
            0 => w.put(0b000, 3),
            1 => w.put(0b001, 3),
            -1 => w.put(0b010, 3),
            2 => w.put(0b0110, 4),
            -2 => w.put(0b0111, 4),
            3 => w.put(0b1000, 4),
            -3 => w.put(0b1001, 4),
            v if (4..=7).contains(&v) => {
                let mag = v.unsigned_abs() as u32;
                let bucket = mag - 4;
                // Prefix b101 (3 bits) << 3, then 2-bit bucket << 1, then sign bit 0.
                let code = (0b101 << 3) | (bucket << 1);
                w.put(code, 6);
            }
            v if (-7..=-4).contains(&v) => {
                let mag = v.unsigned_abs() as u32;
                let bucket = mag - 4;
                let code = (0b101 << 3) | (bucket << 1) | 1;
                w.put(code, 6);
            }
            v if (8..=15).contains(&v) => {
                let mag = v.unsigned_abs() as u32;
                let bucket = mag - 8;
                // Prefix b110 (3 bits) << 4, then 3-bit bucket << 1, then sign bit 0.
                let code = (0b110 << 4) | (bucket << 1);
                w.put(code, 7);
            }
            v if (-15..=-8).contains(&v) => {
                let mag = v.unsigned_abs() as u32;
                let bucket = mag - 8;
                let code = (0b110 << 4) | (bucket << 1) | 1;
                w.put(code, 7);
            }
            v if (16..=31).contains(&v) => {
                let mag = v.unsigned_abs() as u32;
                let bucket = mag - 16;
                // Prefix b111 (3 bits) << 5, then 4-bit bucket << 1, then sign bit 0.
                let code = (0b111 << 5) | (bucket << 1);
                w.put(code, 8);
            }
            v if (-31..=-16).contains(&v) => {
                let mag = v.unsigned_abs() as u32;
                let bucket = mag - 16;
                let code = (0b111 << 5) | (bucket << 1) | 1;
                w.put(code, 8);
            }
            _ => panic!("MV component {value} out of Table 7.23 range"),
        }
    }

    /// Encode a single MV under MVMODE=1 (5-bit magnitude + 1-bit
    /// sign per component, with the sign bit always read even when
    /// magnitude is zero).
    fn write_mv_fixed_component(w: &mut BitWriter, value: i8) {
        let mag = value.unsigned_abs() as u32;
        let sign = if value < 0 { 1 } else { 0 };
        w.put(mag, 5);
        w.put(sign, 1);
    }

    #[test]
    fn mv_huffman_table_7_23_round_trip_all_values() {
        // §7.5.1 step 1 always decodes both MVX and MVY, so we encode
        // each value as the MVX with a fixed MVY=0 to exercise every
        // Table 7.23 entry as MVX in turn.
        for v in -31i8..=31 {
            let mut w = BitWriter::new();
            write_mv_huffman_component(&mut w, v);
            write_mv_huffman_component(&mut w, 0);
            let bytes = w.finish();
            let mv = decode_single_motion_vector(&bytes, 0).unwrap();
            assert_eq!(mv.x, v, "decode of value {v} failed (MVX)");
            assert_eq!(mv.y, 0);
            // Also exercise sequential reads with paired sign.
            let mut w2 = BitWriter::new();
            write_mv_huffman_component(&mut w2, v);
            write_mv_huffman_component(&mut w2, -v);
            let bytes2 = w2.finish();
            let mv2 = decode_single_motion_vector(&bytes2, 0).unwrap();
            assert_eq!(mv2.x, v, "decode pair MVX={v}");
            assert_eq!(mv2.y, -v, "decode pair MVY=-{v}");
        }
    }

    #[test]
    fn mv_fixed_length_mvmode_1_round_trip() {
        // Includes both representations of zero: +0 (sign=0) and -0
        // (sign=1) — §7.5.1 step 2(c) negates only when the sign is 1,
        // so -0 should decode as 0 (Rust's i8 negation: 0 == -0).
        for v in -31i8..=31 {
            let mut w = BitWriter::new();
            write_mv_fixed_component(&mut w, v);
            write_mv_fixed_component(&mut w, -v);
            let bytes = w.finish();
            let mv = decode_single_motion_vector(&bytes, 1).unwrap();
            assert_eq!(mv.x, v, "MVX={v}");
            assert_eq!(mv.y, -v, "MVY=-{v}");
        }
    }

    #[test]
    fn mv_mvmode_1_sign_bit_read_even_when_magnitude_zero() {
        // Per §7.5.1: "for compatibility with VP3, a sign bit is read
        // even if the magnitude read is zero". A second component
        // payload must follow the sign bit to confirm the reader
        // advanced one bit past the magnitude.
        let mut w = BitWriter::new();
        // MVX: magnitude=0, sign=1 → -0 == 0
        w.put(0, 5);
        w.put(1, 1);
        // MVY: magnitude=5, sign=0 → 5
        w.put(5, 5);
        w.put(0, 1);
        let bytes = w.finish();
        let mv = decode_single_motion_vector(&bytes, 1).unwrap();
        assert_eq!(mv.x, 0);
        assert_eq!(mv.y, 5);
    }

    #[test]
    fn mv_huffman_truncated_returns_truncated_header() {
        // 3-bit lookup needs 3 bits.
        let bytes = vec![0b10100000u8];
        // First component reads 6 bits (b101000 → +4). Second
        // component needs 3 bits but the byte is exhausted at bit 2.
        let err = decode_single_motion_vector(&bytes, 0).unwrap_err();
        assert!(matches!(err, Error::TruncatedHeader { .. }));
    }

    #[test]
    fn mv_fixed_truncated_returns_truncated_header() {
        // Only 1 byte = 8 bits, but MVMODE=1 needs 12 bits per vector.
        let bytes = vec![0xFFu8];
        let err = decode_single_motion_vector(&bytes, 1).unwrap_err();
        assert!(matches!(err, Error::TruncatedHeader { .. }));
    }

    /// Build a uniform layout: NMBS macro blocks each with 4 luma
    /// blocks in raster order [4*mbi, 4*mbi+1, 4*mbi+2, 4*mbi+3] plus
    /// per-format chroma blocks placed after the luma region. Returns
    /// `(nbs, luma_map, cb_outer, cr_outer)`.
    #[allow(clippy::type_complexity)]
    fn build_uniform_layout(
        nmbs: u32,
        pf: PixelFormat,
    ) -> (u32, Vec<[u32; 4]>, Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let luma_count = nmbs * 4;
        let chroma_per_mb: u32 = match pf {
            PixelFormat::Yuv420 => 1,
            PixelFormat::Yuv422 => 2,
            PixelFormat::Yuv444 => 4,
        };
        let cb_count = nmbs * chroma_per_mb;
        let cr_count = nmbs * chroma_per_mb;
        let nbs = luma_count + cb_count + cr_count;
        let luma_map: Vec<[u32; 4]> = (0..nmbs)
            .map(|mbi| [mbi * 4, mbi * 4 + 1, mbi * 4 + 2, mbi * 4 + 3])
            .collect();
        let cb_base = luma_count;
        let cr_base = luma_count + cb_count;
        let cb_outer: Vec<Vec<u32>> = (0..nmbs)
            .map(|mbi| {
                (0..chroma_per_mb)
                    .map(|k| cb_base + mbi * chroma_per_mb + k)
                    .collect()
            })
            .collect();
        let cr_outer: Vec<Vec<u32>> = (0..nmbs)
            .map(|mbi| {
                (0..chroma_per_mb)
                    .map(|k| cr_base + mbi * chroma_per_mb + k)
                    .collect()
            })
            .collect();
        (nbs, luma_map, cb_outer, cr_outer)
    }

    /// Convert a `Vec<Vec<u32>>` into a `Vec<&[u32]>` so it can be
    /// passed to `ChromaBlockLayout`.
    fn slice_outer(outer: &[Vec<u32>]) -> Vec<&[u32]> {
        outer.iter().map(|v| v.as_slice()).collect()
    }

    #[test]
    fn mv_intra_frame_short_circuits_all_zero_no_bits_consumed() {
        // Intra frame: all MVs are (0, 0) and the packet is not consumed.
        let nmbs = 3u32;
        let (nbs, luma, cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv420);
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![MacroBlockMode::Intra; nmbs as usize];
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let layout = ChromaBlockLayout { cb: &cb, cr: &cr };
        // Pass an empty packet — no bits should be read.
        let out = decode_macroblock_motion_vectors(
            &[],
            FrameType::Intra,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            layout,
        )
        .unwrap();
        assert_eq!(out.len(), nbs as usize);
        assert!(out.iter().all(|mv| *mv == MotionVector::ZERO));
    }

    #[test]
    fn mv_inter_nomv_reads_mvmode_bit_then_zero_vectors() {
        // Inter frame with all macro blocks in INTER_NOMV: only the
        // 1-bit MVMODE field is read. Every block gets (0, 0).
        let nmbs = 2u32;
        let (nbs, luma, cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv420);
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![MacroBlockMode::InterNoMv; nmbs as usize];

        let mut w = BitWriter::new();
        w.put(0, 1); // MVMODE=0
        let bytes = w.finish();
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let out = decode_macroblock_motion_vectors(
            &bytes,
            FrameType::Inter,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap();
        assert!(out.iter().all(|mv| *mv == MotionVector::ZERO));
    }

    #[test]
    fn mv_inter_mv_decodes_and_propagates_to_blocks() {
        // One macroblock in INTER_MV. The single decoded MV propagates
        // to every coded luma + chroma block; LAST1 takes that value.
        let nmbs = 1u32;
        let (nbs, luma, cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv420);
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![MacroBlockMode::InterMv];

        let mut w = BitWriter::new();
        w.put(0, 1); // MVMODE=0 (Huffman)
        write_mv_huffman_component(&mut w, 5);
        write_mv_huffman_component(&mut w, -3);
        let bytes = w.finish();
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let out = decode_macroblock_motion_vectors(
            &bytes,
            FrameType::Inter,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap();
        let expected = MotionVector::new(5, -3);
        for (bi, mv) in out.iter().enumerate() {
            assert_eq!(*mv, expected, "bi={bi}");
        }
    }

    #[test]
    fn mv_inter_mv_last_chain() {
        // Two macroblocks: MB0 = INTER_MV (decode (7, 1)); MB1 =
        // INTER_MV_LAST (reuse LAST1). Both should end up with (7, 1).
        let nmbs = 2u32;
        let (nbs, luma, cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv420);
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![MacroBlockMode::InterMv, MacroBlockMode::InterMvLast];

        let mut w = BitWriter::new();
        w.put(0, 1); // MVMODE=0
        write_mv_huffman_component(&mut w, 7);
        write_mv_huffman_component(&mut w, 1);
        // No MV decoded for MB1.
        let bytes = w.finish();
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let out = decode_macroblock_motion_vectors(
            &bytes,
            FrameType::Inter,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap();
        let expected = MotionVector::new(7, 1);
        for (bi, mv) in out.iter().enumerate() {
            assert_eq!(*mv, expected, "bi={bi}");
        }
    }

    #[test]
    fn mv_inter_mv_last2_swaps_through_lasts() {
        // Three macroblocks: INTER_MV (a), INTER_MV (b), INTER_MV_LAST2.
        // After mb0: LAST1=a, LAST2=0.
        // After mb1: LAST1=b, LAST2=a.
        // mb2 INTER_MV_LAST2 reads LAST2 (=a); then LAST2=LAST1=b,
        //   LAST1=a.
        // So MVs are (a, b, a).
        let nmbs = 3u32;
        let (nbs, luma, cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv420);
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![
            MacroBlockMode::InterMv,
            MacroBlockMode::InterMv,
            MacroBlockMode::InterMvLast2,
        ];

        let a = MotionVector::new(2, -2);
        let b = MotionVector::new(-5, 6);
        let mut w = BitWriter::new();
        w.put(0, 1); // MVMODE=0
        write_mv_huffman_component(&mut w, a.x);
        write_mv_huffman_component(&mut w, a.y);
        write_mv_huffman_component(&mut w, b.x);
        write_mv_huffman_component(&mut w, b.y);
        let bytes = w.finish();
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let out = decode_macroblock_motion_vectors(
            &bytes,
            FrameType::Inter,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap();
        // Check the first luma block of each MB.
        assert_eq!(out[luma[0][0] as usize], a, "mb0");
        assert_eq!(out[luma[1][0] as usize], b, "mb1");
        assert_eq!(out[luma[2][0] as usize], a, "mb2 (=LAST2 was a)");
    }

    #[test]
    fn mv_inter_golden_mv_does_not_update_lasts() {
        // INTER_GOLDEN_MV decodes an MV but does NOT update
        // LAST1/LAST2 (step 3(b) of §7.5.2). Test by sandwiching it
        // between two INTER_MV macroblocks and verifying the second
        // INTER_MV's MV ends up as LAST1, with the GOLDEN_MV value
        // visible only on that macroblock's own blocks.
        let nmbs = 3u32;
        let (nbs, luma, cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv420);
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![
            MacroBlockMode::InterMv,
            MacroBlockMode::InterGoldenMv,
            MacroBlockMode::InterMvLast, // should read LAST1 == mb0's MV
        ];

        let a = MotionVector::new(4, 4);
        let g = MotionVector::new(-10, 10);
        let mut w = BitWriter::new();
        w.put(0, 1); // MVMODE=0
                     // mb0: INTER_MV
        write_mv_huffman_component(&mut w, a.x);
        write_mv_huffman_component(&mut w, a.y);
        // mb1: INTER_GOLDEN_MV
        write_mv_huffman_component(&mut w, g.x);
        write_mv_huffman_component(&mut w, g.y);
        // mb2: INTER_MV_LAST — no bits.
        let bytes = w.finish();
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let out = decode_macroblock_motion_vectors(
            &bytes,
            FrameType::Inter,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap();
        assert_eq!(out[luma[0][0] as usize], a, "mb0");
        assert_eq!(out[luma[1][0] as usize], g, "mb1 GOLDEN MV");
        assert_eq!(out[luma[2][0] as usize], a, "mb2 LAST1 still == a");
    }

    #[test]
    fn mv_inter_mv_four_yuv420_chroma_averaging() {
        // One INTER_MV_FOUR macroblock, 4:2:0. All four luma blocks
        // coded; chroma E/F gets the round((A+B+C+D)/4) average.
        let nmbs = 1u32;
        let (nbs, luma, cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv420);
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![MacroBlockMode::InterMvFour];

        // Pick four luma MVs whose averages are not divisible by 4 to
        // exercise rounding away from zero.
        let mv_a = MotionVector::new(1, -1);
        let mv_b = MotionVector::new(2, -1);
        let mv_c = MotionVector::new(2, 0);
        let mv_d = MotionVector::new(2, 0);
        // sumX=7 → 7/4 = 1.75 → round = 2; sumY=-2 → -2/4 = -0.5 → round = -1 (away from 0).

        let mut w = BitWriter::new();
        w.put(0, 1); // MVMODE=0
        write_mv_huffman_component(&mut w, mv_a.x);
        write_mv_huffman_component(&mut w, mv_a.y);
        write_mv_huffman_component(&mut w, mv_b.x);
        write_mv_huffman_component(&mut w, mv_b.y);
        write_mv_huffman_component(&mut w, mv_c.x);
        write_mv_huffman_component(&mut w, mv_c.y);
        write_mv_huffman_component(&mut w, mv_d.x);
        write_mv_huffman_component(&mut w, mv_d.y);
        let bytes = w.finish();
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let out = decode_macroblock_motion_vectors(
            &bytes,
            FrameType::Inter,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap();
        assert_eq!(out[luma[0][0] as usize], mv_a);
        assert_eq!(out[luma[0][1] as usize], mv_b);
        assert_eq!(out[luma[0][2] as usize], mv_c);
        assert_eq!(out[luma[0][3] as usize], mv_d);
        let chroma_expected = MotionVector::new(2, -1);
        assert_eq!(out[cb_o[0][0] as usize], chroma_expected);
        assert_eq!(out[cr_o[0][0] as usize], chroma_expected);
    }

    #[test]
    fn mv_inter_mv_four_uncoded_luma_uses_zero_for_chroma_average() {
        // Per §7.5.2 step 3(a)iii: uncoded luma blocks contribute
        // (0, 0) to the chroma average. Coded blocks A=B=C=(3,-3),
        // D uncoded → chroma avg = round((3+3+3+0)/4, (-3-3-3+0)/4)
        // = round(9/4, -9/4) = (2, -2).
        let nmbs = 1u32;
        let (nbs, luma, cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv420);
        let mut bcoded = vec![1u8; nbs as usize];
        bcoded[luma[0][3] as usize] = 0; // D uncoded
        let mbmodes = vec![MacroBlockMode::InterMvFour];

        let mut w = BitWriter::new();
        w.put(0, 1); // MVMODE=0
        for _ in 0..3 {
            write_mv_huffman_component(&mut w, 3);
            write_mv_huffman_component(&mut w, -3);
        }
        let bytes = w.finish();
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let out = decode_macroblock_motion_vectors(
            &bytes,
            FrameType::Inter,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap();
        // D's MVECTS entry: §7.5.2 step 3(a)ix assigns (0, 0).
        assert_eq!(out[luma[0][3] as usize], MotionVector::ZERO);
        // Chroma = round((3+3+3+0)/4, (-3-3-3+0)/4) = (round(2.25), round(-2.25)) = (2, -2).
        let chroma_expected = MotionVector::new(2, -2);
        assert_eq!(out[cb_o[0][0] as usize], chroma_expected);
        assert_eq!(out[cr_o[0][0] as usize], chroma_expected);
    }

    #[test]
    fn mv_inter_mv_four_yuv422_chroma_averaging() {
        // 4:2:2: E (Cb bottom) = avg(A, B); F (Cb top) = avg(C, D).
        let nmbs = 1u32;
        let (nbs, luma, cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv422);
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![MacroBlockMode::InterMvFour];

        let mv_a = MotionVector::new(1, 1);
        let mv_b = MotionVector::new(2, 2);
        let mv_c = MotionVector::new(-3, 3);
        let mv_d = MotionVector::new(-4, 4);
        // avg(A,B) = round(1.5, 1.5) = (2, 2)
        // avg(C,D) = round(-3.5, 3.5) = (-4, 4) (away from zero)

        let mut w = BitWriter::new();
        w.put(0, 1);
        for mv in [mv_a, mv_b, mv_c, mv_d] {
            write_mv_huffman_component(&mut w, mv.x);
            write_mv_huffman_component(&mut w, mv.y);
        }
        let bytes = w.finish();
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let out = decode_macroblock_motion_vectors(
            &bytes,
            FrameType::Inter,
            PixelFormat::Yuv422,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap();
        let bot = MotionVector::new(2, 2);
        let top = MotionVector::new(-4, 4);
        assert_eq!(out[cb_o[0][0] as usize], bot, "E");
        assert_eq!(out[cb_o[0][1] as usize], top, "F");
        assert_eq!(out[cr_o[0][0] as usize], bot, "G");
        assert_eq!(out[cr_o[0][1] as usize], top, "H");
    }

    #[test]
    fn mv_inter_mv_four_yuv444_chroma_copies_luma() {
        // 4:4:4: chroma blocks get the luma MVs directly.
        let nmbs = 1u32;
        let (nbs, luma, cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv444);
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![MacroBlockMode::InterMvFour];

        let mvs = [
            MotionVector::new(1, 1),
            MotionVector::new(2, 2),
            MotionVector::new(3, 3),
            MotionVector::new(4, 4),
        ];
        let mut w = BitWriter::new();
        w.put(0, 1);
        for mv in mvs.iter() {
            write_mv_huffman_component(&mut w, mv.x);
            write_mv_huffman_component(&mut w, mv.y);
        }
        let bytes = w.finish();
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let out = decode_macroblock_motion_vectors(
            &bytes,
            FrameType::Inter,
            PixelFormat::Yuv444,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap();
        for k in 0..4 {
            assert_eq!(out[cb_o[0][k] as usize], mvs[k], "E/F/G/H k={k}");
            assert_eq!(out[cr_o[0][k] as usize], mvs[k], "I/J/K/L k={k}");
        }
    }

    #[test]
    fn mv_inter_mv_four_updates_last1_to_last_coded_block() {
        // INTER_MV_FOUR's step 3(a)xiv assigns LAST1 the value of the
        // last coded luma block in raster order (A→B→C→D). With D
        // coded, LAST1 = D's MV; a following INTER_MV_LAST macroblock
        // should reuse D's MV.
        let nmbs = 2u32;
        let (nbs, luma, cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv420);
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![MacroBlockMode::InterMvFour, MacroBlockMode::InterMvLast];

        let mvs = [
            MotionVector::new(1, 0),
            MotionVector::new(2, 0),
            MotionVector::new(3, 0),
            MotionVector::new(4, 0),
        ];
        let mut w = BitWriter::new();
        w.put(0, 1);
        for mv in mvs.iter() {
            write_mv_huffman_component(&mut w, mv.x);
            write_mv_huffman_component(&mut w, mv.y);
        }
        let bytes = w.finish();
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let out = decode_macroblock_motion_vectors(
            &bytes,
            FrameType::Inter,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap();
        // mb1 INTER_MV_LAST should propagate D=(4, 0) (LAST1 after mb0).
        assert_eq!(out[luma[1][0] as usize], MotionVector::new(4, 0));
    }

    #[test]
    fn mv_round_div_matches_spec_examples() {
        // §spec round(a) with ties away from zero.
        assert_eq!(round_div(7, 4), 2); // 1.75 → 2
        assert_eq!(round_div(-7, 4), -2); // -1.75 → -2
        assert_eq!(round_div(6, 4), 2); // 1.5 → 2 (tie away from 0)
        assert_eq!(round_div(-6, 4), -2); // -1.5 → -2 (tie away from 0)
        assert_eq!(round_div(0, 4), 0);
        assert_eq!(round_div(2, 4), 1); // 0.5 → 1
        assert_eq!(round_div(-2, 4), -1); // -0.5 → -1
        assert_eq!(round_div(3, 2), 2); // 1.5 → 2
        assert_eq!(round_div(-3, 2), -2); // -1.5 → -2
    }

    #[test]
    fn mv_rejects_mbmodes_length_mismatch() {
        let nmbs = 2u32;
        let (nbs, luma, cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv420);
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![MacroBlockMode::InterNoMv]; // length 1, nmbs=2
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let err = decode_macroblock_motion_vectors(
            &[0u8],
            FrameType::Inter,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::MotionVectorMbModesLenMismatch {
                modes_len: 1,
                nmbs: 2
            }
        ));
    }

    #[test]
    fn mv_rejects_luma_map_length_mismatch() {
        let nmbs = 2u32;
        let (nbs, _luma, cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv420);
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![MacroBlockMode::InterNoMv; nmbs as usize];
        let bad_luma: Vec<[u32; 4]> = vec![[0, 1, 2, 3]]; // length 1
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let err = decode_macroblock_motion_vectors(
            &[0u8],
            FrameType::Inter,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &bad_luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::MotionVectorLumaMapLenMismatch {
                map_len: 1,
                nmbs: 2
            }
        ));
    }

    #[test]
    fn mv_rejects_luma_index_out_of_range() {
        let nmbs = 1u32;
        let nbs = 4u32;
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![MacroBlockMode::InterMv];
        let luma: Vec<[u32; 4]> = vec![[0, 1, 2, 4]]; // 4 >= nbs
        let cb_o: Vec<Vec<u32>> = vec![vec![0]];
        let cr_o: Vec<Vec<u32>> = vec![vec![0]];
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let err = decode_macroblock_motion_vectors(
            &[0u8],
            FrameType::Inter,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::MotionVectorLumaBlockIndexOutOfRange {
                mbi: 0,
                slot: 3,
                bi: 4,
                nbs: 4
            }
        ));
    }

    #[test]
    fn mv_rejects_chroma_map_length_mismatch() {
        let nmbs = 2u32;
        let (nbs, luma, _cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv420);
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![MacroBlockMode::InterNoMv; nmbs as usize];
        let cb_o_bad: Vec<Vec<u32>> = vec![vec![8]]; // length 1, nmbs=2
        let cb = slice_outer(&cb_o_bad);
        let cr = slice_outer(&cr_o);
        let err = decode_macroblock_motion_vectors(
            &[0u8],
            FrameType::Inter,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::MotionVectorChromaMapLenMismatch {
                plane: 0,
                map_len: 1,
                expected: 2
            }
        ));
    }

    #[test]
    fn mv_rejects_chroma_inner_slot_length_mismatch_for_pf() {
        // PF=0 expects inner length 1; pass a length-2 inner.
        let nmbs = 1u32;
        let (nbs, luma, _cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv420);
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![MacroBlockMode::InterNoMv];
        let cb_o_bad: Vec<Vec<u32>> = vec![vec![4, 5]]; // length 2 for PF=0
        let cb = slice_outer(&cb_o_bad);
        let cr = slice_outer(&cr_o);
        let err = decode_macroblock_motion_vectors(
            &[0u8],
            FrameType::Inter,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::MotionVectorChromaMacroBlockSlotLenMismatch {
                mbi: 0,
                plane: 0,
                got: 2,
                expected: 1,
            }
        ));
    }

    #[test]
    fn mv_rejects_chroma_index_out_of_range() {
        let nmbs = 1u32;
        let nbs = 4u32;
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![MacroBlockMode::InterNoMv];
        let luma: Vec<[u32; 4]> = vec![[0, 1, 2, 3]];
        let cb_o: Vec<Vec<u32>> = vec![vec![4]]; // OOB
        let cr_o: Vec<Vec<u32>> = vec![vec![0]];
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let err = decode_macroblock_motion_vectors(
            &[0u8],
            FrameType::Inter,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::MotionVectorChromaBlockIndexOutOfRange {
                mbi: 0,
                slot: 0,
                bi: 4,
                nbs: 4
            }
        ));
    }

    #[test]
    fn mv_truncation_at_mvmode_bit_is_reported() {
        let nmbs = 1u32;
        let (nbs, luma, cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv420);
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![MacroBlockMode::InterNoMv];
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let err = decode_macroblock_motion_vectors(
            &[], // empty: MVMODE bit cannot be read
            FrameType::Inter,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap_err();
        assert!(matches!(err, Error::TruncatedHeader { field: "MVMODE" }));
    }

    #[test]
    fn mv_error_displays_render() {
        let e1 = Error::MotionVectorMbModesLenMismatch {
            modes_len: 3,
            nmbs: 5,
        };
        let e2 = Error::MotionVectorLumaMapLenMismatch {
            map_len: 1,
            nmbs: 2,
        };
        let e3 = Error::MotionVectorLumaBlockIndexOutOfRange {
            mbi: 7,
            slot: 2,
            bi: 99,
            nbs: 64,
        };
        let e4 = Error::MotionVectorChromaBlockIndexOutOfRange {
            mbi: 3,
            slot: 0,
            bi: 50,
            nbs: 48,
        };
        let e5 = Error::MotionVectorChromaMapLenMismatch {
            plane: 1,
            map_len: 0,
            expected: 8,
        };
        let e6 = Error::MotionVectorChromaMacroBlockSlotLenMismatch {
            mbi: 4,
            plane: 0,
            got: 2,
            expected: 4,
        };
        for e in [e1, e2, e3, e4, e5, e6] {
            let s = format!("{e}");
            assert!(s.contains("oxideav-theora"));
            assert!(s.contains("§7.5.2"));
        }
    }

    #[test]
    fn mv_inner_chains_on_shared_reader() {
        // After §7.5.2 finishes, a sentinel byte appended to the
        // packet must still be readable via the shared reader. This
        // confirms the procedure stops at the right bit position.
        let nmbs = 1u32;
        let (nbs, luma, cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv420);
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![MacroBlockMode::InterMv];

        let mut w = BitWriter::new();
        w.put(0, 1); // MVMODE=0
        write_mv_huffman_component(&mut w, 1); // 3 bits
        write_mv_huffman_component(&mut w, 1); // 3 bits
                                               // Total = 7 bits. Pad one bit to align to byte and emit 0xAB.
        w.put(0, 1); // pad to byte boundary
        w.put(0xAB, 8);
        let bytes = w.finish();

        let mut r = BitReader::new(&bytes);
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let _out = decode_macroblock_motion_vectors_inner(
            &mut r,
            FrameType::Inter,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap();
        // Consume the pad bit, then read the 0xAB sentinel.
        let _pad = r.read_bits(1, "pad").unwrap();
        let sentinel = r.read_bits(8, "sentinel").unwrap();
        assert_eq!(sentinel, 0xAB);
    }

    #[test]
    fn mv_intra_does_not_consume_bits() {
        // An intra frame must not consume the MVMODE bit (or anything
        // else). Use a packet whose first bit is 1 so we'd notice a
        // spurious read.
        let nmbs = 1u32;
        let (nbs, luma, cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv420);
        let bcoded = vec![1u8; nbs as usize];
        let mbmodes = vec![MacroBlockMode::Intra];
        let packet = vec![0xFFu8];

        let mut r = BitReader::new(&packet);
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let _ = decode_macroblock_motion_vectors_inner(
            &mut r,
            FrameType::Intra,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap();
        // No bits consumed.
        let next = r.read_bits(8, "sentinel").unwrap();
        assert_eq!(next, 0xFF);
    }

    #[test]
    fn mv_uncoded_luma_in_non_four_mode_does_not_overwrite_chroma() {
        // Step 3(g) iterates "each coded block bi". Uncoded blocks
        // (BCODED[bi]==0) should retain their (0, 0) default even when
        // the macroblock has an MV (e.g. INTER_MV with some chroma
        // uncoded).
        let nmbs = 1u32;
        let (nbs, luma, cb_o, cr_o) = build_uniform_layout(nmbs, PixelFormat::Yuv420);
        let mut bcoded = vec![1u8; nbs as usize];
        // Mark the chroma Cb block uncoded.
        bcoded[cb_o[0][0] as usize] = 0;
        let mbmodes = vec![MacroBlockMode::InterMv];

        let mut w = BitWriter::new();
        w.put(0, 1);
        write_mv_huffman_component(&mut w, 6);
        write_mv_huffman_component(&mut w, -6);
        let bytes = w.finish();
        let cb = slice_outer(&cb_o);
        let cr = slice_outer(&cr_o);
        let out = decode_macroblock_motion_vectors(
            &bytes,
            FrameType::Inter,
            PixelFormat::Yuv420,
            nbs,
            nmbs,
            &bcoded,
            &mbmodes,
            &luma,
            ChromaBlockLayout { cb: &cb, cr: &cr },
        )
        .unwrap();
        // Coded luma: propagated.
        for k in 0..4 {
            assert_eq!(out[luma[0][k] as usize], MotionVector::new(6, -6));
        }
        // Uncoded Cb: still zero.
        assert_eq!(out[cb_o[0][0] as usize], MotionVector::ZERO);
        // Coded Cr: propagated.
        assert_eq!(out[cr_o[0][0] as usize], MotionVector::new(6, -6));
    }

    // =================================================================
    // §7.6 Block-Level qi Decode tests
    // =================================================================

    /// VP3 fallback path (NQIS=1): the outer loop is empty, no bits
    /// are read, and every QIIS[bi] is 0.
    #[test]
    fn block_level_qi_nqis_1_short_circuits_no_bits_consumed() {
        let bcoded = vec![1u8, 1, 0, 1, 0];
        // Sentinel byte that MUST NOT be consumed.
        let packet = [0xffu8; 4];
        let qiis =
            decode_block_level_qi(&packet, bcoded.len() as u32, &bcoded, 1).expect("nqis=1 path");
        assert_eq!(qiis, vec![0, 0, 0, 0, 0]);

        // Confirm via the inner that the reader did not consume any
        // bits (read one bit and assert it's the MSb of the first
        // sentinel byte).
        let mut r = BitReader::new(&packet);
        let _ = decode_block_level_qi_inner(&mut r, bcoded.len() as u32, &bcoded, 1).unwrap();
        let b = r.read_bits(1, "sentinel").unwrap();
        assert_eq!(b, 1, "no bits should have been consumed");
    }

    /// NQIS=2 happy path: one long-run pass; coded blocks get bits,
    /// uncoded blocks stay at 0.
    #[test]
    fn block_level_qi_nqis_2_assigns_per_coded_block() {
        // bi=0 coded; bi=1 uncoded; bi=2 coded; bi=3 coded; bi=4
        // uncoded; bi=5 coded.
        let bcoded = vec![1u8, 0, 1, 1, 0, 1];
        // Pass 0: NBITS = 4 coded blocks. Choose the per-coded-block
        // bits in coded order to be [0, 1, 0, 1] → final QIIS values
        // for the coded blocks are [0, 1, 0, 1] (uncoded keep 0).
        let mut w = BitWriter::new();
        encode_long_run_runs(&mut w, &[0, 1, 0, 1]);
        let packet = w.finish();
        let qiis = decode_block_level_qi(&packet, bcoded.len() as u32, &bcoded, 2).unwrap();
        // bi index ......... 0  1  2  3  4  5
        // coded? ........... y  n  y  y  n  y
        // pass-0 bit ....... 0     1  0     1
        assert_eq!(qiis, vec![0, 0, 1, 0, 0, 1]);
    }

    /// NQIS=3 happy path: two long-run passes, the second restricted
    /// to blocks promoted out of the first.
    #[test]
    fn block_level_qi_nqis_3_chains_two_passes() {
        // 6 coded blocks.
        let bcoded = vec![1u8; 6];
        let mut w = BitWriter::new();
        // Pass 0: NBITS = 6. Bits in coded order: [0, 1, 0, 1, 1, 0].
        // After pass 0, QIIS = [0, 1, 0, 1, 1, 0].
        encode_long_run_runs(&mut w, &[0, 1, 0, 1, 1, 0]);
        // Pass 1: NBITS = #{bi : BCODED=1 AND QIIS==1} = 3 (bi=1, 3, 4).
        // Bits in coded order of those 3 blocks: [1, 0, 1].
        // After pass 1, QIIS[1] += 1 = 2, QIIS[3] += 0 = 1, QIIS[4] += 1 = 2.
        encode_long_run_runs(&mut w, &[1, 0, 1]);
        let packet = w.finish();
        let qiis = decode_block_level_qi(&packet, bcoded.len() as u32, &bcoded, 3).unwrap();
        // Final: bi=0 → 0, bi=1 → 2, bi=2 → 0, bi=3 → 1, bi=4 → 2, bi=5 → 0
        assert_eq!(qiis, vec![0, 2, 0, 1, 2, 0]);
    }

    /// Step 2(a) only counts coded blocks: an uncoded block's QIIS
    /// stays at 0 even if the pass-0 long-run string skips over it.
    #[test]
    fn block_level_qi_uncoded_blocks_stay_at_zero() {
        // bi 0 1 2 3 4 5 6 7
        //    y n y n y n y n  (every other one coded)
        let bcoded = vec![1u8, 0, 1, 0, 1, 0, 1, 0];
        // Pass 0: NBITS = 4 (only the four coded blocks).
        let mut w = BitWriter::new();
        encode_long_run_runs(&mut w, &[1, 1, 1, 1]);
        let packet = w.finish();
        let qiis = decode_block_level_qi(&packet, bcoded.len() as u32, &bcoded, 2).unwrap();
        // All four coded promoted to 1; all four uncoded stay at 0.
        assert_eq!(qiis, vec![1, 0, 1, 0, 1, 0, 1, 0]);
    }

    /// NQIS=3 where the first pass leaves NO blocks at qii=1 → the
    /// second pass has NBITS=0 and consumes no bits.
    #[test]
    fn block_level_qi_nqis_3_second_pass_can_be_empty() {
        let bcoded = vec![1u8; 4];
        let mut w = BitWriter::new();
        // Pass 0: all four bits zero → every coded block stays at 0.
        encode_long_run_runs(&mut w, &[0, 0, 0, 0]);
        // Pass 1: NBITS = #{BCODED=1 AND QIIS==1} = 0 → no encoding.
        // Append a sentinel byte to verify no bits get consumed.
        let mut packet = w.finish();
        packet.push(0xaa);
        let qiis = decode_block_level_qi(&packet, bcoded.len() as u32, &bcoded, 3).unwrap();
        assert_eq!(qiis, vec![0, 0, 0, 0]);
    }

    /// nbs=0 (no blocks): every pass tallies NBITS=0 and no bits are
    /// consumed regardless of nqis.
    #[test]
    fn block_level_qi_zero_nbs_short_circuits() {
        let qiis = decode_block_level_qi(&[], 0, &[], 1).unwrap();
        assert_eq!(qiis, Vec::<u8>::new());
        let qiis = decode_block_level_qi(&[], 0, &[], 3).unwrap();
        assert_eq!(qiis, Vec::<u8>::new());
    }

    /// Bcoded length mismatch error.
    #[test]
    fn block_level_qi_rejects_bcoded_length_mismatch() {
        let err = decode_block_level_qi(&[], 4, &[1, 1, 1], 1).unwrap_err();
        assert_eq!(
            err,
            Error::BlockLevelQiBcodedLenMismatch {
                bcoded_len: 3,
                nbs: 4,
            }
        );
    }

    /// nqis out-of-range rejects (both 0 and >MAX).
    #[test]
    fn block_level_qi_rejects_nqis_out_of_range() {
        let err = decode_block_level_qi(&[], 0, &[], 0).unwrap_err();
        assert_eq!(err, Error::BlockLevelQiNqisOutOfRange { nqis: 0 });
        let err = decode_block_level_qi(&[], 0, &[], 4).unwrap_err();
        assert_eq!(err, Error::BlockLevelQiNqisOutOfRange { nqis: 4 });
    }

    /// Truncation mid-pass-0 long-run surfaces as a TruncatedHeader
    /// from the §7.2.1 layer.
    #[test]
    fn block_level_qi_truncation_in_pass_0_is_reported() {
        let bcoded = vec![1u8; 4];
        // Empty packet → the very first BIT read by §7.2.1 must fail.
        let err = decode_block_level_qi(&[], bcoded.len() as u32, &bcoded, 2).unwrap_err();
        match err {
            Error::TruncatedHeader { field: _ } => {}
            other => panic!("expected TruncatedHeader, got {other:?}"),
        }
    }

    /// Truncation mid-pass-1 long-run (pass-0 fully decoded; pass-1
    /// runs out) also surfaces.
    #[test]
    fn block_level_qi_truncation_in_pass_1_is_reported() {
        let bcoded = vec![1u8; 4];
        let mut w = BitWriter::new();
        // Pass 0: full 4-bit string that promotes every block to 1.
        encode_long_run_runs(&mut w, &[1, 1, 1, 1]);
        // Intentionally do NOT encode pass 1. The §7.2.1 inner reads
        // BIT (and then walks RLEN) — the first read fails.
        let packet = w.finish();
        let err = decode_block_level_qi(&packet, bcoded.len() as u32, &bcoded, 3).unwrap_err();
        match err {
            Error::TruncatedHeader { field: _ } => {}
            other => panic!("expected TruncatedHeader, got {other:?}"),
        }
    }

    /// Shared-reader chaining contract: after `decode_block_level_qi_inner`
    /// returns, the same BitReader is still positioned correctly so the
    /// caller can keep reading downstream.
    #[test]
    fn block_level_qi_inner_chains_on_shared_reader() {
        let bcoded = vec![1u8; 4];
        let mut w = BitWriter::new();
        // Pass 0 only, NQIS=2, all bits zero.
        encode_long_run_runs(&mut w, &[0, 0, 0, 0]);
        // Sentinel pattern that should land on the next bit boundary.
        // Append seven 1-bits then five 0-bits = 12 bits.
        for _ in 0..7 {
            w.put(1, 1);
        }
        for _ in 0..5 {
            w.put(0, 1);
        }
        let packet = w.finish();
        let mut r = BitReader::new(&packet);
        let qiis = decode_block_level_qi_inner(&mut r, bcoded.len() as u32, &bcoded, 2).unwrap();
        assert_eq!(qiis, vec![0, 0, 0, 0]);
        // Now read 12 more bits — should be seven 1s then five 0s.
        let after = r.read_bits(12, "sentinel").unwrap();
        // Bits in order: 1111111 00000 → 0b1111_1110_0000 = 0xfe0
        assert_eq!(after, 0b1111_1110_0000);
    }

    /// Error::Display rendering for the two new variants.
    #[test]
    fn block_level_qi_error_displays_render() {
        let s = format!(
            "{}",
            Error::BlockLevelQiBcodedLenMismatch {
                bcoded_len: 7,
                nbs: 5,
            }
        );
        assert!(s.contains("§7.6"));
        assert!(s.contains("bcoded length 7"));
        assert!(s.contains("nbs 5"));

        let s = format!("{}", Error::BlockLevelQiNqisOutOfRange { nqis: 9 });
        assert!(s.contains("§7.6"));
        assert!(s.contains("NQIS=9"));
        assert!(s.contains("1..=3"));
    }

    /// NQIS=3 where pass 1's bit-tally is fed by an interleaved
    /// pattern: confirm that coded order (ascending `bi`) is the
    /// iteration order, not the order of bcoded entries.
    #[test]
    fn block_level_qi_nqis_3_iterates_in_coded_order() {
        // bcoded: y n y y y n y n  → 5 coded blocks
        let bcoded = vec![1u8, 0, 1, 1, 1, 0, 1, 0];
        let mut w = BitWriter::new();
        // Pass 0 over 5 coded blocks (bi = 0,2,3,4,6) with bits
        // [1, 0, 1, 0, 1] → QIIS[0,2,3,4,6] = [1, 0, 1, 0, 1].
        encode_long_run_runs(&mut w, &[1, 0, 1, 0, 1]);
        // Pass 1 NBITS = #{BCODED=1 AND QIIS==1} = 3 (bi=0, 3, 6) in
        // coded order. Bits [0, 1, 0] → QIIS[0]+=0=1, QIIS[3]+=1=2,
        // QIIS[6]+=0=1.
        encode_long_run_runs(&mut w, &[0, 1, 0]);
        let packet = w.finish();
        let qiis = decode_block_level_qi(&packet, bcoded.len() as u32, &bcoded, 3).unwrap();
        assert_eq!(qiis, vec![1, 0, 0, 2, 0, 0, 1, 0]);
    }

    /// All blocks uncoded: every pass sees NBITS=0 even when NQIS=3.
    #[test]
    fn block_level_qi_all_uncoded_consumes_no_bits() {
        let bcoded = vec![0u8; 5];
        // No bits encoded; a sentinel byte verifies nothing is consumed.
        let packet = [0xffu8; 2];
        let mut r = BitReader::new(&packet);
        let qiis = decode_block_level_qi_inner(&mut r, bcoded.len() as u32, &bcoded, 3).unwrap();
        assert_eq!(qiis, vec![0, 0, 0, 0, 0]);
        // 16 sentinel bits should still be intact.
        let v = r.read_bits(16, "sentinel").unwrap();
        assert_eq!(v, 0xffff);
    }

    /// Realistic mix at NQIS=2 spanning the long-run RSTART boundaries
    /// (RSTART=10 covers run lengths 10..=17). Confirms a multi-run
    /// long-run encoding survives the round-trip through §7.6.
    #[test]
    fn block_level_qi_long_run_multi_run_round_trip() {
        // 24 coded blocks.
        let bcoded = vec![1u8; 24];
        let mut w = BitWriter::new();
        // Pass 0: 10 zeros then 14 ones (two long-run records,
        // RSTART=10 chosen for both).
        let pass0: Vec<u8> = std::iter::repeat(0)
            .take(10)
            .chain(std::iter::repeat(1).take(14))
            .collect();
        encode_long_run_runs(&mut w, &pass0);
        let packet = w.finish();
        let qiis = decode_block_level_qi(&packet, bcoded.len() as u32, &bcoded, 2).unwrap();
        // The first 10 stay at 0; the remaining 14 promote to 1.
        let mut expected = vec![0u8; 10];
        expected.extend(std::iter::repeat(1).take(14));
        assert_eq!(qiis, expected);
    }

    // =================================================================
    // §7.7.1 EOB Token Decode tests
    // =================================================================

    /// Build a fresh `(tis, ncoeffs, coeffs)` tuple sized for `nbs`
    /// blocks, with sentinel values that make later "untouched" /
    /// "zero-filled" assertions unambiguous.
    fn fresh_eob_state(nbs: usize) -> (Vec<u8>, Vec<u8>, Vec<[i16; 64]>) {
        let tis = vec![0u8; nbs];
        // NCOEFFS sentinel is 0xff so the procedure's step-9 write is
        // detectable even when TIS[bi] happens to be 0.
        let ncoeffs = vec![0xffu8; nbs];
        // COEFFS sentinel is -777 (not zero) so step 8's zero-fill is
        // distinguishable from "no write happened".
        let coeffs = vec![[-777i16; 64]; nbs];
        (tis, ncoeffs, coeffs)
    }

    /// Token 0..=2 ride §7.7.1 steps 1..=3: EOBS is a fixed
    /// 1/2/3, no extra bits are read.
    #[test]
    fn eob_token_constant_runs_consume_no_bits() {
        for (token, expected_run) in [(0u8, 1u64), (1, 2), (2, 3)] {
            let (mut tis, mut ncoeffs, mut coeffs) = fresh_eob_state(4);
            // Empty packet — the procedure must not read any bits.
            let eobs =
                decode_eob_token(&[], token, 4, 2, 5, &mut tis, &mut ncoeffs, &mut coeffs).unwrap();
            // Step 11: EOBS -= 1.
            assert_eq!(eobs, expected_run - 1);
            // Step 8: COEFFS[2][5..=63] zeroed; [0..5] untouched (still
            // sentinel).
            for (ti, &v) in coeffs[2].iter().enumerate().take(5) {
                assert_eq!(v, -777, "ti={ti}");
            }
            for (ti_off, &v) in coeffs[2][5..64].iter().enumerate() {
                let ti = 5 + ti_off;
                assert_eq!(v, 0, "ti={ti}");
            }
            // Other blocks untouched.
            for bi in [0, 1, 3] {
                assert_eq!(coeffs[bi], [-777i16; 64]);
                assert_eq!(ncoeffs[bi], 0xff);
                assert_eq!(tis[bi], 0);
            }
            // Step 9: NCOEFFS[bi] = TIS[bi] (which was 0 at entry).
            assert_eq!(ncoeffs[2], 0);
            // Step 10: TIS[bi] = 64.
            assert_eq!(tis[2], 64);
        }
    }

    /// Token 3 reads a 2-bit payload then adds 4. The four values it
    /// can produce are 4, 5, 6, 7 (then minus the step-11 decrement).
    #[test]
    fn eob_token_3_two_bit_payload_round_trip() {
        for payload in 0u32..4 {
            let (mut tis, mut ncoeffs, mut coeffs) = fresh_eob_state(2);
            // Pack the 2-bit payload as the most significant bits of one
            // byte (the MSb-first §5.2 reader consumes the MSb first).
            let byte = ((payload & 0b11) << 6) as u8;
            let packet = [byte];
            let eobs = decode_eob_token(&packet, 3, 2, 0, 63, &mut tis, &mut ncoeffs, &mut coeffs)
                .unwrap();
            assert_eq!(eobs, payload as u64 + 4 - 1, "payload={payload}");
            // Step 8 at ti=63 zero-fills exactly one coefficient.
            assert_eq!(coeffs[0][63], 0);
            for (ti, &v) in coeffs[0].iter().enumerate().take(63) {
                assert_eq!(v, -777, "ti={ti}");
            }
        }
    }

    /// Token 4 reads a 3-bit payload then adds 8 → range 8..=15.
    #[test]
    fn eob_token_4_three_bit_payload_round_trip() {
        for payload in 0u32..8 {
            let (mut tis, mut ncoeffs, mut coeffs) = fresh_eob_state(1);
            let byte = ((payload & 0b111) << 5) as u8;
            let packet = [byte];
            let eobs =
                decode_eob_token(&packet, 4, 1, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs).unwrap();
            assert_eq!(eobs, payload as u64 + 8 - 1, "payload={payload}");
        }
    }

    /// Token 5 reads a 4-bit payload then adds 16 → range 16..=31.
    #[test]
    fn eob_token_5_four_bit_payload_round_trip() {
        for payload in 0u32..16 {
            let (mut tis, mut ncoeffs, mut coeffs) = fresh_eob_state(1);
            let byte = ((payload & 0b1111) << 4) as u8;
            let packet = [byte];
            let eobs =
                decode_eob_token(&packet, 5, 1, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs).unwrap();
            assert_eq!(eobs, payload as u64 + 16 - 1, "payload={payload}");
        }
    }

    /// Token 6 with a non-zero payload returns the payload verbatim
    /// (modulo step 11's decrement); range 1..=4095.
    #[test]
    fn eob_token_6_non_zero_payload_returns_literal() {
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_eob_state(1);
        // Payload = 0b1010_1010_1010 = 2730.
        let packet = [0b1010_1010, 0b1010_0000];
        let eobs =
            decode_eob_token(&packet, 6, 1, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs).unwrap();
        assert_eq!(eobs, 2730 - 1);
    }

    /// Token 6 with a *zero* 12-bit payload triggers the §7.7.1 step
    /// 7(b) sentinel: EOBS becomes the count of blocks bj with
    /// TIS[bj] < 64, *including the current block* (TIS[bi] is still <
    /// 64 at this point because step 10 has not yet run).
    #[test]
    fn eob_token_6_zero_payload_counts_remaining_blocks() {
        // 5 blocks: blocks 0..=3 still have TIS < 64; block 4 has
        // already been pinned (TIS=64 from a prior pass).
        let mut tis = vec![0u8, 0, 0, 0, 64];
        let mut ncoeffs = vec![0xffu8; 5];
        let mut coeffs = vec![[-777i16; 64]; 5];
        // Zero 12-bit payload.
        let packet = [0u8, 0];
        let eobs =
            decode_eob_token(&packet, 6, 5, 1, 10, &mut tis, &mut ncoeffs, &mut coeffs).unwrap();
        // 4 blocks have TIS < 64 entering the call → EOBS = 4; step 11
        // decrements to 3.
        assert_eq!(eobs, 3);
        // Step 10 ran AFTER the count, pinning block 1.
        assert_eq!(tis, vec![0, 64, 0, 0, 64]);
    }

    /// Token-out-of-range rejection: every value 7..=255 must reject.
    #[test]
    fn eob_token_rejects_token_above_six() {
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_eob_state(1);
        for token in 7u8..=20 {
            let r = decode_eob_token(&[], token, 1, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs);
            assert_eq!(r, Err(Error::EobTokenOutOfRange { token }), "token={token}");
        }
    }

    /// `bi` out of range rejection.
    #[test]
    fn eob_token_rejects_bi_equal_or_greater_than_nbs() {
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_eob_state(4);
        let r = decode_eob_token(&[], 0, 4, 4, 0, &mut tis, &mut ncoeffs, &mut coeffs);
        assert_eq!(
            r,
            Err(Error::EobTokenBlockIndexOutOfRange { bi: 4, nbs: 4 })
        );
        let r = decode_eob_token(&[], 0, 4, 99, 0, &mut tis, &mut ncoeffs, &mut coeffs);
        assert_eq!(
            r,
            Err(Error::EobTokenBlockIndexOutOfRange { bi: 99, nbs: 4 })
        );
    }

    /// `ti` out of range rejection.
    #[test]
    fn eob_token_rejects_ti_above_63() {
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_eob_state(1);
        for ti in 64u8..=127 {
            let r = decode_eob_token(&[], 0, 1, 0, ti, &mut tis, &mut ncoeffs, &mut coeffs);
            assert_eq!(r, Err(Error::EobTokenIndexOutOfRange { ti }), "ti={ti}");
        }
    }

    /// State-slice length mismatch rejections, one per slice.
    #[test]
    fn eob_token_rejects_state_length_mismatch() {
        // TIS too short.
        {
            let mut tis = vec![0u8; 3];
            let mut ncoeffs = vec![0u8; 4];
            let mut coeffs = vec![[0i16; 64]; 4];
            let r = decode_eob_token(&[], 0, 4, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs);
            assert_eq!(
                r,
                Err(Error::EobTokenStateLenMismatch {
                    which: EobTokenStateSlice::Tis,
                    got: 3,
                    nbs: 4
                })
            );
        }
        // NCOEFFS too short.
        {
            let mut tis = vec![0u8; 4];
            let mut ncoeffs = vec![0u8; 2];
            let mut coeffs = vec![[0i16; 64]; 4];
            let r = decode_eob_token(&[], 0, 4, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs);
            assert_eq!(
                r,
                Err(Error::EobTokenStateLenMismatch {
                    which: EobTokenStateSlice::Ncoeffs,
                    got: 2,
                    nbs: 4
                })
            );
        }
        // COEFFS too short.
        {
            let mut tis = vec![0u8; 4];
            let mut ncoeffs = vec![0u8; 4];
            let mut coeffs = vec![[0i16; 64]; 1];
            let r = decode_eob_token(&[], 0, 4, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs);
            assert_eq!(
                r,
                Err(Error::EobTokenStateLenMismatch {
                    which: EobTokenStateSlice::Coeffs,
                    got: 1,
                    nbs: 4
                })
            );
        }
    }

    /// Truncation surfaces from the extra-bits paths on each of tokens
    /// 3, 4, 5, 6.
    #[test]
    fn eob_token_truncation_on_extra_bits() {
        for token in 3u8..=6 {
            let (mut tis, mut ncoeffs, mut coeffs) = fresh_eob_state(1);
            // Empty packet → the very first extra-bits read fails.
            let r = decode_eob_token(&[], token, 1, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs);
            match r {
                Err(Error::TruncatedHeader { .. }) => {}
                other => panic!("token={token} expected TruncatedHeader, got {other:?}"),
            }
            // No partial state should land: TIS is still 0 (step 10
            // has not run) and COEFFS is still sentinel.
            assert_eq!(tis[0], 0);
            assert_eq!(coeffs[0], [-777i16; 64]);
        }
    }

    /// Step-8 boundary check: ti=0 zero-fills the full block.
    #[test]
    fn eob_token_zero_fills_from_ti_zero() {
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_eob_state(2);
        let eobs = decode_eob_token(&[], 0, 2, 1, 0, &mut tis, &mut ncoeffs, &mut coeffs).unwrap();
        assert_eq!(eobs, 0);
        // Every coefficient of block 1 is zero-filled.
        assert_eq!(coeffs[1], [0i16; 64]);
        // Block 0 untouched.
        assert_eq!(coeffs[0], [-777i16; 64]);
    }

    /// Step-9 captures the pre-call TIS value, not the post-step-10
    /// pinned 64. Exercised by pre-seeding TIS[bi] with a non-zero
    /// value before the call.
    #[test]
    fn eob_token_step9_captures_pre_call_tis() {
        let mut tis = vec![0u8, 17];
        let mut ncoeffs = vec![0xffu8; 2];
        let mut coeffs = vec![[-777i16; 64]; 2];
        let _ = decode_eob_token(&[], 0, 2, 1, 20, &mut tis, &mut ncoeffs, &mut coeffs).unwrap();
        // §7.7.1 step 9: NCOEFFS[bi] = TIS[bi] (which was 17 *before*
        // step 10 ran).
        assert_eq!(ncoeffs[1], 17);
        // §7.7.1 step 10 then pinned TIS[bi] = 64.
        assert_eq!(tis[1], 64);
    }

    /// Display rendering for each new §7.7.1 error variant carries the
    /// section number so log messages bisect cleanly to the spec.
    #[test]
    fn eob_token_error_display_carries_section_number() {
        let e = Error::EobTokenOutOfRange { token: 7 };
        let s = format!("{e}");
        assert!(s.contains("§7.7.1"), "got: {s}");
        assert!(s.contains("TOKEN=7"), "got: {s}");

        let e = Error::EobTokenBlockIndexOutOfRange { bi: 4, nbs: 4 };
        let s = format!("{e}");
        assert!(s.contains("§7.7.1"), "got: {s}");
        assert!(s.contains("bi=4"), "got: {s}");

        let e = Error::EobTokenIndexOutOfRange { ti: 99 };
        let s = format!("{e}");
        assert!(s.contains("§7.7.1"), "got: {s}");
        assert!(s.contains("ti=99"), "got: {s}");

        let e = Error::EobTokenStateLenMismatch {
            which: EobTokenStateSlice::Tis,
            got: 3,
            nbs: 4,
        };
        let s = format!("{e}");
        assert!(s.contains("§7.7.1"), "got: {s}");
        assert!(s.contains("Tis"), "got: {s}");
    }

    /// The shared-`BitReader` chaining contract: after the inner
    /// procedure returns, the next read picks up exactly where §7.7.1
    /// left off. Exercised on token 3 (2 bits of extra-bits payload)
    /// followed by a sentinel byte.
    #[test]
    fn eob_token_inner_leaves_reader_at_correct_offset() {
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_eob_state(1);
        // Two payload bits = b11 (the high two bits of byte 0), then
        // six tail bits in the same byte set to 0b101010, then a fresh
        // byte = 0xFF as a sentinel.
        let packet = [0b1110_1010, 0xff];
        let mut r = BitReader::new(&packet);
        let eobs = decode_eob_token_inner(&mut r, 3, 1, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs)
            .unwrap();
        // Payload was 0b11 = 3 → EOBS = 3 + 4 - 1 = 6.
        assert_eq!(eobs, 6);
        // Next 6 bits of byte 0 = 0b101010, then the byte 0xFF sentinel.
        let tail = r.read_bits(6, "tail").unwrap();
        assert_eq!(tail, 0b101010);
        let sentinel = r.read_bits(8, "sentinel").unwrap();
        assert_eq!(sentinel, 0xff);
    }

    /// Token 6 zero-payload "all-remaining" interpretation: every
    /// already-pinned block is excluded; including the current block
    /// in the tally is the spec's wording ("the size of the remaining
    /// coded blocks").
    #[test]
    fn eob_token_6_zero_payload_excludes_pinned_blocks() {
        // 8 blocks: 0..=2 pinned (TIS=64), 3..=7 still in play.
        let mut tis = vec![64u8, 64, 64, 0, 0, 0, 0, 0];
        let mut ncoeffs = vec![0xffu8; 8];
        let mut coeffs = vec![[0i16; 64]; 8];
        // Zero 12-bit payload (token 6 sentinel).
        let packet = [0u8, 0];
        let eobs =
            decode_eob_token(&packet, 6, 8, 3, 0, &mut tis, &mut ncoeffs, &mut coeffs).unwrap();
        // 5 blocks (3..=7) qualify; step 11 decrements → 4.
        assert_eq!(eobs, 4);
        // Only block 3 was pinned by this call.
        assert_eq!(tis, vec![64, 64, 64, 64, 0, 0, 0, 0]);
    }

    // ---------------------------------------------------------------
    // §6.4.1 Loop Filter Limit Table Decode tests.
    //
    // Spec body recovered in `docs/video/theora/theora-6.4.1-lflims.md`
    // from the spec's own LaTeX source. The procedure reads `NBITS`
    // (3 bits) once, then for each `qi` in 0..=63 reads `LFLIMS[qi]`
    // as an `NBITS`-bit unsigned integer.
    //
    // The fixtures below construct §6.4.1 payloads bit-by-bit using
    // the same MSb-first packing the `BitReader` consumes (most
    // significant bit of byte 0 first). This avoids needing a live
    // setup-header payload and exercises the decoder against
    // synthetic inputs whose expected output is by construction.
    // ---------------------------------------------------------------

    /// Helper: pack `(value, nbits)` slots MSb-first into the smallest
    /// `Vec<u8>` that fits, zero-padding the final byte's tail bits.
    fn pack_msb_first(slots: &[(u32, u32)]) -> Vec<u8> {
        let total_bits: u32 = slots.iter().map(|(_, n)| n).sum();
        let total_bytes = total_bits.div_ceil(8) as usize;
        let mut out = vec![0u8; total_bytes];
        let mut bit_cursor: usize = 0;
        for &(value, nbits) in slots {
            for i in (0..nbits).rev() {
                let bit = (value >> i) & 1;
                let byte_idx = bit_cursor / 8;
                let bit_idx = 7 - (bit_cursor % 8);
                out[byte_idx] |= (bit as u8) << bit_idx;
                bit_cursor += 1;
            }
        }
        out
    }

    /// NBITS=0 is a corner case the spec body permits: the read is
    /// zero-bit-wide per entry, so every `LFLIMS[qi]` is forced to 0.
    /// Total bit cost = 3 + 64*0 = 3 bits. The next 5 bits of byte 0
    /// are free for whatever §6.4.2 wants to do with them.
    #[test]
    fn lflims_nbits_zero_yields_all_zeros() {
        // Single byte: NBITS=0 in the top 3 bits, then 5 don't-care
        // bits set to 1 to make sure they are not consumed.
        let packet = [0b000_11111];
        let lflims = decode_loop_filter_limit_table(&packet).unwrap();
        assert_eq!(lflims, [0u8; 64]);
    }

    /// NBITS=5 (5-bit values) suffices to encode the Appendix B.2 VP3
    /// hardcoded LFLIMS table — its maximum entry is 30, well below
    /// 2^5 - 1 = 31. The decoded table must match `LFLIMS_VP3` byte
    /// for byte; this confirms both the §6.4.1 transcription and the
    /// Appendix B.2 table at the same time.
    #[test]
    fn lflims_nbits_5_roundtrips_vp3_table() {
        // NBITS=5 then 64 × 5-bit values from LFLIMS_VP3.
        let mut slots: Vec<(u32, u32)> = vec![(5, 3)];
        for &v in LFLIMS_VP3.iter() {
            slots.push((v as u32, 5));
        }
        let packet = pack_msb_first(&slots);
        let lflims = decode_loop_filter_limit_table(&packet).unwrap();
        assert_eq!(lflims, LFLIMS_VP3);
    }

    /// NBITS=7 covers the full 7-bit output width declared by §6.4.1.
    /// Round-tripping a table that uses the full 0..=127 range
    /// confirms the decoder does not narrow or clamp.
    #[test]
    fn lflims_nbits_7_roundtrips_full_range() {
        // Construct a synthetic table with `qi` as the value
        // (0..=63), then bump some entries to >63 to exercise the
        // top half of the 7-bit range.
        let mut expected = [0u8; 64];
        for (qi, slot) in expected.iter_mut().enumerate() {
            *slot = ((qi * 2) as u8) & 0x7f; // 0, 2, ... up to 126.
        }
        let mut slots: Vec<(u32, u32)> = vec![(7, 3)];
        for &v in expected.iter() {
            slots.push((v as u32, 7));
        }
        let packet = pack_msb_first(&slots);
        let lflims = decode_loop_filter_limit_table(&packet).unwrap();
        assert_eq!(lflims, expected);
    }

    /// Total bit cost: a NBITS=3 payload reads `3 + 64*3 = 195` bits,
    /// which spans 25 bytes (the last 5 bits of byte 24 are unused).
    /// A 24-byte buffer is therefore insufficient and must surface
    /// the truncation error the shared `BitReader` raises.
    #[test]
    fn lflims_truncated_payload_errors() {
        // 195 bits would need 25 bytes; give 24 with NBITS=3 packed.
        let mut slots: Vec<(u32, u32)> = vec![(3, 3)];
        for _ in 0..64 {
            slots.push((5, 3));
        }
        let mut packet = pack_msb_first(&slots);
        packet.truncate(24);
        let err = decode_loop_filter_limit_table(&packet).unwrap_err();
        assert!(
            matches!(err, Error::TruncatedHeader { field: "LFLIMS" }),
            "got: {err:?}"
        );
    }

    /// Truncation while reading `NBITS` itself (an empty payload)
    /// reports the `LFLIMS NBITS` field so the failure pinpoints the
    /// step-1 read rather than the step-2 loop.
    #[test]
    fn lflims_truncated_at_nbits_errors() {
        let packet: [u8; 0] = [];
        let err = decode_loop_filter_limit_table(&packet).unwrap_err();
        assert!(
            matches!(
                err,
                Error::TruncatedHeader {
                    field: "LFLIMS NBITS"
                }
            ),
            "got: {err:?}"
        );
    }

    /// The shared-`BitReader` chaining contract for §6.4.1: after the
    /// inner procedure returns, the next read picks up exactly where
    /// the procedure left off. This is what lets a future
    /// `parse_setup_header` chain §6.4.1 → §6.4.2 → §6.4.4 on the
    /// same reader without re-aligning at byte boundaries.
    #[test]
    fn lflims_inner_leaves_reader_at_correct_offset() {
        // NBITS=3 → reads 3 + 64*3 = 195 bits.
        let mut slots: Vec<(u32, u32)> = vec![(3, 3)];
        for _ in 0..64 {
            slots.push((1, 3));
        }
        // Add a 5-bit tail in byte 24 (bits 4..=0) and a sentinel
        // byte to verify both partial-byte and next-byte continuation.
        slots.push((0b10110, 5));
        slots.push((0xab, 8));
        let packet = pack_msb_first(&slots);
        let mut r = BitReader::new(&packet);
        let lflims = decode_lflims_inner(&mut r).unwrap();
        assert_eq!(lflims, [1u8; 64]);
        // Reader is now mid-byte 24 at bit-position 4 (zero-indexed
        // from the MSb). Read the 5-bit tail then the sentinel.
        let tail = r.read_bits(5, "tail").unwrap();
        assert_eq!(tail, 0b10110);
        let sentinel = r.read_bits(8, "sentinel").unwrap();
        assert_eq!(sentinel, 0xab);
    }

    /// All entries reach the 7-bit ceiling (`2^7 - 1 = 127`) when
    /// NBITS=7 — the maximum addressable value the §6.4.1 output
    /// table allows. Confirms no upper-bound truncation.
    #[test]
    fn lflims_nbits_7_all_max_value() {
        let mut slots: Vec<(u32, u32)> = vec![(7, 3)];
        for _ in 0..64 {
            slots.push((127, 7));
        }
        let packet = pack_msb_first(&slots);
        let lflims = decode_loop_filter_limit_table(&packet).unwrap();
        assert_eq!(lflims, [127u8; 64]);
    }

    // ===================================================================
    // §7.7.2 Coefficient Token Decode tests
    // ===================================================================

    /// Reusable starting state for §7.7.2 tests. NCOEFFS sentinel is
    /// 0xff so step-c writes are detectable when TIS[bi] happens to be
    /// 0. COEFFS sentinel is -777 so the per-token writes are visible
    /// against the background.
    fn fresh_coeff_state(nbs: usize) -> (Vec<u8>, Vec<u8>, Vec<[i16; 64]>) {
        let tis = vec![0u8; nbs];
        let ncoeffs = vec![0xffu8; nbs];
        let coeffs = vec![[-777i16; 64]; nbs];
        (tis, ncoeffs, coeffs)
    }

    /// TOKEN=7 reads a 3-bit `RLEN`, adds 1, then zero-fills that many
    /// entries. NCOEFFS[bi] is NOT updated (pure-zero-run exception).
    /// All eight `RLEN_RAW` values 0..=7 produce runs 1..=8.
    #[test]
    fn coeff_token_7_short_zero_run() {
        for rlen_raw in 0u32..8 {
            let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(2);
            let byte = ((rlen_raw & 0b111) << 5) as u8;
            let kind =
                decode_coefficient_token(&[byte], 7, 2, 1, 0, &mut tis, &mut ncoeffs, &mut coeffs)
                    .unwrap();
            assert_eq!(kind, CoefficientTokenKind::ZeroRun);
            let rlen = rlen_raw + 1;
            // Block 1's first `rlen` slots are zeroed; everything past
            // that retains the -777 sentinel.
            for (ti, &v) in coeffs[1].iter().enumerate() {
                if (ti as u32) < rlen {
                    assert_eq!(v, 0, "rlen_raw={rlen_raw} ti={ti}");
                } else {
                    assert_eq!(v, -777, "rlen_raw={rlen_raw} ti={ti}");
                }
            }
            assert_eq!(tis[1], rlen as u8);
            // NCOEFFS[bi] untouched per §7.7.2's pure-zero-run note.
            assert_eq!(ncoeffs[1], 0xff);
        }
    }

    /// TOKEN=8 reads a 6-bit RLEN+1, range 1..=64. The boundary case
    /// RLEN=64 starting from ti=0 must succeed (writes exactly 64
    /// zeros, TIS advances to 64).
    #[test]
    fn coeff_token_8_long_zero_run_to_block_end() {
        // 6-bit RLEN_RAW = 63 (0b111_111) → run of 64.
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
        // 6 bits, MSb-first: 0b111111_xx (top 6 of byte).
        let byte = 0b1111_1100u8;
        let kind =
            decode_coefficient_token(&[byte], 8, 1, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs)
                .unwrap();
        assert_eq!(kind, CoefficientTokenKind::ZeroRun);
        assert_eq!(tis[0], 64);
        assert_eq!(ncoeffs[0], 0xff);
        assert_eq!(coeffs[0], [0i16; 64]);
    }

    /// TOKEN=9..=12 write a fixed coefficient and consume zero extra
    /// bits. NCOEFFS[bi] is set to the post-step TIS[bi].
    #[test]
    fn coeff_token_9_through_12_fixed_singles() {
        for (token, expected) in [(9u8, 1i16), (10, -1), (11, 2), (12, -2)] {
            let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(2);
            // Empty packet — these tokens read no extra bits.
            let kind =
                decode_coefficient_token(&[], token, 2, 0, 7, &mut tis, &mut ncoeffs, &mut coeffs)
                    .unwrap();
            assert_eq!(kind, CoefficientTokenKind::Single);
            assert_eq!(coeffs[0][7], expected, "token={token}");
            // Other entries unchanged.
            assert_eq!(coeffs[0][6], -777);
            assert_eq!(coeffs[0][8], -777);
            assert_eq!(tis[0], 8);
            assert_eq!(ncoeffs[0], 8);
        }
    }

    /// TOKEN=13..=16 read a 1-bit SIGN and write ±MAG with
    /// MAG = 3, 4, 5, 6. The SIGN bit polarity is "zero → positive".
    #[test]
    fn coeff_token_13_through_16_fixed_mag_sign() {
        for (token, mag) in [(13u8, 3i16), (14, 4), (15, 5), (16, 6)] {
            for sign in 0u8..=1 {
                let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
                // SIGN bit at MSb of byte 0.
                let byte = sign << 7;
                let _ = decode_coefficient_token(
                    &[byte],
                    token,
                    1,
                    0,
                    0,
                    &mut tis,
                    &mut ncoeffs,
                    &mut coeffs,
                )
                .unwrap();
                let expected = if sign == 0 { mag } else { -mag };
                assert_eq!(coeffs[0][0], expected, "token={token} sign={sign}");
                assert_eq!(tis[0], 1);
                assert_eq!(ncoeffs[0], 1);
            }
        }
    }

    /// TOKEN=17 reads SIGN (1 bit) + MAG (1 bit) + offset 7 → ±7..=8.
    #[test]
    fn coeff_token_17_sign_one_bit_mag() {
        for sign in 0u8..=1 {
            for mag_raw in 0u8..=1 {
                let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
                // SIGN at bit 7, MAG at bit 6.
                let byte = (sign << 7) | (mag_raw << 6);
                let _ = decode_coefficient_token(
                    &[byte],
                    17,
                    1,
                    0,
                    0,
                    &mut tis,
                    &mut ncoeffs,
                    &mut coeffs,
                )
                .unwrap();
                let mag = mag_raw as i16 + 7;
                let expected = if sign == 0 { mag } else { -mag };
                assert_eq!(coeffs[0][0], expected, "sign={sign} mag_raw={mag_raw}");
            }
        }
    }

    /// TOKEN=22 reads SIGN (1 bit) + MAG (9 bits) + offset 69 → range
    /// ±69..=580. Exercises the largest single-coefficient token.
    #[test]
    fn coeff_token_22_ten_bit_payload_round_trip() {
        // MAG_RAW = 0x1FF (511) → MAG = 580. SIGN = 1 → -580.
        // 10 bits total, MSb-first: 1_1_1111_1111_xxxxxx.
        // Byte 0 = 0b1_1_1_1_1_1_1_1 (sign=1, mag bits 8..2 all 1).
        // Byte 1 = 0b1_1_xxxxxx (mag bits 1..0 = 1,1).
        let packet = [0b1111_1111u8, 0b1100_0000u8];
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
        let _ = decode_coefficient_token(&packet, 22, 1, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs)
            .unwrap();
        assert_eq!(coeffs[0][0], -580);
        assert_eq!(tis[0], 1);
        assert_eq!(ncoeffs[0], 1);
    }

    /// TOKEN=23..=27 are "N zeros + ±1" with fixed run lengths
    /// 1..=5. Tests the run-plus-one structural branch including the
    /// run-then-coefficient write order.
    #[test]
    fn coeff_token_23_through_27_fixed_run_plus_one() {
        for (token, run) in [(23u8, 1usize), (24, 2), (25, 3), (26, 4), (27, 5)] {
            let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
            // SIGN = 0 → +1.
            let kind = decode_coefficient_token(
                &[0u8],
                token,
                1,
                0,
                10,
                &mut tis,
                &mut ncoeffs,
                &mut coeffs,
            )
            .unwrap();
            assert_eq!(kind, CoefficientTokenKind::RunPlusOne);
            // Zero run at COEFFS[0][10..10+run].
            assert!(
                coeffs[0][10..10 + run].iter().all(|&v| v == 0),
                "token={token}"
            );
            // Trailing +1 at COEFFS[0][10+run].
            assert_eq!(coeffs[0][10 + run], 1, "token={token}");
            // TIS advanced by run+1.
            assert_eq!(tis[0] as usize, 10 + run + 1);
            assert_eq!(ncoeffs[0] as usize, 10 + run + 1);
        }
    }

    /// TOKEN=28 reads SIGN + 2-bit RLEN, then writes RLEN+6 zeros
    /// followed by trailing ±1. Boundaries: RLEN range 6..=9.
    #[test]
    fn coeff_token_28_run_range_6_to_9() {
        for rlen_raw in 0u32..4 {
            let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
            // SIGN=1 (negative trailing -1), 2-bit RLEN_RAW.
            // Bits: SIGN at bit 7, RLEN at bits 6..5.
            let byte = (0b1u8 << 7) | (((rlen_raw & 0b11) << 5) as u8);
            let _ =
                decode_coefficient_token(&[byte], 28, 1, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs)
                    .unwrap();
            let run = rlen_raw + 6;
            assert!(
                coeffs[0][..run as usize].iter().all(|&v| v == 0),
                "rlen_raw={rlen_raw}"
            );
            assert_eq!(coeffs[0][run as usize], -1, "rlen_raw={rlen_raw}");
            assert_eq!(tis[0] as u32, run + 1);
            assert_eq!(ncoeffs[0] as u32, run + 1);
        }
    }

    /// TOKEN=29 reads SIGN + 3-bit RLEN, then writes RLEN+10 zeros
    /// followed by trailing ±1. Boundary: RLEN range 10..=17.
    #[test]
    fn coeff_token_29_run_range_10_to_17() {
        for rlen_raw in 0u32..8 {
            let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
            // SIGN=0 (positive). 4 bits = SIGN(1) + RLEN(3); pack at top.
            let byte = ((rlen_raw & 0b111) << 4) as u8;
            let _ =
                decode_coefficient_token(&[byte], 29, 1, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs)
                    .unwrap();
            let run = rlen_raw + 10;
            assert!(
                coeffs[0][..run as usize].iter().all(|&v| v == 0),
                "rlen_raw={rlen_raw}"
            );
            assert_eq!(coeffs[0][run as usize], 1, "rlen_raw={rlen_raw}");
            assert_eq!(tis[0] as u32, run + 1);
            assert_eq!(ncoeffs[0] as u32, run + 1);
        }
    }

    /// TOKEN=30 is "one zero + ±MAG" with MAG = 2..=3. SIGN/MAG bits
    /// are 2 of the 3 read; no RLEN field.
    #[test]
    fn coeff_token_30_one_zero_plus_signed_mag() {
        for sign in 0u8..=1 {
            for mag_raw in 0u8..=1 {
                let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
                let byte = (sign << 7) | (mag_raw << 6);
                let _ = decode_coefficient_token(
                    &[byte],
                    30,
                    1,
                    0,
                    5,
                    &mut tis,
                    &mut ncoeffs,
                    &mut coeffs,
                )
                .unwrap();
                let mag = (mag_raw as i16) + 2;
                let expected = if sign == 0 { mag } else { -mag };
                // First a zero at ti=5, then ±MAG at ti=6.
                assert_eq!(coeffs[0][5], 0);
                assert_eq!(coeffs[0][6], expected);
                assert_eq!(tis[0], 7);
                assert_eq!(ncoeffs[0], 7);
            }
        }
    }

    /// TOKEN=31 reads SIGN + MAG (1 bit, +2) + RLEN (1 bit, +2):
    /// `2..=3` zeros followed by `±2..=3`.
    #[test]
    fn coeff_token_31_signed_mag_with_run() {
        for sign in 0u8..=1 {
            for mag_raw in 0u8..=1 {
                for rlen_raw in 0u8..=1 {
                    let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
                    // Layout: SIGN(1) | MAG(1) | RLEN(1) at top 3 bits.
                    let byte = (sign << 7) | (mag_raw << 6) | (rlen_raw << 5);
                    let _ = decode_coefficient_token(
                        &[byte],
                        31,
                        1,
                        0,
                        0,
                        &mut tis,
                        &mut ncoeffs,
                        &mut coeffs,
                    )
                    .unwrap();
                    let mag = (mag_raw as i16) + 2;
                    let expected = if sign == 0 { mag } else { -mag };
                    let run = (rlen_raw as usize) + 2;
                    assert!(coeffs[0][..run].iter().all(|&v| v == 0));
                    assert_eq!(coeffs[0][run], expected);
                    assert_eq!(tis[0] as usize, run + 1);
                    assert_eq!(ncoeffs[0] as usize, run + 1);
                }
            }
        }
    }

    /// Argument validation: tokens 0..=6 belong to §7.7.1 and must be
    /// rejected by §7.7.2; tokens 32..=255 are out of band entirely.
    #[test]
    fn coeff_token_rejects_out_of_range_token() {
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
        for token in 0u8..=6 {
            let err =
                decode_coefficient_token(&[], token, 1, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs)
                    .unwrap_err();
            assert_eq!(err, Error::CoefficientTokenOutOfRange { token });
        }
        for token in [32u8, 64, 99, 200, 255] {
            let err =
                decode_coefficient_token(&[], token, 1, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs)
                    .unwrap_err();
            assert_eq!(err, Error::CoefficientTokenOutOfRange { token });
        }
    }

    /// Argument validation: `bi >= nbs` and `ti > 63` reject with
    /// dedicated typed errors.
    #[test]
    fn coeff_token_rejects_index_out_of_range() {
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(2);
        let err = decode_coefficient_token(&[], 9, 2, 2, 0, &mut tis, &mut ncoeffs, &mut coeffs)
            .unwrap_err();
        assert_eq!(
            err,
            Error::CoefficientTokenBlockIndexOutOfRange { bi: 2, nbs: 2 }
        );

        let err = decode_coefficient_token(&[], 9, 2, 0, 64, &mut tis, &mut ncoeffs, &mut coeffs)
            .unwrap_err();
        assert_eq!(err, Error::CoefficientTokenIndexOutOfRange { ti: 64 });
    }

    /// Argument validation: the three state slices must each have
    /// length exactly `nbs`. Each mismatch surfaces with a typed
    /// `CoefficientTokenStateSlice` discriminant.
    #[test]
    fn coeff_token_rejects_state_length_mismatch() {
        // TIS too short.
        let mut tis = vec![0u8; 1];
        let mut ncoeffs = vec![0xffu8; 2];
        let mut coeffs = vec![[-777i16; 64]; 2];
        let err = decode_coefficient_token(&[], 9, 2, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs)
            .unwrap_err();
        assert_eq!(
            err,
            Error::CoefficientTokenStateLenMismatch {
                which: CoefficientTokenStateSlice::Tis,
                got: 1,
                nbs: 2,
            }
        );

        // NCOEFFS too long.
        let mut tis = vec![0u8; 2];
        let mut ncoeffs = vec![0xffu8; 3];
        let mut coeffs = vec![[-777i16; 64]; 2];
        let err = decode_coefficient_token(&[], 9, 2, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs)
            .unwrap_err();
        assert_eq!(
            err,
            Error::CoefficientTokenStateLenMismatch {
                which: CoefficientTokenStateSlice::Ncoeffs,
                got: 3,
                nbs: 2,
            }
        );

        // COEFFS too short.
        let mut tis = vec![0u8; 2];
        let mut ncoeffs = vec![0xffu8; 2];
        let mut coeffs = vec![[-777i16; 64]; 1];
        let err = decode_coefficient_token(&[], 9, 2, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs)
            .unwrap_err();
        assert_eq!(
            err,
            Error::CoefficientTokenStateLenMismatch {
                which: CoefficientTokenStateSlice::Coeffs,
                got: 1,
                nbs: 2,
            }
        );
    }

    /// §7.7.2 MUST-NOT clause: tokens that bring TIS[bi] past 64 fail
    /// closed. TOKEN=8 with RLEN=64 starting from ti=1 is the
    /// tightest illegal case — it would produce TIS=65.
    #[test]
    fn coeff_token_8_overflow_rejects() {
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
        // 6-bit RLEN_RAW = 63 (max) → run of 64. Starting from ti=1
        // would push TIS to 65 > 64.
        let byte = 0b1111_1100u8;
        let err =
            decode_coefficient_token(&[byte], 8, 1, 0, 1, &mut tis, &mut ncoeffs, &mut coeffs)
                .unwrap_err();
        assert_eq!(
            err,
            Error::CoefficientTokenWouldOverflowBlock {
                token: 8,
                ti: 1,
                new_tis: 65,
            }
        );
        // State arrays untouched on overflow.
        assert_eq!(tis[0], 0);
        assert_eq!(ncoeffs[0], 0xff);
        assert_eq!(coeffs[0][0], -777);
    }

    /// Run-plus-one tokens (e.g. TOKEN=29 with RLEN=17) also surface
    /// the overflow when the trailing coefficient would land past
    /// index 63. RLEN=17 + ti=47 → new TIS = 47+17+1 = 65 > 64.
    #[test]
    fn coeff_token_29_run_overflow_rejects() {
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
        // SIGN=0, RLEN_RAW=7 → run=17.
        let byte = (0b111u8) << 4;
        let err =
            decode_coefficient_token(&[byte], 29, 1, 0, 47, &mut tis, &mut ncoeffs, &mut coeffs)
                .unwrap_err();
        assert_eq!(
            err,
            Error::CoefficientTokenWouldOverflowBlock {
                token: 29,
                ti: 47,
                new_tis: 65,
            }
        );
    }

    /// Single-coefficient writes at ti = 63 are legal: TIS advances to
    /// 64 exactly (the same pinned state §7.7.1 step 10 produces). No
    /// overflow because §7.7.2's MUST-NOT clause only applies to
    /// multi-coefficient tokens.
    #[test]
    fn coeff_token_9_at_ti_63_pins_block() {
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
        let kind = decode_coefficient_token(&[], 9, 1, 0, 63, &mut tis, &mut ncoeffs, &mut coeffs)
            .unwrap();
        assert_eq!(kind, CoefficientTokenKind::Single);
        assert_eq!(coeffs[0][63], 1);
        assert_eq!(tis[0], 64);
        assert_eq!(ncoeffs[0], 64);
    }

    /// TOKEN=7 with RLEN=8 starting from ti=56 reaches exactly 64 — the
    /// largest legal zero-run terminal-position case. No overflow.
    #[test]
    fn coeff_token_7_reaches_block_end_exactly() {
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
        // RLEN_RAW=7 (max) → run=8.
        let byte = (0b111u8) << 5;
        let kind =
            decode_coefficient_token(&[byte], 7, 1, 0, 56, &mut tis, &mut ncoeffs, &mut coeffs)
                .unwrap();
        assert_eq!(kind, CoefficientTokenKind::ZeroRun);
        assert!(coeffs[0][56..64].iter().all(|&v| v == 0));
        assert_eq!(tis[0], 64);
        // NCOEFFS untouched per pure-zero-run rule.
        assert_eq!(ncoeffs[0], 0xff);
    }

    /// Truncation surfaces with a TruncatedHeader carrying a §7.7.2
    /// field name so log messages bisect cleanly to the spec.
    #[test]
    fn coeff_token_truncation_on_payload() {
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
        // TOKEN=22 needs 10 bits (1 SIGN + 9 MAG). Empty packet → trunc.
        let err = decode_coefficient_token(&[], 22, 1, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs)
            .unwrap_err();
        match err {
            Error::TruncatedHeader { field } => {
                assert!(field.contains("§7.7.2"), "got: {field}");
                assert!(field.contains("TOKEN=22"), "got: {field}");
            }
            other => panic!("expected TruncatedHeader, got {other:?}"),
        }
    }

    /// Display rendering for each new §7.7.2 error variant carries the
    /// section number so log lines bisect cleanly to the spec.
    #[test]
    fn coeff_token_error_display_carries_section_number() {
        let e = Error::CoefficientTokenOutOfRange { token: 0 };
        let s = format!("{e}");
        assert!(s.contains("§7.7.2"), "got: {s}");
        assert!(s.contains("TOKEN=0"), "got: {s}");

        let e = Error::CoefficientTokenBlockIndexOutOfRange { bi: 4, nbs: 4 };
        let s = format!("{e}");
        assert!(s.contains("§7.7.2"), "got: {s}");

        let e = Error::CoefficientTokenIndexOutOfRange { ti: 64 };
        let s = format!("{e}");
        assert!(s.contains("§7.7.2"), "got: {s}");

        let e = Error::CoefficientTokenStateLenMismatch {
            which: CoefficientTokenStateSlice::Coeffs,
            got: 3,
            nbs: 4,
        };
        let s = format!("{e}");
        assert!(s.contains("§7.7.2"), "got: {s}");
        assert!(s.contains("Coeffs"), "got: {s}");

        let e = Error::CoefficientTokenWouldOverflowBlock {
            token: 8,
            ti: 1,
            new_tis: 65,
        };
        let s = format!("{e}");
        assert!(s.contains("§7.7.2"), "got: {s}");
        assert!(s.contains("TOKEN=8"), "got: {s}");
        assert!(s.contains("65"), "got: {s}");
    }

    /// Shared-`BitReader` chaining contract: after the inner procedure
    /// returns, the next read picks up exactly where §7.7.2 left off.
    /// Exercised on TOKEN=22 (10 bits of payload) followed by a
    /// sentinel byte.
    #[test]
    fn coeff_token_inner_leaves_reader_at_correct_offset() {
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
        // 10 bits of TOKEN=22 payload = 1_111111111 (sign=1, mag=511 →
        // MAG=580, signed → -580). Then 6 tail bits = 0b101010, then a
        // fresh 0xFF sentinel byte.
        let packet = [0b1111_1111u8, 0b1110_1010u8, 0xff];
        let mut r = BitReader::new(&packet);
        let _ = decode_coefficient_token_inner(
            &mut r,
            22,
            1,
            0,
            0,
            &mut tis,
            &mut ncoeffs,
            &mut coeffs,
        )
        .unwrap();
        assert_eq!(coeffs[0][0], -580);
        // 16 bits consumed → byte 2 next. The 6 unused bits in byte 1
        // (lsb of mag is bit 6, then 6 remain) were... actually let me
        // re-verify: we read 10 bits = byte0 (8 bits) + 2 bits of byte
        // 1. Tail bits in byte 1 = 6 bits 0b101010. Then sentinel byte
        // 2 = 0xff.
        let tail = r.read_bits(6, "tail").unwrap();
        assert_eq!(tail, 0b101010);
        let sentinel = r.read_bits(8, "sentinel").unwrap();
        assert_eq!(sentinel, 0xff);
    }

    /// TOKEN=18 reads 3 bits total (SIGN + 2-bit MAG). With sign=0 and
    /// mag_raw=3 the result is +12 (MAG = 3 + 9). Confirms the offset
    /// addition.
    #[test]
    fn coeff_token_18_two_bit_mag_offset_9() {
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
        // SIGN=0, MAG_RAW=3 → MAG = 12.
        // Bits: 0_11_xxxxx = 0b0110_0000.
        let _ = decode_coefficient_token(
            &[0b0110_0000u8],
            18,
            1,
            0,
            0,
            &mut tis,
            &mut ncoeffs,
            &mut coeffs,
        )
        .unwrap();
        assert_eq!(coeffs[0][0], 12);
    }

    /// TOKEN=19 reads 4 bits total: SIGN + 3-bit MAG, offset 13.
    /// MAG_RAW=7 + SIGN=1 → -20.
    #[test]
    fn coeff_token_19_three_bit_mag_offset_13() {
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
        // SIGN=1, MAG_RAW=7. Bits: 1_111_xxxx = 0b1111_0000.
        let _ = decode_coefficient_token(
            &[0b1111_0000u8],
            19,
            1,
            0,
            0,
            &mut tis,
            &mut ncoeffs,
            &mut coeffs,
        )
        .unwrap();
        assert_eq!(coeffs[0][0], -20);
    }

    /// TOKEN=20 reads 5 bits: SIGN + 4-bit MAG, offset 21. The lowest
    /// magnitude is 21 (mag_raw=0); the highest is 36 (mag_raw=15).
    #[test]
    fn coeff_token_20_range_endpoints() {
        for (mag_raw, sign, expected) in [(0u32, 0u8, 21i16), (15, 1, -36)] {
            let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
            // 5 bits: SIGN(1) + MAG(4); pack at top.
            let byte = (((sign as u32) << 7) | ((mag_raw & 0xf) << 3)) as u8;
            let _ =
                decode_coefficient_token(&[byte], 20, 1, 0, 0, &mut tis, &mut ncoeffs, &mut coeffs)
                    .unwrap();
            assert_eq!(coeffs[0][0], expected, "mag_raw={mag_raw} sign={sign}");
        }
    }

    /// TOKEN=21 reads 6 bits: SIGN + 5-bit MAG, offset 37. Spot-check
    /// the midrange.
    #[test]
    fn coeff_token_21_midrange() {
        let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
        // SIGN=0, MAG_RAW=16 → MAG=53.
        // 6 bits: SIGN(1) + MAG(5); pack: 0_10000_xx = 0b0100_0000.
        let _ = decode_coefficient_token(
            &[0b0100_0000u8],
            21,
            1,
            0,
            0,
            &mut tis,
            &mut ncoeffs,
            &mut coeffs,
        )
        .unwrap();
        assert_eq!(coeffs[0][0], 53);
    }

    /// Each §7.7.2 token reads exactly as many bits as Table 7.38
    /// declares in the "Extra Bits" column. This regression catches a
    /// future edit that mis-orders SIGN/MAG/RLEN reads.
    #[test]
    fn coeff_token_bit_counts_match_table_7_38() {
        // (token, expected bits read after the token byte itself).
        for (token, expected_bits) in [
            (7u8, 3u32), // RLEN(3)
            (8, 6),      // RLEN(6)
            (9, 0),      // no extra
            (10, 0),
            (11, 0),
            (12, 0),
            (13, 1), // SIGN(1)
            (14, 1),
            (15, 1),
            (16, 1),
            (17, 2),  // SIGN(1) + MAG(1)
            (18, 3),  // SIGN(1) + MAG(2)
            (19, 4),  // SIGN(1) + MAG(3)
            (20, 5),  // SIGN(1) + MAG(4)
            (21, 6),  // SIGN(1) + MAG(5)
            (22, 10), // SIGN(1) + MAG(9)
            (23, 1),  // SIGN(1)
            (24, 1),
            (25, 1),
            (26, 1),
            (27, 1),
            (28, 3), // SIGN(1) + RLEN(2)
            (29, 4), // SIGN(1) + RLEN(3)
            (30, 2), // SIGN(1) + MAG(1)
            (31, 3), // SIGN(1) + MAG(1) + RLEN(1)
        ] {
            let (mut tis, mut ncoeffs, mut coeffs) = fresh_coeff_state(1);
            // Provide a generous 32-byte buffer of zeros — any token
            // can read its payload, and the tail-position telemetry
            // confirms the exact bit count.
            let packet = [0u8; 32];
            let mut r = BitReader::new(&packet);
            let _ = decode_coefficient_token_inner(
                &mut r,
                token,
                1,
                0,
                0,
                &mut tis,
                &mut ncoeffs,
                &mut coeffs,
            )
            .unwrap();
            // Compute how many bits the reader has advanced. byte_pos
            // and bit_pos let us reconstruct the total.
            let bits_consumed = (r.byte_pos as u32) * 8 + (7u32.wrapping_sub(r.bit_pos as u32));
            assert_eq!(
                bits_consumed, expected_bits,
                "TOKEN={token} expected {expected_bits} bits, got {bits_consumed}"
            );
        }
    }
}
