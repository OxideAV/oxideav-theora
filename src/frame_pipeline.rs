//! Throughput-oriented §7.11 frame-decode pipeline.
//!
//! The procedures in the crate root follow the specification text step
//! by step (one function per numbered procedure, every intermediate
//! array materialised). This module computes exactly the same frame
//! with the bookkeeping reorganised for speed:
//!
//! * **§7.7 tokens** — each of the 80 §6.4.4 codebooks gets a
//!   prefix-indexed lookup table, so a token costs one table read
//!   instead of a bit-by-bit tree walk, and the token's extra bits come
//!   out of the same 32-bit peek. The §7.7.3 `ti` passes iterate a
//!   shrinking list of still-open blocks rather than every block of the
//!   frame, and coefficients land directly in natural (row-major)
//!   order in a buffer that is kept all-zero between frames.
//! * **§7.8 DC prediction** — one raster sweep per plane over two
//!   block-row buffers instead of per-block neighbour-table lookups.
//! * **§7.9 reconstruction** — the §6.4.3 quantisation matrices are
//!   evaluated once per frame (at most 18 of them), the dequantiser is
//!   a 64-lane multiply, the §7.9.3 inverse DCT runs all eight
//!   rows / columns as lanes and drops the upper four inputs when the
//!   block's coefficient count proves them zero, and predictor rows
//!   are read straight from the reference plane.
//! * **§7.10 loop filter** — the same edge order as §7.10.3 on row
//!   slices, with the limit function in closed form.
//! * **§7.11 steps 7–8** — the new frame is reconstructed into a spare
//!   plane set that is then *swapped* with the previous-frame
//!   reference; only keyframes pay one copy (into the golden slot).
//!
//! Reconstruction has no intra-frame dependency once §7.8 has run, so
//! under a caller-granted thread budget the frame is cut into bands of
//! whole super-block rows that are reconstructed and loop-filtered
//! concurrently. The loop filter *is* order dependent, but only through
//! the four pixel rows straddling a band boundary; each band replays
//! the neighbouring band's operations on a private copy of those rows
//! (see [`filter_band`]), which makes the output identical to the
//! serial order at every budget.

use std::sync::{Barrier, Mutex};

use super::{
    BitReader, Error, FrameGeometry, FrameType, HuffmanNode, HuffmanTable, MacroBlockMode,
    MotionVector, PixelFormat, QuantizationParameters, ReadTokenErr, ReferenceFrameStore,
    TABLE_7_47, ZIGZAG_NATURAL_TO_ZIGZAG, compute_quantization_matrix,
};

// ---------------------------------------------------------------------
// §7.7 token lookup tables
// ---------------------------------------------------------------------

/// Number of leading code bits resolved by one table read.
const LUT_BITS: u32 = 9;
const LUT_SIZE: usize = 1 << LUT_BITS;
/// Table entry for "no code of at most [`LUT_BITS`] bits has this
/// prefix": the caller falls back to the tree walk.
const LUT_MISS: u16 = u16::MAX;

/// Prefix lookup tables for the 80 §6.4.4 codebooks. Entry layout:
/// `token | (code_length << 5)`.
#[derive(Debug, Clone)]
pub(crate) struct TokenLuts {
    tables: Vec<[u16; LUT_SIZE]>,
}

impl TokenLuts {
    pub(crate) fn build(hts: &[HuffmanTable]) -> Self {
        let mut tables = vec![[LUT_MISS; LUT_SIZE]; hts.len()];
        for (lut, table) in tables.iter_mut().zip(hts) {
            // Depth-first over the decode tree, bounded at LUT_BITS:
            // a leaf at depth d owns every index sharing its d-bit
            // prefix.
            let mut stack: Vec<(u32, u32, u32)> = vec![(0, 0, 0)];
            while let Some((idx, depth, prefix)) = stack.pop() {
                match table.nodes.get(idx as usize) {
                    None => {}
                    Some(HuffmanNode::Leaf { token }) => {
                        let shift = LUT_BITS - depth;
                        let base = (prefix << shift) as usize;
                        let entry = u16::from(*token) | ((depth as u16) << 5);
                        lut[base..base + (1usize << shift)].fill(entry);
                    }
                    Some(HuffmanNode::Branch { zero, one }) if depth < LUT_BITS => {
                        stack.push((*one, depth + 1, (prefix << 1) | 1));
                        stack.push((*zero, depth + 1, prefix << 1));
                    }
                    // A branch at the depth bound: left to the walk.
                    Some(HuffmanNode::Branch { .. }) => {}
                }
            }
        }
        Self { tables }
    }
}

/// How a §7.7.1 / §7.7.2 token expands (Tables 7.33 and 7.38). The
/// extra bits are laid out sign first, then magnitude, then run.
#[derive(Clone, Copy)]
struct TokenShape {
    /// Total number of extra bits.
    extra: u8,
    /// 0 = EOB run, 1 = pure zero run, 2 = (zero run then) value.
    kind: u8,
    run_bits: u8,
    run_base: u8,
    mag_bits: u8,
    mag_base: u8,
    /// A sign bit leads the extra bits.
    signed: bool,
    /// Fixed sign for the tokens without a sign bit.
    negative: bool,
}

const KIND_EOB: u8 = 0;
const KIND_ZERO_RUN: u8 = 1;
const KIND_VALUE: u8 = 2;

const fn token_shape(token: u8) -> TokenShape {
    // (kind, run_bits, run_base, mag_bits, mag_base, signed, negative)
    let (kind, run_bits, run_base, mag_bits, mag_base, signed, negative) = match token {
        0 => (KIND_EOB, 0, 1, 0, 0, false, false),
        1 => (KIND_EOB, 0, 2, 0, 0, false, false),
        2 => (KIND_EOB, 0, 3, 0, 0, false, false),
        3 => (KIND_EOB, 2, 4, 0, 0, false, false),
        4 => (KIND_EOB, 3, 8, 0, 0, false, false),
        5 => (KIND_EOB, 4, 16, 0, 0, false, false),
        6 => (KIND_EOB, 12, 0, 0, 0, false, false),
        7 => (KIND_ZERO_RUN, 3, 1, 0, 0, false, false),
        8 => (KIND_ZERO_RUN, 6, 1, 0, 0, false, false),
        9 => (KIND_VALUE, 0, 0, 0, 1, false, false),
        10 => (KIND_VALUE, 0, 0, 0, 1, false, true),
        11 => (KIND_VALUE, 0, 0, 0, 2, false, false),
        12 => (KIND_VALUE, 0, 0, 0, 2, false, true),
        13 => (KIND_VALUE, 0, 0, 0, 3, true, false),
        14 => (KIND_VALUE, 0, 0, 0, 4, true, false),
        15 => (KIND_VALUE, 0, 0, 0, 5, true, false),
        16 => (KIND_VALUE, 0, 0, 0, 6, true, false),
        17 => (KIND_VALUE, 0, 0, 1, 7, true, false),
        18 => (KIND_VALUE, 0, 0, 2, 9, true, false),
        19 => (KIND_VALUE, 0, 0, 3, 13, true, false),
        20 => (KIND_VALUE, 0, 0, 4, 21, true, false),
        21 => (KIND_VALUE, 0, 0, 5, 37, true, false),
        22 => (KIND_VALUE, 0, 0, 9, 69, true, false),
        23 => (KIND_VALUE, 0, 1, 0, 1, true, false),
        24 => (KIND_VALUE, 0, 2, 0, 1, true, false),
        25 => (KIND_VALUE, 0, 3, 0, 1, true, false),
        26 => (KIND_VALUE, 0, 4, 0, 1, true, false),
        27 => (KIND_VALUE, 0, 5, 0, 1, true, false),
        28 => (KIND_VALUE, 2, 6, 0, 1, true, false),
        29 => (KIND_VALUE, 3, 10, 0, 1, true, false),
        30 => (KIND_VALUE, 0, 1, 1, 2, true, false),
        _ => (KIND_VALUE, 1, 2, 1, 2, true, false),
    };
    TokenShape {
        extra: run_bits + mag_bits + signed as u8,
        kind,
        run_bits,
        run_base,
        mag_bits,
        mag_base,
        signed,
        negative,
    }
}

const TOKEN_SHAPES: [TokenShape; 32] = {
    let mut out = [token_shape(0); 32];
    let mut t = 0;
    while t < 32 {
        out[t] = token_shape(t as u8);
        t += 1;
    }
    out
};

/// Zig-zag index → slot in the pipeline's coefficient blocks. Blocks
/// are stored *column-major* — the coefficient at natural row `r`,
/// column `c` lives at `c * 8 + r` — so that slot group `c` is one
/// lane vector (all eight rows of column `c`) for the row pass of the
/// inverse DCT.
const SLOT_OF_ZIGZAG: [u8; 64] = {
    let mut out = [0u8; 64];
    let mut ci = 0;
    while ci < 64 {
        out[ZIGZAG_NATURAL_TO_ZIGZAG[ci] as usize] = ((ci % 8) * 8 + ci / 8) as u8;
        ci += 1;
    }
    out
};

/// For a block whose first `n` zig-zag coefficients may be non-zero:
/// can any of them sit in natural rows 4..=7 (`.0`) or columns 4..=7
/// (`.1`)?
const UPPER_HALF_REACHED: [(bool, bool); 65] = {
    let mut out = [(false, false); 65];
    let mut rows = false;
    let mut cols = false;
    let mut n = 0;
    while n < 64 {
        let slot = SLOT_OF_ZIGZAG[n];
        if slot % 8 >= 4 {
            rows = true;
        }
        if slot / 8 >= 4 {
            cols = true;
        }
        out[n + 1] = (rows, cols);
        n += 1;
    }
    out
};
/// §7.7.3 with the reorganised bookkeeping described in the module
/// docs. `coeffs` must be all-zero on entry; on success it holds the
/// quantised coefficients of every coded block in slot order and
/// `ncoeffs` the §7.7.3 `NCOEFFS` counts.
#[allow(clippy::too_many_arguments)]
fn decode_tokens(
    r: &mut BitReader<'_>,
    nlbs: usize,
    bcoded: &[u8],
    hts: &[HuffmanTable],
    luts: &TokenLuts,
    active: &mut Vec<(u32, u8)>,
    coeffs: &mut [[i16; 64]],
    ncoeffs: &mut [u8],
) -> Result<(), Error> {
    active.clear();
    for (bi, &coded) in bcoded.iter().enumerate() {
        ncoeffs[bi] = 0;
        if coded != 0 {
            active.push((bi as u32, 0));
        }
    }
    let mut eobs: u64 = 0;
    let mut hti_l = 0usize;
    let mut hti_c = 0usize;

    for ti in 0u8..64 {
        if ti < 2 {
            hti_l = r.read_bits(4, "§7.7.3 htiL")? as usize;
            hti_c = r.read_bits(4, "§7.7.3 htiC")? as usize;
        }
        if active.is_empty() {
            continue;
        }
        let hg = match ti {
            0 => 0usize,
            1..=5 => 1,
            6..=14 => 2,
            15..=27 => 3,
            _ => 4,
        };
        let total = active.len();
        let mut keep = 0usize;
        for k in 0..total {
            let (bi, next_ti) = active[k];
            if next_ti != ti {
                // Still inside a zero run that skipped past `ti`.
                active[keep] = (bi, next_ti);
                keep += 1;
                continue;
            }
            let biu = bi as usize;
            ncoeffs[biu] = ti;
            if eobs > 0 {
                eobs -= 1;
                continue;
            }
            let hti = 16 * hg + if biu < nlbs { hti_l } else { hti_c };
            let window = r.peek32();
            let entry = luts
                .tables
                .get(hti)
                .map_or(LUT_MISS, |lut| lut[(window >> (32 - LUT_BITS)) as usize]);
            let (token, code_len, window) = if entry != LUT_MISS {
                ((entry & 31) as u8, u32::from(entry >> 5), window)
            } else {
                // Long code (or a degenerate table): the tree walk
                // consumes the code itself.
                let table = hts
                    .get(hti)
                    .ok_or(Error::DctCoefficientEmptyHuffmanTable { hti: hti as u8 })?;
                if table.entries.is_empty() {
                    return Err(Error::DctCoefficientEmptyHuffmanTable { hti: hti as u8 });
                }
                let token = match table.read_token(r) {
                    Ok(t) => t,
                    Err(ReadTokenErr::OffTree) => {
                        return Err(Error::DctCoefficientHuffmanWalkOffTree { hti: hti as u8 });
                    }
                    Err(ReadTokenErr::Truncated(e)) => return Err(e),
                };
                (token, 0, r.peek32())
            };
            let shape = TOKEN_SHAPES[(token & 31) as usize];
            let extra = u32::from(shape.extra);
            let x = if extra == 0 {
                0
            } else {
                (window << code_len) >> (32 - extra)
            };
            r.skip(code_len + extra, "§7.7 DCT token")?;

            match shape.kind {
                KIND_EOB => {
                    let mut run = u64::from(shape.run_base) + u64::from(x);
                    if run == 0 {
                        // §7.7.1 step 7(b): every coded block that is
                        // still open — the ones kept so far, this one,
                        // and the ones not yet visited in this pass.
                        run = (keep + (total - k)) as u64;
                    }
                    eobs = run - 1;
                }
                KIND_ZERO_RUN => {
                    let new_tis = u32::from(ti) + u32::from(shape.run_base) + x;
                    if new_tis > 64 {
                        return Err(Error::CoefficientTokenWouldOverflowBlock {
                            token,
                            ti,
                            new_tis: new_tis as u16,
                        });
                    }
                    if new_tis < 64 {
                        active[keep] = (bi, new_tis as u8);
                        keep += 1;
                    }
                }
                _ => {
                    let run = u32::from(shape.run_base) + (x & ((1 << shape.run_bits) - 1));
                    let mag_part = (x >> shape.run_bits) & ((1 << shape.mag_bits) - 1);
                    let negative = if shape.signed {
                        (x >> (shape.run_bits + shape.mag_bits)) & 1 != 0
                    } else {
                        shape.negative
                    };
                    let mag = i16::from(shape.mag_base) + mag_part as i16;
                    let pos = u32::from(ti) + run;
                    let new_tis = pos + 1;
                    if new_tis > 64 {
                        return Err(Error::CoefficientTokenWouldOverflowBlock {
                            token,
                            ti,
                            new_tis: new_tis as u16,
                        });
                    }
                    coeffs[biu][SLOT_OF_ZIGZAG[pos as usize] as usize] =
                        if negative { -mag } else { mag };
                    ncoeffs[biu] = new_tis as u8;
                    if new_tis < 64 {
                        active[keep] = (bi, new_tis as u8);
                        keep += 1;
                    }
                }
            }
        }
        active.truncate(keep);
    }

    if eobs > 0 {
        return Err(Error::DctCoefficientLeftoverEobs { residual: eobs });
    }
    Ok(())
}

// ---------------------------------------------------------------------
// §7.8 DC prediction
// ---------------------------------------------------------------------

/// Reference-frame class of a coding mode (Table 7.46): 0 = intra,
/// 1 = previous frame, 2 = golden frame.
#[inline(always)]
fn reference_class(mode: MacroBlockMode) -> u8 {
    match mode {
        MacroBlockMode::Intra => 0,
        MacroBlockMode::InterGoldenNoMv | MacroBlockMode::InterGoldenMv => 2,
        _ => 1,
    }
}

/// Class marker for "no usable neighbour" (off the plane or uncoded).
const NO_CLASS: u8 = 3;

/// §7.8.2 over one plane: undo the DC prediction in raster order.
/// Slot 0 of each coded block's coefficients holds its DC residual on
/// entry and its reconstructed DC on return. `rows` is scratch (class + DC of the
/// previous and current block row, padded by one slot on each side).
fn undo_dc_prediction_plane(
    grid: &[u32],
    bw: usize,
    bcoded: &[u8],
    mbmodes: &[MacroBlockMode],
    mbi_of_block: &[u32],
    coeffs: &mut [[i16; 64]],
    rows: &mut DcRows,
) {
    if bw == 0 {
        return;
    }
    rows.reset(bw);
    let mut last_dc = [0i32; 3];
    for grid_row in grid.chunks_exact(bw) {
        rows.advance();
        let (lower_cls, lower_dc, cur_cls, cur_dc) = rows.split();
        for (x, &bi) in grid_row.iter().enumerate() {
            let bi = bi as usize;
            if bcoded[bi] == 0 {
                cur_cls[x + 1] = NO_CLASS;
                continue;
            }
            let cls = reference_class(mbmodes[mbi_of_block[bi] as usize]);
            let mask = usize::from(cur_cls[x] == cls)
                | usize::from(lower_cls[x] == cls) << 1
                | usize::from(lower_cls[x + 1] == cls) << 2
                | usize::from(lower_cls[x + 2] == cls) << 3;
            let pred = if mask == 0 {
                last_dc[cls as usize]
            } else {
                let near = [
                    i32::from(cur_dc[x]),
                    i32::from(lower_dc[x]),
                    i32::from(lower_dc[x + 1]),
                    i32::from(lower_dc[x + 2]),
                ];
                let weights = TABLE_7_47[mask - 1];
                let mut acc = 0i32;
                for (k, (&w, &v)) in weights.w.iter().zip(&near).enumerate() {
                    if mask & (1 << k) != 0 {
                        acc += w * v;
                    }
                }
                let mut p = acc / weights.pdiv;
                if mask & 7 == 7 {
                    if (p - near[2]).abs() > 128 {
                        p = near[2];
                    } else if (p - near[0]).abs() > 128 {
                        p = near[0];
                    } else if (p - near[1]).abs() > 128 {
                        p = near[1];
                    }
                }
                p
            };
            let value = (i32::from(coeffs[bi][0]).wrapping_add(pred)) as i16;
            coeffs[bi][0] = value;
            last_dc[cls as usize] = i32::from(value);
            cur_cls[x + 1] = cls;
            cur_dc[x + 1] = value;
        }
    }
}

/// Two block rows of (class, DC) for [`undo_dc_prediction_plane`].
#[derive(Debug, Clone, Default)]
pub(crate) struct DcRows {
    cls: [Vec<u8>; 2],
    dc: [Vec<i16>; 2],
    cur: usize,
}

impl DcRows {
    fn reset(&mut self, bw: usize) {
        for v in &mut self.cls {
            v.clear();
            v.resize(bw + 2, NO_CLASS);
        }
        for v in &mut self.dc {
            v.clear();
            v.resize(bw + 2, 0);
        }
        self.cur = 0;
    }

    /// Make the current row the lower row and start a fresh current.
    fn advance(&mut self) {
        self.cur ^= 1;
    }

    /// `(lower classes, lower DCs, current classes, current DCs)`.
    fn split(&mut self) -> (&[u8], &[i16], &mut [u8], &mut [i16]) {
        let [c0, c1] = &mut self.cls;
        let [d0, d1] = &mut self.dc;
        if self.cur == 0 {
            (c1, d1, c0, d0)
        } else {
            (c0, d0, c1, d1)
        }
    }
}

// ---------------------------------------------------------------------
// §7.9.3 inverse DCT on eight lanes
// ---------------------------------------------------------------------

const C1: i32 = 64277;
const C2: i32 = 60547;
const C3: i32 = 54491;
const C4: i32 = 46341;
const C5: i32 = 36410;
const C6: i32 = 25080;
const C7: i32 = 12785;

/// `(c * a) >> 16` for a multiplier below 2¹⁵.
#[inline(always)]
fn scale(a: i16, c: i32) -> i16 {
    ((i32::from(a) * c) >> 16) as i16
}

/// `(c * a) >> 16` for a multiplier in 2¹⁵..2¹⁶, as a 16×16-bit
/// product: `c·a = (c − 2¹⁶)·a + 2¹⁶·a`, and the second term shifts to
/// exactly `a`.
#[inline(always)]
fn scale_wide(a: i16, c: i32) -> i16 {
    scale(a, c - 65536).wrapping_add(a)
}

/// §7.9.3.1 on eight independent lanes: `y[k][lane]` is input `k`,
/// `x[k][lane]` output `k`. With `UPPER` false the inputs `y[4..8]`
/// are known to be zero and their terms drop out.
///
/// The specification computes in wider integers and truncates to 16
/// bits before every multiplication and at the output. Sums and
/// differences commute with that truncation, and every product
/// `(c · v) >> 16` of a 16-bit `v` with `c < 2¹⁶` fits 16 bits, so the
/// whole procedure runs in wrapping 16-bit arithmetic with identical
/// results — eight lanes to a vector register.
#[inline(always)]
fn idct_lanes<const UPPER: bool>(y: &[[i16; 8]; 8], x: &mut [[i16; 8]; 8]) {
    for i in 0..8 {
        let (y0, y1, y2, y3) = (y[0][i], y[1][i], y[2][i], y[3][i]);
        let (y4, y5, y6, y7) = if UPPER {
            (y[4][i], y[5][i], y[6][i], y[7][i])
        } else {
            (0, 0, 0, 0)
        };
        let t0 = scale_wide(y0.wrapping_add(y4), C4);
        let t1 = scale_wide(y0.wrapping_sub(y4), C4);
        let t2 = scale(y2, C6).wrapping_sub(scale_wide(y6, C2));
        let t3 = scale_wide(y2, C2).wrapping_add(scale(y6, C6));
        let t4 = scale(y1, C7).wrapping_sub(scale_wide(y7, C1));
        let t5 = scale_wide(y5, C3).wrapping_sub(scale_wide(y3, C5));
        let t6 = scale_wide(y5, C5).wrapping_add(scale_wide(y3, C3));
        let t7 = scale_wide(y1, C1).wrapping_add(scale(y7, C7));

        let s45 = t4.wrapping_add(t5);
        let d45 = scale_wide(t4.wrapping_sub(t5), C4);
        let s76 = t7.wrapping_add(t6);
        let d76 = scale_wide(t7.wrapping_sub(t6), C4);

        let e0 = t0.wrapping_add(t3);
        let e3 = t0.wrapping_sub(t3);
        let e1 = t1.wrapping_add(t2);
        let e2 = t1.wrapping_sub(t2);
        let f6 = d76.wrapping_add(d45);
        let f5 = d76.wrapping_sub(d45);

        x[0][i] = e0.wrapping_add(s76);
        x[1][i] = e1.wrapping_add(f6);
        x[2][i] = e2.wrapping_add(f5);
        x[3][i] = e3.wrapping_add(s45);
        x[4][i] = e3.wrapping_sub(s45);
        x[5][i] = e2.wrapping_sub(f5);
        x[6][i] = e1.wrapping_sub(f6);
        x[7][i] = e0.wrapping_sub(s76);
    }
}

/// §7.9.3.2 two-dimensional inverse DCT of a dequantised block in
/// slot (column-major) order. `rows_upper` / `cols_upper` say whether
/// natural rows / columns 4..=7 can hold a non-zero coefficient. The
/// result is row-major, already rounded (`(x + 8) >> 4`).
#[inline(always)]
fn idct_block(dqc: &[i16; 64], rows_upper: bool, cols_upper: bool) -> [[i16; 8]; 8] {
    // Row pass: lane = row, input k = column k — exactly the slot
    // groups.
    let y: &[[i16; 8]; 8] = dqc
        .as_chunks::<8>()
        .0
        .try_into()
        .expect("64 coefficients are eight groups of eight");
    let mut x = [[0i16; 8]; 8];
    if cols_upper {
        idct_lanes::<true>(y, &mut x);
    } else {
        idct_lanes::<false>(y, &mut x);
    }
    // x[c][r] is the row-pass output at (row r, column c). Column
    // pass: lane = column, input k = row k.
    let mut z = [[0i16; 8]; 8];
    for (k, lanes) in z.iter_mut().enumerate() {
        for (col, lane) in lanes.iter_mut().enumerate() {
            *lane = x[col][k];
        }
    }
    let mut out = [[0i16; 8]; 8];
    if rows_upper {
        idct_lanes::<true>(&z, &mut out);
    } else {
        idct_lanes::<false>(&z, &mut out);
    }
    // (v + 8) >> 4 without leaving 16 bits.
    for row in &mut out {
        for v in row.iter_mut() {
            *v = (*v >> 4) + (((*v & 15) + 8) >> 4);
        }
    }
    out
}

/// [`crate::inverse_dct_2d`] through the lane kernel: `dqc` in natural
/// (row-major) order, result row-major. Bit-identical to the
/// step-by-step transform.
pub(crate) fn inverse_dct_block(dqc: &[i16; 64]) -> [[i16; 8]; 8] {
    let mut slots = [0i16; 64];
    for (c, group) in slots.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        for (r, slot) in group.iter_mut().enumerate() {
            *slot = dqc[r * 8 + c];
        }
    }
    idct_block(&slots, true, true)
}

// ---------------------------------------------------------------------
// §7.9.3.3 forward DCT on eight lanes (encoder side)
// ---------------------------------------------------------------------

/// `x / 2¹⁶` rounded toward zero.
#[inline(always)]
fn div_2p16(x: i32) -> i32 {
    (x + ((x >> 31) & 0xFFFF)) >> 16
}

/// The one-dimensional forward transform of [`crate::forward_dct_1d`]
/// on eight independent lanes: `x[k][lane]` is input sample `k`,
/// `y[k][lane]` output coefficient `k` (truncated to 16 bits, as the
/// step-by-step procedure stores it).
#[inline(always)]
fn fdct_lanes(x: &[[i32; 8]; 8], y: &mut [[i32; 8]; 8]) {
    for i in 0..8 {
        let a0 = x[0][i].wrapping_add(x[7][i]);
        let a1 = x[1][i].wrapping_add(x[6][i]);
        let a2 = x[2][i].wrapping_add(x[5][i]);
        let a3 = x[3][i].wrapping_add(x[4][i]);
        let a4 = x[3][i].wrapping_sub(x[4][i]);
        let a5 = x[2][i].wrapping_sub(x[5][i]);
        let a6 = x[1][i].wrapping_sub(x[6][i]);
        let a7 = x[0][i].wrapping_sub(x[7][i]);

        let b0 = a0.wrapping_add(a3);
        let b3 = a0.wrapping_sub(a3);
        let b1 = a1.wrapping_add(a2);
        let b2 = a1.wrapping_sub(a2);

        let c6 = div_2p16(C4.wrapping_mul(a6.wrapping_add(a5)));
        let c5 = div_2p16(C4.wrapping_mul(a6.wrapping_sub(a5)));

        let d4 = a4.wrapping_add(c5);
        let d5 = a4.wrapping_sub(c5);
        let d7 = a7.wrapping_add(c6);
        let d6 = a7.wrapping_sub(c6);

        let t = |v: i32| i32::from(v as i16);
        y[0][i] = t(div_2p16(C4.wrapping_mul(b0.wrapping_add(b1))));
        y[4][i] = t(div_2p16(C4.wrapping_mul(b0.wrapping_sub(b1))));
        y[2][i] = t(div_2p16(C2.wrapping_mul(b3)).wrapping_add(div_2p16(C6.wrapping_mul(b2))));
        y[6][i] = t(div_2p16(C6.wrapping_mul(b3)).wrapping_sub(div_2p16(C2.wrapping_mul(b2))));
        y[1][i] = t(div_2p16(C1.wrapping_mul(d7)).wrapping_add(div_2p16(C7.wrapping_mul(d4))));
        y[5][i] = t(div_2p16(C5.wrapping_mul(d6)).wrapping_add(div_2p16(C3.wrapping_mul(d5))));
        y[3][i] = t(div_2p16(C3.wrapping_mul(d6)).wrapping_sub(div_2p16(C5.wrapping_mul(d5))));
        y[7][i] = t(div_2p16(C7.wrapping_mul(d7)).wrapping_sub(div_2p16(C1.wrapping_mul(d4))));
    }
}

/// [`crate::forward_dct_2d`] through the lane kernel: rows first, then
/// columns; the result is in natural (row-major) order. Bit-identical
/// to the step-by-step transform.
pub(crate) fn forward_dct_block(input: &[[i16; 8]; 8]) -> [i16; 64] {
    // Row pass: lane = row, sample k = column k.
    let mut x = [[0i32; 8]; 8];
    for (k, lanes) in x.iter_mut().enumerate() {
        for (row, lane) in lanes.iter_mut().enumerate() {
            *lane = i32::from(input[row][k]);
        }
    }
    let mut y = [[0i32; 8]; 8];
    fdct_lanes(&x, &mut y);
    // y[k][row] is coefficient k of row `row`. Column pass: lane =
    // column, sample k = row k.
    let mut z = [[0i32; 8]; 8];
    for (k, lanes) in z.iter_mut().enumerate() {
        for (col, lane) in lanes.iter_mut().enumerate() {
            *lane = y[col][k];
        }
    }
    let mut o = [[0i32; 8]; 8];
    fdct_lanes(&z, &mut o);
    let mut out = [0i16; 64];
    for (k, lanes) in o.iter().enumerate() {
        for (col, &v) in lanes.iter().enumerate() {
            out[k * 8 + col] = v as i16;
        }
    }
    out
}

// ---------------------------------------------------------------------
// §7.9 reconstruction
// ---------------------------------------------------------------------

/// The §6.4.3 matrices a frame can touch: `[qti][pli][qii]`, in slot
/// order (see [`SLOT_OF_ZIGZAG`]), with element 0 taken from the `qi0` matrix (§7.9.2: the DC
/// coefficient always uses the frame's first `qi`). A matrix that
/// fails to evaluate keeps its error, raised only if a block needs it.
type FrameMatrices = [[[Result<[i16; 64], Error>; 3]; 3]; 2];

fn frame_matrices(params: &QuantizationParameters, qis: &[u8], intra_only: bool) -> FrameMatrices {
    let mut out: FrameMatrices = [[[Err(Error::ReconstructEmptyQis); 3]; 3]; 2];
    let qi0 = qis[0] as usize;
    for (qti, per_plane) in out.iter_mut().enumerate() {
        if intra_only && qti == 1 {
            continue;
        }
        for (pli, per_qii) in per_plane.iter_mut().enumerate() {
            let dc = compute_quantization_matrix(params, qti, pli, qi0);
            for (qii, slot) in per_qii.iter_mut().enumerate() {
                let Some(&qi) = qis.get(qii) else {
                    *slot = Err(Error::ReconstructQiiIndexOutOfRange {
                        qii,
                        nqis: qis.len(),
                    });
                    continue;
                };
                *slot = match (
                    dc,
                    compute_quantization_matrix(params, qti, pli, qi as usize),
                ) {
                    (Ok(dc), Ok(ac)) => {
                        let mut m = [0i16; 64];
                        for (ci, &q) in ac.values.iter().enumerate() {
                            m[(ci % 8) * 8 + ci / 8] = q as i16;
                        }
                        m[0] = dc.values[0] as i16;
                        Ok(m)
                    }
                    (Err(e), _) | (_, Err(e)) => Err(e),
                };
            }
        }
    }
    out
}

/// Per-frame block state shared by every reconstruction band.
struct FrameCtx<'a> {
    bcoded: &'a [u8],
    mbmodes: &'a [MacroBlockMode],
    mvects: &'a [MotionVector],
    qiis: &'a [u8],
    ncoeffs: &'a [u8],
    mbi_of_block: &'a [u32],
    mats: &'a FrameMatrices,
}

/// One colour plane's geometry and reference samples.
#[derive(Clone, Copy)]
struct PlaneCtx<'a> {
    pli: usize,
    /// Plane size in blocks.
    bw: usize,
    bh: usize,
    /// Raster position → coded-order block index.
    grid: &'a [u32],
    /// Raster position → "is this block coded" for the current frame.
    coded: &'a [u8],
    prev: &'a [u8],
    golden: &'a [u8],
    /// Motion-vector divisors (2 = half-pixel units, 4 = quarter).
    div_x: i32,
    div_y: i32,
}

/// Split one motion-vector component into its two §7.9.1 sample
/// offsets: truncated toward zero and rounded away from zero.
#[inline(always)]
fn split_component(v: i32, div: i32) -> (i32, i32) {
    let toward = v / div;
    let away = if v % div == 0 {
        toward
    } else {
        toward + v.signum()
    };
    (toward, away)
}

/// Fetch an 8×8 predictor at integer offset `(rx, ry)`, clamping
/// coordinates to the plane (§7.9.1.2 step 1).
#[cold]
fn clamped_block(plane: &[u8], w: usize, h: usize, rx: i32, ry: i32) -> [[u8; 8]; 8] {
    let mut out = [[0u8; 8]; 8];
    for (j, row) in out.iter_mut().enumerate() {
        let y = (ry + j as i32).clamp(0, h as i32 - 1) as usize;
        for (i, px) in row.iter_mut().enumerate() {
            let x = (rx + i as i32).clamp(0, w as i32 - 1) as usize;
            *px = plane[y * w + x];
        }
    }
    out
}

/// Reconstruct block rows `y0..y1` of one plane into `dst` (pixel rows
/// `8·y0 .. 8·y1`). `coeffs` holds the coefficient blocks whose
/// coded-order index starts at `coeff_base`; every block consumed is
/// returned to all-zero.
fn reconstruct_rows(
    f: &FrameCtx<'_>,
    p: &PlaneCtx<'_>,
    y0: usize,
    y1: usize,
    dst: &mut [u8],
    coeffs: &mut [[i16; 64]],
    coeff_base: usize,
) -> Result<(), Error> {
    let stride = p.bw * 8;
    let height = p.bh * 8;
    for y in y0..y1 {
        let band_row = &mut dst[(y - y0) * 8 * stride..(y - y0 + 1) * 8 * stride];
        let src_row = y * 8 * stride;
        for x in 0..p.bw {
            let bi = p.grid[y * p.bw + x] as usize;
            let px = x * 8;
            if f.bcoded[bi] == 0 {
                // §7.9.4 step 2(e): copy from the previous frame.
                for j in 0..8 {
                    let o = j * stride + px;
                    band_row[o..o + 8].copy_from_slice(&p.prev[src_row + o..src_row + o + 8]);
                }
                continue;
            }
            let mode = f.mbmodes[f.mbi_of_block[bi] as usize];
            let cls = reference_class(mode);
            let qti = usize::from(cls != 0);
            let nc = f.ncoeffs[bi];
            let block = &mut coeffs[bi - coeff_base];

            // Residual.
            let res: [[i16; 8]; 8] = if nc < 2 {
                // §7.9.4 step 2(f)i: DC-only shortcut.
                let m = f.mats[qti][p.pli][0].as_ref().map_err(|e| *e)?;
                let dc = (i32::from(block[0])
                    .wrapping_mul(i32::from(m[0]))
                    .wrapping_add(15)
                    >> 5) as i16;
                block[0] = 0;
                [[dc; 8]; 8]
            } else {
                let m = f.mats[qti][p.pli][f.qiis[bi] as usize]
                    .as_ref()
                    .map_err(|e| *e)?;
                let mut dqc = [0i16; 64];
                for ((d, &c), &q) in dqc.iter_mut().zip(block.iter()).zip(m.iter()) {
                    *d = c.wrapping_mul(q);
                }
                *block = [0i16; 64];
                let (rows_upper, cols_upper) = UPPER_HALF_REACHED[nc as usize];
                idct_block(&dqc, rows_upper, cols_upper)
            };

            // Predictor + residual.
            if cls == 0 {
                for (j, res_row) in res.iter().enumerate() {
                    let o = j * stride + px;
                    for (d, &r) in band_row[o..o + 8].iter_mut().zip(res_row) {
                        *d = (128 + i32::from(r)).clamp(0, 255) as u8;
                    }
                }
                continue;
            }
            let refp = if cls == 1 { p.prev } else { p.golden };
            let mv = f.mvects[bi];
            let (ax, bx) = split_component(i32::from(mv.x), p.div_x);
            let (ay, by) = split_component(i32::from(mv.y), p.div_y);
            let (x0, y0p) = (px as i32, (y * 8) as i32);
            let inside = |dx: i32, dy: i32| {
                x0 + dx >= 0
                    && y0p + dy >= 0
                    && x0 + dx + 8 <= stride as i32
                    && y0p + dy + 8 <= height as i32
            };
            if ax == bx && ay == by {
                if inside(ax, ay) {
                    let base = ((y0p + ay) as usize) * stride + (x0 + ax) as usize;
                    for (j, res_row) in res.iter().enumerate() {
                        let o = j * stride + px;
                        let s = &refp[base + j * stride..base + j * stride + 8];
                        for ((d, &r), &pv) in band_row[o..o + 8].iter_mut().zip(res_row).zip(s) {
                            *d = (i32::from(pv) + i32::from(r)).clamp(0, 255) as u8;
                        }
                    }
                } else {
                    let pred = clamped_block(refp, stride, height, x0 + ax, y0p + ay);
                    for (j, res_row) in res.iter().enumerate() {
                        let o = j * stride + px;
                        for ((d, &r), &pv) in
                            band_row[o..o + 8].iter_mut().zip(res_row).zip(&pred[j])
                        {
                            *d = (i32::from(pv) + i32::from(r)).clamp(0, 255) as u8;
                        }
                    }
                }
            } else if inside(ax, ay) && inside(bx, by) {
                let base1 = ((y0p + ay) as usize) * stride + (x0 + ax) as usize;
                let base2 = ((y0p + by) as usize) * stride + (x0 + bx) as usize;
                for (j, res_row) in res.iter().enumerate() {
                    let o = j * stride + px;
                    let s1 = &refp[base1 + j * stride..base1 + j * stride + 8];
                    let s2 = &refp[base2 + j * stride..base2 + j * stride + 8];
                    for (((d, &r), &p1), &p2) in
                        band_row[o..o + 8].iter_mut().zip(res_row).zip(s1).zip(s2)
                    {
                        let pv = (i32::from(p1) + i32::from(p2)) >> 1;
                        *d = (pv + i32::from(r)).clamp(0, 255) as u8;
                    }
                }
            } else {
                let pred1 = clamped_block(refp, stride, height, x0 + ax, y0p + ay);
                let pred2 = clamped_block(refp, stride, height, x0 + bx, y0p + by);
                for (j, res_row) in res.iter().enumerate() {
                    let o = j * stride + px;
                    for (((d, &r), &p1), &p2) in band_row[o..o + 8]
                        .iter_mut()
                        .zip(res_row)
                        .zip(&pred1[j])
                        .zip(&pred2[j])
                    {
                        let pv = (i32::from(p1) + i32::from(p2)) >> 1;
                        *d = (pv + i32::from(r)).clamp(0, 255) as u8;
                    }
                }
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// §7.10 loop filter
// ---------------------------------------------------------------------

/// §7.10.1 limit function in closed form: `sign(r) · max(0, min(|r|,
/// 2L − |r|))`.
#[inline(always)]
fn limit(r: i32, l: i32) -> i32 {
    let a = r.abs();
    let m = a.min(2 * l - a).max(0);
    if r < 0 { -m } else { m }
}

/// The limit function tabulated for one frame's `L`: entry `r + 256`
/// for the filter response `r`, which lies in `-128..=128`.
type LimitTable = [i8; 512];

fn limit_table(l: i32) -> LimitTable {
    let mut out = [0i8; 512];
    for (slot, r) in out.iter_mut().zip(-256i32..256) {
        *slot = limit(r, l) as i8;
    }
    out
}

/// §7.10.2 horizontal filter (across a vertical block edge) on one
/// pixel row: the four samples start at `fx`.
#[inline(always)]
fn filter_across(row: &mut [u8], fx: usize, lut: &LimitTable) {
    let p: &mut [u8; 4] = (&mut row[fx..fx + 4]).try_into().expect("four samples");
    let (p1, p2) = (i32::from(p[1]), i32::from(p[2]));
    let r = (i32::from(p[0]) - 3 * p1 + 3 * p2 - i32::from(p[3]) + 4) >> 3;
    let d = i32::from(lut[(r + 256) as usize & 511]);
    p[1] = (p1 + d).clamp(0, 255) as u8;
    p[2] = (p2 - d).clamp(0, 255) as u8;
}

/// §7.10.2 horizontal filter on the eight pixel rows of `rows` (a
/// whole block edge): the four samples of each row start at `fx`.
#[inline(always)]
fn filter_across_block(rows: &mut [u8], stride: usize, fx: usize, lut: &LimitTable) {
    for row in rows.chunks_exact_mut(stride) {
        filter_across(row, fx, lut);
    }
}

/// §7.10.2 vertical filter (across a horizontal block edge) on eight
/// columns: `r0..r3` are the four pixel rows, bottom to top, each an
/// 8-sample slice.
#[inline(always)]
fn filter_along(r0: &[u8], r1: &mut [u8], r2: &mut [u8], r3: &[u8], l: i32) {
    for (((&a, b), c), &d) in r0.iter().zip(r1.iter_mut()).zip(r2.iter_mut()).zip(r3) {
        let (p1, p2) = (i32::from(*b), i32::from(*c));
        let r = (i32::from(a) - 3 * p1 + 3 * p2 - i32::from(d) + 4) >> 3;
        let delta = limit(r, l);
        *b = (p1 + delta).clamp(0, 255) as u8;
        *c = (p2 - delta).clamp(0, 255) as u8;
    }
}

/// Split four consecutive pixel rows starting at row `fy` of `buf`
/// into the `(r0, r1, r2, r3)` column window `[x, x + 8)`.
#[inline(always)]
fn four_rows(
    buf: &mut [u8],
    stride: usize,
    fy: usize,
    x: usize,
) -> (&[u8], &mut [u8], &mut [u8], &[u8]) {
    let rows = &mut buf[fy * stride..(fy + 4) * stride];
    let (lo, hi) = rows.split_at_mut(2 * stride);
    let (r0, r1) = lo.split_at_mut(stride);
    let (r2, r3) = hi.split_at_mut(stride);
    (
        &r0[x..x + 8],
        &mut r1[x..x + 8],
        &mut r2[x..x + 8],
        &r3[x..x + 8],
    )
}

/// §7.10.3 over block rows `y0..y1` of one plane. `band` is the
/// plane's pixel rows `8·y0 .. 8·y1`.
///
/// The filter visits blocks in raster order and, for every coded
/// block, filters its left edge, bottom edge, right edge (when the
/// right neighbour is uncoded) and top edge (when the upper neighbour
/// is uncoded) in place. Processing of block row `k` therefore
/// interacts with row `k − 1` only through pixel rows `8k − 2 ..
/// 8k + 2`, and that four-row strip evolves on its own while row
/// `k − 1` is processed: the only operations touching it are row
/// `k − 1`'s left/right-edge filters (each pixel row independently)
/// and its top-edge filters.
///
/// A band that does not start at the bottom of the plane receives
/// `below` — a private copy of the unfiltered pixel rows `8·y0 − 2`
/// and `8·y0 − 1` — and first replays exactly those strip operations
/// of row `y0 − 1` on (`below`, its own first two rows). A band that
/// does not end at the top receives `above`, a private copy of the
/// unfiltered pixel rows `8·y1` and `8·y1 + 1`, as the far side of
/// its last row's top-edge filters. On return `below[stride..]` is the
/// final content of pixel row `8·y0 − 1`, which the caller writes back
/// (the lower band's own version of that row lacks this band's
/// bottom-edge filters); writes to `above` are discarded.
fn filter_band(
    band: &mut [u8],
    p: &PlaneCtx<'_>,
    y0: usize,
    y1: usize,
    mut below: Option<&mut [u8]>,
    mut above: Option<&mut [u8]>,
    l: i32,
) {
    let (bw, bh) = (p.bw, p.bh);
    let stride = bw * 8;
    let lut = &limit_table(l);

    if let Some(below) = below.as_deref_mut() {
        // Replay block row y0 − 1 on the boundary strip.
        let lower = &p.coded[(y0 - 1) * bw..y0 * bw];
        let upper = &p.coded[y0 * bw..(y0 + 1) * bw];
        for x in 0..bw {
            if lower[x] == 0 {
                continue;
            }
            let (b0, b1) = below.split_at_mut(stride);
            if x > 0 {
                filter_across(b0, x * 8 - 2, lut);
                filter_across(b1, x * 8 - 2, lut);
            }
            if x + 1 < bw && lower[x + 1] == 0 {
                filter_across(b0, x * 8 + 6, lut);
                filter_across(b1, x * 8 + 6, lut);
            }
            if upper[x] == 0 {
                let (n0, n1) = band[..2 * stride].split_at_mut(stride);
                let c = x * 8;
                filter_along(
                    &b0[c..c + 8],
                    &mut b1[c..c + 8],
                    &mut n0[c..c + 8],
                    &n1[c..c + 8],
                    l,
                );
            }
        }
    }

    for y in y0..y1 {
        let top = (y - y0) * 8;
        let row_coded = &p.coded[y * bw..(y + 1) * bw];
        let up_coded = (y + 1 < bh).then(|| &p.coded[(y + 1) * bw..(y + 2) * bw]);
        for x in 0..bw {
            if row_coded[x] == 0 {
                continue;
            }
            let c = x * 8;
            if x > 0 {
                filter_across_block(
                    &mut band[top * stride..(top + 8) * stride],
                    stride,
                    c - 2,
                    lut,
                );
            }
            if y > 0 {
                if y == y0 {
                    let below = below
                        .as_deref_mut()
                        .expect("a band above the plane bottom carries its lower strip");
                    let (b0, b1) = below.split_at_mut(stride);
                    let (n0, n1) = band[..2 * stride].split_at_mut(stride);
                    filter_along(
                        &b0[c..c + 8],
                        &mut b1[c..c + 8],
                        &mut n0[c..c + 8],
                        &n1[c..c + 8],
                        l,
                    );
                } else {
                    let (r0, r1, r2, r3) = four_rows(band, stride, top - 2, c);
                    filter_along(r0, r1, r2, r3, l);
                }
            }
            if x + 1 < bw && row_coded[x + 1] == 0 {
                filter_across_block(
                    &mut band[top * stride..(top + 8) * stride],
                    stride,
                    c + 6,
                    lut,
                );
            }
            if let Some(up_coded) = up_coded
                && up_coded[x] == 0
            {
                if y + 1 == y1 {
                    let above = above
                        .as_deref_mut()
                        .expect("a band below the plane top carries its upper strip");
                    let (n0, n1) =
                        band[(top + 6) * stride..(top + 8) * stride].split_at_mut(stride);
                    let (a0, a1) = above.split_at_mut(stride);
                    filter_along(
                        &n0[c..c + 8],
                        &mut n1[c..c + 8],
                        &mut a0[c..c + 8],
                        &a1[c..c + 8],
                        l,
                    );
                } else {
                    let (r0, r1, r2, r3) = four_rows(band, stride, top + 6, c);
                    filter_along(r0, r1, r2, r3, l);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------
// Bands
// ---------------------------------------------------------------------

/// The four pixel rows straddling an interior band boundary at block
/// row `k`, as left by reconstruction (before any filtering), plus the
/// finished row handed back by the upper band.
#[derive(Default)]
struct Strip {
    /// Pixel rows `8k − 2` and `8k − 1` (posted by the lower band).
    lower: Vec<u8>,
    /// Pixel rows `8k` and `8k + 1` (posted by the upper band).
    upper: Vec<u8>,
    /// Final content of pixel row `8k − 1` (posted by the upper band
    /// after filtering).
    patched: Vec<u8>,
}

/// One band of one plane, with exclusive ownership of its output rows
/// and coefficient blocks.
struct BandTask<'a> {
    p: PlaneCtx<'a>,
    /// Block rows `y0..y1`.
    y0: usize,
    y1: usize,
    /// Index of this band within its plane, and the plane's band count.
    band: usize,
    nbands: usize,
    dst: &'a mut [u8],
    /// The band's coefficient blocks; the first one has coded-order
    /// index `coeff_base`.
    coeffs: &'a mut [[i16; 64]],
    coeff_base: usize,
}

/// Reconstruct, then loop-filter, one worker's bands. With `sync` the
/// worker posts its boundary rows, meets the other workers at the
/// barrier once every band is reconstructed, and filters against
/// private copies of the neighbouring rows (see [`filter_band`]);
/// without it the bands span whole planes (or the filter is off).
fn run_bands(
    tasks: &mut [BandTask<'_>],
    frame: &FrameCtx<'_>,
    sync: Option<(&Barrier, &[Vec<Mutex<Strip>>; 3])>,
    l: i32,
) -> Result<(), Error> {
    let mut result = Ok(());
    for t in tasks.iter_mut() {
        if result.is_ok() {
            result = reconstruct_rows(frame, &t.p, t.y0, t.y1, t.dst, t.coeffs, t.coeff_base);
        }
        if let Some((_, strips)) = sync {
            let two = 2 * t.p.bw * 8;
            let strips = &strips[t.p.pli];
            if t.band > 0 {
                strips[t.band - 1].lock().expect("strip lock").upper = t.dst[..two].to_vec();
            }
            if t.band + 1 < t.nbands {
                strips[t.band].lock().expect("strip lock").lower =
                    t.dst[t.dst.len() - two..].to_vec();
            }
        }
    }
    let Some((barrier, strips)) = sync else {
        if l != 0 && result.is_ok() {
            for t in tasks.iter_mut().filter(|t| t.nbands == 1) {
                filter_band(t.dst, &t.p, t.y0, t.y1, None, None, l);
            }
        }
        return result;
    };
    barrier.wait();
    result?;
    for t in tasks.iter_mut() {
        let strips = &strips[t.p.pli];
        let stride = t.p.bw * 8;
        let mut below =
            (t.band > 0).then(|| strips[t.band - 1].lock().expect("strip lock").lower.clone());
        let mut above = (t.band + 1 < t.nbands)
            .then(|| strips[t.band].lock().expect("strip lock").upper.clone());
        filter_band(
            t.dst,
            &t.p,
            t.y0,
            t.y1,
            below.as_deref_mut(),
            above.as_deref_mut(),
            l,
        );
        if let Some(below) = below {
            strips[t.band - 1].lock().expect("strip lock").patched = below[stride..].to_vec();
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Frame driver
// ---------------------------------------------------------------------

/// Decoder state that persists across frames for the fast pipeline.
#[derive(Debug, Clone)]
pub(crate) struct PipelineState {
    luts: TokenLuts,
    /// Quantised coefficients in natural order; all-zero between
    /// frames.
    coeffs: Vec<[i16; 64]>,
    ncoeffs: Vec<u8>,
    active: Vec<(u32, u8)>,
    dc_rows: DcRows,
    /// Raster "is coded" maps of the three planes.
    coded: [Vec<u8>; 3],
    /// The plane set the next frame is reconstructed into (swapped
    /// with the previous-frame reference afterwards).
    spare: [Vec<u8>; 3],
    /// First coded-order block index of each plane.
    plane_base: [usize; 3],
    /// Worker budget (1 = serial).
    pub(crate) threads: usize,
}

impl PipelineState {
    pub(crate) fn new(g: &FrameGeometry, hts: &[HuffmanTable]) -> Self {
        let nbs = g.nbs as usize;
        let len_y = (g.dims_y.width as usize) * (g.dims_y.height as usize);
        let len_c = (g.dims_c.width as usize) * (g.dims_c.height as usize);
        let n: [usize; 3] = core::array::from_fn(|pli| g.plane_raster_order[pli].len());
        Self {
            luts: TokenLuts::build(hts),
            coeffs: vec![[0i16; 64]; nbs],
            ncoeffs: vec![0u8; nbs],
            active: Vec::new(),
            dc_rows: DcRows::default(),
            coded: [vec![1u8; n[0]], vec![1u8; n[1]], vec![1u8; n[2]]],
            spare: [vec![0u8; len_y], vec![0u8; len_c], vec![0u8; len_c]],
            plane_base: [0, n[0], n[0] + n[1]],
            threads: 1,
        }
    }
}

/// The block-level arrays §7.3–§7.6 hand to the back half of the
/// pipeline.
pub(crate) struct FrontEnd<'a> {
    pub(crate) ftype: FrameType,
    pub(crate) qis: &'a [u8],
    pub(crate) bcoded: &'a [u8],
    pub(crate) mbmodes: &'a [MacroBlockMode],
    pub(crate) mvects: &'a [MotionVector],
    pub(crate) qiis: &'a [u8],
}

/// §7.11 steps 1(f)–8 for one non-empty packet: decode the tokens
/// from `r` (positioned after the §7.6 data), undo DC prediction,
/// reconstruct, loop-filter, and promote the result into `refs`. On
/// error `refs` is untouched.
#[allow(clippy::too_many_arguments)]
pub(crate) fn finish_frame(
    r: &mut BitReader<'_>,
    fe: &FrontEnd<'_>,
    g: &FrameGeometry,
    hts: &[HuffmanTable],
    params: &QuantizationParameters,
    lflims: &[u8; 64],
    state: &mut PipelineState,
    refs: &mut ReferenceFrameStore,
) -> Result<(), Error> {
    let result = finish_frame_inner(r, fe, g, hts, params, lflims, state, refs);
    if result.is_err() {
        // A failed frame may leave coefficients behind; restore the
        // all-zero invariant.
        state.coeffs.fill([0i16; 64]);
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn finish_frame_inner(
    r: &mut BitReader<'_>,
    fe: &FrontEnd<'_>,
    g: &FrameGeometry,
    hts: &[HuffmanTable],
    params: &QuantizationParameters,
    lflims: &[u8; 64],
    state: &mut PipelineState,
    refs: &mut ReferenceFrameStore,
) -> Result<(), Error> {
    let qi0 = *fe.qis.first().ok_or(Error::ReconstructEmptyQis)?;
    let l = i32::from(lflims[(qi0 & 63) as usize]);
    let extents: [(usize, usize); 3] = core::array::from_fn(|pli| {
        (
            g.block_extent[pli].0 as usize,
            g.block_extent[pli].1 as usize,
        )
    });

    // Cut every plane into bands of whole super-block rows (four block
    // rows): a band is a contiguous run of pixel rows *and* of
    // coded-order block indices, so each worker owns its slice of the
    // output plane and of the coefficient buffer outright.
    let workers = state.threads.max(1);
    let mut bounds: [Vec<usize>; 3] = Default::default();
    for (pli, b) in bounds.iter_mut().enumerate() {
        let bh = extents[pli].1;
        let sb_rows = bh.div_ceil(4);
        let nbands = workers.min(sb_rows).max(1);
        b.extend((0..=nbands).map(|i| (sb_rows * i / nbands * 4).min(bh)));
    }

    let nlbs = (g.nmbs as usize) * 4;
    decode_tokens(
        r,
        nlbs,
        fe.bcoded,
        hts,
        &state.luts,
        &mut state.active,
        &mut state.coeffs,
        &mut state.ncoeffs,
    )?;

    for pli in 0..3 {
        undo_dc_prediction_plane(
            &g.plane_raster_order[pli],
            extents[pli].0,
            fe.bcoded,
            fe.mbmodes,
            &g.mbi_of_block,
            &mut state.coeffs,
            &mut state.dc_rows,
        );
        // Raster coded map for the loop filter (all-ones on a
        // keyframe, where every block is coded).
        let map = &mut state.coded[pli];
        if fe.ftype == FrameType::Intra {
            map.fill(1);
        } else {
            for (m, &bi) in map.iter_mut().zip(&g.plane_raster_order[pli]) {
                *m = fe.bcoded[bi as usize];
            }
        }
    }

    let mats = frame_matrices(params, fe.qis, fe.ftype == FrameType::Intra);
    let (div_cx, div_cy) = match g.pf {
        PixelFormat::Yuv420 => (4, 4),
        PixelFormat::Yuv422 => (4, 2),
        PixelFormat::Yuv444 => (2, 2),
    };
    let prev: [&[u8]; 3] = [&refs.previous_y, &refs.previous_cb, &refs.previous_cr];
    let golden: [&[u8]; 3] = [&refs.golden_y, &refs.golden_cb, &refs.golden_cr];
    let planes: [PlaneCtx<'_>; 3] = core::array::from_fn(|pli| PlaneCtx {
        pli,
        bw: extents[pli].0,
        bh: extents[pli].1,
        grid: &g.plane_raster_order[pli],
        coded: &state.coded[pli],
        prev: prev[pli],
        golden: golden[pli],
        div_x: if pli == 0 { 2 } else { div_cx },
        div_y: if pli == 0 { 2 } else { div_cy },
    });
    let frame = FrameCtx {
        bcoded: fe.bcoded,
        mbmodes: fe.mbmodes,
        mvects: fe.mvects,
        qiis: fe.qiis,
        ncoeffs: &state.ncoeffs,
        mbi_of_block: &g.mbi_of_block,
        mats: &mats,
    };

    let nthreads = bounds.iter().map(|b| b.len() - 1).max().unwrap_or(1);
    let strips: [Vec<Mutex<Strip>>; 3] = core::array::from_fn(|pli| {
        (2..bounds[pli].len())
            .map(|_| Mutex::new(Strip::default()))
            .collect()
    });
    let mut work: Vec<Vec<BandTask<'_>>> = (0..nthreads).map(|_| Vec::new()).collect();
    let mut coeff_rest: &mut [[i16; 64]] = &mut state.coeffs;
    for (p, dst) in planes.iter().zip(state.spare.iter_mut()) {
        let (mut plane_coeffs, rest) = coeff_rest.split_at_mut(p.bw * p.bh);
        coeff_rest = rest;
        let mut dst_rest: &mut [u8] = dst;
        let b = &bounds[p.pli];
        for (band, w) in b.windows(2).enumerate() {
            let (y0, y1) = (w[0], w[1]);
            let (band_dst, rest) = dst_rest.split_at_mut((y1 - y0) * 8 * p.bw * 8);
            dst_rest = rest;
            let (band_coeffs, rest) = plane_coeffs.split_at_mut((y1 - y0) * p.bw);
            plane_coeffs = rest;
            work[band].push(BandTask {
                p: *p,
                y0,
                y1,
                band,
                nbands: b.len() - 1,
                dst: band_dst,
                coeffs: band_coeffs,
                coeff_base: state.plane_base[p.pli] + y0 * p.bw,
            });
        }
    }

    if nthreads == 1 {
        run_bands(&mut work[0], &frame, None, l)?;
    } else {
        // The loop filter needs every band reconstructed first (it
        // reads the unfiltered rows across each boundary), hence the
        // barrier between the two phases. With a zero limit the filter
        // is the identity and the bands never meet.
        let barrier = Barrier::new(nthreads);
        let sync = (l != 0).then_some((&barrier, &strips));
        let frame = &frame;
        let results: Vec<Result<(), Error>> = std::thread::scope(|scope| {
            let mut iter = work.iter_mut();
            let own = iter.next().expect("at least one band");
            let handles: Vec<_> = iter
                .map(|tasks| scope.spawn(move || run_bands(tasks, frame, sync, l)))
                .collect();
            let mut results = vec![run_bands(own, frame, sync, l)];
            for h in handles {
                results.push(h.join().expect("reconstruction worker panicked"));
            }
            results
        });
        drop(work);
        for r in results {
            r?;
        }
        // Each interior boundary's lower pixel row was finished by the
        // band above it; write it back.
        for ((p, dst), plane_strips) in planes.iter().zip(state.spare.iter_mut()).zip(&strips) {
            let stride = p.bw * 8;
            for (i, strip) in plane_strips.iter().enumerate() {
                let strip = strip.lock().expect("strip lock");
                if strip.patched.len() == stride {
                    let row = bounds[p.pli][i + 1] * 8 - 1;
                    dst[row * stride..(row + 1) * stride].copy_from_slice(&strip.patched);
                }
            }
        }
    }

    // §7.11 steps 7–8: the new frame becomes the previous-frame
    // reference by swapping buffers; a keyframe also refreshes the
    // golden reference.
    let [sy, scb, scr] = &mut state.spare;
    core::mem::swap(&mut refs.previous_y, sy);
    core::mem::swap(&mut refs.previous_cb, scb);
    core::mem::swap(&mut refs.previous_cr, scr);
    if fe.ftype == FrameType::Intra {
        refs.golden_y.copy_from_slice(&refs.previous_y);
        refs.golden_cb.copy_from_slice(&refs.previous_cb);
        refs.golden_cr.copy_from_slice(&refs.previous_cr);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_data;
    use crate::{dequantize_block, inverse_dct_2d};

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
    }

    /// Natural (row-major) index of a slot.
    fn natural_of_slot(slot: usize) -> usize {
        (slot % 8) * 8 + slot / 8
    }

    /// The 16-bit lane IDCT (every zero-input specialisation) equals
    /// the step-by-step §7.9.3.2 transform, including on inputs that
    /// overflow the intermediate sums.
    #[test]
    fn lane_idct_matches_stepwise_transform() {
        let mut rng = Lcg(0x1234_5678_9abc_def0);
        for trial in 0..6000 {
            let n = 2 + (rng.next() as usize % 63);
            let amp: i32 = match trial % 4 {
                0 => 64,
                1 => 2048,
                _ => 32768,
            };
            let mut natural = [0i16; 64];
            let mut slots = [0i16; 64];
            for &slot in &SLOT_OF_ZIGZAG[..n] {
                let v = ((rng.next() % (2 * amp) as u32) as i32 - amp) as i16;
                let slot = slot as usize;
                slots[slot] = v;
                natural[natural_of_slot(slot)] = v;
            }
            let want = inverse_dct_2d(&natural);
            let (rows_upper, cols_upper) = UPPER_HALF_REACHED[n];
            assert_eq!(idct_block(&slots, rows_upper, cols_upper), want, "n={n}");
            assert_eq!(idct_block(&slots, true, true), want, "n={n} (full)");
        }
    }

    /// Slot-order storage times the slot-order frame matrix equals the
    /// §7.9.2 dequantiser (DC from the `qi0` matrix, AC from the
    /// block's own `qi`).
    #[test]
    fn slot_order_dequantiser_matches() {
        let setup = crate::decode_setup_header(&fixture_data::FIXTURE_SETUP_PACKET).unwrap();
        let params = &setup.quantization_parameters;
        let qis = [20u8, 5, 48];
        let mats = frame_matrices(params, &qis, false);
        let mut rng = Lcg(42);
        for (qti, per_plane) in mats.iter().enumerate() {
            for (pli, per_qii) in per_plane.iter().enumerate() {
                for (qii, &qi) in qis.iter().enumerate() {
                    let dc =
                        compute_quantization_matrix(params, qti, pli, qis[0] as usize).unwrap();
                    let ac = compute_quantization_matrix(params, qti, pli, qi as usize).unwrap();
                    let m = per_qii[qii].unwrap();
                    let mut zz = [0i16; 64];
                    let mut slots = [0i16; 64];
                    for (i, z) in zz.iter_mut().enumerate() {
                        *z = (rng.next() % 1200) as i16 - 600;
                        slots[SLOT_OF_ZIGZAG[i] as usize] = *z;
                    }
                    let want = dequantize_block(&zz, &dc, &ac);
                    for slot in 0..64 {
                        assert_eq!(
                            slots[slot].wrapping_mul(m[slot]),
                            want[natural_of_slot(slot)],
                            "qti={qti} pli={pli} qii={qii} slot={slot}"
                        );
                    }
                }
            }
        }
    }

    /// The encoder-side lane transforms equal the step-by-step ones
    /// on residual-range and on overflowing inputs.
    #[test]
    fn lane_transforms_match_stepwise_for_the_encoder() {
        let mut rng = Lcg(0xfeed_beef);
        for trial in 0..3000 {
            let amp: u32 = if trial % 3 == 0 { 65536 } else { 511 };
            let mut block = [[0i16; 8]; 8];
            for row in &mut block {
                for v in row.iter_mut() {
                    *v = ((rng.next() % amp) as i32 - (amp as i32) / 2) as i16;
                }
            }
            let fwd = crate::forward_dct_2d(&block);
            assert_eq!(forward_dct_block(&block), fwd);
            assert_eq!(inverse_dct_block(&fwd), inverse_dct_2d(&fwd));
        }
    }

    /// The closed-form limit equals the piecewise §7.10.1 definition.
    #[test]
    fn closed_form_limit_matches_piecewise() {
        for l in 0u8..=127 {
            for r in -300..=300 {
                assert_eq!(limit(r, i32::from(l)), crate::lflim(r, l), "r={r} l={l}");
            }
        }
    }
    /// Decode `packets` through the step-by-step §7 chain and through
    /// this pipeline at several worker budgets; every frame and the
    /// reference store after it must agree.
    fn assert_matches_stepwise(ident_pkt: &[u8], setup_pkt: &[u8], packets: &[&[u8]]) {
        use crate::{FrameDecoder, decode_identification_header, decode_setup_header};
        let ident = decode_identification_header(ident_pkt).unwrap();
        let setup = decode_setup_header(setup_pkt).unwrap();
        let mut oracle = FrameDecoder::new(ident.clone(), setup.clone()).unwrap();
        let mut fast: Vec<FrameDecoder> = [1usize, 2, 3, 4, 8, 64]
            .iter()
            .map(|&n| {
                let mut d = FrameDecoder::new(ident.clone(), setup.clone()).unwrap();
                d.set_threads(n);
                d
            })
            .collect();
        for (fi, pkt) in packets.iter().enumerate() {
            let want = oracle.decode_frame_stepwise(pkt).unwrap();
            for d in &mut fast {
                let n = d.threads();
                let got = d.decode_frame(pkt).unwrap();
                assert_eq!(got, want, "frame {fi}, {n} worker(s)");
                assert_eq!(
                    d.reference_store(),
                    oracle.reference_store(),
                    "references after frame {fi}, {n} worker(s)"
                );
            }
        }
    }

    #[test]
    fn pipeline_matches_stepwise_on_all_modes_fixture() {
        assert_matches_stepwise(
            &fixture_data::AMM_IDENT_PACKET,
            &fixture_data::PF_SETUP_PACKET,
            &[
                &fixture_data::AMM_DATA_PACKET_0,
                &fixture_data::AMM_DATA_PACKET_1,
                &fixture_data::AMM_DATA_PACKET_2,
                &fixture_data::AMM_DATA_PACKET_3,
                &fixture_data::AMM_DATA_PACKET_4,
                &fixture_data::AMM_DATA_PACKET_5,
                &fixture_data::AMM_DATA_PACKET_6,
                &fixture_data::AMM_DATA_PACKET_7,
                &fixture_data::AMM_DATA_PACKET_8,
                &fixture_data::AMM_DATA_PACKET_9,
                &fixture_data::AMM_DATA_PACKET_10,
                &fixture_data::AMM_DATA_PACKET_11,
                &fixture_data::AMM_DATA_PACKET_12,
                &fixture_data::AMM_DATA_PACKET_13,
                &fixture_data::AMM_DATA_PACKET_14,
                &fixture_data::AMM_DATA_PACKET_15,
                &fixture_data::AMM_DATA_PACKET_16,
                &fixture_data::AMM_DATA_PACKET_17,
                &fixture_data::AMM_DATA_PACKET_18,
                &fixture_data::AMM_DATA_PACKET_19,
                &fixture_data::AMM_DATA_PACKET_20,
                &fixture_data::AMM_DATA_PACKET_21,
                &fixture_data::AMM_DATA_PACKET_22,
                &fixture_data::AMM_DATA_PACKET_23,
            ],
        );
    }

    #[test]
    fn pipeline_matches_stepwise_on_422_and_444_fixtures() {
        assert_matches_stepwise(
            &fixture_data::P422_IDENT_PACKET,
            &fixture_data::PF_SETUP_PACKET,
            &[
                &fixture_data::P422_DATA_PACKET_0,
                &fixture_data::P422_DATA_PACKET_1,
                &fixture_data::P422_DATA_PACKET_2,
            ],
        );
        assert_matches_stepwise(
            &fixture_data::P444_IDENT_PACKET,
            &fixture_data::PF_SETUP_PACKET,
            &[
                &fixture_data::P444_DATA_PACKET_0,
                &fixture_data::P444_DATA_PACKET_1,
                &fixture_data::P444_DATA_PACKET_2,
            ],
        );
    }

    #[test]
    fn pipeline_matches_stepwise_on_hd_and_empty_packet_fixtures() {
        assert_matches_stepwise(
            &fixture_data::HD1080_IDENT_PACKET,
            &fixture_data::HD1080_SETUP_PACKET,
            &[
                &fixture_data::HD1080_DATA_PACKET_0,
                &fixture_data::HD1080_DATA_PACKET_1,
            ],
        );
        assert_matches_stepwise(
            &fixture_data::KI30_IDENT_PACKET,
            &fixture_data::FIXTURE_SETUP_PACKET,
            &[
                &fixture_data::KI30_DATA_PACKET_0,
                &fixture_data::KI30_DATA_PACKET_1,
                &fixture_data::KI30_DATA_PACKET_2,
                &fixture_data::KI30_DATA_PACKET_3,
                &fixture_data::KI30_DATA_PACKET_4,
                &fixture_data::KI30_DATA_PACKET_5,
                &fixture_data::KI30_DATA_PACKET_6,
                &fixture_data::KI30_DATA_PACKET_7,
            ],
        );
    }

    /// The prefix tables resolve every code of at most `LUT_BITS` bits
    /// to the token and length the tree walk finds, and leave longer
    /// codes to the walk.
    #[test]
    fn token_luts_agree_with_tree_walk() {
        let setup = crate::decode_setup_header(&fixture_data::FIXTURE_SETUP_PACKET).unwrap();
        let luts = TokenLuts::build(&setup.huffman_tables[..]);
        for (table, lut) in setup.huffman_tables.iter().zip(&luts.tables) {
            let mut resolved = 0usize;
            for e in &table.entries {
                if u32::from(e.len) <= LUT_BITS {
                    let shift = LUT_BITS - u32::from(e.len);
                    for low in 0..(1u32 << shift) {
                        let got = lut[((e.code << shift) | low) as usize];
                        assert_eq!(got, u16::from(e.token) | (u16::from(e.len) << 5));
                    }
                    resolved += 1usize << shift;
                }
            }
            let misses = lut.iter().filter(|&&e| e == LUT_MISS).count();
            assert_eq!(resolved + misses, LUT_SIZE);
        }
    }
}
