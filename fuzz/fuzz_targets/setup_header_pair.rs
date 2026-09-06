#![no_main]

//! Structured §6.4 setup-header writer / parser pair.
//!
//! `decode_headers` throws arbitrary bytes at the parser; this target
//! drives the *writer* with arbitrary **valid** tables instead. Fuzz
//! bytes perturb the encoder's default tables — the §6.4.1 loop-filter
//! limits (any 7-bit value, including the all-zero `NBITS = 0` table),
//! the §6.4.2 AC / DC scale ladders (any 16-bit value), the base
//! matrices (any 1..=255 entry, 1..=8 of them), and a fresh
//! quant-range layout per `(qti, pli)` (1..=4 ranges whose sizes sum
//! to 63, base-matrix indices within `NBMS`) — then assert the pair's
//! contract on every such bundle:
//!
//! * `encode_setup_header` must accept the tables (they are valid by
//!   construction);
//! * `parse_setup_header` must accept the bytes;
//! * `decode_setup_header` must reproduce the tables **exactly**
//!   (`SetupHeaderTables: PartialEq`), and a second encode of the
//!   decoded tables must be byte-identical to the first (the writer is
//!   canonical on its own output);
//! * `compute_quantization_matrix` must succeed for every `(qti, pli,
//!   qi)` on the decoded tables — a layout the writer emits is one the
//!   §6.4.3 interpolation can always evaluate.

use libfuzzer_sys::fuzz_target;
use oxideav_theora::{
    compute_quantization_matrix, decode_setup_header, encode_setup_header, parse_setup_header,
    SetupHeaderTables,
};

fuzz_target!(|data: &[u8]| {
    let mut setup = SetupHeaderTables::encoder_defaults();
    let mut at = 0usize;
    let mut next = || -> u8 {
        let b = data.get(at).copied().unwrap_or(0);
        at += 1;
        b
    };

    // §6.4.1 loop-filter limits: 7-bit values; a zero first byte
    // selects the all-zero table (NBITS = 0 on the wire).
    if next() != 0 {
        for v in setup.loop_filter_limits.iter_mut() {
            *v = next() & 0x7f;
        }
    } else {
        setup.loop_filter_limits = [0; 64];
    }

    // §6.4.2 scale ladders (16-bit, unconstrained by the syntax).
    let qp = &mut setup.quantization_parameters;
    if next() & 1 != 0 {
        for v in qp.ac_scale.iter_mut() {
            *v = u16::from_le_bytes([next(), next()]);
        }
        for v in qp.dc_scale.iter_mut() {
            *v = u16::from_le_bytes([next(), next()]);
        }
    }

    // Base matrices: 1..=8 of them, entries 1..=255.
    let nbms = 1 + (next() % 8) as usize;
    let mut bms: Vec<[u8; 64]> = Vec::with_capacity(nbms);
    for i in 0..nbms {
        let mut m = if i < qp.base_matrices.len() {
            qp.base_matrices[i]
        } else {
            [16u8; 64]
        };
        if next() & 1 != 0 {
            for v in m.iter_mut() {
                *v = next().max(1);
            }
        }
        bms.push(m);
    }
    qp.num_base_matrices = nbms as u16;
    qp.base_matrices = bms;

    // Quant-range layouts: per (qti, pli), 1..=4 ranges summing to
    // 63, base-matrix indices within NBMS.
    for qti in 0..2 {
        for pli in 0..3 {
            let nqr = 1 + (next() % 4) as usize;
            let mut sizes = [0u8; 63];
            let mut left = 63u32;
            for r in 0..nqr {
                let remaining_ranges = (nqr - r) as u32;
                let size = if r + 1 == nqr {
                    left
                } else {
                    // Leave at least one for each later range.
                    let max = left - (remaining_ranges - 1);
                    1 + (next() as u32 % max)
                };
                sizes[r] = size as u8;
                left -= size;
            }
            let mut bmis = [0u16; 64];
            for b in bmis.iter_mut().take(nqr + 1) {
                *b = (next() as usize % nbms) as u16;
            }
            qp.num_quant_ranges[qti][pli] = nqr as u8;
            qp.quant_range_sizes[qti][pli] = sizes;
            qp.quant_range_base_matrix_indices[qti][pli] = bmis;
        }
    }

    let bytes = encode_setup_header(&setup).expect("valid tables must serialize");
    parse_setup_header(&bytes).expect("the writer's output must parse");
    let back = decode_setup_header(&bytes).expect("the writer's output must decode");
    assert_eq!(back, setup, "§6.4 encode∘decode must be the identity on valid tables");
    let again = encode_setup_header(&back).expect("decoded tables must re-serialize");
    assert_eq!(again, bytes, "the writer must be canonical on its own output");
    for qti in 0..2 {
        for pli in 0..3 {
            for qi in 0..64 {
                compute_quantization_matrix(&back.quantization_parameters, qti, pli, qi)
                    .expect("every (qti, pli, qi) must evaluate on a written layout");
            }
        }
    }
});
