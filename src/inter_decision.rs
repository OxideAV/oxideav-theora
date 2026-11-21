//! Rate-distortion mode decision for inter frames, split so that the
//! expensive work is a pure per-macro-block function.
//!
//! The §7.5.2 motion-vector syntax couples macro blocks through its
//! running `LAST1` / `LAST2` predictors: whether an `INTER_MV` vector is
//! free (it re-spells as `INTER_MV_LAST` / `INTER_MV_LAST2`) depends on
//! the decisions of every macro block before it in coded order. Only
//! that *bit charge* is order dependent, though. The motion searches,
//! the transform / quantise / reconstruct pricing of every candidate
//! mode, the intra alternative and the four-vector alternative depend
//! on nothing but the macro block itself, the reference frames, and
//! hints fixed before the frame starts. So the decision runs in two
//! stages:
//!
//! 1. **Analysis** — for each macro block independently: motion search
//!    on the previous and golden references, then distortion and token
//!    rate of every candidate that can be named without knowing the
//!    predictors. Macro blocks are analysed concurrently under a caller
//!    budget.
//! 2. **Resolution** — one serial sweep in coded order that tracks
//!    `LAST1` / `LAST2`, adds the motion-vector bit charge each
//!    candidate would actually pay, prices the two predictor vectors
//!    themselves (from the analysis when it happened to cover them,
//!    otherwise on the spot), and picks the cheapest mode.
//!
//! Every per-macro-block result is a function of its inputs alone and
//! the resolution sweep is serial, so the plan — and therefore the
//! packet — is identical at every worker budget.

use super::{
    Error, FrameEncoder, MacroBlockMode, MotionVector, QuantizationMatrix, ReferenceFrame,
    ReferencePlaneSet, SourceFrame, TokenBitCosts, isqrt_u64, mv_huffman_bits,
};

/// Run `f(i)` for `i` in `0..n` on up to `threads` workers and return
/// the results in index order. Each worker takes one contiguous index
/// range; with `threads <= 1` (or a single item) everything runs on
/// the calling thread.
pub(crate) fn par_map<T, F>(threads: usize, n: usize, f: F) -> Vec<T>
where
    T: Send,
    F: Fn(usize) -> T + Sync,
{
    let workers = threads.min(n).max(1);
    if workers == 1 {
        return (0..n).map(f).collect();
    }
    let bounds: Vec<usize> = (0..=workers).map(|w| n * w / workers).collect();
    let f = &f;
    std::thread::scope(|scope| {
        let handles: Vec<_> = bounds
            .windows(2)
            .skip(1)
            .map(|w| {
                let (lo, hi) = (w[0], w[1]);
                scope.spawn(move || (lo..hi).map(f).collect::<Vec<T>>())
            })
            .collect();
        let mut out: Vec<T> = (bounds[0]..bounds[1]).map(f).collect();
        out.reserve(n - out.len());
        for h in handles {
            out.extend(h.join().expect("analysis worker panicked"));
        }
        out
    })
}

/// The three most frequent vectors of a field (ties go to the vector
/// met first in iteration order; unused slots are zero).
fn most_common_vectors(field: impl Iterator<Item = MotionVector>) -> [MotionVector; 3] {
    // Components lie in -31..=31 (§7.5); anything else is not counted.
    let mut counts = vec![0u32; 63 * 63];
    let mut order: Vec<MotionVector> = Vec::new();
    for mv in field {
        let (x, y) = (mv.x as i32 + 31, mv.y as i32 + 31);
        if (0..63).contains(&x) && (0..63).contains(&y) {
            let slot = &mut counts[(x * 63 + y) as usize];
            if *slot == 0 {
                order.push(mv);
            }
            *slot += 1;
        }
    }
    let count_of = |mv: MotionVector| counts[((mv.x as i32 + 31) * 63 + mv.y as i32 + 31) as usize];
    let mut top = [MotionVector::ZERO; 3];
    let mut taken = [false; 3];
    for &mv in &order {
        // Insert into the running top three, keeping earlier vectors
        // ahead of later ones with the same count.
        let c = count_of(mv);
        let mut at = 3;
        for k in 0..3 {
            if !taken[k] || c > count_of(top[k]) {
                at = k;
                break;
            }
        }
        if at < 3 {
            for k in (at + 1..3).rev() {
                top[k] = top[k - 1];
                taken[k] = taken[k - 1];
            }
            top[at] = mv;
            taken[at] = true;
        }
    }
    top
}

/// Distortion and rate of one uniform-vector candidate, before the
/// §7.5 motion-vector bits (which depend on the `LAST` predictors).
#[derive(Clone, Copy)]
struct UniformCost {
    mode: MacroBlockMode,
    mv: MotionVector,
    distortion: u64,
    /// Token bits plus the §7.4 mode code.
    rate: u32,
}

/// Everything about one macro block that can be computed without
/// knowing the decisions of its predecessors.
struct MbAnalysis {
    prev_mv: MotionVector,
    /// Candidates priced ahead of the resolution sweep.
    uniform: [Option<UniformCost>; 6],
    intra_cost: u64,
    four_cost: u64,
    four_luma: [MotionVector; 4],
}

impl MbAnalysis {
    fn lookup(&self, mode: MacroBlockMode, mv: MotionVector) -> Option<UniformCost> {
        self.uniform
            .iter()
            .flatten()
            .find(|c| c.mode == mode && c.mv == mv)
            .copied()
    }
}

/// The mode decision's output: one vector and mode per macro block,
/// plus the four luma vectors of `INTER_MV_FOUR` macro blocks.
pub(crate) struct ModeDecision {
    pub(crate) mb_mv: Vec<MotionVector>,
    pub(crate) mbmodes: Vec<MacroBlockMode>,
    pub(crate) four_mv_luma: Vec<Option<[MotionVector; 4]>>,
    /// The previous-reference vector each macro block's search settled
    /// on, whatever mode won — the next frame's search seeds.
    pub(crate) searched_prev: Vec<MotionVector>,
}

impl FrameEncoder {
    /// The rate-distortion mode decision of an inter frame (see the
    /// module docs). `qmats` / `qmats_intra` are the per-plane inter /
    /// intra matrices at the frame's first `qi`; `lambda` prices one
    /// bit in squared-error units.
    pub(crate) fn decide_modes_rd(
        &self,
        frame: &SourceFrame,
        refs: &ReferencePlaneSet<'_>,
        qmats: &[QuantizationMatrix; 3],
        qmats_intra: &[QuantizationMatrix; 3],
        token_costs: &TokenBitCosts,
        lambda: u64,
    ) -> Result<ModeDecision, Error> {
        let g = &self.geometry;
        let nmbs = g.nmbs as usize;
        // SAD-domain λ for the in-search MV rate penalty: λ prices SSD
        // per bit; per-sample absolute error scales as its square root
        // over the 256-sample macro block, so √λ is the right order.
        let lambda_sad = isqrt_u64(lambda) as u32;

        // Right after a keyframe the previous frame *is* the golden
        // frame: every golden candidate then duplicates a
        // previous-reference one (same predictor, never a cheaper
        // spelling), so the golden search and candidates are skipped.
        let golden_distinct = refs.golden_y.samples != refs.previous_y.samples
            || refs.golden_cb.samples != refs.previous_cb.samples
            || refs.golden_cr.samples != refs.previous_cr.samples;

        // --- Stage 1: per-macro-block analysis. ----------------------
        // Motion search first, on its own, so the pricing step can see
        // the vectors of neighbouring macro blocks.
        let hint = |mbi: usize| -> MotionVector {
            self.motion_hint
                .as_deref()
                .and_then(|h| h.get(mbi))
                .copied()
                .unwrap_or(MotionVector::ZERO)
        };
        let first_pass: Vec<(MotionVector, MotionVector)> = par_map(self.threads, nmbs, |mbi| {
            // Seeds fixed before the frame starts: the vector this
            // macro block and its coded-order neighbours ended up with
            // in the previous inter frame.
            let seeds = [
                hint(mbi),
                hint(mbi.wrapping_sub(1)),
                hint(mbi + 1),
                hint(mbi.wrapping_sub(2)),
            ];
            let (prev_mv, _) = self.search_macro_block_mv_seeded_priced(
                frame,
                refs,
                mbi,
                ReferenceFrame::Previous,
                &seeds,
                lambda_sad,
            );
            let gold_mv = if golden_distinct {
                self.search_macro_block_mv_seeded_priced(
                    frame,
                    refs,
                    mbi,
                    ReferenceFrame::Golden,
                    &[],
                    lambda_sad,
                )
                .0
            } else {
                MotionVector::ZERO
            };
            (prev_mv, gold_mv)
        });
        // Second pass: every macro block may adopt a vector that won
        // elsewhere — its four neighbours' and the frame's most common
        // ones — when that vector is cheaper for it too. A descent
        // that stopped in a poor local minimum is pulled onto the
        // motion its surroundings agree on, which is what makes the
        // LAST1 / LAST2 spellings (and smooth fields) reachable. Both
        // inputs are first-pass results, so the pass is still a pure
        // per-macro-block function.
        let common_prev = most_common_vectors(first_pass.iter().map(|&(p, _)| p));
        let common_gold = most_common_vectors(first_pass.iter().map(|&(_, g)| g));
        let searched: Vec<(MotionVector, MotionVector)> = par_map(self.threads, nmbs, |mbi| {
            let (prev0, gold0) = first_pass[mbi];
            let around = self.mb_neighbors[mbi];
            let gather = |pick: fn(&(MotionVector, MotionVector)) -> MotionVector,
                          common: &[MotionVector; 3]| {
                let mut cands = [MotionVector::ZERO; 7];
                for (slot, &n) in cands.iter_mut().zip(around.iter()) {
                    if let Some(v) = first_pass.get(n as usize) {
                        *slot = pick(v);
                    }
                }
                cands[4..].copy_from_slice(common);
                cands
            };
            let prev_mv = self.adopt_better_macro_block_mv(
                frame,
                refs,
                mbi,
                ReferenceFrame::Previous,
                prev0,
                &gather(|v| v.0, &common_prev),
                lambda_sad,
            );
            let gold_mv = if golden_distinct {
                self.adopt_better_macro_block_mv(
                    frame,
                    refs,
                    mbi,
                    ReferenceFrame::Golden,
                    gold0,
                    &gather(|v| v.1, &common_gold),
                    lambda_sad,
                )
            } else {
                MotionVector::ZERO
            };
            (prev_mv, gold_mv)
        });

        let analysis: Vec<Result<MbAnalysis, Error>> = par_map(self.threads, nmbs, |mbi| {
            let (prev_mv, gold_mv) = searched[mbi];
            let mut uniform: [Option<UniformCost>; 6] = [None; 6];
            let mut n = 0usize;
            let mut price = |mode: MacroBlockMode, mv: MotionVector| -> Result<(), Error> {
                if uniform[..n]
                    .iter()
                    .flatten()
                    .any(|c| c.mode == mode && c.mv == mv)
                {
                    return Ok(());
                }
                let (distortion, rate) = self.mb_uniform_mode_parts(
                    frame,
                    refs,
                    qmats,
                    token_costs,
                    mbi,
                    mode,
                    mv,
                    lambda,
                )?;
                uniform[n] = Some(UniformCost {
                    mode,
                    mv,
                    distortion,
                    rate,
                });
                n += 1;
                Ok(())
            };
            price(MacroBlockMode::InterNoMv, MotionVector::ZERO)?;
            if prev_mv != MotionVector::ZERO {
                price(MacroBlockMode::InterMv, prev_mv)?;
            }
            if golden_distinct {
                price(MacroBlockMode::InterGoldenNoMv, MotionVector::ZERO)?;
                if gold_mv != MotionVector::ZERO {
                    price(MacroBlockMode::InterGoldenMv, gold_mv)?;
                }
            }
            // The likeliest values of LAST1 / LAST2 when the sweep
            // reaches this macro block: the vectors found for its two
            // coded-order predecessors. Pricing them here keeps that
            // work off the serial sweep (which prices any predictor
            // this guess misses).
            for back in 1..=2usize {
                if let Some(&(v, _)) = mbi.checked_sub(back).and_then(|i| searched.get(i))
                    && v != MotionVector::ZERO
                {
                    price(MacroBlockMode::InterMv, v)?;
                }
            }
            let intra_cost =
                self.mb_intra_mode_cost(frame, qmats_intra, token_costs, mbi, lambda)?;
            // Per-block vectors start from the motions present around
            // this macro block — its own and its four neighbours' — and
            // are refined locally: a block that needs its own vector
            // sits on a boundary between two of those motions.
            let mut four_seeds = [prev_mv; 5];
            for (slot, &n) in four_seeds[1..]
                .iter_mut()
                .zip(self.mb_neighbors[mbi].iter())
            {
                if let Some(&(v, _)) = searched.get(n as usize) {
                    *slot = v;
                }
            }
            let four_luma = self.mb_four_mv_vectors(frame, refs, mbi, &four_seeds, lambda_sad);
            let four_cost = if four_luma.iter().all(|&v| v == four_luma[0]) {
                // Four equal vectors predict exactly what the uniform
                // mode with that vector predicts (the §7.5.2 chroma
                // average of equal vectors is the vector), at a
                // strictly higher header cost: never the cheapest
                // spelling, so not worth pricing.
                u64::MAX
            } else {
                self.mb_four_mv_cost(frame, refs, qmats, token_costs, mbi, lambda, four_luma)?
            };
            Ok(MbAnalysis {
                prev_mv,
                uniform,
                intra_cost,
                four_cost,
                four_luma,
            })
        });

        // --- Stage 2: resolution sweep in coded order. ----------------
        let mut mb_mv = vec![MotionVector::ZERO; nmbs];
        let mut mbmodes = vec![MacroBlockMode::InterNoMv; nmbs];
        let mut four_mv_luma: Vec<Option<[MotionVector; 4]>> = vec![None; nmbs];
        let mut last1 = MotionVector::ZERO;
        let mut last2 = MotionVector::ZERO;
        for (mbi, a) in analysis.into_iter().enumerate() {
            let a = a?;
            let mut best_mode = MacroBlockMode::InterNoMv;
            let mut best_mv = MotionVector::ZERO;
            let mut best_cost = u64::MAX;
            let mut consider = |c: UniformCost| {
                // An `INTER_MV` whose vector matches a running
                // predictor re-spells as a LAST mode and spends no MV
                // bits; the golden explicit mode always pays; the
                // zero-vector modes carry none.
                let mv_bits = match c.mode {
                    MacroBlockMode::InterMv if c.mv == last1 || c.mv == last2 => 0,
                    MacroBlockMode::InterMv | MacroBlockMode::InterGoldenMv => {
                        mv_huffman_bits(c.mv)
                    }
                    _ => 0,
                };
                let cost = c.distortion + lambda.saturating_mul(u64::from(c.rate + mv_bits));
                if cost < best_cost {
                    best_cost = cost;
                    best_mode = c.mode;
                    best_mv = c.mv;
                }
            };
            // The searched candidates, in a fixed order (ties keep the
            // earlier one) ...
            for (mode, mv) in [
                (MacroBlockMode::InterNoMv, MotionVector::ZERO),
                (MacroBlockMode::InterMv, a.prev_mv),
                (MacroBlockMode::InterGoldenNoMv, MotionVector::ZERO),
                (MacroBlockMode::InterGoldenMv, searched[mbi].1),
            ] {
                if let Some(c) = a.lookup(mode, mv) {
                    consider(c);
                }
            }
            // ... then the predictors themselves: a vector that lost
            // the search by a few SAD units still wins the
            // rate-distortion race when it costs no bits. The analysis
            // priced its guess at them; anything it missed is priced
            // here.
            for seed in [last1, last2] {
                if seed == MotionVector::ZERO || seed == a.prev_mv {
                    continue;
                }
                let c = match a.lookup(MacroBlockMode::InterMv, seed) {
                    Some(c) => c,
                    None => {
                        let (distortion, rate) = self.mb_uniform_mode_parts(
                            frame,
                            refs,
                            qmats,
                            token_costs,
                            mbi,
                            MacroBlockMode::InterMv,
                            seed,
                            lambda,
                        )?;
                        UniformCost {
                            mode: MacroBlockMode::InterMv,
                            mv: seed,
                            distortion,
                            rate,
                        }
                    }
                };
                consider(c);
            }
            if a.intra_cost < best_cost {
                best_cost = a.intra_cost;
                best_mode = MacroBlockMode::Intra;
                best_mv = MotionVector::ZERO;
            }
            let chosen = if a.four_cost < best_cost {
                four_mv_luma[mbi] = Some(a.four_luma);
                MacroBlockMode::InterMvFour
            } else {
                mb_mv[mbi] = best_mv;
                best_mode
            };
            mbmodes[mbi] = chosen;

            // Advance the predictors the way the final §7.5.2 recode
            // will: a reused vector moves to the front, a new explicit
            // vector pushes the history, a four-vector macro block
            // contributes its (approximate) representative vector.
            match chosen {
                MacroBlockMode::InterMv => {
                    if best_mv == last1 {
                    } else if best_mv == last2 {
                        std::mem::swap(&mut last1, &mut last2);
                    } else {
                        last2 = last1;
                        last1 = best_mv;
                    }
                }
                MacroBlockMode::InterMvFour => {
                    last2 = last1;
                    last1 = a.prev_mv;
                }
                _ => {}
            }
        }

        Ok(ModeDecision {
            mb_mv,
            mbmodes,
            four_mv_luma,
            searched_prev: searched.iter().map(|&(prev, _)| prev).collect(),
        })
    }
}
