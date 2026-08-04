#![no_main]

//! §A.2.3 granule-position mapping properties on fuzz-derived
//! values.
//!
//! The mapping is pure integer arithmetic, so the oracle is
//! algebraic: for any `(granule_position, kfgshift)` the split /
//! join pair must round-trip, `granule_position_for_frame` must be
//! consistent with its three inverses, and a
//! [`GranulePositionTracker`] must emit non-decreasing granule
//! positions whose splits reproduce the keyframe flags that were
//! pushed. Every function must `return` a `Result` for any input —
//! out-of-range `kfgshift` (> 31), overflowing keyframe counts, and
//! offsets outrunning the shift are `Err`, never a panic.

use libfuzzer_sys::fuzz_target;
use oxideav_theora::{
    frame_index_from_granule_position, granule_position_for_frame, join_granule_position,
    keyframe_index_from_granule_position, split_granule_position, GranulePositionTracker,
};

fuzz_target!(|data: &[u8]| {
    if data.len() < 10 {
        return;
    }
    let gp = u64::from_be_bytes(data[..8].try_into().unwrap());
    let kfgshift = data[8]; // deliberately unclamped: >31 must Err cleanly

    // --- split/join round-trip -------------------------------------
    match split_granule_position(gp, kfgshift) {
        Ok(split) => {
            assert!(kfgshift <= 31, "split accepted an out-of-range kfgshift");
            if gp <= i64::MAX as u64 {
                let back = join_granule_position(split, kfgshift)
                    .expect("join of a split carriable granule position");
                assert_eq!(back, gp, "join∘split must be the identity");
            }
        }
        Err(_) => assert!(kfgshift > 31, "split rejected an in-range kfgshift"),
    }

    // --- forward mapping vs inverses -------------------------------
    let shift = kfgshift % 32;
    let frame_index = gp % (1u64 << 40); // keep well inside carriable range
    let back_span = u64::from(data[9]);
    let keyframe_index = frame_index.saturating_sub(back_span);
    if let Ok(fgp) = granule_position_for_frame(frame_index, keyframe_index, shift) {
        let split =
            split_granule_position(fgp, shift).expect("forward-mapped value must split");
        assert_eq!(
            split.keyframe_count,
            keyframe_index + 1,
            "§A.2.3 high half is the one-based keyframe index"
        );
        assert_eq!(
            split.frames_since_keyframe,
            frame_index - keyframe_index,
            "§A.2.3 low half is the offset from the keyframe"
        );
        assert_eq!(
            frame_index_from_granule_position(fgp, shift)
                .expect("forward-mapped value marks a frame"),
            frame_index,
            "frame-index inverse must undo the forward mapping"
        );
        assert_eq!(
            keyframe_index_from_granule_position(fgp, shift)
                .expect("forward-mapped value has a seek anchor"),
            keyframe_index,
            "keyframe-index inverse must undo the forward mapping"
        );
    }

    // --- tracker over a fuzz keyframe-flag sequence ----------------
    if let Ok(mut tracker) = GranulePositionTracker::new(shift) {
        let mut last: Option<u64> = None;
        for (i, &b) in data[10..].iter().take(64).enumerate() {
            // The first pushed frame must be a keyframe for the push
            // to succeed; after that the flag is fuzz-chosen.
            let is_keyframe = i == 0 || b & 1 == 1;
            match tracker.push_frame(is_keyframe) {
                Ok(gp) => {
                    if let Some(prev) = last {
                        assert!(
                            gp > prev,
                            "tracker granule positions must strictly increase"
                        );
                    }
                    last = Some(gp);
                    let split = split_granule_position(gp, shift)
                        .expect("tracker output must split");
                    if is_keyframe {
                        assert_eq!(
                            split.frames_since_keyframe, 0,
                            "a keyframe's low half is zero"
                        );
                    }
                }
                Err(_) => {
                    // Legal only when the keyframe interval outran the
                    // shift's representable offset (or the index
                    // overflowed, unreachable in 64 pushes).
                    assert!(
                        !is_keyframe,
                        "a keyframe granule position is always representable"
                    );
                    break;
                }
            }
        }
    }
});
