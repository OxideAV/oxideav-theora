# Self-encoded corpus — generation + external-validation notes (rounds 413 / 437 / 444)

`encoded_corpus.rs` pins twelve deterministic encoder scenarios by
SHA-256, at two levels per scenario:

* **wire** — the length-prefixed packet chain (`u32` LE length + bytes,
  three §6 header packets then every §7 data packet, in emission order);
* **reconstruction** — this crate's own `TheoraDecoder` output for that
  chain (§2.2-cropped, top-down planar frames, concatenated).

The corpus is *self-generating*: the test rebuilds every stream from the
in-file content generator on each run, so there are no binary fixtures
to stage. `CORPUS_PRINT=1 cargo test --test encoded_corpus -- --nocapture`
prints fresh `Pin` lines when a deliberate encoder change re-spells the
streams; update the digests in the same commit as the change.

## What the digests were validated against

In round 413 the same scenario family (at larger sizes: 320×240 /
176×144 / 160×128 / 130×98 / 128×96) was validated through an
**external, black-box decoder**:

1. Each scenario's packets were produced by `TheoraEncoder` exactly as
   in the test (same constructors, same options).
2. A throwaway helper project (kept outside this crate — this crate has
   no container dependency) wrapped the packets into `.ogv` using the
   published `oxideav-ogg` 0.1.8 crate: the three header packets
   Xiph-laced into `CodecParameters::extradata`, each data packet's
   `Packet::pts` set to its **frame index** (the muxer derives §A.2.3
   granule positions itself — pre-packing a granule double-packs it),
   `unit_boundary` on keyframes, 4096-byte page target.
3. Every `.ogv` passed `oggz-validate` (page framing / CRC / granule
   monotonicity).
4. Every `.ogv` was decoded black-box with `ffmpeg` 8.1
   (`-f rawvideo -pix_fmt yuv420p|yuv422p|yuv444p`) and the raw planar
   output compared byte-for-byte against this crate's own decoder
   reconstruction of the same packets.

**Verdict: 11/11 scenarios pixel-exact** (all three pixel formats, the
non-MB-aligned picture region with its odd 65×49-class chroma window,
multi-`qi` adaptive-quantization streams, rate-controlled streams with
the `NOMBR` declaration, GOP-tuned custom Huffman setup headers, scene
cuts, four-MV and golden-frame inter strategies, and zero-byte
duplicate-frame packets).

One presentation-level caveat, not a wire defect: for a stream whose
zero-byte duplicate packets sit between two coded frames, a player's
constant-frame-rate resampler may fill a duplicated presentation slot
with the *nearest* decoded frame (the upcoming keyframe) where this
crate's decoder emits the §7.11 previous-frame copy; with
timestamp-passthrough decoding the coded frames themselves match
byte-for-byte.

## Externally-measured operating points (round 413, 320×240, 30 fps)

* Fixed-`qi` I+P (interval 8, 16 frames), luma PSNR via the external
  decode: monotone from 29.5 dB (qi 8) to 51.4 dB (qi 63) with monotone
  rate — the pre-round-413 λ ramp folded this curve over above qi 52.
* Target-bitrate loop, 150-frame runs: 400 kb/s target → +0.5 %
  measured, 800 kb/s → −0.9 %; a 200 kb/s target is below this
  content's qi-0 rate floor (~310 kb/s) and saturates cleanly at
  `qi_min`.

## Round 437 — decode-corner scenarios

Three further pins (`encoded_corpus_decode_corner_digests_are_stable`)
target wire states the staged fixture corpus never reaches on the
decode side. The self-encoded stream is the vehicle: the encoder is a
validated stream source, so exotic setup-header configurations become
real decodable streams.

* **`lflims127`** — every §6.4.1 loop-filter limit forced to 127, so
  the serialized table is 7 bits wide (asserted on the setup packet)
  and `lflim()` runs at the ceiling `L = 127` on every edge of an
  I+P GOP. The staged fixtures only exercise limits 0 and 15.
* **`lflims0`** — the all-zero limit table serializes at `NBITS = 0`:
  sixty-four §5.2.5 *zero-bit* integer reads on the wire (asserted),
  and the §7.10 loop filter is skipped at every qi.
* **`multiqrange`** — three transmitted quant ranges per `(qti, pli)`
  set (21 + 21 + 21, alternating base matrices) plus adaptive
  quantization with candidate qis 40 / 10 / 60, so §6.4.3
  interpolation crosses interior boundaries of a non-VP3 range layout
  on both frame types and on several qis per frame.

External validation (2026-08-04): the exact pinned streams
(176×144, 8 frames, interval 6) were muxed to `.ogv` through the
published `oxideav-ogg` crate (Xiph-laced extradata, frame-index pts,
keyframe unit boundaries, 4096-byte page target), passed
`oggz-validate`, and were black-box decoded by `ffmpeg` 8.1
(`-f rawvideo -pix_fmt yuv420p`); all three matched this crate's own
reconstruction **byte-for-byte** (304128 bytes each). Verdict:
3/3 pixel-exact.

## Rules

* Digest changes must be intentional, explained in the commit message,
  and accompanied by a re-run of the external route above when the
  wire moved.
* The generator and scenario parameters are part of the pin: changing
  either is a re-pin, not a fix.

## Round 444 — the composed rate-control + adaptive-quant scenario

Round 444 made `with_target_bitrate` and `with_adaptive_quant`
compose (the loop owns each adaptive frame's `QIS[0]`; the caller's
candidates ride as the per-block AC alternatives) — previously the
loop was inert whenever adaptive quantization was enabled, so this
stream shape did not exist. Scenario 12 (`rcadaptive`: 176×144
4:2:0, interval 8, 16 frames, `qis = [40, 24, 56]`, 150 kb/s
target) pins it, and the round-413 route was re-run on it end to
end: packets rebuilt exactly as in the test, muxed to `.ogv`
through the sibling container crate (same recipe — Xiph-laced
extradata, `pts` = frame index, `unit_boundary` on keyframes,
4096-byte page target), `oggz-validate` clean, black-box-decoded to
raw planar video, byte-compared against this crate's own
`TheoraDecoder` reconstruction.

**Verdict: pixel-exact (16/16 scenarios cumulative across rounds
413 + 437 + 444).** The wire evidence of composition is asserted in
the test itself independent of the digests: every data frame
carries a multi-entry `QIS` list and at least one frame's `QIS[0]`
leaves the seed quantizer.
