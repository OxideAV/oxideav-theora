# Self-encoded corpus — generation + external-validation notes (round 413)

`encoded_corpus.rs` pins eleven deterministic encoder scenarios by
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

## Rules

* Digest changes must be intentional, explained in the commit message,
  and accompanied by a re-run of the external route above when the
  wire moved.
* The generator and scenario parameters are part of the pin: changing
  either is a re-pin, not a fix.
