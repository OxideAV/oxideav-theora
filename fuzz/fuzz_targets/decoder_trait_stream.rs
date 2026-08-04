#![no_main]

//! Hostile bytes through the framework decode path: the
//! `make_decoder` factory's length-prefixed extradata walk, then a
//! packet stream through `oxideav_core::Decoder::send_packet` /
//! `receive_frame`.
//!
//! # Input framing
//!
//! ```text
//! [0..2)   u16-BE  N = extradata length
//! [2..2+N)         CodecParameters::extradata bytes (the factory's
//!                  own u16-BE length-prefixed §6 header chain)
//! [2+N..)          up to 8 u16-BE length-prefixed stream packets
//! ```
//!
//! Splitting the buffer this way lets one mutation perturb either
//! the extradata chain (factory parse, header-order collection,
//! ident/setup rejection) or the live packet stream (inline
//! headers, reserved header types `0x83`–`0xFF`, video data,
//! zero-byte duplicate markers) independently.
//!
//! # Why the geometry cap
//!
//! `FrameDecoder` allocates its reference planes at the coded
//! dimensions the §6.2 identification header declares. `FMBW` /
//! `FMBH` are 16-bit, so a valid-but-enormous header (65535×65535
//! macro blocks ≈ 10⁶×10⁶ pixels) is a legitimate *resource*
//! request, not a decoder bug — letting the allocator OOM on it
//! would mask the logic bugs this harness exists to find. The
//! harness therefore pre-scans every packet (extradata chain and
//! stream alike) and drops identification headers declaring more
//! than [`MAX_CODED_PIXELS`], mirroring the sanity limit a real
//! demuxer applies. The library itself is deliberately left free of
//! an arbitrary built-in size policy.

use libfuzzer_sys::fuzz_target;
use oxideav_core::{CodecId, CodecParameters, Packet, TimeBase};
use oxideav_theora::{decode_identification_header, make_decoder, THEORA_CODEC_ID};

/// Upper bound on the declared coded-frame area (≈ 4 Mpixel — HD
/// class). Larger declarations are resource requests the harness
/// skips.
const MAX_CODED_PIXELS: u64 = 1 << 22;

/// Cap on stream packets per input (each accepted video packet is a
/// full reconstruction pass).
const MAX_PACKETS: usize = 8;

/// True when `pkt` is an identification header declaring a coded
/// area past [`MAX_CODED_PIXELS`].
fn declares_oversized_geometry(pkt: &[u8]) -> bool {
    if pkt.first() != Some(&0x80) {
        return false;
    }
    match decode_identification_header(pkt) {
        Ok(ident) => {
            (ident.coded_width() as u64) * (ident.coded_height() as u64) > MAX_CODED_PIXELS
        }
        Err(_) => false,
    }
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 {
        return;
    }
    let n = u16::from_be_bytes([data[0], data[1]]) as usize;
    let rest = &data[2..];
    let n = n.min(rest.len());
    let (extradata, mut stream) = rest.split_at(n);

    // Pre-scan the extradata chain for oversized geometry
    // declarations before the factory walks it.
    {
        let mut off = 0usize;
        while off + 2 <= extradata.len() {
            let len = ((extradata[off] as usize) << 8) | (extradata[off + 1] as usize);
            off += 2;
            let end = match off.checked_add(len) {
                Some(e) if e <= extradata.len() => e,
                _ => break,
            };
            if declares_oversized_geometry(&extradata[off..end]) {
                return;
            }
            off = end;
        }
    }

    let mut params = CodecParameters::video(CodecId::new(THEORA_CODEC_ID));
    params.extradata = extradata.to_vec();
    let Ok(mut dec) = make_decoder(&params) else {
        return;
    };

    let tb = TimeBase::from_rate(30);
    let mut packets = 0usize;
    while stream.len() >= 2 && packets < MAX_PACKETS {
        let len = u16::from_be_bytes([stream[0], stream[1]]) as usize;
        stream = &stream[2..];
        let take = len.min(stream.len());
        let (body, tail) = stream.split_at(take);
        stream = tail;
        if declares_oversized_geometry(body) {
            return;
        }
        let _ = dec.send_packet(&Packet::new(0, tb, body.to_vec()));
        // Drain everything the packet produced; receive_frame returns
        // NeedMore once the pending queue is empty.
        while dec.receive_frame().is_ok() {}
        packets += 1;
    }
    let _ = dec.flush();
    while dec.receive_frame().is_ok() {}
});
