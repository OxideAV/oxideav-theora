#![no_main]

//! Arbitrary bytes at every §6 header-packet decoder.
//!
//! The three header decoders are the stream's front door: a demuxer
//! routes de-framed packets by [`classify_packet`] and then hands the
//! bytes to `decode_identification_header` (§6.2),
//! `parse_comment_header` (§6.3), and `parse_setup_header` /
//! `decode_setup_header` (§6.4 — loop-filter limits, quantization
//! parameters, and the 80 DCT-token Huffman codebooks). All of them
//! must `return` a `Result` for any input — never panic, abort,
//! overflow, or index out of bounds.
//!
//! Two self-inverse oracles ride along (this crate's §6 serializers
//! are documented exact inverses of the decoders):
//!
//! * an accepted identification header re-encodes **byte-exactly**
//!   onto the packet prefix it was decoded from (§6.2 is a
//!   fixed-width field list — there are no non-canonical spellings);
//! * accepted setup tables survive `encode_setup_header` →
//!   `decode_setup_header` back to an equal `SetupHeaderTables`
//!   (the bit-packed §6.4.5 body has freedom in NBITS choices, so
//!   the oracle is a decode fixpoint, not byte identity).

use libfuzzer_sys::fuzz_target;
use oxideav_theora::{
    classify_packet, decode_identification_header, decode_setup_header,
    encode_identification_header, encode_setup_header, parse_comment_header, parse_setup_header,
};

fuzz_target!(|data: &[u8]| {
    let _ = classify_packet(data);

    if let Ok(ident) = decode_identification_header(data) {
        let enc = encode_identification_header(&ident)
            .expect("a decoded identification header must re-encode");
        assert!(
            data.starts_with(&enc),
            "ident re-encode must reproduce the accepted §6.2 prefix"
        );
        // The re-encoded packet must decode back to the same header.
        let back = decode_identification_header(&enc)
            .expect("a re-encoded identification header must decode");
        assert_eq!(back, ident, "§6.2 decode∘encode must be the identity");
    }

    let _ = parse_comment_header(data);
    let _ = parse_setup_header(data);

    if let Ok(tables) = decode_setup_header(data) {
        let bytes =
            encode_setup_header(&tables).expect("accepted setup tables must serialize (§6.4.5)");
        let back = decode_setup_header(&bytes)
            .expect("a serialized §6.4 setup header must decode back");
        assert_eq!(
            back, tables,
            "§6.4 decode∘encode∘decode must be a fixpoint on accepted tables"
        );
    }
});
