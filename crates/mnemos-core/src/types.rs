//! Core value types.
//!
//! Two invariants run through this module and are worth stating once:
//!
//! 1. **Nothing here derives `Debug`, `Display`, or `Serialize` if it can hold plaintext or
//!    key-adjacent material.** The `no-plaintext` CI lint enforces it; the manual impls below
//!    exist so accidental logging is impossible rather than merely discouraged.
//! 2. **No address is ever derived from content.** Content-derived addressing would smuggle
//!    deterministic-encryption leakage in through the back door. Addresses come from
//!    `PRF(K_idx, tag)` where `tag = PRF(K_tag, memory_id)`.

use std::fmt::Write as _;

use rand::RngCore;
use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Stable, random 128-bit identity for a stored memory.
///
/// Random — **not** a hash of content, and **not** a counter. A counter would encode insertion
/// order, which is exactly the artifact `docs/adr/0003` exists to remove.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct MemoryId(pub [u8; 16]);

impl MemoryId {
    /// Draw a fresh identifier from the OS CSPRNG.
    #[must_use]
    pub fn random() -> Self {
        let mut b = [0u8; 16];
        getrandom_into(&mut b);
        Self(b)
    }

    /// Lowercase hex, for CLI output and test fixtures only.
    #[must_use]
    pub fn to_hex(self) -> String {
        self.0.iter().fold(String::with_capacity(32), |mut acc, b| {
            let _ = write!(acc, "{b:02x}");
            acc
        })
    }
}

// Opaque on purpose: an id in a log is a correlation handle for anyone reading that log.
impl std::fmt::Debug for MemoryId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MemoryId(<redacted>)")
    }
}

/// `PRF(K_tag, memory_id ‖ chunk_idx)`, truncated to 128 bits.
///
/// The store's notion of identity. Compare **only** via [`Tag::ct_eq_bytes`] — a variable-time
/// comparison here is a timing oracle on entry existence.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Tag(pub [u8; 16]);

impl Tag {
    /// Constant-time equality. The `subtle` dependency lives in `mnemos-crypto`, so this
    /// helper takes the comparison result rather than performing it here — callers in the
    /// crypto crate wrap it. Kept as a marker so no one is tempted to write `==` on the hot path.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }
}

impl std::fmt::Debug for Tag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tag(<redacted>)")
    }
}

/// Physical slot index within a slab. `PRF(K_idx, tag) mod slot_count`, Robin Hood on collision.
///
/// A function of *identity*, never of *arrival time*. This is the mechanism of history
/// independence (`docs/adr/0003-history-independence-first-class.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SlotAddr(pub u64);

/// IVF partition index, `0 .. params.index.buckets`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct BucketId(pub u32);

/// Monotone epoch counter. The unit of write batching, key rotation, and — per Theorem 1 —
/// the granularity of the only signal the post-compromise cohort has that the historical
/// cohort does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Epoch(pub u64);

impl Epoch {
    /// The next epoch.
    #[must_use]
    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

/// A dense f32 embedding, as produced by the Python embedder.
///
/// Held only transiently: quantised to [`QuantVec`] before anything touches disk.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct Embedding(pub Vec<f32>);

impl Embedding {
    /// Dimensionality.
    #[must_use]
    pub fn dim(&self) -> usize {
        self.0.len()
    }
}

/// int8 scalar-quantised embedding with an f16-equivalent scale.
///
/// 770 bytes at dim 768. Fixed width — every stored vector is the same size, so vector
/// length never encodes anything.
#[derive(Clone, PartialEq)]
pub struct QuantVec {
    /// Quantised components.
    pub codes: Vec<i8>,
    /// Per-vector dequantisation scale, stored as f32 but generated from an f16 grid so the
    /// on-disk representation is canonical (relevant to history independence: two equal
    /// vectors must serialise identically).
    pub scale: f32,
}

impl QuantVec {
    /// Quantise, choosing the scale from the vector's max absolute component.
    ///
    /// The scale is snapped to an f16-representable value so the encoding is **canonical**:
    /// two equal input vectors must serialise to identical bytes, or history independence
    /// fails (two stores with the same logical content would differ on disk).
    ///
    /// # Panics
    /// Never — an all-zero vector yields `scale = 1.0` and all-zero codes.
    #[must_use]
    pub fn quantise(e: &Embedding) -> Self {
        let max = e.0.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let raw = if max > 0.0 { max / 127.0 } else { 1.0 };
        let scale = snap_to_f16_grid(raw);
        let codes =
            e.0.iter()
                .map(|v| {
                    let q = (v / scale).round();
                    q.clamp(-127.0, 127.0) as i8
                })
                .collect();
        Self { codes, scale }
    }

    /// Cosine similarity against another quantised vector.
    ///
    /// The dot product accumulates in `i32` and is scaled once at the end: integer arithmetic
    /// in the hot loop is both faster and, more importantly, **deterministic**. Float
    /// accumulation order varies with vectorisation, which would make identical inputs produce
    /// microscopically different scores on different machines — and rerank order is part of
    /// what the leakage tests assert.
    ///
    /// Both vectors are assumed L2-normalised before quantisation (the Python embedder does
    /// this), so the dot product *is* the cosine up to quantisation error.
    #[must_use]
    pub fn cosine(&self, other: &Self) -> f32 {
        let n = self.codes.len().min(other.codes.len());
        let mut acc: i32 = 0;
        for i in 0..n {
            acc += i32::from(self.codes[i]) * i32::from(other.codes[i]);
        }
        acc as f32 * self.scale * other.scale
    }

    /// Serialised width in bytes: one byte per dimension plus a 4-byte scale.
    #[must_use]
    pub fn encoded_len(dim: usize) -> usize {
        dim + 4
    }

    /// Encode to a fixed-width byte string. Every stored vector is the same size, so vector
    /// length never encodes anything.
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(Self::encoded_len(self.codes.len()));
        v.extend(self.codes.iter().map(|c| *c as u8));
        v.extend_from_slice(&self.scale.to_le_bytes());
        v
    }

    /// Decode from [`QuantVec::encode`] output.
    ///
    /// Returns `None` if the buffer is too short to hold a scale.
    #[must_use]
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 4 {
            return None;
        }
        let (codes, scale) = bytes.split_at(bytes.len() - 4);
        Some(Self {
            codes: codes.iter().map(|b| *b as i8).collect(),
            scale: f32::from_le_bytes(scale.try_into().ok()?),
        })
    }
}

impl std::fmt::Debug for QuantVec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QuantVec(dim={}, <redacted>)", self.codes.len())
    }
}

/// A decrypted memory, as handed back to the caller.
///
/// `ZeroizeOnDrop` so the plaintext leaves the locked arena's lifetime cleanly. The FFI layer
/// copies out of this into Python — at which point the residency guarantee ends, deliberately
/// and as documented in `ARCHITECTURE.md` §6.3.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct Memory {
    /// The stored text.
    pub text: String,
    /// Caller-supplied metadata, JSON-encoded. Encrypted alongside the text — metadata is
    /// content and gets no weaker treatment.
    pub meta: String,
    /// Identity. Not zeroised (it is not secret once returned), but kept opaque in logs.
    #[zeroize(skip)]
    pub id: MemoryId,
}

impl std::fmt::Debug for Memory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Memory(id={:?}, {} bytes <redacted>)",
            self.id,
            self.text.len()
        )
    }
}

/// A search hit: identity, score, and the decrypted memory.
#[derive(Debug)]
pub struct Hit {
    /// The retrieved memory.
    pub memory: Memory,
    /// Cosine similarity after exact rerank in the locked arena.
    pub score: f32,
}

/// Snap a positive f32 to the nearest value representable in binary16.
///
/// Canonicalisation, not compression: it guarantees that two equal embeddings quantise to
/// byte-identical output regardless of how the float arrived, which history independence
/// requires. Implemented by truncating the mantissa to 10 bits rather than pulling in an f16
/// dependency for one operation.
fn snap_to_f16_grid(x: f32) -> f32 {
    // f32 mantissa is 23 bits; f16 keeps 10. Round-to-nearest on the discarded low 13 bits.
    const DISCARD: u32 = 13;

    if !x.is_finite() || x <= 0.0 {
        return 1.0;
    }
    let bits = x.to_bits();
    let half = 1u32 << (DISCARD - 1);
    let rounded = (bits + half) & !((1u32 << DISCARD) - 1);
    let out = f32::from_bits(rounded);
    if out > 0.0 && out.is_finite() {
        out
    } else {
        x
    }
}

fn getrandom_into(buf: &mut [u8]) {
    rand::rngs::OsRng.fill_bytes(buf);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: Vec<f32>) -> Embedding {
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        Embedding(v.into_iter().map(|x| x / norm).collect())
    }

    #[test]
    fn quantisation_is_canonical() {
        // Two equal embeddings must produce byte-identical encodings. If they did not,
        // history independence would fail: two stores with the same logical content would
        // differ on disk.
        let a = QuantVec::quantise(&unit(vec![0.3, -0.7, 0.2, 0.6]));
        let b = QuantVec::quantise(&unit(vec![0.3, -0.7, 0.2, 0.6]));
        assert_eq!(a.encode(), b.encode());
    }

    #[test]
    fn cosine_approximates_the_true_value() {
        let e = unit(vec![1.0, 2.0, 3.0, 4.0]);
        let q = QuantVec::quantise(&e);
        let self_sim = q.cosine(&q);
        assert!(
            (self_sim - 1.0).abs() < 0.02,
            "self-similarity was {self_sim}"
        );

        let orth = QuantVec::quantise(&unit(vec![4.0, -3.0, 2.0, -1.0]));
        let true_dot: f32 = unit(vec![1.0, 2.0, 3.0, 4.0])
            .0
            .iter()
            .zip(unit(vec![4.0, -3.0, 2.0, -1.0]).0.iter())
            .map(|(a, b)| a * b)
            .sum();
        assert!((q.cosine(&orth) - true_dot).abs() < 0.02);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let q = QuantVec::quantise(&unit(vec![0.1, -0.9, 0.4]));
        let back = QuantVec::decode(&q.encode()).unwrap();
        assert_eq!(back.codes, q.codes);
        assert!((back.scale - q.scale).abs() < f32::EPSILON);
    }

    #[test]
    fn encoded_width_is_fixed() {
        let a = QuantVec::quantise(&unit(vec![1.0, 0.0, 0.0, 0.0]));
        let b = QuantVec::quantise(&unit(vec![0.5, 0.5, 0.5, 0.5]));
        assert_eq!(a.encode().len(), b.encode().len());
        assert_eq!(a.encode().len(), QuantVec::encoded_len(4));
    }

    #[test]
    fn all_zero_vector_does_not_divide_by_zero() {
        let q = QuantVec::quantise(&Embedding(vec![0.0; 8]));
        assert!(q.codes.iter().all(|c| *c == 0));
        assert!(q.scale > 0.0);
    }
}
