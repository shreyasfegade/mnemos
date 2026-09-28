//! GGM puncturable PRF over AES-128 — the forward-privacy engine.
//!
//! Diana-style (Bost–Minaud–Ohrimenko, CCS 2017): symmetric primitives only, no public-key
//! operations on the hot path. Sophos achieves forward privacy with a trapdoor permutation,
//! which costs an RSA-scale operation per update; at our write rates that would dominate the
//! `write_p50_ms = 8` budget outright.
//!
//! ## What forward privacy buys, and what it does not
//!
//! Advancing the token chain after each epoch means a search token issued at epoch `e` cannot
//! be used to recognise entries written at epoch `e+1`. That defeats the induced-write attack
//! (A4): the adversary's injected write reveals nothing about earlier queries.
//!
//! **It does not protect the snapshot.** The tree's shape on disk encodes how many times each
//! partition has been updated, and that is what attack A3 harvests from the Sophos and Diana
//! baselines. Mnemos's answer is not in this module — it is the padding and PRF slot addressing
//! in `mnemos-index` and `mnemos-store`. This module provides component (i) of dual-epoch
//! security; component (iii) lives elsewhere, and the whole paper is about needing both.
//!
//! Keeping that distinction visible here, in the module that implements forward privacy, is
//! deliberate: it is the single most likely place for a future reader to conclude the job is
//! done.
//!
//! ## The construction
//!
//! A complete binary tree of depth `D`. Leaf `e` is the search token for epoch `e`. The
//! length-doubling PRG at each node is `G(k) = AES_k(0) ‖ AES_k(1)`, which is the standard
//! GGM instantiation from a block cipher.
//!
//! After advancing past epoch `e`, the client keeps the **minimal set of subtree roots
//! covering `[e+1, 2^D)`** — the residual key set. Leaves `≤ e` become unreachable: that is the
//! puncture, and it is what forward privacy is.
//!
//! ## The leak this module reproduces on purpose
//!
//! [`suffix_cover`] returns `popcount_zeros(from) + 1` nodes. **The residual set's size is a
//! function of how many times the chain has been advanced.** An adversary reading a Diana
//! index off disk counts keys per keyword and reads update counts straight off the structure —
//! no counter needed, no key needed. `mnemos-baselines::diana` exposes exactly that, and
//! [`TokenChain::serialise`] is where Mnemos declines to.
//!
//! [`TokenChain::serialise`] pads to [`TokenChain::serialised_len`], a constant. Same
//! primitive, one extra property, different outcome — which is the cheapest illustration in
//! the codebase of the paper's thesis that forward privacy and snapshot resistance are
//! independent axes.

use aes::cipher::generic_array::GenericArray;
use aes::cipher::{BlockEncrypt, KeyInit};
use aes::Aes128;
use hmac::{Hmac, Mac};
use sha2::Sha256;
use zeroize::Zeroize;

use mnemos_core::error::{Error, Result};
use mnemos_core::types::{BucketId, Epoch};

use crate::kdf::DerivedKey;

/// HKDF/HMAC domain separator for GGM roots. Distinct from every `info` string in
/// [`crate::kdf`], so a bucket's tree root can never collide with a derived key.
const ROOT_INFO: &[u8] = b"mnemos/v1/pprf/root";

/// A GGM tree node key, addressed by depth and index.
///
/// `index` is the node's position within its level, most-significant-bit-first, so the subtree
/// rooted here covers leaves `[index << (D - depth), (index + 1) << (D - depth))`.
///
/// **Not `Copy`.** A node key derives every token in its subtree, so it is zeroised on drop —
/// and a type with a destructor cannot be `Copy`. Losing implicit copies here is the point:
/// every duplication of a node key is now visible at the call site.
#[derive(Clone)]
pub struct NodeKey {
    /// AES-128 key at this node.
    pub key: [u8; 16],
    /// Depth from the root. `0` is the root; `D` is a leaf.
    pub depth: u8,
    /// Index within the level.
    pub index: u64,
}

impl std::fmt::Debug for NodeKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never print the key: a node key derives every token beneath it.
        write!(
            f,
            "NodeKey(depth={}, index={}, <redacted>)",
            self.depth, self.index
        )
    }
}

impl Drop for NodeKey {
    fn drop(&mut self) {
        self.key.zeroize();
    }
}

/// Length-doubling PRG from AES-128: `G(k) = (AES_k(0), AES_k(1))`.
///
/// The two constant blocks are distinct, so the halves are independent under the assumption
/// that AES is a PRP. Two calls rather than one because a single block cipher call produces
/// only 128 bits and GGM needs 256.
fn prg(key: &[u8; 16]) -> ([u8; 16], [u8; 16]) {
    let cipher = Aes128::new(GenericArray::from_slice(key));

    let mut left = [0u8; 16];
    let mut right = [0u8; 16];
    right[15] = 1;

    cipher.encrypt_block(GenericArray::from_mut_slice(&mut left));
    cipher.encrypt_block(GenericArray::from_mut_slice(&mut right));
    (left, right)
}

/// Derive a bucket's GGM root from the epoch-independent root key.
///
/// Per-bucket separation matters: a single tree shared across buckets would let a token for
/// one partition recognise writes in another, collapsing the partitioning the whole index
/// depends on.
fn bucket_root(root: &DerivedKey, bucket: BucketId) -> Result<[u8; 16]> {
    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(&root.0).map_err(|_| Error::Crypto)?;
    mac.update(ROOT_INFO);
    mac.update(&bucket.0.to_le_bytes());
    let out = mac.finalize().into_bytes();
    let mut k = [0u8; 16];
    k.copy_from_slice(&out[..16]);
    Ok(k)
}

/// The minimal set of subtree roots whose leaves are exactly `[from, 2^depth)`.
///
/// **This function's output length is the Diana leak.** It returns one node per zero bit in
/// `from`, plus the leaf `from` itself — so `|cover|` is a direct function of how far the
/// chain has been advanced. Reproducing that faithfully is the point: `mnemos-baselines`
/// needs the real leak to attack, and [`TokenChain::serialise`] needs something real to pad.
///
/// Returns an empty vector when `from >= 2^depth` (the chain is exhausted).
#[must_use]
pub fn suffix_cover(root: &[u8; 16], depth: u8, from: u64) -> Vec<NodeKey> {
    if depth >= 64 || from >= (1u64 << depth) {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(depth as usize + 1);
    let mut cur = *root;
    let mut idx: u64 = 0;

    for level in 0..depth {
        let bit = (from >> (depth - 1 - level)) & 1;
        let (l, r) = prg(&cur);
        if bit == 0 {
            // Every leaf under the right sibling is >= `from`, so the whole subtree is kept
            // and we descend left to keep splitting.
            out.push(NodeKey {
                key: r,
                depth: level + 1,
                index: idx * 2 + 1,
            });
            cur = l;
            idx *= 2;
        } else {
            // Every leaf under the left sibling is < `from`; drop it. That drop is the
            // puncture.
            cur = r;
            idx = idx * 2 + 1;
        }
    }
    out.push(NodeKey {
        key: cur,
        depth,
        index: from,
    });
    out
}

/// The minimal set of subtree roots whose leaves are exactly `[0, upto]`.
///
/// The mirror of [`suffix_cover`], and the one a **Diana search token** is made of: the client
/// hands the server a constrained key that derives every leaf up to the current counter and
/// nothing beyond it. Being unable to derive leaf `upto + 1` is precisely forward privacy —
/// a token issued now cannot recognise the next write.
///
/// Like its mirror, `|cover|` is a function of `upto` (one node per set bit, plus the leaf), so
/// a cached constrained key on disk leaks the counter's Hamming weight. `mnemos-baselines`
/// exhibits that; Mnemos pads it away.
#[must_use]
pub fn prefix_cover(root: &[u8; 16], depth: u8, upto: u64) -> Vec<NodeKey> {
    if depth >= 64 || upto >= (1u64 << depth) {
        return Vec::new();
    }

    let mut out = Vec::with_capacity(depth as usize + 1);
    let mut cur = *root;
    let mut idx: u64 = 0;

    for level in 0..depth {
        let bit = (upto >> (depth - 1 - level)) & 1;
        let (l, r) = prg(&cur);
        if bit == 1 {
            // Every leaf under the left sibling is <= `upto`, so keep the whole subtree and
            // descend right to keep splitting.
            out.push(NodeKey {
                key: l,
                depth: level + 1,
                index: idx * 2,
            });
            cur = r;
            idx = idx * 2 + 1;
        } else {
            cur = l;
            idx *= 2;
        }
    }
    out.push(NodeKey {
        key: cur,
        depth,
        index: upto,
    });
    out
}

/// Evaluate a leaf directly from the tree root.
///
/// The client always holds the root, so it can address any leaf; the constrained keys handed
/// out by [`prefix_cover`] are what restrict a *server*.
#[must_use]
pub fn eval_leaf(root: &[u8; 16], depth: u8, leaf: u64) -> [u8; 16] {
    let mut cur = *root;
    for level in 0..depth {
        let bit = (leaf >> (depth - 1 - level)) & 1;
        let (l, r) = prg(&cur);
        cur = if bit == 0 { l } else { r };
    }
    cur
}

/// Derive a GGM tree root for a bucket. Exposed so `mnemos-baselines` builds its trees from
/// the same derivation Mnemos uses.
///
/// # Errors
/// [`mnemos_core::Error::Crypto`] on HMAC failure.
pub fn tree_root(root: &DerivedKey, bucket: BucketId) -> Result<[u8; 16]> {
    bucket_root(root, bucket)
}

/// Evaluate leaf `leaf` from a residual set, if the set still covers it.
///
/// Returns `None` when the leaf has been punctured away — which is exactly the case the
/// forward-privacy argument turns on, so it is a value and not an error: callers are expected
/// to hit it.
#[must_use]
pub fn eval_from_cover(cover: &[NodeKey], depth: u8, leaf: u64) -> Option<[u8; 16]> {
    let node = cover.iter().find(|n| {
        let shift = depth - n.depth;
        (leaf >> shift) == n.index
    })?;

    let mut cur = node.key;
    for level in node.depth..depth {
        let bit = (leaf >> (depth - 1 - level)) & 1;
        let (l, r) = prg(&cur);
        cur = if bit == 0 { l } else { r };
    }
    Some(cur)
}

/// Per-partition puncturable token chain.
pub struct TokenChain {
    /// Which partition this chain serves.
    pub bucket: BucketId,
    /// Epoch of the most recent advance.
    pub epoch: Epoch,
    /// Tree depth; `2^depth` epochs are addressable.
    depth: u8,
    /// First epoch still reachable. Leaves below this have been punctured.
    next: u64,
    /// The residual key set covering `[next, 2^depth)`.
    ///
    /// **Its length is a function of `next`** — see [`suffix_cover`]. That is why
    /// [`TokenChain::serialise`] pads.
    residual: Vec<NodeKey>,
}

impl std::fmt::Debug for TokenChain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TokenChain(bucket={:?}, epoch={:?}, residual={} nodes)",
            self.bucket,
            self.epoch,
            self.residual.len()
        )
    }
}

impl TokenChain {
    /// Start a chain for a partition.
    ///
    /// # Errors
    /// [`mnemos_core::Error::Crypto`] on key-derivation failure.
    pub fn new(root: &DerivedKey, bucket: BucketId) -> Result<Self> {
        Self::with_depth(root, bucket, 32)
    }

    /// Start a chain with an explicit depth. Tests use shallow trees; production uses 32,
    /// matching `params.toml` `crypto.pprf_depth`.
    ///
    /// # Errors
    /// [`mnemos_core::Error::Crypto`] on key-derivation failure, or if `depth` is not in
    /// `1..=63`.
    pub fn with_depth(root: &DerivedKey, bucket: BucketId, depth: u8) -> Result<Self> {
        if depth == 0 || depth >= 64 {
            return Err(Error::Crypto);
        }
        let r = bucket_root(root, bucket)?;
        let residual = suffix_cover(&r, depth, 0);
        Ok(Self {
            bucket,
            epoch: Epoch(0),
            depth,
            next: 0,
            residual,
        })
    }

    /// Number of nodes currently retained.
    ///
    /// Exposed **because it is the leak**: `mnemos-baselines::diana` reports it as the
    /// snapshot observable an adversary reads off a Diana index. Mnemos never persists it.
    #[must_use]
    pub fn residual_len(&self) -> usize {
        self.residual.len()
    }

    /// First epoch still reachable.
    #[must_use]
    pub fn next_epoch(&self) -> u64 {
        self.next
    }

    /// Search token valid for entries written at `epoch`.
    ///
    /// # Errors
    /// [`mnemos_core::Error::Crypto`] if the epoch has been punctured away or is beyond the
    /// tree's depth. Both are the same error deliberately: distinguishing "already punctured"
    /// from "out of range" would tell a caller how far the chain has advanced.
    pub fn search_token(&self, epoch: Epoch) -> Result<[u8; 16]> {
        eval_from_cover(&self.residual, self.depth, epoch.0).ok_or(Error::Crypto)
    }

    /// Puncture the chain at `epoch`, so the token cannot recognise later writes.
    ///
    /// Called once per partition per epoch flush. **The whole of forward privacy is this
    /// call**, so it must happen for every partition touched in the batch — including those
    /// touched only by dummy records, or the set of punctured partitions would itself reveal
    /// which records were real.
    ///
    /// Idempotent below the current position: advancing to an epoch already punctured is a
    /// no-op rather than an error, because a caller that flushes twice in one epoch must not
    /// observe a difference.
    ///
    /// # Errors
    /// [`mnemos_core::Error::Crypto`] if `epoch` exceeds the tree's capacity.
    pub fn advance(&mut self, epoch: Epoch) -> Result<()> {
        let target = epoch.0.saturating_add(1);
        if target >= (1u64 << self.depth) {
            return Err(Error::Crypto);
        }
        if target <= self.next {
            self.epoch = epoch;
            return Ok(());
        }

        // Re-cover from the surviving residual rather than from the root: the root is gone
        // after the first advance, which is what makes the puncture irreversible. Re-deriving
        // from a retained root would leave earlier leaves recoverable and silently void
        // forward privacy.
        let mut fresh = Vec::with_capacity(self.depth as usize + 1);
        for level in 0..self.depth {
            let bit = (target >> (self.depth - 1 - level)) & 1;
            if bit == 0 {
                let index = (target >> (self.depth - level - 1)) | 1;
                if let Some(key) = self.node_key(level + 1, index) {
                    fresh.push(NodeKey {
                        key,
                        depth: level + 1,
                        index,
                    });
                }
            }
        }
        if let Some(key) = eval_from_cover(&self.residual, self.depth, target) {
            fresh.push(NodeKey {
                key,
                depth: self.depth,
                index: target,
            });
        }

        self.residual = fresh;
        self.next = target;
        self.epoch = epoch;
        Ok(())
    }

    /// Derive an interior node from the residual set, if it is still covered.
    fn node_key(&self, depth: u8, index: u64) -> Option<[u8; 16]> {
        // A node at (depth, index) is the ancestor of leaf `index << (D - depth)`; walking to
        // that leaf's position and stopping early gives the node key.
        let first_leaf = index << (self.depth - depth);
        let node = self.residual.iter().find(|n| {
            let shift = self.depth - n.depth;
            (first_leaf >> shift) == n.index
        })?;
        if node.depth > depth {
            return None;
        }

        let mut cur = node.key;
        for level in node.depth..depth {
            let bit = (first_leaf >> (self.depth - 1 - level)) & 1;
            let (l, r) = prg(&cur);
            cur = if bit == 0 { l } else { r };
        }
        Some(cur)
    }

    /// Serialised size in bytes.
    ///
    /// **Must be constant across partitions.** A variable-size chain on disk is precisely the
    /// per-partition update-count leak that A3 exploits against Diana — reproducing it here
    /// would defeat the entire scheme. Pad to the depth-32 worst case.
    #[must_use]
    pub fn serialised_len(depth: u32) -> usize {
        // header: depth(1) + bucket(4) + epoch(8) + next(8) = 21
        // worst-case cover: depth + 1 nodes of (key 16 + depth 1 + index 8) = 25 bytes each
        21 + (depth as usize + 1) * 25
    }

    /// Serialise to exactly [`TokenChain::serialised_len`] bytes.
    ///
    /// The residual set is written first and the remainder filled with **random** bytes, not
    /// zeros. Zero padding would make the real length recoverable by inspection, which is the
    /// entire leak this function exists to close — the trailing region has to be
    /// indistinguishable from key material, and random bytes are.
    #[must_use]
    pub fn serialise(&self) -> Vec<u8> {
        use rand::RngCore;

        let total = Self::serialised_len(u32::from(self.depth));
        let mut out = Vec::with_capacity(total);
        out.push(self.depth);
        out.extend_from_slice(&self.bucket.0.to_le_bytes());
        out.extend_from_slice(&self.epoch.0.to_le_bytes());
        out.extend_from_slice(&self.next.to_le_bytes());

        for n in &self.residual {
            out.extend_from_slice(&n.key);
            out.push(n.depth);
            out.extend_from_slice(&n.index.to_le_bytes());
        }

        let mut pad = vec![0u8; total - out.len()];
        rand::rngs::OsRng.fill_bytes(&mut pad);
        out.extend_from_slice(&pad);
        out
    }

    /// Reconstruct from [`TokenChain::serialise`] output.
    ///
    /// The retained-node count is recomputed from `next` rather than stored, which is what
    /// makes the padding safe to discard: the length was never needed on disk in the first
    /// place.
    ///
    /// # Errors
    /// [`mnemos_core::Error::Format`] if the buffer is short or malformed.
    pub fn deserialise(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 21 {
            return Err(Error::Format("token chain header truncated".into()));
        }
        let depth = bytes[0];
        if depth == 0 || depth >= 64 {
            return Err(Error::Format("token chain depth out of range".into()));
        }
        if bytes.len() < Self::serialised_len(u32::from(depth)) {
            return Err(Error::Format("token chain body truncated".into()));
        }

        let bucket = BucketId(u32::from_le_bytes(
            bytes[1..5].try_into().map_err(|_| Error::Crypto)?,
        ));
        let epoch = Epoch(u64::from_le_bytes(
            bytes[5..13].try_into().map_err(|_| Error::Crypto)?,
        ));
        let next = u64::from_le_bytes(bytes[13..21].try_into().map_err(|_| Error::Crypto)?);

        // Recomputed, never read from disk — see the doc comment.
        let count = if next >= (1u64 << depth) {
            0
        } else {
            (0..depth)
                .filter(|l| (next >> (depth - 1 - l)) & 1 == 0)
                .count()
                + 1
        };

        let mut residual = Vec::with_capacity(count);
        for i in 0..count {
            let off = 21 + i * 25;
            let mut key = [0u8; 16];
            key.copy_from_slice(&bytes[off..off + 16]);
            let d = bytes[off + 16];
            let index = u64::from_le_bytes(
                bytes[off + 17..off + 25]
                    .try_into()
                    .map_err(|_| Error::Crypto)?,
            );
            residual.push(NodeKey {
                key,
                depth: d,
                index,
            });
        }

        Ok(Self {
            bucket,
            epoch,
            depth,
            next,
            residual,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> DerivedKey {
        DerivedKey([7u8; 32])
    }

    #[test]
    fn tokens_are_deterministic_and_distinct_per_epoch() {
        let a = TokenChain::with_depth(&root(), BucketId(3), 8).unwrap();
        let b = TokenChain::with_depth(&root(), BucketId(3), 8).unwrap();
        for e in 0..8 {
            assert_eq!(
                a.search_token(Epoch(e)).unwrap(),
                b.search_token(Epoch(e)).unwrap()
            );
        }
        let t0 = a.search_token(Epoch(0)).unwrap();
        let t1 = a.search_token(Epoch(1)).unwrap();
        assert_ne!(t0, t1, "distinct epochs must yield distinct tokens");
    }

    #[test]
    fn buckets_are_separated() {
        let a = TokenChain::with_depth(&root(), BucketId(1), 8).unwrap();
        let b = TokenChain::with_depth(&root(), BucketId(2), 8).unwrap();
        assert_ne!(
            a.search_token(Epoch(4)).unwrap(),
            b.search_token(Epoch(4)).unwrap()
        );
    }

    #[test]
    fn advancing_punctures_the_past_and_preserves_the_future() {
        // This is forward privacy, as a test. A token for a punctured epoch must be
        // unrecoverable; every later token must still be derivable AND unchanged, because a
        // puncture that altered future tokens would break search rather than protect it.
        let mut c = TokenChain::with_depth(&root(), BucketId(0), 10).unwrap();
        let future: Vec<[u8; 16]> = (0..64).map(|e| c.search_token(Epoch(e)).unwrap()).collect();

        c.advance(Epoch(20)).unwrap();

        for e in 0..=20u64 {
            assert!(
                c.search_token(Epoch(e)).is_err(),
                "epoch {e} survived a puncture at 20 — forward privacy is void"
            );
        }
        for e in 21..64u64 {
            assert_eq!(
                c.search_token(Epoch(e)).unwrap(),
                future[e as usize],
                "puncture altered the token for a future epoch {e}"
            );
        }
    }

    #[test]
    fn repeated_advances_stay_consistent() {
        let mut c = TokenChain::with_depth(&root(), BucketId(9), 12).unwrap();
        let mut expected = Vec::new();
        for e in 0..200u64 {
            expected.push(c.search_token(Epoch(e + 1)).ok());
            c.advance(Epoch(e)).unwrap();
            assert!(c.search_token(Epoch(e)).is_err());
        }
        // Advancing to an already-punctured epoch is a no-op, not an error: a double flush in
        // one epoch must not be observable.
        let before = c.residual_len();
        c.advance(Epoch(10)).unwrap();
        assert_eq!(c.residual_len(), before);
    }

    #[test]
    fn residual_set_size_leaks_the_update_count() {
        // The NEGATIVE control for the whole module. This asserts the leak EXISTS in the raw
        // structure — because `serialised_length_is_independent_of_update_count` below would
        // otherwise pass vacuously if the cover happened to be constant-size.
        let mut sizes = std::collections::HashSet::new();
        for advances in [0u64, 1, 7, 63, 255, 1023] {
            let mut c = TokenChain::with_depth(&root(), BucketId(0), 16).unwrap();
            if advances > 0 {
                c.advance(Epoch(advances - 1)).unwrap();
            }
            sizes.insert(c.residual_len());
        }
        assert!(
            sizes.len() > 1,
            "residual sizes were constant, so the Diana baseline has no leak to attack and \
             the padding test below proves nothing"
        );
    }

    #[test]
    fn serialised_length_is_independent_of_update_count() {
        // The regression test for the exact leak A3 exploits in the baselines.
        // If this ever fails, Mnemos has become one of its own victims.
        let mut lens = std::collections::HashSet::new();
        for advances in [0u64, 1, 7, 63, 255, 1023] {
            let mut c = TokenChain::with_depth(&root(), BucketId(0), 16).unwrap();
            if advances > 0 {
                c.advance(Epoch(advances - 1)).unwrap();
            }
            lens.insert(c.serialise().len());
        }
        assert_eq!(
            lens.len(),
            1,
            "serialised chain length varied with update count: {lens:?}"
        );
        assert_eq!(*lens.iter().next().unwrap(), TokenChain::serialised_len(16));
    }

    #[test]
    fn serialisation_roundtrips_through_the_padding() {
        let mut c = TokenChain::with_depth(&root(), BucketId(5), 12).unwrap();
        c.advance(Epoch(37)).unwrap();
        let back = TokenChain::deserialise(&c.serialise()).unwrap();

        assert_eq!(back.next_epoch(), c.next_epoch());
        assert_eq!(back.residual_len(), c.residual_len());
        for e in 38..80u64 {
            assert_eq!(
                back.search_token(Epoch(e)).unwrap(),
                c.search_token(Epoch(e)).unwrap()
            );
        }
        assert!(back.search_token(Epoch(37)).is_err());
    }

    #[test]
    fn prefix_cover_derives_the_past_and_not_the_future() {
        // The Diana search-token property. A constrained key for counter `c` must reach every
        // leaf up to `c` — otherwise search silently loses results — and must NOT reach `c+1`,
        // which is forward privacy.
        let r = bucket_root(&root(), BucketId(4)).unwrap();
        for upto in [0u64, 1, 5, 8, 100, 255] {
            let cover = prefix_cover(&r, 9, upto);
            for leaf in 0..=upto {
                assert_eq!(
                    eval_from_cover(&cover, 9, leaf),
                    Some(eval_leaf(&r, 9, leaf)),
                    "constrained key for {upto} could not derive leaf {leaf}"
                );
            }
            for leaf in (upto + 1)..(upto + 16).min(512) {
                assert!(
                    eval_from_cover(&cover, 9, leaf).is_none(),
                    "constrained key for {upto} derived FUTURE leaf {leaf} — forward privacy is void"
                );
            }
        }
    }

    #[test]
    fn prefix_cover_size_leaks_the_counters_hamming_weight() {
        // Why a cached constrained key on disk is a leak at all, and the negative control for
        // the padded serialisation. `|cover| = popcount(upto) + 1`.
        let r = bucket_root(&root(), BucketId(0)).unwrap();
        for upto in [0u64, 1, 3, 7, 15, 31, 63, 127] {
            assert_eq!(
                prefix_cover(&r, 16, upto).len(),
                upto.count_ones() as usize + 1,
                "cover size for {upto} did not track its Hamming weight"
            );
        }
    }

    #[test]
    fn padding_is_not_zeros() {
        // Zero padding would make the true residual length recoverable by inspection, which is
        // the leak the padding exists to close.
        let mut c = TokenChain::with_depth(&root(), BucketId(0), 16).unwrap();
        c.advance(Epoch(0)).unwrap();
        let bytes = c.serialise();
        let used = 21 + c.residual_len() * 25;
        assert!(
            used < bytes.len(),
            "test needs a chain that actually gets padded"
        );
        assert!(
            bytes[used..].iter().any(|b| *b != 0),
            "padding region is all zeros — the real length is readable off disk"
        );
    }
}
