//! Forward-private search tokens — a thin wrapper over `mnemos-crypto::pprf`.
//!
//! One [`TokenChain`] per partition. Chains advance once per epoch flush, for **every** touched
//! partition including those touched only by dummy records.
//!
//! The serialised state of every chain is padded to the same constant length. Without that
//! padding the chain state would encode how many times its partition has been updated —
//! reproducing, inside Mnemos, exactly the leak that attack A3 exploits in the Sophos and
//! Diana baselines.
//!
//! ## Why every partition advances, not just the written ones
//!
//! The obvious implementation advances only the chains whose partitions received a record this
//! epoch. That is wrong, and wrong in a way that silently undoes the padding one layer up: the
//! WAL pads each batch to `epoch_writes` records so an observer cannot tell how many were real,
//! but if only *real* writes advanced a chain, the set of advanced chains would name exactly the
//! partitions that received real data. The dummy padding would have been paid for and then
//! given away.
//!
//! So [`TokenTable::advance_all`] is the operation the engine calls, and [`TokenTable::advance`]
//! exists only for tests that need to exercise a subset deliberately.
//!
//! ## What this buys, and what it does not
//!
//! Forward privacy means a token issued at epoch `e` cannot recognise a record written at epoch
//! `e+1`. It does **not** protect the snapshot: the structure-transfer attack (finding F2) never
//! looks at a token, it looks at region sizes. Both are needed, which is the whole argument of
//! §5 of the paper — this module is component (i) of dual-epoch security and nothing more.

use mnemos_core::error::{Error, Result};
use mnemos_core::types::{BucketId, Epoch};
use mnemos_crypto::kdf::DerivedKey;

pub use mnemos_crypto::pprf::TokenChain;

/// Tree depth. `2^32` epochs is far beyond any store's life at `epoch_ms = 500`
/// (roughly 68 years of continuous flushing), and matches `params.toml` `crypto.pprf_depth`.
const DEPTH: u8 = 32;

/// All per-partition token chains.
pub struct TokenTable {
    /// One chain per partition, in `BucketId` order.
    chains: Vec<TokenChain>,
}

impl TokenTable {
    /// Derive a fresh table for `buckets` partitions.
    ///
    /// # Errors
    /// [`Error::Crypto`] on key-derivation failure.
    pub fn derive(root: &DerivedKey, buckets: u32) -> Result<Self> {
        let chains = (0..buckets)
            .map(|b| TokenChain::with_depth(root, BucketId(b), DEPTH))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { chains })
    }

    /// Number of partitions served.
    #[must_use]
    pub fn len(&self) -> usize {
        self.chains.len()
    }

    /// Whether the table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.chains.is_empty()
    }

    /// The search token for a partition at an epoch.
    ///
    /// # Errors
    /// [`Error::Crypto`] if the epoch has been punctured away or is out of range, or
    /// [`Error::Format`] if the partition does not exist.
    pub fn token(&self, bucket: BucketId, epoch: Epoch) -> Result<[u8; 16]> {
        self.chain(bucket)?.search_token(epoch)
    }

    /// Advance **every** chain. This is what an epoch flush calls.
    ///
    /// Advancing all of them, unconditionally, is the point — see the module docs.
    ///
    /// # Errors
    /// [`Error::Crypto`] if the epoch exceeds the tree's capacity.
    pub fn advance_all(&mut self, epoch: Epoch) -> Result<()> {
        for chain in &mut self.chains {
            chain.advance(epoch)?;
        }
        Ok(())
    }

    /// Advance the chains for the given partitions.
    ///
    /// **Not for the write path.** Advancing a subset makes the advanced set reveal which
    /// partitions held real records; [`TokenTable::advance_all`] is the operation the engine
    /// uses. This exists so a test can construct that leak deliberately and assert it is
    /// detectable.
    ///
    /// # Errors
    /// Crypto or I/O failures.
    pub fn advance(&mut self, buckets: &[BucketId], epoch: Epoch) -> Result<()> {
        for b in buckets {
            self.chain_mut(*b)?.advance(epoch)?;
        }
        Ok(())
    }

    /// Serialised size. **Constant, independent of update history.**
    #[must_use]
    pub fn on_disk_len(&self) -> usize {
        self.chains.len() * TokenChain::serialised_len(u32::from(DEPTH))
    }

    /// Serialise every chain, each padded to the same constant width.
    ///
    /// The concatenation is therefore a function of `buckets` alone. Two stores with identical
    /// contents and wildly different update histories produce byte-strings of identical length,
    /// which is the property [`TokenTable::on_disk_len`] advertises and
    /// `serialised_length_is_independent_of_update_count` in `mnemos-crypto` proves per chain.
    #[must_use]
    pub fn serialise(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.on_disk_len());
        for chain in &self.chains {
            out.extend_from_slice(&chain.serialise());
        }
        out
    }

    /// Reconstruct from [`TokenTable::serialise`] output.
    ///
    /// # Errors
    /// [`Error::Format`] if the buffer is not exactly `buckets` chains long.
    pub fn deserialise(bytes: &[u8], buckets: u32) -> Result<Self> {
        let width = TokenChain::serialised_len(u32::from(DEPTH));
        if bytes.len() != buckets as usize * width {
            return Err(Error::Format(format!(
                "token table is {} bytes; expected {} for {buckets} partitions",
                bytes.len(),
                buckets as usize * width
            )));
        }
        let chains = bytes
            .chunks_exact(width)
            .map(TokenChain::deserialise)
            .collect::<Result<Vec<_>>>()?;
        Ok(Self { chains })
    }

    fn chain(&self, bucket: BucketId) -> Result<&TokenChain> {
        self.chains
            .get(bucket.0 as usize)
            .ok_or_else(|| Error::Format("bucket id out of range".into()))
    }

    fn chain_mut(&mut self, bucket: BucketId) -> Result<&mut TokenChain> {
        self.chains
            .get_mut(bucket.0 as usize)
            .ok_or_else(|| Error::Format("bucket id out of range".into()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(buckets: u32) -> TokenTable {
        TokenTable::derive(&DerivedKey([5u8; 32]), buckets).unwrap()
    }

    #[test]
    fn tokens_differ_across_partitions_and_epochs() {
        let t = table(8);
        let a = t.token(BucketId(0), Epoch(3)).unwrap();
        let b = t.token(BucketId(1), Epoch(3)).unwrap();
        let c = t.token(BucketId(0), Epoch(4)).unwrap();
        assert_ne!(
            a, b,
            "partitions share a token — one partition's token would search another"
        );
        assert_ne!(
            a, c,
            "epochs share a token — that is no forward privacy at all"
        );
    }

    #[test]
    fn advancing_makes_the_current_epoch_unrecognisable() {
        // Forward privacy, at the table level: after the flush for epoch e, no token in the
        // table can recognise anything from epoch e or earlier.
        let mut t = table(4);
        t.advance_all(Epoch(0)).unwrap();
        for b in 0..4u32 {
            assert!(t.token(BucketId(b), Epoch(0)).is_err());
            assert!(t.token(BucketId(b), Epoch(1)).is_ok());
        }
    }

    #[test]
    fn advance_all_leaves_no_partition_behind() {
        // The leak this module's docs warn about, asserted as a difference. Advancing a subset
        // leaves the untouched partitions at an older position, which is exactly the signal
        // that would identify which partitions held real records.
        let mut subset = table(4);
        subset
            .advance(&[BucketId(0), BucketId(1)], Epoch(0))
            .unwrap();
        assert!(
            subset.token(BucketId(0), Epoch(0)).is_err(),
            "advanced chain still readable"
        );
        assert!(
            subset.token(BucketId(2), Epoch(0)).is_ok(),
            "the subset advance was a no-op, so this test proves nothing"
        );

        let mut all = table(4);
        all.advance_all(Epoch(0)).unwrap();
        for b in 0..4u32 {
            assert!(all.token(BucketId(b), Epoch(0)).is_err());
        }
    }

    #[test]
    fn serialised_length_does_not_track_update_history() {
        // The regression test for the leak Mnemos would otherwise reproduce inside itself.
        let mut lens = std::collections::HashSet::new();
        for advances in [0u64, 1, 17, 250, 4096] {
            let mut t = table(16);
            for e in 0..advances {
                t.advance_all(Epoch(e)).unwrap();
            }
            lens.insert(t.serialise().len());
            assert_eq!(t.serialise().len(), t.on_disk_len());
        }
        assert_eq!(
            lens.len(),
            1,
            "token table length varied with update count: {lens:?}"
        );
    }

    #[test]
    fn serialisation_roundtrips() {
        let mut t = table(8);
        for e in 0..40u64 {
            t.advance_all(Epoch(e)).unwrap();
        }
        let back = TokenTable::deserialise(&t.serialise(), 8).unwrap();
        for b in 0..8u32 {
            assert_eq!(
                back.token(BucketId(b), Epoch(100)).unwrap(),
                t.token(BucketId(b), Epoch(100)).unwrap()
            );
            assert!(
                back.token(BucketId(b), Epoch(39)).is_err(),
                "puncture did not survive"
            );
        }
    }

    #[test]
    fn a_wrong_length_buffer_is_rejected() {
        let t = table(4);
        let mut bytes = t.serialise();
        bytes.truncate(bytes.len() - 1);
        assert!(TokenTable::deserialise(&bytes, 4).is_err());
    }
}
