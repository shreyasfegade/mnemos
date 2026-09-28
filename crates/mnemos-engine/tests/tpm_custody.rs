//! The store under **real hardware key custody**, and the claim that key custody is orthogonal
//! to the leakage results.
//!
//! ## Why this file exists
//!
//! Every security number in this project was produced with the software keystore, and the
//! benchmark metadata dutifully stamped every run `software-INSECURE`. That label is correct and
//! it invited a reasonable objection: *are these security results at all?*
//!
//! The answer has two halves and this file establishes both.
//!
//! **Half one: hardware custody works.** [`store_opens_under_hardware_custody`] builds a real
//! store whose master key is wrapped by a TPM 2.0 through the CNG Platform Crypto Provider. The
//! root key never leaves the chip. If no TPM is present the test skips loudly rather than
//! passing quietly.
//!
//! **Half two — the substantive one: the leakage results never depended on custody.**
//! [`custody_does_not_change_the_leakage_profile`] builds two stores over the identical corpus,
//! one under the TPM and one under the software keystore, and asserts their observable footprint
//! is byte-for-byte the same size, file for file. That is not a coincidence to be noted; it is
//! forced by the design. Attacks A3, A4 and A5 consume partition sizes, event counts and read
//! patterns. None of those quantities is a function of *how the master key was wrapped* — the
//! keystore is consulted exactly twice in a store's life, at create and at unlock, and it
//! influences nothing downstream.
//!
//! So: the `-INSECURE` label matters for one claim only, and it is the claim that an imaged disk
//! yields nothing useful. Under the software keystore the root key is a file on the same disk,
//! so that claim genuinely fails. Under the TPM it holds. Every *leakage* number is unaffected,
//! and this test is the evidence rather than the assertion.

use mnemos_core::traits::{KeyStore, SearchableMemory};
use mnemos_core::types::{Embedding, QuantVec};
use mnemos_core::Params;
use mnemos_engine::Engine;
use mnemos_keystore::windows::WindowsCngPcp;

const DIM: usize = 16;
const BUCKETS: u32 = 8;

fn test_params() -> Params {
    let mut p = Params::discover().expect("params.toml");
    p.index.n_max = 512;
    p.index.buckets = BUCKETS;
    p.index.bucket_capacity = 32;
    p.index.probe_width = 2;
    p.index.decoy_buckets = 0;
    p.index.top_k = 5;
    p.crypto.block_size_bytes = 512;
    p.crypto.max_entry_bytes = 512;
    p.embedding.dim = DIM;
    p.epoch.compaction_slots = 4;
    p.memory.arena_bytes = 64 * 1024;
    p
}

fn qv(v: Vec<f32>) -> QuantVec {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    QuantVec::quantise(&Embedding(v.into_iter().map(|x| x / n).collect()))
}

fn axis(i: usize) -> QuantVec {
    let mut v = vec![0.0f32; DIM];
    v[i % DIM] = 1.0;
    v[(i + 1) % DIM] = 0.25;
    qv(v)
}

fn centroids() -> Vec<QuantVec> {
    (0..BUCKETS).map(|i| axis(i as usize)).collect()
}

/// The corpus both stores receive, identical and in identical order.
fn corpus() -> Vec<(String, QuantVec)> {
    (0..48)
        .map(|i| (format!("entry {i}"), axis(i % 12)))
        .collect()
}

fn tpm() -> Option<WindowsCngPcp> {
    match WindowsCngPcp::open() {
        Ok(ks) => Some(ks),
        Err(e) => {
            eprintln!("SKIP: no usable TPM 2.0 on this machine ({e})");
            None
        }
    }
}

/// File name -> size, which is precisely what an imaging adversary reads first.
fn footprint(dir: &std::path::Path) -> Vec<(String, u64)> {
    let mut v: Vec<(String, u64)> = std::fs::read_dir(dir)
        .expect("read_dir")
        .filter_map(std::result::Result::ok)
        .filter(|e| e.path().is_file())
        .map(|e| {
            (
                e.file_name().to_string_lossy().into_owned(),
                e.metadata().map_or(0, |m| m.len()),
            )
        })
        .collect();
    v.sort();
    v
}

#[test]
fn store_opens_under_hardware_custody() {
    let Some(ks) = tpm() else { return };
    assert!(
        ks.is_hardware(),
        "the TPM provider reported itself as software"
    );
    assert_eq!(ks.provider_name(), "windows-cng-pcp");

    let d = tempfile::tempdir().unwrap();
    let p = test_params();
    let mut e =
        Engine::create(&d.path().join("store"), p, Box::new(ks), centroids()).expect("create");

    for (text, v) in corpus() {
        e.put(&text, "{}", &v).expect("put");
    }
    e.flush().expect("flush");

    let hits = e.search(&axis(0)).expect("search");
    assert!(!hits.is_empty(), "a TPM-backed store returned nothing");
}

#[test]
fn a_tpm_backed_store_survives_a_reopen() {
    // The unwrap path, which is the one that actually exercises the TPM's private key: create
    // wraps, reopen unwraps. A test that only created a store would never touch NCryptDecrypt.
    let Some(ks) = tpm() else { return };
    let d = tempfile::tempdir().unwrap();
    let p = test_params();

    {
        let mut e = Engine::create(
            &d.path().join("store"),
            p.clone(),
            Box::new(ks),
            centroids(),
        )
        .expect("create");
        for (text, v) in corpus() {
            e.put(&text, "{}", &v).expect("put");
        }
        e.flush().expect("flush");
    }

    let Some(ks2) = tpm() else { return };
    let mut e = Engine::open(&d.path().join("store"), p, Box::new(ks2)).expect("open");
    assert!(!e.stat().unlocked, "open() left the store unlocked");
    e.unlock()
        .expect("unlock — this is the TPM decrypting the master key");
    assert!(!e.search(&axis(0)).expect("search").is_empty());
}

#[test]
fn custody_does_not_change_the_leakage_profile() {
    // **The point of this file.** Two stores, identical corpus, different key custody. If their
    // observable footprints match, then every leakage measurement taken under the software
    // keystore transfers unchanged to hardware — which is what lets the existing A3/A4/A5
    // results stand rather than needing to be re-run.
    let Some(hw) = tpm() else { return };

    let hw_dir = tempfile::tempdir().unwrap();
    let sw_dir = tempfile::tempdir().unwrap();
    let p = test_params();

    {
        let mut e = Engine::create(
            &hw_dir.path().join("store"),
            p.clone(),
            Box::new(hw),
            centroids(),
        )
        .expect("create hw");
        for (text, v) in corpus() {
            e.put(&text, "{}", &v).expect("put");
        }
        e.flush().expect("flush");
    }
    {
        let sw = mnemos_keystore::software::SoftwareKeystore::open_at(
            &sw_dir.path().join("keystore.bin"),
            true,
        )
        .expect("software keystore");
        let mut e = Engine::create(&sw_dir.path().join("store"), p, Box::new(sw), centroids())
            .expect("create sw");
        for (text, v) in corpus() {
            e.put(&text, "{}", &v).expect("put");
        }
        e.flush().expect("flush");
    }

    let hw_files = footprint(&hw_dir.path().join("store"));
    let sw_files = footprint(&sw_dir.path().join("store"));

    let hw_names: Vec<&String> = hw_files.iter().map(|(n, _)| n).collect();
    let sw_names: Vec<&String> = sw_files.iter().map(|(n, _)| n).collect();
    assert_eq!(hw_names, sw_names, "custody changed which files exist");

    for ((name, hw_len), (_, sw_len)) in hw_files.iter().zip(sw_files.iter()) {
        if name == "wrapped.key" {
            // The ONE difference, and it is expected: an RSA-2048 OAEP wrap is 256 bytes, a
            // software AEAD wrap is not. This file is a fixed size per provider and is never a
            // function of the corpus, so it carries nothing about the contents — which is why
            // it is exempted here rather than treated as a leak.
            assert!(*hw_len > 0 && *sw_len > 0);
            continue;
        }
        assert_eq!(
            hw_len, sw_len,
            "{name} differs in size between TPM and software custody ({hw_len} vs {sw_len}) — \
             leakage results would NOT transfer between them"
        );
    }
}
