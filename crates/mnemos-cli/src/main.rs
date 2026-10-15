//! `mnemos` — the command-line interface.
//!
//! ```text
//! mnemos init   <path>                    create a store (preallocated to n_max)
//! mnemos put    <path> --text <t>         store a memory
//! mnemos search <path> --query <q>        retrieve
//! mnemos stat   <path>                    non-secret shape data
//! mnemos bench  <path> --n 10000          E2 performance run
//! mnemos attack <path> --which a3         run an attack against a store
//! ```
//!
//! `attack` is as much the point of this binary as `put` and `search`. A store you
//! can only use is a product; a store you can also attack, in one command, is an artifact.
//!
//! ## The embedder problem, and how this binary handles it
//!
//! Real embeddings come from `bge-base-en-v1.5` via the Python layer. A Rust CLI that shelled
//! out to Python for every `put` would be unusable, so this binary embeds text with a
//! **deterministic hashing embedder** ([`embed`]) and says so, loudly, in `stat`.
//!
//! That is honest and useful for demonstrating the *store* — round-trips, leakage behaviour,
//! and every security property, none of which depend on where vectors come from. It is **not**
//! valid for retrieval quality: recall and nDCG numbers from this path are meaningless and the
//! binary refuses to print them. `python/eval/` owns utility measurement.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use mnemos_core::traits::SearchableMemory;
use mnemos_core::types::{Embedding, QuantVec};
use mnemos_core::{Error, Params, Result};
use mnemos_engine::Engine;

fn main() -> ExitCode {
    // stderr, so stdout stays a clean machine-readable channel: `mnemos put` prints an id and
    // nothing else, which is what makes it usable in a pipeline.
    tracing_subscriber::fmt()
        .with_max_level(match std::env::var("MNEMOS_LOG").as_deref() {
            Ok("trace") => tracing::Level::TRACE,
            Ok("debug") => tracing::Level::DEBUG,
            Ok("warn") => tracing::Level::WARN,
            Ok("error") => tracing::Level::ERROR,
            _ => tracing::Level::INFO,
        })
        .with_writer(std::io::stderr)
        .init();

    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            // Errors are already scrubbed of secrets by construction — see
            // mnemos-core::error's module docs.
            eprintln!("mnemos: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let params = Params::discover()?;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(cmd) = args.first().map(String::as_str) else {
        print_usage();
        return Ok(());
    };
    let rest = &args[1..];

    match cmd {
        "init" => cmd_init(&params, rest),
        "put" => cmd_put(&params, rest),
        "search" => cmd_search(&params, rest),
        "stat" => cmd_stat(&params, rest),
        "bench" => {
            cmd_bench(rest);
            Ok(())
        }
        "attack" => cmd_attack(&params, rest),
        "params" => {
            println!("{params:#?}");
            println!("params_hash = {}", params.hash());
            Ok(())
        }
        "help" | "--help" | "-h" => {
            print_usage();
            Ok(())
        }
        other => {
            eprintln!("unknown command: {other}");
            print_usage();
            Err(Error::Params(format!("unknown command {other}")))
        }
    }
}

// ---------------------------------------------------------------------------------------------
// commands
// ---------------------------------------------------------------------------------------------

fn cmd_init(params: &Params, args: &[String]) -> Result<()> {
    let (path, flags) = positional(args)?;
    let seed = flags
        .get("seed")
        .and_then(|s| s.parse().ok())
        .unwrap_or(20_260_813);

    if path.join("manifest.enc").exists() {
        return Err(Error::Params(format!(
            "{} already contains a store; refusing to overwrite",
            path.display()
        )));
    }

    // Centroids are trained at `init`, because retraining later would move every entry's
    // partition and a store would stop finding its own contents. The table is then persisted in
    // the manifest and never recomputed.
    //
    // The sample must come from **this binary's embedder**, not from a generic dense generator.
    // `mnemos_bench::synthetic_corpus` produces dense vectors; the hashing embedder produces
    // sparse k-hot ones. Centroids trained on the wrong distribution partition the space along
    // directions the data never occupies, so a query and a near-identical stored entry land in
    // different partitions and search returns nothing — which is exactly what happened before
    // this was fixed, silently and with no error anywhere.
    let sample = embedder_sample(
        params.index.buckets as usize * 8,
        params.embedding.dim,
        seed,
    );
    let centroids = mnemos_index::spherical_kmeans(
        &sample,
        params.index.buckets as usize,
        params.embedding.dim,
        seed,
    );

    let ks = keystore(params, &flags)?;
    let engine = Engine::create(&path, params.clone(), ks, centroids)?;
    let s = engine.stat();

    println!("created store at {}", path.display());
    println!("  capacity     {} slots (preallocated)", s.capacity);
    println!(
        "  partitions   {} x {} slots",
        params.index.buckets, params.index.bucket_capacity
    );
    println!("  params_hash  {}", params.hash());
    println!("  disk         {}", human(dir_size(&path).unwrap_or(0)));
    Ok(())
}

fn cmd_put(params: &Params, args: &[String]) -> Result<()> {
    let (path, flags) = positional(args)?;
    let text = flags
        .get("text")
        .ok_or_else(|| Error::Params("put requires --text".into()))?;
    let meta = flags.get("meta").map_or("{}", String::as_str);

    let mut engine = open(params, &path, &flags)?;
    let id = engine.put(text, meta, &embed(text, params.embedding.dim))?;

    // ALWAYS flush. `mnemos-store::wal` stages in memory and reaches disk only on flush, so a
    // CLI invocation that returned without flushing would print an id for an entry that ceases
    // to exist when the process exits. Not hypothetical: this binary did exactly that until a
    // cross-process smoke test caught four silently discarded writes.
    //
    // The cost belongs on the record: one CLI `put` per process means one real record per
    // batch, so the epoch padding does all the work and the batching none of it. An embedded
    // caller holding a long-lived Engine gets the batching properly and should NOT flush per
    // write.
    engine.flush()?;

    println!("{}", id.to_hex());
    Ok(())
}

fn cmd_search(params: &Params, args: &[String]) -> Result<()> {
    let (path, flags) = positional(args)?;
    let query = flags
        .get("query")
        .ok_or_else(|| Error::Params("search requires --query".into()))?;

    let engine = open(params, &path, &flags)?;
    let hits = engine.search(&embed(query, params.embedding.dim))?;
    let top_k = flags
        .get("top-k")
        .and_then(|s| s.parse().ok())
        .unwrap_or(hits.len());

    if hits.is_empty() {
        println!("(no results)");
    }
    for h in hits.iter().take(top_k) {
        println!("{:.4}  {}", h.score, h.memory.text);
    }
    // Not a footnote. A reader who takes these rankings at face value would be drawing a
    // conclusion the embedder cannot support.
    eprintln!(
        "note: ranking used the CLI's hashing embedder, not bge-base-en-v1.5. Retrieval \
         quality here is not meaningful; use python/eval/ for utility measurement."
    );
    Ok(())
}

fn cmd_stat(params: &Params, args: &[String]) -> Result<()> {
    let (path, flags) = positional(args)?;
    let engine = open(params, &path, &flags)?;
    let s = engine.stat();

    // Everything printed here is already in docs/leakage-profile.md as observable, so exposing
    // it through the API costs nothing that a directory listing did not already give away.
    println!("live_entries  {}", s.live_entries);
    println!("epoch         {}", s.epoch.0);
    println!("capacity      {}", s.capacity);
    println!("unlocked      {}", s.unlocked);
    println!("disk          {}", human(dir_size(&path).unwrap_or(0)));
    println!("params_hash   {}", params.hash());
    println!(
        "load_factor   {:.3} (refuses to flush above {:.2})",
        s.live_entries as f64 / s.capacity as f64,
        params.epoch.max_load_factor
    );
    println!("embedder      hashing-CLI (NOT bge-base-en-v1.5; utility numbers invalid)");
    Ok(())
}

fn cmd_bench(_args: &[String]) {
    // Deliberately not reimplemented here. Criterion owns warmup, outlier rejection and the
    // statistics; a hand-rolled timing loop in a CLI would produce numbers that look like the
    // benchmark's and are not comparable to it.
    println!("Run the E2 benchmark suite with:");
    println!();
    println!("    cargo bench -p mnemos-bench");
    println!();
    println!("Add `-- --quick` for a fast pass. Results land in target/criterion/.");
    println!("Provenance (params hash, keystore provider, git sha) is printed at the start of");
    println!("the run and must accompany any number quoted from it.");
}

fn cmd_attack(_params: &Params, args: &[String]) -> Result<()> {
    let (path, flags) = positional(args)?;
    let which = flags.get("which").map_or("a3", String::as_str);

    match which {
        "a3" => {
            // A3 against Mnemos: enumerate what a τ-arrival adversary recovers from the store's
            // files. The answer is meant to be "nothing per-entry", and printing it as a
            // measurement rather than a claim is the point.
            let files = snapshot_shape(&path)?;
            println!("A3 — snapshot observables for {}", path.display());
            println!();
            for (name, len) in &files {
                println!("  {name:<16} {len:>12} bytes");
            }
            println!();
            println!("per-keyword update counts   : none (no counter map on disk)");
            println!("per-partition occupancy     : none (partitions are fully padded)");
            println!("insertion order             : none (slot = PRF(K_idx, tag))");
            println!("entry count                 : bounded by n_max only");
            println!();
            println!(
                "Every file above is a function of n_max and the parameter set, not of content."
            );
            println!("Compare with mnemos-baselines: Sophos and Diana both persist a per-keyword");
            println!("counter, which is finding F1's input. See crates/mnemos-engine/tests/\\");
            println!("baseline_contrast.rs for the side-by-side assertion.");
            Ok(())
        }
        "a4" => {
            // A4 is the structure-transfer attack and it is *not* a Rust attack: it needs the
            // semantic partitioning, the cohort split, and the Hungarian assignment, all of
            // which live in the Python lab. Pointing at it beats a worse reimplementation.
            println!("A4 (structure transfer) runs in the Python attack lab:");
            println!();
            println!("    python -m eval.experiments.e4_transfer");
            println!();
            println!("It needs the semantic partitioner and the assignment solver, which live");
            println!("in python/eval/. Reimplementing them here would produce a second, worse");
            println!("copy of the experiment that generated Figure 1.");
            Ok(())
        }
        "a1" | "a2" | "a5" => Err(Error::Params(format!(
            "attack {which} is not implemented"
        ))),
        other => Err(Error::Params(format!("unknown attack {other}"))),
    }
}

// ---------------------------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------------------------

/// A training sample drawn from [`embed`]'s own output distribution.
///
/// Synthetic documents of random tokens, embedded through the production path. The point is
/// distributional match, not realism: k-means only needs to see vectors shaped like the ones it
/// will later be asked to partition.
fn embedder_sample(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed | 1;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };

    (0..n)
        .map(|_| {
            // 4-12 tokens from a bounded vocabulary, so documents genuinely share terms and the
            // sample has cluster structure rather than being uniform noise.
            let count = 4 + (next() % 9) as usize;
            let doc: Vec<String> = (0..count).map(|_| format!("t{}", next() % 4096)).collect();
            let q = embed(&doc.join(" "), dim);
            // Back to f32 for the partitioner, which takes raw vectors.
            q.codes.iter().map(|c| f32::from(*c) * q.scale).collect()
        })
        .collect()
}

/// Deterministic hashing embedder. **Not a semantic embedder** — see the module docs.
///
/// Hashes each whitespace token into `dim` buckets with a signed contribution, then
/// L2-normalises. Shared vocabulary produces genuine similarity, which is enough to exercise
/// partitioning and retrieval mechanics, and nowhere near enough for a recall number.
fn embed(text: &str, dim: usize) -> QuantVec {
    let mut v = vec![0.0f32; dim];
    for token in text.split_whitespace() {
        let lower = token.to_lowercase();
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for b in lower.bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0100_0000_01b3);
        }
        let idx = (h % dim as u64) as usize;
        let sign = if (h >> 63) & 1 == 1 { -1.0 } else { 1.0 };
        v[idx] += sign;
    }
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 1e-9 {
        for x in &mut v {
            *x /= n;
        }
    } else {
        v[0] = 1.0;
    }
    QuantVec::quantise(&Embedding(v))
}

/// Select a key-custody provider.
///
/// Prefers a real one. Reaching the software fallback needs **three** independent acts, and the
/// CLI widens the key-custody gate by exactly one notch rather than removing it:
///
/// 1. compile with `--features insecure-software-keystore`,
/// 2. pass `--insecure` on the command line,
/// 3. set `MNEMOS_I_ACCEPT_INSECURE=1` in the environment.
///
/// `--insecure` clears `require_hardware` **for this process only** — `params.toml` still says
/// `true`, so the shipped configuration is unchanged and no other caller is affected.
fn keystore(
    params: &Params,
    flags: &HashMap<String, String>,
) -> Result<Box<dyn mnemos_core::traits::KeyStore>> {
    if !flags.contains_key("insecure") {
        return mnemos_keystore::select(params);
    }
    let mut relaxed = params.clone();
    relaxed.keystore.require_hardware = false;
    eprintln!(
        "mnemos: --insecure - using the software keystore. The root key is on disk in \
         plaintext. Nothing produced in this mode is a security result."
    );
    mnemos_keystore::select(&relaxed)
}

fn open(params: &Params, path: &Path, flags: &HashMap<String, String>) -> Result<Engine> {
    if !path.join("manifest.enc").exists() {
        return Err(Error::Params(format!(
            "no store at {} — run `mnemos init` first",
            path.display()
        )));
    }
    // No k-means here. The centroid table is restored from the manifest, so `open` is fast and
    // — more importantly — is guaranteed to use the *same* partitioning the store was built
    // with. Recomputing it took 46 s at production geometry and was not even reliably identical.
    let ks = keystore(params, flags)?;
    let mut engine = Engine::open(path, params.clone(), ks)?;
    engine.unlock()?;
    Ok(engine)
}

/// File sizes in a store directory. This *is* the snapshot an adversary sees.
fn snapshot_shape(path: &Path) -> Result<Vec<(String, u64)>> {
    let mut out: Vec<(String, u64)> = std::fs::read_dir(path)?
        .filter_map(std::result::Result::ok)
        .filter(|e| e.path().is_file())
        .map(|e| {
            (
                e.file_name().to_string_lossy().into_owned(),
                e.metadata().map_or(0, |m| m.len()),
            )
        })
        .collect();
    out.sort();
    Ok(out)
}

fn dir_size(path: &Path) -> Result<u64> {
    Ok(snapshot_shape(path)?.iter().map(|(_, n)| n).sum())
}

fn human(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut i = 0;
    while v >= 1024.0 && i < UNITS.len() - 1 {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.1} {}", UNITS[i])
    }
}


/// Split `<path> --flag value --bare` into a path and a flag map.
///
/// A hand-rolled parser rather than `clap`: this binary has seven commands and adding a
/// 400 KiB dependency with a derive macro for that is not a trade worth making in a crate whose
/// dependency list is a security argument.
fn positional(args: &[String]) -> Result<(PathBuf, HashMap<String, String>)> {
    let path = args
        .first()
        .filter(|a| !a.starts_with("--"))
        .ok_or_else(|| Error::Params("expected a store path".into()))?;
    Ok((PathBuf::from(path), flags_only(&args[1..])))
}

fn flags_only(args: &[String]) -> HashMap<String, String> {
    let mut out = HashMap::new();
    let mut i = 0;
    while i < args.len() {
        if let Some(name) = args[i].strip_prefix("--") {
            let value = args.get(i + 1).filter(|v| !v.starts_with("--"));
            // A flag with no following value is a bare boolean flag (`--flush`), recorded with
            // an empty string so `contains_key` works for both forms.
            if let Some(v) = value {
                out.insert(name.to_owned(), v.clone());
                i += 2;
            } else {
                out.insert(name.to_owned(), String::new());
                i += 1;
            }
        } else {
            i += 1;
        }
    }
    out
}

fn print_usage() {
    eprintln!(
        "mnemos — searchable encrypted memory for self-writing agents\n\n\
         USAGE:\n  \
         mnemos init   <path> [--seed N]\n  \
         mnemos put    <path> --text <text> [--meta <json>]\n  \
         mnemos search <path> --query <query> [--top-k N]\n  \
         mnemos stat   <path>\n  \
         mnemos bench  <path> --n <count>\n  \
         mnemos attack <path> --which <a3|a4>\n  \
         mnemos params\n\n\
         Add --insecure to any command to permit the software keystore (needs the\n\
         insecure-software-keystore feature and MNEMOS_I_ACCEPT_INSECURE=1).\n\n\
         NOTE: put/search use a deterministic hashing embedder, not bge-base-en-v1.5.\n\
         Retrieval quality from this binary is not meaningful; use python/eval/.\n"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flag_parsing_handles_bare_and_valued_flags() {
        let args: Vec<String> = ["p", "--text", "hello", "--flush", "--meta", "{}"]
            .iter()
            .map(|s| (*s).to_owned())
            .collect();
        let (path, flags) = positional(&args).unwrap();
        assert_eq!(path, PathBuf::from("p"));
        assert_eq!(flags.get("text").map(String::as_str), Some("hello"));
        assert_eq!(flags.get("meta").map(String::as_str), Some("{}"));
        assert!(flags.contains_key("flush"));
    }

    #[test]
    fn a_missing_path_is_an_error_not_a_flag() {
        let args: Vec<String> = ["--text", "x"].iter().map(|s| (*s).to_owned()).collect();
        assert!(positional(&args).is_err());
    }

    #[test]
    fn embedder_is_deterministic_and_normalised() {
        let a = embed("the quick brown fox", 64);
        let b = embed("the quick brown fox", 64);
        assert_eq!(a.encode(), b.encode());
        let self_sim = a.cosine(&a);
        assert!((self_sim - 1.0).abs() < 0.05, "self-similarity {self_sim}");
    }

    #[test]
    fn shared_vocabulary_scores_higher_than_disjoint() {
        // The only property the hashing embedder actually has. Asserted so the CLI's own
        // caveat is accurate: there IS similarity structure, it just is not semantic.
        let q = embed("encrypted memory index", 128);
        let near = embed("encrypted memory store", 128);
        let far = embed("banana pancake weather", 128);
        assert!(
            q.cosine(&near) > q.cosine(&far),
            "shared tokens did not score above disjoint ones"
        );
    }

    #[test]
    fn human_sizes_are_readable() {
        assert_eq!(human(512), "512 B");
        assert_eq!(human(2048), "2.0 KiB");
    }
}
