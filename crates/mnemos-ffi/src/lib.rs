//! # mnemos-ffi
//!
//! `PyO3` bindings. **The only surface Python may touch.**
//!
//! Deliberately thin: it exposes `mnemos-engine`'s public API and contains no logic. Logic here
//! would be logic outside the audited core, and every line of it would need its own argument
//! about plaintext residency.
//!
//! ## The residency boundary lives here
//!
//! Python holds plaintext before `put` and after `search`. That is unavoidable — the embedder
//! is Python — and it is declared, argued and bounded in `ARCHITECTURE.md` §6.3 rather than
//! hidden. Two rules keep the gap as narrow as it can be:
//!
//! 1. **Take plaintext as early as possible, return it as late as possible.** Text crosses into
//!    Rust in the same call that encrypts it.
//! 2. **Never return key material, tags, or slot addresses to Python.** Python gets memories and
//!    scores. Anything that would let Python reconstruct the store's internal structure stays
//!    on this side of the boundary — otherwise the attack harness could accidentally learn more
//!    than the adversary is entitled to, and every attack result would be suspect.
//!
//! Rule 2 is why `python/eval/attacks/` imports `mnemos.trace` (the event stream) rather than
//! this module. It is enforced by inspection of this file's signatures: no method returns a
//! `Tag`, a `SlotAddr`, a key, or a bucket id. `set_trace_sink` writes the *adversary's* event
//! stream to disk, which is a different thing and deliberately the only structural channel out.
//!
//! ## Why `flush` is exposed and `mcp_server` still does not call it
//!
//! Tests and the demo need deterministic flush points. Application code must not use it: the
//! epoch batching is what bounds Δ, so a caller that flushes per write has disabled the central
//! security mechanism. The restriction belongs in the layer that has application semantics —
//! `python/mnemos/mcp_server.py` — not in a binding that would then be untestable.

#![forbid(unsafe_code)]
// `clippy::useless_conversion` fires inside the code `#[pymethods]` generates around every
// `PyResult` return type, not on anything written here. Scoped to this crate so the workspace
// lint policy is unchanged everywhere the lint is meaningful.
#![allow(
    clippy::useless_conversion,
    reason = "false positive in the #[pymethods] expansion"
)]

use std::path::PathBuf;
use std::sync::Arc;

use mnemos_core::event::{EventSink, LeakEvent};
use mnemos_core::traits::SearchableMemory;
use mnemos_core::types::{Embedding, MemoryId, QuantVec};
use mnemos_core::Params;
use mnemos_engine::Engine;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Map a core error onto the closest Python exception.
///
/// `Locked` becomes `RuntimeError` rather than a custom type on purpose: callers should handle
/// it, and a custom exception class in a thin binding is an API that has to be documented,
/// versioned and imported for one condition.
fn err(e: mnemos_core::Error) -> PyErr {
    match e {
        mnemos_core::Error::Io(io) => PyIOError::new_err(io.to_string()),
        mnemos_core::Error::Params(m) | mnemos_core::Error::Format(m) => PyValueError::new_err(m),
        other => PyRuntimeError::new_err(other.to_string()),
    }
}

/// Writes `LeakEvent`s to a JSONL file for `python/eval` to consume.
///
/// **This is the adversary's transcript**, and it is the only structural information that leaves
/// the Rust side. `python/eval/attacks/` reads the file through `mnemos.trace`, which strips the
/// `decoy` flag — a field the evaluation needs and the adversary cannot have.
struct JsonlSink {
    path: PathBuf,
    file: parking_lot::Mutex<Option<std::fs::File>>,
}

impl JsonlSink {
    fn new(path: PathBuf) -> std::io::Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            path,
            file: parking_lot::Mutex::new(Some(file)),
        })
    }
}

impl EventSink for JsonlSink {
    fn emit(&self, event: LeakEvent) {
        use std::io::Write as _;
        // A failed write must not take down the store. The trace is an evaluation artifact;
        // losing a line degrades a measurement, whereas panicking here would abort a flush
        // half-way and corrupt the thing being measured.
        if let Some(f) = self.file.lock().as_mut() {
            if let Ok(line) = serde_json::to_string(&event) {
                let _ = writeln!(f, "{line}");
            }
        }
    }
}

impl std::fmt::Debug for JsonlSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JsonlSink({})", self.path.display())
    }
}

/// A Mnemos store, from Python.
///
/// Construct with [`PyEngine::create`] or [`PyEngine::open`]. `open` starts **locked**; call
/// `unlock` before use.
#[pyclass(name = "Engine", module = "mnemos_native")]
pub struct PyEngine {
    inner: Engine,
    dim: usize,
}

impl PyEngine {
    /// Quantise a Python-supplied vector through **exactly** the production path.
    ///
    /// Not a convenience. If Python quantised and handed over codes, two representations of the
    /// same vector could differ by a rounding step, and two stores holding the same logical
    /// content would stop being byte-identical — which is the history-independence property
    /// `docs/adr/0003` exists to guarantee. Taking floats and quantising here makes that
    /// impossible to get wrong from the Python side.
    fn quantise(&self, v: Vec<f32>) -> PyResult<QuantVec> {
        if v.len() != self.dim {
            return Err(PyValueError::new_err(format!(
                "vector has {} dimensions but this store expects {}",
                v.len(),
                self.dim
            )));
        }
        Ok(QuantVec::quantise(&Embedding(v)))
    }
}

#[pymethods]
impl PyEngine {
    /// Create a new store, preallocated to `n_max`. Starts **unlocked**.
    ///
    /// `centroids` is the partition table, one vector per partition. It is fixed at creation and
    /// persisted in the manifest: retraining later would move every entry's partition and the
    /// store would stop finding its own contents.
    #[staticmethod]
    #[pyo3(signature = (path, centroids, params_path=None, allow_insecure_keystore=false))]
    fn create(
        path: &str,
        centroids: Vec<Vec<f32>>,
        params_path: Option<&str>,
        allow_insecure_keystore: bool,
    ) -> PyResult<Self> {
        let params = load_params(params_path)?;
        let dim = params.embedding.dim;

        if centroids.len() != params.index.buckets as usize {
            return Err(PyValueError::new_err(format!(
                "expected {} centroids, got {}",
                params.index.buckets,
                centroids.len()
            )));
        }
        let table: Vec<QuantVec> = centroids
            .into_iter()
            .map(|c| QuantVec::quantise(&Embedding(c)))
            .collect();

        let ks = select_keystore(&params, allow_insecure_keystore)?;
        let inner = Engine::create(std::path::Path::new(path), params, ks, table).map_err(err)?;
        Ok(Self { inner, dim })
    }

    /// Open an existing store. Starts **locked** — call `unlock`.
    ///
    /// Takes no centroid table: it is restored from the encrypted manifest.
    #[staticmethod]
    #[pyo3(signature = (path, params_path=None, allow_insecure_keystore=false))]
    fn open(
        path: &str,
        params_path: Option<&str>,
        allow_insecure_keystore: bool,
    ) -> PyResult<Self> {
        let params = load_params(params_path)?;
        let dim = params.embedding.dim;
        let ks = select_keystore(&params, allow_insecure_keystore)?;
        let inner = Engine::open(std::path::Path::new(path), params, ks).map_err(err)?;
        Ok(Self { inner, dim })
    }

    /// Unwrap the master key from hardware into the locked arena.
    fn unlock(&mut self) -> PyResult<()> {
        self.inner.unlock().map_err(err)
    }

    /// Zeroise the master key and the locked arena.
    fn lock(&mut self) {
        self.inner.lock();
    }

    /// Store a memory. Returns its hex id.
    ///
    /// Returns as soon as the record is staged; it becomes durable at the next epoch flush.
    #[pyo3(signature = (text, embedding, meta="{}"))]
    fn put(&mut self, text: &str, embedding: Vec<f32>, meta: &str) -> PyResult<String> {
        let v = self.quantise(embedding)?;
        self.inner
            .put(text, meta, &v)
            .map(MemoryId::to_hex)
            .map_err(err)
    }

    /// Retrieve the nearest memories as `(text, meta, score)`.
    ///
    /// Constant-volume: the same number of slot reads happens whatever matches, so the returned
    /// list being short does not mean less work was done.
    ///
    /// No id is returned. Ids are a correlation handle and Python does not need one to read;
    /// `put` and `update` are the only places an id crosses the boundary.
    fn search(&self, embedding: Vec<f32>) -> PyResult<Vec<(String, String, f32)>> {
        let v = self.quantise(embedding)?;
        Ok(self
            .inner
            .search(&v)
            .map_err(err)?
            .into_iter()
            .map(|h| (h.memory.text.clone(), h.memory.meta.clone(), h.score))
            .collect())
    }

    /// Replace a memory. **Returns a NEW id** — updates are delete-then-put, never in-place,
    /// because an in-place rewrite is a same-slot write and therefore the correlation signal a
    /// live adversary most wants (ADR-0004).
    #[pyo3(signature = (memory_id, text, embedding, meta="{}"))]
    fn update(
        &mut self,
        memory_id: &str,
        text: &str,
        embedding: Vec<f32>,
        meta: &str,
    ) -> PyResult<String> {
        let id = parse_id(memory_id)?;
        let v = self.quantise(embedding)?;
        self.inner
            .update(id, text, meta, &v)
            .map(MemoryId::to_hex)
            .map_err(err)
    }

    /// Delete a memory (Type-II backward privacy).
    fn delete(&mut self, memory_id: &str) -> PyResult<()> {
        self.inner.delete(parse_id(memory_id)?).map_err(err)
    }

    /// **Evaluation only.** The index bucket holding `memory_id`, found by its tag, or `None`.
    ///
    /// Scores an attack against where a record actually is -- quantised assignment and spills
    /// included -- rather than where a float recomputation says it should be.
    fn actual_bucket(&self, memory_id: &str) -> PyResult<Option<u32>> {
        self.inner.actual_bucket(parse_id(memory_id)?).map_err(err)
    }

    /// Force an epoch flush.
    ///
    /// **Tests and the demo only.** Calling this per write defeats the batching that bounds Δ;
    /// see the module docs.
    fn flush(&mut self) -> PyResult<u64> {
        self.inner.flush().map(|e| e.0).map_err(err)
    }

    /// Non-secret shape data. Everything here is already observable from a directory listing.
    fn stat<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stat();
        let d = PyDict::new_bound(py);
        d.set_item("live_entries", s.live_entries)?;
        d.set_item("epoch", s.epoch.0)?;
        d.set_item("capacity", s.capacity)?;
        d.set_item("unlocked", s.unlocked)?;
        Ok(d)
    }

    /// Start writing `LeakEvent`s as JSONL to `path` — the adversary's transcript.
    ///
    /// Appends, so a trace spanning several sessions is one file.
    fn set_trace_sink(&mut self, path: &str) -> PyResult<()> {
        let sink =
            JsonlSink::new(PathBuf::from(path)).map_err(|e| PyIOError::new_err(e.to_string()))?;
        self.inner.set_event_sink(Arc::new(sink));
        Ok(())
    }

    fn __repr__(&self) -> String {
        let s = self.inner.stat();
        format!(
            "<mnemos_native.Engine entries={} epoch={} unlocked={}>",
            s.live_entries, s.epoch.0, s.unlocked
        )
    }
}

/// A Mnemos v2 store, from Python. See `crates/mnemos-v2`.
///
/// Unlike v1, search results carry the record id, because consolidation (update and delete of a
/// retrieved memory) is part of what the experiments exercise.
#[pyclass(name = "EngineV2", module = "mnemos_native")]
pub struct PyEngineV2 {
    inner: mnemos_v2::EngineV2,
    dim: usize,
}

fn parse_ablation(d: Option<&Bound<'_, PyDict>>) -> PyResult<mnemos_v2::Ablation> {
    let mut a = mnemos_v2::Ablation::default();
    let Some(d) = d else { return Ok(a) };
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        let on: bool = v.extract()?;
        match key.as_str() {
            "growing_regions" => a.growing_regions = on,
            "unsealed_slots" => a.unsealed_slots = on,
            "slot_writes_only" => a.slot_writes_only = on,
            "first_free_slot" => a.first_free_slot = on,
            "no_padding" => a.no_padding = on,
            "spill" => a.spill = on,
            "identity_permutation" => a.identity_permutation = on,
            "immediate_delete" => a.immediate_delete = on,
            "result_reads" => a.result_reads = on,
            other => return Err(PyValueError::new_err(format!("unknown ablation {other:?}"))),
        }
    }
    Ok(a)
}

#[pymethods]
impl PyEngineV2 {
    /// Create a v2 store. Starts **unlocked**.
    #[staticmethod]
    #[pyo3(signature = (path, centroids, params_path=None, allow_insecure_keystore=false, ablation=None))]
    fn create(
        path: &str,
        centroids: Vec<Vec<f32>>,
        params_path: Option<&str>,
        allow_insecure_keystore: bool,
        ablation: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let params = load_params(params_path)?;
        let dim = params.embedding.dim;
        let table: Vec<QuantVec> = centroids
            .into_iter()
            .map(|c| QuantVec::quantise(&Embedding(c)))
            .collect();
        let ks = select_keystore(&params, allow_insecure_keystore)?;
        let abl = parse_ablation(ablation)?;
        let inner = mnemos_v2::EngineV2::create(std::path::Path::new(path), params, ks, table, abl)
            .map_err(err)?;
        Ok(Self { inner, dim })
    }

    /// Open a v2 store. Starts **locked**.
    #[staticmethod]
    #[pyo3(signature = (path, params_path=None, allow_insecure_keystore=false, ablation=None))]
    fn open(
        path: &str,
        params_path: Option<&str>,
        allow_insecure_keystore: bool,
        ablation: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let params = load_params(params_path)?;
        let dim = params.embedding.dim;
        let ks = select_keystore(&params, allow_insecure_keystore)?;
        let abl = parse_ablation(ablation)?;
        let inner =
            mnemos_v2::EngineV2::open(std::path::Path::new(path), params, ks, abl).map_err(err)?;
        Ok(Self { inner, dim })
    }

    fn unlock(&mut self) -> PyResult<()> {
        self.inner.unlock().map_err(err)
    }

    /// Flush anything staged or deleted, then lock. Raises on a failed flush.
    fn close(&mut self) -> PyResult<()> {
        self.inner.close().map_err(err)
    }

    fn lock(&mut self) {
        self.inner.lock();
    }

    #[pyo3(signature = (text, embedding, meta="{}"))]
    fn put(&mut self, text: &str, embedding: Vec<f32>, meta: &str) -> PyResult<String> {
        let v = quantise_checked(self.dim, embedding)?;
        self.inner
            .put(text, meta, &v)
            .map(MemoryId::to_hex)
            .map_err(err)
    }

    /// `(id, text, meta, score)` for the top results.
    fn search(&self, embedding: Vec<f32>) -> PyResult<Vec<(String, String, String, f32)>> {
        let v = quantise_checked(self.dim, embedding)?;
        Ok(self
            .inner
            .search(&v)
            .map_err(err)?
            .into_iter()
            .map(|h| {
                (
                    h.memory.id.to_hex(),
                    h.memory.text.clone(),
                    h.memory.meta.clone(),
                    h.score,
                )
            })
            .collect())
    }

    #[pyo3(signature = (memory_id, text, embedding, meta="{}"))]
    fn update(
        &mut self,
        memory_id: &str,
        text: &str,
        embedding: Vec<f32>,
        meta: &str,
    ) -> PyResult<String> {
        let id = parse_id(memory_id)?;
        let v = quantise_checked(self.dim, embedding)?;
        self.inner
            .update(id, text, meta, &v)
            .map(MemoryId::to_hex)
            .map_err(err)
    }

    fn delete(&mut self, memory_id: &str) -> PyResult<()> {
        self.inner.delete(parse_id(memory_id)?).map_err(err)
    }

    fn flush(&mut self) -> PyResult<u64> {
        self.inner.flush().map(|e| e.0).map_err(err)
    }

    /// Flush if the schedule says one is due; returns the new epoch or `None`.
    fn tick(&mut self) -> PyResult<Option<u64>> {
        Ok(self.inner.tick().map_err(err)?.map(|e| e.0))
    }

    fn stat<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stat();
        let st = self.inner.stats();
        let g = self.inner.geometry();
        let d = PyDict::new_bound(py);
        d.set_item("live_entries", s.live_entries)?;
        d.set_item("epoch", s.epoch.0)?;
        d.set_item("capacity", s.capacity)?;
        d.set_item("unlocked", s.unlocked)?;
        d.set_item("staged", self.inner.staged_len())?;
        d.set_item("tombstones", self.inner.tombstones_len())?;
        d.set_item("flushes", st.flushes)?;
        d.set_item("region_writes", st.region_writes)?;
        d.set_item("region_bytes_written", st.region_bytes_written)?;
        d.set_item("manifest_bytes_written", st.manifest_bytes_written)?;
        d.set_item("refused", st.refused)?;
        d.set_item("ghost_writes", st.ghost_writes)?;
        d.set_item("spilled", st.spilled)?;
        d.set_item("purged", st.purged)?;
        d.set_item("slot_len", g.slot_len())?;
        d.set_item("region_len", g.region_len())?;
        d.set_item("regions_per_search", g.regions_per_search())?;
        d.set_item("max_record_bytes", g.max_record_bytes())?;
        Ok(d)
    }

    fn set_trace_sink(&mut self, path: &str) -> PyResult<()> {
        let sink =
            JsonlSink::new(PathBuf::from(path)).map_err(|e| PyIOError::new_err(e.to_string()))?;
        self.inner.set_event_sink(Arc::new(sink));
        Ok(())
    }

    /// **Evaluation ground truth only**: the physical region of a vector. Never an adversary input.
    fn ground_truth_region(&self, embedding: Vec<f32>) -> PyResult<u32> {
        let v = quantise_checked(self.dim, embedding)?;
        self.inner.ground_truth_region(&v).map_err(err)
    }

    /// **Evaluation ground truth only**: physical region of each logical partition.
    fn ground_truth_permutation(&self) -> Vec<u32> {
        self.inner.ground_truth_permutation()
    }
}

/// A Mnemos-M store, from Python. See `crates/mnemos-mem`.
///
/// The surface is deliberately the same as [`PyEngineV2`]'s, so an experiment can drive either
/// design through one code path and the comparison measures the design rather than the harness.
/// Two methods differ, and both are evaluation-only: `content_region` answers what a
/// content-placed design *would* have done with a vector, and `actual_region` answers where a
/// record really is. An adversary has neither.
#[pyclass(name = "EngineM", module = "mnemos_native")]
pub struct PyEngineM {
    inner: mnemos_mem::EngineM,
    dim: usize,
}

fn parse_m_ablation(d: Option<&Bound<'_, PyDict>>) -> PyResult<mnemos_mem::Ablation> {
    let mut a = mnemos_mem::Ablation::default();
    let Some(d) = d else { return Ok(a) };
    for (k, v) in d.iter() {
        let key: String = k.extract()?;
        let on: bool = v.extract()?;
        match key.as_str() {
            "content_placement" => a.content_placement = on,
            "on_demand_flush" => a.on_demand_flush = on,
            "disk_search" => a.disk_search = on,
            "no_padding" => a.no_padding = on,
            other => return Err(PyValueError::new_err(format!("unknown ablation {other:?}"))),
        }
    }
    Ok(a)
}

#[pymethods]
impl PyEngineM {
    /// Create a Mnemos-M store. Starts **unlocked**.
    #[staticmethod]
    #[pyo3(signature = (path, centroids, params_path=None, allow_insecure_keystore=false, ablation=None))]
    fn create(
        path: &str,
        centroids: Vec<Vec<f32>>,
        params_path: Option<&str>,
        allow_insecure_keystore: bool,
        ablation: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let params = load_params(params_path)?;
        let dim = params.embedding.dim;
        let table: Vec<QuantVec> = centroids
            .into_iter()
            .map(|c| QuantVec::quantise(&Embedding(c)))
            .collect();
        let ks = select_keystore(&params, allow_insecure_keystore)?;
        let abl = parse_m_ablation(ablation)?;
        let inner = mnemos_mem::EngineM::create(std::path::Path::new(path), params, ks, table, abl)
            .map_err(err)?;
        Ok(Self { inner, dim })
    }

    /// Open a Mnemos-M store. Starts **locked**.
    #[staticmethod]
    #[pyo3(signature = (path, params_path=None, allow_insecure_keystore=false, ablation=None))]
    fn open(
        path: &str,
        params_path: Option<&str>,
        allow_insecure_keystore: bool,
        ablation: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let params = load_params(params_path)?;
        let dim = params.embedding.dim;
        let ks = select_keystore(&params, allow_insecure_keystore)?;
        let abl = parse_m_ablation(ablation)?;
        let inner =
            mnemos_mem::EngineM::open(std::path::Path::new(path), params, ks, abl).map_err(err)?;
        Ok(Self { inner, dim })
    }

    fn unlock(&mut self) -> PyResult<()> {
        self.inner.unlock().map_err(err)
    }

    fn close(&mut self) -> PyResult<()> {
        self.inner.close().map_err(err)
    }

    fn lock(&mut self) {
        self.inner.lock();
    }

    #[pyo3(signature = (text, embedding, meta="{}"))]
    fn put(&mut self, text: &str, embedding: Vec<f32>, meta: &str) -> PyResult<String> {
        let v = quantise_checked(self.dim, embedding)?;
        self.inner
            .put(text, meta, &v)
            .map(MemoryId::to_hex)
            .map_err(err)
    }

    /// `(id, text, meta, score)` for the top results.
    fn search(&self, embedding: Vec<f32>) -> PyResult<Vec<(String, String, String, f32)>> {
        let v = quantise_checked(self.dim, embedding)?;
        Ok(self
            .inner
            .search(&v)
            .map_err(err)?
            .into_iter()
            .map(|h| {
                (
                    h.memory.id.to_hex(),
                    h.memory.text.clone(),
                    h.memory.meta.clone(),
                    h.score,
                )
            })
            .collect())
    }

    #[pyo3(signature = (memory_id, text, embedding, meta="{}"))]
    fn update(
        &mut self,
        memory_id: &str,
        text: &str,
        embedding: Vec<f32>,
        meta: &str,
    ) -> PyResult<String> {
        let id = parse_id(memory_id)?;
        let v = quantise_checked(self.dim, embedding)?;
        self.inner
            .update(id, text, meta, &v)
            .map(MemoryId::to_hex)
            .map_err(err)
    }

    fn delete(&mut self, memory_id: &str) -> PyResult<()> {
        self.inner.delete(parse_id(memory_id)?).map_err(err)
    }

    fn flush(&mut self) -> PyResult<u64> {
        self.inner.flush().map(|e| e.0).map_err(err)
    }

    /// Flush if the clock says one is due; returns the new epoch or `None`.
    ///
    /// This is how the clocked schedule is driven: an application calls it on a timer, and a
    /// flush happens every Δ whether or not the agent wrote anything.
    fn tick(&mut self) -> PyResult<Option<u64>> {
        Ok(self.inner.tick().map_err(err)?.map(|e| e.0))
    }

    fn stat<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stat();
        let st = self.inner.stats();
        let g = self.inner.geometry();
        let d = PyDict::new_bound(py);
        d.set_item("live_entries", s.live_entries)?;
        d.set_item("epoch", s.epoch.0)?;
        d.set_item("capacity", s.capacity)?;
        d.set_item("unlocked", s.unlocked)?;
        d.set_item("staged", self.inner.staged_len())?;
        d.set_item("tombstones", self.inner.tombstones_len())?;
        d.set_item("flushes", st.flushes)?;
        d.set_item("empty_flushes", st.empty_flushes)?;
        d.set_item("region_writes", st.region_writes)?;
        d.set_item("region_bytes_written", st.region_bytes_written)?;
        d.set_item("manifest_bytes_written", st.manifest_bytes_written)?;
        d.set_item("refused", st.refused)?;
        d.set_item("purged", st.purged)?;
        d.set_item("region_reads", st.region_reads)?;
        d.set_item("resident_bytes", st.resident_bytes)?;
        d.set_item("slot_len", g.slot_len())?;
        d.set_item("region_len", g.region_len())?;
        d.set_item("max_record_bytes", g.max_record_bytes())?;
        Ok(d)
    }

    fn set_trace_sink(&mut self, path: &str) -> PyResult<()> {
        let sink =
            JsonlSink::new(PathBuf::from(path)).map_err(|e| PyIOError::new_err(e.to_string()))?;
        self.inner.set_event_sink(Arc::new(sink));
        Ok(())
    }

    /// **Evaluation ground truth only**: the region a content-placed design would have used.
    fn content_region(&self, embedding: Vec<f32>) -> PyResult<u32> {
        let v = quantise_checked(self.dim, embedding)?;
        self.inner.content_region(&v).map_err(err)
    }

    /// **Evaluation ground truth only**: the region a record actually occupies.
    fn actual_region(&self, memory_id: &str) -> PyResult<Option<u32>> {
        self.inner.actual_region(parse_id(memory_id)?).map_err(err)
    }
}

/// The plaintext IVF baseline, from Python.
///
/// This is `mnemos-baselines`' in-tree index, not a numpy reimplementation. It shares
/// `QuantVec` and `spherical_kmeans` with the encrypted engines, so a latency ratio measured
/// against it is a statement about the security machinery rather than about two different
/// libraries. The earlier performance experiment used a numpy IVF whose `add` rebuilt its whole
/// array, which made the plaintext write path *slower* than the encrypted one at 10^4 records
/// and the ratio meaningless.
#[pyclass(name = "PlaintextIvf", module = "mnemos_native")]
pub struct PyPlaintextIvf {
    inner: mnemos_baselines::plain::PlaintextIvf,
    dim: usize,
}

#[pymethods]
impl PyPlaintextIvf {
    /// Build over an already-trained centroid table.
    #[new]
    #[pyo3(signature = (centroids, probe_width=4, top_k=10))]
    fn new(centroids: Vec<Vec<f32>>, probe_width: usize, top_k: usize) -> PyResult<Self> {
        let dim = centroids.first().map_or(0, Vec::len);
        let table: Vec<QuantVec> = centroids
            .into_iter()
            .map(|c| QuantVec::quantise(&Embedding(c)))
            .collect();
        let inner =
            mnemos_baselines::plain::PlaintextIvf::from_centroids(table, probe_width, top_k)
                .map_err(err)?;
        Ok(Self { inner, dim })
    }

    #[pyo3(signature = (text, embedding, meta="{}"))]
    fn put(&mut self, text: &str, embedding: Vec<f32>, meta: &str) -> PyResult<String> {
        let v = quantise_checked(self.dim, embedding)?;
        self.inner
            .put(text, meta, &v)
            .map(MemoryId::to_hex)
            .map_err(err)
    }

    /// `(id, text, meta, score)` from the IVF search, at the configured probe width.
    fn search(&self, embedding: Vec<f32>) -> PyResult<Vec<(String, String, String, f32)>> {
        let v = quantise_checked(self.dim, embedding)?;
        Ok(self
            .inner
            .search(&v)
            .map_err(err)?
            .into_iter()
            .map(|h| {
                (
                    h.memory.id.to_hex(),
                    h.memory.text.clone(),
                    h.memory.meta.clone(),
                    h.score,
                )
            })
            .collect())
    }

    /// Brute-force nearest neighbours: the recall ceiling for this quantisation.
    #[pyo3(signature = (embedding, top_k=10))]
    fn exact_search(
        &self,
        embedding: Vec<f32>,
        top_k: usize,
    ) -> PyResult<Vec<(String, String, String, f32)>> {
        let v = quantise_checked(self.dim, embedding)?;
        Ok(self
            .inner
            .exact_search(&v, top_k)
            .map_err(err)?
            .into_iter()
            .map(|h| {
                (
                    h.memory.id.to_hex(),
                    h.memory.text.clone(),
                    h.memory.meta.clone(),
                    h.score,
                )
            })
            .collect())
    }

    /// How many partitions this query would read. Variable, unlike the encrypted engines.
    fn probes_for(&self, embedding: Vec<f32>) -> PyResult<usize> {
        let v = quantise_checked(self.dim, embedding)?;
        Ok(self.inner.probes_for(&v).len())
    }

    fn stat<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.inner.stat();
        let d = PyDict::new_bound(py);
        d.set_item("live_entries", s.live_entries)?;
        d.set_item("epoch", s.epoch.0)?;
        d.set_item("staged", 0)?;
        Ok(d)
    }
}

fn quantise_checked(dim: usize, v: Vec<f32>) -> PyResult<QuantVec> {
    if v.len() != dim {
        return Err(PyValueError::new_err(format!(
            "vector has {} dimensions but this store expects {dim}",
            v.len()
        )));
    }
    Ok(QuantVec::quantise(&Embedding(v)))
}

fn parse_id(hex: &str) -> PyResult<MemoryId> {
    if hex.len() != 32 {
        return Err(PyValueError::new_err("memory id must be 32 hex characters"));
    }
    let mut out = [0u8; 16];
    for (i, byte) in out.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
            .map_err(|_| PyValueError::new_err("memory id is not valid hex"))?;
    }
    Ok(MemoryId(out))
}

fn load_params(path: Option<&str>) -> PyResult<Params> {
    match path {
        Some(p) => Params::load(p).map_err(err),
        None => Params::discover().map_err(err),
    }
}

/// Select a key-custody provider, optionally permitting the CI fallback.
///
/// `allow_insecure` clears `require_hardware` **for this call only** — `params.toml` still says
/// `true`, so the shipped configuration is unchanged and no other caller is affected. Reaching
/// the software keystore still needs the compile feature and `MNEMOS_I_ACCEPT_INSECURE=1`, so
/// this widens ADR-0005's gate by exactly one notch and keeps the opt-in visible at the call
/// site rather than hidden in an environment variable.
fn select_keystore(
    params: &Params,
    allow_insecure: bool,
) -> PyResult<Box<dyn mnemos_core::traits::KeyStore>> {
    if !allow_insecure {
        return mnemos_keystore::select(params).map_err(err);
    }
    let mut relaxed = params.clone();
    relaxed.keystore.require_hardware = false;
    mnemos_keystore::select(&relaxed).map_err(err)
}

/// Train a partition table from a sample — the **same** spherical k-means the index uses.
///
/// Exposed so Python does not grow a second implementation. Two partitioners that disagree
/// would make every recall comparison a measurement of the partitioner rather than of the
/// security machinery.
#[pyfunction]
#[allow(clippy::needless_pass_by_value, reason = "PyO3 hands us an owned Vec")]
#[pyo3(signature = (sample, buckets, dim, seed=20_260_813))]
fn train_centroids(
    // By value because PyO3 extracts an owned Vec from the Python list; a reference would just
    // move the same allocation one line later.
    sample: Vec<Vec<f32>>,
    buckets: usize,
    dim: usize,
    seed: u64,
) -> PyResult<Vec<Vec<f32>>> {
    // `seed` has a default, so `train_centroids(sample, buckets, seed)` is a legal call in which
    // the seed silently lands in `dim`. A plausible seed like 20_260_827 then asks for centroids
    // twenty million wide: several GB per centroid table, an OOM abort, and no clue why. Checked
    // against the sample's actual width, which is the only value `dim` can correctly take.
    if let Some(first) = sample.first() {
        if first.len() != dim {
            return Err(PyValueError::new_err(format!(
                "train_centroids: dim = {dim} but the sample vectors are {} wide. \
                 The signature is (sample, buckets, dim, seed); a seed passed in dim's \
                 position would allocate a centroid table of that width.",
                first.len()
            )));
        }
    }
    Ok(mnemos_index::spherical_kmeans(&sample, buckets, dim, seed)
        .into_iter()
        .map(|q| q.codes.iter().map(|c| f32::from(*c) * q.scale).collect())
        .collect())
}

/// `(k_probe + d) × C` — the constant-volume invariant, from the one place that defines it.
#[pyfunction]
#[pyo3(signature = (params_path=None))]
fn slots_read_per_search(params_path: Option<&str>) -> PyResult<u32> {
    Ok(load_params(params_path)?.slots_read_per_search())
}

/// Which keystore would actually guard the master key here.
///
/// The reproduction harness stamps this into `provenance.json`, and it has to be *asked* rather
/// than assumed: a hardcoded provider string next to the system, never compared to the system,
/// is the same class of defect as a leakage table that is not a test. A run on
/// `software-INSECURE` is not a security result, and six weeks later nobody remembers which
/// machine produced which JSON.
///
/// # Errors
/// If `params.toml` cannot be read.
#[pyfunction]
#[pyo3(signature = (params_path=None, allow_insecure_keystore=false))]
fn keystore_provider(params_path: Option<&str>, allow_insecure_keystore: bool) -> PyResult<String> {
    let params = load_params(params_path)?;
    match select_keystore(&params, allow_insecure_keystore) {
        Ok(ks) => Ok(ks.provider_name().to_owned()),
        // "unavailable" is a legitimate answer and a much better one than a guess. The caller
        // records what it was told.
        Err(e) => Ok(format!("unavailable: {e}")),
    }
}

/// The `mnemos_native` extension module.
#[pymodule]
fn mnemos_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyEngineV2>()?;
    m.add_class::<PyEngineM>()?;
    m.add_class::<PyPlaintextIvf>()?;
    m.add_function(wrap_pyfunction!(train_centroids, m)?)?;
    m.add_function(wrap_pyfunction!(slots_read_per_search, m)?)?;
    m.add_function(wrap_pyfunction!(keystore_provider, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
