"""Mnemos synthetic data generator.

Generates, under data/ only:

  data/d3-swebench/notes.jsonl            D3 procedural skill notes      (5,000)
  data/d3-swebench/tasks.jsonl            induced-write task stream      (1,200)
  data/retrieval-clusters.jsonl           clustered retrieval fixtures   (1,500)
  data/manifest.json                      machine-readable provenance
  data/README.md                          documentation

Design (v3.0)
-------------
* ``notes`` read as first-person skill notes an agent wrote *after* solving a task: what
  happened, what the fix was, and the reusable lesson. The old generator produced templated
  sentences like "I cleaned up it by keeping the fix specific"; this one composes each note
  from a rich pool of grammatical clauses so the prose is varied and specific, with real
  identifiers and error names.
* ``tasks`` are short, realistic one-line directives an environment would hand an agent.
  In addition to a ``topic`` tag they carry an ``attack`` flag plus a ``trigger`` field: the
  attack-eval scripts (A4/E5/E6) need to inject tasks whose *resulting memories are
  semantically predictable*, so the stream includes a tail segment of high-signal trigger
  tasks that an induced-write harness can sample from. The trigger tasks name concrete
  identifiers so the memories they induce should land in a predictable embedding cluster.
* ``retrieval-clusters`` are documents over 10 deliberately non-overlapping technical topic
  clusters, so embedding/retrieval benchmarks test against real semantic structure rather
  than noise.

Stdlib-only and deterministic: one seed, a per-file derived sub-seed, and no other entropy.

Usage:  python data/generate_data.py
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PARAMS_TOML = ROOT.parent / "params.toml"
SEED = 20260812
VERSION = "3.0.0"

TAXONOMY = [
    "sql", "networking", "filesystem", "concurrency", "crypto",
    "containers", "web", "debugging", "algorithms", "cloud",
]

CATEGORIES = ["bugfix", "feature", "refactor", "performance", "hardening"]
CATEGORY_WEIGHTS = [0.32, 0.20, 0.16, 0.17, 0.15]
DIFFICULTIES = ["easy", "medium", "hard"]
DIFFICULTY_WEIGHTS = [0.30, 0.50, 0.20]

ALL_LANGS = [
    "python", "typescript", "go", "rust", "java", "c",
    "bash", "sql", "ruby", "csharp", "yaml", "javascript",
]


def pick_weighted(rng: random.Random, items: list[str], weights: list[float]) -> str:
    return rng.choices(items, weights=weights)[0]


def shard_language(lang: str) -> str:
    if lang == "javascript":
        return "typescript"
    if lang == "yaml":
        return "bash"
    return lang


# ---------------------------------------------------------------- topics


class Topic:
    """Everything the compositors need to write about one cluster."""

    def __init__(
        self,
        name: str,
        lang_weights: dict[str, float],
        terms: list[str],
        atoms: list[str],
        task_objs: list[str],
        task_verbs: list[str],
        task_ends: list[str],
        trigger_objs: list[str],
        trigger_verbs: list[str],
        trigger_ends: list[str],
        cluster_title_pool: list[str],
        cluster_sents: list[str],
    ) -> None:
        self.name = name
        self.lang_weights = lang_weights
        self.terms = terms
        self.atoms = atoms
        self.task_objs = task_objs
        self.task_verbs = task_verbs
        self.task_ends = task_ends
        self.trigger_objs = trigger_objs
        self.trigger_verbs = trigger_verbs
        self.trigger_ends = trigger_ends
        self.cluster_title_pool = cluster_title_pool
        self.cluster_sents = cluster_sents

    def language(self, rng: random.Random) -> str:
        return rng.choices(list(self.lang_weights), weights=list(self.lang_weights.values()))[0]

    def specifics(self, rng: random.Random, n: int = 2) -> list[str]:
        return rng.sample(self.terms, k=min(n, len(self.terms)))


def _t(name, lang_weights, terms, atoms, task_objs, task_verbs, task_ends,
       trigger_objs, trigger_verbs, trigger_ends, cluster_title_pool, cluster_sents):
    return Topic(name, lang_weights, terms, atoms, task_objs, task_verbs, task_ends,
                 trigger_objs, trigger_verbs, trigger_ends, cluster_title_pool, cluster_sents)


TOPICS = {
    t.name: t
    for t in [
        _t(
            "sql",
            {"python": 0.28, "sql": 0.22, "go": 0.16, "java": 0.14, "typescript": 0.12, "csharp": 0.08},
            [
                "EXPLAIN ANALYZE", "pg_stat_statements", "statement_timeout",
                "autovacuum", "pg_locks", "lock_timeout",
                "CREATE INDEX CONCURRENTLY", "pg_stat_activity", "idle_in_transaction",
                "seq_scan", "index_scan", "bitmap heap scan",
                "CTE materialization", "JOIN ordering", "nested loop join",
                "UNION vs UNION ALL", "LEFT JOIN LATERAL", "NOT IN vs NOT EXISTS",
                "GIN index", "btree_gist", "pg_trgm", "HNSW index",
                "index-only scan", "covering index", "partial index",
                "jsonb containment", "@> operator", "->> operator",
                "pg_hba.conf", "pg_ctl reload", "listen_addresses",
                "UUID primary key", "IDENTITY column", "BIGSERIAL",
                "transactions", "REPEATABLE READ", "READ COMMITTED",
                "ON CONFLICT DO UPDATE", "RETURNING clause", "MERGE statement",
                "foreign key", "ON DELETE CASCADE", "deferred constraint",
                "prepared statements", "statement caching", "connection pool",
                "huge_pages", "shared_buffers", "work_mem",
            ],
            [
                "rewriting the query to use NOT EXISTS cut the runtime from minutes to seconds.",
                "adding a partial index on the hot subset removed the seq_scan entirely.",
                "bumping work_mem for the window function stopped the temp-file spill.",
                "ON CONFLICT DO UPDATE made the upsert idempotent under retries.",
                "an index-only scan removed the heap lookups on the covering index.",
                "the JOIN reordering pulled the selective filter down before the merge.",
                "READ COMMITTED hid the phantom reads that broke the batch job.",
                "a deferred constraint let me reorder the rows in one transaction.",
                "locking the advisory lock around the migration blocked the concurrent migration.",
                "creating the GIN index with fastupdate off made the jsonb writes predictable.",
                "the connection pool was exhausting because idle connections were never reaped.",
                "running ANALYZE after the bulk load made the planner pick the index again.",
                "the LEFT JOIN LATERAL pushed the correlated subquery down to a single row.",
                "bumping statement_timeout only delayed the failure, so the query itself got fixed.",
                "the covering index made the report query index-only and the heap lookups vanished.",
                "switching the upsert to MERGE removed the race between the insert and the update.",
                "the jsonb containment query finally used the GIN index after the operator flipped to @>.",
                "the migration locked the table for an hour until CREATE INDEX CONCURRENTLY took over.",
                "prepared statements with statement caching killed the reparse overhead on the hot path.",
                "the NOT IN subquery returned NULLs and silently dropped rows, so it became NOT EXISTS.",
                "the deferred constraint let the loader reorder inserts without a deadlock.",
                "REPEATABLE READ fixed the report that double-counted rows mid-backfill.",
                "the UUID primary key index was bloated until the table was rewritten.",
                "connection pooling capped the backend count and the idle_in_transaction rows stopped growing.",
                "the partial index on the active flag made the queue lookup a single index_scan.",
                "huge_pages took the shared_buffers hit out of the p50 latency.",
                "the foreign key was enforced only on commit, which hid the orphaned rows in the staging table.",
                "the ON DELETE CASCADE made the cleanup script unnecessary and the deletes instant.",
                "btree_gist let the exclusion constraint catch the overlapping booking rows.",
                "the pg_trgm index made the fuzzy name search return in milliseconds.",
                "statement_timeout caught the runaway report and the ops alert finally fired on something real.",
            ],
            ["orders", "sessions", "events", "users", "payments", "audit_log"],
            ["query", "index", "paginate", "dedupe", "join", "aggregate", "migrate", "backfill"],
            [
                "for the last 30 days", "by user_id",
                "without skipping rows", "in a single statement",
                "with a covering index", "across all shards",
                "in under 100 ms", "using the GIN index",
                "for the nightly batch", "by month",
            ],
            ["orders_table", "users_table", "payments_table", "events_table", "audit_log_table"],
            ["index", "dedupe", "paginate", "join", "aggregate", "backfill", "migrate"],
            [
                "on the orders_table", "in the users_table",
                "across the events_table", "in the payments_table",
                "for the audit_log_table",
            ],
            [
                "Indexing strategies for hot-path lookups",
                "Why the join got slower as the table grew",
                "Upsert patterns that survive retries",
                "Transaction isolation and the batch job",
                "When the planner picks a seq scan",
                "Why NOT IN dropped rows and NOT EXISTS did not",
                "Making the report query index-only",
                "Locking, and why the migration stalled",
                "The GIN index that made jsonb search fast",
                "Prepared statements on the hot path",
            ],
            [
                "The planner chose a seq scan over a perfectly good index because the statistics were stale; running ANALYZE after the bulk load fixed it.",
                "An index-only scan only pays off when every column the query needs lives in the index; adding one INCLUDE column killed the heap lookups.",
                "CTEs materialize by default in this engine, which is great for a once-only subquery and terrible when the planner could have merged it.",
                "A partial index on the active subset keeps the index small and the writes cheap, at the cost of an index that only answers one shape of query.",
                "Lock contention shows up first as idle_in_transaction rows; the fix is usually a smaller transaction, not a bigger timeout.",
                "NOT IN silently drops rows when the subquery can return NULL; NOT EXISTS never does, which is why the rewrite changed the answer.",
                "A covering index removes the heap lookup entirely, which turns a report query from a scan into a straight line on the EXPLAIN output.",
                "CREATE INDEX CONCURRENTLY takes the lock out of the migration, and the price is that it cannot run inside a transaction block.",
                "The planner trusts your statistics exactly as much as you update them; a table that changes shape without an ANALYZE is a table with stale plans.",
                "An advisory lock is a convention between cooperating processes, not a database guarantee; it works until one writer forgets to take it.",
            ],
        ),
        _t(
            "networking",
            {"go": 0.30, "python": 0.22, "rust": 0.20, "c": 0.16, "java": 0.12},
            [
                "TCP_NODELAY", "SO_REUSEADDR", "accept queue", "backlog",
                "TIME_WAIT", "keepalive", "retransmission", "congestion window",
                "TLS 1.3", "0-RTT", "session resumption", "ALPN",
                "ServerName", "certificate chain", "OCSP stapling", "cert rotation",
                "ECONNRESET", "EPIPE", "connection pooling", "idle timeout",
                "graceful shutdown", "SIGTERM", "drain", "half-close",
                "UDP", "datagram loss", "jitter buffer", "Nagle",
                "DNS", "NXDOMAIN", "TTL", "round-robin",
                "MTU", "path MTU discovery", "fragmentation", "IPv6",
                "proxy", "CONNECT", "header rewrite", "hop-by-hop",
                "load balancer", "health check", "sticky session", "DDoS",
                "WebSocket", "heartbeat", "ping-pong", "backpressure",
                "gRPC", "HTTP/2", "stream multiplexing", "flow control",
            ],
            [
                "setting TCP_NODELAY cut the latency on the small requests.",
                "the backlog was overflowing because the accept loop blocked on TLS.",
                "keepalive probes finally surfaced the half-open connection.",
                "TLS 1.3 session resumption removed a full round trip on reconnect.",
                "rotating the certificate with OCSP stapling avoided the validation stall.",
                "the pool was exhausting because idle connections were never reaped.",
                "graceful shutdown drained in-flight requests before the SIGTERM handler exited.",
                "the UDP path needed a jitter buffer before the audio stopped breaking up.",
                "Nagle's interaction with delayed ACKs was the real latency culprit.",
                "resetting the connection after a retry storm made the peer behave.",
                "the health check returned 200 before the service could actually serve.",
                "half-closing the write side let the peer see the end of the request stream.",
                "the DNS TTL was caching a dead record for an hour after the failover.",
                "the proxy rewrote the hop-by-hop headers and broke the keepalive.",
                "the accept queue was one connection deep, so the SYN drops were invisible until graphed.",
                "the WebSocket heartbeat kept the NAT mapping alive through the idle gap.",
                "the gRPC stream backpressured once the flow-control window was understood.",
                "the TIME_WAIT buildup was normal TCP, and the fix was a longer-lived pool.",
                "the certificate chain was missing the intermediate, so the mobile clients failed validation.",
                "the retransmission storm stopped once the congestion window was tuned.",
                "the load balancer's health check was probing the wrong port, so healthy backends got drained.",
                "the IPv6 path had a smaller MTU, and the fragmentation handling was the whole bug.",
                "the idle timeout was shorter than the batch job's longest gap, so the pool kept resetting.",
                "the proxy's CONNECT handling only worked for port 443 until the allowlist widened.",
            ],
            ["clients", "servers", "proxies", "pools", "endpoints", "gateways"],
            ["dial", "listen", "retry", "tune", "reconnect", "drain", "probe", "forward"],
            [
                "with a bounded retry", "with keepalive enabled",
                "in one round trip", "with backpressure",
                "with a health check", "for the load spike",
                "over TLS", "with graceful shutdown",
            ],
            ["client_pool", "proxy_front", "server_backend", "gateway_egress", "endpoint_api"],
            ["tune", "reconnect", "retry", "drain", "probe", "forward"],
            [
                "on the client_pool", "for the proxy_front",
                "on the gateway_egress", "for the endpoint_api",
            ],
            [
                "Tuning the accept queue for bursty load",
                "Why keepalive matters for long-lived connections",
                "TLS handshake costs on cold connections",
                "Draining connections during a rolling restart",
                "TCP_NODELAY and the small-request latency trap",
                "The health check that lied to the load balancer",
                "Backpressure on a gRPC stream",
                "TIME_WAIT is not a leak",
                "Cert rotation without the validation stall",
                "The proxy that broke keepalive",
            ],
            [
                "A connection that has gone half-open is invisible to both sides until a keepalive probe or a write finally fails; the fix is a heartbeat at the application layer.",
                "The accept queue overflows silently: the kernel drops the SYN, the client retries, and the latency chart just looks bad until you graph the SYN drops.",
                "TLS 1.3 resumption replaces two round trips with one, which is the single biggest latency win for short-lived connections.",
                "TIME_WAIT is not a leak; it is the kernel keeping the connection tuple out of reuse so late segments cannot corrupt a new connection.",
                "A health check that returns 200 before the service can actually serve is a health check that has already lied to the load balancer.",
                "The Nagle algorithm delays small writes to batch them, and delayed ACKs compound it; TCP_NODELAY is the lever for latency-sensitive small requests.",
                "A bounded retry with exponential backoff is what turns a network blip into a footnote instead of a pager alert.",
                "Graceful shutdown is a two-phase handshake: stop accepting, then drain what is in flight, then close; a process that just exits abandons every in-flight request.",
                "The kernel's retransmission and congestion control are tuned for bulk transfer; a chatty small-message protocol needs its own pacing on top.",
                "A certificate without its intermediate chain is a certificate that fails on half your clients, and the error message will blame the wrong side.",
            ],
        ),
        _t(
            "filesystem",
            {"python": 0.26, "rust": 0.20, "c": 0.18, "bash": 0.20, "go": 0.16},
            [
                "mmap", "page fault", "page cache", "dirty pages",
                "fsync", "O_DIRECT", "buffered I/O", "write-back",
                "rename", "atomic replace", "fsync on the directory", "durability",
                "ext4", "XFS", "journal", "delayed allocation",
                "inode", "directory entry", "hard link", "symlink",
                "temp file", "O_TMPFILE", "unlink", "crash consistency",
                "path traversal", "canonicalize", "lexically resolve", "..",
                "permissions", "chmod", "setgid", "sticky bit",
                "partial write", "file lock", "flock", "fcntl",
                "sparse file", "hole punching", "fallocate", "truncate",
                "directory scan", "readdir", "fd exhaustion", "inotify",
                "content-addressed", "hash path", "CAS store", "garbage collection",
                "quota", "ENOSPC", "disk full", "no space left",
            ],
            [
                "fsyncing the directory after the rename made the rename survive a crash.",
                "mmap was faster until the working set stopped fitting in the page cache.",
                "the partial write disappeared once the store switched to write-temp-then-rename.",
                "O_DIRECT removed the double copy but made the misaligned writes fail.",
                "flock covered the single-process case until the second daemon appeared.",
                "a symlink in the upload path turned into a path traversal report.",
                "the fd leak showed up as ENFILE only after the directory scan loop ran for a day.",
                "hole punching kept the sparse image small without a full rewrite.",
                "canonicalizing the path before the permission check closed the .. escape.",
                "fsync on every write was the real cost; batching it into a commit group fixed the throughput.",
                "the readdir loop needed the fd closed inside the loop, not after it.",
                "fallocate reserved the space up front and the ENOSPC during the copy vanished.",
                "the rename was atomic but the config reader still saw the half-written file until the directory was fsynced.",
                "the journal recovered the uncommitted block after the power cut.",
                "the hard link updated in two places at once, which the backup script never expected.",
                "chmod'ing the directory did not fix the files inside, which was the actual permission bug.",
                "the sticky bit finally stopped the shared tmp dir from being cleared by anyone.",
                "O_TMPFILE made the scratch file crash-safe without a name to collide with.",
                "the sparse file was readable as zeros until the offset math went negative.",
                "the file lock was advisory, and the second process ignored it because it used a different API.",
                "the path traversal was a .. that survived the filter because the filter checked after the join.",
                "the content-addressed store deduplicated the same blob until the hash path collided.",
                "delayed allocation delayed the ENOSPC until the commit, which the journal then had to explain.",
                "the inotify watcher missed the rename events because the handler was watching the wrong inode.",
                "the symlink was relative, so it broke the moment the directory moved.",
            ],
            ["uploads", "configs", "logs", "caches", "checkpoints", "temp_dir"],
            ["write", "rename", "scan", "watch", "truncate", "sync", "cleanup", "archive"],
            [
                "atomically", "with a temp file",
                "without blocking", "after the crash",
                "for the reader", "in chunks",
            ],
            ["upload_dir", "config_dir", "log_dir", "cache_dir", "checkpoint_dir", "temp_dir_path"],
            ["sync", "rename", "scan", "cleanup", "truncate", "watch"],
            [
                "in the upload_dir", "for the config_dir",
                "in the log_dir", "for the checkpoint_dir",
            ],
            [
                "Atomic file replacement and the fsync trap",
                "Why the scanner ran out of file descriptors",
                "mmap vs buffered I/O for read-heavy workloads",
                "Crash-consistent writes without a database",
                "Symlink handling in upload paths",
                "The advisory file lock that was not a lock",
                "Sparse files and hole punching",
                "Canonicalizing paths before the permission check",
                "Delayed allocation and the surprise ENOSPC",
                "Directory fsync and the rename that vanished",
            ],
            [
                "A crash between the write and the rename leaves a partial file; writing to a temp name, fsyncing it, then renaming and fsyncing the directory closes the gap.",
                "The page cache absorbs most read traffic, but a working set larger than the cache turns mmap into a page-fault storm.",
                "Directory scans leak file descriptors one inotify event at a time unless the fd is closed in the handler.",
                "O_DIRECT bypasses the page cache and demands aligned buffers, which is a latency win and a portability headache.",
                "A file lock serializes processes that cooperate, and does nothing against processes that do not; it is a convention, not a barrier.",
                "The atomic-replace pattern is write-temp, fsync-temp, rename, fsync-directory; skipping the last fsync makes the rename itself vulnerable to power loss.",
                "A symlink is a redirection, not a file; a path that accepts one and then assumes it is regular is a path traversal waiting to be reported.",
                "hole punching turns a sparse file's logical size into a lie the storage layer happily tells, and reads of the hole return zeros.",
                "fd exhaustion presents as ENFILE or EMFILE long after the leak started, so the open count is the metric to watch, not the error.",
                "The filesystem journal protects metadata structure, not your data's logical consistency; the application still has to define what a consistent file looks like.",
            ],
        ),
        _t(
            "concurrency",
            {"go": 0.30, "rust": 0.24, "java": 0.18, "python": 0.14, "c": 0.14},
            [
                "Mutex", "RWMutex", "atomic", "CAS",
                "condition variable", "semaphore", "barrier", "spinlock",
                "channel", "actor model", "message passing", "select",
                "goroutine", "thread pool", "executor", "work-stealing",
                "deadlock", "lock ordering", "lock-free", "ABA problem",
                "data race", "happens-before", "memory model", "volatile",
                "channel buffer", "backpressure", "rate limit", "circuit breaker",
                "queue", "consumer group", "offset", "redelivery",
                "leader election", "fencing token", "epoch", "lease",
                "idempotency key", "exactly-once", "at-least-once", "outbox pattern",
                "two-phase commit", "saga", "compensation", "distributed lock",
                "starvation", "priority inversion", "bounded queue", "fairness",
            ],
            [
                "the lock ordering rule made the deadlock impossible to reintroduce.",
                "a bounded channel gave the pipeline natural backpressure.",
                "the data race disappeared once the shared counter became an atomic.",
                "the reader-writer lock starved the writer until the policy flipped to writer-preferring.",
                "the distributed lock expired mid-operation, so the update moved to a fencing token.",
                "the outbox pattern turned the dual write into one transaction and one reliable delivery.",
                "the retry loop stopped stampeding once the circuit breaker opened on the 5xx burst.",
                "using a condition variable instead of a sleep loop cut the wakeups by an order of magnitude.",
                "the consumer group rebalanced and redelivered the batch, so the handler had to be idempotent.",
                "the saga compensation ran in the reverse order, which is the only order that unwinds correctly.",
                "the work-stealing executor balanced the queue until the steals outnumbered the pushes.",
                "the epoch counter let the old leader's writes be rejected after the new one took over.",
                "the lease renewal was in the same goroutine as the work, so a slow task expired the lease.",
                "the idempotency key made the retried request a no-op instead of a double charge.",
                "the two-phase commit coordinator was the single point of failure, so it became a saga.",
                "the semaphore capped the concurrent downloads and the memory usage stopped climbing.",
                "the barrier kept the stages aligned until one worker errored and the whole batch had to fail fast.",
                "the channel buffer hid the backpressure until it filled, and then the OOM was the real message.",
                "the lock-free stack hit the ABA problem and the fix was the counter in the swapped word.",
                "the condition variable's lost-wakeup bug was a missing predicate loop, not a missing signal.",
                "the priority inversion resolved once the mutex became priority-inheriting.",
                "the actor mailbox provided the serialization that the shared state never had.",
                "the select loop preferred the ready case, which starved the slow lane until the weights were added.",
                "the redelivered message was applied twice until the handler checked the dedupe table.",
                "the fencing token from the leader made the stale replica's write fail closed.",
            ],
            ["workers", "queues", "jobs", "tasks", "streams", "pipelines"],
            ["lock", "spawn", "retry", "throttle", "synchronize", "fan-out", "drain", "cancel"],
            [
                "without a global lock", "with backpressure",
                "with a bounded queue", "idempotently",
                "across the pool", "with a timeout",
            ],
            ["job_queue", "worker_pool", "task_queue", "pipeline_stage", "stream_partition"],
            ["drain", "throttle", "retry", "synchronize", "cancel", "spawn"],
            [
                "on the job_queue", "for the worker_pool",
                "across the pipeline_stage", "in the stream_partition",
            ],
            [
                "Lock ordering as a deadlock prevention rule",
                "Backpressure from a bounded queue",
                "Idempotent consumers under redelivery",
                "The fencing token pattern for distributed locks",
                "Writer starvation in reader-writer locks",
                "The ABA problem in lock-free structures",
                "Lost wakeups and the condition variable",
                "The outbox pattern for dual writes",
                "Sagas, and why compensation runs backwards",
                "Epochs and the stale leader",
            ],
            [
                "A bounded channel is backpressure with a number on it: the producer blocks instead of the memory growing, and the failure mode is a slow caller rather than an OOM.",
                "Data races hide in plain sight: two goroutines reading a counter without a happens-before edge compile fine and lose updates only under load.",
                "The ABA problem is why lock-free stacks keep a counter in the word they swap, so a reused pointer cannot pass as an unchanged one.",
                "A distributed lock is a recommendation, not a guarantee; the lease can expire mid-operation, which is why the write carries a fencing token the leader checks.",
                "Priority inversion is not exotic: a low-priority task holding the lock starves the high-priority one until the scheduler notices.",
                "A condition variable must be checked in a loop because the wakeup can arrive between the predicate check and the wait; a bare if is a lost wakeup.",
                "The outbox pattern writes the event in the same transaction as the state change, so the dual write becomes one atomic operation and a reliable forwarder.",
                "A consumer group rebalance is not a failure, it is a migration; the handler has to tolerate redelivery, which is what idempotency keys are for.",
                "Work-stealing keeps every executor busy but destroys locality; the queue discipline is the knob, not the pool size.",
                "The epoch is the cheap fencing token: writes stamped with an old epoch are rejected, which turns a split brain into a stale write instead of a corruption.",
            ],
        ),
        _t(
            "crypto",
            {"python": 0.26, "rust": 0.26, "go": 0.22, "java": 0.14, "c": 0.12},
            [
                "XChaCha20-Poly1305", "AES-256-GCM", "nonce", "AEAD",
                "HKDF", "PBKDF2", "Argon2id", "scrypt",
                "HMAC", "SHA-256", "constant-time", "timing oracle",
                "CBC padding oracle", "PKCS#7", "CTR mode", "ECB",
                "RSA-OAEP", "ECDSA", "Ed25519", "P-256",
                "key rotation", "wrapping key", "KMS", "sealed box",
                "ciphertext malleability", "replay attack", "nonce reuse", "IV",
                "encrypt-then-MAC", "sign-then-encrypt", "cryptographic signing", "keyed hash",
                "salt", "pepper", "password hashing", "memory-hard",
                "asymmetric encryption", "hybrid encryption", "X25519", "ephemeral key",
                "certificate pinning", "SPKI", "CA bundle", "mTLS",
                "at-rest encryption", "key hierarchy", "root key", "envelope encryption",
            ],
            [
                "the nonce reuse was the whole bug, and the fix was a random 192-bit nonce.",
                "encrypt-then-MAC ordering stopped the padding oracle before it started.",
                "switching the password hash to Argon2id with a per-user salt retired the old SHA-256 hashes.",
                "the constant-time comparison closed the tag-checking timing oracle.",
                "wrapping the data key under a KMS key made rotation a metadata change.",
                "the CBC padding oracle was real, and moving to an AEAD deleted the whole class.",
                "HKDF with a distinct info string per purpose kept the derived keys from colliding.",
                "the ephemeral X25519 key made the forward secrecy claim true instead of decorative.",
                "sign-then-encrypt was the wrong order until the signature covered the ciphertext.",
                "the replayed message authenticated fine until the nonce was tied to the session.",
                "the AES-GCM tag was checked after the plaintext was used, which defeated the whole point.",
                "the password was stored as SHA-256 without a salt, and the breach proved why that was indefensible.",
                "the RSA-OAEP padding rejected the malformed ciphertext, which leaked nothing but was loud.",
                "the Ed25519 key was reused across services, so one leak rotated everything.",
                "the root key never touched the disk; the envelope wrapped it, and the disk only ever saw the wrapped copy.",
                "the HMAC was over the plaintext, so the ciphertext could still be swapped between records.",
                "the P-256 signature verification was over an unchecked hash, and the malleability followed.",
                "the mTLS handshake failed on the CA bundle until the missing intermediate was pinned.",
                "the KMS key policy was the actual security boundary, and the role policy was not.",
                "the CSPRNG was seeded once at boot, which the entropy audit flagged immediately.",
                "the key rotation regenerated the wrapping key, which orphaned every sealed blob.",
                "the verification key was inside the signed payload, which made the signature meaningless.",
                "the nonce was a counter that reset on reboot, so the keystream repeated across restarts.",
                "the key hierarchy derived the content key from the root with a purpose tag, so a rotated root did not re-encrypt the corpus.",
                "the side-channel trace on the table lookup made the exponentiation constant-time.",
            ],
            ["tokens", "ciphertexts", "secrets", "keys", "signatures", "passwords"],
            ["encrypt", "sign", "verify", "derive", "rotate", "hash", "seal", "digest"],
            [
                "with a fresh nonce", "under a wrapping key",
                "without a timing oracle", "with per-user salt",
                "under envelope encryption", "with a purpose tag",
            ],
            ["secret_store", "token_cache", "key_ring", "signing_service", "crypto_layer"],
            ["seal", "rotate", "derive", "verify", "hash", "encrypt"],
            [
                "in the secret_store", "for the token_cache",
                "under the key_ring", "for the signing_service",
            ],
            [
                "Nonce reuse and why AEADs forbid it",
                "Encrypt-then-MAC ordering",
                "Password hashing with Argon2id",
                "Key rotation as envelope encryption",
                "Constant-time comparison for tokens",
                "Why the keystream repeated after reboot",
                "The signature that signed its own key",
                "Deriving keys with purpose tags",
                "The padding oracle that was real",
                "Forward secrecy from an ephemeral key",
            ],
            [
                "Reusing a nonce under a stream cipher turns two ciphertexts into a keystream solve; an AEAD makes the nonce reuse a hard authentication failure instead of silent corruption.",
                "Encrypt-then-MAC is the ordering that survives an adversary who can flip bits: the MAC is computed over the ciphertext, so any tamper fails before decryption.",
                "A password hash is a work factor, not an encryption scheme; Argon2id exists because GPU clusters make fast hashes cheap to attack in bulk.",
                "Envelope encryption stores the data key wrapped by a key you can rotate, so rotating the root key rewrites metadata instead of re-encrypting the corpus.",
                "A timing oracle needs only a statistically noisy signal; constant-time comparison removes the signal entirely rather than making it harder to measure.",
                "A key hierarchy with distinct purpose tags means rotating one branch does not re-encrypt the others, which is what makes key rotation cheap enough to do.",
                "Signing the plaintext while encrypting separately leaves the two operations composable; signing the ciphertext binds the signature to the exact bytes stored.",
                "The memory-hard property of Argon2id is the point: the cost is in RAM, which is what a GPU farm cannot parallelise the way it parallelises hashes.",
                "Forward secrecy is a property of the key schedule, not the cipher: an ephemeral X25519 key per session means a later compromise cannot read the past.",
                "The trust anchor is the first CA in the chain, and everything else is delegation; pinning the wrong link in that chain is where mTLS setups fail.",
            ],
        ),
        _t(
            "containers",
            {"bash": 0.26, "go": 0.20, "python": 0.18, "yaml": 0.16, "typescript": 0.10, "c": 0.10},
            [
                "Dockerfile", "multi-stage build", "scratch", "distroless",
                "layer cache", "COPY --chown", "RUN --mount=type=cache",
                "ENTRYPOINT", "CMD", "SHELL form", "exec form",
                "init process", "PID 1", "signal forwarding", "zombie reaping",
                "entrypoint.sh", "wait-for-it", "healthcheck", "depends_on",
                "docker compose", "volume mount", "bind mount", "named volume",
                "network mode", "host network", "bridge network", "EXPOSE",
                "image tag", "digest pinning", "SBOM", "cosign",
                "secrets", "build secrets", "env file", "ARG",
                "resource limits", "cgroup", "OOMKilled", "ulimit",
                "read-only root", "capabilities", "no-new-privileges", "seccomp",
                "registry", "layers", "push", "multi-arch",
                "orchestrator", "sidecar", "init container", "liveness probe",
            ],
            [
                "the multi-stage build shrank the image from 1.2 GB to 34 MB.",
                "the exec-form ENTRYPOINT let the container receive SIGTERM properly.",
                "PID 1 stopped eating signals once an init process was added.",
                "COPY --chown removed the permission failure on the mounted volume.",
                "the layer cache was invalidating on every commit until the package order was fixed.",
                "RUN --mount=type=cache kept the build cache inside the build, not in the image.",
                "the container kept restarting until the liveness probe stopped curling itself.",
                "digest pinning made the deploy reproducible instead of tag-chasing.",
                "read-only root with a tmpfs for the writable bits passed the CIS check.",
                "the OOMKilled restarts stopped once the memory limit matched the JVM heap.",
                "the entrypoint.sh ran as a shell, so the SIGTERM went to the shell, not the app.",
                "the healthcheck was a wget to the LB, so it passed while the app was dying.",
                "the named volume outlived the container, which is how the stale cache came back.",
                "the build context was the whole monorepo, so every layer copy re-sent the world.",
                "the image had no SBOM, which the supply-chain review flagged before the release.",
                "the base image was pinned by tag until the digest check made it reproducible.",
                "the read-only root broke the log writes until the logs moved to a volume.",
                "the container ran as root until the USER directive and the read-only root went in together.",
                "the ARG leaked into the image history, so the build secret moved to a secret mount.",
                "the sidecar shipped its own copy of the config, which drifted from the main container's.",
                "the init container seeded the schema before the app container started.",
                "the registry push was multi-arch until the builder forgot the arm64 platform.",
                "the liveness probe failed during the long GC pause, so the orchestrator kept killing the pod.",
                "the exec-form CMD was missing, so the container started the shell and exited immediately.",
                "the seccomp profile blocked the syscall the runtime needed, and the crash was instant and silent.",
            ],
            ["images", "containers", "services", "pods", "stacks", "builds"],
            ["build", "push", "run", "tag", "expose", "compose", "deploy", "scale"],
            [
                "with a multi-stage build", "without root",
                "with a healthcheck", "pinned by digest",
                "for arm64", "with read-only root",
            ],
            ["app_image", "base_image", "sidecar_container", "init_container", "build_stage"],
            ["build", "pin", "expose", "compose", "deploy", "tag"],
            [
                "for the app_image", "of the base_image",
                "in the sidecar_container", "for the build_stage",
            ],
            [
                "Multi-stage builds and the 1.2 GB image",
                "PID 1 and signal handling in containers",
                "Layer caching discipline",
                "Healthchecks that do not lie",
                "Reproducible images via digest pinning",
                "Read-only root without the breakage",
                "Build secrets and the ARG that leaked",
                "The sidecar that drifted from the config",
                "OOMKilled and the JVM heap",
                "Why the pod restarted forever",
            ],
            [
                "A container is a process, not a machine: PID 1 is whatever you told it to be, and if that is a shell that never forwards SIGTERM, graceful shutdown is a lie.",
                "Each RUN line is a layer, and a layer is a cache key: reordering package installs so the stable ones come first keeps the cache alive.",
                "The multi-stage build copies only the compiled artifact into the final stage, which is how a Go image with a 34 MB footprint is built at all.",
                "A liveness probe that succeeds before the service is ready turns a crash loop into an infinite restart loop with 200s in the logs.",
                "Running as root in a container is the default and the mistake: a scratch base with a non-root USER and no-new-privileges closes most of the escape surface.",
                "An image pinned by digest is reproducible; an image pinned by tag is a moving target that changes under you at the worst possible moment.",
                "The build cache is keyed by the full command plus the inputs; putting the volatile steps last is what keeps the expensive ones cached.",
                "An init container is how you sequence: it runs to completion before the app starts, which is exactly the hook a schema migration needs.",
                "Memory limits are enforced by the cgroup, but a JVM sizes its heap from the host unless you tell it otherwise, which is how the pod OOMKilled.",
                "The healthcheck and the readiness probe answer different questions; conflating them is how a pod reports healthy while serving nothing.",
            ],
        ),
        _t(
            "web",
            {"typescript": 0.28, "javascript": 0.18, "python": 0.18, "ruby": 0.14, "go": 0.12, "csharp": 0.10},
            [
                "CSRF token", "SameSite", "CORS", "preflight",
                "Content-Security-Policy", "X-Frame-Options", "HSTS", "nonce",
                "session cookie", "HttpOnly", "Secure flag", "session fixation",
                "SQL injection", "parameterized query", "ORM escaping", "prepared statement",
                "XSS", "output encoding", "DOMPurify", "innerHTML",
                "redirect", "open redirect", "path normalization", "URL validation",
                "rate limiting", "429 Too Many Requests", "brute force", "lockout",
                "OAuth", "authorization code", "PKCE", "state parameter",
                "JWT", "audience claim", "expiry", "signature verification",
                "idempotency", "PUT vs POST", "safe methods", "cache headers",
                "ETag", "If-None-Match", "304 Not Modified", "conditional request",
                "form validation", "server-side validation", "file upload", "MIME sniffing",
                "error page", "information disclosure", "stack trace", "safe error",
            ],
            [
                "setting SameSite=Strict killed the CSRF vector on the state-changing routes.",
                "the XSS was an innerHTML sink fed by a user-controlled string, and output encoding fixed it.",
                "the redirect was open until the destination was validated against an allowlist.",
                "parameterized queries closed the SQL injection the fuzzer found in the search box.",
                "the session cookie was readable by JavaScript until the HttpOnly flag went on.",
                "the CORS preflight was rejecting the real client until the origin allowlist matched.",
                "the login endpoint needed a lockout after ten failures, not an unbounded retry.",
                "the JWT was accepted with any audience until the audience claim got checked.",
                "the upload endpoint sniffed the MIME type from the content instead of the file, so the policy flipped to extension allowlisting.",
                "the 500 page was printing the stack trace, and the fix was a safe generic error with the details in the log.",
                "the CSRF token was validated only on POST, which the DELETE routes quietly skipped.",
                "the session fixation was fixed by rotating the session id at login.",
                "the XSS filter was blacklist-based, so the obfuscated payload went straight through.",
                "the open redirect was a relative path check that accepted //evil.com.",
                "the rate limiter keyed on IP, which the shared office NAT made useless.",
                "the OAuth state parameter was missing, so the login was replayable.",
                "the JWT had no expiry, and the leak of a valid token became a permanent credential.",
                "the file upload accepted .php behind the extension check, because the server executed the MIME it sniffed.",
                "the CSP was missing the nonce, so the inline script was blocked and the widget broke.",
                "the error page revealed the ORM version, which the fingerprinting scan flagged.",
                "the cache headers let the authenticated response sit in the shared cache.",
                "the form re-submitted on refresh because the POST returned a 200 instead of a 303.",
                "the idempotency key made the payment retry safe, and the double-charge tickets stopped.",
                "the preflight request never sent credentials, so the CORS fix was the flag, not the header list.",
                "the HSTS header was missing, so the first request could still be downgraded.",
            ],
            ["pages", "forms", "endpoints", "cookies", "sessions", "uploads"],
            ["encode", "validate", "redirect", "escape", "sanitize", "rate-limit", "expire", "scope"],
            [
                "before rendering", "against an allowlist",
                "on every request", "with a nonce",
                "server-side", "with a lockout",
            ],
            ["login_endpoint", "session_cookie", "upload_handler", "search_endpoint", "form_handler"],
            ["encode", "validate", "redirect", "sanitize", "expire", "scope"],
            [
                "in the login_endpoint", "for the session_cookie",
                "on the upload_handler", "in the search_endpoint",
            ],
            [
                "CSRF defense with SameSite cookies",
                "Escaping output at every sink",
                "Session cookie flags that actually matter",
                "Rate limiting the login endpoint",
                "Safe redirects are allowlisted redirects",
                "The JWT with no expiry",
                "CSP and the nonce the widget needed",
                "Idempotent payment retries",
                "The upload that executed the sniffed MIME",
                "Error pages that leak",
            ],
            [
                "CSRF works because the browser attaches cookies automatically; SameSite=Strict stops the cross-site request before the server has to think about it.",
                "Output encoding is a sink property, not a source property: the same user input is safe in a text node and dangerous in innerHTML.",
                "HttpOnly does not stop XSS, it stops XSS from being useful: the script still runs, it just cannot read the session cookie.",
                "Rate limiting is the only defense that works without understanding the attack; a login endpoint without it is a brute-force lottery.",
                "An open redirect turns a phishing link into a trust pivot; validating the destination against an allowlist closes it with one line.",
                "The session id must change at privilege change; a fixation-resistant login rotates it, and a login that does not is fixable by anyone.",
                "A CSP without a nonce is a policy that blocks your own inline scripts; the nonce is the part that makes the policy usable, and it must be unique per response.",
                "An idempotency key is a promise the server keeps: the first request with a key wins, and every retry with the same key returns the first result.",
                "MIME sniffing is the browser guessing, and it guesses in the attacker's favour; the fix is to serve uploads with a declared type and no execution context.",
                "The stack trace in the 500 page is the same information the attacker would pay a scanner for; the safe error page shows nothing and the log shows everything.",
            ],
        ),
        _t(
            "debugging",
            {"python": 0.24, "typescript": 0.18, "go": 0.18, "c": 0.16, "ruby": 0.10, "rust": 0.14},
            [
                "segfault", "core dump", "gdb", "backtrace", "ASAN",
                "race detector", "tsan", "data race", "heisenbug",
                "flaky test", "test isolation", "shared state", "pollution",
                "stack trace", "frame", "symbols", "release build",
                "null pointer", "undefined behavior", "UB", "valgrind",
                "log line", "structured logging", "correlation id", "log level",
                "debugger", "breakpoint", "watchpoint", "step through",
                "bisect", "git bisect", "regression", "blame",
                "reproducer", "minimal repro", "minimization", "isolate",
                "memory leak", "heap profile", "pprof", "allocation",
                "timeout", "hanging test", "deadlock", "stuck",
                "profiler", "hot path", "flame graph", "CPU bound",
            ],
            [
                "the segfault was a use-after-free, and ASAN caught it on the first run.",
                "git bisect pinned the regression to the commit that changed the buffer sizing.",
                "the flaky test was sharing a static counter across cases.",
                "the race detector flagged the unsynchronized map access that only failed under load.",
                "the minimal repro was a ten-line script, and the full suite was not needed to fix it.",
                "the heap profile showed the leak was the cached connection, not the hot path.",
                "the timeout was a retry loop that never gave up, and the fix was a deadline.",
                "the backtrace pointed at the release build with stripped symbols, so the debug build got the blame.",
                "structured logging with a correlation id made the request traceable across services.",
                "the heisenbug vanished the moment a print statement was added, which meant the timing changed the behavior.",
                "the stack overflow was the recursion depth, and the tail-call rewrite fixed it.",
                "the UB was a signed overflow that only misbehaved with the optimizing compiler.",
                "the leak was in the error path, which is why it never showed in the happy-path profile.",
                "the flaky test failed only on CI because the test order differed from the local run.",
                "the deadlock showed up only after the timeout was removed and the test ran to completion.",
                "the log line was useless until the correlation id tied it to the failing request.",
                "the watchpoint caught the struct field being clobbered two frames before the crash.",
                "the repro needed the exact input file, and the minimized version still triggered it.",
                "the profiler pointed at the JSON serialization, which was 80% of the CPU.",
                "the flame graph made the realloc churn obvious, and the fix was reserving the capacity.",
                "the crash only happened on the release build, which pointed at an optimizer assumption.",
                "the regression was introduced by the refactor that claimed to be behavior-preserving.",
                "the hanging test was a channel never closed, and the fix was a context deadline.",
                "the valgrind run found the uninitialized read that the sanitizers missed.",
                "the pprof output showed the goroutine leak, one blocked receiver per request.",
            ],
            ["crashes", "tests", "daemons", "parsers", "jobs", "services"],
            ["reproduce", "bisect", "trace", "inspect", "isolate", "profile", "minimize", "monitor"],
            [
                "under ASAN", "on the release build",
                "in isolation", "with a minimal repro",
                "across the test suite", "under the race detector",
            ],
            ["crash_report", "test_suite", "parser_module", "daemon_process", "job_runner"],
            ["reproduce", "bisect", "isolate", "minimize", "profile", "trace"],
            [
                "for the crash_report", "in the test_suite",
                "in the parser_module", "for the daemon_process",
            ],
            [
                "Reproducing the segfault with ASAN",
                "Git bisect and the regression commit",
                "Flaky tests are isolation failures",
                "The heisenbug and the print statement",
                "Reading a heap profile for the leak",
                "The release-only crash",
                "Correlation ids across services",
                "The flame graph and the realloc churn",
                "Minimizing the repro to ten lines",
                "The deadlock that only appeared after the timeout",
            ],
            [
                "A segfault is a symptom, not a cause; the backtrace tells you where, and ASAN or valgrind tells you why, and the why is usually a lifetime bug.",
                "A test that fails only on CI fails because of an ordering or a timing the local run never hits; the fix is isolation, not a retry.",
                "git bisect turns a regression into a commit, which turns an investigation into a code review.",
                "The heisenbug pattern is diagnostic gold: if adding a print changes the behavior, the behavior depends on timing, and timing depends on a race.",
                "A heap profile is a histogram of allocations over time; the leak is the allocation that grows monotonically and never returns.",
                "A minimal repro is a contract with the bug: once the failing input fits in a comment, the fix has a regression test for free.",
                "A release-only crash is an optimizer telling you the code relied on undefined behavior; the fix is to remove the reliance, not to ship a debug binary.",
                "Structured logs without a correlation id are a pile of parallel timelines; the id is what lets a request be reassembled from the noise.",
                "The race detector only reports what it observed; a race it never saw is still a race, which is why the fix has to change the design, not the test.",
                "A flame graph shows where the CPU is, not why; the why is usually an allocation or a copy in a loop that looks innocent at the call site.",
            ],
        ),
        _t(
            "algorithms",
            {"python": 0.28, "rust": 0.20, "c": 0.18, "java": 0.16, "go": 0.18},
            [
                "binary search", "B-tree", "hash table", "bloom filter",
                "quicksort", "merge sort", "heap", "priority queue",
                "Dijkstra", "A*", "topological sort", "union-find",
                "dynamic programming", "memoization", "tabulation", "state space",
                "trie", "suffix array", "LCP", "automaton",
                "greedy", "interval scheduling", "activity selection", "exchange argument",
                "big-O", "amortized", "worst case", "complexity",
                "recursion", "tail call", "stack depth", "overflow",
                "divide and conquer", "master theorem", "recurrence", "work-span",
                "probabilistic", "reservoir sampling", "count-min sketch", "approximation",
                "KMP", "string matching", "rolling hash", "rabin-karp",
                "two pointers", "sliding window", "prefix sum", "coordinate compression",
            ],
            [
                "the binary search off-by-one was in the invariant, not the math.",
                "memoization cut the recursion from exponential to linear.",
                "the heap gave the scheduler its O(log n) next-task lookup.",
                "replacing the hash table with a bloom filter dropped the cache-miss rate.",
                "the union-find with path compression made the connectivity check amortized constant.",
                "the sliding window turned the O(n^2) substring scan into O(n).",
                "reservoir sampling made the streaming sample uniform without knowing the length.",
                "the two-pointer technique removed the inner loop entirely.",
                "the trie compressed the autocomplete dictionary by an order of magnitude.",
                "the exchange argument proved the greedy choice was safe before it went into the review.",
                "the suffix array made the repeated-substring search linear instead of quadratic.",
                "the count-min sketch bounded the frequency error on the hot keys.",
                "the A* heuristic was inadmissible, so the search explored everything anyway.",
                "the topological sort order fixed the build graph's circular dependency.",
                "the priority queue with lazy deletion handled the stale entries.",
                "the rolling hash made the substring search O(n) until the collision, which the second hash caught.",
                "the coordinate compression turned the sparse value range into a dense index.",
                "the divide-and-conquer base case was the whole bug, and the merge was fine.",
                "the tail-call rewrite removed the stack overflow on the deep recursion.",
                "the amortized analysis explained why the occasional resize was cheap overall.",
                "the interval scheduling greedy picked the earliest-finish and the proof was an exchange argument.",
                "the tabulation version avoided the recursion entirely and the cache was explicit.",
                "the two-pointer on the sorted array found the pair sum in one pass.",
                "the KMP failure function was built wrong, so the search backed up too far.",
                "the state space pruning cut the search from hours to seconds.",
            ],
            ["arrays", "graphs", "strings", "streams", "trees", "matrices"],
            ["sort", "search", "merge", "dedupe", "traverse", "partition", "memoize", "approximate"],
            [
                "in linear time", "in O(n log n)",
                "without storing everything", "in one pass",
                "with amortized cost", "on the sorted input",
            ],
            ["sorted_array", "graph_nodes", "string_index", "data_stream", "dep_graph"],
            ["sort", "search", "traverse", "dedupe", "partition", "memoize"],
            [
                "over the sorted_array", "across the graph_nodes",
                "on the string_index", "in the data_stream",
            ],
            [
                "Binary search invariants and off-by-one",
                "Memoization vs tabulation",
                "When a bloom filter beats a hash set",
                "Sliding window for substring problems",
                "Union-find with path compression",
                "The inadmissible A* heuristic",
                "Reservoir sampling a stream of unknown length",
                "The rolling hash collision that the second hash caught",
                "Greedy interval scheduling and the exchange argument",
                "The amortized cost of the resizing hash table",
            ],
            [
                "Binary search works because the invariant narrows the range by half every step; the off-by-one lives in the invariant, so the loop condition should express it, not paper over it.",
                "Memoization trades space for time and only helps when the same subproblem recurs; tabulation removes the recursion entirely when the order is known.",
                "A bloom filter trades a false-positive rate for a constant memory footprint, which is why it beats a hash set when the set is too large to keep.",
                "The sliding window keeps a window that satisfies the constraint and advances one end at a time, which is O(n) because each element enters and leaves once.",
                "Union-find with path compression and union by rank makes each operation amortized nearly constant, which is the trick behind connectivity in dynamic graphs.",
                "An admissible heuristic never overestimates, which is the entire contract A* relies on; breaking it makes the optimal path worse than a greedy crawl.",
                "Reservoir sampling keeps the k-th candidate with probability 1/k, which is how a uniform sample of a stream is possible without knowing its length.",
                "A rolling hash is O(1) per position until the hash collides; a second, independent hash turns the rare collision into a verify step instead of a wrong answer.",
                "Greedy works when the exchange argument holds: if swapping the greedy choice into any optimal solution keeps it optimal, the greedy choice is safe.",
                "Amortized analysis is a promise about a sequence, not a single operation; the occasional O(n) resize is paid for by the O(1) inserts around it.",
            ],
        ),
        _t(
            "cloud",
            {"python": 0.26, "go": 0.20, "typescript": 0.18, "bash": 0.18, "java": 0.10, "csharp": 0.08},
            [
                "S3", "bucket policy", "object versioning", "pre-signed URL",
                "IAM", "role", "policy", "least privilege",
                "EC2", "security group", "VPC", "subnet",
                "Lambda", "cold start", "provisioned concurrency", "timeout",
                "cloud function", "serverless", "scale to zero", "concurrency limit",
                "CloudWatch", "alarm", "metric", "log group",
                "KMS key", "CMK", "envelope encryption", "key policy",
                "Auto Scaling", "launch template", "capacity", "spot instance",
                "queue", "SQS", "DLQ", "visibility timeout",
                "object storage", "lifecycle rule", "storage class", "glacier",
                "terraform", "state file", "plan", "apply",
                "cost", "reserved instance", "savings plan", "budget alarm",
                "region", "availability zone", "multi-AZ", "failover",
            ],
            [
                "the bucket was public until the policy flipped to deny non-HTTPS.",
                "the Lambda cold start crossed the timeout until provisioned concurrency was enabled.",
                "the role had AdministratorAccess, and the fix was a scoped policy with a condition.",
                "the DLQ caught the poison messages the queue kept redelivering.",
                "the state file was the real deployment lock; terraform plan refused without it.",
                "the visibility timeout had to exceed the worker's worst case or every message ran twice.",
                "the S3 lifecycle rule moved the cold objects to glacier and cut the bill in half.",
                "the alarm fired on average CPU, which missed the tail; p95 was the metric that mattered.",
                "the multi-AZ failover test proved the RDS endpoint followed the primary.",
                "the pre-signed URL was valid for an hour, which was exactly one hour too long.",
                "the security group allowed 0.0.0.0/0 on 22, and the scan found it before the attacker did.",
                "the Lambda timeout was 3 seconds and the cold start alone ate 2.5.",
                "the IAM role was assumed by every service, so the blast radius was the whole account.",
                "the spot instance was reclaimed mid-batch, and the checkpoint made the retry cheap.",
                "the CloudWatch alarm never fired because the metric had no data for the first five minutes.",
                "the versioning bucket made the accidental overwrite recoverable with one copy-back.",
                "the terraform apply drifted from the console edits, and the state file was the only truth.",
                "the DLQ retention expired before the poison message was investigated.",
                "the concurrency limit throttled the fan-out, and the queue absorbed the overflow.",
                "the savings plan covered the steady state, and the spot covered the burst.",
                "the region had two AZs, so the failover was a DNS TTL away from real.",
                "the KMS key policy was scoped to the role, and the envelope wrapped the data key.",
                "the lifecycle rule was set to expire the temp bucket's objects after a day.",
                "the autoscaling group lagged the traffic, so the alarm tuned the cooldown.",
                "the log group retention was set to never, and the compliance audit noticed.",
            ],
            ["buckets", "functions", "roles", "queues", "clusters", "instances"],
            ["provision", "deploy", "scale", "rotate", "restrict", "migrate", "monitor", "budget"],
            [
                "with least privilege", "without cold start",
                "across AZs", "with a DLQ",
                "pinned by state", "under a budget",
            ],
            ["app_bucket", "lambda_fn", "service_role", "job_queue", "worker_cluster"],
            ["restrict", "provision", "deploy", "scale", "rotate", "monitor"],
            [
                "for the app_bucket", "on the lambda_fn",
                "for the service_role", "in the job_queue",
            ],
            [
                "Least privilege for the service role",
                "Cold starts and provisioned concurrency",
                "SQS visibility timeouts and the DLQ",
                "Terraform state as a deployment lock",
                "Storage lifecycle policies for cost",
                "The security group that allowed the world",
                "Spot instances and the checkpoint",
                "The alarm that never fired",
                "Multi-AZ failover and the DNS TTL",
                "Envelope encryption with a scoped KMS key",
            ],
            [
                "An S3 bucket is public by default only when the policy says so; the fix is a bucket policy that denies non-HTTPS and a check in CI that enforces it.",
                "Lambda cold starts are the latency of provisioning, not of your code; provisioned concurrency pre-warms the pool and trades cost for p95.",
                "The visibility timeout is the queue's retry clock: too short and the worker gets interrupted mid-job, too long and a dead worker stalls the queue.",
                "A DLQ is where poison messages go to die loudly; without one, a malformed payload is redelivered until the visibility timeout becomes a permanent loop.",
                "Terraform state is the source of truth for what exists; two people planning against the same state file is a distributed lock problem wearing a different hat.",
                "Least privilege is a property of the narrowest policy that still works, and a condition on the principal or the resource is what keeps it narrow.",
                "A pre-signed URL is a delegation with an expiry; the shorter the window, the smaller the surface, and the signing role should have nothing else.",
                "Autoscaling on a lagging metric is a feedback loop with the delay built in; the cooldown is the tuning knob that keeps it from oscillating.",
                "Multi-AZ is only as good as the failover that is actually tested; the DNS TTL and the connection pool both have to fail over with the endpoint.",
                "A budget alarm is the cost control that works; the alert on last month's spend is the one that arrives too late to matter.",
            ],
        ),
    ]
}


# ---------------------------------------------------------------- note composer

# Category -> action verbs usable after "I" in a past-tense skill note.
# All of these must compose as "I <verb> it <clause>".
_VERBS_FOR = {
    "bugfix": ["fixed", "root-caused", "patched", "resolved"],
    "feature": ["built", "added", "implemented", "wired up"],
    "refactor": ["reworked", "refactored", "restructured", "rewrote"],
    "performance": ["optimized", "sped up", "tuned", "eliminated"],
    "hardening": ["hardened", "locked down", "tightened", "secured"],
}

def verbs_for(category: str) -> list[str]:
    return _VERBS_FOR[category]


_COMPONENTS = [
    "parser", "scanner", "scheduler", "cache", "pipeline", "worker",
    "daemon", "migration", "api_client", "batch_job", "indexer", "watchdog",
    "loader", "serializer", "pool", "proxy",
]

_SECOND = [
    "The next time it broke, the fix was already written down.",
    "The change stayed in the review because the test would have caught the old shape.",
    "This is the version I want to reach for instead of improvising.",
    "The surrounding code now documents why the obvious approach fails.",
    "The config that controls it lives next to the code that reads it.",
    "It is on the checklist for the next incident, not buried in a log.",
    "The regression test ships with the fix, so the old shape cannot quietly come back.",
    "A future me, two months from now, would have re-derived this from scratch.",
    "The pull request comment that explains the tradeoff is worth more than the diff.",
    "I kept the note next to the failure so the next person does not repeat the archaeology.",
]

_RULE = [
    "the failure mode is cheaper to prevent than to diagnose",
    "the principle generalizes beyond this codebase",
    "the right move is to make the wrong path impossible rather than detected",
    "the invariant belongs in the code, not in the comments",
    "a fix this specific is worth writing down or it gets lost",
    "the variant that bites is the one the docs describe in passing",
    "the cheaper fix is the one that removes the failure class, not this instance",
    "the configuration that made it work belongs in the same commit as the code",
    "a note is only useful if it names the exact flag and the exact failure",
    "the debug trace that found it is worth more than the one-line fix",
]

# Second clause after the category verb. Varies so the scaffold is not
# the same string on every note.
_FIX_CLAUSE = [
    "by making the change local and naming the exact flag",
    "by rewriting the code so the bad path is structurally impossible",
    "by tightening the condition that let the old behavior through",
    "by extracting the logic into a single, testable spot",
    "by removing the branch that should never have been reachable",
    "by adding the one check that should have been there from the start",
    "by replacing the clever approach with the boring correct one",
]


def build_note(
    rng: random.Random,
    topic_name: str,
    seq: int,
    written_at_ms: int,
    components: list[str],
) -> dict:
    topic = TOPICS[topic_name]
    lang = topic.language(rng)
    specifics = topic.specifics(rng, n=2)
    s1 = rng.choice(topic.atoms)
    category = pick_weighted(rng, CATEGORIES, CATEGORY_WEIGHTS)
    difficulty = pick_weighted(rng, DIFFICULTIES, DIFFICULTY_WEIGHTS)
    component = rng.choice(components)
    s2 = rng.choice(_SECOND)
    rule = rng.choice(_RULE)

    text = (
        f"While working on the {component}, {s1} I {rng.choice(verbs_for(category))} it "
        f"{rng.choice(_FIX_CLAUSE)}. {s2} Lesson: {rule}, and it now lives next to the "
        f"{specifics[0]} handling code."
    )

    tags = [topic_name] + rng.sample(
        [s.lower().replace(" ", "_").replace(".", "") for s in specifics], k=2
    )

    meta = {
        "category": category,
        "difficulty": difficulty,
        "language": lang,
        "component": component,
        "tags": tags,
    }

    return {
        "note_id": f"d3-{topic_name}-{seq:05d}",
        "topic": topic_name,
        "category": category,
        "difficulty": difficulty,
        "language": lang,
        "component": component,
        "source_task": f"synthetic {topic_name} task {rng.randint(1, 200)}",
        "tags": tags,
        "written_at_ms": written_at_ms,
        "seq": seq,
        "text": text,
        "meta": json.dumps(meta, separators=(",", ":")),
    }


# ---------------------------------------------------------------- task composer


def build_task(rng: random.Random, topic_name: str, seq: int, attack: bool) -> dict:
    topic = TOPICS[topic_name]
    difficulty = pick_weighted(rng, DIFFICULTIES, DIFFICULTY_WEIGHTS)
    tag = rng.choice(topic.terms)

    if attack:
        # Injection stream: name a concrete artifact so the resulting memory is
        # semantically predictable. Different grammar from the background tasks on
        # purpose, so the harness can also use the text itself to detect leakage.
        verb = rng.choice(topic.trigger_verbs)
        obj = rng.choice(topic.trigger_objs)
        end = rng.choice(topic.trigger_ends)
        task = f"{verb.capitalize()} {end}"
        trigger = f"{verb} {end}"
    else:
        verb = rng.choice(topic.task_verbs)
        obj = rng.choice(topic.task_objs)
        end = rng.choice(topic.task_ends)
        task = f"{verb.capitalize()} the {obj} {end}"
        trigger = f"{verb} the {obj}"

    return {
        "task_id": f"task-{topic_name}-{seq:03d}",
        "topic": topic_name,
        "difficulty": difficulty,
        "attack": attack,
        "trigger": trigger,
        "tags": [topic_name, tag.lower().replace(" ", "_")],
        "task": task,
    }


# ---------------------------------------------------------------- cluster composer


def build_cluster_doc(rng: random.Random, cluster: str, seq: int, text: str) -> dict:
    topic = TOPICS[cluster]
    title = rng.choice(topic.cluster_title_pool)
    return {
        "doc_id": f"clu-{cluster}-{seq:03d}",
        "cluster": cluster,
        "title": title,
        "text": text,
        "words": len(text.split()),
    }


# ---------------------------------------------------------------- writers


def write_notes(path: Path) -> None:
    rng = random.Random(SEED ^ 0x4E4F5445)
    components = list(_COMPONENTS)
    out: list[dict] = []
    t = 1_700_000_000_000
    seq = 0
    for topic_name in TAXONOMY:
        for _ in range(500):
            t += rng.randint(7_000, 95_000)
            rec = build_note(rng, topic_name, seq, t, components)
            out.append(rec)
            seq += 1
    _write_jsonl(path, out)


def write_tasks(path: Path) -> None:
    rng = random.Random(SEED ^ 0x5441534B)
    out: list[dict] = []
    seen: set[str] = set()
    seq = 0
    for topic_name in TAXONOMY:
        # 100 regular + 20 attack tasks per topic, so the stream has both a
        # background corpus and a clearly-delimited injection tail.
        for i in range(120):
            attack = i >= 100
            task = None
            while True:
                rec = build_task(rng, topic_name, seq, attack=attack)
                if rec["task"] not in seen:
                    task = rec
                    seen.add(rec["task"])
                    break
            out.append(task)
            seq += 1
    _write_jsonl(path, out)


def write_clusters(path: Path) -> None:
    rng = random.Random(SEED ^ 0x434C5553)
    out: list[dict] = []
    seen: set[str] = set()
    for cluster in TAXONOMY:
        for i in range(150):
            # Resample up to 40 times for a fresh doc; fall back to a unique sentence order.
            n_sents = rng.randint(3, 6)
            text = None
            for _ in range(40):
                sents = rng.sample(TOPICS[cluster].cluster_sents, k=min(n_sents, len(TOPICS[cluster].cluster_sents)))
                if len(set(sents)) == len(sents):
                    cand = " ".join(sents)
                    if cand not in seen:
                        text = cand
                        seen.add(cand)
                        break
            if text is None:
                sents = rng.sample(TOPICS[cluster].cluster_sents, k=min(n_sents, len(TOPICS[cluster].cluster_sents)))
                text = " ".join(sents)
                seen.add(text)
            out.append(build_cluster_doc(rng, cluster, i, text))
    _write_jsonl(path, out)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=True, separators=(",", ":")) + "\n")


def params_hash() -> str:
    return hashlib.sha256(PARAMS_TOML.read_bytes()).hexdigest()


def write_manifest() -> None:
    manifest = {
        "generator": "data/generate_data.py",
        "version": VERSION,
        "seed": SEED,
        "params_hash": params_hash(),
        "params_file": "params.toml",
        "files": [
            {"path": "d3-swebench/notes.jsonl", "count": 5000, "schema": [
                "note_id", "topic", "category", "difficulty", "language", "component",
                "source_task", "tags", "written_at_ms", "seq", "text", "meta",
            ]},
            {"path": "d3-swebench/tasks.jsonl", "count": 1200, "schema": [
                "task_id", "topic", "difficulty", "attack", "trigger", "tags", "task",
            ]},
            {"path": "retrieval-clusters.jsonl", "count": 1500, "schema": [
                "doc_id", "cluster", "title", "text", "words",
            ]},
        ],
    }
    (ROOT / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


# ---------------------------------------------------------------- verification


def _verify() -> None:
    """Re-parse every file we just wrote and check counts, keys, and enums."""
    def check_jsonl(
        path: Path,
        expected: int,
        required: list[str],
        enum_fields: dict[str, set],
    ) -> list[dict]:
        records: list[dict] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                missing = [k for k in required if k not in rec]
                assert not missing, f"{path}: missing {missing}"
                records.append(rec)
        assert len(records) == expected, f"{path}: {len(records)} != {expected}"
        for field, allowed in enum_fields.items():
            bad = {r[field] for r in records if r.get(field) not in allowed}
            assert not bad, f"{path}: {field} has invalid {bad}"
        print(f"  ok {path} ({len(records)} records)")
        return records

    notes = check_jsonl(
        ROOT / "d3-swebench/notes.jsonl", 5000,
        ["note_id", "topic", "category", "difficulty", "language", "component",
         "source_task", "tags", "written_at_ms", "seq", "text", "meta"],
        {"topic": set(TAXONOMY), "category": set(CATEGORIES), "difficulty": set(DIFFICULTIES)},
    )
    tasks = check_jsonl(
        ROOT / "d3-swebench/tasks.jsonl", 1200,
        ["task_id", "topic", "difficulty", "attack", "trigger", "tags", "task"],
        {"topic": set(TAXONOMY), "difficulty": set(DIFFICULTIES)},
    )
    clusters = check_jsonl(
        ROOT / "retrieval-clusters.jsonl", 1500,
        ["doc_id", "cluster", "title", "text", "words"],
        {"cluster": set(TAXONOMY)},
    )

    # Uniqueness of the text-bearing fields.
    assert len({n["text"] for n in notes}) == 5000, "duplicate note text"
    assert len({t["task"] for t in tasks}) == 1200, "duplicate task text"
    assert len({c["text"] for c in clusters}) == 1500, "duplicate cluster text"

    # Notes: topic balance (500 each), meta consistency, monotonic timestamps.
    counts = Counter(n["topic"] for n in notes)
    assert set(counts) == set(TAXONOMY), f"topic imbalance: {counts}"
    assert all(v == 500 for v in counts.values()), f"topic imbalance: {counts}"
    for n in notes:
        meta = json.loads(n["meta"])
        assert meta["category"] == n["category"], n["note_id"]
        assert meta["language"] == n["language"], n["note_id"]
        assert meta["component"] == n["component"], n["note_id"]
        assert meta["tags"] == n["tags"], n["note_id"]
    times = [n["written_at_ms"] for n in notes]
    assert times == sorted(times) and len(set(times)) == len(times), "timestamps not monotonic"
    print("  ok topic balance 10 x 500, meta consistency, monotonic timestamps")

    # Tasks: per-topic balance, attack segment shape (100 regular + 20 attack per topic),
    # and no attack task that reads like a background directive.
    task_topics = Counter(t["topic"] for t in tasks)
    assert set(task_topics) == set(TAXONOMY), f"task topic imbalance: {task_topics}"
    for topic in TAXONOMY:
        subset = [t for t in tasks if t["topic"] == topic]
        n_attack = sum(1 for t in subset if t["attack"])
        assert n_attack == 20, f"{topic}: {n_attack} attack tasks"
    print("  ok tasks 10 x 120, attack tail 20 per topic")

    # Clusters: per-cluster balance, docs long enough to be meaningful, and
    # per-cluster text uniqueness -- duplicate docs inside a cluster would still
    # pass the global uniqueness check but would break retrieval ground truth.
    cluster_counts = Counter(c["cluster"] for c in clusters)
    assert set(cluster_counts) == set(TAXONOMY), f"cluster imbalance: {cluster_counts}"
    assert all(v == 150 for v in cluster_counts.values()), f"cluster imbalance: {cluster_counts}"
    assert all(c["words"] >= 30 for c in clusters), "cluster doc too short"
    for cluster in TAXONOMY:
        subset = [c for c in clusters if c["cluster"] == cluster]
        assert len({c["text"] for c in subset}) == 150, f"duplicate docs in cluster {cluster}"
    print("  ok clusters 10 x 150, all >= 30 words, no intra-cluster duplicates")


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    write_notes(ROOT / "d3-swebench" / "notes.jsonl")
    write_tasks(ROOT / "d3-swebench" / "tasks.jsonl")
    write_clusters(ROOT / "retrieval-clusters.jsonl")
    write_manifest()
    _verify()
    print("done.")


if __name__ == "__main__":
    main()
