#!/usr/bin/env bash
# Parallel runner for the gocryptfs store experiments (E16 extension, E32, E6 extension).
#
#   wsl -d Ubuntu -- bash /mnt/c/.../python/eval/e8/run_par_wsl.sh e16|e32|e6
#
# Each job gets its own gocryptfs volume and its own Postgres data directory under $WORK, so jobs
# share nothing and run concurrently. A job whose output already exists is skipped, so an
# interrupted run resumes where it stopped. JOBS (default 4) bounds concurrency: the work is
# per-write snapshotting, which is CPU- and fsync-bound, and WSL here has 32 cores.
set -u

MODE="${1:?usage: run_par_wsl.sh e16|e32|e6}"
REPO=/mnt/c/Users/Admin/Desktop/Projects/Code/03_Apps-Products/cybsersec
E8="$REPO/python/eval/e8"
export GC="$HOME/bin/gocryptfs"
export WORK="$HOME/mnemos-par"
export E8
export PASS="mnemos-par-password"
export PER_JOB_TIMEOUT="${PER_JOB_TIMEOUT:-3600}"
JOBS="${JOBS:-4}"

[ -x "$GC" ] || { echo "gocryptfs missing at $GC"; exit 1; }
mkdir -p "$WORK"

# One job: store seed hist budget checkpoint_every outfile python
one() {
  local store="$1" seed="$2" hist="$3" budget="$4" g="$5" out="$6" py="$7"
  if [ -s "$out" ]; then
    echo "skip $(basename "$out") (exists)"
    return 0
  fi
  local tag
  tag="$(basename "$out" .json)"
  local cipher="$WORK/$tag.cipher" plain="$WORK/$tag.plain"
  fusermount -u "$plain" 2>/dev/null || true
  rm -rf "$cipher" "$plain"
  mkdir -p "$cipher" "$plain"
  if ! "$GC" -init -q -extpass "echo -n $PASS" "$cipher" >/dev/null 2>&1; then
    echo "FAIL $tag: gocryptfs init"; return 1
  fi
  if ! "$GC" -q -extpass "echo -n $PASS" "$cipher" "$plain" >/dev/null 2>&1; then
    echo "FAIL $tag: gocryptfs mount"; rm -rf "$cipher" "$plain"; return 1
  fi
  local t0=$SECONDS
  timeout "$PER_JOB_TIMEOUT" "$py" "$E8/run_e6.py" \
    --emb "$E8/data/emb.npy" --labels "$E8/data/labels.npy" \
    --store "$store" --plain "$plain/store" --cipher "$cipher" \
    --out "$out.tmp" --hist "$hist" --budget "$budget" --seed "$seed" \
    --checkpoint-every "$g" > "$WORK/$tag.log" 2>&1
  local rc=$?
  fusermount -u "$plain" 2>/dev/null || true
  rm -rf "$cipher" "$plain"
  if [ $rc -eq 0 ] && [ -s "$out.tmp" ]; then
    mv "$out.tmp" "$out"
    echo "done $tag in $((SECONDS - t0))s"
  else
    rm -f "$out.tmp"
    echo "FAIL $tag rc=$rc after $((SECONDS - t0))s (log: $WORK/$tag.log)"
    tail -5 "$WORK/$tag.log" | sed 's/^/    /'
    return 1
  fi
}
export -f one

jobs_e16() {
  # The E16 workload unchanged (hist 150, budget 20, g = 1); seeds 0-2 already exist.
  local out="${OUT16:-$E8/results16}"; mkdir -p "$out"
  for s in ${E16_SEEDS:-$(seq 3 19)}; do
    echo "pgvector $s 150 20 1 $out/pgvector.s$s.json $HOME/e8venv/bin/python"
  done
}

jobs_e32() {
  # Capture granularity: a larger induction budget so g = 16 still yields 20 versions.
  local out="${OUT32:-$E8/results32}"; mkdir -p "$out"
  for g in ${E32_G:-1 4 16}; do
    for s in ${E32_SEEDS:-$(seq 0 9)}; do
      echo "pgvector $s 150 40 $g $out/pgvector.g$g.s$s.json $HOME/e8venv/bin/python"
    done
  done
}

jobs_e6() {
  # The E6 workload unchanged; seeds 0-2 already exist. Slowest store first so it is not last.
  local out="${OUT6:-$E8/results6}"; mkdir -p "$out"
  for st in ${E6_STORES:-lancedb_indexed faiss chroma lancedb sqlitevec append_control}; do
    for s in ${E6_SEEDS:-$(seq 3 9)}; do
      if [ "$st" = "lancedb_indexed" ]; then h=100; b=10; else h=300; b=30; fi
      echo "$st $s $h $b 1 $out/$st.s$s.json python3"
    done
  done
}

case "$MODE" in
  e16) LIST="$(jobs_e16)" ;;
  e32) LIST="$(jobs_e32)" ;;
  e6) LIST="$(jobs_e6)" ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac

N=$(printf '%s\n' "$LIST" | grep -c .)
echo "$MODE: $N jobs, $JOBS at a time"
printf '%s\n' "$LIST" | xargs -P "$JOBS" -L 1 bash -c 'one "$@"' _
RC=$?
FAILED=$(printf '%s\n' "$LIST" | awk '{print $6}' | while read -r f; do [ -s "$f" ] || echo "$f"; done | grep -c .)
echo "$MODE DONE: $((N - FAILED))/$N outputs present"
[ "$FAILED" -eq 0 ] || exit 1
exit $RC
