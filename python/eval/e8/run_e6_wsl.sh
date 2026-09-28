#!/usr/bin/env bash
# E6: run the redone deployed-store experiment inside a gocryptfs volume.
#
# Invoked from Windows as:
#   wsl -d Ubuntu -- bash /mnt/c/.../python/eval/e8/run_e6_wsl.sh
#
# The ciphertext directory lives on the Linux filesystem, not under /mnt/c: FUSE over the 9p
# mount is slow enough to distort the per-write snapshotting this experiment does.
set -u

REPO=/mnt/c/Users/Admin/Desktop/Projects/Code/03_Apps-Products/cybsersec
E8="$REPO/python/eval/e8"
OUTDIR="$E8/results6"
GC="$HOME/bin/gocryptfs"
WORK="$HOME/mnemos-e6"
PASS="mnemos-e6-password"

mkdir -p "$OUTDIR" "$WORK"

if [ ! -x "$GC" ]; then
  echo "gocryptfs missing at $GC"
  exit 1
fi

STORES="${STORES:-faiss chroma lancedb lancedb_indexed sqlitevec append_control}"

SEEDS="${SEEDS:-0 1 2}"

for STORE in $STORES; do
 for SEED in $SEEDS; do
  echo "=== $STORE seed $SEED ==="
  CIPHER="$WORK/$STORE.s$SEED.cipher"
  PLAIN="$WORK/$STORE.s$SEED.plain"

  fusermount -u "$PLAIN" 2>/dev/null || true
  rm -rf "$CIPHER" "$PLAIN"
  mkdir -p "$CIPHER" "$PLAIN"

  if ! "$GC" -init -q -extpass "echo -n $PASS" "$CIPHER" >/dev/null 2>&1; then
    echo "  gocryptfs init failed"
    continue
  fi
  if ! "$GC" -q -extpass "echo -n $PASS" "$CIPHER" "$PLAIN" >/dev/null 2>&1; then
    echo "  gocryptfs mount failed"
    continue
  fi

  # lancedb_indexed rebuilds its IVF-PQ index after every insert, which is what makes it
  # content-placed and also makes it O(n) per write. It gets a smaller workload so it finishes;
  # the result records the reduced budget so the comparison is not read as like-for-like.
  H="${HIST:-300}"
  BU="${BUDGET:-30}"
  if [ "$STORE" = "lancedb_indexed" ]; then
    H=100
    BU=10
  fi

  timeout "${PER_STORE_TIMEOUT:-2400}" python3 "$E8/run_e6.py" \
    --emb "$E8/data/emb.npy" \
    --labels "$E8/data/labels.npy" \
    --store "$STORE" \
    --plain "$PLAIN/store" \
    --cipher "$CIPHER" \
    --out "$OUTDIR/$STORE.s$SEED.json" \
    --hist "$H" \
    --budget "$BU" \
    --seed "$SEED" 2>&1 | tail -25
  if [ $? -eq 124 ]; then
    echo "  TIMED OUT after ${PER_STORE_TIMEOUT:-2400}s"
  fi

  fusermount -u "$PLAIN" 2>/dev/null || true
  rm -rf "$CIPHER" "$PLAIN"
 done
done

echo "E6 DONE"
