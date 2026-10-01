#!/usr/bin/env bash
# E16: structure transfer against a real, deployed content-placed on-disk store -- pgvector ivfflat
# -- inside a gocryptfs volume. a reader: attack a real system with incremental content-addressed
# persistent list layout, or say plainly the growing layout is hypothetical.
#
# pgvector needs the e8venv python (pgserver + pgvector); it runs on the Linux filesystem, not /mnt/c,
# because Postgres over the 9p mount is too slow for per-insert checkpointing. The ciphertext the
# attack diffs is the gocryptfs cipher directory backing the plaintext mount that holds the DB.
#
#   wsl -d Ubuntu -- bash /mnt/c/.../python/eval/e8/run_e16_wsl.sh
set -u

REPO=/mnt/c/Users/Admin/Desktop/Projects/Code/03_Apps-Products/cybsersec
E8="$REPO/python/eval/e8"
OUTDIR="$E8/results16"
GC="$HOME/bin/gocryptfs"
WORK="$HOME/mnemos-e16"
PASS="mnemos-e16-password"
PY="$HOME/e8venv/bin/python"

mkdir -p "$OUTDIR" "$WORK"
[ -x "$GC" ] || { echo "gocryptfs missing at $GC"; exit 1; }
[ -x "$PY" ] || { echo "e8venv python missing at $PY"; exit 1; }

STORE=pgvector
SEEDS="${SEEDS:-0 1 2}"
H="${HIST:-150}"
BU="${BUDGET:-20}"

for SEED in $SEEDS; do
  echo "=== $STORE seed $SEED ==="
  CIPHER="$WORK/$STORE.s$SEED.cipher"
  PLAIN="$WORK/$STORE.s$SEED.plain"
  fusermount -u "$PLAIN" 2>/dev/null || true
  rm -rf "$CIPHER" "$PLAIN"
  mkdir -p "$CIPHER" "$PLAIN"
  "$GC" -init -q -extpass "echo -n $PASS" "$CIPHER" >/dev/null 2>&1 || { echo "init failed"; continue; }
  "$GC" -q -extpass "echo -n $PASS" "$CIPHER" "$PLAIN" >/dev/null 2>&1 || { echo "mount failed"; continue; }

  timeout "${PER_STORE_TIMEOUT:-3600}" "$PY" "$E8/run_e6.py" \
    --emb "$E8/data/emb.npy" \
    --labels "$E8/data/labels.npy" \
    --store "$STORE" \
    --plain "$PLAIN/store" \
    --cipher "$CIPHER" \
    --out "$OUTDIR/$STORE.s$SEED.json" \
    --hist "$H" \
    --budget "$BU" \
    --seed "$SEED" 2>&1 | tail -30
  [ $? -eq 124 ] && echo "  TIMED OUT"

  fusermount -u "$PLAIN" 2>/dev/null || true
  rm -rf "$CIPHER" "$PLAIN"
done
echo "E16 DONE"
