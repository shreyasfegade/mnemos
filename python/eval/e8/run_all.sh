#!/usr/bin/env bash
# Orchestrate E8 inside WSL: mount a gocryptfs volume, run each store's transfer measurement
# against the ciphertext, unmount. Run from the repo's python/eval/e8 directory in WSL.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
GC=~/e8bin/gocryptfs
WORK=~/e8work
DATA="$HERE/data"
OUT="$HERE/results"
mkdir -p "$OUT"
source ~/e8venv/bin/activate

for store in faiss chroma lancedb sqlitevec; do
  echo "=== $store ==="
  rm -rf "$WORK"; mkdir -p "$WORK/cipher" "$WORK/plain"
  echo -n "e8-passphrase" > "$WORK/pw"
  "$GC" -init -passfile "$WORK/pw" -q "$WORK/cipher"
  "$GC" -passfile "$WORK/pw" -q "$WORK/cipher" "$WORK/plain"
  set +e
  python3 "$HERE/run_e8.py" \
    --emb "$DATA/emb.npy" --labels "$DATA/labels.npy" \
    --store "$store" --cipher "$WORK/plain/store" --ciphertext "$WORK/cipher" \
    --out "$OUT/${store}.json" --budget 40
  rc=$?
  set -e
  fusermount -u "$WORK/plain" || true
  [ $rc -ne 0 ] && echo "$store FAILED (rc=$rc)"
done
rm -rf "$WORK"
echo "=== E8 done ==="
for f in "$OUT"/*.json; do echo "--- $f"; cat "$f"; echo; done
