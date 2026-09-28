#!/usr/bin/env bash
# E8 driver, run inside WSL:  wsl -d Ubuntu -- bash /mnt/c/.../python/eval/e8/run_all_wsl.sh
set -u
REPO=/mnt/c/Users/Admin/Desktop/Projects/Code/03_Apps-Products/cybsersec
E8="$REPO/python/eval/e8"
GC="$HOME/e8bin/gocryptfs"
OUT="$E8/results"
export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$OUT"

for store in faiss sqlitevec chroma lancedb; do
  echo "=== $store ==="
  W="$HOME/e8work_$store"
  rm -rf "$W"; mkdir -p "$W/cipher" "$W/plain"
  echo -n "e8pass" > "$W/pw"
  "$GC" -init -passfile "$W/pw" -q "$W/cipher"
  "$GC" -passfile "$W/pw" -q "$W/cipher" "$W/plain"
  python3 "$E8/run_e8.py" --emb "$E8/data/emb.npy" --labels "$E8/data/labels.npy" \
    --store "$store" --cipher "$W/plain/store" --ciphertext "$W/cipher" \
    --out "$OUT/$store.json" --budget 40 || echo "$store FAILED"
  fusermount -u "$W/plain" || true
  rm -rf "$W"
done
echo "=== E8 DONE ==="
for f in "$OUT"/*.json; do echo "--- $f"; cat "$f"; echo; done
