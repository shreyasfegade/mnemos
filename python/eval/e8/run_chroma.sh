#!/usr/bin/env bash
set -u
REPO=/mnt/c/Users/Admin/Desktop/Projects/Code/03_Apps-Products/cybsersec
E8="$REPO/python/eval/e8"
GC="$HOME/e8bin/gocryptfs"
export PATH="$HOME/.local/bin:$PATH"
export ANONYMIZED_TELEMETRY=False
W="$HOME/e8chroma"
rm -rf "$W"; mkdir -p "$W/cipher" "$W/plain"
echo -n "e8pass" > "$W/pw"
"$GC" -init -passfile "$W/pw" -q "$W/cipher"
"$GC" -passfile "$W/pw" -q "$W/cipher" "$W/plain"
python3 "$E8/run_e8.py" --emb "$E8/data/emb.npy" --labels "$E8/data/labels.npy" \
  --store chroma --cipher "$W/plain/store" --ciphertext "$W/cipher" \
  --out "$E8/results/chroma.json" --budget 40 2>&1 | tail -25
fusermount -u "$W/plain" || true
rm -rf "$W"
