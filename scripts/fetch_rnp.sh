#!/bin/bash
# Fetch Runs N' Poses release files from Zenodo (DOI 10.5281/zenodo.14794785) into data/raw/runs_n_poses.
# Resumable: uses curl -C - so partial files continue. Order: small metadata first, ground truth last.
set -u
OUT="$(dirname "$0")/../data/raw/runs_n_poses"; mkdir -p "$OUT"
for f in annotations.csv inputs.json predictions.tar.gz posebusters_results.tar.gz all_similarity_scores.parquet ground_truth.tar.gz; do
  echo "== $f $(date)"
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    curl -fsSL -C - --retry 3 --retry-delay 30 --connect-timeout 60 -m 7200 \
      -o "$OUT/$f" -w "http=%{http_code} bytes=%{size_download}\n" \
      "https://zenodo.org/records/14794785/files/$f?download=1" && break
    echo "attempt $attempt failed, sleeping"; sleep 300
  done
done
echo "== done $(date)"
