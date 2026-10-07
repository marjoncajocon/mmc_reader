#!/usr/bin/env mmc
#
# getdata.sh - download training datasets into build/data
# Run it from the mmc shell (D:\mmc-shell\mmc.exe):
#
#   ./getdata.sh emnist    handwritten letters and digits (NIST, ~560 MB)
#   ./getdata.sh iam       handwritten English lines (IAM, ~270 MB,
#                          non-commercial research use only)
#   ./getdata.sh list      show what is downloaded
#
# Datasets are never committed: build/ is ignored by git, and
# ./build.sh clean keeps build/data.
#

set -e

DATA="build/data"
EMNIST_URL="https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
IAM_URL="https://huggingface.co/datasets/Teklia/IAM-line/resolve/main/data"

emnist () {
  dir="$DATA/emnist"
  mkdir -p "$dir"
  if [ ! -f "$dir/gzip.zip" ]; then
    echo "downloading EMNIST..."
    curl -fL --retry 3 -o "$dir/gzip.zip" "$EMNIST_URL"
  fi
  echo "unpacking the 'byclass' split (0-9 A-Z a-z)..."
  unzip -o -j "$dir/gzip.zip" "gzip/emnist-byclass-*" -d "$dir"
  for f in "$dir"/emnist-byclass-*.gz; do
    [ -f "$f" ] || continue
    gunzip -f "$f"
  done
  echo "done: $dir"
}

# IAM handwritten lines (via Teklia on Hugging Face). The IAM database is
# free for non-commercial research only: see
# https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
iam () {
  dir="$DATA/iam"
  mkdir -p "$dir"
  for s in train validation test; do
    if [ ! -f "$dir/$s.parquet" ]; then
      echo "downloading IAM $s..."
      curl -fL --retry 3 -o "$dir/$s.parquet" "$IAM_URL/$s.parquet"
    fi
  done
  if [ ! -d "$DATA/.pylib/pyarrow" ]; then
    echo "installing pyarrow into $DATA/.pylib (only for unpacking)..."
    python -m pip install --quiet --target "$DATA/.pylib" pyarrow
  fi
  python iamconv.py "$dir"
  echo "done: $dir"
}

case "${1:-}" in
  emnist)
    emnist
    ;;
  iam)
    iam
    ;;
  list)
    if [ -d "$DATA" ]; then
      ls -la "$DATA"/*
    else
      echo "nothing downloaded yet"
    fi
    ;;
  *)
    echo "usage: ./getdata.sh [emnist|iam|list]"
    exit 1
    ;;
esac
