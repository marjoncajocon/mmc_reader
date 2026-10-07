#!/usr/bin/env mmc
#
# getdata.sh - download training datasets into build/data
# Run it from the mmc shell (D:\mmc-shell\mmc.exe):
#
#   ./getdata.sh emnist    handwritten letters and digits (NIST, ~560 MB)
#   ./getdata.sh iam       handwritten English lines (IAM, ~270 MB,
#                          non-commercial research use only)
#   ./getdata.sh fonts     ~800 Google Fonts files: sans, serif, mono,
#                          display, handwriting (OFL / Apache, ~60 MB)
#   ./getdata.sh books     public domain books from Project Gutenberg,
#                          English and Spanish (~25 MB), for real text
#   ./getdata.sh all       all of the above
#   ./getdata.sh list      show what is downloaded
#
# Datasets are never committed: build/ is ignored by git, and
# ./build.sh clean keeps build/data.
#

set -e

DATA="build/data"
EMNIST_URL="https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip"
IAM_URL="https://huggingface.co/datasets/Teklia/IAM-line/resolve/main/data"
FONTS_META="https://fonts.google.com/metadata/fonts"
BOOKS_URL="https://www.gutenberg.org/cache/epub"

# Gutenberg book numbers: 19 English novels, 3 Spanish (for ñ and accents)
BOOKS="1342 1661 2701 98 84 11 1400 345 76 174 120 1260 768 158 5200"
BOOKS="$BOOKS 2591 1952 219 64317 2000 15353 17013"

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

# python packages for datatool.py go in build/data/.pylib, not in your
# Python
pylib () {
  if [ ! -d "$DATA/.pylib/$1" ]; then
    echo "installing $1 into $DATA/.pylib (only for unpacking)..."
    python -m pip install --quiet --target "$DATA/.pylib" "$1"
  fi
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
  pylib pyarrow
  python datatool.py iam "$dir"
  echo "done: $dir"
}

# Google Fonts: the most popular open source families of each kind
fonts () {
  dir="$DATA/fonts"
  mkdir -p "$dir"
  if [ ! -f "$dir/metadata.json" ]; then
    curl -fsL --retry 3 -o "$dir/metadata.json" "$FONTS_META"
  fi
  python datatool.py fontscript "$dir/metadata.json" "$dir" > "$dir/get.sh"
  echo "downloading fonts (files already there are skipped)..."
  mmc "$dir/get.sh"
  python datatool.py fontlist "$dir/metadata.json" > "$dir/list.txt"
  echo "done: $dir"
}

books () {
  dir="$DATA/books"
  mkdir -p "$dir"
  for id in $BOOKS; do
    if [ ! -s "$dir/pg$id.txt" ]; then
      echo "downloading book $id..."
      curl -fsL --retry 3 -o "$dir/pg$id.txt" "$BOOKS_URL/$id/pg$id.txt"
    fi
  done
  rm -f "$dir/list.txt"
  for id in $BOOKS; do
    if [ -s "$dir/pg$id.txt" ]; then
      echo "pg$id.txt" >> "$dir/list.txt"
    fi
  done
  echo "done: $dir"
}

case "${1:-}" in
  emnist)
    emnist
    ;;
  iam)
    iam
    ;;
  fonts)
    fonts
    ;;
  books)
    books
    ;;
  all)
    emnist
    iam
    fonts
    books
    ;;
  list)
    if [ -d "$DATA" ]; then
      ls -la "$DATA"/*
    else
      echo "nothing downloaded yet"
    fi
    ;;
  *)
    echo "usage: ./getdata.sh [emnist|iam|fonts|books|all|list]"
    exit 1
    ;;
esac
