# mmc_reader

Read the text in images: printed documents, screenshots, hand-printed
forms and cursive handwriting. The OCR engine is written **from scratch**
(no tesseract, no machine-learning libraries) in C-style C++11, and its
models are trained by our own programs.

```
$ mmc_reader scan.png
The quick brown fox jumps over the lazy dog.
Señor Niño paid $1,250.00 for the piña in Parañaque.
```

- English plus `ñ Ñ`, Spanish accents (`á é í ó ú ü ¿ ¡`), smart quotes,
  dashes, `₱ € £ ¥ ¢`, `° © ® ™ × ÷ ± § ¶ ½ ¼ ¾` and all ASCII symbols.
- Light or dark backgrounds, table lines and scanner dust are cleaned up.
- File names and output are UTF-8 on every system.
- One small `.exe` per program; no DLLs besides Windows' own.
- Images only (PNG, JPG, BMP, GIF, TGA, PSD, PNM). For PDF, extract the
  pages to images with another tool first.

## Build

Needs [zig](https://ziglang.org) (used as the C++ compiler) and a
bash-like shell; the project uses the mmc shell (`D:\mmc-shell\mmc.exe`).

```sh
./build.sh                  # release build -> build/release/
./build.sh debug            # debug build   -> build/debug/
./build.sh cross-release    # Windows, Linux, macOS -> build/cross-release/
./build.sh clean            # remove build output (keeps build/data)
```

## Use

The models (`*.mrm`) must be next to `mmc_reader` (or give `-d dir`).

```sh
./build/release/mmc_reader picture.png               # printed text
./build/release/mmc_reader -l print scan.jpg         # printed, line model
./build/release/mmc_reader -l hand form.jpg          # hand-printed letters
./build/release/mmc_reader -l cursive note.jpg       # cursive handwriting
./build/release/mmc_reader -o out.txt a.png b.png    # several files, to a file
./build/release/mmc_reader -i picture.png            # image info only
./build/release/mmc_reader -h                        # help
```

| Model | Reads | How |
|-------|-------|-----|
| `eng.mrm` (default) | printed text | one letter at a time |
| `print.mrm` | printed text | a whole line at a time (like tesseract 4) |
| `hand.mrm` | printed and hand-printed (block) letters | one letter at a time |
| `cursive.mrm` | joined handwriting | a whole line at a time |

Exit status: 0 all files read, 1 some file failed, 2 bad command line.

## How it works

```
image -> gray -> black/white (dark mode, table lines and dust removed)
      -> text lines -> letters or whole lines
      -> neural network -> text
```

- **Letter models** (`eng`, `hand`): each letter is cut out and
  classified by a small neural network, using its shape and its size and
  place in the line. Broken letters are glued, touching letters split.
- **Line models** (`print`, `cursive`): a whole line goes through a
  convolutional network and two LSTMs (left-to-right and right-to-left),
  trained with CTC, so letters never need to be cut apart.

Everything (the networks, their training, CTC, the image code) is plain
C-style code in the project root; see [CONTRIBUTING.md](CONTRIBUTING.md)
section 7 for the details.

## Train the models

Training data goes in `build/data` (not in git):

```sh
./getdata.sh fonts      # ~800 Google Fonts (OFL / Apache)
./getdata.sh books      # 22 Project Gutenberg books (public domain)
./getdata.sh emnist     # handwritten letters (NIST)
./getdata.sh iam        # handwritten lines (non-commercial research only)
```

```sh
./build/release/mmc_train -n 2500000 -l 384,192 -o build/release/eng.mrm
./build/release/mmc_train -H build/data/emnist -n 3000000 -l 384,192 \
  -o build/release/hand.mrm
./build/release/mmc_trainseq -P -e 40 -L 8000 -o build/release/print.mrm
./build/release/mmc_trainseq -e 60 -y 40 -o build/release/cursive.mrm
```

Each trainer tests the new model on data it never saw (other fonts or
other writers) and prints the character error rate. Training uses the
CPU only; run one trainer at a time on machines with little memory.

## Accuracy so far

Character errors on test data the models never trained on:

| Test | Model | Errors |
|------|-------|--------|
| clean printed pages | `eng` | 3.9% |
| printed pages with special characters | `eng` | 7.0% |
| hand-printed lines, unseen EMNIST writers | `hand` | 18.4% |
| IAM handwritten lines, unseen writers | `cursive` | 19.0% |

Real scans with stamps, signatures and tables are harder; tesseract is
still more accurate. Known limits are listed in CONTRIBUTING.md
section 7.

## Layout

All code is flat in the project root, Lua style (`mr*.cpp` modules,
`mmc_*.cpp` programs); third-party code is in `vendor/` (only
stb_image and stb_truetype). Read [CONTRIBUTING.md](CONTRIBUTING.md)
before changing anything: C-style code, no `class`, 2-space indent, no
tabs, C++11 only.

## License

The project license is not chosen yet. Third-party parts:
`vendor/README.md`. The IAM dataset, and so `cursive.mrm` trained on it,
is for **non-commercial research only**.
