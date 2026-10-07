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

Needs [zig](https://ziglang.org) (used as the C++ compiler, from `$CXX`,
default `zig c++`) and a bash-like shell; the project uses the mmc shell
(`D:\mmc-shell\mmc.exe`). Every build makes the three programs
`mmc_reader`, `mmc_train` and `mmc_trainseq`.

| Command | Builds | Output |
|---------|--------|--------|
| `./build.sh` | release (same as `release`) | `build/release/` |
| `./build.sh release` | optimized, vectorized math | `build/release/` |
| `./build.sh debug` | no optimization, debug info, `MR_DEBUG` checks | `build/debug/` |
| `./build.sh gpu` | release + GPU support (OpenCL, `MR_USE_GPU=1`) | `build/gpu/` |
| `./build.sh cross-release` | release for every default target | `build/cross-release/<target>/` |
| `./build.sh cross-release T...` | release for the given zig targets | `build/cross-release/<T>/` |
| `./build.sh cross-debug [T...]` | debug for the default / given targets | `build/cross-debug/<T>/` |
| `./build.sh targets` | nothing; lists the default targets | |
| `./build.sh clean` | nothing; removes build output, keeps `build/data` | |

Default cross targets: `x86_64-windows-gnu`, `x86_64-linux-gnu`,
`aarch64-linux-gnu`, `x86_64-macos`, `aarch64-macos`. Any zig target
works, for example:

```sh
./build.sh cross-release x86_64-linux-musl        # fully static Linux
./build.sh cross-release x86_64-windows-gnu aarch64-macos
```

Extra compiler flags and libraries:

```sh
CFLAGS="-fsanitize=address" ./build.sh debug
LIBS="path/to/libx.a" ./build.sh
CXX="g++" ./build.sh                              # another compiler
```

From outside the mmc shell: `D:\mmc-shell\mmc.exe build.sh release`.

Each program is one standalone file: on Windows it needs no DLLs
besides Windows' own (no zig, MinGW or Visual C++ runtime to install).

### GPU build

`./build.sh gpu` adds GPU support through OpenCL. The OpenCL driver
(`OpenCL.dll` / `libOpenCL.so`) is loaded when the program runs, so
nothing extra is needed to build, and the program still runs on a PC
without a GPU. Check the GPU:

```sh
./build/gpu/mmc_trainseq -G
GPU: Intel(R) UHD Graphics 730, 24 compute units, 5.0 GB
GPU self-test: ok
```

Status: the GPU is found and tested, but training and reading still run
on the CPU; the GPU versions of the network layers are future work.

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
./build/release/mmc_trainseq -P -k 256 -e 150 -L 16000 -S 1.0 \
  -o build/release/print.mrm           # stops once below 1% errors
./build/release/mmc_trainseq -e 60 -y 40 -o build/release/cursive.mrm
```

A stopped line-model training can go on from its saved model with
`-R model` (it only overwrites the model when it gets better).

Each trainer tests the new model on data it never saw (other fonts or
other writers) and prints the character error rate. Training uses the
CPU (all cores); run one trainer at a time on machines with little
memory. Line-model options: `-k` LSTM size, `-L` lines per epoch, `-S`
stop below an error rate, `-R` resume, `-t` test, `-G` check the GPU.

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
