# Contributing to mmc_reader

mmc_reader reads text from images (OCR) with its **own engine, written
from scratch**: no tesseract. A small neural network, trained by our own
`mmc_train` program from TrueType fonts, recognizes each letter. The first
target is English plus `ñ` / `Ñ`.

It is compiled as **C++11** (`.cpp` files), but written the way a
**C programmer** writes C: plain `struct`s and functions, **no `class`**,
like Lua's sources. C++ is used only so we can link both C and C++
libraries from one codebase when needed.

Read this whole file before sending a change. Code that does not follow it
will be asked to change.

---

## 1. Project layout

The layout follows the Lua source tree: **every source file lives flat in
the project root**. No `src/`, `include/`, `lib/` or nested module folders.
The only subfolder for code is `vendor/`.

```
mmc_reader/
  mr.h            public API (the only header users of the library include)
  mrconf.h        configuration, feature switches, linkage macros
  mrlimits.h      internal basic types, limits, helper macros
  mrstate.cpp/.h  the mr_State struct, create/close, errors (mrS_)
  mrmem.cpp/.h    memory allocation           (mrM_)
  mrbuf.cpp/.h    growable byte/string buffer (mrB_)
  mrfile.cpp/.h   fopen with UTF-8 paths      (mrF_)
  mrrand.cpp/.h   random numbers              (mrR_)
  mrimage.cpp/.h  image loading and pixels    (mrI_)
  mrbin.cpp/.h    gray -> black/white bitmap  (mrK_)
  mrlayout.cpp/.h lines, letter segments, spaces (mrL_)
  mrglyph.cpp/.h  segment -> network input    (mrG_)
  mrnet.cpp/.h    neural network, training, model file (mrN_)
  mrocr.cpp/.h    OCR pipeline: image -> text (mrO_)
  mrseq.cpp/.h    line model: CNN + LSTM + CTC (mrQ_)
  mrthread.cpp/.h run work on all CPU cores   (mrX_)
  mrgpu.cpp/.h    GPU through OpenCL (gpu build) (mrU_)
  mrfont.cpp/.h   TrueType text drawing, for training (mrT_)
  mrhand.cpp/.h   EMNIST handwriting lines, for training (mrH_)
  mrpdf.cpp/.h    PDF pages (stub for now)    (mrP_)
  mrapi.cpp       implementation of mr.h      (mr_)
  mmc_reader.cpp  the reader program (like lua.c)
  mmc_train.cpp   letter model trainer (like luac.c)
  mmc_trainseq.cpp line model trainer (handwriting)
  build.sh        build script (run from the mmc shell, see section 6)
  getdata.sh      downloads training datasets into build/data
  datatool.py     dataset helper run by getdata.sh (IAM, font list)
  CONTRIBUTING.md
  vendor/
    README.md     list of vendored libraries, versions, licenses
    stb/          stb_image.h, stb_truetype.h
```

PDF is not supported for now: `mrpdf.cpp` is a stub that returns
`MR_ERRNOTSUP` (`MR_USE_PDF` in `mrconf.h`). Extract PDF pages to images
with another tool first.

Rules:

- Sources use `.cpp`, headers use `.h`. Never `.c`, `.cc`, `.cxx`, `.hpp`.
- One module = one `.cpp` + one `.h` pair, named `mr<module>.cpp` /
  `mr<module>.h`, all lowercase, no underscores, short (like `lstring.c`,
  `lmem.c` in Lua).
- Programs are `mmc_<name>.cpp` and listed in `PROGS` in `build.sh`; every
  other `.cpp` is library code linked into each program.
- `mmc_reader.cpp` uses only the public API in `mr.h`, exactly like `lua.c`
  uses only `lua.h` / `lauxlib.h`. The trainers (`mmc_train.cpp`,
  `mmc_trainseq.cpp`) are tools for us, so, like `luac.c`, they may use
  the internal headers.
- Scripts (`build.sh`, `getdata.sh`) run in the mmc shell. `datatool.py`
  is the only Python: it unpacks a dataset format (Parquet) and reads the
  Google Fonts list (JSON), formats we will not
  write a reader for. It is never part of the program.
- Do not create new folders. If you think one is needed, open an issue first.

---

## 2. Language standard

The standard is **C++11 only**. Every `.cpp` file is compiled with:

```
zig c++ -std=c++11 -Wall -Wextra -pedantic
```

and must build **without warnings**. Do not use anything newer than C++11.

The code is C++, but written in **C style**: data lives in plain `struct`s,
behavior lives in free functions that take a pointer to the struct.
**Never use `class`.** If it would not look normal to a C programmer, do not
write it.

```cpp
/* yes: plain struct + functions */
typedef struct mr_Buffer {
  char *data;
  size_t len;
  size_t cap;
} mr_Buffer;

void mrB_init (mr_Buffer *b);
int mrB_addstr (mr_State *R, mr_Buffer *b, const char *s);
void mrB_free (mr_State *R, mr_Buffer *b);

/* no */
class Buffer {
 public:
  Buffer();
  void add(const std::string &s);
 private:
  std::vector<char> data;
};
```

### 2.1 Allowed

Only the **C-compatible subset of C++11**:

- `struct` holding **data only**: fields, nothing else.
- `union`, `enum`, `typedef`, functions, function pointers.
- `static` functions for anything not exported.
- `static inline` functions (never plain `inline` or `extern inline`;
  the rules differ between C and C++).
- `const`, `<stdint.h>`, `<stddef.h>`, `<stdbool.h>` (or plain `int` for flags,
  like Lua).
- C standard headers: `<stdio.h>`, `<stdlib.h>`, `<string.h>`, ...
  (never `<cstdio>`, `<cstring>`, ...).
- `goto` for cleanup/error paths (see 4.6).

### 2.2 Forbidden

C++ features (they do not exist in C):

- **`class`**. Also inside a `struct`: member functions, constructors,
  destructors, `public`/`private`/`protected`, inheritance, `virtual`,
  `static` members, default member initializers.
- `namespace`, `using`, templates, function/operator overloading.
- References (`int &x`), `new` / `delete`, `auto`, lambdas, `constexpr`,
  `nullptr`, range-for, `enum class`, default arguments.
- Exceptions (`try`/`catch`/`throw`) and RTTI.
- STL / C++ standard library (`std::string`, `std::vector`, `<iostream>`, ...).
- `//` comments: **do not use them**; use `/* */` like Lua.

C habits that are invalid or different in C++ (they will not compile, or
behave differently, under `-std=c++11`):

- Designated initializers (`{ .x = 1 }`) and compound literals.
- Variable-length arrays and flexible array members (`char data[];`).
- `restrict`, `_Generic`, `_Atomic`, `_Thread_local`, complex numbers.
- Implicit conversion from `void *` -- always cast (use `mr_cast`):
  ```c
  p = mr_cast(Node *, mrM_malloc(R, sizeof(Node)));
  ```
- Implicit `int` -> `enum` conversion -- cast it.
- Empty parameter lists: write `int f (void)`, never `int f ()`.
- String literals are `const char *`. Never assign them to `char *`.
- Using C++ keywords as identifiers: `new`, `delete`, `class`, `this`,
  `template`, `typename`, `private`, `public`, `operator`, `try`, `catch`,
  `throw`, `namespace`, `virtual`, `friend`, `export`, ...
- File-scope `const` variables: C++ gives them internal linkage. Write
  `static const` for private constants, and `extern const` in the header plus
  a definition in one `.cpp` file for shared ones.
- `_Static_assert`, `_Noreturn`, `_Alignof`: use the macros from `mrconf.h`
  (`mr_static_assert`, `MR_NORETURN`, ...) which pick the right spelling.
- `setjmp` / `longjmp`: do not use. Report errors with return codes (4.6).

---

## 3. Formatting

- **Indent with 2 spaces. Never use the tab character**, anywhere, in any
  of our files (`.cpp`, `.h`, `build.sh`, `.md`). Configure your
  editor to insert spaces.
- Max line length: 80 columns.
- Unix line endings (`LF`), UTF-8, one newline at end of file, no trailing
  whitespace.
- Braces: opening brace on the same line, closing brace on its own line.
- `else` on the same line as the closing brace: `} else {`.
- A single-statement body may omit braces only if it fits on the same line.
- Function definitions and declarations put **a space before the parameter
  list** (Lua style). Calls do **not**:
  ```c
  static int readpage (mr_State *R, int n) {     /* definition */
    return mrP_render(R, n);                     /* call */
  }
  ```
- One space after keywords: `if (`, `for (`, `while (`, `switch (`, `return x;`.
- Pointer `*` sticks to the name: `char *s`, `const mr_Image *img`.
- `switch`: `case` labels indented one level inside the `switch`.

Example:

```c
/*
** Load an image from 'path' and run OCR on it.
** Returns MR_OK or an error code; on success '*out' holds the text.
*/
int mr_readimage (mr_State *R, const char *path, mr_Buffer *out) {
  mr_Image *img = NULL;
  int status;
  status = mrI_load(R, path, &img);
  if (status != MR_OK) return status;
  switch (img->channels) {
    case 1: break;
    case 3:
    case 4:
      status = mrI_togray(R, img);
      break;
    default:
      status = MR_ERRFORMAT;
      break;
  }
  if (status == MR_OK)
    status = mrO_recognize(R, img, out);
  mrI_free(R, img);
  return status;
}
```

---

## 4. Naming and structure

### 4.1 Prefixes (Lua style)

| Kind                         | Prefix         | Example                    |
|------------------------------|----------------|----------------------------|
| Public API functions         | `mr_`          | `mr_open`, `mr_readpdf`    |
| Public types                 | `mr_` + Capital| `mr_State`, `mr_Buffer`    |
| Public macros / constants    | `MR_`          | `MR_OK`, `MR_ERRMEM`       |
| Internal module functions    | `mr<X>_`       | `mrM_malloc`, `mrB_addstr` |
| Static (file-local) functions| no prefix      | `readheader`, `skipws`     |

Internal module letters: `S` state, `M` memory, `B` buffer, `F` file,
`R` random, `I` image, `K` black/white bitmap, `L` layout, `G` glyph,
`N` network, `O` ocr, `T` TrueType, `P` pdf, `H` handwriting data,
`Q` line model (sequence), `X` threads, `U` GPU. Pick a new unused capital
letter for a new module and add it to this list.

### 4.2 Identifiers

- Functions and variables: all lowercase, words run together, short
  (`getline`, `numpages`, `nchars`) -- like Lua. Use `_` only after a prefix.
- Types: `typedef struct mr_Image { ... } mr_Image;` -- C++ does not need
  the typedef, but we keep it (Lua style) so `mr.h` can also be used from C.
- Macros: UPPER_CASE, except function-like macros that behave like functions,
  which may be lowercase (`mr_cast`, `mrM_new`), as in Lua.
- The interpreter-style state is passed first and named `R`
  (Lua uses `L`): `int mrI_load (mr_State *R, ...)`.

### 4.3 Headers

Every header:

```c
/*
** mrimage.h
** Image loading and pixel access
** See Copyright Notice in mr.h
*/

#ifndef mrimage_h
#define mrimage_h

#include "mrlimits.h"

typedef struct mr_Image {
  int width, height, channels;
  unsigned char *pixels;
} mr_Image;

MRI_FUNC int mrI_load (mr_State *R, const char *path, mr_Image **out);
MRI_FUNC void mrI_free (mr_State *R, mr_Image *img);

#endif
```

- Include guard: `<filename>_h` in lowercase, like Lua.
- Only the public header `mr.h` wraps its declarations in
  `MR_BEGIN_DECLS` / `MR_END_DECLS` (`extern "C"`, defined in `mrconf.h`),
  so the library can also be called from C programs.
- Public functions are marked `MR_API`, internal ones `MRI_FUNC`.
- Include only what the header itself needs.

### 4.4 Source files

Every `.cpp` file starts with the same comment block, then defines its module
name and includes `mrconf.h`-based headers before system headers:

```c
/*
** mrimage.cpp
** Image loading and pixel access
** See Copyright Notice in mr.h
*/

#define mrimage_cpp
#define MR_CORE

#include "mrimage.h"
#include "mrmem.h"

#include <string.h>
```

Group long files into sections the way Lua does:

```c
/*
** {======================================================
** PNG / JPEG decoding
** =======================================================
*/

...

/* }====================================================== */
```

### 4.5 Memory

- Library code never calls `malloc`/`free` directly outside `mrmem.cpp`.
  Use `mrM_malloc`, `mrM_realloc`, `mrM_free` (and helper macros like
  `mrM_new(R, T)`), which go through the allocator stored in `mr_State`,
  like Lua's `lua_Alloc`.
- Like `lua_Alloc`, the allocator gets the old size of every block, so
  `mrM_free(R, p, size)` must be given the size the block was allocated
  with.
- `mmc_reader.cpp` is a program, not library code, and may use `malloc`
  for its own needs (like `lua.c`).
- Every allocation has exactly one clear owner. Document who frees it.

### 4.6 Errors

- Functions that can fail return `int` status: `MR_OK` (0) or an `MR_ERR*`
  code. Results come back through out-parameters.
- A human-readable message can be stored with `mr_seterror(R, fmt, ...)`
  and read with `mr_geterror(R)`.
- Clean up with a single `goto` label at the end of the function. Because
  C++ forbids jumping over initializations, **declare and initialize all
  variables at the top of the block, before the first `goto`**:

```c
int mr_readpdf (mr_State *R, const char *path, mr_Buffer *out) {
  mr_Pdf *pdf = NULL;
  mr_Image *page = NULL;
  int i, n, status;
  status = mrP_open(R, path, &pdf);
  if (status != MR_OK) goto done;
  n = mrP_numpages(pdf);
  for (i = 0; i < n; i++) {
    status = mrP_render(R, pdf, i, &page);
    if (status != MR_OK) goto done;
    status = mrO_recognize(R, page, out);
    mrI_free(R, page);
    page = NULL;
    if (status != MR_OK) goto done;
  }
 done:
  mrI_free(R, page);
  mrP_close(R, pdf);
  return status;
}
```

- No global mutable state. Everything lives in `mr_State`.

### 4.7 Comments

- `/* */` only.
- Block comments before functions use the Lua `/* ** ... */` form.
- Comment *why*, not *what*. Refer to variables in comments with single
  quotes: `/* 'n' counts pages already rendered */`.

---

## 5. Third-party libraries (`vendor/`)

All external code goes in `vendor/<libname>/`, one folder per library.

- Add the library's original `LICENSE` file inside its folder.
- Record name, version/commit, source URL and license in `vendor/README.md`.
- **Do not edit vendored code.** If a patch is unavoidable, keep it minimal
  and list it in `vendor/README.md`.
- Our style rules (2 spaces, C-style, etc.) do **not** apply to vendor code.
- Our code must not include vendor headers from public headers (`mr.h`).
  Only the module that wraps a library includes it.
- Prefer small C libraries, or C++ libraries that offer a C API. Check the
  license is compatible before adding anything (e.g. an AGPL library makes
  the whole program AGPL).

### 5.1 C libraries

Include them directly from the module that uses them, with the path
relative to `vendor/`:

```c
#include "stb/stb_image.h"
```

`build.sh` passes `-isystem vendor`, so warnings inside vendored code are
not shown; only our own code must be warning-free. A header-only library
(like stb) defines its implementation macro in the one module that uses it
(`#define STB_IMAGE_IMPLEMENTATION` in `mrimage.cpp`).

If a C header has no `extern "C"` guards of its own, wrap the include in
`extern "C" { ... }` so its functions link with C names.

### 5.2 C++-only libraries

When a library has only a C++ API, write **one wrapper module** for it
(e.g. `mrxyz.cpp` / `mrxyz.h`). It is the **only** file that includes the
library's headers, and the only place where C++ features are allowed, and
only as much as the library's API forces (calling its methods, holding its
objects, catching its exceptions):

- `mrxyz.h` exposes plain functions and an opaque struct
  (`typedef struct mr_Xyz mr_Xyz;`), so no library type leaks out.
- It must catch every exception thrown by the library and convert it to an
  `MR_ERR*` code. No exception may leave the wrapper.
- Keep it as small as possible; logic belongs in the C-style modules.

An optional library can be turned off with a `MR_USE_<LIB>` switch in
`mrconf.h`; the wrapper then compiles to a stub that returns
`MR_ERRNOTSUP`.

---

## 6. Building

The project is built with **`build.sh`** from the **mmc shell**
(`D:\mmc-shell\mmc.exe`, or `mmc-term.exe` for the terminal window).
The compiler is **zig** (`D:\env\zig`), taken from `$CXX`, which
`D:\mmc-shell\etc\profile` sets to `zig c++`.

```sh
./build.sh                        # release build (default)
./build.sh release                # -O2 -DNDEBUG
./build.sh debug                  # -O0 -g -DMR_DEBUG
./build.sh gpu                    # release + MR_USE_GPU=1 (OpenCL)
./build.sh cross-release [T...]   # release, cross-compiled by zig
./build.sh cross-debug [T...]     # debug, cross-compiled by zig
./build.sh targets                # list the default cross targets
./build.sh clean                  # remove build output, keep build/data
```

From outside the mmc shell: `D:\mmc-shell\mmc.exe build.sh debug`.

### 6.1 Cross compilation

zig can build for other systems from Windows. `T` is a zig target triple.
With no `T`, every target in the `TARGETS` list in `build.sh` is built:

| Target               | System                 |
|----------------------|------------------------|
| `x86_64-windows-gnu` | Windows 64-bit         |
| `x86_64-linux-gnu`   | Linux 64-bit           |
| `aarch64-linux-gnu`  | Linux ARM64            |
| `x86_64-macos`       | macOS Intel            |
| `aarch64-macos`      | macOS Apple Silicon    |

```sh
./build.sh cross-release                          # all targets
./build.sh cross-debug x86_64-linux-gnu           # one target
./build.sh cross-release x86_64-linux-musl aarch64-macos
```

Note: `x86_64-windows.7-gnu` (Windows 7) does not build C++ with the
current zig (its bundled libunwind needs newer Windows APIs). Use
`x86_64-windows-gnu`, or for Windows 7 use another compiler (e.g. MinGW
`g++`) by changing `CXX` / `ZIG` in `build.sh`.

Windows builds link `-lshell32` (`WINLIBS` in `build.sh`), used to read
the command line as UTF-8.

Cross builds use `$ZIG c++ -target T` (`$ZIG` defaults to `zig`).
Vendor libraries must be built for each target too: pass them per target
with `LIBS=...`, or prefer libraries that ship as source (`stb`, ...) so
`build.sh` can compile them with the rest.

### 6.2 Output folders

All build output goes in `build/`, one folder per mode (and per target for
cross builds):

```
build/
  release/
    mmc_reader.exe
    mmc_reader.pdb   (debug symbols, written by zig on Windows)
    mmc_train.exe
    eng.mrm          (trained model, see section 7)
    obj/
      mmc_reader.o
      mrimage.o
      ...
  debug/
    mmc_reader.exe
    mmc_reader.pdb
    obj/
      ...
  cross-release/
    x86_64-windows-gnu/
      mmc_reader.exe
      obj/
    x86_64-linux-gnu/
      mmc_reader
      obj/
    aarch64-macos/
      mmc_reader
      obj/
    ...
  cross-debug/
    x86_64-linux-gnu/
      mmc_reader
      obj/
    ...
```

### 6.3 Rules

- `build.sh` compiles every `*.cpp` in the root with
  `-std=c++11 -Wall -Wextra -pedantic` plus the mode flags, then links
  each program in `PROGS` (`mmc_reader`, `mmc_train`) with all the
  library objects.
- On Windows a running `.exe` cannot be replaced: stop a long
  `mmc_train` run before rebuilding the same mode.
- Release builds add `-fassociative-math -fno-signed-zeros -fno-trapping-math`
  (`FASTMATH`): the compiler may reorder float sums and so use vector
  instructions, about 4x faster training. Never `-ffast-math`: it drops
  the NaN/Inf checks of the model loaders.
- The GPU build loads OpenCL at run time (`LoadLibrary` / `dlopen`, no
  SDK or header). `mrgpu.cpp` declares the few OpenCL functions it uses.
  Other builds compile `mrgpu.cpp` to stubs that return `MR_ERRNOTSUP`,
  so callers never need `#if`. Network kernels on the GPU are future
  work: until then the GPU build trains and reads on the CPU.
- Nothing is ever written outside `build/`. Never commit `build/`.
- Downloaded training datasets go in `build/data/`; `clean` keeps it.
- Debug-only code goes inside `#if defined(MR_DEBUG)`.
- Extra flags: `CFLAGS="-fsanitize=address" ./build.sh debug`, and libraries:
  `LIBS="vendor/x/libx.a" ./build.sh`. Add permanent libraries to `LIBS`
  inside `build.sh`.
- Keep `build.sh` to plain sh/bash features so `mmc --check build.sh`
  passes.

Before sending a change, run `./build.sh` and fix every warning.

---

## 7. The OCR engine and training

### 7.1 How a page is read

```
image -> gray (mrimage)
  -> black/white, Otsu threshold; dark mode is detected (mrbin)
  -> reader only: table borders, rules and underlines erased (long
     straight ink runs, one pixel of tilt allowed) (mrbin)
  -> scanner dust smaller than any '.' ignored (mrlayout)
  -> connected ink parts -> text lines -> letter segments;
     parts on top of each other are one letter: i j ñ : ; ! ? = (mrlayout)
  -> baseline, x-height (lower of two height groups), word gaps by
     2-means per line (mrlayout)
  -> each segment: 32x32 shape + 5 size/position numbers (mrglyph)
  -> neural network -> letter + confidence (mrnet)
  -> unsure neighbors are tried glued together: broken letters, %, ½;
     refused when the network says "not one letter" (mrocr)
  -> "not one letter" segments are split at their thinnest columns;
     the cut where both halves are sure letters wins, and halves are
     split again, up to 8 letters (mrocr)
  -> two single quote marks in a row become one double quote:
     '' -> "   ‘‘ -> “   ’’ -> ”   (mrocr)
  -> UTF-8 text, one line per text line (mrocr)
```

The model knows 135 letters and one more class:

| Group | Letters |
|-------|---------|
| ASCII | `!` .. `~` (letters, digits, all 32 symbols) |
| Filipino / Spanish | `ñ Ñ á é í ó ú ü Á É Í Ó Ú Ü ¿ ¡` |
| Typographic | `“ ” ‘ ’ – — • …` |
| Currency | `₱ € £ ¥ ¢` (and `$` from ASCII) |
| Symbols | `° © ® ™ × ÷ ± § ¶ ½ ¼ ¾` |
| **reject** (`MR_REJECT`) | "this is not one letter": two letters touching or glued |

Spaces come from the gaps, not from the network. An ellipsis drawn as
three separate dots reads as `...`.

### 7.2 Training a model

`mmc_train` draws random English-like lines with TrueType fonts, at
random sizes, contrast, noise and blur. The special letters appear where
they appear in real text: `₱1,250.00`, `25°C`, `99¢`, `“quoted”`, `it’s`,
`• item`, `§ 4`, `Acme™`, `7 × 8`, `¿Qué?`, `¡Hola!`, `José`, `canción`. Each line goes through **the same layout
code** as real images, and every segment is matched to the letter it came
from. Segments that cover two letters, and some neighbor pairs glued on
purpose, become **reject** samples. Rare symbols are mixed in more often
until every class has enough examples. Every sample is new, so it never
trains on the same image twice.

```sh
./build/release/mmc_train -o build/release/eng.mrm          # Windows fonts
./build/release/mmc_train -n 3000000 -l 384,192 -o build/release/eng.mrm
./build/release/mmc_train -o my.mrm a.ttf b.ttf c.ttc#1     # chosen fonts
./build/release/mmc_train -t build/release/eng.mrm -c 500   # test a model
```

| Option | Meaning | Default |
|--------|---------|---------|
| `-o file`  | output model | `eng.mrm` |
| `-n count` | training samples | 1000000 (about 10 minutes at 256,128) |
| `-l sizes` | hidden layer sizes | `256,128` |
| `-r rate`  | learning rate (Adam) | `0.001` |
| `-s seed`  | random seed | `1` |
| `-t model` | test a model instead of training | |
| `-c lines` | lines drawn for the test | 300 |

After training (and with `-t`), it reads new random lines with the full
OCR and prints the **character error rate**, with some wrong lines.

Fonts must have every ASCII letter and `ñ`/`Ñ`, or they are skipped. Other
letters a font lacks (many fonts have no `₱`) are simply not drawn with
that font; `mmc_train` prints a note for letters fewer than a quarter of
the fonts have. To add a letter: put it in `extras` in `mmc_train.cpp`,
raise `NEXTRAS`, give it natural places in the text generator, and
retrain. The model file stores its own letter list, so `mmc_reader` needs
no change.

### 7.3 Using a model

`mmc_reader` loads `<name>.mrm` (`-l name`, default `eng`) from the
program's folder, or from `-d dir`. There are three models:

| Model | Kind | Reads | Trained by |
|-------|------|-------|------------|
| `eng.mrm` | letter model | printed text | `mmc_train` |
| `hand.mrm` | letter model | printed text and hand-printed (block) letters | `mmc_train -H` |
| `print.mrm` | line model | printed text, line by line (like tesseract 4) | `mmc_trainseq -P` |
| `cursive.mrm` | line model | joined handwriting, line by line | `mmc_trainseq` |

```sh
./build/release/mmc_reader page.png              # eng
./build/release/mmc_reader -l hand form.jpg      # hand-printed form
./build/release/mmc_reader -l cursive letter.jpg # cursive
```

The file says which kind it is (`MRNN` letter model, `MRSQ` line model),
so the reader picks the right path by itself. Files are little-endian and
checked against the program: a model made for another input size is
refused, so change `MODELVERSION` in `mrnet.cpp` (or `SEQVERSION` in
`mrseq.cpp`) when the input or format changes.

### 7.4 Handwriting

**Data.** Datasets go in `build/data` (git-ignored; `clean` keeps it):

```sh
./getdata.sh emnist   # handwritten letters/digits, NIST, ~560 MB
./getdata.sh iam      # handwritten English lines, ~270 MB
./getdata.sh fonts    # ~800 Google Fonts (OFL/Apache), ~66 MB
./getdata.sh books    # 22 Project Gutenberg books (public domain), 16 MB
```

The IAM database is free for **non-commercial research only**
(https://fki.tic.heia-fr.ch/databases/iam-handwriting-database). Do not
ship `cursive.mrm` in a commercial product without checking that license.

**Hand-print (`hand.mrm`).** EMNIST has only single 28x28 letters, so
`mrhand` composes handwritten *lines*: each letter is a sample from a
real writer, scaled to its kind (small, tall, below the line), placed on
a wobbly baseline with uneven gaps, sometimes touching. Accents and
`ñ` are drawn over the base letter; symbols EMNIST lacks come from
handwriting-like fonts. Those lines then go through the normal layout
and letter matching. `-d dir` saves example lines to look at.

```sh
./build/release/mmc_train -H build/data/emnist -n 3000000 -l 384,192 \
  -o build/release/hand.mrm          # half printed, half handwritten
```

**Cursive (`cursive.mrm`).** A whole text line is scaled to 32 pixels
high and read at once, so letters never need to be cut apart:

```
line -> 4 conv layers (16, 32, 64, 64 filters, 3x3, ReLU, max pool)
     -> one 128-number column every 4 pixels
     -> LSTM left-to-right + LSTM right-to-left (128 cells each)
     -> 137 classes (blank, space, ASCII, the extras) per column
     -> CTC: drop repeats and blanks -> text
```

`mmc_trainseq` learns from IAM lines (6,482 lines, 650 writers) plus
synthetic lines drawn with cursive fonts (which bring `ñ`, accents and
symbols IAM lacks). It uses every CPU core (`mrthread`), checks the IAM
validation writers after each pass and keeps the best model.

```sh
./build/release/mmc_trainseq -e 60 -y 40 -o build/release/cursive.mrm
./build/release/mmc_trainseq -t build/release/cursive.mrm  # IAM test
```

The gradients of `mrseq` were checked against numeric ones; repeat
that check after changing the network.

### 7.5 Real scans

Scanned forms are harder than drawn lines. The reader (not the trainers)
therefore also:

- erases table lines before looking for letters (`mrK_removelines`);
- for line models, splits each row at gaps wider than 2.5 x-heights and
  reads the pieces separately, so a form's label and value (often on
  slightly different baselines) are two clean lines (`CHUNKGAP`);
- leaves out text that is mostly not letters or digits, or that the
  letter model is unsure about (`JUNKCONF`): stamps, signatures, dust.

### 7.6 Printed line model (`print.mrm`)

The same CNN + LSTM + CTC network as `cursive.mrm`, trained only on
drawn printed lines (`mmc_trainseq -P`). It reads a whole line at once,
like tesseract 4, so it needs no letter cutting and uses the neighbors
of each letter (`l`/`I`, `rn`/`m`, `o`/`O` get easier).

- Text: sentences from the Gutenberg books (English and Spanish), with
  prices (`₱1,250.00`), dates, times, phone numbers, emails, `25°C`,
  `7 × 8`, quotes and symbols mixed in, and some ALL CAPS lines.
- Fonts: the Windows fonts plus the Google Fonts sans, serif, mono and
  display kinds; every 10th text font (decorative ones never) is held
  back, and the validation and `-t` test lines are drawn only with those
  unseen fonts.
- Each line has a random size (10 to 60 pixels), stretch, contrast,
  light-on-dark, noise, blur, thicker or thinner ink and a slight skew;
  new lines every epoch. Lines that black/white turns into noise or
  fragments are dropped (`readable`): they teach nothing.
- **Grayscale input** (model file version 2): black/white is only used to
  find the line and its ink; the network gets the gray pixels near that
  ink, scaled by the line's own paper and ink levels (`mrQ_normalize`).
  Thin strokes and faint text survive this, unlike black/white. Version 1
  (black/white) models are still read.
- `-t model -D dir` saves the test lines read badly as `.pgm` images:
  look at them before changing the network; most bad lines so far were
  bad data, not a weak network.

```sh
./build/release/mmc_trainseq -P -e 40 -L 8000 -o build/release/print.mrm
./build/release/mmc_trainseq -P -t build/release/print.mrm   # unseen fonts
./build/release/mmc_trainseq -P -R build/release/print.mrm -e 15 \
  -r 0.00015 -o build/release/print.mrm  # go on training a saved model
./build/release/mmc_reader -l print page.png
```

The handwriting model also uses the books (instead of only IAM's own
sentences) and the Google handwriting fonts for its drawn lines.

### 7.7 Known limits

- Made for clean printed text: screenshots and good scans. Phone photos
  need better thresholding, deskew and perspective fixes first.
- Touching letters are split by trying cuts at thin columns; letters
  that overlap a lot (heavy kerning, bold at small sizes) can still be
  misread.
- Very small text (x-height under about 8 pixels): word gaps and letter
  gaps overlap, so some spaces are lost or added.
- Some shapes are the same in many fonts: `l` `I` `|` `1`, `O` `0`,
  and in ALL CAPS lines `c`/`C`, `o`/`O`, `s`/`S`. A word list or
  language model would fix most of these later.
- One column only: side-by-side columns are read as one line.
- Handwriting: lines must be roughly straight; phone photos with uneven
  light or slanted pages need cleanup first. EMNIST and IAM have no
  `ñ` or accents, so those come only from drawn marks and fonts and are
  read less well in real handwriting.

---

## 8. Submitting changes

1. One logical change per commit/PR. Do not mix refactoring with features.
2. Commit message: short summary line (max 72 chars), blank line, then
   explanation of *why*.
3. Make sure: no `class`, no tab characters, no trailing whitespace, `./build.sh`
   builds with C++11 and no warnings, no new folders besides `vendor/<lib>`.
4. Update this file if you add a module prefix, a config switch, or a
   vendored library.

Quick tab check:

```sh
grep -nP "\t" *.cpp *.h build.sh
```

It must print nothing.
