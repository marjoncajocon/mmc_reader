# Contributing to mmc_reader

mmc_reader reads text from images and PDF files (OCR). It is compiled as
**C++11** (`.cpp` files), but written the way a **C programmer** writes C:
plain `struct`s and functions, **no `class`**, like Lua's sources. C++ is
used only so we can link both C libraries (stb_image, leptonica, mupdf, ...)
and C++ libraries (tesseract, ...) from one codebase.

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
  mrconf.h        configuration and C/C++ compatibility macros
  mrlimits.h      internal basic types, limits, helper macros
  mrmem.cpp/.h    memory allocation           (mrM_)
  mrbuf.cpp/.h    growable byte/string buffer (mrB_)
  mrimage.cpp/.h  image loading and pixels    (mrI_)
  mrpdf.cpp/.h    PDF loading and rendering   (mrP_)
  mrocr.cpp/.h    OCR engine interface        (mrO_)
  mrtess.cpp/.h   wrapper for a C++-only vendor API (mrT_, see section 5)
  mrapi.cpp       implementation of mr.h      (mr_)
  mrinit.cpp      library setup / teardown
  mmc_reader.cpp  standalone command-line program (like lua.c)
  build.sh        build script (run from the mmc shell, see section 6)
  CONTRIBUTING.md
  vendor/
    README.md     list of vendored libraries, versions, licenses
    stb/
    tesseract/
    ...
```

Rules:

- Sources use `.cpp`, headers use `.h`. Never `.c`, `.cc`, `.cxx`, `.hpp`.
- One module = one `.cpp` + one `.h` pair, named `mr<module>.cpp` /
  `mr<module>.h`, all lowercase, no underscores, short (like `lstring.c`,
  `lmem.c` in Lua).
- The program entry point is `mmc_reader.cpp`. It uses only the public API in
  `mr.h`, exactly like `lua.c` uses only `lua.h` / `lauxlib.h`.
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

Internal module letters: `M` memory, `B` buffer, `I` image, `P` pdf,
`O` ocr, `T` tesseract wrapper. Pick a new unused capital letter for a new
module and add it to this table.

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

- Never call `malloc`/`free` directly outside `mrmem.cpp`. Use `mrM_malloc`,
  `mrM_realloc`, `mrM_free` (and helper macros like `mrM_new(R, T)`), which go
  through the allocator stored in `mr_State`, like Lua's `lua_Alloc`.
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
- Prefer C libraries, or C++ libraries that offer a C API
  (e.g. tesseract's `tesseract/capi.h`). Check the license is compatible
  before adding anything (note: mupdf is AGPL, pdfium is BSD).

### 5.1 C libraries

Include them directly from the module that uses them. Headers that lack
their own `extern "C"` guards must be wrapped:

```c
MR_BEGIN_DECLS
#include "vendor/stb/stb_image.h"
MR_END_DECLS
```

### 5.2 C++-only libraries

When a library has only a C++ API, write **one wrapper module** for it
(e.g. `mrtess.cpp` / `mrtess.h`). It is the **only** file that includes the
library's headers, and the only place where C++ features are allowed, and
only as much as the library's API forces (calling its methods, holding its
objects, catching its exceptions):

- `mrtess.h` exposes plain functions and an opaque struct
  (`typedef struct mr_Tess mr_Tess;`), so no library type leaks out.
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
./build.sh cross-release [T...]   # release, cross-compiled by zig
./build.sh cross-debug [T...]     # debug, cross-compiled by zig
./build.sh targets                # list the default cross targets
./build.sh clean                  # remove the whole build/ folder
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
`x86_64-windows-gnu`.

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
  them all.
- Nothing is ever written outside `build/`. Never commit `build/`.
- Debug-only code goes inside `#if defined(MR_DEBUG)`.
- Extra flags: `CFLAGS="-fsanitize=address" ./build.sh debug`, and libraries:
  `LIBS="vendor/x/libx.a" ./build.sh`. Add permanent libraries to `LIBS`
  inside `build.sh`.
- Keep `build.sh` to plain sh/bash features so `mmc --check build.sh`
  passes.

Before sending a change, run `./build.sh` and fix every warning.

---

## 7. Submitting changes

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
