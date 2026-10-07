# Vendored libraries

Third-party code used by mmc_reader. One folder per library. Do not edit
these files; see CONTRIBUTING.md section 5.

| Folder | Library            | Version | Source                                    | License                      | Used by       |
|--------|--------------------|---------|-------------------------------------------|------------------------------|---------------|
| `stb/` | stb_image.h        | 2.30 (commit 2c980bb) | https://github.com/nothings/stb | MIT or public domain (`stb/LICENSE`) | `mrimage.cpp` |

## Patches

None.

## Planned

| Purpose | Candidates | Notes |
|---------|------------|-------|
| OCR     | tesseract (+ leptonica) | Apache-2.0 / BSD-2. C++ with a C API (`capi.h`). Turn on with `MR_USE_TESSERACT`. |
| PDF     | pdfium, mupdf | pdfium: BSD-3. mupdf: AGPL-3 (makes the whole program AGPL). Turn on with `MR_USE_PDF`. |
