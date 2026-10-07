# Vendored libraries

Third-party code used by mmc_reader. One folder per library. Do not edit
these files; see CONTRIBUTING.md section 5.

| Folder | Library        | Version | Source                          | License                              | Used by        |
|--------|----------------|---------|---------------------------------|--------------------------------------|----------------|
| `stb/` | stb_image.h    | 2.30    | https://github.com/nothings/stb | MIT or public domain (`stb/LICENSE`) | `mrimage.cpp`  |
| `stb/` | stb_truetype.h | 1.26    | https://github.com/nothings/stb | MIT or public domain (`stb/LICENSE`) | `mrfont.cpp`   |

Both taken from stb commit `2c980bb`.

## Patches

None.

## Not vendored

- **OCR**: our own engine (CONTRIBUTING.md section 7), no tesseract.
- **PDF**: not supported for now; extract pages to images with another
  tool. `mrpdf.cpp` is a stub (`MR_USE_PDF`).
- **Fonts** used for training are not part of the project; `mmc_train`
  reads them from the system (`C:/Windows/Fonts`) or from paths you give.
