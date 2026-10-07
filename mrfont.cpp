/*
** mrfont.cpp
** TrueType fonts: draw text lines to make training images
** See Copyright Notice in mr.h
*/

#define mrfont_cpp
#define MR_CORE

#include "mrfont.h"
#include "mrfile.h"
#include "mrmem.h"
#include "mrstate.h"

#include <math.h>
#include <string.h>

#define STB_TRUETYPE_IMPLEMENTATION
#define STBTT_STATIC
#include "stb/stb_truetype.h"


struct mr_Font {
  stbtt_fontinfo info;
  mr_byte *data;
  size_t size;
};


int mrT_load (mr_State *R, const char *path, int index, mr_Font **out) {
  FILE *file;
  mr_Font *f;
  long len;
  int offset;
  *out = NULL;
  file = mrF_open(path, "rb");
  if (file == NULL)
    return mrS_error(R, MR_ERRFILE, "cannot open font '%s'", path);
  if (fseek(file, 0, SEEK_END) != 0 || (len = ftell(file)) <= 0 ||
      fseek(file, 0, SEEK_SET) != 0) {
    fclose(file);
    return mrS_error(R, MR_ERRFILE, "cannot read font '%s'", path);
  }
  f = mrM_new(R, mr_Font);
  if (f == NULL) {
    fclose(file);
    return MR_ERRMEM;
  }
  f->size = mr_cast(size_t, len);
  f->data = mr_cast(mr_byte *, mrM_malloc(R, f->size));
  if (f->data == NULL) {
    fclose(file);
    mrM_delete(R, f);
    return MR_ERRMEM;
  }
  if (fread(f->data, 1, f->size, file) != f->size) {
    fclose(file);
    mrT_free(R, f);
    return mrS_error(R, MR_ERRFILE, "cannot read font '%s'", path);
  }
  fclose(file);
  offset = stbtt_GetFontOffsetForIndex(f->data, index);
  if (offset < 0 || !stbtt_InitFont(&f->info, f->data, offset)) {
    mrT_free(R, f);
    return mrS_error(R, MR_ERRFORMAT, "'%s' is not a TrueType font", path);
  }
  *out = f;
  return MR_OK;
}


void mrT_free (mr_State *R, mr_Font *f) {
  if (f == NULL) return;
  if (f->data != NULL) mrM_free(R, f->data, f->size);
  mrM_delete(R, f);
}


int mrT_hasglyph (const mr_Font *f, uint32_t cp) {
  return stbtt_FindGlyphIndex(&f->info, mr_cast(int, cp)) != 0;
}


float mrT_xheight (const mr_Font *f) {
  int asc, desc, gap, x0, y0, x1, y1;
  stbtt_GetFontVMetrics(&f->info, &asc, &desc, &gap);
  if (!stbtt_GetCodepointBox(&f->info, 'x', &x0, &y0, &x1, &y1) ||
      asc - desc <= 0)
    return 0.5f;
  return mr_cast(float, y1) / mr_cast(float, asc - desc);
}


int mrT_drawline (mr_State *R, const mr_Font *f, const uint32_t *cps,
                  int n, const mr_Draw *d, mr_Image **out, mr_Box *boxes,
                  int *baseline) {
  const stbtt_fontinfo *info = &f->info;
  mr_Image *img = NULL;
  float sy, sx, x, width = 0;
  int asc, desc, gap, i, w, h, base, status;
  int *glyphs = NULL;
  *out = NULL;
  stbtt_GetFontVMetrics(info, &asc, &desc, &gap);
  sy = d->height / mr_cast(float, asc - desc);
  sx = sy * d->stretch;
  glyphs = mrM_newarray(R, n + 1, int);
  if (glyphs == NULL) return MR_ERRMEM;
  /* measure */
  for (i = 0; i < n; i++) {
    int adv, lsb;
    glyphs[i] = stbtt_FindGlyphIndex(info, mr_cast(int, cps[i]));
    stbtt_GetGlyphHMetrics(info, glyphs[i], &adv, &lsb);
    width += adv * sx + d->spacing;
    if (i + 1 < n)
      width += sx * stbtt_GetGlyphKernAdvance(info, glyphs[i],
               stbtt_FindGlyphIndex(info, mr_cast(int, cps[i + 1])));
  }
  w = mr_cast(int, ceilf(width)) + 2 * d->pad + 2;
  h = mr_cast(int, ceilf(d->height)) + 2 * d->pad + 2;
  base = d->pad + 1 + mr_cast(int, floorf(asc * sy));
  status = mrI_new(R, w, h, 1, &img);
  if (status != MR_OK) goto done;
  memset(img->pixels, d->paper, mr_cast(size_t, w) * mr_cast(size_t, h));
  /* draw */
  x = mr_cast(float, d->pad + 1);
  for (i = 0; i < n; i++) {
    int adv, lsb, x0, y0, x1, y1, gx, gy, bw, bh;
    float fx = floorf(x);
    stbtt_GetGlyphHMetrics(info, glyphs[i], &adv, &lsb);
    stbtt_GetGlyphBitmapBoxSubpixel(info, glyphs[i], sx, sy, x - fx, 0,
                                    &x0, &y0, &x1, &y1);
    bw = x1 - x0;
    bh = y1 - y0;
    boxes[i].x0 = boxes[i].x1 = mr_cast(int, fx);
    boxes[i].y0 = boxes[i].y1 = base;
    if (bw > 0 && bh > 0 && cps[i] != ' ') {
      mr_byte *cov = mr_cast(mr_byte *,
                             mrM_malloc(R, mr_cast(size_t, bw) * bh));
      int bx0 = bw, by0 = bh, bx1 = -1, by1 = -1;
      if (cov == NULL) {
        status = MR_ERRMEM;
        goto done;
      }
      stbtt_MakeGlyphBitmapSubpixel(info, cov, bw, bh, bw, sx, sy, x - fx,
                                    0, glyphs[i]);
      for (gy = 0; gy < bh; gy++) {
        for (gx = 0; gx < bw; gx++) {
          int a = cov[gy * bw + gx];
          int px = mr_cast(int, fx) + x0 + gx, py = base + y0 + gy;
          mr_byte *p;
          int v;
          if (a == 0 || px < 0 || py < 0 || px >= w || py >= h) continue;
          p = &img->pixels[mr_cast(size_t, py) * w + px];
          /* blend toward the ink color, keep the strongest ink */
          v = d->paper + (d->ink - d->paper) * a / 255;
          if ((d->ink < d->paper) ? (v < *p) : (v > *p))
            *p = mr_cast(mr_byte, v);
          if (a >= 96) {  /* pixels likely to become ink */
            if (gx < bx0) bx0 = gx;
            if (gy < by0) by0 = gy;
            if (gx > bx1) bx1 = gx;
            if (gy > by1) by1 = gy;
          }
        }
      }
      mrM_free(R, cov, mr_cast(size_t, bw) * bh);
      if (bx1 >= 0) {
        boxes[i].x0 = mr_cast(int, fx) + x0 + bx0;
        boxes[i].x1 = mr_cast(int, fx) + x0 + bx1 + 1;
        boxes[i].y0 = base + y0 + by0;
        boxes[i].y1 = base + y0 + by1 + 1;
      }
    }
    x += adv * sx + d->spacing;
    if (i + 1 < n)
      x += sx * stbtt_GetGlyphKernAdvance(info, glyphs[i], glyphs[i + 1]);
  }
  *out = img;
  img = NULL;
  if (baseline != NULL) *baseline = base;
 done:
  if (img != NULL) mrI_free(R, img);
  mrM_freearray(R, glyphs, n + 1);
  return status;
}
