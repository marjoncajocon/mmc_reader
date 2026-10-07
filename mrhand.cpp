/*
** mrhand.cpp
** Handwritten letters (EMNIST) drawn into text lines, for training
** See Copyright Notice in mr.h
*/

#define mrhand_cpp
#define MR_CORE

#include "mrhand.h"
#include "mrfile.h"
#include "mrmem.h"
#include "mrstate.h"

#include <math.h>
#include <stdio.h>
#include <string.h>


#define SIDE  28  /* EMNIST images are 28x28 */
#define AREA  (SIDE * SIDE)
#define NHAND  62  /* 'byclass': 0-9 A-Z a-z */

/* kinds of accent drawn over a handwritten base letter */
#define ACUTE  1
#define TILDE  2
#define DOTS  3


struct mr_Hand {
  mr_byte *pix;  /* AREA bytes per sample, ink = 255, rows top down */
  mr_byte *box;  /* x0 y0 x1 y1 of the ink of each sample */
  size_t n;
  size_t first[NHAND + 1];  /* samples of class c: first[c] .. first[c+1] */
};


/*
** {======================================================
** Loading
** =======================================================
*/

static int cpclass (uint32_t cp) {
  if (cp >= '0' && cp <= '9') return mr_cast(int, cp - '0');
  if (cp >= 'A' && cp <= 'Z') return mr_cast(int, cp - 'A') + 10;
  if (cp >= 'a' && cp <= 'z') return mr_cast(int, cp - 'a') + 36;
  return -1;
}


static int readu32 (FILE *f, uint32_t *v) {
  mr_byte b[4];
  if (fread(b, 1, 4, f) != 4) return 0;
  *v = mr_cast(uint32_t, b[0]) << 24 | mr_cast(uint32_t, b[1]) << 16 |
       mr_cast(uint32_t, b[2]) << 8 | mr_cast(uint32_t, b[3]);
  return 1;
}


static FILE *openidx (mr_State *R, const char *dir, const char *split,
                      const char *kind, uint32_t magic, uint32_t *count) {
  char path[MR_PATHSIZE];
  uint32_t m;
  FILE *f;
  snprintf(path, sizeof(path), "%s/emnist-byclass-%s-%s", dir, split, kind);
  f = mrF_open(path, "rb");
  if (f == NULL) {
    mrS_error(R, MR_ERRFILE, "cannot open '%s' (run ./getdata.sh emnist)",
              path);
    return NULL;
  }
  if (!readu32(f, &m) || m != magic || !readu32(f, count)) {
    fclose(f);
    mrS_error(R, MR_ERRFORMAT, "'%s' is not an EMNIST file", path);
    return NULL;
  }
  return f;
}


/* store sample 'src' (transposed in the file) with its ink box */
static void putsample (mr_Hand *h, size_t at, const mr_byte *src) {
  mr_byte *dst = h->pix + at * AREA, *b = h->box + at * 4;
  int x, y, x0 = SIDE, y0 = SIDE, x1 = -1, y1 = -1;
  for (y = 0; y < SIDE; y++) {
    for (x = 0; x < SIDE; x++) {
      mr_byte v = src[x * SIDE + y];
      dst[y * SIDE + x] = v;
      if (v >= 64) {
        if (x < x0) x0 = x;
        if (y < y0) y0 = y;
        if (x > x1) x1 = x;
        if (y > y1) y1 = y;
      }
    }
  }
  if (x1 < 0) { x0 = y0 = 0; x1 = y1 = SIDE - 1; }
  b[0] = mr_cast(mr_byte, x0);
  b[1] = mr_cast(mr_byte, y0);
  b[2] = mr_cast(mr_byte, x1 + 1);
  b[3] = mr_cast(mr_byte, y1 + 1);
}


int mrH_load (mr_State *R, const char *dir, const char *split, int maxper,
              mr_Hand **out) {
  FILE *fl = NULL, *fi = NULL;
  mr_byte *labels = NULL, img[AREA];
  size_t fill[NHAND], i;
  uint32_t nl = 0, ni = 0, rows, cols;
  mr_Hand *h = NULL;
  int c, status = MR_OK;
  *out = NULL;
  fl = openidx(R, dir, split, "labels-idx1-ubyte", 0x801, &nl);
  if (fl == NULL) return MR_ERRFILE;
  fi = openidx(R, dir, split, "images-idx3-ubyte", 0x803, &ni);
  if (fi == NULL) {
    status = MR_ERRFILE;
    goto done;
  }
  if (!readu32(fi, &rows) || !readu32(fi, &cols) || rows != SIDE ||
      cols != SIDE || ni != nl) {
    status = mrS_error(R, MR_ERRFORMAT, "EMNIST files do not match");
    goto done;
  }
  labels = mrM_newarray(R, nl, mr_byte);
  if (labels == NULL) goto nomem;
  if (fread(labels, 1, nl, fl) != nl) {
    status = mrS_error(R, MR_ERRFORMAT, "EMNIST labels are cut short");
    goto done;
  }
  h = mrM_new(R, mr_Hand);
  if (h == NULL) goto nomem;
  memset(h, 0, sizeof(*h));
  /* how many samples of each class to keep */
  memset(fill, 0, sizeof(fill));
  for (i = 0; i < nl; i++)
    if (labels[i] < NHAND && fill[labels[i]] < mr_cast(size_t, maxper))
      fill[labels[i]]++;
  h->first[0] = 0;
  for (c = 0; c < NHAND; c++) h->first[c + 1] = h->first[c] + fill[c];
  h->n = h->first[NHAND];
  h->pix = mrM_newarray(R, h->n * AREA, mr_byte);
  h->box = mrM_newarray(R, h->n * 4, mr_byte);
  if (h->pix == NULL || h->box == NULL) goto nomem;
  memset(fill, 0, sizeof(fill));
  for (i = 0; i < nl; i++) {
    int l = labels[i];
    if (fread(img, 1, AREA, fi) != AREA) {
      status = mrS_error(R, MR_ERRFORMAT, "EMNIST images are cut short");
      goto done;
    }
    if (l >= NHAND || h->first[l] + fill[l] >= h->first[l + 1]) continue;
    putsample(h, h->first[l] + fill[l], img);
    fill[l]++;
  }
  *out = h;
  h = NULL;
  goto done;
 nomem:
  status = MR_ERRMEM;
 done:
  if (labels != NULL) mrM_freearray(R, labels, nl);
  if (fl != NULL) fclose(fl);
  if (fi != NULL) fclose(fi);
  mrH_free(R, h);
  return status;
}


void mrH_free (mr_State *R, mr_Hand *h) {
  if (h == NULL) return;
  if (h->pix != NULL) mrM_freearray(R, h->pix, h->n * AREA);
  if (h->box != NULL) mrM_freearray(R, h->box, h->n * 4);
  mrM_delete(R, h);
}


size_t mrH_count (const mr_Hand *h) {
  return h->n;
}

/* }====================================================== */


/*
** {======================================================
** Letters: which EMNIST class, which accent, how tall
** =======================================================
*/

/* split an accented letter into its base letter and accent */
static uint32_t baseof (uint32_t cp, int *accent) {
  *accent = 0;
  switch (cp) {
    case 0xE1: *accent = ACUTE; return 'a';
    case 0xE9: *accent = ACUTE; return 'e';
    case 0xED: *accent = ACUTE; return 0x131;  /* dotless i */
    case 0xF3: *accent = ACUTE; return 'o';
    case 0xFA: *accent = ACUTE; return 'u';
    case 0xFC: *accent = DOTS; return 'u';
    case 0xF1: *accent = TILDE; return 'n';
    case 0xC1: *accent = ACUTE; return 'A';
    case 0xC9: *accent = ACUTE; return 'E';
    case 0xCD: *accent = ACUTE; return 'I';
    case 0xD3: *accent = ACUTE; return 'O';
    case 0xDA: *accent = ACUTE; return 'U';
    case 0xDC: *accent = DOTS; return 'U';
    case 0xD1: *accent = TILDE; return 'N';
    default: return cp;
  }
}


int mrH_has (const mr_Hand *h, uint32_t cp) {
  int accent;
  uint32_t base = baseof(cp, &accent);
  int c = (base == 0x131) ? cpclass('l') : cpclass(base);
  return c >= 0 && h->first[c + 1] > h->first[c];
}


/* top and bottom of a letter, in x-heights above the baseline */
static void extent (uint32_t base, float *top, float *bottom) {
  *top = 1.0f;
  *bottom = 0.0f;
  if ((base >= 'A' && base <= 'Z') || (base >= '0' && base <= '9'))
    *top = 1.45f;
  else if (strchr("bdfhklt", mr_cast(int, base)) != NULL)
    *top = 1.5f;
  else if (base == 'i')
    *top = 1.3f;
  else if (strchr("gpqy", mr_cast(int, base)) != NULL)
    *bottom = -0.45f;
  else if (base == 'j') {
    *top = 1.3f;
    *bottom = -0.45f;
  }
}

/* }====================================================== */


/*
** {======================================================
** Drawing
** =======================================================
*/

typedef struct Canvas {
  mr_Image *img;
  int ink, paper;
  mr_Box inkbox;  /* ink drawn for the current letter */
  int any;
} Canvas;


/* put pen coverage 'a' (0..255) at (x, y) */
static void blend (Canvas *cv, int x, int y, int a) {
  mr_Image *img = cv->img;
  mr_byte *p;
  int v;
  if (a <= 0 || x < 0 || y < 0 || x >= img->width || y >= img->height)
    return;
  if (a > 255) a = 255;
  p = &img->pixels[mr_cast(size_t, y) * img->width + x];
  v = cv->paper + (cv->ink - cv->paper) * a / 255;
  if ((cv->ink < cv->paper) ? (v < *p) : (v > *p))
    *p = mr_cast(mr_byte, v);
  if (a >= 96) {
    if (!cv->any) {
      cv->inkbox.x0 = cv->inkbox.x1 = x;
      cv->inkbox.y0 = cv->inkbox.y1 = y;
      cv->any = 1;
    }
    if (x < cv->inkbox.x0) cv->inkbox.x0 = x;
    if (y < cv->inkbox.y0) cv->inkbox.y0 = y;
    if (x + 1 > cv->inkbox.x1) cv->inkbox.x1 = x + 1;
    if (y + 1 > cv->inkbox.y1) cv->inkbox.y1 = y + 1;
  }
}


/* bilinear sample of an EMNIST image at (fx, fy) */
static float sample (const mr_byte *pix, float fx, float fy) {
  int x0 = mr_cast(int, floorf(fx)), y0 = mr_cast(int, floorf(fy));
  float tx = fx - x0, ty = fy - y0, v = 0;
  int dx, dy;
  for (dy = 0; dy <= 1; dy++) {
    for (dx = 0; dx <= 1; dx++) {
      int x = x0 + dx, y = y0 + dy;
      float w = (dx ? tx : 1 - tx) * (dy ? ty : 1 - ty);
      if (x >= 0 && y >= 0 && x < SIDE && y < SIDE)
        v += w * pix[y * SIDE + x];
    }
  }
  return v;
}


/* draw sample 's' scaled into the box (x, y, w, h) */
static void drawsample (Canvas *cv, const mr_Hand *h, size_t s, float x,
                        float y, float w, float hh) {
  const mr_byte *pix = h->pix + s * AREA, *b = h->box + s * 4;
  float bw = mr_cast(float, b[2] - b[0]), bh = mr_cast(float, b[3] - b[1]);
  int px, py;
  int x0 = mr_cast(int, floorf(x)), y0 = mr_cast(int, floorf(y));
  int x1 = mr_cast(int, ceilf(x + w)), y1 = mr_cast(int, ceilf(y + hh));
  for (py = y0; py < y1; py++) {
    for (px = x0; px < x1; px++) {
      float fx = b[0] + (px + 0.5f - x) * bw / w - 0.5f;
      float fy = b[1] + (py + 0.5f - y) * bh / hh - 0.5f;
      blend(cv, px, py, mr_cast(int, sample(pix, fx, fy)));
    }
  }
}


/* a round pen stroke from (ax, ay) to (bx, by) of width 'th' */
static void stroke (Canvas *cv, float ax, float ay, float bx, float by,
                    float th) {
  float r = th / 2, dx = bx - ax, dy = by - ay;
  float len2 = dx * dx + dy * dy;
  int x0 = mr_cast(int, floorf((ax < bx ? ax : bx) - r - 1));
  int x1 = mr_cast(int, ceilf((ax > bx ? ax : bx) + r + 1));
  int y0 = mr_cast(int, floorf((ay < by ? ay : by) - r - 1));
  int y1 = mr_cast(int, ceilf((ay > by ? ay : by) + r + 1));
  int x, y;
  for (y = y0; y <= y1; y++) {
    for (x = x0; x <= x1; x++) {
      float px = x + 0.5f - ax, py = y + 0.5f - ay, t = 0, ex, ey, d;
      if (len2 > 0) {
        t = (px * dx + py * dy) / len2;
        if (t < 0) t = 0;
        if (t > 1) t = 1;
      }
      ex = px - t * dx;
      ey = py - t * dy;
      d = sqrtf(ex * ex + ey * ey);
      blend(cv, x, y, mr_cast(int, (r + 0.5f - d) * 255));
    }
  }
}


/* the accent over a letter whose ink is 'b' */
static void drawaccent (Canvas *cv, mr_Rand *rng, int accent,
                        const mr_Box *b, float xh) {
  float cx = (b->x0 + b->x1) / 2.0f + mrR_range(rng, -0.08f, 0.08f) * xh;
  float top = b->y0 - mrR_range(rng, 0.12f, 0.25f) * xh;
  float th = xh * mrR_range(rng, 0.09f, 0.15f);
  if (th < 1.2f) th = 1.2f;
  switch (accent) {
    case ACUTE:
      stroke(cv, cx - 0.1f * xh, top, cx + 0.14f * xh, top - 0.3f * xh, th);
      break;
    case DOTS:
      stroke(cv, cx - 0.2f * xh, top - 0.1f * xh, cx - 0.2f * xh,
             top - 0.1f * xh, th * 1.4f);
      stroke(cv, cx + 0.2f * xh, top - 0.1f * xh, cx + 0.2f * xh,
             top - 0.1f * xh, th * 1.4f);
      break;
    case TILDE: {
      float w = (b->x1 - b->x0) * mrR_range(rng, 0.6f, 0.9f), amp = 0.1f * xh;
      float y = top - 0.12f * xh, prevx = cx - w / 2, prevy = y;
      int k;
      for (k = 1; k <= 8; k++) {
        float t = k / 8.0f;
        float nx = cx - w / 2 + w * t;
        float ny = y - amp * sinf(t * 6.2831853f);
        stroke(cv, prevx, prevy, nx, ny, th);
        prevx = nx;
        prevy = ny;
      }
      break;
    }
    default: break;
  }
}


/* draw letter 'cp' with the font, its baseline at 'base'; returns width */
static int drawfont (mr_State *R, Canvas *cv, const mr_Font *font,
                     uint32_t cp, float x, int base, float xh, float *w) {
  mr_Draw d;
  mr_Image *g = NULL;
  mr_Box box;
  int gbase, gx, gy, status;
  float ratio = mrT_xheight(font);
  d.height = xh / ((ratio > 0.2f) ? ratio : 0.5f);
  d.stretch = 1;
  d.spacing = 0;
  d.ink = 0;
  d.paper = 255;
  d.pad = 2;
  status = mrT_drawline(R, font, &cp, 1, &d, &g, &box, &gbase);
  if (status != MR_OK) return status;
  *w = 0;
  if (box.x1 > box.x0) {
    int dx = mr_cast(int, x) - box.x0, dy = base - gbase;
    for (gy = box.y0; gy < box.y1; gy++)
      for (gx = box.x0; gx < box.x1; gx++)
        blend(cv, gx + dx, gy + dy,
              255 - g->pixels[mr_cast(size_t, gy) * g->width + gx]);
    *w = mr_cast(float, box.x1 - box.x0);
  }
  mrI_free(R, g);
  return MR_OK;
}


int mrH_drawline (mr_State *R, const mr_Hand *h, mr_Rand *rng,
                  const mr_Font *font, const uint32_t *cps, int n,
                  const mr_HandDraw *d, mr_Image **out, mr_Box *boxes) {
  Canvas cv;
  float xh = d->xheight, x, drift = 0;
  int i, w, hgt, base, status;
  *out = NULL;
  w = mr_cast(int, n * 2.2f * xh) + 2 * d->pad + 4;
  hgt = mr_cast(int, 3.0f * xh) + 2 * d->pad + 4;
  base = d->pad + 2 + mr_cast(int, 2.0f * xh);
  status = mrI_new(R, w, hgt, 1, &cv.img);
  if (status != MR_OK) return status;
  memset(cv.img->pixels, d->paper, mr_cast(size_t, w) * hgt);
  cv.ink = d->ink;
  cv.paper = d->paper;
  x = mr_cast(float, d->pad + 2);
  for (i = 0; i < n; i++) {
    uint32_t cp = cps[i];
    float gw = 0;
    int by;
    drift += mrR_normal(rng) * 0.03f * xh;
    if (drift > 0.12f * xh) drift = 0.12f * xh;
    if (drift < -0.12f * xh) drift = -0.12f * xh;
    by = base + mr_cast(int, drift);
    cv.any = 0;
    boxes[i].x0 = boxes[i].x1 = mr_cast(int, x);
    boxes[i].y0 = boxes[i].y1 = by;
    if (cp == ' ') {
      x += mrR_range(rng, 0.6f, 1.1f) * xh;
      continue;
    }
    if (mrH_has(h, cp)) {
      int accent, c;
      uint32_t b0 = baseof(cp, &accent);
      float top, bottom, gh, scale;
      size_t s;
      const mr_byte *bb;
      c = (b0 == 0x131) ? cpclass('l') : cpclass(b0);
      extent(b0 == 0x131 ? 'a' : b0, &top, &bottom);
      scale = mrR_range(rng, 0.92f, 1.08f);
      gh = (top - bottom) * xh * scale;
      s = h->first[c] + mr_cast(size_t, mrR_int(rng, mr_cast(int,
                                    h->first[c + 1] - h->first[c])));
      bb = h->box + s * 4;
      gw = gh * (bb[2] - bb[0]) / mr_cast(float, bb[3] - bb[1]) *
           mrR_range(rng, 0.85f, 1.15f);
      if (gw < 1.5f) gw = 1.5f;
      drawsample(&cv, h, s, x, by - top * xh * scale, gw, gh);
      if (accent != 0 && cv.any) {
        mr_Box body = cv.inkbox;
        drawaccent(&cv, rng, accent, &body, xh);
      }
    }
    else if (font != NULL) {
      status = drawfont(R, &cv, font, cp, x, by, xh, &gw);
      if (status != MR_OK) {
        mrI_free(R, cv.img);
        return status;
      }
    }
    if (cv.any) boxes[i] = cv.inkbox;
    x += gw;
    if (mrR_int(rng, 100) < d->touch)
      x -= mrR_range(rng, 0, 0.12f) * xh;  /* letters touch */
    else
      x += mrR_range(rng, 0.08f, 0.32f) * xh;
    if (x > w - d->pad - 2 * xh) {  /* no room left */
      i++;
      break;
    }
  }
  for (; i < n; i++) {  /* letters that did not fit: empty boxes */
    boxes[i].x0 = boxes[i].x1 = mr_cast(int, x);
    boxes[i].y0 = boxes[i].y1 = base;
  }
  *out = cv.img;
  return MR_OK;
}

/* }====================================================== */
