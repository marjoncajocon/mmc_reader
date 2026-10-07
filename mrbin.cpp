/*
** mrbin.cpp
** Black/white bitmaps: turn a gray image into ink and paper
** See Copyright Notice in mr.h
*/

#define mrbin_cpp
#define MR_CORE

#include "mrbin.h"
#include "mrmem.h"
#include "mrstate.h"

#include <string.h>


int mrK_new (mr_State *R, int width, int height, mr_Bitmap **out) {
  mr_Bitmap *bm;
  size_t size;
  *out = NULL;
  if (width <= 0 || height <= 0)
    return mrS_error(R, MR_ERRARG, "bad bitmap size %dx%d", width, height);
  if (mr_cast(size_t, width) > MR_MAXSIZE / mr_cast(size_t, height))
    return mrS_error(R, MR_ERRMEM, "bitmap too large");
  size = mr_cast(size_t, width) * mr_cast(size_t, height);
  bm = mrM_new(R, mr_Bitmap);
  if (bm == NULL) return MR_ERRMEM;
  bm->bits = mr_cast(mr_byte *, mrM_malloc(R, size));
  if (bm->bits == NULL) {
    mrM_delete(R, bm);
    return MR_ERRMEM;
  }
  memset(bm->bits, 0, size);
  bm->width = width;
  bm->height = height;
  *out = bm;
  return MR_OK;
}


/*
** Otsu's method: the gray level that best splits the histogram into two
** groups (dark and light). Pixels <= the result are dark.
*/
static int otsu (const size_t *hist, size_t total) {
  double sumall = 0, sumdark = 0, best = -1;
  size_t ndark = 0;
  int t, result = 127;
  for (t = 0; t < 256; t++)
    sumall += mr_cast(double, t) * mr_cast(double, hist[t]);
  for (t = 0; t < 255; t++) {
    size_t nlight;
    double mdark, mlight, between;
    ndark += hist[t];
    sumdark += mr_cast(double, t) * mr_cast(double, hist[t]);
    if (ndark == 0) continue;
    nlight = total - ndark;
    if (nlight == 0) break;
    mdark = sumdark / mr_cast(double, ndark);
    mlight = (sumall - sumdark) / mr_cast(double, nlight);
    between = mr_cast(double, ndark) * mr_cast(double, nlight) *
              (mdark - mlight) * (mdark - mlight);
    if (between > best) {
      best = between;
      result = t;
    }
  }
  return result;
}


/*
** Ink is the smaller of the two groups, so light text on a dark
** background (dark mode screenshots) works too.
*/
int mrK_fromgray (mr_State *R, const mr_Image *gray, mr_Bitmap **out) {
  size_t hist[256];
  size_t i, n, ndark = 0;
  mr_Bitmap *bm;
  int t, inkdark, status;
  mr_assert(gray->channels == 1);
  status = mrK_new(R, gray->width, gray->height, &bm);
  if (status != MR_OK) {
    *out = NULL;
    return status;
  }
  n = mr_cast(size_t, gray->width) * mr_cast(size_t, gray->height);
  memset(hist, 0, sizeof(hist));
  for (i = 0; i < n; i++)
    hist[gray->pixels[i]]++;
  t = otsu(hist, n);
  for (i = 0; i <= mr_cast(size_t, t); i++)
    ndark += hist[i];
  if (ndark == n) {  /* one flat color: no ink at all */
    *out = bm;
    return MR_OK;
  }
  inkdark = (ndark <= n - ndark);
  for (i = 0; i < n; i++) {
    int dark = (gray->pixels[i] <= t);
    bm->bits[i] = mr_cast(mr_byte, (dark == inkdark) ? 1 : 0);
  }
  *out = bm;
  return MR_OK;
}


/* shortest run that counts as a line: a part of the page size */
#define LINEPART  25
#define MINLINE  60


/* ink at (x, y) or one pixel beside it across the run direction */
static int nearink (const mr_Bitmap *bm, int x, int y, int horizontal) {
  int d;
  for (d = -1; d <= 1; d++) {
    int nx = horizontal ? x : x + d, ny = horizontal ? y + d : y;
    if (nx >= 0 && ny >= 0 && nx < bm->width && ny < bm->height &&
        mrK_at(bm, nx, ny))
      return 1;
  }
  return 0;
}


/*
** Mark in 'mask' the ink of runs at least 'minrun' long, row by row or
** column by column. A run may move one pixel sideways, so slightly
** tilted scanned lines still count as one run.
*/
static void markruns (const mr_Bitmap *bm, mr_byte *mask, int minrun,
                      int horizontal) {
  int w = bm->width, h = bm->height;
  int outer = horizontal ? h : w, inner = horizontal ? w : h;
  int a, b;
  for (a = 0; a < outer; a++) {
    int start = -1;
    for (b = 0; b <= inner; b++) {
      int x = horizontal ? b : a, y = horizontal ? a : b;
      int ink = (b < inner) && nearink(bm, x, y, horizontal);
      if (ink && start < 0) start = b;
      if (!ink && start >= 0) {
        if (b - start >= minrun) {
          int k;
          for (k = start; k < b; k++) {
            int mx = horizontal ? k : a, my = horizontal ? a : k;
            if (mrK_at(bm, mx, my)) mask[mr_cast(size_t, my) * w + mx] = 1;
          }
        }
        start = -1;
      }
    }
  }
}


int mrK_removelines (mr_State *R, mr_Bitmap *bm) {
  size_t i, n = mr_cast(size_t, bm->width) * mr_cast(size_t, bm->height);
  int minh = bm->width / LINEPART, minv = bm->height / LINEPART;
  mr_byte *mask = mr_cast(mr_byte *, mrM_malloc(R, n));
  if (mask == NULL) return MR_ERRMEM;
  memset(mask, 0, n);
  if (minh < MINLINE) minh = MINLINE;
  if (minv < MINLINE) minv = MINLINE;
  markruns(bm, mask, minh, 1);
  markruns(bm, mask, minv, 0);
  for (i = 0; i < n; i++)
    if (mask[i]) bm->bits[i] = 0;
  mrM_free(R, mask, n);
  return MR_OK;
}


void mrK_free (mr_State *R, mr_Bitmap *bm) {
  if (bm == NULL) return;
  mrM_free(R, bm->bits,
           mr_cast(size_t, bm->width) * mr_cast(size_t, bm->height));
  mrM_delete(R, bm);
}
