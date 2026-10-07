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


void mrK_free (mr_State *R, mr_Bitmap *bm) {
  if (bm == NULL) return;
  mrM_free(R, bm->bits,
           mr_cast(size_t, bm->width) * mr_cast(size_t, bm->height));
  mrM_delete(R, bm);
}
