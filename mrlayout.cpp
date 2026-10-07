/*
** mrlayout.cpp
** Page layout: connected parts -> text lines -> letter segments
** See Copyright Notice in mr.h
*/

#define mrlayout_cpp
#define MR_CORE

#include "mrlayout.h"
#include "mrmem.h"
#include "mrstate.h"

#include <stdlib.h>
#include <string.h>


/* bands shorter than this part of the usual line height are marks
   (dots, tildes, accents) that belong to a nearby line */
#define SMALLBAND  0.45

/* bands taller than this many usual line heights are split */
#define TALLBAND  1.8

/* components taller than this many usual heights are not text */
#define TALLCOMP  5

/* first guess of the gap between words, in x-heights */
#define SPACEGUESS  0.5

/* line gaps bigger than this many usual line steps mean an empty line */
#define BLANKLINE  1.7


typedef struct Band {
  int y0, y1;
} Band;


/*
** {======================================================
** Small helpers
** =======================================================
*/

int mrL_xoverlap (const mr_Box *a, const mr_Box *b) {
  int lo = (a->x0 > b->x0) ? a->x0 : b->x0;
  int hi = (a->x1 < b->x1) ? a->x1 : b->x1;
  return hi - lo;
}


int mrL_yoverlap (const mr_Box *a, const mr_Box *b) {
  int lo = (a->y0 > b->y0) ? a->y0 : b->y0;
  int hi = (a->y1 < b->y1) ? a->y1 : b->y1;
  return hi - lo;
}


void mrL_join (mr_Box *a, const mr_Box *b) {
  if (b->x0 < a->x0) a->x0 = b->x0;
  if (b->y0 < a->y0) a->y0 = b->y0;
  if (b->x1 > a->x1) a->x1 = b->x1;
  if (b->y1 > a->y1) a->y1 = b->y1;
}


static int cmpint (const void *a, const void *b) {
  int x = *mr_cast(const int *, a);
  int y = *mr_cast(const int *, b);
  return (x > y) - (x < y);
}


static int cmpdouble (const void *a, const void *b) {
  double x = *mr_cast(const double *, a);
  double y = *mr_cast(const double *, b);
  return (x > y) - (x < y);
}


/* median of 'v' (sorted in place); 0 if empty */
static int median (int *v, size_t n) {
  if (n == 0) return 0;
  qsort(v, n, sizeof(int), cmpint);
  return v[n / 2];
}


/* components sorted by left edge */
typedef struct CompRef {
  int x0;
  int index;
} CompRef;


static int cmpcompref (const void *a, const void *b) {
  const CompRef *x = mr_cast(const CompRef *, a);
  const CompRef *y = mr_cast(const CompRef *, b);
  if (x->x0 != y->x0) return (x->x0 > y->x0) - (x->x0 < y->x0);
  return (x->index > y->index) - (x->index < y->index);
}

/* }====================================================== */


/*
** {======================================================
** Connected components (8-connected flood fill)
** =======================================================
*/

static int label (mr_State *R, const mr_Bitmap *bm, mr_Layout *lo) {
  size_t *stack = NULL;
  size_t nstack = 0, capstack = 0;
  size_t p, n;
  int w = bm->width, h = bm->height;
  int status = MR_OK;
  n = mr_cast(size_t, w) * mr_cast(size_t, h);
  lo->labels = mrM_newarray(R, n, int);
  if (lo->labels == NULL) return MR_ERRMEM;
  memset(lo->labels, 0, n * sizeof(int));
  for (p = 0; p < n; p++) {
    mr_Comp *c;
    int id;
    if (!bm->bits[p] || lo->labels[p] != 0) continue;
    c = mrM_growvector(R, lo->comps, lo->ncomps, &lo->capcomps, mr_Comp);
    if (c == NULL) goto nomem;
    lo->comps = c;
    id = mr_cast(int, lo->ncomps++);
    c = &lo->comps[id];
    c->box.x0 = w;
    c->box.y0 = h;
    c->box.x1 = 0;
    c->box.y1 = 0;
    c->area = 0;
    c->seg = -1;
    lo->labels[p] = id + 1;
    {
      size_t *ns = mrM_growvector(R, stack, 0, &capstack, size_t);
      if (ns == NULL) goto nomem;
      stack = ns;
    }
    stack[0] = p;
    nstack = 1;
    while (nstack > 0) {
      size_t q = stack[--nstack];
      int x = mr_cast(int, q % mr_cast(size_t, w));
      int y = mr_cast(int, q / mr_cast(size_t, w));
      int dx, dy;
      if (x < c->box.x0) c->box.x0 = x;
      if (y < c->box.y0) c->box.y0 = y;
      if (x + 1 > c->box.x1) c->box.x1 = x + 1;
      if (y + 1 > c->box.y1) c->box.y1 = y + 1;
      c->area++;
      for (dy = -1; dy <= 1; dy++) {
        for (dx = -1; dx <= 1; dx++) {
          int nx = x + dx, ny = y + dy;
          size_t r;
          size_t *ns;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          r = mr_cast(size_t, ny) * mr_cast(size_t, w) + mr_cast(size_t, nx);
          if (!bm->bits[r] || lo->labels[r] != 0) continue;
          lo->labels[r] = id + 1;
          ns = mrM_growvector(R, stack, nstack, &capstack, size_t);
          if (ns == NULL) goto nomem;
          stack = ns;
          stack[nstack++] = r;
        }
      }
    }
  }
  goto done;
 nomem:
  status = MR_ERRMEM;
 done:
  if (stack != NULL) mrM_freearray(R, stack, capstack);
  return status;
}

/* }====================================================== */


/*
** {======================================================
** Text lines
** =======================================================
*/

/*
** Mark components that are not letters: very tall ones (pictures,
** vertical rules) and long thin ones (horizontal rules, underlines).
** Returns the usual component height.
*/
static int filtercomps (mr_State *R, mr_Layout *lo, int *usualh) {
  int *hs;
  size_t i, n = 0;
  int medh;
  hs = mrM_newarray(R, lo->ncomps + 1, int);
  if (hs == NULL) return MR_ERRMEM;
  for (i = 0; i < lo->ncomps; i++) {
    mr_Comp *c = &lo->comps[i];
    if (c->area >= 3) hs[n++] = c->box.y1 - c->box.y0;
  }
  medh = median(hs, n);
  mrM_freearray(R, hs, lo->ncomps + 1);
  if (medh < 1) medh = 1;
  for (i = 0; i < lo->ncomps; i++) {
    mr_Comp *c = &lo->comps[i];
    int ch = c->box.y1 - c->box.y0;
    int cw = c->box.x1 - c->box.x0;
    if (ch > TALLCOMP * medh || (cw > 15 * medh && ch * 2 < medh))
      c->seg = -2;  /* ignored */
    else if (medh >= 12 && c->area * 40 < medh * medh)
      c->seg = -2;  /* scanner dust: smaller than any '.' at this size */
  }
  *usualh = medh;
  return MR_OK;
}


static int pushband (mr_State *R, Band **bands, size_t *nb, size_t *cap,
                     int y0, int y1) {
  Band *b = mrM_growvector(R, *bands, *nb, cap, Band);
  if (b == NULL) return MR_ERRMEM;
  *bands = b;
  b[*nb].y0 = y0;
  b[*nb].y1 = y1;
  (*nb)++;
  return MR_OK;
}


/*
** Split band 'b' at its weakest row when it holds two text lines whose
** descenders and ascenders touch. 'cov' counts components per row.
*/
static int splitrow (const int *cov, const Band *b) {
  int h = b->y1 - b->y0;
  int y, besty = -1, best = 0, top = 0;
  for (y = b->y0; y < b->y1; y++)
    if (cov[y] > top) top = cov[y];
  for (y = b->y0 + h * 3 / 10; y < b->y1 - h * 3 / 10; y++) {
    if (besty < 0 || cov[y] < best) {
      best = cov[y];
      besty = y;
    }
  }
  if (besty < 0 || best * 4 > top) return -1;
  return besty;
}


static int findbands (mr_State *R, mr_Layout *lo, Band **out,
                      size_t *nout, size_t *capout) {
  int *cov = NULL, *hs = NULL;
  Band *bands = NULL, *merged = NULL;
  size_t nb = 0, capb = 0, nm = 0, capm = 0, i;
  int y, start = -1, medb, status = MR_OK, rounds;
  size_t ncov = mr_cast(size_t, lo->height) + 1;
  *out = NULL;
  *nout = 0;
  *capout = 0;
  cov = mrM_newarray(R, ncov, int);
  if (cov == NULL) return MR_ERRMEM;
  memset(cov, 0, ncov * sizeof(int));
  for (i = 0; i < lo->ncomps; i++) {
    mr_Comp *c = &lo->comps[i];
    if (c->seg == -2) continue;
    for (y = c->box.y0; y < c->box.y1; y++) cov[y]++;
  }
  for (y = 0; y <= lo->height; y++) {
    int on = (y < lo->height && cov[y] > 0);
    if (on && start < 0) start = y;
    if (!on && start >= 0) {
      status = pushband(R, &bands, &nb, &capb, start, y);
      if (status != MR_OK) goto done;
      start = -1;
    }
  }
  /* split bands that hold more than one line */
  for (rounds = 0; rounds < 8 && nb > 0; rounds++) {
    int split = 0;
    hs = mrM_newarray(R, nb, int);
    if (hs == NULL) goto nomem;
    for (i = 0; i < nb; i++) hs[i] = bands[i].y1 - bands[i].y0;
    medb = median(hs, nb);
    mrM_freearray(R, hs, nb);
    hs = NULL;
    nm = 0;
    for (i = 0; i < nb; i++) {
      Band b = bands[i];
      int cut = -1;
      if (b.y1 - b.y0 > TALLBAND * medb) cut = splitrow(cov, &b);
      if (cut > b.y0 && cut < b.y1) {
        status = pushband(R, &merged, &nm, &capm, b.y0, cut);
        if (status == MR_OK)
          status = pushband(R, &merged, &nm, &capm, cut, b.y1);
        split = 1;
      }
      else
        status = pushband(R, &merged, &nm, &capm, b.y0, b.y1);
      if (status != MR_OK) goto done;
    }
    {  /* swap */
      Band *t = bands;
      size_t tc = capb;
      bands = merged;
      capb = capm;
      nb = nm;
      merged = t;
      capm = tc;
    }
    if (!split) break;
  }
  /* join small bands (dots, accents) to the nearest line */
  if (nb > 1) {
    hs = mrM_newarray(R, nb, int);
    if (hs == NULL) goto nomem;
    for (i = 0; i < nb; i++) hs[i] = bands[i].y1 - bands[i].y0;
    medb = median(hs, nb);
    mrM_freearray(R, hs, nb);
    hs = NULL;
    i = 0;
    while (i < nb && nb > 1) {
      int bh = bands[i].y1 - bands[i].y0;
      int gapup = (i > 0) ? bands[i].y0 - bands[i - 1].y1 : -1;
      int gapdown = (i + 1 < nb) ? bands[i + 1].y0 - bands[i].y1 : -1;
      size_t into;
      if (bh >= SMALLBAND * medb) {
        i++;
        continue;
      }
      if (gapdown >= 0 && (gapup < 0 || gapdown <= gapup)) into = i + 1;
      else into = i - 1;
      if ((into > i ? gapdown : gapup) > medb) {  /* too far: own line */
        i++;
        continue;
      }
      if (bands[i].y0 < bands[into].y0) bands[into].y0 = bands[i].y0;
      if (bands[i].y1 > bands[into].y1) bands[into].y1 = bands[i].y1;
      memmove(&bands[i], &bands[i + 1], (nb - i - 1) * sizeof(Band));
      nb--;
      if (into < i) i--;  /* look again at the grown band */
    }
  }
  *out = bands;
  *nout = nb;
  *capout = capb;
  bands = NULL;
  goto done;
 nomem:
  status = MR_ERRMEM;
 done:
  mrM_freearray(R, cov, ncov);
  if (bands != NULL) mrM_freearray(R, bands, capb);
  if (merged != NULL) mrM_freearray(R, merged, capm);
  return status;
}

/* }====================================================== */


/*
** {======================================================
** Segments and line metrics
** =======================================================
*/

static int newseg (mr_State *R, mr_Layout *lo, const mr_Box *box) {
  mr_Seg *s = mrM_growvector(R, lo->segs, lo->nsegs, &lo->capsegs, mr_Seg);
  if (s == NULL) return MR_ERRMEM;
  lo->segs = s;
  s[lo->nsegs].box = *box;
  s[lo->nsegs].space = 0;
  lo->nsegs++;
  return MR_OK;
}


/*
** Two parts of one letter stand on top of each other: the dot of 'i' and
** 'j', the tilde of 'ñ', the dots of ':' ';' '!' '?', the bars of '='.
*/
static int stacked (const mr_Box *a, const mr_Box *b) {
  int wa = a->x1 - a->x0, wb = b->x1 - b->x0;
  int minw = (wa < wb) ? wa : wb;
  return mrL_yoverlap(a, b) <= 0 && mrL_xoverlap(a, b) * 2 >= minw;
}


static int buildsegs (mr_State *R, mr_Layout *lo, const CompRef *refs,
                      size_t n, size_t first) {
  size_t i;
  for (i = 0; i < n; i++) {
    mr_Comp *c = &lo->comps[refs[i].index];
    size_t k, best = 0;
    int found = 0, bestov = 0;
    /* look back at the last few segments of this line */
    for (k = lo->nsegs; k > first && k + 3 > lo->nsegs; k--) {
      mr_Seg *s = &lo->segs[k - 1];
      int ov = mrL_xoverlap(&s->box, &c->box);
      if (stacked(&s->box, &c->box) && (!found || ov > bestov)) {
        found = 1;
        best = k - 1;
        bestov = ov;
      }
    }
    if (found) {
      mrL_join(&lo->segs[best].box, &c->box);
      c->seg = mr_cast(int, best);
    }
    else {
      int status = newseg(R, lo, &c->box);
      if (status != MR_OK) return status;
      c->seg = mr_cast(int, lo->nsegs - 1);
    }
  }
  return MR_OK;
}


/*
** The x-height from the heights 'v' of letters on the baseline. Heights
** come in groups: small letters, then capitals and tall letters, then
** accented capitals. The x-height is the lowest group that holds at
** least a fifth of the letters (ALL CAPS lines give the cap height).
*/
static int xheight (int *v, size_t m) {
  size_t i, j, need = (m + 4) / 5;
  if (m == 0) return 0;
  if (need < 2) need = (m < 2) ? m : 2;
  qsort(v, m, sizeof(int), cmpint);
  for (i = 0; i < m; i++) {
    /* letters within 10% (at least 1 pixel) above v[i] */
    int top = v[i] + ((v[i] / 10 > 1) ? v[i] / 10 : 1);
    for (j = i; j < m && v[j] <= top; j++) {}
    if (j - i >= need) return v[(i + j - 1) / 2];  /* middle of group */
  }
  return v[m / 2];
}


static int linemetrics (mr_State *R, mr_Layout *lo, mr_Line *ln) {
  int *v;
  size_t i, n = ln->count, m = 0;
  int medh, base, xh;
  if (n == 0) {
    ln->baseline = ln->box.y1;
    ln->xheight = 1;
    return MR_OK;
  }
  v = mrM_newarray(R, n, int);
  if (v == NULL) return MR_ERRMEM;
  for (i = 0; i < n; i++) {
    mr_Box *b = &lo->segs[ln->first + i].box;
    v[i] = b->y1 - b->y0;
  }
  medh = median(v, n);
  for (i = 0; i < n; i++) v[i] = lo->segs[ln->first + i].box.y1;
  base = median(v, n);
  /* heights of letters that sit on the baseline (not marks like '.') */
  for (i = 0; i < n; i++) {
    mr_Box *b = &lo->segs[ln->first + i].box;
    int d = b->y1 - base;
    if (d < 0) d = -d;
    if (d * 6 <= medh + 1 && (base - b->y0) * 5 >= medh * 2)
      v[m++] = base - b->y0;
  }
  xh = (m > 0) ? xheight(v, m) : medh;
  mrM_freearray(R, v, n);
  ln->baseline = base;
  ln->xheight = (xh > 1) ? xh : 1;
  return MR_OK;
}


/*
** Mark word gaps. The gaps of a line fall into two groups, between
** letters and between words; find them with 2-means.
*/
static void findspaces (mr_Layout *lo, mr_Line *ln) {
  double lo_c = 0.15, hi_c = 0.9, cut = SPACEGUESS;
  double xh = mr_cast(double, ln->xheight);
  size_t i;
  int it;
  if (ln->count < 2) return;
  for (it = 0; it < 10; it++) {
    double slo = 0, shi = 0;
    size_t nlo = 0, nhi = 0;
    for (i = 1; i < ln->count; i++) {
      mr_Seg *a = &lo->segs[ln->first + i - 1];
      mr_Seg *b = &lo->segs[ln->first + i];
      double g = (b->box.x0 - a->box.x1) / xh;
      if (g > 1.5) g = 1.5;  /* column gaps must not pull the cut up */
      if (g < cut) { slo += g; nlo++; }
      else { shi += g; nhi++; }
    }
    if (nlo > 0) lo_c = slo / mr_cast(double, nlo);
    if (nhi > 0) hi_c = shi / mr_cast(double, nhi);
    cut = (lo_c + hi_c) / 2;
    if (cut < 0.35) cut = 0.35;
    if (cut > 0.9) cut = 0.9;
  }
  for (i = 1; i < ln->count; i++) {
    mr_Seg *a = &lo->segs[ln->first + i - 1];
    mr_Seg *b = &lo->segs[ln->first + i];
    if ((b->box.x0 - a->box.x1) / xh >= cut) b->space = 1;
  }
}


static int markblanks (mr_State *R, mr_Layout *lo) {
  double *steps;
  size_t i, n = lo->nlines;
  double step;
  if (n < 3) return MR_OK;
  steps = mrM_newarray(R, n - 1, double);
  if (steps == NULL) return MR_ERRMEM;
  for (i = 1; i < n; i++)
    steps[i - 1] = lo->lines[i].baseline - lo->lines[i - 1].baseline;
  qsort(steps, n - 1, sizeof(double), cmpdouble);
  step = steps[(n - 1) / 2];
  mrM_freearray(R, steps, n - 1);
  for (i = 1; i < n; i++) {
    double d = lo->lines[i].baseline - lo->lines[i - 1].baseline;
    if (d > BLANKLINE * step) lo->lines[i].blank = 1;
  }
  return MR_OK;
}

/* }====================================================== */


void mrL_init (mr_Layout *lo) {
  memset(lo, 0, sizeof(*lo));
}


int mrL_analyze (mr_State *R, const mr_Bitmap *bm, mr_Layout *lo) {
  Band *bands = NULL;
  CompRef *refs = NULL;
  size_t nb = 0, capb = 0, nrefs = 0, b, i;
  int usualh, status;
  lo->width = bm->width;
  lo->height = bm->height;
  status = label(R, bm, lo);
  if (status != MR_OK) goto done;
  if (lo->ncomps == 0) goto done;
  status = filtercomps(R, lo, &usualh);
  if (status != MR_OK) goto done;
  status = findbands(R, lo, &bands, &nb, &capb);
  if (status != MR_OK || nb == 0) goto done;
  refs = mrM_newarray(R, lo->ncomps, CompRef);
  if (refs == NULL) {
    status = MR_ERRMEM;
    goto done;
  }
  for (b = 0; b < nb; b++) {
    mr_Line *ln;
    nrefs = 0;
    /* components whose middle is in this band (or nearest to it) */
    for (i = 0; i < lo->ncomps; i++) {
      mr_Comp *c = &lo->comps[i];
      int mid = (c->box.y0 + c->box.y1) / 2;
      int inb = (mid >= bands[b].y0 && mid < bands[b].y1);
      if (c->seg != -1) continue;  /* ignored or already placed */
      if (!inb) {
        /* outside every band: take the band it overlaps */
        size_t k;
        int inany = 0;
        for (k = 0; k < nb && !inany; k++)
          inany = (mid >= bands[k].y0 && mid < bands[k].y1);
        if (inany || c->box.y1 <= bands[b].y0 || c->box.y0 >= bands[b].y1)
          continue;
      }
      refs[nrefs].x0 = c->box.x0;
      refs[nrefs].index = mr_cast(int, i);
      nrefs++;
    }
    if (nrefs == 0) continue;
    qsort(refs, nrefs, sizeof(CompRef), cmpcompref);
    ln = mrM_growvector(R, lo->lines, lo->nlines, &lo->caplines, mr_Line);
    if (ln == NULL) {
      status = MR_ERRMEM;
      goto done;
    }
    lo->lines = ln;
    ln = &lo->lines[lo->nlines];
    memset(ln, 0, sizeof(*ln));
    ln->first = lo->nsegs;
    status = buildsegs(R, lo, refs, nrefs, ln->first);
    if (status != MR_OK) goto done;
    ln->count = lo->nsegs - ln->first;
    ln->box = lo->segs[ln->first].box;
    for (i = 1; i < ln->count; i++)
      mrL_join(&ln->box, &lo->segs[ln->first + i].box);
    status = linemetrics(R, lo, ln);
    if (status != MR_OK) goto done;
    findspaces(lo, ln);
    lo->nlines++;
  }
  status = markblanks(R, lo);
 done:
  if (bands != NULL) mrM_freearray(R, bands, capb);
  if (refs != NULL) mrM_freearray(R, refs, lo->ncomps);
  return status;
}


int mrL_inseg (const mr_Layout *lo, size_t seg, int x, int y) {
  int l = lo->labels[mr_cast(size_t, y) * mr_cast(size_t, lo->width) + x];
  return l != 0 && lo->comps[l - 1].seg == mr_cast(int, seg);
}


int mrL_inkbox (const mr_Layout *lo, size_t seg, const mr_Box *clip,
                mr_Box *out) {
  int x, y, found = 0;
  for (y = clip->y0; y < clip->y1; y++) {
    for (x = clip->x0; x < clip->x1; x++) {
      if (!mrL_inseg(lo, seg, x, y)) continue;
      if (!found) {
        out->x0 = out->x1 = x;
        out->y0 = out->y1 = y;
        found = 1;
      }
      if (x < out->x0) out->x0 = x;
      if (x + 1 > out->x1) out->x1 = x + 1;
      if (y + 1 > out->y1) out->y1 = y + 1;
    }
  }
  return found;
}


void mrL_mergenext (mr_Layout *lo, mr_Line *ln, size_t i) {
  size_t g = ln->first + i, k;
  int gi = mr_cast(int, g);
  mr_assert(i + 1 < ln->count);
  mrL_join(&lo->segs[g].box, &lo->segs[g + 1].box);
  for (k = 0; k < lo->ncomps; k++) {
    int s = lo->comps[k].seg;
    if (s == gi + 1) lo->comps[k].seg = gi;
    else if (s > gi + 1) lo->comps[k].seg = s - 1;
  }
  memmove(&lo->segs[g + 1], &lo->segs[g + 2],
          (lo->nsegs - g - 2) * sizeof(mr_Seg));
  lo->nsegs--;
  ln->count--;
  for (k = 0; k < lo->nlines; k++)
    if (lo->lines[k].first > ln->first) lo->lines[k].first--;
}


void mrL_free (mr_State *R, mr_Layout *lo) {
  if (lo->labels != NULL)
    mrM_freearray(R, lo->labels,
                  mr_cast(size_t, lo->width) * mr_cast(size_t, lo->height));
  if (lo->comps != NULL) mrM_freearray(R, lo->comps, lo->capcomps);
  if (lo->segs != NULL) mrM_freearray(R, lo->segs, lo->capsegs);
  if (lo->lines != NULL) mrM_freearray(R, lo->lines, lo->caplines);
  mrL_init(lo);
}
