/*
** mrocr.cpp
** OCR: gray image -> text, with our own trained network
** See Copyright Notice in mr.h
*/

#define mrocr_cpp
#define MR_CORE

#include "mrocr.h"
#include "mrbin.h"
#include "mrglyph.h"
#include "mrlayout.h"
#include "mrmem.h"
#include "mrstate.h"

#include <string.h>


/* below this confidence a letter may be a broken piece of another */
#define LOWCONF  0.7f

/* a glued letter must be at least this sure */
#define GLUECONF  0.5f

/* both halves of a split must be at least this sure */
#define SPLITCONF  0.5f

/* cut points tried per split, and how deep splits go (2^3 = 8 letters) */
#define MAXCUTS  8
#define MAXDEPTH  3
#define MAXPARTS  (1 << MAXDEPTH)


/* a letter found by splitting a segment */
typedef struct Part {
  int cls;
  float conf;
} Part;


/* everything needed while reading one image */
typedef struct Reader {
  mr_State *R;
  const mr_Net *net;
  mr_Layout lo;
  float *in;  /* network input */
  float *work;  /* network scratch */
  int *cls;  /* class of each segment of the current line */
  float *conf;  /* its confidence */
  mr_byte *rej;  /* 1 if the segment is not one letter */
  size_t cap;  /* size of 'cls', 'conf' and 'rej' */
  int *cols;  /* ink per column, for splitting */
  size_t capcols;
} Reader;


/*
** {======================================================
** Classifying
** =======================================================
*/

/*
** Classify the ink of segments 'a' and 'b' (-1 for none) in 'box'.
** Returns 1 if the network says it is not one letter (MR_REJECT); then
** '*cls' is still the most likely real letter.
*/
static int classify (Reader *rd, mr_Line *ln, const mr_Box *box, long a,
                     long b, int *cls, float *conf) {
  const mr_Net *net = rd->net;
  const float *p;
  int k, best = -1, n = mrN_nclasses(net);
  mrG_extract(&rd->lo, ln, box, a, b, rd->in);
  *cls = mrN_classify(net, rd->in, rd->work, conf);
  if (net->classes[*cls] != MR_REJECT) return 0;
  p = mrN_probs(net, rd->work);
  for (k = 0; k < n; k++)
    if (net->classes[k] != MR_REJECT && (best < 0 || p[k] > p[best]))
      best = k;
  *cls = best;
  *conf = p[best];
  return 1;
}


static int reserve (Reader *rd, size_t n) {
  mr_State *R = rd->R;
  int *c;
  float *f;
  mr_byte *r;
  if (n <= rd->cap) return MR_OK;
  c = mrM_newarray(R, n, int);
  f = mrM_newarray(R, n, float);
  r = mrM_newarray(R, n, mr_byte);
  if (c == NULL || f == NULL || r == NULL) {
    if (c != NULL) mrM_freearray(R, c, n);
    if (f != NULL) mrM_freearray(R, f, n);
    if (r != NULL) mrM_freearray(R, r, n);
    return MR_ERRMEM;
  }
  if (rd->cap > 0) {
    mrM_freearray(R, rd->cls, rd->cap);
    mrM_freearray(R, rd->conf, rd->cap);
    mrM_freearray(R, rd->rej, rd->cap);
  }
  rd->cls = c;
  rd->conf = f;
  rd->rej = r;
  rd->cap = n;
  return MR_OK;
}

/* }====================================================== */


/*
** {======================================================
** Gluing broken letters
** =======================================================
*/

/* glue segments 'i' to 'i + n' of 'ln' into one letter 'c' */
static void gluesegs (Reader *rd, mr_Line *ln, size_t i, size_t n, int c,
                      float cf) {
  size_t k, rest;
  for (k = 0; k < n; k++) mrL_mergenext(&rd->lo, ln, i);
  rd->cls[i] = c;
  rd->conf[i] = cf;
  rd->rej[i] = 0;
  rest = ln->count - i - 1;
  memmove(&rd->cls[i + 1], &rd->cls[i + 1 + n], rest * sizeof(int));
  memmove(&rd->conf[i + 1], &rd->conf[i + 1 + n], rest * sizeof(float));
  memmove(&rd->rej[i + 1], &rd->rej[i + 1 + n], rest * sizeof(mr_byte));
}


/*
** Try to glue segments 'i' to 'i + n' into one letter that the network
** is surer about than about any of the pieces.
*/
static int tryglue (Reader *rd, mr_Line *ln, size_t i, size_t n) {
  size_t g = ln->first + i, k;
  mr_Box box = rd->lo.segs[g].box;
  float highest = 0, cf;
  int c;
  for (k = 0; k <= n; k++) {
    if (k > 0) mrL_join(&box, &rd->lo.segs[g + k].box);
    if (rd->conf[i + k] > highest) highest = rd->conf[i + k];
  }
  if (classify(rd, ln, &box, mr_cast(long, g), mr_cast(long, g + n), &c,
               &cf) ||
      cf < GLUECONF || cf <= highest)
    return 0;
  gluesegs(rd, ln, i, n, c, cf);
  return 1;
}


/*
** Glue neighbor segments when the network is more sure about them
** together: broken letters, and letters in several parts like '%' (two
** pieces after the slash is glued) or '½' (three pieces).
*/
static void glue (Reader *rd, mr_Line *ln) {
  size_t i = 0;
  while (i + 1 < ln->count) {
    size_t g = ln->first + i;
    mr_Seg *a = &rd->lo.segs[g], *b = &rd->lo.segs[g + 1];
    float lowest = (rd->conf[i] < rd->conf[i + 1]) ? rd->conf[i]
                                                   : rd->conf[i + 1];
    int gap = b->box.x0 - a->box.x1;
    if (!b->space && (gap <= 0 || lowest < LOWCONF)) {
      if (tryglue(rd, ln, i, 1))
        continue;  /* maybe glue the next one too */
      if (i + 2 < ln->count && !rd->lo.segs[g + 2].space &&
          rd->lo.segs[g + 2].box.x0 <= b->box.x1 && tryglue(rd, ln, i, 2))
        continue;
    }
    i++;
  }
}

/* }====================================================== */


/*
** {======================================================
** Splitting touching letters
** =======================================================
*/

/* the columns of 'box' with the least ink of segment 'g' */
static int findcuts (Reader *rd, mr_Line *ln, long g, const mr_Box *box,
                     int *cuts) {
  int w = box->x1 - box->x0, x, y, n = 0, k;
  int minw = ln->xheight / 4;
  size_t need = mr_cast(size_t, w);
  if (minw < 2) minw = 2;
  if (w < 2 * minw + 1) return 0;
  if (need > rd->capcols) {
    int *c = mrM_newarray(rd->R, need, int);
    if (c == NULL) return 0;
    if (rd->cols != NULL) mrM_freearray(rd->R, rd->cols, rd->capcols);
    rd->cols = c;
    rd->capcols = need;
  }
  for (x = 0; x < w; x++) {
    int count = 0;
    for (y = box->y0; y < box->y1; y++)
      count += mrL_inseg(&rd->lo, mr_cast(size_t, g), box->x0 + x, y);
    rd->cols[x] = count;
  }
  /* local minima, kept sorted by ink (fewest first) */
  for (x = minw; x < w - minw; x++) {
    int v = rd->cols[x];
    if (v > rd->cols[x - 1] || v > rd->cols[x + 1]) continue;
    if (n == MAXCUTS && v >= rd->cols[cuts[n - 1] - box->x0]) continue;
    if (n < MAXCUTS) n++;
    for (k = n - 1; k > 0 && rd->cols[cuts[k - 1] - box->x0] > v; k--)
      cuts[k] = cuts[k - 1];
    cuts[k] = box->x0 + x;
  }
  return n;
}


static void addpart (Part *parts, int *n, int cls, float conf) {
  if (*n < MAXPARTS) {
    parts[*n].cls = cls;
    parts[*n].conf = conf;
    (*n)++;
  }
}


/*
** Segment 'g' (ink inside 'box') is not one letter: find the cut that
** gives two sure letters, and split the halves again if needed.
*/
static void split (Reader *rd, mr_Line *ln, long g, const mr_Box *box,
                   int depth, Part *parts, int *n) {
  int cuts[MAXCUTS], ncuts, k, best = -1;
  int cl = 0, cr = 0, rl = 0, rr = 0, wholecls;
  float bestscore = 0, fl = 0, fr = 0, wholeconf;
  mr_Box bl, br;
  int wholerej = classify(rd, ln, box, g, -1, &wholecls, &wholeconf);
  if (!wholerej || depth >= MAXDEPTH) {
    addpart(parts, n, wholecls, wholeconf);
    return;
  }
  ncuts = findcuts(rd, ln, g, box, cuts);
  for (k = 0; k < ncuts; k++) {
    mr_Box l = *box, r = *box, il, ir;
    int c1, c2, j1, j2;
    float f1, f2, score;
    l.x1 = cuts[k];
    r.x0 = cuts[k];
    if (!mrL_inkbox(&rd->lo, mr_cast(size_t, g), &l, &il) ||
        !mrL_inkbox(&rd->lo, mr_cast(size_t, g), &r, &ir))
      continue;
    j1 = classify(rd, ln, &il, g, -1, &c1, &f1);
    j2 = classify(rd, ln, &ir, g, -1, &c2, &f2);
    /* a half that is still "not one letter" may split again later */
    score = ((j1 ? 0.5f : f1) < (j2 ? 0.5f : f2)) ? (j1 ? 0.5f : f1)
                                                   : (j2 ? 0.5f : f2);
    if (score > bestscore) {
      bestscore = score;
      best = k;
      bl = il;
      br = ir;
      cl = c1; cr = c2;
      rl = j1; rr = j2;
      fl = f1; fr = f2;
    }
  }
  if (best < 0 || bestscore < SPLITCONF) {  /* no good cut */
    addpart(parts, n, wholecls, wholeconf);
    return;
  }
  if (rl) split(rd, ln, g, &bl, depth + 1, parts, n);
  else addpart(parts, n, cl, fl);
  if (rr) split(rd, ln, g, &br, depth + 1, parts, n);
  else addpart(parts, n, cr, fr);
}

/* }====================================================== */


/*
** {======================================================
** Text output
** =======================================================
*/

/* last letter written, to join quote marks */
typedef struct Last {
  uint32_t cp;
  size_t len;  /* its size in bytes */
} Last;


/* two single quote marks in a row are one double quote */
static uint32_t doublequote (uint32_t prev, uint32_t cp) {
  if (prev != cp) return 0;
  switch (cp) {
    case '\'': return '"';
    case 0x2018: return 0x201C;  /* ‘‘ -> “ */
    case 0x2019: return 0x201D;  /* ’’ -> ” */
    default: return 0;
  }
}


static int emit (mr_State *R, mr_Buffer *out, uint32_t cp, Last *last) {
  uint32_t dq = doublequote(last->cp, cp);
  size_t before;
  int status;
  if (dq != 0) {  /* replace the previous mark */
    out->len -= last->len;
    out->data[out->len] = '\0';
    cp = dq;
  }
  before = out->len;
  status = mrB_addutf8(R, out, cp);
  last->cp = (dq != 0) ? 0 : cp;  /* do not join three marks */
  last->len = out->len - before;
  return status;
}


static int readline (Reader *rd, mr_Line *ln, mr_Buffer *out) {
  mr_State *R = rd->R;
  Last last;
  size_t i;
  int status = reserve(rd, ln->count);
  if (status != MR_OK) return status;
  last.cp = 0;
  last.len = 0;
  for (i = 0; i < ln->count; i++) {
    size_t g = ln->first + i;
    rd->rej[i] = mr_cast(mr_byte,
                         classify(rd, ln, &rd->lo.segs[g].box,
                                  mr_cast(long, g), -1, &rd->cls[i],
                                  &rd->conf[i]));
  }
  glue(rd, ln);
  for (i = 0; i < ln->count; i++) {
    const mr_Seg *s = &rd->lo.segs[ln->first + i];
    Part parts[MAXPARTS];
    int np = 0, k;
    if (s->space) {
      status = mrB_addchar(R, out, ' ');
      if (status != MR_OK) return status;
      last.cp = ' ';
      last.len = 1;
    }
    if (rd->rej[i])
      split(rd, ln, mr_cast(long, ln->first + i), &s->box, 0, parts, &np);
    else
      addpart(parts, &np, rd->cls[i], rd->conf[i]);
    for (k = 0; k < np; k++) {
      status = emit(R, out, rd->net->classes[parts[k].cls], &last);
      if (status != MR_OK) return status;
    }
  }
  return mrB_addchar(R, out, '\n');
}

/* }====================================================== */


int mrO_run (mr_State *R, const mr_Net *net, const mr_Image *img,
             mr_Buffer *out) {
  Reader rd;
  mr_Bitmap *bm = NULL;
  size_t i;
  int status;
  memset(&rd, 0, sizeof(rd));
  rd.R = R;
  rd.net = net;
  mrL_init(&rd.lo);
  rd.in = mrM_newarray(R, MR_GLYPHINPUT, float);
  rd.work = mrM_newarray(R, mrN_worksize(net), float);
  if (rd.in == NULL || rd.work == NULL) {
    status = MR_ERRMEM;
    goto done;
  }
  status = mrK_fromgray(R, img, &bm);
  if (status != MR_OK) goto done;
  status = mrL_analyze(R, bm, &rd.lo);
  if (status != MR_OK) goto done;
  for (i = 0; i < rd.lo.nlines; i++) {
    mr_Line *ln = &rd.lo.lines[i];
    if (ln->blank) {
      status = mrB_addchar(R, out, '\n');
      if (status != MR_OK) goto done;
    }
    status = readline(&rd, ln, out);
    if (status != MR_OK) goto done;
  }
 done:
  if (rd.in != NULL) mrM_freearray(R, rd.in, MR_GLYPHINPUT);
  if (rd.work != NULL) mrM_freearray(R, rd.work, mrN_worksize(net));
  if (rd.cap > 0) {
    mrM_freearray(R, rd.cls, rd.cap);
    mrM_freearray(R, rd.conf, rd.cap);
    mrM_freearray(R, rd.rej, rd.cap);
  }
  if (rd.cols != NULL) mrM_freearray(R, rd.cols, rd.capcols);
  mrL_free(R, &rd.lo);
  mrK_free(R, bm);
  return status;
}


/*
** {======================================================
** Line models (handwriting): read each text line at once
** =======================================================
*/

/* the ink of line 'ln' only (not of the lines above and below) */
static int lineink (mr_State *R, const mr_Layout *lo, const mr_Line *ln,
                    mr_byte **bits, int *w, int *h) {
  int x, y, x0 = ln->box.x0, y0 = ln->box.y0;
  long first = mr_cast(long, ln->first);
  long last = first + mr_cast(long, ln->count);
  *w = ln->box.x1 - x0;
  *h = ln->box.y1 - y0;
  *bits = mrM_newarray(R, mr_cast(size_t, *w) * *h, mr_byte);
  if (*bits == NULL) return MR_ERRMEM;
  for (y = 0; y < *h; y++) {
    for (x = 0; x < *w; x++) {
      int l = lo->labels[mr_cast(size_t, y0 + y) * lo->width + x0 + x];
      long s = (l != 0) ? lo->comps[l - 1].seg : -1;
      (*bits)[mr_cast(size_t, y) * *w + x] =
        mr_cast(mr_byte, s >= first && s < last);
    }
  }
  return MR_OK;
}


static int runseq (mr_State *R, const mr_Image *img, mr_Buffer *out) {
  mr_Bitmap *bm = NULL;
  mr_Layout lo;
  float *line;
  size_t i;
  int status;
  size_t nline = mr_cast(size_t, MR_SEQH) * MR_SEQMAXW;
  mrL_init(&lo);
  line = mrM_newarray(R, nline, float);
  if (line == NULL) return MR_ERRMEM;
  status = mrK_fromgray(R, img, &bm);
  if (status == MR_OK) status = mrL_analyze(R, bm, &lo);
  for (i = 0; status == MR_OK && i < lo.nlines; i++) {
    const mr_Line *ln = &lo.lines[i];
    mr_byte *bits = NULL;
    int w, h, lw;
    if (ln->blank) {
      status = mrB_addchar(R, out, '\n');
      if (status != MR_OK) break;
    }
    status = lineink(R, &lo, ln, &bits, &w, &h);
    if (status != MR_OK) break;
    lw = mrQ_normalize(bits, w, h, 1.0f, line);
    mrM_freearray(R, bits, mr_cast(size_t, w) * h);
    if (lw > 0) status = mrQ_read(R, R->seq, R->seqwork, line, lw, out);
    if (status == MR_OK) status = mrB_addchar(R, out, '\n');
  }
  mrM_freearray(R, line, nline);
  mrL_free(R, &lo);
  mrK_free(R, bm);
  return status;
}

/* }====================================================== */


/* '<datapath>/<lang>.mrm': a letter model or a line model */
int mrO_loadmodel (mr_State *R) {
  char path[MR_PATHSIZE + MR_LANGSIZE + 8];
  size_t n = strlen(R->datapath);
  int status;
  if (R->net != NULL || R->seq != NULL) return MR_OK;
  if (n > 0) {
    memcpy(path, R->datapath, n);
    if (path[n - 1] != '/' && path[n - 1] != '\\') path[n++] = '/';
  }
  memcpy(path + n, R->lang, strlen(R->lang));
  n += strlen(R->lang);
  memcpy(path + n, MR_MODELEXT, sizeof(MR_MODELEXT));
  if (!mrQ_isseq(path)) return mrN_load(R, path, &R->net);
  status = mrQ_load(R, path, &R->seq);
  if (status == MR_OK) status = mrQ_newwork(R, R->seq, &R->seqwork);
  return status;
}


int mrO_recognize (mr_State *R, const mr_Image *img, mr_Buffer *out) {
  int status = mrO_loadmodel(R);
  if (status != MR_OK) return status;
  if (R->seq != NULL) return runseq(R, img, out);
  return mrO_run(R, R->net, img, out);
}
