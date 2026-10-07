/*
** mrseq.cpp
** Line recognizer for handwriting: CNN + bidirectional LSTM + CTC
** See Copyright Notice in mr.h
**
** Network, for a line MR_SEQH (32) pixels high and W wide:
**   conv 3x3  1->16, ReLU, max pool 2x2    -> 16 x 16 x W/2
**   conv 3x3 16->32, ReLU, max pool 2x2    -> 32 x  8 x W/4
**   conv 3x3 32->64, ReLU, max pool 2x1    -> 64 x  4 x W/4
**   conv 3x3 64->64, ReLU, max pool 2x1    -> 64 x  2 x W/4
**   each of the T = W/4 columns is a vector of 128 features
**   LSTM forward and LSTM backward over the columns, H cells each
**   linear layer -> K classes (class 0 = blank), softmax
** Trained with CTC, so the text needs no position for each letter.
*/

#define mrseq_cpp
#define MR_CORE

#include "mrseq.h"
#include "mrfile.h"
#include "mrmem.h"
#include "mrstate.h"

#include <math.h>
#include <string.h>


#define NCONV  4
#define FEAT  128  /* 64 channels x 2 rows */
#define MAXT  (MR_SEQMAXW / MR_SEQSTEP)
#define MAXS  (2 * MAXT + 1)  /* CTC states: blank, label, blank, ... */

#define SEQMAGIC  "MRSQ"
#define SEQVERSION  1

#define NEGINF  (-1e30f)


static const int convin[NCONV] = { 1, 16, 32, 64 };
static const int convout[NCONV] = { 16, 32, 64, 64 };
static const int poolw[NCONV] = { 2, 2, 1, 1 };  /* pool height is 2 */


/* where each part of the network starts in 'p' */
typedef struct Parts {
  size_t cw[NCONV], cb[NCONV];  /* conv weights, biases */
  size_t wx[2], wh[2], lb[2];  /* LSTM per direction */
  size_t wo, bo;  /* output layer */
  size_t total;
} Parts;


struct mr_SeqWork {
  int K, H;
  size_t np;
  float *grad;  /* np */
  float *act[NCONV];  /* conv output (ReLU), before pooling */
  float *pin[NCONV + 1];  /* pooled output = next input; pin[0] unused */
  float *dact;  /* backward scratch, largest 'act' */
  float *dpin[NCONV + 1];  /* gradients of pooled outputs */
  float *feat, *dfeat;  /* T x FEAT */
  float *st[2];  /* LSTM state per step: i f g o c h (6H each) */
  float *hcat, *dhcat;  /* T x 2H */
  float *prob, *dlogit;  /* T x K */
  float *z, *dz, *dhnext, *dcnext, *zero;  /* LSTM scratch */
  float *alpha, *beta;  /* CTC, T x S */
  int *ext;  /* CTC extended labels */
};


/*
** {======================================================
** Shapes
** =======================================================
*/

static void parts (int K, int H, Parts *pt) {
  size_t at = 0;
  int l, d;
  for (l = 0; l < NCONV; l++) {
    pt->cw[l] = at;
    at += mr_cast(size_t, convout[l]) * convin[l] * 9;
    pt->cb[l] = at;
    at += mr_cast(size_t, convout[l]);
  }
  for (d = 0; d < 2; d++) {
    pt->wx[d] = at;
    at += mr_cast(size_t, 4 * H) * FEAT;
    pt->wh[d] = at;
    at += mr_cast(size_t, 4 * H) * H;
    pt->lb[d] = at;
    at += mr_cast(size_t, 4 * H);
  }
  pt->wo = at;
  at += mr_cast(size_t, K) * 2 * H;
  pt->bo = at;
  at += mr_cast(size_t, K);
  pt->total = at;
}


/* input height and width of conv layer 'l' for a line 'w' wide */
static int convh (int l) {
  return MR_SEQH >> l;
}


static int convw (int l, int w) {
  int k;
  for (k = 0; k < l; k++) w /= poolw[k];
  return w;
}


static size_t actsize (int l) {  /* largest 'act' of layer 'l' */
  return mr_cast(size_t, convout[l]) * convh(l) * convw(l, MR_SEQMAXW);
}


static size_t pinsize (int l) {  /* largest pooled output of layer l-1 */
  return mr_cast(size_t, convout[l - 1]) * convh(l) * convw(l, MR_SEQMAXW);
}

/* }====================================================== */


/*
** {======================================================
** Network and model file
** =======================================================
*/

int mrQ_new (mr_State *R, int nclasses, int hidden, mr_Seq **out) {
  mr_Seq *q;
  Parts pt;
  *out = NULL;
  if (nclasses < 2 || nclasses > 100000 || hidden < 4 || hidden > 1024)
    return mrS_error(R, MR_ERRARG, "bad line model size");
  parts(nclasses, hidden, &pt);
  q = mrM_new(R, mr_Seq);
  if (q == NULL) return MR_ERRMEM;
  memset(q, 0, sizeof(*q));
  q->nclasses = nclasses;
  q->hidden = hidden;
  q->np = pt.total;
  q->classes = mrM_newarray(R, nclasses, uint32_t);
  q->p = mrM_newarray(R, q->np, float);
  if (q->classes == NULL || q->p == NULL) {
    mrQ_free(R, q);
    return MR_ERRMEM;
  }
  memset(q->classes, 0, nclasses * sizeof(uint32_t));
  memset(q->p, 0, q->np * sizeof(float));
  *out = q;
  return MR_OK;
}


void mrQ_free (mr_State *R, mr_Seq *q) {
  if (q == NULL) return;
  if (q->classes != NULL) mrM_freearray(R, q->classes, q->nclasses);
  if (q->p != NULL) mrM_freearray(R, q->p, q->np);
  mrM_delete(R, q);
}


void mrQ_randomize (mr_Seq *q, mr_Rand *rng) {
  Parts pt;
  size_t i;
  int l, d, H = q->hidden, K = q->nclasses;
  float s;
  parts(K, H, &pt);
  memset(q->p, 0, q->np * sizeof(float));
  for (l = 0; l < NCONV; l++) {
    size_t n = mr_cast(size_t, convout[l]) * convin[l] * 9;
    s = sqrtf(2.0f / (convin[l] * 9));
    for (i = 0; i < n; i++) q->p[pt.cw[l] + i] = mrR_normal(rng) * s;
  }
  s = 1.0f / sqrtf(mr_cast(float, H));
  for (d = 0; d < 2; d++) {
    for (i = 0; i < mr_cast(size_t, 4 * H) * FEAT; i++)
      q->p[pt.wx[d] + i] = mrR_range(rng, -s, s);
    for (i = 0; i < mr_cast(size_t, 4 * H) * H; i++)
      q->p[pt.wh[d] + i] = mrR_range(rng, -s, s);
    for (i = 0; i < mr_cast(size_t, H); i++)
      q->p[pt.lb[d] + H + i] = 1.0f;  /* forget gate starts open */
  }
  s = sqrtf(1.0f / (2 * H));
  for (i = 0; i < mr_cast(size_t, K) * 2 * H; i++)
    q->p[pt.wo + i] = mrR_normal(rng) * s;
}


static int put32 (FILE *f, uint32_t v) {
  mr_byte b[4];
  b[0] = mr_cast(mr_byte, v);
  b[1] = mr_cast(mr_byte, v >> 8);
  b[2] = mr_cast(mr_byte, v >> 16);
  b[3] = mr_cast(mr_byte, v >> 24);
  return fwrite(b, 1, 4, f) == 4;
}


static int get32 (FILE *f, uint32_t *v) {
  mr_byte b[4];
  if (fread(b, 1, 4, f) != 4) return 0;
  *v = mr_cast(uint32_t, b[0]) | mr_cast(uint32_t, b[1]) << 8 |
       mr_cast(uint32_t, b[2]) << 16 | mr_cast(uint32_t, b[3]) << 24;
  return 1;
}


int mrQ_save (mr_State *R, const mr_Seq *q, const char *path) {
  FILE *f = mrF_open(path, "wb");
  size_t i;
  int ok, k;
  if (f == NULL) return mrS_error(R, MR_ERRFILE, "cannot create model file");
  ok = fwrite(SEQMAGIC, 1, 4, f) == 4 && put32(f, SEQVERSION) &&
       put32(f, MR_SEQH) && put32(f, mr_cast(uint32_t, q->nclasses)) &&
       put32(f, mr_cast(uint32_t, q->hidden));
  for (k = 0; ok && k < q->nclasses; k++) ok = put32(f, q->classes[k]);
  for (i = 0; ok && i < q->np; i++) {
    uint32_t u;
    memcpy(&u, &q->p[i], 4);
    ok = put32(f, u);
  }
  if (fclose(f) != 0) ok = 0;
  return ok ? MR_OK : mrS_error(R, MR_ERRFILE, "cannot write model file");
}


int mrQ_isseq (const char *path) {
  char magic[4];
  FILE *f = mrF_open(path, "rb");
  int is;
  if (f == NULL) return 0;
  is = fread(magic, 1, 4, f) == 4 && memcmp(magic, SEQMAGIC, 4) == 0;
  fclose(f);
  return is;
}


int mrQ_load (mr_State *R, const char *path, mr_Seq **out) {
  FILE *f = mrF_open(path, "rb");
  char magic[4];
  uint32_t version, h, K, H, u;
  mr_Seq *q = NULL;
  size_t i;
  int k, status;
  *out = NULL;
  if (f == NULL)
    return mrS_error(R, MR_ERRFILE, "cannot open model '%s'", path);
  if (fread(magic, 1, 4, f) != 4 || memcmp(magic, SEQMAGIC, 4) != 0 ||
      !get32(f, &version) || !get32(f, &h) || !get32(f, &K) ||
      !get32(f, &H) || version != SEQVERSION || h != MR_SEQH)
    goto bad;
  status = mrQ_new(R, mr_cast(int, K), mr_cast(int, H), &q);
  if (status != MR_OK) {
    fclose(f);
    return status;
  }
  for (k = 0; k < q->nclasses; k++)
    if (!get32(f, &q->classes[k])) goto bad;
  for (i = 0; i < q->np; i++) {
    if (!get32(f, &u)) goto bad;
    memcpy(&q->p[i], &u, 4);
    if (!isfinite(q->p[i])) goto bad;
  }
  fclose(f);
  *out = q;
  return MR_OK;
 bad:
  fclose(f);
  mrQ_free(R, q);
  return mrS_error(R, MR_ERRFORMAT, "line model '%s' is broken or old",
                   path);
}

/* }====================================================== */


/*
** {======================================================
** Work space
** =======================================================
*/

int mrQ_newwork (mr_State *R, const mr_Seq *q, mr_SeqWork **out) {
  mr_SeqWork *w;
  int l, H = q->hidden, K = q->nclasses;
  size_t maxact = 0;
  *out = NULL;
  w = mrM_new(R, mr_SeqWork);
  if (w == NULL) return MR_ERRMEM;
  memset(w, 0, sizeof(*w));
  w->K = K;
  w->H = H;
  w->np = q->np;
  w->grad = mrM_newarray(R, q->np, float);
  for (l = 0; l < NCONV; l++) {
    w->act[l] = mrM_newarray(R, actsize(l), float);
    w->pin[l + 1] = mrM_newarray(R, pinsize(l + 1), float);
    w->dpin[l + 1] = mrM_newarray(R, pinsize(l + 1), float);
    if (actsize(l) > maxact) maxact = actsize(l);
  }
  w->dpin[0] = mrM_newarray(R, 1, float);  /* not used */
  w->dact = mrM_newarray(R, maxact, float);
  w->feat = mrM_newarray(R, mr_cast(size_t, MAXT) * FEAT, float);
  w->dfeat = mrM_newarray(R, mr_cast(size_t, MAXT) * FEAT, float);
  w->st[0] = mrM_newarray(R, mr_cast(size_t, MAXT) * 6 * H, float);
  w->st[1] = mrM_newarray(R, mr_cast(size_t, MAXT) * 6 * H, float);
  w->hcat = mrM_newarray(R, mr_cast(size_t, MAXT) * 2 * H, float);
  w->dhcat = mrM_newarray(R, mr_cast(size_t, MAXT) * 2 * H, float);
  w->prob = mrM_newarray(R, mr_cast(size_t, MAXT) * K, float);
  w->dlogit = mrM_newarray(R, mr_cast(size_t, MAXT) * K, float);
  w->z = mrM_newarray(R, 4 * H, float);
  w->dz = mrM_newarray(R, 4 * H, float);
  w->dhnext = mrM_newarray(R, H, float);
  w->dcnext = mrM_newarray(R, H, float);
  w->zero = mrM_newarray(R, H, float);
  w->alpha = mrM_newarray(R, mr_cast(size_t, MAXT) * MAXS, float);
  w->beta = mrM_newarray(R, mr_cast(size_t, MAXT) * MAXS, float);
  w->ext = mrM_newarray(R, MAXS, int);
  if (!w->grad || !w->dact || !w->feat || !w->dfeat || !w->st[0] ||
      !w->st[1] || !w->hcat || !w->dhcat || !w->prob || !w->dlogit ||
      !w->z || !w->dz || !w->dhnext || !w->dcnext || !w->zero ||
      !w->alpha || !w->beta || !w->ext || !w->dpin[0]) {
    mrQ_freework(R, w);
    return MR_ERRMEM;
  }
  for (l = 0; l < NCONV; l++) {
    if (!w->act[l] || !w->pin[l + 1] || !w->dpin[l + 1]) {
      mrQ_freework(R, w);
      return MR_ERRMEM;
    }
  }
  memset(w->zero, 0, H * sizeof(float));
  memset(w->grad, 0, q->np * sizeof(float));
  *out = w;
  return MR_OK;
}


#define FREEARR(R, p, n)  if ((p) != NULL) mrM_freearray(R, p, n)

void mrQ_freework (mr_State *R, mr_SeqWork *w) {
  int l, H, K;
  size_t maxact = 0;
  if (w == NULL) return;
  H = w->H;
  K = w->K;
  for (l = 0; l < NCONV; l++) {
    FREEARR(R, w->act[l], actsize(l));
    FREEARR(R, w->pin[l + 1], pinsize(l + 1));
    FREEARR(R, w->dpin[l + 1], pinsize(l + 1));
    if (actsize(l) > maxact) maxact = actsize(l);
  }
  FREEARR(R, w->dpin[0], 1);
  FREEARR(R, w->grad, w->np);
  FREEARR(R, w->dact, maxact);
  FREEARR(R, w->feat, mr_cast(size_t, MAXT) * FEAT);
  FREEARR(R, w->dfeat, mr_cast(size_t, MAXT) * FEAT);
  FREEARR(R, w->st[0], mr_cast(size_t, MAXT) * 6 * H);
  FREEARR(R, w->st[1], mr_cast(size_t, MAXT) * 6 * H);
  FREEARR(R, w->hcat, mr_cast(size_t, MAXT) * 2 * H);
  FREEARR(R, w->dhcat, mr_cast(size_t, MAXT) * 2 * H);
  FREEARR(R, w->prob, mr_cast(size_t, MAXT) * K);
  FREEARR(R, w->dlogit, mr_cast(size_t, MAXT) * K);
  FREEARR(R, w->z, 4 * H);
  FREEARR(R, w->dz, 4 * H);
  FREEARR(R, w->dhnext, H);
  FREEARR(R, w->dcnext, H);
  FREEARR(R, w->zero, H);
  FREEARR(R, w->alpha, mr_cast(size_t, MAXT) * MAXS);
  FREEARR(R, w->beta, mr_cast(size_t, MAXT) * MAXS);
  FREEARR(R, w->ext, MAXS);
  mrM_delete(R, w);
}


float *mrQ_grad (mr_SeqWork *wk) {
  return wk->grad;
}


void mrQ_cleargrad (const mr_Seq *q, mr_SeqWork *wk) {
  memset(wk->grad, 0, q->np * sizeof(float));
}

/* }====================================================== */


/*
** {======================================================
** Line image
** =======================================================
*/

int mrQ_normalize (const mr_byte *bits, int w, int h, float stretch,
                   float *line) {
  int x, y, x0 = w, y0 = h, x1 = -1, y1 = -1, ow, inner = MR_SEQH - 4;
  float s, sx;
  for (y = 0; y < h; y++) {
    for (x = 0; x < w; x++) {
      if (!bits[mr_cast(size_t, y) * w + x]) continue;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
  }
  if (x1 < 0) return 0;
  s = mr_cast(float, inner) / (y1 - y0 + 1);
  sx = s * stretch;
  ow = mr_cast(int, (x1 - x0 + 1) * sx) + 4;
  if (ow > MR_SEQMAXW) {  /* too long: squeeze */
    sx = mr_cast(float, MR_SEQMAXW - 4) / (x1 - x0 + 1);
    ow = MR_SEQMAXW;
  }
  ow = (ow + MR_SEQSTEP - 1) / MR_SEQSTEP * MR_SEQSTEP;
  if (ow < 4 * MR_SEQSTEP) ow = 4 * MR_SEQSTEP;
  if (ow > MR_SEQMAXW) ow = MR_SEQMAXW;
  for (y = 0; y < MR_SEQH; y++) {
    for (x = 0; x < ow; x++) {
      int count = 0, sy, su;
      for (sy = 0; sy < 3; sy++) {
        for (su = 0; su < 3; su++) {
          float fx = (x - 2 + (su + 0.5f) / 3) / sx;
          float fy = (y - 2 + (sy + 0.5f) / 3) / s;
          int ix = x0 + mr_cast(int, floorf(fx));
          int iy = y0 + mr_cast(int, floorf(fy));
          if (fx < 0 || fy < 0 || ix > x1 || iy > y1) continue;
          count += bits[mr_cast(size_t, iy) * w + ix];
        }
      }
      line[y * ow + x] = count / 9.0f;
    }
  }
  return ow;
}

/* }====================================================== */


/*
** {======================================================
** Forward
** =======================================================
*/

static float sigm (float x) {
  return 1.0f / (1.0f + expf(-x));
}


/* 3x3 convolution with zero padding, then ReLU */
static void convfwd (const float *wt, const float *bias, int cin, int cout,
                     int h, int w, const float *in, float *out) {
  int co, ci, ky, kx, y, x;
  for (co = 0; co < cout; co++) {
    float *o = out + mr_cast(size_t, co) * h * w;
    for (x = 0; x < h * w; x++) o[x] = bias[co];
    for (ci = 0; ci < cin; ci++) {
      for (ky = 0; ky < 3; ky++) {
        for (kx = 0; kx < 3; kx++) {
          float wv = wt[((co * cin + ci) * 3 + ky) * 3 + kx];
          int xa = (kx == 0) ? 1 : 0, xb = (kx == 2) ? w - 1 : w;
          for (y = 0; y < h; y++) {
            int iy = y + ky - 1;
            const float *src;
            float *dst;
            if (iy < 0 || iy >= h) continue;
            src = in + (mr_cast(size_t, ci) * h + iy) * w + (kx - 1);
            dst = o + mr_cast(size_t, y) * w;
            for (x = xa; x < xb; x++) dst[x] += wv * src[x];
          }
        }
      }
    }
    for (x = 0; x < h * w; x++)
      if (o[x] < 0) o[x] = 0;
  }
}


/* max pooling 2 x pw */
static void poolfwd (int c, int h, int w, int pw, const float *in,
                     float *out) {
  int k, y, x, oh = h / 2, ow = w / pw;
  for (k = 0; k < c; k++) {
    for (y = 0; y < oh; y++) {
      for (x = 0; x < ow; x++) {
        const float *a = in + (mr_cast(size_t, k) * h + 2 * y) * w + pw * x;
        float m = a[0];
        if (a[w] > m) m = a[w];
        if (pw == 2) {
          if (a[1] > m) m = a[1];
          if (a[w + 1] > m) m = a[w + 1];
        }
        out[(mr_cast(size_t, k) * oh + y) * ow + x] = m;
      }
    }
  }
}


static void lstmfwd (const float *wx, const float *wh, const float *b,
                     int H, int T, const float *feat, float *st,
                     float *z, const float *zero, int rev) {
  int k, j, i;
  for (k = 0; k < T; k++) {
    int t = rev ? T - 1 - k : k;
    int tp = rev ? t + 1 : t - 1;
    const float *x = feat + mr_cast(size_t, t) * FEAT;
    const float *hp = (k == 0) ? zero : st + mr_cast(size_t, tp) * 6 * H
                                            + 5 * H;
    const float *cp = (k == 0) ? zero : st + mr_cast(size_t, tp) * 6 * H
                                            + 4 * H;
    float *s = st + mr_cast(size_t, t) * 6 * H;
    for (j = 0; j < 4 * H; j++) {
      const float *rx = wx + mr_cast(size_t, j) * FEAT;
      const float *rh = wh + mr_cast(size_t, j) * H;
      float v = b[j];
      for (i = 0; i < FEAT; i++) v += rx[i] * x[i];
      for (i = 0; i < H; i++) v += rh[i] * hp[i];
      z[j] = v;
    }
    for (j = 0; j < H; j++) {
      float ig = sigm(z[j]), fg = sigm(z[H + j]);
      float gg = tanhf(z[2 * H + j]), og = sigm(z[3 * H + j]);
      float c = fg * cp[j] + ig * gg;
      s[j] = ig;
      s[H + j] = fg;
      s[2 * H + j] = gg;
      s[3 * H + j] = og;
      s[4 * H + j] = c;
      s[5 * H + j] = og * tanhf(c);
    }
  }
}


/* the whole forward pass; returns T and leaves softmax in wk->prob */
static int forward (const mr_Seq *q, mr_SeqWork *wk, const float *line,
                    int width) {
  Parts pt;
  const float *p = q->p;
  const float *in = line;
  int l, t, k, j, T = width / MR_SEQSTEP, H = q->hidden, K = q->nclasses;
  parts(K, H, &pt);
  for (l = 0; l < NCONV; l++) {
    int h = convh(l), w = convw(l, width);
    convfwd(p + pt.cw[l], p + pt.cb[l], convin[l], convout[l], h, w, in,
            wk->act[l]);
    poolfwd(convout[l], h, w, poolw[l], wk->act[l], wk->pin[l + 1]);
    in = wk->pin[l + 1];
  }
  /* columns -> feature vectors: f = channel * 2 + row */
  for (t = 0; t < T; t++)
    for (k = 0; k < 64; k++)
      for (j = 0; j < 2; j++)
        wk->feat[mr_cast(size_t, t) * FEAT + k * 2 + j] =
          in[(mr_cast(size_t, k) * 2 + j) * T + t];
  lstmfwd(p + pt.wx[0], p + pt.wh[0], p + pt.lb[0], H, T, wk->feat,
          wk->st[0], wk->z, wk->zero, 0);
  lstmfwd(p + pt.wx[1], p + pt.wh[1], p + pt.lb[1], H, T, wk->feat,
          wk->st[1], wk->z, wk->zero, 1);
  for (t = 0; t < T; t++) {
    float *hc = wk->hcat + mr_cast(size_t, t) * 2 * H;
    float *pr = wk->prob + mr_cast(size_t, t) * K, m, sum = 0;
    memcpy(hc, wk->st[0] + mr_cast(size_t, t) * 6 * H + 5 * H,
           H * sizeof(float));
    memcpy(hc + H, wk->st[1] + mr_cast(size_t, t) * 6 * H + 5 * H,
           H * sizeof(float));
    for (k = 0; k < K; k++) {
      const float *row = p + pt.wo + mr_cast(size_t, k) * 2 * H;
      float v = p[pt.bo + k];
      for (j = 0; j < 2 * H; j++) v += row[j] * hc[j];
      pr[k] = v;
    }
    m = pr[0];
    for (k = 1; k < K; k++)
      if (pr[k] > m) m = pr[k];
    for (k = 0; k < K; k++) {
      pr[k] = expf(pr[k] - m);
      sum += pr[k];
    }
    for (k = 0; k < K; k++) pr[k] /= sum;
  }
  return T;
}


int mrQ_read (mr_State *R, const mr_Seq *q, mr_SeqWork *wk,
              const float *line, int width, mr_Buffer *out) {
  int T = forward(q, wk, line, width), t, k, prev = MR_SEQBLANK;
  for (t = 0; t < T; t++) {  /* best path: drop repeats and blanks */
    const float *pr = wk->prob + mr_cast(size_t, t) * q->nclasses;
    int best = 0;
    for (k = 1; k < q->nclasses; k++)
      if (pr[k] > pr[best]) best = k;
    if (best != MR_SEQBLANK && best != prev) {
      int status = mrB_addutf8(R, out, q->classes[best]);
      if (status != MR_OK) return status;
    }
    prev = best;
  }
  return MR_OK;
}

/* }====================================================== */


/*
** {======================================================
** CTC loss
** =======================================================
*/

static float lse (float a, float b) {
  float m = (a > b) ? a : b;
  if (m <= NEGINF) return NEGINF;
  return m + logf(expf(a - m) + expf(b - m));
}


static float logp (const mr_SeqWork *wk, int t, int k) {
  float v = wk->prob[mr_cast(size_t, t) * wk->K + k];
  return logf(v > 1e-30f ? v : 1e-30f);
}


/*
** Loss and gradient (on the logits, before softmax) for 'target'.
** Returns a negative number if T is too short for the text.
*/
static float ctc (mr_SeqWork *wk, int T, const int *target, int n) {
  int S = 2 * n + 1, s, t, k, need = n, K = wk->K;
  int *e = wk->ext;
  float *A = wk->alpha, *B = wk->beta, lp;
  for (k = 1; k < n; k++)
    if (target[k] == target[k - 1]) need++;
  if (n == 0 || need > T || S > MAXS) return -1;
  for (s = 0; s < S; s++) e[s] = (s & 1) ? target[s / 2] : MR_SEQBLANK;
  /* alpha: probability of the prefix ending in state s at time t */
  for (s = 0; s < S; s++) A[s] = NEGINF;
  A[0] = logp(wk, 0, e[0]);
  A[1] = logp(wk, 0, e[1]);
  for (t = 1; t < T; t++) {
    float *a = A + mr_cast(size_t, t) * S, *ap = a - S;
    for (s = 0; s < S; s++) {
      float v = ap[s];
      if (s >= 1) v = lse(v, ap[s - 1]);
      if (s >= 2 && e[s] != MR_SEQBLANK && e[s] != e[s - 2])
        v = lse(v, ap[s - 2]);
      a[s] = (v <= NEGINF) ? NEGINF : v + logp(wk, t, e[s]);
    }
  }
  /* beta: probability of the rest after time t, from state s */
  for (s = 0; s < S; s++)
    B[mr_cast(size_t, T - 1) * S + s] = (s >= S - 2) ? 0 : NEGINF;
  for (t = T - 2; t >= 0; t--) {
    float *b = B + mr_cast(size_t, t) * S, *bn = b + S;
    for (s = 0; s < S; s++) {
      float v = bn[s] + logp(wk, t + 1, e[s]);
      if (s + 1 < S) v = lse(v, bn[s + 1] + logp(wk, t + 1, e[s + 1]));
      if (s + 2 < S && e[s + 2] != MR_SEQBLANK && e[s + 2] != e[s])
        v = lse(v, bn[s + 2] + logp(wk, t + 1, e[s + 2]));
      b[s] = v;
    }
  }
  lp = lse(A[mr_cast(size_t, T - 1) * S + S - 1],
           A[mr_cast(size_t, T - 1) * S + S - 2]);
  if (lp <= NEGINF / 2) return -1;
  /* gradient: probability minus how much each class is used */
  for (t = 0; t < T; t++) {
    float *g = wk->dlogit + mr_cast(size_t, t) * K;
    const float *a = A + mr_cast(size_t, t) * S;
    const float *b = B + mr_cast(size_t, t) * S;
    memcpy(g, wk->prob + mr_cast(size_t, t) * K, K * sizeof(float));
    for (s = 0; s < S; s++) {
      float v = a[s] + b[s] - lp;
      if (v > -60) g[e[s]] -= expf(v);
    }
  }
  return -lp;
}

/* }====================================================== */


/*
** {======================================================
** Backward
** =======================================================
*/

static void lstmbwd (const float *wx, const float *wh, float *gwx,
                     float *gwh, float *gb, int H, int T, const float *feat,
                     const float *st, const float *dhcat, int dir,
                     float *dfeat, mr_SeqWork *wk) {
  float *dz = wk->dz, *dhn = wk->dhnext, *dcn = wk->dcnext;
  int k, j, i, rev = dir;
  memset(dhn, 0, H * sizeof(float));
  memset(dcn, 0, H * sizeof(float));
  for (k = T - 1; k >= 0; k--) {
    int t = rev ? T - 1 - k : k;
    int tp = rev ? t + 1 : t - 1;
    const float *s = st + mr_cast(size_t, t) * 6 * H;
    const float *hp = (k == 0) ? wk->zero : st + mr_cast(size_t, tp) * 6 * H
                                                + 5 * H;
    const float *cp = (k == 0) ? wk->zero : st + mr_cast(size_t, tp) * 6 * H
                                                + 4 * H;
    const float *x = feat + mr_cast(size_t, t) * FEAT;
    const float *dho = dhcat + mr_cast(size_t, t) * 2 * H + dir * H;
    float *dx = dfeat + mr_cast(size_t, t) * FEAT;
    for (j = 0; j < H; j++) {
      float ig = s[j], fg = s[H + j], gg = s[2 * H + j], og = s[3 * H + j];
      float tc = tanhf(s[4 * H + j]);
      float dh = dho[j] + dhn[j];
      float dc = dh * og * (1 - tc * tc) + dcn[j];
      dz[j] = dc * gg * ig * (1 - ig);
      dz[H + j] = dc * cp[j] * fg * (1 - fg);
      dz[2 * H + j] = dc * ig * (1 - gg * gg);
      dz[3 * H + j] = dh * tc * og * (1 - og);
      dcn[j] = dc * fg;
    }
    memset(dhn, 0, H * sizeof(float));
    for (j = 0; j < 4 * H; j++) {
      float d = dz[j];
      const float *rx = wx + mr_cast(size_t, j) * FEAT;
      const float *rh = wh + mr_cast(size_t, j) * H;
      float *gx = gwx + mr_cast(size_t, j) * FEAT;
      float *gh = gwh + mr_cast(size_t, j) * H;
      if (d == 0) continue;
      gb[j] += d;
      for (i = 0; i < FEAT; i++) {
        gx[i] += d * x[i];
        dx[i] += d * rx[i];
      }
      for (i = 0; i < H; i++) {
        gh[i] += d * hp[i];
        dhn[i] += d * rh[i];
      }
    }
  }
}


/* spread pooled gradients back to the max of each window */
static void poolbwd (int c, int h, int w, int pw, const float *act,
                     const float *dout, float *dact) {
  int k, y, x, oh = h / 2, ow = w / pw;
  memset(dact, 0, mr_cast(size_t, c) * h * w * sizeof(float));
  for (k = 0; k < c; k++) {
    for (y = 0; y < oh; y++) {
      for (x = 0; x < ow; x++) {
        size_t base = (mr_cast(size_t, k) * h + 2 * y) * w + pw * x;
        size_t best = base;
        float d = dout[(mr_cast(size_t, k) * oh + y) * ow + x];
        if (act[base + w] > act[best]) best = base + w;
        if (pw == 2) {
          if (act[base + 1] > act[best]) best = base + 1;
          if (act[base + w + 1] > act[best]) best = base + w + 1;
        }
        if (act[best] > 0) dact[best] += d;  /* ReLU: no gradient at 0 */
      }
    }
  }
}


static void convbwd (const float *wt, float *gw, float *gb, int cin,
                     int cout, int h, int w, const float *in,
                     const float *dout, float *din) {
  int co, ci, ky, kx, y, x;
  if (din != NULL)
    memset(din, 0, mr_cast(size_t, cin) * h * w * sizeof(float));
  for (co = 0; co < cout; co++) {
    const float *d = dout + mr_cast(size_t, co) * h * w;
    float sum = 0;
    for (x = 0; x < h * w; x++) sum += d[x];
    gb[co] += sum;
    for (ci = 0; ci < cin; ci++) {
      for (ky = 0; ky < 3; ky++) {
        for (kx = 0; kx < 3; kx++) {
          size_t wi = ((mr_cast(size_t, co) * cin + ci) * 3 + ky) * 3 + kx;
          float wv = wt[wi], g = 0;
          int xa = (kx == 0) ? 1 : 0, xb = (kx == 2) ? w - 1 : w;
          for (y = 0; y < h; y++) {
            int iy = y + ky - 1;
            const float *src, *dr;
            if (iy < 0 || iy >= h) continue;
            src = in + (mr_cast(size_t, ci) * h + iy) * w + (kx - 1);
            dr = d + mr_cast(size_t, y) * w;
            for (x = xa; x < xb; x++) g += dr[x] * src[x];
            if (din != NULL) {
              float *di = din + (mr_cast(size_t, ci) * h + iy) * w + (kx - 1);
              for (x = xa; x < xb; x++) di[x] += wv * dr[x];
            }
          }
          gw[wi] += g;
        }
      }
    }
  }
}


float mrQ_learn (const mr_Seq *q, mr_SeqWork *wk, const float *line,
                 int width, const int *target, int n) {
  Parts pt;
  const float *p = q->p;
  float *g = wk->grad, loss;
  int T, t, k, j, l, H = q->hidden, K = q->nclasses;
  parts(K, H, &pt);
  T = forward(q, wk, line, width);
  loss = ctc(wk, T, target, n);
  if (loss < 0) return -1;
  /* output layer */
  memset(wk->dhcat, 0, mr_cast(size_t, T) * 2 * H * sizeof(float));
  for (t = 0; t < T; t++) {
    const float *hc = wk->hcat + mr_cast(size_t, t) * 2 * H;
    const float *dl = wk->dlogit + mr_cast(size_t, t) * K;
    float *dh = wk->dhcat + mr_cast(size_t, t) * 2 * H;
    for (k = 0; k < K; k++) {
      const float *row = p + pt.wo + mr_cast(size_t, k) * 2 * H;
      float *grow = g + pt.wo + mr_cast(size_t, k) * 2 * H, d = dl[k];
      g[pt.bo + k] += d;
      for (j = 0; j < 2 * H; j++) {
        grow[j] += d * hc[j];
        dh[j] += d * row[j];
      }
    }
  }
  /* LSTMs */
  memset(wk->dfeat, 0, mr_cast(size_t, T) * FEAT * sizeof(float));
  lstmbwd(p + pt.wx[0], p + pt.wh[0], g + pt.wx[0], g + pt.wh[0],
          g + pt.lb[0], H, T, wk->feat, wk->st[0], wk->dhcat, 0, wk->dfeat,
          wk);
  lstmbwd(p + pt.wx[1], p + pt.wh[1], g + pt.wx[1], g + pt.wh[1],
          g + pt.lb[1], H, T, wk->feat, wk->st[1], wk->dhcat, 1, wk->dfeat,
          wk);
  /* features -> last pooled output */
  for (t = 0; t < T; t++)
    for (k = 0; k < 64; k++)
      for (j = 0; j < 2; j++)
        wk->dpin[NCONV][(mr_cast(size_t, k) * 2 + j) * T + t] =
          wk->dfeat[mr_cast(size_t, t) * FEAT + k * 2 + j];
  /* conv layers, last to first */
  for (l = NCONV - 1; l >= 0; l--) {
    int h = convh(l), w = convw(l, width);
    const float *in = (l == 0) ? line : wk->pin[l];
    poolbwd(convout[l], h, w, poolw[l], wk->act[l], wk->dpin[l + 1],
            wk->dact);
    convbwd(p + pt.cw[l], g + pt.cw[l], g + pt.cb[l], convin[l],
            convout[l], h, w, in, wk->dact, (l == 0) ? NULL : wk->dpin[l]);
  }
  return loss;
}

/* }====================================================== */
