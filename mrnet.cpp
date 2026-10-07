/*
** mrnet.cpp
** Neural network: a small multi-layer perceptron, its training and its
** model file
** See Copyright Notice in mr.h
*/

#define mrnet_cpp
#define MR_CORE

#include "mrnet.h"
#include "mrfile.h"
#include "mrglyph.h"
#include "mrmem.h"
#include "mrstate.h"

#include <math.h>
#include <string.h>


#define MODELMAGIC  "MRNN"
#define MODELVERSION  2
#define MAXSIZE  65536

#define ADAM_B1  0.9f
#define ADAM_B2  0.999f
#define ADAM_EPS  1e-8f


struct mr_Trainer {
  mr_State *R;
  int nlayers;
  int sizes[MR_MAXLAYERS + 1];
  float *gw[MR_MAXLAYERS], *gb[MR_MAXLAYERS];  /* gradients */
  float *mw[MR_MAXLAYERS], *mb[MR_MAXLAYERS];  /* Adam 1st moments */
  float *vw[MR_MAXLAYERS], *vb[MR_MAXLAYERS];  /* Adam 2nd moments */
  float *act[MR_MAXLAYERS + 1];  /* outputs of each layer, one sample */
  float *delta, *delta2;  /* back-propagated errors */
  long step;
};


static size_t wcount (const int *sizes, int l) {
  return mr_cast(size_t, sizes[l + 1]) * mr_cast(size_t, sizes[l]);
}


static int maxsize (const int *sizes, int nlayers) {
  int l, m = 0;
  for (l = 0; l <= nlayers; l++)
    if (sizes[l] > m) m = sizes[l];
  return m;
}


/*
** {======================================================
** Network
** =======================================================
*/

int mrN_new (mr_State *R, int nlayers, const int *sizes, mr_Net **out) {
  mr_Net *net;
  int l;
  *out = NULL;
  if (nlayers < 1 || nlayers > MR_MAXLAYERS)
    return mrS_error(R, MR_ERRARG, "bad number of layers %d", nlayers);
  for (l = 0; l <= nlayers; l++)
    if (sizes[l] < 1 || sizes[l] > MAXSIZE)
      return mrS_error(R, MR_ERRARG, "bad layer size %d", sizes[l]);
  net = mrM_new(R, mr_Net);
  if (net == NULL) return MR_ERRMEM;
  memset(net, 0, sizeof(*net));
  net->nlayers = nlayers;
  for (l = 0; l <= nlayers; l++) net->sizes[l] = sizes[l];
  net->classes = mrM_newarray(R, sizes[nlayers], uint32_t);
  if (net->classes == NULL) goto nomem;
  memset(net->classes, 0, sizes[nlayers] * sizeof(uint32_t));
  for (l = 0; l < nlayers; l++) {
    net->w[l] = mrM_newarray(R, wcount(sizes, l), float);
    net->b[l] = mrM_newarray(R, sizes[l + 1], float);
    if (net->w[l] == NULL || net->b[l] == NULL) goto nomem;
    memset(net->w[l], 0, wcount(sizes, l) * sizeof(float));
    memset(net->b[l], 0, sizes[l + 1] * sizeof(float));
  }
  *out = net;
  return MR_OK;
 nomem:
  mrN_free(R, net);
  return MR_ERRMEM;
}


void mrN_free (mr_State *R, mr_Net *net) {
  int l;
  if (net == NULL) return;
  for (l = 0; l < net->nlayers; l++) {
    if (net->w[l] != NULL)
      mrM_freearray(R, net->w[l], wcount(net->sizes, l));
    if (net->b[l] != NULL)
      mrM_freearray(R, net->b[l], net->sizes[l + 1]);
  }
  if (net->classes != NULL)
    mrM_freearray(R, net->classes, net->sizes[net->nlayers]);
  mrM_delete(R, net);
}


/* He initialization, good for ReLU */
void mrN_randomize (mr_Net *net, mr_Rand *rng) {
  int l;
  for (l = 0; l < net->nlayers; l++) {
    size_t i, n = wcount(net->sizes, l);
    float sd = sqrtf(2.0f / mr_cast(float, net->sizes[l]));
    for (i = 0; i < n; i++) net->w[l][i] = mrR_normal(rng) * sd;
    memset(net->b[l], 0, net->sizes[l + 1] * sizeof(float));
  }
}


int mrN_nclasses (const mr_Net *net) {
  return net->sizes[net->nlayers];
}


size_t mrN_worksize (const mr_Net *net) {
  return 2 * mr_cast(size_t, maxsize(net->sizes, net->nlayers));
}


/* out = W * in + b, then ReLU unless it is the last layer */
static void layer (const mr_Net *net, int l, const float *in, float *out) {
  int i, j, nin = net->sizes[l], nout = net->sizes[l + 1];
  int last = (l == net->nlayers - 1);
  for (i = 0; i < nout; i++) {
    const float *row = net->w[l] + mr_cast(size_t, i) * nin;
    float s = net->b[l][i];
    for (j = 0; j < nin; j++) s += row[j] * in[j];
    out[i] = (!last && s < 0) ? 0 : s;
  }
}


static void softmax (float *v, int n) {
  int i;
  float m = v[0], sum = 0;
  for (i = 1; i < n; i++)
    if (v[i] > m) m = v[i];
  for (i = 0; i < n; i++) {
    v[i] = expf(v[i] - m);
    sum += v[i];
  }
  for (i = 0; i < n; i++) v[i] /= sum;
}


static int argmax (const float *v, int n) {
  int i, best = 0;
  for (i = 1; i < n; i++)
    if (v[i] > v[best]) best = i;
  return best;
}


int mrN_classify (const mr_Net *net, const float *in, float *work,
                  float *conf) {
  size_t half = mrN_worksize(net) / 2;
  float *bufs[2];
  const float *cur = in;
  int l, best, nc = mrN_nclasses(net);
  bufs[0] = work;
  bufs[1] = work + half;
  for (l = 0; l < net->nlayers; l++) {
    layer(net, l, cur, bufs[l & 1]);
    cur = bufs[l & 1];
  }
  softmax(bufs[(net->nlayers - 1) & 1], nc);
  best = argmax(cur, nc);
  if (conf != NULL) *conf = cur[best];
  return best;
}


const float *mrN_probs (const mr_Net *net, const float *work) {
  return work + ((net->nlayers - 1) & 1) * (mrN_worksize(net) / 2);
}

/* }====================================================== */


/*
** {======================================================
** Model file (all numbers little-endian)
**   "MRNN"  u32 version  u32 glyphsize  u32 nfeat  u32 nlayers
**   u32 sizes[nlayers + 1]  u32 classes[nclasses]
**   per layer: f32 w[...]  f32 b[...]
** =======================================================
*/

static int putu32 (FILE *f, uint32_t v) {
  mr_byte b[4];
  b[0] = mr_cast(mr_byte, v);
  b[1] = mr_cast(mr_byte, v >> 8);
  b[2] = mr_cast(mr_byte, v >> 16);
  b[3] = mr_cast(mr_byte, v >> 24);
  return fwrite(b, 1, 4, f) == 4;
}


static int getu32 (FILE *f, uint32_t *v) {
  mr_byte b[4];
  if (fread(b, 1, 4, f) != 4) return 0;
  *v = mr_cast(uint32_t, b[0]) | mr_cast(uint32_t, b[1]) << 8 |
       mr_cast(uint32_t, b[2]) << 16 | mr_cast(uint32_t, b[3]) << 24;
  return 1;
}


static int putfloats (FILE *f, const float *v, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    uint32_t u;
    memcpy(&u, &v[i], 4);
    if (!putu32(f, u)) return 0;
  }
  return 1;
}


static int getfloats (FILE *f, float *v, size_t n) {
  size_t i;
  for (i = 0; i < n; i++) {
    uint32_t u;
    if (!getu32(f, &u)) return 0;
    memcpy(&v[i], &u, 4);
    if (!isfinite(v[i])) return 0;
  }
  return 1;
}


int mrN_save (mr_State *R, const mr_Net *net, const char *path) {
  FILE *f;
  int l, ok;
  mr_static_assert(sizeof(float) == 4, "float must be 32 bits");
  f = mrF_open(path, "wb");
  if (f == NULL)
    return mrS_error(R, MR_ERRFILE, "cannot create model file");
  ok = fwrite(MODELMAGIC, 1, 4, f) == 4 && putu32(f, MODELVERSION) &&
       putu32(f, MR_GLYPHSIZE) && putu32(f, MR_GLYPHFEAT) &&
       putu32(f, mr_cast(uint32_t, net->nlayers));
  for (l = 0; ok && l <= net->nlayers; l++)
    ok = putu32(f, mr_cast(uint32_t, net->sizes[l]));
  for (l = 0; ok && l < mrN_nclasses(net); l++)
    ok = putu32(f, net->classes[l]);
  for (l = 0; ok && l < net->nlayers; l++)
    ok = putfloats(f, net->w[l], wcount(net->sizes, l)) &&
         putfloats(f, net->b[l], net->sizes[l + 1]);
  if (fclose(f) != 0) ok = 0;
  if (!ok) return mrS_error(R, MR_ERRFILE, "cannot write model file");
  return MR_OK;
}


int mrN_load (mr_State *R, const char *path, mr_Net **out) {
  FILE *f;
  char magic[4];
  uint32_t version, gsize, nfeat, nl, v;
  int sizes[MR_MAXLAYERS + 1];
  mr_Net *net = NULL;
  int l, status;
  *out = NULL;
  f = mrF_open(path, "rb");
  if (f == NULL)
    return mrS_error(R, MR_ERRFILE, "cannot open model '%s'", path);
  if (fread(magic, 1, 4, f) != 4 || memcmp(magic, MODELMAGIC, 4) != 0 ||
      !getu32(f, &version) || !getu32(f, &gsize) || !getu32(f, &nfeat) ||
      !getu32(f, &nl))
    goto bad;
  if (version != MODELVERSION || gsize != MR_GLYPHSIZE ||
      nfeat != MR_GLYPHFEAT || nl < 1 || nl > MR_MAXLAYERS) {
    fclose(f);
    return mrS_error(R, MR_ERRFORMAT,
                     "model '%s' is for another version of mmc_reader",
                     path);
  }
  for (l = 0; l <= mr_cast(int, nl); l++) {
    if (!getu32(f, &v) || v < 1 || v > MAXSIZE) goto bad;
    sizes[l] = mr_cast(int, v);
  }
  if (sizes[0] != MR_GLYPHINPUT) goto bad;
  status = mrN_new(R, mr_cast(int, nl), sizes, &net);
  if (status != MR_OK) {
    fclose(f);
    return status;
  }
  for (l = 0; l < mrN_nclasses(net); l++)
    if (!getu32(f, &net->classes[l]) || net->classes[l] > 0x10FFFF)
      goto bad;
  for (l = 0; l < net->nlayers; l++)
    if (!getfloats(f, net->w[l], wcount(net->sizes, l)) ||
        !getfloats(f, net->b[l], net->sizes[l + 1]))
      goto bad;
  fclose(f);
  *out = net;
  return MR_OK;
 bad:
  fclose(f);
  mrN_free(R, net);
  return mrS_error(R, MR_ERRFORMAT, "model '%s' is broken", path);
}

/* }====================================================== */


/*
** {======================================================
** Training
** =======================================================
*/

int mrN_newtrainer (mr_State *R, const mr_Net *net, mr_Trainer **out) {
  mr_Trainer *t;
  int l, m = maxsize(net->sizes, net->nlayers);
  *out = NULL;
  t = mrM_new(R, mr_Trainer);
  if (t == NULL) return MR_ERRMEM;
  memset(t, 0, sizeof(*t));
  t->R = R;
  t->nlayers = net->nlayers;
  for (l = 0; l <= net->nlayers; l++) t->sizes[l] = net->sizes[l];
  for (l = 0; l < net->nlayers; l++) {
    size_t nw = wcount(net->sizes, l);
    size_t nb = mr_cast(size_t, net->sizes[l + 1]);
    t->gw[l] = mrM_newarray(R, nw, float);
    t->mw[l] = mrM_newarray(R, nw, float);
    t->vw[l] = mrM_newarray(R, nw, float);
    t->gb[l] = mrM_newarray(R, nb, float);
    t->mb[l] = mrM_newarray(R, nb, float);
    t->vb[l] = mrM_newarray(R, nb, float);
    if (!t->gw[l] || !t->mw[l] || !t->vw[l] || !t->gb[l] || !t->mb[l] ||
        !t->vb[l])
      goto nomem;
    memset(t->mw[l], 0, nw * sizeof(float));
    memset(t->vw[l], 0, nw * sizeof(float));
    memset(t->mb[l], 0, nb * sizeof(float));
    memset(t->vb[l], 0, nb * sizeof(float));
  }
  for (l = 1; l <= net->nlayers; l++) {
    t->act[l] = mrM_newarray(R, net->sizes[l], float);
    if (t->act[l] == NULL) goto nomem;
  }
  t->delta = mrM_newarray(R, m, float);
  t->delta2 = mrM_newarray(R, m, float);
  if (t->delta == NULL || t->delta2 == NULL) goto nomem;
  *out = t;
  return MR_OK;
 nomem:
  mrN_freetrainer(R, t);
  return MR_ERRMEM;
}


void mrN_freetrainer (mr_State *R, mr_Trainer *t) {
  int l, m;
  if (t == NULL) return;
  m = maxsize(t->sizes, t->nlayers);
  for (l = 0; l < t->nlayers; l++) {
    size_t nw = wcount(t->sizes, l);
    size_t nb = mr_cast(size_t, t->sizes[l + 1]);
    if (t->gw[l]) mrM_freearray(R, t->gw[l], nw);
    if (t->mw[l]) mrM_freearray(R, t->mw[l], nw);
    if (t->vw[l]) mrM_freearray(R, t->vw[l], nw);
    if (t->gb[l]) mrM_freearray(R, t->gb[l], nb);
    if (t->mb[l]) mrM_freearray(R, t->mb[l], nb);
    if (t->vb[l]) mrM_freearray(R, t->vb[l], nb);
  }
  for (l = 1; l <= t->nlayers; l++)
    if (t->act[l]) mrM_freearray(R, t->act[l], t->sizes[l]);
  if (t->delta) mrM_freearray(R, t->delta, m);
  if (t->delta2) mrM_freearray(R, t->delta2, m);
  mrM_delete(R, t);
}


/* forward and backward pass for one sample; adds to the gradients */
static float backprop (mr_Trainer *t, const mr_Net *net, const float *in,
                       int label, int *right) {
  int l, i, j, L = net->nlayers, nc = mrN_nclasses(net);
  float *out, loss, *d = t->delta, *dn = t->delta2;
  t->act[0] = mr_cast(float *, in);  /* read only */
  for (l = 0; l < L; l++) layer(net, l, t->act[l], t->act[l + 1]);
  out = t->act[L];
  softmax(out, nc);
  *right = (argmax(out, nc) == label);
  loss = -logf(out[label] > 1e-12f ? out[label] : 1e-12f);
  for (i = 0; i < nc; i++) d[i] = out[i] - (i == label ? 1.0f : 0.0f);
  for (l = L - 1; l >= 0; l--) {
    int nin = net->sizes[l], nout = net->sizes[l + 1];
    const float *a = t->act[l];
    for (i = 0; i < nout; i++) {
      float di = d[i], *g = t->gw[l] + mr_cast(size_t, i) * nin;
      if (di == 0) continue;
      for (j = 0; j < nin; j++) g[j] += di * a[j];
      t->gb[l][i] += di;
    }
    if (l > 0) {  /* error of the layer below, through ReLU */
      for (j = 0; j < nin; j++) dn[j] = 0;
      for (i = 0; i < nout; i++) {
        const float *row = net->w[l] + mr_cast(size_t, i) * nin;
        float di = d[i];
        if (di == 0) continue;
        for (j = 0; j < nin; j++) dn[j] += row[j] * di;
      }
      for (j = 0; j < nin; j++)
        if (a[j] <= 0) dn[j] = 0;
      {
        float *tmp = d;
        d = dn;
        dn = tmp;
      }
    }
  }
  return loss;
}


static void adam (float *p, const float *g, float *m, float *v, size_t n,
                  float scale, float rate, float c1, float c2) {
  size_t i;
  for (i = 0; i < n; i++) {
    float gi = g[i] * scale;
    m[i] = ADAM_B1 * m[i] + (1 - ADAM_B1) * gi;
    v[i] = ADAM_B2 * v[i] + (1 - ADAM_B2) * gi * gi;
    p[i] -= rate * (m[i] / c1) / (sqrtf(v[i] / c2) + ADAM_EPS);
  }
}


float mrN_train (mr_Trainer *t, mr_Net *net, const float *in,
                 const int *labels, int n, float rate, int *correct) {
  int l, s, right, ok = 0;
  float loss = 0, c1, c2;
  size_t nin = mr_cast(size_t, net->sizes[0]);
  if (n <= 0) return 0;
  for (l = 0; l < net->nlayers; l++) {
    memset(t->gw[l], 0, wcount(net->sizes, l) * sizeof(float));
    memset(t->gb[l], 0, net->sizes[l + 1] * sizeof(float));
  }
  for (s = 0; s < n; s++) {
    loss += backprop(t, net, in + mr_cast(size_t, s) * nin, labels[s],
                     &right);
    ok += right;
  }
  t->step++;
  c1 = 1 - powf(ADAM_B1, mr_cast(float, t->step));
  c2 = 1 - powf(ADAM_B2, mr_cast(float, t->step));
  for (l = 0; l < net->nlayers; l++) {
    adam(net->w[l], t->gw[l], t->mw[l], t->vw[l], wcount(net->sizes, l),
         1.0f / n, rate, c1, c2);
    adam(net->b[l], t->gb[l], t->mb[l], t->vb[l],
         mr_cast(size_t, net->sizes[l + 1]), 1.0f / n, rate, c1, c2);
  }
  if (correct != NULL) *correct = ok;
  return loss / mr_cast(float, n);
}

/* }====================================================== */
