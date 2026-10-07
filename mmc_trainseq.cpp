/*
** mmc_trainseq.cpp
** Train the handwriting line model (CNN + LSTM + CTC) from IAM lines and
** cursive fonts (like luac.c, it uses internal headers)
** See Copyright Notice in mr.h
*/

#define mmc_trainseq_cpp

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mr.h"
#include "mrbin.h"
#include "mrbuf.h"
#include "mrfile.h"
#include "mrfont.h"
#include "mrimage.h"
#include "mrmem.h"
#include "mrrand.h"
#include "mrseq.h"
#include "mrstate.h"
#include "mrthread.h"


#define MAXTEXT  256  /* code points per line */
#define MAXFONTS  16
#define DEF_OUTPUT  "cursive.mrm"
#define DEF_EPOCHS  30
#define DEF_HIDDEN  128
#define DEF_RATE  0.0005f
#define DEF_SYNTH  30  /* percent of synthetic (font) lines */
#define VALLINES  400  /* validation lines checked per epoch */
#define CLIP  5.0f  /* largest gradient norm */

#define ADAM_B1  0.9f
#define ADAM_B2  0.999f
#define ADAM_EPS  1e-8f


static const char *progname = "mmc_trainseq";


/* cursive and handwriting-like fonts for synthetic lines (Windows) */
static const char *const scriptfonts[] = {
  "segoesc.ttf", "segoescb.ttf", "LHANDW.TTF", "FREESCPT.TTF",
  "VLADIMIR.TTF", "FRSCRIPT.TTF", "BRADHITC.TTF", "segoepr.ttf",
  "Inkfree.ttf", "PRISTINA.TTF", "GIGI.TTF", "comic.ttf", NULL
};


/* letters beyond ASCII the model learns (same as the letter model) */
static const uint32_t extras[] = {
  0xD1, 0xF1, 0x201C, 0x201D, 0x2018, 0x2019, 0x2013, 0x2014, 0x2022,
  0x2026, 0x20B1, 0x20AC, 0xA3, 0xA5, 0xA2, 0xB0, 0xA9, 0xAE, 0x2122,
  0xD7, 0xF7, 0xB1, 0xA7, 0xB6, 0xBD, 0xBC, 0xBE, 0xE1, 0xE9, 0xED,
  0xF3, 0xFA, 0xFC, 0xC1, 0xC9, 0xCD, 0xD3, 0xDA, 0xDC, 0xBF, 0xA1
};
#define NEXTRAS  (mr_cast(int, sizeof(extras) / sizeof(extras[0])))
#define NCLASSES  (2 + 94 + NEXTRAS)  /* blank, space, '!'..'~', extras */


/* one line ready for training: 32 rows of bytes, and its text */
typedef struct Sample {
  mr_byte *pix;  /* MR_SEQH x w */
  int w;
  int *target;  /* class of each letter */
  int n;
} Sample;


typedef struct Set {
  Sample *s;
  int n, cap;
} Set;


typedef struct Options {
  const char *iam;
  const char *output;
  const char *test;
  int epochs, hidden, threads, synth, testlines;
  float rate;
  long seed;
} Options;


/* shared by the worker threads during one batch */
typedef struct Batch {
  const mr_Seq *q;
  mr_SeqWork *wk[MR_MAXTHREADS];
  float *line[MR_MAXTHREADS];
  const Sample *items[MR_MAXTHREADS][8];
  int nitems[MR_MAXTHREADS];
  float stretch[MR_MAXTHREADS][8];
  float loss[MR_MAXTHREADS];
  int used[MR_MAXTHREADS];
} Batch;


typedef struct Trainer {
  mr_State *R;
  mr_Rand rng;
  mr_Seq *q;
  uint32_t classes[NCLASSES];
  mr_Font *fonts[MAXFONTS];
  int nfonts;
  Set train, val, test, text;  /* 'text': IAM lines for synthetic text */
  float *m, *v;  /* Adam moments */
  long step;
} Trainer;


/*
** {======================================================
** Small helpers
** =======================================================
*/

static void l_message (const char *msg) {
  fprintf(stderr, "%s: %s\n", progname, msg);
  fflush(stderr);
}


static int classof (const Trainer *t, uint32_t cp) {
  int k;
  for (k = 1; k < NCLASSES; k++)
    if (t->classes[k] == cp) return k;
  return -1;
}


/* decode UTF-8 into classes; returns the count, -1 if a letter is unknown */
static int totarget (const Trainer *t, const char *ss, int *out, int max) {
  const unsigned char *s = mr_cast(const unsigned char *, ss);
  int n = 0;
  while (*s && n < max) {
    uint32_t c = *s++;
    int k;
    if (c >= 0xC0 && c < 0xE0 && (s[0] & 0xC0) == 0x80)
      c = ((c & 0x1F) << 6) | (*s++ & 0x3F);
    else if (c >= 0xE0 && c < 0xF0 && (s[0] & 0xC0) == 0x80 &&
             (s[1] & 0xC0) == 0x80) {
      c = ((c & 0x0F) << 12) | ((s[0] & 0x3F) << 6) | (s[1] & 0x3F);
      s += 2;
    }
    k = classof(t, c);
    if (k < 0) return -1;
    out[n++] = k;
  }
  return n;
}


static int addsample (mr_State *R, Set *set, const float *line, int w,
                      const int *target, int n) {
  Sample *s;
  int i;
  if (set->n == set->cap) {
    size_t cap = mr_cast(size_t, set->cap);
    Sample *ns = mrM_growvector(R, set->s, set->n, &cap, Sample);
    if (ns == NULL) return MR_ERRMEM;
    set->s = ns;
    set->cap = mr_cast(int, cap);
  }
  s = &set->s[set->n];
  s->w = w;
  s->n = n;
  s->pix = mrM_newarray(R, mr_cast(size_t, MR_SEQH) * w, mr_byte);
  s->target = mrM_newarray(R, n > 0 ? n : 1, int);
  if (s->pix == NULL || s->target == NULL) return MR_ERRMEM;
  for (i = 0; i < MR_SEQH * w; i++)
    s->pix[i] = mr_cast(mr_byte, line[i] * 255 + 0.5f);
  memcpy(s->target, target, n * sizeof(int));
  set->n++;
  return MR_OK;
}


static void freeset (mr_State *R, Set *set) {
  int i;
  for (i = 0; i < set->n; i++) {
    mrM_freearray(R, set->s[i].pix, mr_cast(size_t, MR_SEQH) * set->s[i].w);
    mrM_freearray(R, set->s[i].target, set->s[i].n > 0 ? set->s[i].n : 1);
  }
  if (set->s != NULL) mrM_freearray(R, set->s, set->cap);
  memset(set, 0, sizeof(*set));
}


/* gray image -> ink bitmap -> normalized line; returns its width */
static int prepare (mr_State *R, const mr_Image *img, float stretch,
                    float *line, int *status) {
  mr_Bitmap *bm = NULL;
  int w = 0;
  *status = mrK_fromgray(R, img, &bm);
  if (*status == MR_OK)
    w = mrQ_normalize(bm->bits, bm->width, bm->height, stretch, line);
  mrK_free(R, bm);
  return w;
}

/* }====================================================== */


/*
** {======================================================
** Data: IAM lines and synthetic cursive lines
** =======================================================
*/

static int loadiam (Trainer *t, const char *dir, const char *split, Set *set,
                    int max, float *line) {
  char path[MR_PATHSIZE], buf[2048];
  int target[MAXTEXT], skipped = 0;
  FILE *lst;
  snprintf(path, sizeof(path), "%s/%s.txt", dir, split);
  lst = mrF_open(path, "rb");
  if (lst == NULL)
    return mrS_error(t->R, MR_ERRFILE,
                     "cannot open '%s' (run ./getdata.sh iam)", path);
  while (fgets(buf, sizeof(buf), lst) != NULL && (max <= 0 || set->n < max)) {
    char *tab = strchr(buf, '\t'), *end;
    mr_Image *img = NULL;
    int n, w, status;
    if (tab == NULL) continue;
    *tab = '\0';
    end = tab + 1 + strcspn(tab + 1, "\r\n");
    *end = '\0';
    n = totarget(t, tab + 1, target, MAXTEXT);
    if (n <= 0) {
      skipped++;
      continue;
    }
    snprintf(path, sizeof(path), "%s/%s/%s", dir, split, buf);
    status = mrI_load(t->R, path, &img);
    if (status == MR_OK) status = mrI_togray(t->R, img);
    if (status != MR_OK) {
      mrI_free(t->R, img);
      fclose(lst);
      return status;
    }
    w = prepare(t->R, img, 1.0f, line, &status);
    mrI_free(t->R, img);
    if (status != MR_OK) {
      fclose(lst);
      return status;
    }
    if (w == 0 || w / MR_SEQSTEP < n + 2) {
      skipped++;
      continue;
    }
    status = addsample(t->R, set, line, w, target, n);
    if (status != MR_OK) {
      fclose(lst);
      return status;
    }
  }
  fclose(lst);
  printf("IAM %s: %d lines (%d skipped)\n", split, set->n, skipped);
  return MR_OK;
}


/* a line of text from IAM, with some letters swapped for extras */
static int synthtext (Trainer *t, int *target) {
  const Sample *src = &t->text.s[mrR_int(&t->rng, t->text.n)];
  int n = src->n, i;
  memcpy(target, src->target, n * sizeof(int));
  for (i = 0; i < n; i++) {
    uint32_t c = t->classes[target[i]];
    if (c == 'n' && mrR_int(&t->rng, 6) == 0) c = 0xF1;
    else if (c == 'N' && mrR_int(&t->rng, 6) == 0) c = 0xD1;
    else if (strchr("aeiou", mr_cast(int, c)) != NULL &&
             mrR_int(&t->rng, 30) == 0)
      c = extras[27 + (strchr("aeiou", mr_cast(int, c)) - "aeiou")];
    else if (c != ' ' && mrR_int(&t->rng, 60) == 0)
      c = extras[mrR_int(&t->rng, NEXTRAS)];
    target[i] = classof(t, c);
  }
  return n;
}


/* draw a synthetic cursive line; returns its width (0 if it failed) */
static int synthline (Trainer *t, float *line, int *target, int *n) {
  uint32_t cps[MAXTEXT];
  const mr_Font *f;
  mr_Draw d;
  mr_Image *img = NULL;
  mr_Box boxes[MAXTEXT];
  int i, w, status, fi;
  if (t->nfonts == 0 || t->text.n == 0) return 0;
  fi = mrR_int(&t->rng, t->nfonts);
  f = t->fonts[fi];
  *n = synthtext(t, target);
  for (i = 0; i < *n; i++) {
    cps[i] = t->classes[target[i]];
    if (cps[i] != ' ' && !mrT_hasglyph(f, cps[i])) {  /* font lacks it */
      cps[i] = 'e';
      target[i] = classof(t, 'e');
    }
  }
  d.height = mrR_range(&t->rng, 28, 60);
  d.stretch = mrR_range(&t->rng, 0.85f, 1.2f);
  d.spacing = 0;
  d.ink = mrR_int(&t->rng, 90);
  d.paper = 170 + mrR_int(&t->rng, 86);
  d.pad = 4;
  status = mrT_drawline(t->R, f, cps, *n, &d, &img, boxes, NULL);
  if (status != MR_OK) return 0;
  w = prepare(t->R, img, 1.0f, line, &status);
  mrI_free(t->R, img);
  if (status != MR_OK || w / MR_SEQSTEP < *n + 2) return 0;
  return w;
}


static int loadfonts (Trainer *t) {
  char path[MR_PATHSIZE];
  int i;
  for (i = 0; scriptfonts[i] != NULL && t->nfonts < MAXFONTS; i++) {
    mr_Font *f;
    snprintf(path, sizeof(path), "C:/Windows/Fonts/%s", scriptfonts[i]);
    if (mrT_load(t->R, path, 0, &f) == MR_OK) t->fonts[t->nfonts++] = f;
  }
  printf("script fonts: %d\n", t->nfonts);
  return MR_OK;
}

/* }====================================================== */


/*
** {======================================================
** Training
** =======================================================
*/

/* stretch a stored sample horizontally into 'line'; returns the width */
static int unpack (const Sample *s, float stretch, float *line) {
  int w = mr_cast(int, s->w * stretch);
  int x, y;
  w = (w + MR_SEQSTEP - 1) / MR_SEQSTEP * MR_SEQSTEP;
  if (w < 4 * MR_SEQSTEP) w = 4 * MR_SEQSTEP;
  if (w > MR_SEQMAXW) w = MR_SEQMAXW;
  for (y = 0; y < MR_SEQH; y++) {
    const mr_byte *row = s->pix + mr_cast(size_t, y) * s->w;
    for (x = 0; x < w; x++) {
      float fx = (x + 0.5f) * s->w / w - 0.5f;
      int x0 = mr_cast(int, floorf(fx));
      float a = fx - x0, v0, v1;
      v0 = (x0 >= 0 && x0 < s->w) ? row[x0] : 0;
      v1 = (x0 + 1 >= 0 && x0 + 1 < s->w) ? row[x0 + 1] : 0;
      line[y * w + x] = ((1 - a) * v0 + a * v1) / 255.0f;
    }
  }
  return w;
}


static void work (void *ctx, int i) {
  Batch *b = mr_cast(Batch *, ctx);
  int k;
  b->loss[i] = 0;
  b->used[i] = 0;
  for (k = 0; k < b->nitems[i]; k++) {
    const Sample *s = b->items[i][k];
    int w = unpack(s, b->stretch[i][k], b->line[i]);
    float l = mrQ_learn(b->q, b->wk[i], b->line[i], w, s->target, s->n);
    if (l >= 0) {
      b->loss[i] += l;
      b->used[i]++;
    }
  }
}


static void adam (Trainer *t, const float *g, size_t n, float rate,
                  float scale) {
  float c1, c2, *p = t->q->p;
  size_t i;
  t->step++;
  c1 = 1 - powf(ADAM_B1, mr_cast(float, t->step));
  c2 = 1 - powf(ADAM_B2, mr_cast(float, t->step));
  for (i = 0; i < n; i++) {
    float gi = g[i] * scale;
    t->m[i] = ADAM_B1 * t->m[i] + (1 - ADAM_B1) * gi;
    t->v[i] = ADAM_B2 * t->v[i] + (1 - ADAM_B2) * gi * gi;
    p[i] -= rate * (t->m[i] / c1) / (sqrtf(t->v[i] / c2) + ADAM_EPS);
  }
}


/* edit distance between class strings */
static int distance (const int *a, int na, const int *b, int nb, int *row) {
  int i, j;
  for (j = 0; j <= nb; j++) row[j] = j;
  for (i = 1; i <= na; i++) {
    int diag = row[0];
    row[0] = i;
    for (j = 1; j <= nb; j++) {
      int up = row[j], best = diag + (a[i - 1] != b[j - 1]);
      if (up + 1 < best) best = up + 1;
      if (row[j - 1] + 1 < best) best = row[j - 1] + 1;
      row[j] = best;
      diag = up;
    }
  }
  return row[nb];
}


/* character error rate (percent) on the first 'max' lines of 'set' */
static float evaluate (Trainer *t, mr_SeqWork *wk, float *line,
                       const Set *set, int max, int show) {
  mr_Buffer text;
  int got[MAXTEXT * 2], row[MAXTEXT * 2 + 1], i, shown = 0;
  long errs = 0, total = 0;
  mrB_init(&text);
  for (i = 0; i < set->n && i < max; i++) {
    const Sample *s = &set->s[i];
    int w = unpack(s, 1.0f, line), n, d;
    mrB_reset(&text);
    if (mrQ_read(t->R, t->q, wk, line, w, &text) != MR_OK) break;
    n = totarget(t, mrB_cstr(&text), got, MAXTEXT * 2);
    if (n < 0) n = 0;
    d = distance(s->target, s->n, got, n, row);
    errs += d;
    total += s->n;
    if (d > 0 && shown < show) {
      mr_Buffer want;
      int k;
      mrB_init(&want);
      for (k = 0; k < s->n; k++)
        mrB_addutf8(t->R, &want, t->classes[s->target[k]]);
      printf("  want: %s\n  got:  %s\n", mrB_cstr(&want), mrB_cstr(&text));
      mrB_free(t->R, &want);
      shown++;
    }
  }
  mrB_free(t->R, &text);
  return total ? 100.0f * errs / total : 0;
}


static int train (Trainer *t, const Options *opt) {
  mr_State *R = t->R;
  Batch b;
  Set synth;
  float *line0 = NULL, best = 1e9f;
  int e, i, nt = opt->threads, status = MR_OK;
  int target[MAXTEXT];
  size_t np = t->q->np;
  clock_t start = clock();
  memset(&b, 0, sizeof(b));
  memset(&synth, 0, sizeof(synth));
  b.q = t->q;
  t->m = mrM_newarray(R, np, float);
  t->v = mrM_newarray(R, np, float);
  line0 = mrM_newarray(R, mr_cast(size_t, MR_SEQH) * MR_SEQMAXW, float);
  if (t->m == NULL || t->v == NULL || line0 == NULL)
    goto nomem;
  memset(t->m, 0, np * sizeof(float));
  memset(t->v, 0, np * sizeof(float));
  for (i = 0; i < nt; i++) {
    status = mrQ_newwork(R, t->q, &b.wk[i]);
    if (status != MR_OK) goto done;
    b.line[i] = mrM_newarray(R, mr_cast(size_t, MR_SEQH) * MR_SEQMAXW,
                             float);
    if (b.line[i] == NULL) goto nomem;
  }
  printf("model: %d classes, LSTM %d, %lu weights, %d threads\n",
         t->q->nclasses, t->q->hidden, mr_cast(unsigned long, np), nt);
  for (e = 1; e <= opt->epochs; e++) {
    float rate = opt->rate, loss = 0, cer;
    long used = 0;
    int nsynth = t->train.n * opt->synth / (100 - opt->synth), pos;
    if (e > opt->epochs * 2 / 3) rate *= 0.3f;  /* slow down at the end */
    /* fresh synthetic lines each epoch */
    freeset(R, &synth);
    for (i = 0; i < nsynth; i++) {
      int n, w = synthline(t, line0, target, &n);
      if (w > 0) {
        status = addsample(R, &synth, line0, w, target, n);
        if (status != MR_OK) goto done;
      }
    }
    {  /* shuffled order over IAM + synthetic */
      int total = t->train.n + synth.n;
      int *no = mrM_newarray(R, total + 1, int);
      if (no == NULL) goto nomem;
      for (i = 0; i < total; i++) no[i] = i;
      for (i = total - 1; i > 0; i--) {
        int j = mrR_int(&t->rng, i + 1), tmp = no[i];
        no[i] = no[j];
        no[j] = tmp;
      }
      for (pos = 0; pos < total; ) {
        int k, w0;
        float norm = 0, scale;
        size_t j;
        for (i = 0; i < nt; i++) {
          b.nitems[i] = 0;
          for (k = 0; k < 2 && pos < total; k++, pos++) {
            int idx = no[pos];
            b.items[i][k] = (idx < t->train.n) ? &t->train.s[idx]
                                               : &synth.s[idx - t->train.n];
            b.stretch[i][k] = mrR_range(&t->rng, 0.8f, 1.2f);
            b.nitems[i]++;
          }
        }
        mrX_parallel(nt, work, &b);
        /* sum gradients into worker 0 */
        w0 = 0;
        for (i = 0; i < nt; i++) {
          loss += b.loss[i];
          w0 += b.used[i];
          if (i > 0) {
            float *g0 = mrQ_grad(b.wk[0]), *gi = mrQ_grad(b.wk[i]);
            for (j = 0; j < np; j++) g0[j] += gi[j];
            mrQ_cleargrad(t->q, b.wk[i]);
          }
        }
        if (w0 > 0) {
          float *g0 = mrQ_grad(b.wk[0]);
          for (j = 0; j < np; j++) norm += g0[j] * g0[j];
          norm = sqrtf(norm) / w0;
          scale = 1.0f / w0;
          if (norm > CLIP) scale *= CLIP / norm;
          adam(t, g0, np, rate, scale);
          used += w0;
        }
        mrQ_cleargrad(t->q, b.wk[0]);
      }
      mrM_freearray(R, no, total + 1);
    }
    cer = evaluate(t, b.wk[0], b.line[0], &t->val, VALLINES, 0);
    printf("epoch %3d  loss %.3f  lines %ld  validation %.2f%% errors  "
           "%.0fs\n", e, used ? loss / used : 0.0f, used, cer,
           mr_cast(double, clock() - start) / CLOCKS_PER_SEC);
    fflush(stdout);
    if (cer < best) {
      best = cer;
      status = mrQ_save(R, t->q, opt->output);
      if (status != MR_OK) goto done;
      printf("  saved %s\n", opt->output);
    }
  }
  goto done;
 nomem:
  status = MR_ERRMEM;
 done:
  freeset(R, &synth);
  for (i = 0; i < nt; i++) {
    mrQ_freework(R, b.wk[i]);
    if (b.line[i] != NULL)
      mrM_freearray(R, b.line[i], mr_cast(size_t, MR_SEQH) * MR_SEQMAXW);
  }
  if (line0 != NULL)
    mrM_freearray(R, line0, mr_cast(size_t, MR_SEQH) * MR_SEQMAXW);
  if (t->m != NULL) mrM_freearray(R, t->m, np);
  if (t->v != NULL) mrM_freearray(R, t->v, np);
  t->m = t->v = NULL;
  return status;
}

/* }====================================================== */


static void print_usage (const char *bad) {
  if (bad != NULL) fprintf(stderr, "%s: bad option '%s'\n", progname, bad);
  fprintf(stderr,
    "usage: %s [options]\n"
    "Train the handwriting line model from IAM lines and cursive fonts.\n"
    "Available options are:\n"
    "  -I dir    IAM folder (default build/data/iam, see getdata.sh)\n"
    "  -o file   output model (default " DEF_OUTPUT ")\n"
    "  -e count  passes over the training lines (default %d)\n"
    "  -k cells  LSTM cells per direction (default %d)\n"
    "  -r rate   learning rate (default 0.0005)\n"
    "  -j count  threads (default: CPU cores - 1)\n"
    "  -y pct    percent of synthetic font lines (default %d)\n"
    "  -s seed   random seed (default 1)\n"
    "  -t model  test 'model' on the IAM test lines instead\n"
    "  -c lines  test lines (default all)\n"
    "  -h        print this help\n",
    progname, DEF_EPOCHS, DEF_HIDDEN, DEF_SYNTH);
}


static int collectargs (int argc, char **argv, Options *opt) {
  int i, cpus = mrX_cpus();
  opt->iam = "build/data/iam";
  opt->output = DEF_OUTPUT;
  opt->test = NULL;
  opt->epochs = DEF_EPOCHS;
  opt->hidden = DEF_HIDDEN;
  opt->threads = (cpus > 1) ? cpus - 1 : 1;
  opt->synth = DEF_SYNTH;
  opt->testlines = 0;
  opt->rate = DEF_RATE;
  opt->seed = 1;
  for (i = 1; i < argc; i++) {
    const char *a = argv[i], *val;
    if (a[0] != '-' || a[1] == '\0' || a[2] != '\0') goto bad;
    if (a[1] == 'h') {
      print_usage(NULL);
      return -1;
    }
    if (i + 1 >= argc) goto bad;
    val = argv[++i];
    switch (a[1]) {
      case 'I': opt->iam = val; break;
      case 'o': opt->output = val; break;
      case 't': opt->test = val; break;
      case 'e': opt->epochs = atoi(val); break;
      case 'k': opt->hidden = atoi(val); break;
      case 'j': opt->threads = atoi(val); break;
      case 'y': opt->synth = atoi(val); break;
      case 'c': opt->testlines = atoi(val); break;
      case 'r': opt->rate = mr_cast(float, atof(val)); break;
      case 's': opt->seed = atol(val); break;
      default: goto bad;
    }
  }
  if (opt->threads < 1) opt->threads = 1;
  if (opt->threads > MR_MAXTHREADS) opt->threads = MR_MAXTHREADS;
  if (opt->synth < 0) opt->synth = 0;
  if (opt->synth > 90) opt->synth = 90;
  return 0;
 bad:
  print_usage(argv[i]);
  return 2;
}


int main (int argc, char **argv) {
  Options opt;
  Trainer t;
  float *line = NULL;
  mr_SeqWork *wk = NULL;
  int i, status = collectargs(argc, argv, &opt);
  if (status != 0) return (status < 0) ? EXIT_SUCCESS : status;
  memset(&t, 0, sizeof(t));
  t.R = mr_open();
  if (t.R == NULL) {
    l_message("not enough memory");
    return EXIT_FAILURE;
  }
  mrR_seed(&t.rng, mr_cast(uint64_t, opt.seed));
  t.classes[0] = 0;  /* blank */
  t.classes[1] = ' ';
  for (i = 0; i < 94; i++) t.classes[2 + i] = mr_cast(uint32_t, '!' + i);
  for (i = 0; i < NEXTRAS; i++) t.classes[96 + i] = extras[i];
  line = mrM_newarray(t.R, mr_cast(size_t, MR_SEQH) * MR_SEQMAXW, float);
  if (line == NULL) {
    status = MR_ERRMEM;
    goto done;
  }
  if (opt.test != NULL) {
    status = mrQ_load(t.R, opt.test, &t.q);
    if (status == MR_OK)
      status = loadiam(&t, opt.iam, "test", &t.test, opt.testlines, line);
    if (status == MR_OK) status = mrQ_newwork(t.R, t.q, &wk);
    if (status == MR_OK)
      printf("test: %d lines, %.2f%% character errors\n", t.test.n,
             evaluate(&t, wk, line, &t.test, t.test.n, 8));
    goto done;
  }
  status = loadiam(&t, opt.iam, "train", &t.train, 0, line);
  if (status == MR_OK)
    status = loadiam(&t, opt.iam, "validation", &t.val, 0, line);
  if (status != MR_OK) goto done;
  t.text = t.train;  /* sentences for synthetic lines (shared, not owned) */
  loadfonts(&t);
  status = mrQ_new(t.R, NCLASSES, opt.hidden, &t.q);
  if (status != MR_OK) goto done;
  memcpy(t.q->classes, t.classes, sizeof(t.classes));
  mrQ_randomize(t.q, &t.rng);
  status = train(&t, &opt);
 done:
  if (status != MR_OK) l_message(mr_geterror(t.R));
  mrQ_freework(t.R, wk);
  if (line != NULL)
    mrM_freearray(t.R, line, mr_cast(size_t, MR_SEQH) * MR_SEQMAXW);
  mrQ_free(t.R, t.q);
  freeset(t.R, &t.train);
  freeset(t.R, &t.val);
  freeset(t.R, &t.test);
  for (i = 0; i < t.nfonts; i++) mrT_free(t.R, t.fonts[i]);
  mr_close(t.R);
  return (status == MR_OK) ? EXIT_SUCCESS : EXIT_FAILURE;
}
