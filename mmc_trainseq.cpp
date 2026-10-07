/*
** mmc_trainseq.cpp
** Train a line model (CNN + LSTM + CTC): printed text from many fonts
** (-P), or handwriting from IAM lines plus handwriting fonts.
** Like luac.c, it uses internal headers.
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
#include "mrgpu.h"
#include "mrimage.h"
#include "mrmem.h"
#include "mrrand.h"
#include "mrseq.h"
#include "mrstate.h"
#include "mrthread.h"


#define MAXTEXT  256  /* code points per line */
#define MAXFONTS  1024
#define PERWORKER  2  /* lines per thread per step */
#define DEF_EPOCHS  30
#define DEF_HIDDEN  128
#define DEF_RATE  0.0005f
#define DEF_SYNTH  30  /* handwriting: percent of synthetic (font) lines */
#define DEF_LINES  16000  /* print: lines per epoch */
#define VALLINES  400  /* validation lines checked per epoch */
#define HOLDOUT  10  /* print: every 10th font only for validation */
#define CLIP  5.0f  /* largest gradient norm */

#define ADAM_B1  0.9f
#define ADAM_B2  0.999f
#define ADAM_EPS  1e-8f


static const char *progname = "mmc_trainseq";


/* Windows fonts for printed lines */
static const char *const printfonts[] = {
  "arial.ttf", "arialbd.ttf", "ariali.ttf", "times.ttf", "timesbd.ttf",
  "timesi.ttf", "calibri.ttf", "calibrib.ttf", "cambria.ttc", "verdana.ttf",
  "verdanab.ttf", "tahoma.ttf", "georgia.ttf", "georgiab.ttf",
  "segoeui.ttf", "segoeuib.ttf", "consola.ttf", "cour.ttf", "trebuc.ttf",
  "candara.ttf", "corbel.ttf", "constan.ttf", "pala.ttf", "BOOKOS.TTF",
  "GARA.TTF", "CENTURY.TTF", "framd.ttf", "lucon.ttf", "micross.ttf",
  "seguisb.ttf", NULL
};


/* Windows cursive and handwriting-like fonts */
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
#define SPACE  1  /* class of ' ' */


/* one stored line: 32 rows of bytes, and its text */
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
  const char *iam, *fonts, *books, *output, *test;
  const char *resume;  /* go on training this model (-R) */
  const char *dumpdir;  /* -D */
  int force;  /* -F: a new model may replace an existing file */
  int print, epochs, hidden, threads, synth, lines, testlines;
  int gpucheck;  /* -G: only check the GPU */
  float stopat;  /* -S: stop below this validation error (percent) */
  float rate;
  long seed;
} Options;


typedef struct Trainer {
  mr_State *R;
  mr_Rand rng;
  mr_Seq *q;
  int print;  /* 1 printed text, 0 handwriting */
  uint32_t classes[NCLASSES];
  mr_Font *fonts[MAXFONTS];  /* training fonts */
  int nfonts;
  mr_Font *vfonts[MAXFONTS];  /* print: held out for validation */
  int nvfonts;
  mr_byte *corpus;  /* book text as classes */
  size_t ncorpus;
  Set train, val, test;
  float *m, *v;  /* Adam moments */
  long step;
  const char *dumpdir;  /* -D: save badly read test lines here */
  int gray;  /* the model reads grayscale (else black/white) */
} Trainer;


/* one line to learn: a stored sample, or NULL to draw a new one */
typedef struct Item {
  const Sample *s;
  float stretch;
} Item;


/* shared by the worker threads during one step */
typedef struct Batch {
  Trainer *t;
  mr_SeqWork *wk[MR_MAXTHREADS];
  float *line[MR_MAXTHREADS];
  mr_Rand rng[MR_MAXTHREADS];
  Item items[MR_MAXTHREADS][PERWORKER];
  int nitems[MR_MAXTHREADS];
  float loss[MR_MAXTHREADS];
  int used[MR_MAXTHREADS];
} Batch;


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


/* next code point of UTF-8 text '*ps' (0 at the end) */
static uint32_t nextcp (const unsigned char **ps) {
  const unsigned char *s = *ps;
  uint32_t c = *s;
  if (c == 0) return 0;
  s++;
  if (c >= 0xC0 && c < 0xE0 && (s[0] & 0xC0) == 0x80)
    c = ((c & 0x1F) << 6) | (*s++ & 0x3F);
  else if (c >= 0xE0 && c < 0xF0 && (s[0] & 0xC0) == 0x80 &&
           (s[1] & 0xC0) == 0x80) {
    c = ((c & 0x0F) << 12) | ((s[0] & 0x3F) << 6) | (s[1] & 0x3F);
    s += 2;
  }
  *ps = s;
  return c;
}


/* UTF-8 into classes; returns the count, -1 if a letter is unknown */
static int totarget (const Trainer *t, const char *ss, int *out, int max) {
  const unsigned char *s = mr_cast(const unsigned char *, ss);
  int n = 0;
  uint32_t c;
  while ((c = nextcp(&s)) != 0 && n < max) {
    int k = classof(t, c);
    if (k < 0) return -1;
    out[n++] = k;
  }
  return n;
}


static uint32_t upper (uint32_t c) {
  if (c >= 'a' && c <= 'z') return c - 32;
  if (c >= 0xE0 && c <= 0xFE && c != 0xF7) return c - 0x20;
  return c;
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


/*
** A drawn line that black/white turned into noise (too much ink) or into
** fragments (too little ink for its 'n' letters) teaches nothing.
*/
static int readable (const mr_Bitmap *bm, int n) {
  int x, y, x0 = bm->width, y0 = bm->height, x1 = -1, y1 = -1;
  long ink = 0, area, h;
  for (y = 0; y < bm->height; y++) {
    for (x = 0; x < bm->width; x++) {
      if (!mrK_at(bm, x, y)) continue;
      ink++;
      if (x < x0) x0 = x;
      if (x > x1) x1 = x;
      if (y < y0) y0 = y;
      if (y > y1) y1 = y;
    }
  }
  if (x1 < 0) return 0;
  area = mr_cast(long, x1 - x0 + 1) * (y1 - y0 + 1);
  h = y1 - y0 + 1;
  if (ink * 100 > area * 45) return 0;  /* noise */
  if (n > 0 && ink * 100 < mr_cast(long, n) * h * h * 3) return 0;  /* faded */
  return 1;
}


/*
** Gray image -> ink bitmap -> normalized line (gray or black/white, as
** the model reads); returns its width, 0 if the line is unreadable
** ('n' letters; 0 = do not check).
*/
static int prepare (Trainer *t, const mr_Image *img, int n, float *line,
                    int *status) {
  mr_Bitmap *bm = NULL;
  int w = 0;
  *status = mrK_fromgray(t->R, img, &bm);
  if (*status == MR_OK && (n == 0 || readable(bm, n)))
    w = mrQ_normalize(bm->bits, t->gray ? img->pixels : NULL, bm->width,
                      bm->height, 1.0f, line);
  mrK_free(t->R, bm);
  return w;
}


/* the line is long enough for CTC: one step per letter, plus repeats */
static int fits (int w, const int *target, int n) {
  int k, need = n;
  for (k = 1; k < n; k++)
    if (target[k] == target[k - 1]) need++;
  return w / MR_SEQSTEP >= need + 2;
}

/* }====================================================== */


/*
** {======================================================
** Data: IAM lines, books, fonts
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
    if (status == MR_OK) w = prepare(t, img, 0, line, &status);
    else w = 0;
    mrI_free(t->R, img);
    if (status != MR_OK) {
      fclose(lst);
      return status;
    }
    if (w == 0 || !fits(w, target, n)) {
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


/* append the text of one Gutenberg book, without its license header */
static int addbook (Trainer *t, const char *path, size_t *cap) {
  FILE *f = mrF_open(path, "rb");
  char *buf;
  long len;
  const unsigned char *s, *end;
  const char *a, *b;
  int lastspace = 1;
  if (f == NULL) return MR_OK;  /* missing book: skip */
  if (fseek(f, 0, SEEK_END) != 0 || (len = ftell(f)) <= 0 ||
      fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return MR_OK;
  }
  buf = mr_cast(char *, mrM_malloc(t->R, mr_cast(size_t, len) + 1));
  if (buf == NULL) {
    fclose(f);
    return MR_ERRMEM;
  }
  len = mr_cast(long, fread(buf, 1, mr_cast(size_t, len), f));
  fclose(f);
  buf[len] = '\0';
  a = strstr(buf, "*** START OF");
  a = (a != NULL) ? strchr(a, '\n') : buf;
  b = strstr(buf, "*** END OF");
  s = mr_cast(const unsigned char *, a ? a : buf);
  end = mr_cast(const unsigned char *, b ? b : buf + len);
  while (s < end) {
    uint32_t c = nextcp(&s);
    int k;
    mr_byte *nc;
    if (c == 0) break;
    if (c == '_' || c == '\r') continue;  /* _italics_ marks */
    if (c == ' ' || c == '\n' || c == '\t') {
      if (lastspace) continue;
      k = SPACE;
    }
    else if ((k = classof(t, c)) < 0)
      continue;
    nc = mrM_growvector(t->R, t->corpus, t->ncorpus, cap, mr_byte);
    if (nc == NULL) {
      mrM_free(t->R, buf, mr_cast(size_t, len) + 1);
      return MR_ERRMEM;
    }
    t->corpus = nc;
    t->corpus[t->ncorpus++] = mr_cast(mr_byte, k);
    lastspace = (k == SPACE);
  }
  mrM_free(t->R, buf, mr_cast(size_t, len) + 1);
  return MR_OK;
}


static int loadbooks (Trainer *t, const char *dir, size_t *cap) {
  char path[MR_PATHSIZE], name[256];
  FILE *lst;
  int nbooks = 0, status = MR_OK;
  snprintf(path, sizeof(path), "%s/list.txt", dir);
  lst = mrF_open(path, "rb");
  if (lst == NULL) {
    printf("no books in %s (run ./getdata.sh books)\n", dir);
    return MR_OK;
  }
  while (status == MR_OK && fgets(name, sizeof(name), lst) != NULL) {
    name[strcspn(name, "\r\n")] = '\0';
    if (name[0] == '\0') continue;
    snprintf(path, sizeof(path), "%s/%s", dir, name);
    status = addbook(t, path, cap);
    nbooks++;
  }
  fclose(lst);
  printf("books: %d, %lu letters of text\n", nbooks,
         mr_cast(unsigned long, t->ncorpus));
  return status;
}


/*
** Add one font: to the training list, or every HOLDOUT-th to validation.
** Decorative ('display' = 0) fonts are never held out: no OCR reads them
** reliably, so they would not measure normal text. They are counted all
** the same, so the held-out text fonts stay the same.
*/
static void addfont (Trainer *t, const char *path, int *count, int text) {
  mr_Font *f;
  if (mrT_load(t->R, path, 0, &f) != MR_OK) return;
  if (!mrT_hasglyph(f, 'a') || !mrT_hasglyph(f, 'A') ||
      !mrT_hasglyph(f, '0') || !mrT_hasglyph(f, '.')) {
    mrT_free(t->R, f);
    return;
  }
  (*count)++;
  if (t->print && text && *count % HOLDOUT == 0 && t->nvfonts < MAXFONTS)
    t->vfonts[t->nvfonts++] = f;
  else if (t->nfonts < MAXFONTS)
    t->fonts[t->nfonts++] = f;
  else
    mrT_free(t->R, f);
}


/*
** Windows fonts, then the Google Fonts in 'dir' (see getdata.sh fonts):
** print and display kinds for printed text, handwriting for handwriting.
*/
static void loadfonts (Trainer *t, const char *dir) {
  char path[MR_PATHSIZE], line[512];
  const char *const *win = t->print ? printfonts : scriptfonts;
  FILE *lst;
  int i, count = 0;
  for (i = 0; win[i] != NULL; i++) {
    snprintf(path, sizeof(path), "C:/Windows/Fonts/%s", win[i]);
    addfont(t, path, &count, 1);
  }
  snprintf(path, sizeof(path), "%s/list.txt", dir);
  lst = mrF_open(path, "rb");
  while (lst != NULL && fgets(line, sizeof(line), lst) != NULL) {
    char *cat = line, *fam, *styles, *st, family[256];
    size_t k, n = 0;
    line[strcspn(line, "\r\n")] = '\0';
    fam = strchr(cat, '\t');
    if (fam == NULL) continue;
    *fam++ = '\0';
    styles = strchr(fam, '\t');
    if (styles == NULL) continue;
    *styles++ = '\0';
    if ((strcmp(cat, "Handwriting") == 0) == (t->print != 0)) continue;
    for (k = 0; fam[k] != '\0' && n + 1 < sizeof(family); k++)
      if (fam[k] != ' ') family[n++] = fam[k];
    family[n] = '\0';
    for (st = strtok(styles, ","); st != NULL; st = strtok(NULL, ",")) {
      snprintf(path, sizeof(path), "%s/%s/%s-%s.ttf", dir, cat, family, st);
      addfont(t, path, &count, strcmp(cat, "Display") != 0);
    }
  }
  if (lst != NULL) fclose(lst);
  printf("fonts: %d for training, %d held out for validation\n", t->nfonts,
         t->nvfonts);
}

/* }====================================================== */


/*
** {======================================================
** Synthetic lines
** =======================================================
*/

/* a random piece of book text, from word start to word end */
static int booktext (const Trainer *t, mr_Rand *rng, int *target) {
  int len = 12 + mrR_int(rng, 50), n = 0;
  size_t at;
  if (t->ncorpus < 1000) return 0;
  at = mr_cast(size_t, mrR_int(rng, mr_cast(int, t->ncorpus - 200)));
  while (at < t->ncorpus && t->corpus[at] != SPACE) at++;
  at++;
  while (at < t->ncorpus && n < MAXTEXT - 40) {
    int k = t->corpus[at++];
    if (k == SPACE && n >= len) break;
    target[n++] = k;
  }
  return n;
}


/* a short piece of printed-document text: numbers, money, symbols... */
static int token (Trainer *t, mr_Rand *rng, int *out) {
  static const char *const money[] = { "$", "\xE2\x82\xB1", "\xE2\x82\xAC",
                                       "\xC2\xA3", "\xC2\xA5" };
  static const char *const names[] = { "juan", "maria", "info", "sales",
                                       "j.dela.cruz", "admin" };
  static const char *const hosts[] = { "example.com", "mail.ph", "acme.org",
                                       "gmail.com" };
  static const char *const syms[] = {
    "\xE2\x80\xA2", "\xC2\xA9", "\xC2\xAE", "\xE2\x84\xA2", "\xC2\xA7",
    "\xC2\xB6", "\xC2\xBD", "\xC2\xBC", "\xC2\xBE", "\xC3\x97", "\xC3\xB7",
    "\xC2\xB1", "\xE2\x80\x93", "\xE2\x80\x94", "\xE2\x80\xA6", "#", "&",
    "*", "+", "=", "<", ">", "|", "\\", "/", "~", "^", "_", "`", "@", "%",
    "[", "]", "{", "}", "(", ")"
  };
  char buf[96];
  int r = mrR_int(rng, 12);
  int a = mrR_int(rng, 10000), b = mrR_int(rng, 100);
  switch (r) {
    case 0:
      snprintf(buf, sizeof(buf), "%s%d,%03d.%02d", money[mrR_int(rng, 5)],
               1 + a % 99, mrR_int(rng, 1000), b);
      break;
    case 1:
      snprintf(buf, sizeof(buf), "%s%d.%02d", money[mrR_int(rng, 5)], a, b);
      break;
    case 2:
      snprintf(buf, sizeof(buf), "%02d/%02d/%04d", 1 + b % 28, 1 + a % 12,
               1950 + a % 80);
      break;
    case 3:
      snprintf(buf, sizeof(buf), "%04d-%02d-%02d", 1950 + a % 80,
               1 + a % 12, 1 + b % 28);
      break;
    case 4:
      snprintf(buf, sizeof(buf), "%d:%02d", 1 + a % 12, b % 60);
      break;
    case 5:
      snprintf(buf, sizeof(buf), "%d.%d%%", a % 100, b % 10);
      break;
    case 6:
      snprintf(buf, sizeof(buf), "+63 9%02d %03d %04d", b, a % 1000, a);
      break;
    case 7:
      snprintf(buf, sizeof(buf), "%s%d@%s", names[mrR_int(rng, 6)], b,
               hosts[mrR_int(rng, 4)]);
      break;
    case 8:
      snprintf(buf, sizeof(buf), "%d\xC2\xB0%s", a % 45, b % 2 ? "C" : "F");
      break;
    case 9:
      snprintf(buf, sizeof(buf), "%d %s %d = %d", a % 20,
               (b % 2) ? "\xC3\x97" : "\xC3\xB7", 1 + b % 12, a % 97);
      break;
    case 10:
      snprintf(buf, sizeof(buf), "\xE2\x80\x9C%s\xE2\x80\x9D",
               names[mrR_int(rng, 6)]);
      break;
    default:
      snprintf(buf, sizeof(buf), "%s", syms[mrR_int(rng, 37)]);
      break;
  }
  return totarget(t, buf, out, 64);
}


/* text for a synthetic line, using what 'font' can draw */
static int gentext (Trainer *t, mr_Rand *rng, const mr_Font *font,
                    int *target) {
  int n = booktext(t, rng, target), i, k;
  int tok[64];
  if (n == 0 && t->train.n > 0) {  /* no books: IAM sentences */
    const Sample *s = &t->train.s[mrR_int(rng, t->train.n)];
    n = s->n;
    memcpy(target, s->target, n * sizeof(int));
  }
  if (t->print && mrR_int(rng, 100) < 40) {  /* insert document bits */
    int times = 1 + mrR_int(rng, 3);
    for (k = 0; k < times; k++) {
      int m = token(t, rng, tok), at = mrR_int(rng, n + 1);
      if (m <= 0 || n + m + 2 >= MAXTEXT) continue;
      while (at < n && target[at] != SPACE) at++;  /* at a word gap */
      memmove(&target[at + m + 1], &target[at],
              (n - at) * sizeof(int));
      target[at] = SPACE;
      memcpy(&target[at + 1], tok, m * sizeof(int));
      n += m + 1;
    }
  }
  if (t->print && mrR_int(rng, 100) < 5)  /* ALL CAPS line */
    for (i = 0; i < n; i++)
      target[i] = classof(t, upper(t->classes[target[i]]));
  for (i = 0; i < n; i++) {
    uint32_t c = t->classes[target[i]];
    if (c == 'n' && mrR_int(rng, 25) == 0) c = 0xF1;
    else if (c != ' ' && mrR_int(rng, 120) == 0)
      c = extras[mrR_int(rng, NEXTRAS)];
    if (c != ' ' && !mrT_hasglyph(font, c)) c = 'e';  /* font lacks it */
    target[i] = classof(t, c);
  }
  while (n > 0 && target[n - 1] == SPACE) n--;
  while (n > 0 && target[0] == SPACE) {
    memmove(target, target + 1, (n - 1) * sizeof(int));
    n--;
  }
  return n;
}


static void addnoise (mr_Rand *rng, mr_Image *img, float sd) {
  size_t i, n = mr_cast(size_t, img->width) * img->height;
  for (i = 0; i < n; i++) {
    int v = img->pixels[i] + mr_cast(int, mrR_normal(rng) * sd);
    img->pixels[i] = mr_cast(mr_byte, v < 0 ? 0 : v > 255 ? 255 : v);
  }
}


static void blur (mr_Image *img, mr_byte *tmp) {
  int w = img->width, h = img->height, x, y;
  memcpy(tmp, img->pixels, mr_cast(size_t, w) * h);
  for (y = 0; y < h; y++) {
    for (x = 0; x < w; x++) {
      int s = 0, c = 0, dx, dy;
      for (dy = -1; dy <= 1; dy++) {
        for (dx = -1; dx <= 1; dx++) {
          int nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          s += tmp[mr_cast(size_t, ny) * w + nx];
          c++;
        }
      }
      img->pixels[mr_cast(size_t, y) * w + x] = mr_cast(mr_byte, s / c);
    }
  }
}


/*
** Thicker ink (grow = 1) or thinner ink (grow = 0): the darkest or the
** lightest of each 3x3 window, like over- or under-inked scans.
*/
static void morph (mr_Image *img, mr_byte *tmp, int darkink, int grow) {
  int w = img->width, h = img->height, x, y, dx, dy;
  int takemin = (darkink == grow);
  memcpy(tmp, img->pixels, mr_cast(size_t, w) * h);
  for (y = 0; y < h; y++) {
    for (x = 0; x < w; x++) {
      int v = tmp[mr_cast(size_t, y) * w + x];
      for (dy = -1; dy <= 1; dy++) {
        for (dx = -1; dx <= 1; dx++) {
          int nx = x + dx, ny = y + dy, u;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          u = tmp[mr_cast(size_t, ny) * w + nx];
          if (takemin ? (u < v) : (u > v)) v = u;
        }
      }
      img->pixels[mr_cast(size_t, y) * w + x] = mr_cast(mr_byte, v);
    }
  }
}


/* a slightly skewed line, like a page put crooked on the scanner */
static void skew (mr_Image *img, mr_byte *tmp, float slope, int paper) {
  int w = img->width, h = img->height, x, y;
  memcpy(tmp, img->pixels, mr_cast(size_t, w) * h);
  for (x = 0; x < w; x++) {
    int shift = mr_cast(int, floorf((x - w / 2) * slope + 0.5f));
    for (y = 0; y < h; y++) {
      int sy = y - shift;
      img->pixels[mr_cast(size_t, y) * w + x] =
        (sy >= 0 && sy < h) ? tmp[mr_cast(size_t, sy) * w + x]
                            : mr_cast(mr_byte, paper);
    }
  }
}


/*
** Draw a new line with one of 'fonts' into 'line'; returns its width,
** 0 if it failed. Safe to call from several threads.
*/
static int synthline (Trainer *t, mr_Rand *rng, mr_Font *const *fonts,
                      int nfonts, float *line, int *target, int *n) {
  uint32_t cps[MAXTEXT];
  mr_Box boxes[MAXTEXT];
  const mr_Font *f;
  mr_Draw d;
  mr_Image *img = NULL;
  int i, w, status;
  float u = mrR_float(rng);
  if (nfonts == 0) return 0;
  f = fonts[mrR_int(rng, nfonts)];
  *n = gentext(t, rng, f, target);
  if (*n <= 0) return 0;
  for (i = 0; i < *n; i++) cps[i] = t->classes[target[i]];
  d.height = t->print ? 10 + 50 * u * sqrtf(u) : mrR_range(rng, 28, 60);
  d.stretch = mrR_range(rng, 0.85f, 1.15f);
  d.spacing = (t->print && mrR_int(rng, 10) == 0)
              ? mrR_range(rng, -0.5f, 0.8f) : 0;
  d.ink = mrR_int(rng, 90);
  d.paper = 170 + mrR_int(rng, 86);
  if (t->print && mrR_int(rng, 100) < 15) {  /* light text on dark */
    int tmp = d.paper;
    d.paper = d.ink;
    d.ink = tmp;
  }
  d.pad = 4;
  status = mrT_drawline(t->R, f, cps, *n, &d, &img, boxes, NULL);
  if (status != MR_OK) return 0;
  if (mrR_int(rng, 100) < 20 && d.height > 16) {
    size_t sz = mr_cast(size_t, img->width) * img->height;
    mr_byte *tmp = mr_cast(mr_byte *, mrM_malloc(t->R, sz));
    if (tmp != NULL) {
      blur(img, tmp);
      mrM_free(t->R, tmp, sz);
    }
  }
  if (t->print) {  /* scan effects */
    size_t sz = mr_cast(size_t, img->width) * img->height;
    int r = mrR_int(rng, 100);
    mr_byte *tmp = mr_cast(mr_byte *, mrM_malloc(t->R, sz));
    if (tmp != NULL) {
      if (r < 12 && d.height > 14) morph(img, tmp, d.ink < d.paper, 1);
      else if (r < 22 && d.height > 30) morph(img, tmp, d.ink < d.paper, 0);
      if (mrR_int(rng, 100) < 30)
        skew(img, tmp, mrR_range(rng, -0.015f, 0.015f), d.paper);
      mrM_free(t->R, tmp, sz);
    }
  }
  if (mrR_int(rng, 100) < 30 && abs(d.ink - d.paper) >= 100)
    addnoise(rng, img, mrR_range(rng, 3, 12));
  w = prepare(t, img, *n, line, &status);
  mrI_free(t->R, img);
  if (status != MR_OK || w == 0 || !fits(w, target, *n)) return 0;
  return w;
}


/* a fixed set of printed lines from the held-out fonts */
static int printset (Trainer *t, Set *set, int count, uint64_t seed,
                     float *line) {
  mr_Rand rng;
  int target[MAXTEXT], i, tries = 0;
  mrR_seed(&rng, seed);
  for (i = 0; i < count && tries < count * 4; tries++) {
    int n, w = synthline(t, &rng, t->vfonts, t->nvfonts, line, target, &n);
    if (w > 0) {
      int status = addsample(t->R, set, line, w, target, n);
      if (status != MR_OK) return status;
      i++;
    }
  }
  printf("held-out font lines: %d\n", set->n);
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
  Trainer *t = b->t;
  int target[MAXTEXT], k;
  b->loss[i] = 0;
  b->used[i] = 0;
  for (k = 0; k < b->nitems[i]; k++) {
    const Item *it = &b->items[i][k];
    const int *tg;
    int w, n;
    float l;
    if (it->s != NULL) {
      w = unpack(it->s, it->stretch, b->line[i]);
      tg = it->s->target;
      n = it->s->n;
    }
    else {
      w = synthline(t, &b->rng[i], t->fonts, t->nfonts, b->line[i], target,
                    &n);
      tg = target;
    }
    if (w == 0) continue;
    l = mrQ_learn(t->q, b->wk[i], b->line[i], w, tg, n);
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


/* save a stored line as a .pgm image (black text on white) */
static void savepgm (const Sample *s, const char *path) {
  FILE *f = mrF_open(path, "wb");
  int k;
  if (f == NULL) return;
  fprintf(f, "P5\n%d %d\n255\n", s->w, MR_SEQH);
  for (k = 0; k < s->w * MR_SEQH; k++) fputc(255 - s->pix[k], f);
  fclose(f);
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
    if (t->dumpdir != NULL && d * 4 > s->n) {  /* badly read: keep it */
      char path[MR_PATHSIZE];
      snprintf(path, sizeof(path), "%s/bad%04d.pgm", t->dumpdir, i);
      savepgm(s, path);
      printf("  bad%04d: %d of %d letters wrong\n", i, d, s->n);
    }
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
  float best = 1e9f;
  int *order = NULL, e, i, nt = opt->threads, status = MR_OK, norder = 0;
  size_t np = t->q->np, j;
  clock_t start = clock();
  memset(&b, 0, sizeof(b));
  b.t = t;
  t->m = mrM_newarray(R, np, float);
  t->v = mrM_newarray(R, np, float);
  if (t->m == NULL || t->v == NULL) goto nomem;
  memset(t->m, 0, np * sizeof(float));
  memset(t->v, 0, np * sizeof(float));
  for (i = 0; i < nt; i++) {
    status = mrQ_newwork(R, t->q, &b.wk[i]);
    if (status != MR_OK) goto done;
    b.line[i] = mrM_newarray(R, mr_cast(size_t, MR_SEQH) * MR_SEQMAXW,
                             float);
    if (b.line[i] == NULL) goto nomem;
    mrR_seed(&b.rng[i], mr_cast(uint64_t, opt->seed) * 7919u + i);
  }
  /* each epoch: the stored lines (handwriting) and new drawn ones (-1) */
  norder = t->print ? opt->lines
                    : t->train.n + t->train.n * opt->synth /
                                   (100 - opt->synth);
  order = mrM_newarray(R, norder + 1, int);
  if (order == NULL) goto nomem;
  printf("model: %d classes, LSTM %d, %lu weights, %d threads, "
         "%d lines per epoch\n", t->q->nclasses, t->q->hidden,
         mr_cast(unsigned long, np), nt, norder);
  if (opt->resume != NULL) {  /* only save what beats the start */
    best = evaluate(t, b.wk[0], b.line[0], &t->val, VALLINES, 0);
    printf("resumed %s: validation %.2f%% errors\n", opt->resume, best);
    fflush(stdout);
  }
  for (e = 1; e <= opt->epochs; e++) {
    float rate = opt->rate, loss = 0, cer;
    long used = 0;
    int pos;
    if (e > opt->epochs * 2 / 3) rate *= 0.3f;  /* slow down at the end */
    for (i = 0; i < norder; i++) order[i] = (i < t->train.n) ? i : -1;
    for (i = norder - 1; i > 0; i--) {
      int k = mrR_int(&t->rng, i + 1), tmp = order[i];
      order[i] = order[k];
      order[k] = tmp;
    }
    for (pos = 0; pos < norder; ) {
      int k, w0 = 0;
      float norm = 0, scale;
      for (i = 0; i < nt; i++) {
        b.nitems[i] = 0;
        for (k = 0; k < PERWORKER && pos < norder; k++, pos++) {
          Item *it = &b.items[i][k];
          it->s = (order[pos] >= 0) ? &t->train.s[order[pos]] : NULL;
          it->stretch = mrR_range(&t->rng, 0.8f, 1.2f);
          b.nitems[i]++;
        }
      }
      mrX_parallel(nt, work, &b);
      for (i = 0; i < nt; i++) {  /* sum gradients into worker 0 */
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
      fflush(stdout);
    }
    if (cer < opt->stopat) {
      printf("validation below %.2f%%: done\n", opt->stopat);
      break;
    }
  }
  goto done;
 nomem:
  status = MR_ERRMEM;
 done:
  for (i = 0; i < nt; i++) {
    mrQ_freework(R, b.wk[i]);
    if (b.line[i] != NULL)
      mrM_freearray(R, b.line[i], mr_cast(size_t, MR_SEQH) * MR_SEQMAXW);
  }
  if (order != NULL) mrM_freearray(R, order, norder + 1);
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
    "Train a line model: handwriting (IAM + handwriting fonts), or\n"
    "printed text (-P, many fonts). Both use book text when present.\n"
    "Available options are:\n"
    "  -P        printed text instead of handwriting\n"
    "  -I dir    IAM folder (default build/data/iam)\n"
    "  -F dir    Google Fonts folder (default build/data/fonts)\n"
    "  -B dir    books folder (default build/data/books)\n"
    "  -o file   output model (default cursive.mrm, or print.mrm with -P)\n"
    "  -e count  epochs (default %d)\n"
    "  -L count  printed lines per epoch with -P (default %d)\n"
    "  -k cells  LSTM cells per direction (default %d)\n"
    "  -r rate   learning rate (default 0.0005)\n"
    "  -j count  threads (default: CPU cores - 1)\n"
    "  -y pct    handwriting: percent of font lines (default %d)\n"
    "  -s seed   random seed (default 1)\n"
    "  -D dir    with -t: save badly read lines as .pgm in 'dir'\n"
    "  -S pct    stop when validation errors are below 'pct'\n"
    "  -G        check the GPU (in a ./build.sh gpu build) and stop\n"
    "  -F        let a new model replace an existing output file\n"
    "  -R model  go on training 'model' (keeps its size)\n"
    "  -t model  test 'model' instead (IAM test, or held-out fonts)\n"
    "  -c lines  test lines (default all IAM, or 1000)\n"
    "  -h        print this help\n",
    progname, DEF_EPOCHS, DEF_LINES, DEF_HIDDEN, DEF_SYNTH);
}


static int collectargs (int argc, char **argv, Options *opt) {
  int i, cpus = mrX_cpus();
  opt->iam = "build/data/iam";
  opt->fonts = "build/data/fonts";
  opt->books = "build/data/books";
  opt->output = NULL;
  opt->test = NULL;
  opt->resume = NULL;
  opt->dumpdir = NULL;
  opt->force = 0;
  opt->print = 0;
  opt->gpucheck = 0;
  opt->stopat = 0;
  opt->epochs = DEF_EPOCHS;
  opt->hidden = DEF_HIDDEN;
  opt->threads = (cpus > 1) ? cpus - 1 : 1;
  opt->synth = DEF_SYNTH;
  opt->lines = DEF_LINES;
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
    if (a[1] == 'P') {
      opt->print = 1;
      continue;
    }
    if (a[1] == 'G') {
      opt->gpucheck = 1;
      continue;
    }
    if (a[1] == 'F') {
      opt->force = 1;
      continue;
    }
    if (i + 1 >= argc) goto bad;
    val = argv[++i];
    switch (a[1]) {
      case 'I': opt->iam = val; break;
      case 'F': opt->fonts = val; break;
      case 'B': opt->books = val; break;
      case 'o': opt->output = val; break;
      case 't': opt->test = val; break;
      case 'R': opt->resume = val; break;
      case 'D': opt->dumpdir = val; break;
      case 'S': opt->stopat = mr_cast(float, atof(val)); break;
      case 'e': opt->epochs = atoi(val); break;
      case 'L': opt->lines = atoi(val); break;
      case 'k': opt->hidden = atoi(val); break;
      case 'j': opt->threads = atoi(val); break;
      case 'y': opt->synth = atoi(val); break;
      case 'c': opt->testlines = atoi(val); break;
      case 'r': opt->rate = mr_cast(float, atof(val)); break;
      case 's': opt->seed = atol(val); break;
      default: goto bad;
    }
  }
  if (opt->output == NULL) opt->output = opt->print ? "print.mrm"
                                                    : "cursive.mrm";
  if (opt->threads < 1) opt->threads = 1;
  if (opt->threads > MR_MAXTHREADS) opt->threads = MR_MAXTHREADS;
  if (opt->synth < 0) opt->synth = 0;
  if (opt->synth > 90) opt->synth = 90;
  if (opt->lines < 100) opt->lines = 100;
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
  size_t capcorpus = 0;
  int i, status = collectargs(argc, argv, &opt);
  if (status != 0) return (status < 0) ? EXIT_SUCCESS : status;
  memset(&t, 0, sizeof(t));
  t.R = mr_open();
  if (t.R == NULL) {
    l_message("not enough memory");
    return EXIT_FAILURE;
  }
  t.print = opt.print;
  t.dumpdir = opt.dumpdir;
  if (opt.gpucheck) {
    mr_Gpu *g = NULL;
    status = mrU_open(t.R, &g);
    if (status == MR_OK) {
      printf("GPU: %s, %d compute units, %.1f GB\n", mrU_name(g),
             mrU_units(g), mrU_memory(g));
      status = mrU_selftest(t.R, g);
      if (status == MR_OK) printf("GPU self-test: ok\n");
      printf("note: training still runs on the CPU; the GPU network "
             "kernels are not written yet\n");
    }
    mrU_close(t.R, g);
    goto done;
  }
  mrR_seed(&t.rng, mr_cast(uint64_t, opt.seed));
  t.classes[0] = 0;  /* blank */
  t.classes[SPACE] = ' ';
  for (i = 0; i < 94; i++) t.classes[2 + i] = mr_cast(uint32_t, '!' + i);
  for (i = 0; i < NEXTRAS; i++) t.classes[96 + i] = extras[i];
  line = mrM_newarray(t.R, mr_cast(size_t, MR_SEQH) * MR_SEQMAXW, float);
  if (line == NULL) {
    status = MR_ERRMEM;
    goto done;
  }
  status = loadbooks(&t, opt.books, &capcorpus);
  if (status != MR_OK) goto done;
  loadfonts(&t, opt.fonts);
  if (opt.test != NULL) {  /* test an existing model */
    status = mrQ_load(t.R, opt.test, &t.q);
    if (status == MR_OK) {
      t.gray = t.q->gray;
      status = mrQ_newwork(t.R, t.q, &wk);
    }
    if (status == MR_OK && t.print)
      status = printset(&t, &t.test, opt.testlines > 0 ? opt.testlines
                                                        : 1000, 4242, line);
    else if (status == MR_OK)
      status = loadiam(&t, opt.iam, "test", &t.test, opt.testlines, line);
    if (status == MR_OK)
      printf("test: %d lines, %.2f%% character errors\n", t.test.n,
             evaluate(&t, wk, line, &t.test, t.test.n, 8));
    goto done;
  }
  if (opt.resume == NULL && !opt.force) {
    FILE *old = mrF_open(opt.output, "rb");
    if (old != NULL) {
      fclose(old);
      status = mrS_error(t.R, MR_ERRARG,
                         "'%s' already exists: continue it with -R %s, "
                         "use another -o name, or replace it with -F",
                         opt.output, opt.output);
      goto done;
    }
  }
  if (opt.resume != NULL) {
    status = mrQ_load(t.R, opt.resume, &t.q);
    if (status == MR_OK && (t.q->nclasses != NCLASSES ||
        memcmp(t.q->classes, t.classes, sizeof(t.classes)) != 0))
      status = mrS_error(t.R, MR_ERRARG, "'%s' has other letters",
                         opt.resume);
    if (status != MR_OK) goto done;
  }
  else {
    status = mrQ_new(t.R, NCLASSES, opt.hidden, &t.q);
    if (status != MR_OK) goto done;
    memcpy(t.q->classes, t.classes, sizeof(t.classes));
    mrQ_randomize(t.q, &t.rng);
  }
  t.gray = t.q->gray;
  printf("input: %s\n", t.gray ? "grayscale" : "black/white");
  if (t.print) {
    if (t.nvfonts == 0) {
      status = mrS_error(t.R, MR_ERRARG, "not enough fonts");
      goto done;
    }
    status = printset(&t, &t.val, VALLINES, 777, line);
  }
  else {
    status = loadiam(&t, opt.iam, "train", &t.train, 0, line);
    if (status == MR_OK)
      status = loadiam(&t, opt.iam, "validation", &t.val, 0, line);
  }
  if (status != MR_OK) goto done;
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
  if (t.corpus != NULL) mrM_freearray(t.R, t.corpus, capcorpus);
  for (i = 0; i < t.nfonts; i++) mrT_free(t.R, t.fonts[i]);
  for (i = 0; i < t.nvfonts; i++) mrT_free(t.R, t.vfonts[i]);
  mr_close(t.R);
  return (status == MR_OK) ? EXIT_SUCCESS : EXIT_FAILURE;
}
