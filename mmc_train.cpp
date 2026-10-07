/*
** mmc_train.cpp
** Train an OCR model from fonts (like luac.c, it uses internal headers)
** See Copyright Notice in mr.h
*/

#define mmc_train_cpp

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
#include "mrglyph.h"
#include "mrhand.h"
#include "mrlayout.h"
#include "mrmem.h"
#include "mrnet.h"
#include "mrocr.h"
#include "mrrand.h"
#include "mrstate.h"


#define MAXFONTS  64
#define MAXLINE  64  /* code points per training line */
#define POOL  4096  /* samples per training round */
#define BATCH  32
#define NEXTRAS  41  /* letters beyond ASCII, see 'extras' */
#define NBASE  (94 + 2)  /* '!' .. '~' plus 'Ñ' 'ñ': every font needs them */
#define NLETTERS  (94 + NEXTRAS)
#define REJECT  NLETTERS  /* class of "not one letter" */
#define NCLASSES  (NLETTERS + 1)
#define TESTLINES  300

#define DEF_SAMPLES  1000000L
#define DEF_RATE  0.001f
#define DEF_OUTPUT  "eng" MR_MODELEXT
#define DEF_HANDPCT  50  /* percent of handwritten lines with -H */
#define HANDPER  4000  /* EMNIST samples kept per letter */


static const char *progname = "mmc_train";


/* fonts tried when none are given (Windows) */
static const char *const deffonts[] = {
  "arial.ttf", "arialbd.ttf", "ariali.ttf", "times.ttf", "timesbd.ttf",
  "timesi.ttf", "calibri.ttf", "calibrib.ttf", "cambria.ttc", "verdana.ttf",
  "verdanab.ttf", "tahoma.ttf", "georgia.ttf", "georgiab.ttf",
  "segoeui.ttf", "segoeuib.ttf", "consola.ttf", "cour.ttf", "trebuc.ttf",
  "candara.ttf", "corbel.ttf", "constan.ttf", "pala.ttf", "BOOKOS.TTF",
  "GARA.TTF", "CENTURY.TTF", "framd.ttf", "lucon.ttf", "micross.ttf",
  "seguisb.ttf", NULL
};


/* handwriting-like fonts, for the symbols EMNIST does not have */
static const char *const handfonts[] = {
  "segoepr.ttf", "segoeprb.ttf", "Inkfree.ttf", "comic.ttf", "comicbd.ttf",
  "BRADHITC.TTF", "mvboli.ttf", "PRISTINA.TTF", NULL
};


/* letters beyond ASCII; the first two are needed in every font */
static const uint32_t extras[NEXTRAS] = {
  0xD1, 0xF1,  /* Ñ ñ */
  0x201C, 0x201D, 0x2018, 0x2019, 0x2013, 0x2014,  /* “ ” ‘ ’ – — */
  0x2022, 0x2026,  /* • … */
  0x20B1, 0x20AC, 0xA3, 0xA5, 0xA2,  /* ₱ € £ ¥ ¢ */
  0xB0, 0xA9, 0xAE, 0x2122, 0xD7, 0xF7,  /* ° © ® ™ × ÷ */
  0xB1, 0xA7, 0xB6, 0xBD, 0xBC, 0xBE,  /* ± § ¶ ½ ¼ ¾ */
  0xE1, 0xE9, 0xED, 0xF3, 0xFA, 0xFC,  /* á é í ó ú ü */
  0xC1, 0xC9, 0xCD, 0xD3, 0xDA, 0xDC,  /* Á É Í Ó Ú Ü */
  0xBF, 0xA1  /* ¿ ¡ */
};


/* common words, to get real letter shapes next to each other */
static const char *const words[] = {
  "the", "of", "and", "to", "in", "is", "you", "that", "it", "he", "was",
  "for", "on", "are", "as", "with", "his", "they", "at", "be", "this",
  "have", "from", "or", "one", "had", "by", "word", "but", "not", "what",
  "all", "were", "we", "when", "your", "can", "said", "there", "use",
  "each", "which", "she", "do", "how", "their", "if", "will", "up",
  "other", "about", "out", "many", "then", "them", "these", "so", "some",
  "her", "would", "make", "like", "him", "into", "time", "has", "look",
  "two", "more", "write", "go", "see", "number", "no", "way", "could",
  "people", "my", "than", "first", "water", "been", "call", "who", "oil",
  "its", "now", "find", "long", "down", "day", "did", "get", "come",
  "made", "may", "part", "quick", "brown", "fox", "jumps", "over", "lazy",
  "dog", "zone", "jazz", "quiz", "exam", "box", "vivid", "kept", "yield",
  "invoice", "total", "amount", "date", "name", "address", "phone",
  "email", "page", "report", "office", "street", "city", "receipt",
  "señor", "señora", "niño", "niña", "año", "mañana", "piña", "España",
  "Parañaque", "Peñafrancia", "Santo", "Niño", "Dasmariñas", "Muñoz",
  "Ñora", "baño", "cañon", "doña", "dueño", "pequeño", "it's", "don't",
  "can't", "you're", "we'll", "José", "María", "Ramón", "canción",
  "jalapeño", "pingüino", "Bogotá", "Perú", "más", "sí", "corazón",
  "Ángel", "árbol", "último", "Úrsula", "Óscar", "Ícaro", "Él", "café",
  "menú", "teléfono", "público", "rápido", "acción", "también", "Güiro",
  NULL
};


/* English letter frequencies (per 1000) */
static const int letterfreq[26] = {
  82, 15, 28, 43, 127, 22, 20, 61, 70, 2, 8, 40, 24,
  67, 75, 19, 1, 60, 63, 91, 28, 10, 24, 2, 20, 1
};


typedef struct Gen {
  mr_State *R;
  mr_Rand rng;
  mr_Font *fonts[MAXFONTS];
  int nfonts;
  int font;  /* font of the line being made */
  mr_byte has[MAXFONTS][NCLASSES];  /* which letters each font has */
  mr_byte ishand[MAXFONTS];  /* 1 for handwriting-like fonts */
  int nhand;  /* how many of those */
  mr_Hand *hand;  /* EMNIST training writers (NULL without -H) */
  mr_Hand *handtest;  /* EMNIST test writers */
  int handpct;  /* percent of handwritten training lines */
  uint32_t classes[NCLASSES];
  long counts[NCLASSES];  /* samples made per class */
  float *pool;  /* POOL inputs */
  int labels[POOL];
  int npool;
  long lines, badlines, chars, used;
} Gen;


typedef struct Options {
  const char *output;
  const char *test;  /* model to test, NULL to train */
  const char *handdir;  /* EMNIST folder (-H), NULL for print only */
  const char *dumpdir;  /* save example lines here (-d), then stop */
  int handpct;
  long samples;
  long seed;
  float rate;
  int hidden[MR_MAXLAYERS];
  int nhidden;
  int testlines;
  int first;  /* first font argument */
} Options;


/*
** {======================================================
** Messages
** =======================================================
*/

static void print_usage (const char *badoption) {
  if (badoption != NULL)
    fprintf(stderr, "%s: bad option '%s'\n", progname, badoption);
  fprintf(stderr,
    "usage: %s [options] [font[#index]...]\n"
    "Train an OCR model by drawing text with TrueType fonts.\n"
    "With no fonts, common Windows fonts are used.\n"
    "Available options are:\n"
    "  -o file   output model (default " DEF_OUTPUT ")\n"
    "  -n count  training samples (default %ld)\n"
    "  -l sizes  hidden layer sizes (default 256,128)\n"
    "  -r rate   learning rate (default 0.001)\n"
    "  -s seed   random seed (default 1)\n"
    "  -t model  test 'model' instead of training\n"
    "  -c lines  lines to test (default %d)\n"
    "  -H dir    also learn handwriting from EMNIST in 'dir'\n"
    "            (build/data/emnist, see getdata.sh)\n"
    "  -m pct    percent of handwritten lines with -H (default %d)\n"
    "  -d dir    save 20 example training lines as .pgm images in 'dir'\n"
    "  -h        print this help\n",
    progname, DEF_SAMPLES, TESTLINES, DEF_HANDPCT);
}


static void l_message (const char *msg) {
  fprintf(stderr, "%s: %s\n", progname, msg);
  fflush(stderr);
}


static int report (mr_State *R, int status) {
  if (status != MR_OK) l_message(mr_geterror(R));
  return status;
}

/* }====================================================== */


/*
** {======================================================
** Text generation
** =======================================================
*/

static int classof (const Gen *g, uint32_t cp) {
  int i;
  for (i = 0; i < NCLASSES; i++)
    if (g->classes[i] == cp) return i;
  return -1;
}


/* 1 if the current font can draw 'cp' */
static int candraw (const Gen *g, uint32_t cp) {
  int c;
  if (cp == ' ') return 1;
  c = classof(g, cp);
  return c >= 0 && g->has[g->font][c];
}


/* decode UTF-8 'ss' into 'out'; returns the count */
static int decode (const char *ss, uint32_t *out, int max) {
  const unsigned char *s = mr_cast(const unsigned char *, ss);
  int n = 0;
  while (*s && n < max) {
    uint32_t c = *s++;
    if (c >= 0xC0 && c < 0xE0 && (s[0] & 0xC0) == 0x80)
      c = ((c & 0x1F) << 6) | (*s++ & 0x3F);
    else if (c >= 0xE0 && c < 0xF0 && (s[0] & 0xC0) == 0x80 &&
             (s[1] & 0xC0) == 0x80) {
      c = ((c & 0x0F) << 12) | ((s[0] & 0x3F) << 6) | (s[1] & 0x3F);
      s += 2;
    }
    out[n++] = c;
  }
  return n;
}


static uint32_t upper (uint32_t c) {
  if (c >= 'a' && c <= 'z') return c - 32;
  if (c >= 0xE0 && c <= 0xFE && c != 0xF7) return c - 0x20;  /* á -> Á */
  return c;
}


static uint32_t randletter (Gen *g) {
  int r = mrR_int(&g->rng, 1000), i;
  for (i = 0; i < 26; i++) {
    r -= letterfreq[i];
    if (r < 0) return 'a' + i;
  }
  return 'e';
}


static uint32_t pick (Gen *g, const uint32_t *list, int n) {
  return list[mrR_int(&g->rng, n)];
}


/* one of the 8 letters with the fewest samples that the font can draw */
static uint32_t rareclass (Gen *g) {
  int idx[8], n = 0, i, k;
  for (i = 0; i < NLETTERS; i++) {
    if (!g->has[g->font][i]) continue;
    if (n < 8) idx[n++] = i;
    else {
      int worst = 0;
      for (k = 1; k < 8; k++)
        if (g->counts[idx[k]] > g->counts[idx[worst]]) worst = k;
      if (g->counts[i] < g->counts[idx[worst]]) idx[worst] = i;
    }
  }
  return (n > 0) ? g->classes[idx[mrR_int(&g->rng, n)]] : 'e';
}


static int put (uint32_t *line, int n, uint32_t c) {
  if (n < MAXLINE) line[n++] = c;
  return n;
}


/* a number, maybe with money, percent, degree or fraction signs */
static int gennumber (Gen *g, uint32_t *w) {
  static const uint32_t money[] = { '$', 0x20B1, 0x20AC, 0xA3, 0xA5 };
  static const uint32_t frac[] = { 0xBD, 0xBC, 0xBE };
  int len = 0, digits = 1 + mrR_int(&g->rng, 6), i, r;
  if (mrR_int(&g->rng, 100) < 15) w[len++] = pick(g, money, 5);
  for (i = 0; i < digits; i++) {
    if (i > 0 && (digits - i) % 3 == 0 && mrR_int(&g->rng, 3) == 0)
      w[len++] = ',';
    w[len++] = '0' + mrR_int(&g->rng, 10);
  }
  if (mrR_int(&g->rng, 6) == 0) {
    w[len++] = '.';
    w[len++] = '0' + mrR_int(&g->rng, 10);
    w[len++] = '0' + mrR_int(&g->rng, 10);
  }
  r = mrR_int(&g->rng, 100);
  if (r < 8) w[len++] = '%';
  else if (r < 12) {
    w[len++] = 0xB0;  /* ° */
    if (mrR_int(&g->rng, 2)) w[len++] = mrR_int(&g->rng, 2) ? 'C' : 'F';
  }
  else if (r < 14) w[len++] = 0xA2;  /* ¢ */
  else if (r < 17) w[len++] = pick(g, frac, 3);
  return len;
}


/* random accents on the vowels of a made-up word */
static uint32_t accent (Gen *g, uint32_t c) {
  if (mrR_int(&g->rng, 100) >= 3) return c;
  switch (c) {
    case 'a': return 0xE1;
    case 'e': return 0xE9;
    case 'i': return 0xED;
    case 'o': return 0xF3;
    case 'u': return mrR_int(&g->rng, 4) ? 0xFA : 0xFC;
    case 'n': return 0xF1;
    default: return c;
  }
}


static int genword (Gen *g, uint32_t *line, int n, int allcaps) {
  static const uint32_t punct[] = {
    '.', ',', ',', ',', '.', ';', ':', '!', '?', '-', 0x2026
  };
  static const uint32_t marks[] = { 0x2122, 0xAE, 0xA9 };  /* ™ ® © */
  uint32_t w[40];
  int len, i, r = mrR_int(&g->rng, 100);
  int wrap = mrR_int(&g->rng, 100);
  uint32_t open = 0, close = 0;
  if (r < 9)
    len = gennumber(g, w);
  else if (r < 55) {  /* a common word */
    int nw = 0;
    while (words[nw] != NULL) nw++;
    len = decode(words[mrR_int(&g->rng, nw)], w, 24);
    for (i = 0; i < len; i++)  /* it's -> it’s */
      if (w[i] == '\'' && mrR_int(&g->rng, 2)) w[i] = 0x2019;
  }
  else {  /* a made-up word */
    len = 1 + mrR_int(&g->rng, 9);
    for (i = 0; i < len; i++) w[i] = accent(g, randletter(g));
  }
  r = mrR_int(&g->rng, 100);
  for (i = 0; i < len; i++) {
    if (allcaps || r < 7 || (r < 25 && i == 0)) w[i] = upper(w[i]);
  }
  if (wrap < 3) { open = 0x201C; close = 0x201D; }  /* “ ” */
  else if (wrap < 5) { open = 0x2018; close = 0x2019; }  /* ‘ ’ */
  else if (wrap < 7) open = close = '"';
  else if (wrap < 8) open = close = '\'';
  else if (wrap < 10) { open = '('; close = ')'; }
  else if (wrap < 11) { open = '['; close = ']'; }
  else if (wrap < 12) { open = 0xBF; close = '?'; }  /* ¿ ? */
  else if (wrap < 13) { open = 0xA1; close = '!'; }  /* ¡ ! */
  if (open) n = put(line, n, open);
  for (i = 0; i < len; i++) n = put(line, n, w[i]);
  if (close) n = put(line, n, close);
  if (len > 0 && w[0] >= 'A' && w[0] <= 'Z' && mrR_int(&g->rng, 50) == 0)
    n = put(line, n, pick(g, marks, 3));
  if (mrR_int(&g->rng, 100) < 14)
    n = put(line, n, pick(g, punct, 11));
  return n;
}


/* what goes between two words */
static int gengap (Gen *g, uint32_t *line, int n) {
  static const uint32_t seps[] = {
    0x2013, 0x2014, 0xD7, 0xF7, 0xB1, '=', '+', '/'  /* – — × ÷ ± */
  };
  int r = mrR_int(&g->rng, 100);
  if (r < 2) return put(line, n, '-');  /* hyphenated-word */
  n = put(line, n, ' ');
  if (r < 6) {
    n = put(line, n, pick(g, seps, 8));
    n = put(line, n, ' ');
  }
  return n;
}


/* a line of text for the current font; returns the count */
static int genline (Gen *g, uint32_t *line) {
  static const uint32_t starts[] = { 0x2022, 0x2022, 0x2022, 0xA7, 0xB6 };
  int n = 0, target = 8 + mrR_int(&g->rng, 34), k, rare;
  int allcaps = (mrR_int(&g->rng, 100) < 5);
  if (mrR_int(&g->rng, 100) < 6) {  /* • item, § 3, ¶ 2 */
    n = put(line, n, pick(g, starts, 5));
    n = put(line, n, ' ');
  }
  while (n < target) {
    if (n > 0 && line[n - 1] != ' ') n = gengap(g, line, n);
    n = genword(g, line, n, allcaps);
  }
  /* mix in rare letters so every class gets examples */
  rare = mrR_int(&g->rng, 4);
  for (k = 0; k < rare && n + 3 < MAXLINE; k++) {
    int at = mrR_int(&g->rng, n + 1);
    uint32_t c = rareclass(g);
    int alone = mrR_int(&g->rng, 2);
    memmove(&line[at + 1 + alone], &line[at],
            mr_cast(size_t, n - at) * sizeof(uint32_t));
    line[at] = c;
    if (alone) line[at + 1] = ' ';
    n += 1 + alone;
  }
  for (k = 0; k < n; k++)  /* letters this font does not have */
    if (!candraw(g, line[k])) line[k] = randletter(g);
  while (n > 0 && line[n - 1] == ' ') n--;
  return n;
}

/* }====================================================== */


/*
** {======================================================
** Drawing with noise
** =======================================================
*/

static void addnoise (Gen *g, mr_Image *img, float sd) {
  size_t i, n = mr_cast(size_t, img->width) * img->height;
  for (i = 0; i < n; i++) {
    int v = img->pixels[i] + mr_cast(int, mrR_normal(&g->rng) * sd);
    img->pixels[i] = mr_cast(mr_byte, v < 0 ? 0 : v > 255 ? 255 : v);
  }
}


static int blur (mr_State *R, mr_Image *img) {
  int w = img->width, h = img->height, x, y;
  size_t n = mr_cast(size_t, w) * h;
  mr_byte *src = mr_cast(mr_byte *, mrM_malloc(R, n));
  if (src == NULL) return MR_ERRMEM;
  memcpy(src, img->pixels, n);
  for (y = 0; y < h; y++) {
    for (x = 0; x < w; x++) {
      int s = 0, c = 0, dx, dy;
      for (dy = -1; dy <= 1; dy++) {
        for (dx = -1; dx <= 1; dx++) {
          int nx = x + dx, ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
          s += src[mr_cast(size_t, ny) * w + nx];
          c++;
        }
      }
      img->pixels[mr_cast(size_t, y) * w + x] = mr_cast(mr_byte, s / c);
    }
  }
  mrM_free(R, src, n);
  return MR_OK;
}


/* a handwritten line from EMNIST letters (test writers if 'test') */
static int drawhand (Gen *g, const uint32_t *line, int n, mr_Image **img,
                     mr_Box *boxes, int test) {
  mr_HandDraw d;
  float u = mrR_float(&g->rng);
  d.xheight = 7 + 17 * u * sqrtf(u);
  d.paper = 180 + mrR_int(&g->rng, 76);
  d.ink = mrR_int(&g->rng, 111);
  if (mrR_int(&g->rng, 100) < 5) {  /* chalk on a dark board */
    int t = d.paper;
    d.paper = d.ink;
    d.ink = t;
  }
  d.pad = 4 + mrR_int(&g->rng, 9);
  d.touch = 8;
  return mrH_drawline(g->R, test ? g->handtest : g->hand, &g->rng,
                      g->fonts[g->font], line, n, &d, img, boxes);
}


static int drawline (Gen *g, const uint32_t *line, int n, mr_Image **img,
                     mr_Box *boxes, int hand) {
  mr_Draw d;
  float u = mrR_float(&g->rng);
  int status;
  const mr_Font *font = g->fonts[g->font];
  if (hand) {
    status = drawhand(g, line, n, img, boxes, hand == 2);
    if (status != MR_OK) return status;
    if (mrR_int(&g->rng, 100) < 15)
      addnoise(g, *img, mrR_range(&g->rng, 3, 14));
    return MR_OK;
  }
  d.height = 10 + 46 * u * sqrtf(u);  /* more small text */
  d.stretch = mrR_range(&g->rng, 0.85f, 1.15f);
  d.spacing = (mrR_int(&g->rng, 10) == 0) ? mrR_range(&g->rng, -0.5f, 0)
                                          : mrR_range(&g->rng, 0, 1.0f);
  d.paper = 170 + mrR_int(&g->rng, 86);
  d.ink = mrR_int(&g->rng, 91);
  if (mrR_int(&g->rng, 100) < 15) {  /* light text on dark */
    int t = d.paper;
    d.paper = d.ink;
    d.ink = t;
  }
  d.pad = 4 + mrR_int(&g->rng, 9);
  status = mrT_drawline(g->R, font, line, n, &d, img, boxes, NULL);
  if (status != MR_OK) return status;
  if (mrR_int(&g->rng, 100) < 20 && d.height > 16) {
    status = blur(g->R, *img);
    if (status != MR_OK) return status;
  }
  if (mrR_int(&g->rng, 100) < 30)
    addnoise(g, *img, mrR_range(&g->rng, 3, 18));
  return MR_OK;
}

/* }====================================================== */


/*
** {======================================================
** Samples: run the real layout code, then match segments to letters
** =======================================================
*/

static int boxarea (const mr_Box *b) {
  int w = b->x1 - b->x0, h = b->y1 - b->y0;
  return (w > 0 && h > 0) ? w * h : 0;
}


static int interarea (const mr_Box *a, const mr_Box *b) {
  int w = mrL_xoverlap(a, b), h = mrL_yoverlap(a, b);
  return (w > 0 && h > 0) ? w * h : 0;
}


/* which letter owns segment 's'; -1 if none, -2 if it touches two */
static int owner (const mr_Seg *s, const uint32_t *line,
                  const mr_Box *boxes, int n) {
  int i, best = -1, bestov = 0, second = 0;
  for (i = 0; i < n; i++) {
    int ov;
    if (line[i] == ' ' || boxarea(&boxes[i]) == 0) continue;
    ov = interarea(&s->box, &boxes[i]);
    if (ov > bestov) {
      second = bestov;
      bestov = ov;
      best = i;
    }
    else if (ov > second)
      second = ov;
  }
  if (best >= 0 && second * 3 > bestov) return -2;
  return best;
}


static int addsamples (Gen *g, const uint32_t *line, const mr_Box *boxes,
                       int n, const mr_Image *img) {
  mr_Bitmap *bm = NULL;
  mr_Layout lo;
  int *own = NULL;
  size_t i, nown = 0;
  int status, k;
  mrL_init(&lo);
  for (k = 0; k < n; k++)
    if (line[k] != ' ') g->chars++;
  status = mrK_fromgray(g->R, img, &bm);
  if (status != MR_OK) goto done;
  status = mrL_analyze(g->R, bm, &lo);
  if (status != MR_OK) goto done;
  if (lo.nlines != 1) {
    g->badlines++;
    goto done;
  }
  nown = lo.lines[0].count;
  own = mrM_newarray(g->R, nown + 1, int);
  if (own == NULL) {
    status = MR_ERRMEM;
    goto done;
  }
  for (i = 0; i < nown; i++)
    own[i] = owner(&lo.segs[i], line, boxes, n);
  /* glue segments of one letter (like '%' or a broken stroke) */
  i = 0;
  while (i + 1 < lo.lines[0].count) {
    if (own[i] >= 0 && own[i] == own[i + 1]) {
      mrL_mergenext(&lo, &lo.lines[0], i);
      memmove(&own[i + 1], &own[i + 2],
              (lo.lines[0].count - i - 1) * sizeof(int));
    }
    else i++;
  }
  for (i = 0; i < lo.lines[0].count && g->npool < POOL; i++) {
    const mr_Seg *s = &lo.segs[i];
    const mr_Box *cb;
    int c, inter, uni;
    if (own[i] < 0) continue;
    cb = &boxes[own[i]];
    inter = interarea(&s->box, cb);
    uni = boxarea(&s->box) + boxarea(cb) - inter;
    if (inter * 2 < uni) continue;  /* matched badly */
    c = classof(g, line[own[i]]);
    if (c < 0) continue;
    mrG_extract(&lo, &lo.lines[0], &s->box, mr_cast(long, i), -1,
                g->pool + mr_cast(size_t, g->npool) * MR_GLYPHINPUT);
    g->labels[g->npool++] = c;
    g->counts[c]++;
    g->used++;
  }
  /* "not one letter": real touching pairs, and neighbors glued together,
     so the reader does not glue two letters into a wrong one */
  for (i = 0; i < lo.lines[0].count && g->npool < POOL; i++) {
    mr_Box box = lo.segs[i].box;
    long other = -1;
    if (own[i] == -2) {
      if (mrR_int(&g->rng, 2)) continue;
    }
    else if (own[i] >= 0 && i + 1 < lo.lines[0].count && own[i + 1] >= 0 &&
             own[i + 1] != own[i] && !lo.segs[i + 1].space &&
             mrR_int(&g->rng, 100) < 12) {
      other = mr_cast(long, i + 1);
      mrL_join(&box, &lo.segs[i + 1].box);
    }
    else continue;
    mrG_extract(&lo, &lo.lines[0], &box, mr_cast(long, i), other,
                g->pool + mr_cast(size_t, g->npool) * MR_GLYPHINPUT);
    g->labels[g->npool++] = REJECT;
    g->counts[REJECT]++;
  }
 done:
  if (own != NULL) mrM_freearray(g->R, own, nown + 1);
  mrL_free(g->R, &lo);
  mrK_free(g->R, bm);
  return status;
}


/* choose the font of the next line: a handwriting-like one if 'hand' */
static void pickfont (Gen *g, int hand) {
  int k, i;
  if (!hand || g->nhand == 0) {
    g->font = mrR_int(&g->rng, g->nfonts);
    return;
  }
  k = mrR_int(&g->rng, g->nhand);
  for (i = 0; i < g->nfonts; i++)
    if (g->ishand[i] && k-- == 0) break;
  g->font = i;
}


static int fillpool (Gen *g) {
  uint32_t line[MAXLINE];
  mr_Box boxes[MAXLINE];
  g->npool = 0;
  while (g->npool < POOL) {
    mr_Image *img = NULL;
    int n, status;
    int hand = (g->hand != NULL && mrR_int(&g->rng, 100) < g->handpct);
    pickfont(g, hand);
    n = genline(g, line);
    status = drawline(g, line, n, &img, boxes, hand);
    if (status == MR_OK) status = addsamples(g, line, boxes, n, img);
    mrI_free(g->R, img);
    if (status != MR_OK) return status;
    g->lines++;
  }
  return MR_OK;
}

/* }====================================================== */


/*
** {======================================================
** Test: read new lines with the full OCR and count errors
** =======================================================
*/

/* edit distance between code point strings */
static int distance (mr_State *R, const uint32_t *a, int na,
                     const uint32_t *b, int nb) {
  int *row = mrM_newarray(R, nb + 1, int);
  int i, j, d;
  if (row == NULL) return na > nb ? na : nb;
  for (j = 0; j <= nb; j++) row[j] = j;
  for (i = 1; i <= na; i++) {
    int diag = row[0];
    row[0] = i;
    for (j = 1; j <= nb; j++) {
      int up = row[j];
      int best = diag + (a[i - 1] != b[j - 1]);
      if (up + 1 < best) best = up + 1;
      if (row[j - 1] + 1 < best) best = row[j - 1] + 1;
      row[j] = best;
      diag = up;
    }
  }
  d = row[nb];
  mrM_freearray(R, row, nb + 1);
  return d;
}


static void printcps (mr_State *R, const char *label, const uint32_t *s,
                      int n) {
  mr_Buffer b;
  int i;
  mrB_init(&b);
  for (i = 0; i < n; i++) mrB_addutf8(R, &b, s[i]);
  printf("  %s %s\n", label, mrB_cstr(&b));
  mrB_free(R, &b);
}


/* 'hand': 0 print, 1 handwriting from the EMNIST test writers */
static int testmodel (Gen *g, const mr_Net *net, int nlines, int hand) {
  uint32_t line[MAXLINE], got[MAXLINE * 2];
  mr_Box boxes[MAXLINE];
  mr_Buffer text;
  long errs = 0, total = 0;
  int i, exact = 0, shown = 0, status = MR_OK;
  mrB_init(&text);
  for (i = 0; i < nlines; i++) {
    mr_Image *img = NULL;
    int n, ngot, d;
    pickfont(g, hand);
    n = genline(g, line);
    status = drawline(g, line, n, &img, boxes, hand ? 2 : 0);
    if (status == MR_OK) {
      mrB_reset(&text);
      status = mrO_run(g->R, net, img, &text);
    }
    mrI_free(g->R, img);
    if (status != MR_OK) break;
    ngot = decode(mrB_cstr(&text), got, MAXLINE * 2);
    while (ngot > 0 && (got[ngot - 1] == '\n' || got[ngot - 1] == ' '))
      ngot--;
    d = distance(g->R, line, n, got, ngot);
    errs += d;
    total += n;
    if (d == 0) exact++;
    else if (shown < 8) {
      printcps(g->R, "want:", line, n);
      printcps(g->R, "got: ", got, ngot);
      shown++;
    }
  }
  mrB_free(g->R, &text);
  if (status == MR_OK)
    printf("test (%s): %d lines, %.2f%% character errors, "
           "%.1f%% lines exact\n", hand ? "handwriting" : "print", nlines,
           total ? 100.0 * errs / total : 0.0,
           nlines ? 100.0 * exact / nlines : 0.0);
  return status;
}

/* }====================================================== */


/*
** {======================================================
** Training
** =======================================================
*/

static void shuffle (Gen *g, int *order, int n) {
  int i;
  for (i = n - 1; i > 0; i--) {
    int j = mrR_int(&g->rng, i + 1), t = order[i];
    order[i] = order[j];
    order[j] = t;
  }
}


static int train (Gen *g, const Options *opt) {
  mr_State *R = g->R;
  mr_Net *net = NULL;
  mr_Trainer *tr = NULL;
  float *batch = NULL;
  int sizes[MR_MAXLAYERS + 1], order[POOL], blabels[BATCH];
  int nl, i, status;
  long done = 0;
  clock_t start = clock();
  sizes[0] = MR_GLYPHINPUT;
  for (i = 0; i < opt->nhidden; i++) sizes[i + 1] = opt->hidden[i];
  nl = opt->nhidden + 1;
  sizes[nl] = NCLASSES;
  status = mrN_new(R, nl, sizes, &net);
  if (status != MR_OK) goto done;
  memcpy(net->classes, g->classes, sizeof(g->classes));
  mrN_randomize(net, &g->rng);
  status = mrN_newtrainer(R, net, &tr);
  if (status != MR_OK) goto done;
  batch = mrM_newarray(R, mr_cast(size_t, BATCH) * MR_GLYPHINPUT, float);
  if (batch == NULL) {
    status = MR_ERRMEM;
    goto done;
  }
  printf("network:");
  for (i = 0; i <= nl; i++) printf(" %d", sizes[i]);
  printf("  fonts: %d  samples: %ld\n", g->nfonts, opt->samples);
  while (done < opt->samples) {
    float loss = 0, frac = mr_cast(float, done) / opt->samples;
    float rate = opt->rate;
    int right = 0, steps = 0;
    if (frac > 0.6f)  /* slow down at the end */
      rate *= 1 - 0.9f * (frac - 0.6f) / 0.4f;
    status = fillpool(g);
    if (status != MR_OK) goto done;
    for (i = 0; i < POOL; i++) order[i] = i;
    shuffle(g, order, POOL);
    for (i = 0; i + BATCH <= POOL; i += BATCH) {
      int k, ok;
      for (k = 0; k < BATCH; k++) {
        memcpy(batch + mr_cast(size_t, k) * MR_GLYPHINPUT,
               g->pool + mr_cast(size_t, order[i + k]) * MR_GLYPHINPUT,
               MR_GLYPHINPUT * sizeof(float));
        blabels[k] = g->labels[order[i + k]];
      }
      loss += mrN_train(tr, net, batch, blabels, BATCH, rate, &ok);
      right += ok;
      steps++;
    }
    done += POOL;
    printf("%8ld  loss %.4f  right %5.1f%%  skipped %4.1f%%  %.0fs\n",
           done, loss / steps, 100.0 * right / (steps * BATCH),
           g->chars ? 100.0 * (g->chars - g->used) / g->chars : 0.0,
           mr_cast(double, clock() - start) / CLOCKS_PER_SEC);
    fflush(stdout);
  }
  status = mrN_save(R, net, opt->output);
  if (status != MR_OK) goto done;
  printf("saved %s\n", opt->output);
  status = testmodel(g, net, opt->testlines, 0);
  if (status == MR_OK && g->hand != NULL)
    status = testmodel(g, net, opt->testlines, 1);
 done:
  if (batch != NULL)
    mrM_freearray(R, batch, mr_cast(size_t, BATCH) * MR_GLYPHINPUT);
  mrN_freetrainer(R, tr);
  mrN_free(R, net);
  return status;
}

/* }====================================================== */


/*
** {======================================================
** Example lines (-d): see what the network learns from
** =======================================================
*/

static int savepgm (mr_State *R, const mr_Image *img, const char *path) {
  FILE *f = mrF_open(path, "wb");
  size_t n = mr_cast(size_t, img->width) * img->height;
  int ok;
  if (f == NULL) return mrS_error(R, MR_ERRFILE, "cannot create '%s'", path);
  ok = fprintf(f, "P5\n%d %d\n255\n", img->width, img->height) > 0 &&
       fwrite(img->pixels, 1, n, f) == n;
  if (fclose(f) != 0) ok = 0;
  return ok ? MR_OK : mrS_error(R, MR_ERRFILE, "cannot write '%s'", path);
}


static int dumplines (Gen *g, const char *dir) {
  uint32_t line[MAXLINE];
  mr_Box boxes[MAXLINE];
  char path[MR_PATHSIZE];
  FILE *truth;
  int i, status = MR_OK;
  snprintf(path, sizeof(path), "%s/lines.txt", dir);
  truth = mrF_open(path, "wb");
  if (truth == NULL)
    return mrS_error(g->R, MR_ERRFILE, "cannot create '%s'", path);
  for (i = 0; i < 20 && status == MR_OK; i++) {
    mr_Image *img = NULL;
    mr_Buffer b;
    int n, k, hand = (g->hand != NULL && i % 2 == 1);
    pickfont(g, hand);
    n = genline(g, line);
    status = drawline(g, line, n, &img, boxes, hand);
    if (status == MR_OK) {
      snprintf(path, sizeof(path), "%s/line%02d.pgm", dir, i);
      status = savepgm(g->R, img, path);
    }
    mrI_free(g->R, img);
    mrB_init(&b);
    for (k = 0; k < n; k++) mrB_addutf8(g->R, &b, line[k]);
    fprintf(truth, "line%02d %s %s\n", i, hand ? "hand " : "print",
            mrB_cstr(&b));
    mrB_free(g->R, &b);
  }
  fclose(truth);
  if (status == MR_OK) printf("saved 20 lines in %s\n", dir);
  return status;
}

/* }====================================================== */


static int parsehidden (const char *s, Options *opt) {
  opt->nhidden = 0;
  while (*s) {
    char *end;
    long v = strtol(s, &end, 10);
    if (end == s || v < 1 || v > 4096 || opt->nhidden >= MR_MAXLAYERS - 1)
      return 0;
    opt->hidden[opt->nhidden++] = mr_cast(int, v);
    s = end;
    if (*s == ',') s++;
    else if (*s != '\0') return 0;
  }
  return opt->nhidden > 0;
}


static int collectargs (int argc, char **argv, Options *opt) {
  int i;
  opt->output = DEF_OUTPUT;
  opt->test = NULL;
  opt->samples = DEF_SAMPLES;
  opt->seed = 1;
  opt->rate = DEF_RATE;
  opt->hidden[0] = 256;
  opt->hidden[1] = 128;
  opt->nhidden = 2;
  opt->testlines = TESTLINES;
  opt->handdir = NULL;
  opt->dumpdir = NULL;
  opt->handpct = DEF_HANDPCT;
  for (i = 1; i < argc; i++) {
    const char *a = argv[i], *val;
    if (a[0] != '-' || a[1] == '\0') break;
    if (a[2] != '\0') goto bad;
    if (a[1] == 'h') {
      print_usage(NULL);
      return -1;
    }
    if (i + 1 >= argc) goto bad;
    val = argv[++i];
    switch (a[1]) {
      case 'o': opt->output = val; break;
      case 't': opt->test = val; break;
      case 'n': opt->samples = atol(val); break;
      case 's': opt->seed = atol(val); break;
      case 'r': opt->rate = mr_cast(float, atof(val)); break;
      case 'c': opt->testlines = atoi(val); break;
      case 'H': opt->handdir = val; break;
      case 'd': opt->dumpdir = val; break;
      case 'm': opt->handpct = atoi(val); break;
      case 'l':
        if (!parsehidden(val, opt)) goto bad;
        break;
      default: goto bad;
    }
  }
  if (opt->samples < POOL || opt->rate <= 0 || opt->testlines < 1) {
    print_usage(NULL);
    return 2;
  }
  opt->first = i;
  return 0;
 bad:
  print_usage(argv[i]);
  return 2;
}


/*
** Load a font "path" or "path#index". Fonts without every ASCII letter
** and 'ñ' are skipped; other missing letters are just not drawn with it.
*/
static int addfont (Gen *g, const char *arg, int quiet) {
  char path[MR_PATHSIZE];
  const char *hash = strrchr(arg, '#');
  int index = 0, c, status;
  mr_Font *f;
  size_t len = hash ? mr_cast(size_t, hash - arg) : strlen(arg);
  if (len >= sizeof(path) || g->nfonts >= MAXFONTS) return MR_ERRARG;
  memcpy(path, arg, len);
  path[len] = '\0';
  if (hash) index = atoi(hash + 1);
  status = mrT_load(g->R, path, index, &f);
  if (status != MR_OK) {
    if (!quiet) report(g->R, status);
    return status;
  }
  for (c = 0; c < NBASE; c++) {
    if (!mrT_hasglyph(f, g->classes[c])) {
      if (!quiet)
        fprintf(stderr, "%s: %s: no letter U+%04X, skipped\n", progname,
                path, mr_cast(unsigned, g->classes[c]));
      mrT_free(g->R, f);
      return MR_ERRFORMAT;
    }
  }
  for (c = 0; c < NLETTERS; c++)
    g->has[g->nfonts][c] = mr_cast(mr_byte, mrT_hasglyph(f, g->classes[c]));
  g->fonts[g->nfonts++] = f;
  return MR_OK;
}


/* tell which letters no font, or few fonts, can draw */
static void coverage (const Gen *g) {
  int c, i;
  for (c = NBASE; c < NLETTERS; c++) {
    int n = 0;
    for (i = 0; i < g->nfonts; i++) n += g->has[i][c];
    if (n * 4 < g->nfonts)
      fprintf(stderr, "%s: note: only %d of %d fonts have U+%04X\n",
              progname, n, g->nfonts, mr_cast(unsigned, g->classes[c]));
  }
}


int main (int argc, char **argv) {
  Options opt;
  Gen g;
  mr_Net *net = NULL;
  int i, status;
  status = collectargs(argc, argv, &opt);
  if (status != 0) return (status < 0) ? EXIT_SUCCESS : status;
  memset(&g, 0, sizeof(g));
  g.R = mr_open();
  if (g.R == NULL) {
    l_message("not enough memory");
    return EXIT_FAILURE;
  }
  mrR_seed(&g.rng, mr_cast(uint64_t, opt.seed));
  for (i = 0; i < 94; i++) g.classes[i] = mr_cast(uint32_t, '!' + i);
  for (i = 0; i < NEXTRAS; i++) g.classes[94 + i] = extras[i];
  g.classes[REJECT] = MR_REJECT;
  if (opt.first < argc) {
    for (i = opt.first; i < argc; i++) addfont(&g, argv[i], 0);
  }
  else {
    char path[MR_PATHSIZE];
    for (i = 0; deffonts[i] != NULL; i++) {
      snprintf(path, sizeof(path), "C:/Windows/Fonts/%s", deffonts[i]);
      addfont(&g, path, 1);
    }
  }
  if (g.nfonts == 0) {
    l_message("no usable fonts; give some .ttf files");
    status = 2;
    goto done;
  }
  if (opt.handdir != NULL) {
    char path[MR_PATHSIZE];
    for (i = 0; handfonts[i] != NULL; i++) {
      snprintf(path, sizeof(path), "C:/Windows/Fonts/%s", handfonts[i]);
      if (addfont(&g, path, 1) == MR_OK) {
        g.ishand[g.nfonts - 1] = 1;
        g.nhand++;
      }
    }
    printf("loading EMNIST from %s...\n", opt.handdir);
    fflush(stdout);
    status = mrH_load(g.R, opt.handdir, "train", HANDPER, &g.hand);
    if (status == MR_OK)
      status = mrH_load(g.R, opt.handdir, "test", HANDPER / 4, &g.handtest);
    if (status != MR_OK) {
      report(g.R, status);
      status = EXIT_FAILURE;
      goto done;
    }
    g.handpct = opt.handpct;
    printf("handwriting: %lu + %lu samples, %d handwriting fonts, "
           "%d%% of lines\n", mr_cast(unsigned long, mrH_count(g.hand)),
           mr_cast(unsigned long, mrH_count(g.handtest)), g.nhand,
           g.handpct);
  }
  coverage(&g);
  g.pool = mrM_newarray(g.R, mr_cast(size_t, POOL) * MR_GLYPHINPUT, float);
  if (g.pool == NULL) {
    l_message("not enough memory");
    status = EXIT_FAILURE;
    goto done;
  }
  if (opt.dumpdir != NULL)
    status = dumplines(&g, opt.dumpdir);
  else if (opt.test != NULL) {
    printf("fonts: %d\n", g.nfonts);
    status = mrN_load(g.R, opt.test, &net);
    if (status == MR_OK) status = testmodel(&g, net, opt.testlines, 0);
    if (status == MR_OK && g.hand != NULL)
      status = testmodel(&g, net, opt.testlines, 1);
  }
  else
    status = train(&g, &opt);
  report(g.R, status);
  status = (status == MR_OK) ? EXIT_SUCCESS : EXIT_FAILURE;
 done:
  mrN_free(g.R, net);
  if (g.pool != NULL)
    mrM_freearray(g.R, g.pool, mr_cast(size_t, POOL) * MR_GLYPHINPUT);
  for (i = 0; i < g.nfonts; i++) mrT_free(g.R, g.fonts[i]);
  mrH_free(g.R, g.hand);
  mrH_free(g.R, g.handtest);
  mr_close(g.R);
  return status;
}
