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


/* everything needed while reading one image */
typedef struct Reader {
  mr_State *R;
  const mr_Net *net;
  mr_Layout lo;
  float *in;  /* network input */
  float *work;  /* network scratch */
  int *cls;  /* class of each segment of the current line */
  float *conf;  /* its confidence */
  size_t cap;  /* size of 'cls' and 'conf' */
} Reader;


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


/*
** Glue neighbor segments when the network is more sure about them
** together: broken letters, '%' and other letters in several parts.
*/
static void glue (Reader *rd, mr_Line *ln) {
  size_t i = 0;
  while (i + 1 < ln->count) {
    size_t g = ln->first + i;
    mr_Seg *a = &rd->lo.segs[g], *b = &rd->lo.segs[g + 1];
    float lowest = (rd->conf[i] < rd->conf[i + 1]) ? rd->conf[i]
                                                   : rd->conf[i + 1];
    float highest = (rd->conf[i] > rd->conf[i + 1]) ? rd->conf[i]
                                                    : rd->conf[i + 1];
    int gap = b->box.x0 - a->box.x1;
    if (!b->space && (gap <= 0 || lowest < LOWCONF)) {
      mr_Box box = a->box;
      int c;
      float cf;
      mrL_join(&box, &b->box);
      if (!classify(rd, ln, &box, mr_cast(long, g), mr_cast(long, g + 1),
                    &c, &cf) &&
          cf >= GLUECONF && cf > highest) {
        mrL_mergenext(&rd->lo, ln, i);
        rd->cls[i] = c;
        rd->conf[i] = cf;
        memmove(&rd->cls[i + 1], &rd->cls[i + 2],
                (ln->count - i - 1) * sizeof(int));
        memmove(&rd->conf[i + 1], &rd->conf[i + 2],
                (ln->count - i - 1) * sizeof(float));
        continue;  /* maybe glue the next one too */
      }
    }
    i++;
  }
}


static int readline (Reader *rd, mr_Line *ln, mr_Buffer *out) {
  mr_State *R = rd->R;
  size_t i;
  uint32_t prev = 0;
  int status;
  if (ln->count > rd->cap) {
    int *c = mrM_newarray(R, ln->count, int);
    float *f = mrM_newarray(R, ln->count, float);
    if (c == NULL || f == NULL) {
      if (c != NULL) mrM_freearray(R, c, ln->count);
      if (f != NULL) mrM_freearray(R, f, ln->count);
      return MR_ERRMEM;
    }
    if (rd->cls != NULL) mrM_freearray(R, rd->cls, rd->cap);
    if (rd->conf != NULL) mrM_freearray(R, rd->conf, rd->cap);
    rd->cls = c;
    rd->conf = f;
    rd->cap = ln->count;
  }
  for (i = 0; i < ln->count; i++) {
    size_t g = ln->first + i;
    classify(rd, ln, &rd->lo.segs[g].box, mr_cast(long, g), -1,
             &rd->cls[i], &rd->conf[i]);
  }
  glue(rd, ln);
  for (i = 0; i < ln->count; i++) {
    const mr_Seg *s = &rd->lo.segs[ln->first + i];
    uint32_t cp = rd->net->classes[rd->cls[i]];
    if (s->space) {
      status = mrB_addchar(R, out, ' ');
      if (status != MR_OK) return status;
      prev = ' ';
    }
    if (cp == '\'' && prev == '\'') {  /* two quotes: '"' */
      out->data[out->len - 1] = '"';
      prev = '"';
      continue;
    }
    status = mrB_addutf8(R, out, cp);
    if (status != MR_OK) return status;
    prev = cp;
  }
  return mrB_addchar(R, out, '\n');
}


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
  if (rd.cls != NULL) mrM_freearray(R, rd.cls, rd.cap);
  if (rd.conf != NULL) mrM_freearray(R, rd.conf, rd.cap);
  mrL_free(R, &rd.lo);
  mrK_free(R, bm);
  return status;
}


int mrO_loadmodel (mr_State *R) {
  char path[MR_PATHSIZE + MR_LANGSIZE + 8];
  size_t n = strlen(R->datapath);
  if (R->net != NULL) return MR_OK;
  if (n > 0) {
    memcpy(path, R->datapath, n);
    if (path[n - 1] != '/' && path[n - 1] != '\\') path[n++] = '/';
  }
  memcpy(path + n, R->lang, strlen(R->lang));
  n += strlen(R->lang);
  memcpy(path + n, MR_MODELEXT, sizeof(MR_MODELEXT));
  return mrN_load(R, path, &R->net);
}


int mrO_recognize (mr_State *R, const mr_Image *img, mr_Buffer *out) {
  int status = mrO_loadmodel(R);
  if (status != MR_OK) return status;
  return mrO_run(R, R->net, img, out);
}
