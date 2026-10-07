/*
** mrglyph.cpp
** Letter segment -> neural network input
** See Copyright Notice in mr.h
*/

#define mrglyph_cpp
#define MR_CORE

#include "mrglyph.h"

#include <math.h>


/* empty border around the shape, in cells */
#define PAD  2

/* samples per cell side when scaling */
#define SUB  3


static float clampf (float v, float lo, float hi) {
  return (v < lo) ? lo : (v > hi) ? hi : v;
}


/* 1 if pixel (x, y) is ink of a segment from 'sega' to 'segb' */
static int member (const mr_Layout *lo, long sega, long segb, int x, int y) {
  int l = lo->labels[mr_cast(size_t, y) * mr_cast(size_t, lo->width) + x];
  long s;
  if (l == 0) return 0;
  s = lo->comps[l - 1].seg;
  return s >= sega && s <= ((segb < 0) ? sega : segb);
}


void mrG_extract (const mr_Layout *lo, const mr_Line *ln, const mr_Box *box,
                  long sega, long segb, float *in) {
  int w = box->x1 - box->x0, h = box->y1 - box->y0;
  int big = (w > h) ? w : h;
  float scale = mr_cast(float, MR_GLYPHSIZE - 2 * PAD) / mr_cast(float, big);
  float ox = (MR_GLYPHSIZE - w * scale) / 2;
  float oy = (MR_GLYPHSIZE - h * scale) / 2;
  float xh = mr_cast(float, ln->xheight);
  float *feat = in + MR_GLYPHSIZE * MR_GLYPHSIZE;
  int u, v, su, sv;
  for (v = 0; v < MR_GLYPHSIZE; v++) {
    for (u = 0; u < MR_GLYPHSIZE; u++) {
      int count = 0;
      for (sv = 0; sv < SUB; sv++) {
        for (su = 0; su < SUB; su++) {
          float fx = (u + (su + 0.5f) / SUB - ox) / scale;
          float fy = (v + (sv + 0.5f) / SUB - oy) / scale;
          int x, y;
          if (fx < 0 || fy < 0) continue;
          x = box->x0 + mr_cast(int, fx);
          y = box->y0 + mr_cast(int, fy);
          if (x >= box->x1 || y >= box->y1) continue;
          if (member(lo, sega, segb, x, y)) count++;
        }
      }
      in[v * MR_GLYPHSIZE + u] = mr_cast(float, count) / (SUB * SUB);
    }
  }
  feat[0] = clampf((ln->baseline - box->y0) / xh, -4, 4) * 0.5f;
  feat[1] = clampf((box->y1 - ln->baseline) / xh, -4, 4) * 0.5f;
  feat[2] = clampf(w / xh, 0, 8) * 0.25f;
  feat[3] = clampf(h / xh, 0, 8) * 0.25f;
  feat[4] = clampf(log2f(mr_cast(float, w) / mr_cast(float, h)), -4, 4) * 0.5f;
}
