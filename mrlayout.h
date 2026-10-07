/*
** mrlayout.h
** Page layout: connected parts -> text lines -> letter segments
** See Copyright Notice in mr.h
*/

#ifndef mrlayout_h
#define mrlayout_h

#include "mrlimits.h"
#include "mrbin.h"


/* a rectangle; x1 and y1 are one past the last pixel */
typedef struct mr_Box {
  int x0, y0, x1, y1;
} mr_Box;


/* a connected group of ink pixels */
typedef struct mr_Comp {
  mr_Box box;
  int area;  /* number of pixels */
  int seg;  /* segment it belongs to, -1 if ignored (noise, rules) */
} mr_Comp;


/* one letter candidate: one or more components */
typedef struct mr_Seg {
  mr_Box box;
  int space;  /* 1 if a space comes before this segment */
} mr_Seg;


typedef struct mr_Line {
  mr_Box box;
  int baseline;  /* y of the baseline */
  int xheight;  /* height of 'x' in pixels, at least 1 */
  int blank;  /* 1 if an empty line comes before this one */
  size_t first;  /* index of the first segment in 'segs' */
  size_t count;  /* number of segments */
} mr_Line;


typedef struct mr_Layout {
  int width, height;
  int *labels;  /* per pixel: component index + 1, 0 = paper */
  mr_Comp *comps;
  size_t ncomps, capcomps;
  mr_Seg *segs;
  size_t nsegs, capsegs;
  mr_Line *lines;
  size_t nlines, caplines;
} mr_Layout;


MRI_FUNC void mrL_init (mr_Layout *lo);
MRI_FUNC int mrL_analyze (mr_State *R, const mr_Bitmap *bm, mr_Layout *lo);
MRI_FUNC void mrL_free (mr_State *R, mr_Layout *lo);

/* 1 if pixel (x, y) is ink of segment 'seg' */
MRI_FUNC int mrL_inseg (const mr_Layout *lo, size_t seg, int x, int y);

/* merge segment 'i + 1' into segment 'i' of 'ln' (glued letter parts) */
MRI_FUNC void mrL_mergenext (mr_Layout *lo, mr_Line *ln, size_t i);

MRI_FUNC int mrL_xoverlap (const mr_Box *a, const mr_Box *b);
MRI_FUNC int mrL_yoverlap (const mr_Box *a, const mr_Box *b);
MRI_FUNC void mrL_join (mr_Box *a, const mr_Box *b);

#endif
