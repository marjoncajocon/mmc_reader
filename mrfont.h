/*
** mrfont.h
** TrueType fonts: draw text lines to make training images
** See Copyright Notice in mr.h
*/

#ifndef mrfont_h
#define mrfont_h

#include <stdint.h>

#include "mrlimits.h"
#include "mrimage.h"
#include "mrlayout.h"


/* an open font; its fields belong to mrfont.cpp */
typedef struct mr_Font mr_Font;


/* how to draw a line */
typedef struct mr_Draw {
  float height;  /* line height in pixels (ascent + descent) */
  float stretch;  /* horizontal scale, 1 = normal */
  float spacing;  /* extra space between letters, in pixels */
  int ink;  /* gray level of the text */
  int paper;  /* gray level of the background */
  int pad;  /* empty border, in pixels */
} mr_Draw;


/* 'index' picks a font in a collection (.ttc); 0 for plain .ttf */
MRI_FUNC int mrT_load (mr_State *R, const char *path, int index,
                       mr_Font **out);
MRI_FUNC void mrT_free (mr_State *R, mr_Font *f);
MRI_FUNC int mrT_hasglyph (const mr_Font *f, uint32_t cp);

/* height of 'x' as a part of the line height (ascent + descent) */
MRI_FUNC float mrT_xheight (const mr_Font *f);

/*
** Draw code points 'cps[0..n-1]' as one gray line. 'boxes[i]' gets the
** ink box of letter i (empty, x0 == x1, for spaces). '*baseline' (if
** not NULL) gets the y of the baseline in the image.
*/
MRI_FUNC int mrT_drawline (mr_State *R, const mr_Font *f,
                           const uint32_t *cps, int n, const mr_Draw *d,
                           mr_Image **out, mr_Box *boxes, int *baseline);

#endif
