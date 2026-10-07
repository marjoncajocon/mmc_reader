/*
** mrhand.h
** Handwritten letters (EMNIST) drawn into text lines, for training
** See Copyright Notice in mr.h
*/

#ifndef mrhand_h
#define mrhand_h

#include <stdint.h>

#include "mrlimits.h"
#include "mrfont.h"
#include "mrimage.h"
#include "mrlayout.h"
#include "mrrand.h"


/* loaded EMNIST samples; its fields belong to mrhand.cpp */
typedef struct mr_Hand mr_Hand;


/* how to draw a handwritten line */
typedef struct mr_HandDraw {
  float xheight;  /* height of small letters, in pixels */
  int ink;  /* gray level of the pen */
  int paper;  /* gray level of the paper */
  int pad;  /* empty border, in pixels */
  int touch;  /* percent of letters that touch the next one */
} mr_HandDraw;


/*
** Load the EMNIST 'byclass' split 'split' ("train" or "test") from
** folder 'dir' (unpacked .idx files, see getdata.sh), at most 'maxper'
** samples per letter.
*/
MRI_FUNC int mrH_load (mr_State *R, const char *dir, const char *split,
                       int maxper, mr_Hand **out);
MRI_FUNC void mrH_free (mr_State *R, mr_Hand *h);
MRI_FUNC size_t mrH_count (const mr_Hand *h);

/* 1 if 'cp' is drawn from handwriting (EMNIST, maybe plus an accent) */
MRI_FUNC int mrH_has (const mr_Hand *h, uint32_t cp);

/*
** Draw 'cps[0..n-1]' as one handwritten line. Letters 'mrH_has' does
** not know are drawn with 'font' (a handwriting-like font). 'boxes[i]'
** gets the ink box of letter i, like 'mrT_drawline'.
*/
MRI_FUNC int mrH_drawline (mr_State *R, const mr_Hand *h, mr_Rand *rng,
                           const mr_Font *font, const uint32_t *cps, int n,
                           const mr_HandDraw *d, mr_Image **out,
                           mr_Box *boxes);

#endif
