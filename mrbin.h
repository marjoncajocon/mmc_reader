/*
** mrbin.h
** Black/white bitmaps: turn a gray image into ink and paper
** See Copyright Notice in mr.h
*/

#ifndef mrbin_h
#define mrbin_h

#include "mrlimits.h"
#include "mrimage.h"


/* one byte per pixel: 1 = ink (text), 0 = paper (background) */
typedef struct mr_Bitmap {
  int width;
  int height;
  mr_byte *bits;
} mr_Bitmap;


#define mrK_at(bm, x, y)  \
  ((bm)->bits[mr_cast(size_t, (y)) * mr_cast(size_t, (bm)->width) + (x)])


MRI_FUNC int mrK_new (mr_State *R, int width, int height, mr_Bitmap **out);
MRI_FUNC int mrK_fromgray (mr_State *R, const mr_Image *gray,
                           mr_Bitmap **out);
MRI_FUNC void mrK_free (mr_State *R, mr_Bitmap *bm);

#endif
