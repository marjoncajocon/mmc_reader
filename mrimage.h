/*
** mrimage.h
** Image loading and pixel access
** See Copyright Notice in mr.h
*/

#ifndef mrimage_h
#define mrimage_h

#include "mrlimits.h"


/*
** 8 bits per channel, rows top to bottom, no padding between rows.
** 'channels': 1 gray, 2 gray+alpha, 3 RGB, 4 RGBA.
*/
typedef struct mr_Image {
  int width;
  int height;
  int channels;
  mr_byte *pixels;  /* width * height * channels bytes */
} mr_Image;


MRI_FUNC int mrI_new (mr_State *R, int width, int height, int channels,
                      mr_Image **out);
MRI_FUNC int mrI_load (mr_State *R, const char *path, mr_Image **out);
MRI_FUNC int mrI_loadmem (mr_State *R, const mr_byte *data, size_t size,
                          mr_Image **out);
MRI_FUNC int mrI_info (mr_State *R, const char *path, int *width,
                       int *height, int *channels);
MRI_FUNC int mrI_probe (const char *path);
MRI_FUNC int mrI_togray (mr_State *R, mr_Image *img);
MRI_FUNC void mrI_free (mr_State *R, mr_Image *img);

#endif
