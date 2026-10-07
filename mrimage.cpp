/*
** mrimage.cpp
** Image loading and pixel access
** See Copyright Notice in mr.h
*/

#define mrimage_cpp
#define MR_CORE

#include "mrimage.h"
#include "mrfile.h"
#include "mrmem.h"
#include "mrstate.h"

#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_HDR
#define STBI_NO_LINEAR
#include "stb/stb_image.h"


static size_t imagesize (int width, int height, int channels) {
  return mr_cast(size_t, width) * mr_cast(size_t, height) *
         mr_cast(size_t, channels);
}


int mrI_new (mr_State *R, int width, int height, int channels,
             mr_Image **out) {
  mr_Image *img;
  size_t size;
  *out = NULL;
  if (width <= 0 || height <= 0 || channels < 1 || channels > 4)
    return mrS_error(R, MR_ERRARG, "bad image size %dx%dx%d",
                     width, height, channels);
  if (mr_cast(size_t, width) > MR_MAXSIZE / mr_cast(size_t, height) /
                               mr_cast(size_t, channels))
    return mrS_error(R, MR_ERRMEM, "image too large");
  size = imagesize(width, height, channels);
  img = mrM_new(R, mr_Image);
  if (img == NULL) return MR_ERRMEM;
  img->pixels = mr_cast(mr_byte *, mrM_malloc(R, size));
  if (img->pixels == NULL) {
    mrM_delete(R, img);
    return MR_ERRMEM;
  }
  img->width = width;
  img->height = height;
  img->channels = channels;
  *out = img;
  return MR_OK;
}


/*
** stb_image allocates with its own malloc; copy the pixels into memory
** from our allocator so 'mrI_free' works the same for every image.
*/
static int fromstb (mr_State *R, stbi_uc *data, int w, int h, int ch,
                    mr_Image **out) {
  int status;
  if (data == NULL)
    return mrS_error(R, MR_ERRFORMAT, "cannot decode image: %s",
                     stbi_failure_reason());
  status = mrI_new(R, w, h, ch, out);
  if (status == MR_OK)
    memcpy((*out)->pixels, data, imagesize(w, h, ch));
  stbi_image_free(data);
  return status;
}


int mrI_load (mr_State *R, const char *path, mr_Image **out) {
  stbi_uc *data;
  int w, h, ch;
  FILE *f;
  *out = NULL;
  f = mrF_open(path, "rb");
  if (f == NULL)
    return mrS_error(R, MR_ERRFILE, "cannot open file");
  data = stbi_load_from_file(f, &w, &h, &ch, 0);
  fclose(f);
  return fromstb(R, data, w, h, ch, out);
}


int mrI_loadmem (mr_State *R, const mr_byte *data, size_t size,
                 mr_Image **out) {
  stbi_uc *pix;
  int w, h, ch;
  *out = NULL;
  if (size > INT_MAX)
    return mrS_error(R, MR_ERRARG, "image data too large");
  pix = stbi_load_from_memory(data, mr_cast(int, size), &w, &h, &ch, 0);
  return fromstb(R, pix, w, h, ch, out);
}


int mrI_info (mr_State *R, const char *path, int *width, int *height,
              int *channels) {
  int ok;
  FILE *f = mrF_open(path, "rb");
  if (f == NULL)
    return mrS_error(R, MR_ERRFILE, "cannot open file");
  ok = stbi_info_from_file(f, width, height, channels);
  fclose(f);
  if (!ok)
    return mrS_error(R, MR_ERRFORMAT, "not a known image format: %s",
                     stbi_failure_reason());
  return MR_OK;
}


/*
** Return 1 if 'path' is an image stb_image can read, 0 otherwise.
*/
int mrI_probe (const char *path) {
  int w, h, ch, ok;
  FILE *f = mrF_open(path, "rb");
  if (f == NULL) return 0;
  ok = stbi_info_from_file(f, &w, &h, &ch);
  fclose(f);
  return ok;
}


/*
** Convert to 1-channel gray. Transparent pixels are put on a white
** background, which is what OCR expects for text on paper. On failure
** the image is left unchanged.
*/
int mrI_togray (mr_State *R, mr_Image *img) {
  size_t i, n;
  const mr_byte *src;
  mr_byte *dst;
  int ch = img->channels;
  if (ch == 1) return MR_OK;
  n = imagesize(img->width, img->height, 1);
  dst = mr_cast(mr_byte *, mrM_malloc(R, n));
  if (dst == NULL) return MR_ERRMEM;
  src = img->pixels;
  for (i = 0; i < n; i++, src += ch) {
    unsigned v, a;
    if (ch >= 3)
      v = (src[0] * 299u + src[1] * 587u + src[2] * 114u) / 1000u;
    else
      v = src[0];
    if (ch == 2 || ch == 4) {
      a = src[ch - 1];
      v = (v * a + 255u * (255u - a)) / 255u;
    }
    dst[i] = mr_cast(mr_byte, v);
  }
  mrM_free(R, img->pixels, n * mr_cast(size_t, ch));
  img->pixels = dst;
  img->channels = 1;
  return MR_OK;
}


void mrI_free (mr_State *R, mr_Image *img) {
  if (img == NULL) return;
  mrM_free(R, img->pixels, imagesize(img->width, img->height,
                                     img->channels));
  mrM_delete(R, img);
}
