/*
** mrapi.cpp
** Implementation of the public API in mr.h
** See Copyright Notice in mr.h
*/

#define mrapi_cpp
#define MR_CORE

#include "mrstate.h"
#include "mrbuf.h"
#include "mrfile.h"
#include "mrimage.h"
#include "mrocr.h"
#include "mrpdf.h"

#include <string.h>


#define MINDPI  36
#define MAXDPI  1200

/* separator between PDF pages in the text, like pdftotext */
#define PAGESEP  '\f'


/*
** {======================================================
** Settings
** =======================================================
*/

/* the model depends on the language and data path: load it again */
static void dropmodel (mr_State *R) {
  mrN_free(R, R->net);
  R->net = NULL;
  mrQ_freework(R, R->seqwork);
  R->seqwork = NULL;
  mrQ_free(R, R->seq);
  R->seq = NULL;
}


int mr_setlang (mr_State *R, const char *lang) {
  if (lang == NULL || lang[0] == '\0' ||
      mrS_copystr(R->lang, sizeof(R->lang), lang) != MR_OK)
    return mrS_error(R, MR_ERRARG, "bad OCR language");
  dropmodel(R);
  return MR_OK;
}


int mr_setdatapath (mr_State *R, const char *path) {
  if (path == NULL) path = "";
  if (mrS_copystr(R->datapath, sizeof(R->datapath), path) != MR_OK)
    return mrS_error(R, MR_ERRARG, "OCR data path too long");
  dropmodel(R);
  return MR_OK;
}


int mr_setdpi (mr_State *R, int dpi) {
  if (dpi < MINDPI || dpi > MAXDPI)
    return mrS_error(R, MR_ERRARG, "dpi must be %d to %d", MINDPI, MAXDPI);
  R->dpi = dpi;
  return MR_OK;
}

/* }====================================================== */


/*
** {======================================================
** Reading
** =======================================================
*/

int mr_filekind (const char *path) {
  mr_byte head[5];
  size_t n = mrF_head(path, head, sizeof(head));
  if (n == sizeof(head) && memcmp(head, "%PDF-", 5) == 0)
    return MR_KINDPDF;
  if (n > 0 && mrI_probe(path))
    return MR_KINDIMAGE;
  return MR_KINDUNKNOWN;
}


/*
** OCR one image into the text buffer. Takes ownership of 'img'.
*/
static int readimg (mr_State *R, mr_Image *img) {
  int status = mrI_togray(R, img);
  if (status == MR_OK)
    status = mrO_recognize(R, img, &R->text);
  mrI_free(R, img);
  return status;
}


int mr_readimage (mr_State *R, const char *path) {
  mr_Image *img = NULL;
  int status = mrI_load(R, path, &img);
  if (status != MR_OK) return status;
  return readimg(R, img);
}


int mr_readpdf (mr_State *R, const char *path) {
  mr_Pdf *pdf = NULL;
  mr_Image *page = NULL;
  int i, n, status;
  status = mrP_open(R, path, &pdf);
  if (status != MR_OK) goto done;
  n = mrP_numpages(pdf);
  for (i = 0; i < n; i++) {
    if (i > 0) {
      status = mrB_addchar(R, &R->text, PAGESEP);
      if (status != MR_OK) goto done;
    }
    status = mrP_render(R, pdf, i, R->dpi, &page);
    if (status != MR_OK) goto done;
    status = readimg(R, page);  /* frees 'page' */
    page = NULL;
    if (status != MR_OK) goto done;
  }
 done:
  mrI_free(R, page);
  mrP_close(R, pdf);
  return status;
}


int mr_readfile (mr_State *R, const char *path) {
  switch (mr_filekind(path)) {
    case MR_KINDPDF:
      return mr_readpdf(R, path);
    case MR_KINDIMAGE:
      return mr_readimage(R, path);
    default: {
      mr_byte c;
      if (mrF_head(path, &c, 1) == 0)  /* missing, unreadable or empty */
        return mrS_error(R, MR_ERRFILE, "cannot read file");
      return mrS_error(R, MR_ERRFORMAT,
                       "not a known image or PDF file");
    }
  }
}


int mr_imageinfo (mr_State *R, const char *path, int *width, int *height,
                  int *channels) {
  return mrI_info(R, path, width, height, channels);
}

/* }====================================================== */


/*
** {======================================================
** Results
** =======================================================
*/

const char *mr_text (mr_State *R, size_t *len) {
  if (len != NULL) *len = R->text.len;
  return mrB_cstr(&R->text);
}


void mr_cleartext (mr_State *R) {
  mrB_reset(&R->text);
}


const char *mr_geterror (mr_State *R) {
  return R->errmsg;
}


const char *mr_statusname (int status) {
  switch (status) {
    case MR_OK: return "ok";
    case MR_ERRMEM: return "out of memory";
    case MR_ERRFILE: return "file error";
    case MR_ERRFORMAT: return "bad format";
    case MR_ERRNOTSUP: return "not supported";
    case MR_ERROCR: return "OCR error";
    case MR_ERRARG: return "bad argument";
    default: return "unknown error";
  }
}


const char *mr_version (void) {
  return MR_VERSION;
}

/* }====================================================== */
