/*
** mrpdf.cpp
** PDF loading and page rendering
** See Copyright Notice in mr.h
*/

#define mrpdf_cpp
#define MR_CORE

#include "mrpdf.h"
#include "mrstate.h"


#if MR_USE_PDF

#error "MR_USE_PDF: the PDF library is not added yet (see vendor/README.md)"

#else

/*
** {======================================================
** Stub: built without a PDF library
** =======================================================
*/

int mrP_available (void) {
  return 0;
}


int mrP_open (mr_State *R, const char *path, mr_Pdf **out) {
  MR_UNUSED(path);
  *out = NULL;
  return mrS_error(R, MR_ERRNOTSUP, "built without PDF support (MR_USE_PDF)");
}


int mrP_numpages (const mr_Pdf *pdf) {
  MR_UNUSED(pdf);
  return 0;
}


int mrP_render (mr_State *R, mr_Pdf *pdf, int page, int dpi,
                mr_Image **out) {
  MR_UNUSED(pdf);
  MR_UNUSED(page);
  MR_UNUSED(dpi);
  *out = NULL;
  return mrS_error(R, MR_ERRNOTSUP, "built without PDF support");
}


void mrP_close (mr_State *R, mr_Pdf *pdf) {
  MR_UNUSED(R);
  MR_UNUSED(pdf);
}

/* }====================================================== */

#endif
