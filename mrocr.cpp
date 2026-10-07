/*
** mrocr.cpp
** OCR engine interface
** See Copyright Notice in mr.h
*/

#define mrocr_cpp
#define MR_CORE

#include "mrocr.h"
#include "mrstate.h"


#if MR_USE_TESSERACT

#error "MR_USE_TESSERACT: the tesseract wrapper (mrtess.cpp) is not added yet"

#else

/*
** {======================================================
** Stub: built without an OCR engine
** =======================================================
*/

int mrO_available (void) {
  return 0;
}


int mrO_recognize (mr_State *R, const mr_Image *img, mr_Buffer *out) {
  MR_UNUSED(img);
  MR_UNUSED(out);
  return mrS_error(R, MR_ERRNOTSUP,
                   "built without an OCR engine (MR_USE_TESSERACT)");
}

/* }====================================================== */

#endif
