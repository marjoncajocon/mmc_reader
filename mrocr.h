/*
** mrocr.h
** OCR engine interface
** See Copyright Notice in mr.h
*/

#ifndef mrocr_h
#define mrocr_h

#include "mrlimits.h"
#include "mrbuf.h"
#include "mrimage.h"


MRI_FUNC int mrO_available (void);

/*
** Recognize the text in 'img' and add it (UTF-8) to 'out'. Uses the
** language and data path set in the state.
*/
MRI_FUNC int mrO_recognize (mr_State *R, const mr_Image *img,
                            mr_Buffer *out);

#endif
