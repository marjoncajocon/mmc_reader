/*
** mrocr.h
** OCR: gray image -> text, with our own trained network
** See Copyright Notice in mr.h
*/

#ifndef mrocr_h
#define mrocr_h

#include "mrlimits.h"
#include "mrbuf.h"
#include "mrimage.h"
#include "mrnet.h"


/*
** Recognize the text in gray image 'img' and add it (UTF-8, one line per
** text line) to 'out'. Loads the model '<datapath>/<lang>.mrm' the
** first time.
*/
MRI_FUNC int mrO_recognize (mr_State *R, const mr_Image *img,
                            mr_Buffer *out);

/* the same with a given network (used by the trainer) */
MRI_FUNC int mrO_run (mr_State *R, const mr_Net *net, const mr_Image *img,
                      mr_Buffer *out);

/* load the model for the state's language now */
MRI_FUNC int mrO_loadmodel (mr_State *R);

#endif
