/*
** mrpdf.h
** PDF loading and page rendering
** See Copyright Notice in mr.h
*/

#ifndef mrpdf_h
#define mrpdf_h

#include "mrlimits.h"
#include "mrimage.h"


/* an open PDF document; its fields belong to mrpdf.cpp */
typedef struct mr_Pdf mr_Pdf;


MRI_FUNC int mrP_available (void);
MRI_FUNC int mrP_open (mr_State *R, const char *path, mr_Pdf **out);
MRI_FUNC int mrP_numpages (const mr_Pdf *pdf);
MRI_FUNC int mrP_render (mr_State *R, mr_Pdf *pdf, int page, int dpi,
                         mr_Image **out);
MRI_FUNC void mrP_close (mr_State *R, mr_Pdf *pdf);

#endif
