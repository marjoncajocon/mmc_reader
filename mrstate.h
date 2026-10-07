/*
** mrstate.h
** The library state
** See Copyright Notice in mr.h
*/

#ifndef mrstate_h
#define mrstate_h

#include "mrlimits.h"
#include "mrbuf.h"


struct mr_State {
  mr_Alloc frealloc;  /* memory allocator */
  void *ud;  /* user data for 'frealloc' */
  mr_Buffer text;  /* text read so far */
  int dpi;  /* PDF render resolution */
  char lang[MR_LANGSIZE];  /* OCR language, e.g. "eng" */
  char datapath[MR_PATHSIZE];  /* OCR data folder; empty = engine default */
  char errmsg[MR_ERRSIZE];  /* last error message */
};


/*
** Set the error message and return 'status', so a failing function can
** do: return mrS_error(R, MR_ERRFILE, "cannot open '%s'", path);
*/
MRI_FUNC int mrS_error (mr_State *R, int status, const char *fmt, ...)
  MR_PRINTF(3, 4);

/* copy 'src' into 'dst' of 'size' bytes; MR_ERRARG if it does not fit */
MRI_FUNC int mrS_copystr (char *dst, size_t size, const char *src);

#endif
