/*
** mr.h
** mmc_reader - read the text in images and PDF files
** This is the public API. It is usable from C and C++.
*/

#ifndef mr_h
#define mr_h

#include <stddef.h>

#include "mrconf.h"


#define MR_VERSION_MAJOR  "0"
#define MR_VERSION_MINOR  "1"
#define MR_VERSION_PATCH  "0"
#define MR_VERSION  "mmc_reader " MR_VERSION_MAJOR "." MR_VERSION_MINOR \
                    "." MR_VERSION_PATCH


/* status codes */
#define MR_OK  0
#define MR_ERRMEM  1      /* out of memory */
#define MR_ERRFILE  2     /* cannot open or read a file */
#define MR_ERRFORMAT  3   /* unknown or broken file format */
#define MR_ERRNOTSUP  4   /* feature not built in (see mrconf.h) */
#define MR_ERROCR  5      /* OCR engine failed */
#define MR_ERRARG  6      /* bad argument */


/* kind of input file, as found by mr_filekind */
#define MR_KINDUNKNOWN  0
#define MR_KINDIMAGE  1
#define MR_KINDPDF  2


MR_BEGIN_DECLS

typedef struct mr_State mr_State;

/*
** Memory allocator, same contract as Lua's 'lua_Alloc':
** nsize == 0 frees 'ptr' and returns NULL; otherwise works like
** realloc. 'osize' is the old size of the block (0 for a new block).
*/
typedef void *(*mr_Alloc) (void *ud, void *ptr, size_t osize, size_t nsize);


/* state */
MR_API mr_State *mr_newstate (mr_Alloc f, void *ud);
MR_API mr_State *mr_open (void);
MR_API void mr_close (mr_State *R);

/* settings */
MR_API int mr_setlang (mr_State *R, const char *lang);
MR_API int mr_setdatapath (mr_State *R, const char *path);
MR_API int mr_setdpi (mr_State *R, int dpi);

/* reading; the text is added to the state's text buffer */
MR_API int mr_filekind (const char *path);
MR_API int mr_readfile (mr_State *R, const char *path);
MR_API int mr_readimage (mr_State *R, const char *path);
MR_API int mr_readpdf (mr_State *R, const char *path);

/* image information without OCR */
MR_API int mr_imageinfo (mr_State *R, const char *path,
                         int *width, int *height, int *channels);

/* results */
MR_API const char *mr_text (mr_State *R, size_t *len);
MR_API void mr_cleartext (mr_State *R);
MR_API const char *mr_geterror (mr_State *R);
MR_API const char *mr_statusname (int status);
MR_API const char *mr_version (void);

MR_END_DECLS


/******************************************************************************
* Copyright (C) 2026 Marjon Mangindo Cajocon.
* License: not chosen yet.
******************************************************************************/

#endif
