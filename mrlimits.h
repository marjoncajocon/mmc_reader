/*
** mrlimits.h
** Internal basic types, limits and helper macros
** See Copyright Notice in mr.h
*/

#ifndef mrlimits_h
#define mrlimits_h

#include <stddef.h>
#include <limits.h>

#include "mr.h"


typedef unsigned char mr_byte;

/* largest size of an object */
#define MR_MAXSIZE  (~mr_cast(size_t, 0))

/* longest file path, in bytes (UTF-8) */
#define MR_PATHSIZE  1024

/* longest OCR language name, e.g. "eng+fil" */
#define MR_LANGSIZE  64

/* default PDF render resolution */
#define MR_DEFDPI  300

#if defined(MR_DEBUG)
#include <assert.h>
#define mr_assert(c)  assert(c)
#else
#define mr_assert(c)  ((void)0)
#endif

#endif
