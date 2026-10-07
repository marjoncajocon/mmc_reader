/*
** mrbuf.h
** Growable byte/string buffer
** See Copyright Notice in mr.h
*/

#ifndef mrbuf_h
#define mrbuf_h

#include "mrlimits.h"


/*
** 'data' is always NUL-terminated once anything has been added, so it
** can be used as a C string. 'len' does not count the NUL.
*/
typedef struct mr_Buffer {
  char *data;
  size_t len;
  size_t cap;
} mr_Buffer;


MRI_FUNC void mrB_init (mr_Buffer *b);
MRI_FUNC int mrB_addlstr (mr_State *R, mr_Buffer *b, const char *s,
                          size_t l);
MRI_FUNC int mrB_addstr (mr_State *R, mr_Buffer *b, const char *s);
MRI_FUNC int mrB_addchar (mr_State *R, mr_Buffer *b, char c);
MRI_FUNC const char *mrB_cstr (const mr_Buffer *b);
MRI_FUNC void mrB_reset (mr_Buffer *b);
MRI_FUNC void mrB_free (mr_State *R, mr_Buffer *b);

#endif
