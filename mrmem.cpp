/*
** mrmem.cpp
** Memory allocation
** See Copyright Notice in mr.h
*/

#define mrmem_cpp
#define MR_CORE

#include "mrmem.h"
#include "mrstate.h"

#include <stdlib.h>


/*
** The default allocator: plain realloc/free, like 'l_alloc' in Lua.
*/
void *mrM_defaultalloc (void *ud, void *ptr, size_t osize, size_t nsize) {
  MR_UNUSED(ud);
  MR_UNUSED(osize);
  if (nsize == 0) {
    free(ptr);
    return NULL;
  }
  return realloc(ptr, nsize);
}


void *mrM_realloc (mr_State *R, void *block, size_t osize, size_t nsize) {
  void *newblock = (*R->frealloc)(R->ud, block, osize, nsize);
  if (newblock == NULL && nsize > 0)
    mrS_error(R, MR_ERRMEM, "not enough memory");
  return newblock;
}


#define MINVECTOR  16


void *mrM_grow (mr_State *R, void *block, size_t n, size_t *cap,
                size_t size) {
  size_t newcap;
  void *newblock;
  if (n < *cap) return block;  /* already has room */
  newcap = (*cap == 0) ? MINVECTOR : *cap;
  while (newcap <= n) {
    if (newcap > MR_MAXSIZE / 2 / size) {
      mrS_error(R, MR_ERRMEM, "memory block too large");
      return NULL;
    }
    newcap *= 2;
  }
  newblock = mrM_realloc(R, block, *cap * size, newcap * size);
  if (newblock != NULL)
    *cap = newcap;
  return newblock;
}


/*
** Allocate 'n' items of 'size' bytes, checking for overflow.
*/
void *mrM_array (mr_State *R, size_t n, size_t size) {
  if (size != 0 && n > MR_MAXSIZE / size) {
    mrS_error(R, MR_ERRMEM, "memory block too large");
    return NULL;
  }
  return mrM_malloc(R, n * size);
}
