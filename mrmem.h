/*
** mrmem.h
** Memory allocation
** See Copyright Notice in mr.h
*/

#ifndef mrmem_h
#define mrmem_h

#include "mrlimits.h"


/*
** All library memory goes through these, so it uses the allocator given
** to 'mr_newstate'. On failure they return NULL and set the error
** message; the caller returns MR_ERRMEM.
*/
#define mrM_malloc(R, s)  mrM_realloc(R, NULL, 0, (s))
#define mrM_free(R, b, s)  ((void)mrM_realloc(R, (b), (s), 0))

#define mrM_new(R, t)  mr_cast(t *, mrM_malloc(R, sizeof(t)))
#define mrM_delete(R, p)  mrM_free(R, (p), sizeof(*(p)))

#define mrM_newarray(R, n, t)  mr_cast(t *, mrM_array(R, (n), sizeof(t)))
#define mrM_freearray(R, b, n)  mrM_free(R, (b), (n) * sizeof(*(b)))


MRI_FUNC void *mrM_realloc (mr_State *R, void *block, size_t osize,
                            size_t nsize);
MRI_FUNC void *mrM_array (mr_State *R, size_t n, size_t size);
MRI_FUNC void *mrM_defaultalloc (void *ud, void *ptr, size_t osize,
                                 size_t nsize);

#endif
