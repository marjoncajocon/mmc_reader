/*
** mrrand.h
** Small random number generator (xorshift64*), same results everywhere
** See Copyright Notice in mr.h
*/

#ifndef mrrand_h
#define mrrand_h

#include <stdint.h>

#include "mrlimits.h"


typedef struct mr_Rand {
  uint64_t s;
} mr_Rand;


MRI_FUNC void mrR_seed (mr_Rand *r, uint64_t seed);
MRI_FUNC uint64_t mrR_next (mr_Rand *r);
MRI_FUNC int mrR_int (mr_Rand *r, int n);  /* 0 .. n-1 */
MRI_FUNC float mrR_float (mr_Rand *r);  /* [0, 1) */
MRI_FUNC float mrR_range (mr_Rand *r, float lo, float hi);  /* [lo, hi) */
MRI_FUNC float mrR_normal (mr_Rand *r);  /* mean 0, deviation 1 */

#endif
