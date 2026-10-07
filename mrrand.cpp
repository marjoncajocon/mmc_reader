/*
** mrrand.cpp
** Small random number generator (xorshift64*), same results everywhere
** See Copyright Notice in mr.h
*/

#define mrrand_cpp
#define MR_CORE

#include "mrrand.h"

#include <math.h>


void mrR_seed (mr_Rand *r, uint64_t seed) {
  /* splitmix64 step, so small seeds still give a good state */
  uint64_t z = seed + 0x9E3779B97F4A7C15ull;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  r->s = (z != 0) ? z : 1;
}


uint64_t mrR_next (mr_Rand *r) {
  uint64_t x = r->s;
  x ^= x >> 12;
  x ^= x << 25;
  x ^= x >> 27;
  r->s = x;
  return x * 0x2545F4914F6CDD1Dull;
}


int mrR_int (mr_Rand *r, int n) {
  if (n <= 1) return 0;
  return mr_cast(int, (mrR_next(r) >> 33) % mr_cast(uint64_t, n));
}


float mrR_float (mr_Rand *r) {
  return mr_cast(float, mrR_next(r) >> 40) / 16777216.0f;  /* 2^24 */
}


float mrR_range (mr_Rand *r, float lo, float hi) {
  return lo + (hi - lo) * mrR_float(r);
}


float mrR_normal (mr_Rand *r) {
  /* Box-Muller */
  float u = mrR_float(r);
  float v = mrR_float(r);
  if (u < 1e-7f) u = 1e-7f;
  return sqrtf(-2.0f * logf(u)) * cosf(6.2831853f * v);
}
