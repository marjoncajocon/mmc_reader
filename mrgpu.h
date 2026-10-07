/*
** mrgpu.h
** GPU access through OpenCL, loaded at run time (only in the GPU build:
** ./build.sh gpu, MR_USE_GPU=1). Other builds get stubs that say
** MR_ERRNOTSUP, so callers never need #if.
** See Copyright Notice in mr.h
*/

#ifndef mrgpu_h
#define mrgpu_h

#include "mrlimits.h"


/* an open GPU; its fields belong to mrgpu.cpp */
typedef struct mr_Gpu mr_Gpu;


/* 1 if this program was built with GPU support */
MRI_FUNC int mrU_built (void);

/*
** Open the first GPU (or any OpenCL device if there is no GPU). Fails
** with MR_ERRNOTSUP if OpenCL or a device is missing; then use the CPU.
*/
MRI_FUNC int mrU_open (mr_State *R, mr_Gpu **out);
MRI_FUNC void mrU_close (mr_State *R, mr_Gpu *g);

/* device name, compute units and memory, e.g. for a startup message */
MRI_FUNC const char *mrU_name (const mr_Gpu *g);
MRI_FUNC int mrU_units (const mr_Gpu *g);
MRI_FUNC double mrU_memory (const mr_Gpu *g);  /* in GB */

/* build and run a small kernel and check its result */
MRI_FUNC int mrU_selftest (mr_State *R, mr_Gpu *g);

#endif
