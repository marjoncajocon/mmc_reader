/*
** mrgpu.cpp
** GPU access through OpenCL, loaded at run time
** See Copyright Notice in mr.h
**
** OpenCL is a C API, and its kernels are compiled by the graphics driver
** when the program runs, so no SDK, header or compiler is needed: the few
** types and functions used are declared below and looked up in
** OpenCL.dll (Windows) or libOpenCL.so (Linux) with LoadLibrary/dlopen.
** Without a GPU driver the program still runs, on the CPU.
**
** Status: device setup and a self-test only. The network kernels
** (convolution, LSTM, CTC) are still to be written; see CONTRIBUTING.md.
*/

#define mrgpu_cpp
#define MR_CORE

#include "mrgpu.h"
#include "mrmem.h"
#include "mrstate.h"

#include <math.h>
#include <string.h>


#if !MR_USE_GPU

/*
** {======================================================
** Stub: built without GPU support
** =======================================================
*/

int mrU_built (void) {
  return 0;
}


int mrU_open (mr_State *R, mr_Gpu **out) {
  *out = NULL;
  return mrS_error(R, MR_ERRNOTSUP,
                   "built without GPU support (./build.sh gpu)");
}


void mrU_close (mr_State *R, mr_Gpu *g) {
  MR_UNUSED(R);
  MR_UNUSED(g);
}


const char *mrU_name (const mr_Gpu *g) {
  MR_UNUSED(g);
  return "";
}


int mrU_units (const mr_Gpu *g) {
  MR_UNUSED(g);
  return 0;
}


double mrU_memory (const mr_Gpu *g) {
  MR_UNUSED(g);
  return 0;
}


int mrU_selftest (mr_State *R, mr_Gpu *g) {
  MR_UNUSED(g);
  return mrS_error(R, MR_ERRNOTSUP, "built without GPU support");
}

/* }====================================================== */

#else

#include <stdint.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#define CLCALL  __stdcall
#else
#include <dlfcn.h>
#define CLCALL
#endif


/*
** {======================================================
** The part of the OpenCL 1.2 API we use (the ABI is stable)
** =======================================================
*/

typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef cl_ulong cl_bitfield;
typedef struct ClPlatform *cl_platform_id;
typedef struct ClDevice *cl_device_id;
typedef struct ClContext *cl_context;
typedef struct ClQueue *cl_command_queue;
typedef struct ClMem *cl_mem;
typedef struct ClProgram *cl_program;
typedef struct ClKernel *cl_kernel;
typedef struct ClEvent *cl_event;

#define CL_SUCCESS  0
#define CL_TRUE  1
#define CL_DEVICE_TYPE_GPU  (mr_cast(cl_bitfield, 1) << 2)
#define CL_DEVICE_TYPE_ALL  mr_cast(cl_bitfield, 0xFFFFFFFF)
#define CL_DEVICE_MAX_COMPUTE_UNITS  0x1002
#define CL_DEVICE_GLOBAL_MEM_SIZE  0x101F
#define CL_DEVICE_NAME  0x102B
#define CL_MEM_READ_WRITE  (mr_cast(cl_bitfield, 1) << 0)
#define CL_MEM_COPY_HOST_PTR  (mr_cast(cl_bitfield, 1) << 5)
#define CL_PROGRAM_BUILD_LOG  0x1183

typedef cl_int (CLCALL *GetPlatformIDs) (cl_uint, cl_platform_id *,
                                         cl_uint *);
typedef cl_int (CLCALL *GetDeviceIDs) (cl_platform_id, cl_bitfield, cl_uint,
                                       cl_device_id *, cl_uint *);
typedef cl_int (CLCALL *GetDeviceInfo) (cl_device_id, cl_uint, size_t,
                                        void *, size_t *);
typedef cl_context (CLCALL *CreateContext) (const intptr_t *, cl_uint,
                                            const cl_device_id *, void *,
                                            void *, cl_int *);
typedef cl_command_queue (CLCALL *CreateCommandQueue) (cl_context,
                                                       cl_device_id,
                                                       cl_bitfield,
                                                       cl_int *);
typedef cl_program (CLCALL *CreateProgramWithSource) (cl_context, cl_uint,
                                                      const char **,
                                                      const size_t *,
                                                      cl_int *);
typedef cl_int (CLCALL *BuildProgram) (cl_program, cl_uint,
                                       const cl_device_id *, const char *,
                                       void *, void *);
typedef cl_int (CLCALL *GetProgramBuildInfo) (cl_program, cl_device_id,
                                              cl_uint, size_t, void *,
                                              size_t *);
typedef cl_kernel (CLCALL *CreateKernel) (cl_program, const char *,
                                          cl_int *);
typedef cl_mem (CLCALL *CreateBuffer) (cl_context, cl_bitfield, size_t,
                                       void *, cl_int *);
typedef cl_int (CLCALL *SetKernelArg) (cl_kernel, cl_uint, size_t,
                                       const void *);
typedef cl_int (CLCALL *EnqueueNDRangeKernel) (cl_command_queue, cl_kernel,
                                               cl_uint, const size_t *,
                                               const size_t *,
                                               const size_t *, cl_uint,
                                               const cl_event *,
                                               cl_event *);
typedef cl_int (CLCALL *EnqueueReadBuffer) (cl_command_queue, cl_mem,
                                            cl_uint, size_t, size_t, void *,
                                            cl_uint, const cl_event *,
                                            cl_event *);
typedef cl_int (CLCALL *Finish) (cl_command_queue);
typedef cl_int (CLCALL *Release) (void *);


/* the functions, looked up by name */
typedef struct Cl {
  GetPlatformIDs getplatformids;
  GetDeviceIDs getdeviceids;
  GetDeviceInfo getdeviceinfo;
  CreateContext createcontext;
  CreateCommandQueue createcommandqueue;
  CreateProgramWithSource createprogramwithsource;
  BuildProgram buildprogram;
  GetProgramBuildInfo getprogrambuildinfo;
  CreateKernel createkernel;
  CreateBuffer createbuffer;
  SetKernelArg setkernelarg;
  EnqueueNDRangeKernel enqueuendrangekernel;
  EnqueueReadBuffer enqueuereadbuffer;
  Finish finish;
  Release releasekernel, releaseprogram, releasememobject;
  Release releasecommandqueue, releasecontext;
} Cl;


struct mr_Gpu {
  void *lib;  /* the OpenCL library */
  Cl cl;
  cl_device_id device;
  cl_context context;
  cl_command_queue queue;
  char name[128];
  int units;
  double memory;
};

/* }====================================================== */


/*
** {======================================================
** Loading OpenCL
** =======================================================
*/

static void *openlib (void) {
#if defined(_WIN32)
  return mr_cast(void *, LoadLibraryA("OpenCL.dll"));
#else
  void *h = dlopen("libOpenCL.so.1", RTLD_NOW);
  return (h != NULL) ? h : dlopen("libOpenCL.so", RTLD_NOW);
#endif
}


static void closelib (void *lib) {
#if defined(_WIN32)
  FreeLibrary(mr_cast(HMODULE, lib));
#else
  dlclose(lib);
#endif
}


/* look up 'name'; a function pointer comes back through 'fp' */
static int sym (void *lib, const char *name, void *fp) {
  void *p;
#if defined(_WIN32)
  FARPROC f = GetProcAddress(mr_cast(HMODULE, lib), name);
  memcpy(&p, &f, sizeof(p));
#else
  p = dlsym(lib, name);
#endif
  memcpy(fp, &p, sizeof(p));
  return p != NULL;
}


static int loadcl (void *lib, Cl *cl) {
  return sym(lib, "clGetPlatformIDs", &cl->getplatformids) &&
         sym(lib, "clGetDeviceIDs", &cl->getdeviceids) &&
         sym(lib, "clGetDeviceInfo", &cl->getdeviceinfo) &&
         sym(lib, "clCreateContext", &cl->createcontext) &&
         sym(lib, "clCreateCommandQueue", &cl->createcommandqueue) &&
         sym(lib, "clCreateProgramWithSource",
             &cl->createprogramwithsource) &&
         sym(lib, "clBuildProgram", &cl->buildprogram) &&
         sym(lib, "clGetProgramBuildInfo", &cl->getprogrambuildinfo) &&
         sym(lib, "clCreateKernel", &cl->createkernel) &&
         sym(lib, "clCreateBuffer", &cl->createbuffer) &&
         sym(lib, "clSetKernelArg", &cl->setkernelarg) &&
         sym(lib, "clEnqueueNDRangeKernel", &cl->enqueuendrangekernel) &&
         sym(lib, "clEnqueueReadBuffer", &cl->enqueuereadbuffer) &&
         sym(lib, "clFinish", &cl->finish) &&
         sym(lib, "clReleaseKernel", &cl->releasekernel) &&
         sym(lib, "clReleaseProgram", &cl->releaseprogram) &&
         sym(lib, "clReleaseMemObject", &cl->releasememobject) &&
         sym(lib, "clReleaseCommandQueue", &cl->releasecommandqueue) &&
         sym(lib, "clReleaseContext", &cl->releasecontext);
}


/* the first GPU of any platform, else the first device of any kind */
static int finddevice (const Cl *cl, cl_device_id *dev) {
  cl_platform_id plats[16];
  cl_uint np = 0, i, n;
  int pass;
  if (cl->getplatformids(16, plats, &np) != CL_SUCCESS || np == 0)
    return 0;
  if (np > 16) np = 16;
  for (pass = 0; pass < 2; pass++) {
    cl_bitfield type = (pass == 0) ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_ALL;
    for (i = 0; i < np; i++)
      if (cl->getdeviceids(plats[i], type, 1, dev, &n) == CL_SUCCESS &&
          n > 0)
        return 1;
  }
  return 0;
}

/* }====================================================== */


int mrU_built (void) {
  return 1;
}


int mrU_open (mr_State *R, mr_Gpu **out) {
  mr_Gpu *g;
  cl_int err;
  cl_uint units = 0;
  cl_ulong mem = 0;
  *out = NULL;
  g = mrM_new(R, mr_Gpu);
  if (g == NULL) return MR_ERRMEM;
  memset(g, 0, sizeof(*g));
  g->lib = openlib();
  if (g->lib == NULL || !loadcl(g->lib, &g->cl)) {
    mrU_close(R, g);
    return mrS_error(R, MR_ERRNOTSUP, "no OpenCL driver found");
  }
  if (!finddevice(&g->cl, &g->device)) {
    mrU_close(R, g);
    return mrS_error(R, MR_ERRNOTSUP, "no OpenCL device found");
  }
  g->cl.getdeviceinfo(g->device, CL_DEVICE_NAME, sizeof(g->name) - 1,
                      g->name, NULL);
  g->cl.getdeviceinfo(g->device, CL_DEVICE_MAX_COMPUTE_UNITS,
                      sizeof(units), &units, NULL);
  g->cl.getdeviceinfo(g->device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem),
                      &mem, NULL);
  g->units = mr_cast(int, units);
  g->memory = mr_cast(double, mem) / (1024.0 * 1024.0 * 1024.0);
  g->context = g->cl.createcontext(NULL, 1, &g->device, NULL, NULL, &err);
  if (g->context != NULL)
    g->queue = g->cl.createcommandqueue(g->context, g->device, 0, &err);
  if (g->context == NULL || g->queue == NULL) {
    mrU_close(R, g);
    return mrS_error(R, MR_ERRNOTSUP, "cannot start the GPU (error %d)",
                     mr_cast(int, err));
  }
  *out = g;
  return MR_OK;
}


void mrU_close (mr_State *R, mr_Gpu *g) {
  if (g == NULL) return;
  if (g->queue != NULL) g->cl.releasecommandqueue(g->queue);
  if (g->context != NULL) g->cl.releasecontext(g->context);
  if (g->lib != NULL) closelib(g->lib);
  mrM_delete(R, g);
}


const char *mrU_name (const mr_Gpu *g) {
  return g->name;
}


int mrU_units (const mr_Gpu *g) {
  return g->units;
}


double mrU_memory (const mr_Gpu *g) {
  return g->memory;
}


/*
** {======================================================
** Self-test: y = a * x + y on the GPU, checked on the CPU
** =======================================================
*/

#define TESTN  4096

static const char *testsrc =
  "__kernel void axpy (float a, __global const float *x,\n"
  "                    __global float *y) {\n"
  "  int i = get_global_id(0);\n"
  "  y[i] = a * x[i] + y[i];\n"
  "}\n";


int mrU_selftest (mr_State *R, mr_Gpu *g) {
  const Cl *cl = &g->cl;
  float *x = NULL, *y = NULL, a = 2.5f;
  cl_program prog = NULL;
  cl_kernel kern = NULL;
  cl_mem bx = NULL, by = NULL;
  size_t n = TESTN, i;
  cl_int err;
  int status = MR_OK;
  x = mrM_newarray(R, TESTN, float);
  y = mrM_newarray(R, TESTN, float);
  if (x == NULL || y == NULL) {
    status = MR_ERRMEM;
    goto done;
  }
  for (i = 0; i < n; i++) {
    x[i] = mr_cast(float, i);
    y[i] = 1.0f;
  }
  prog = cl->createprogramwithsource(g->context, 1, &testsrc, NULL, &err);
  if (prog == NULL || cl->buildprogram(prog, 1, &g->device, "", NULL,
                                       NULL) != CL_SUCCESS) {
    status = mrS_error(R, MR_ERROCR, "GPU: cannot build the test kernel");
    goto done;
  }
  kern = cl->createkernel(prog, "axpy", &err);
  bx = cl->createbuffer(g->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        n * sizeof(float), x, &err);
  by = cl->createbuffer(g->context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                        n * sizeof(float), y, &err);
  if (kern == NULL || bx == NULL || by == NULL ||
      cl->setkernelarg(kern, 0, sizeof(float), &a) != CL_SUCCESS ||
      cl->setkernelarg(kern, 1, sizeof(cl_mem), &bx) != CL_SUCCESS ||
      cl->setkernelarg(kern, 2, sizeof(cl_mem), &by) != CL_SUCCESS ||
      cl->enqueuendrangekernel(g->queue, kern, 1, NULL, &n, NULL, 0, NULL,
                               NULL) != CL_SUCCESS ||
      cl->enqueuereadbuffer(g->queue, by, CL_TRUE, 0, n * sizeof(float), y,
                            0, NULL, NULL) != CL_SUCCESS) {
    status = mrS_error(R, MR_ERROCR, "GPU: the test kernel did not run");
    goto done;
  }
  cl->finish(g->queue);
  for (i = 0; i < n; i++) {
    if (fabsf(y[i] - (a * x[i] + 1.0f)) > 1e-3f) {
      status = mrS_error(R, MR_ERROCR, "GPU: wrong result at %d",
                         mr_cast(int, i));
      break;
    }
  }
 done:
  if (bx != NULL) cl->releasememobject(bx);
  if (by != NULL) cl->releasememobject(by);
  if (kern != NULL) cl->releasekernel(kern);
  if (prog != NULL) cl->releaseprogram(prog);
  if (x != NULL) mrM_freearray(R, x, TESTN);
  if (y != NULL) mrM_freearray(R, y, TESTN);
  return status;
}

/* }====================================================== */

#endif
