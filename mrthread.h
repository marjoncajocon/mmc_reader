/*
** mrthread.h
** Run work on several CPU cores (Win32 threads or pthreads)
** See Copyright Notice in mr.h
*/

#ifndef mrthread_h
#define mrthread_h

#include "mrlimits.h"


#define MR_MAXTHREADS  64


/* work item 'i' of a parallel job; 'ctx' is shared by all items */
typedef void (*mr_Work) (void *ctx, int i);


/* number of CPU cores (at least 1) */
MRI_FUNC int mrX_cpus (void);

/*
** Run work(ctx, 0) .. work(ctx, n - 1) at the same time, one thread
** each (n <= MR_MAXTHREADS), and wait for all. Falls back to running
** them one by one if threads cannot be made.
*/
MRI_FUNC void mrX_parallel (int n, mr_Work work, void *ctx);

#endif
