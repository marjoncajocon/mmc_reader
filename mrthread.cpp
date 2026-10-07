/*
** mrthread.cpp
** Run work on several CPU cores (Win32 threads or pthreads)
** See Copyright Notice in mr.h
*/

#define mrthread_cpp
#define MR_CORE

#include "mrthread.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#include <unistd.h>
#endif


typedef struct Job {
  mr_Work work;
  void *ctx;
  int i;
} Job;


#if defined(_WIN32)

int mrX_cpus (void) {
  SYSTEM_INFO si;
  GetSystemInfo(&si);
  return (si.dwNumberOfProcessors > 0) ? mr_cast(int, si.dwNumberOfProcessors)
                                       : 1;
}


static DWORD WINAPI runjob (LPVOID p) {
  Job *j = mr_cast(Job *, p);
  j->work(j->ctx, j->i);
  return 0;
}


void mrX_parallel (int n, mr_Work work, void *ctx) {
  Job jobs[MR_MAXTHREADS];
  HANDLE th[MR_MAXTHREADS];
  int i, started = 0;
  if (n > MR_MAXTHREADS) n = MR_MAXTHREADS;
  for (i = 1; i < n; i++) {  /* item 0 runs on this thread */
    jobs[i].work = work;
    jobs[i].ctx = ctx;
    jobs[i].i = i;
    th[i] = CreateThread(NULL, 0, runjob, &jobs[i], 0, NULL);
    if (th[i] == NULL) break;
    started = i;
  }
  work(ctx, 0);
  for (i = started + 1; i < n; i++) work(ctx, i);  /* not started */
  for (i = 1; i <= started; i++) {
    WaitForSingleObject(th[i], INFINITE);
    CloseHandle(th[i]);
  }
}

#else

int mrX_cpus (void) {
  long n = sysconf(_SC_NPROCESSORS_ONLN);
  return (n > 0) ? mr_cast(int, n) : 1;
}


static void *runjob (void *p) {
  Job *j = mr_cast(Job *, p);
  j->work(j->ctx, j->i);
  return NULL;
}


void mrX_parallel (int n, mr_Work work, void *ctx) {
  Job jobs[MR_MAXTHREADS];
  pthread_t th[MR_MAXTHREADS];
  int i, started = 0;
  if (n > MR_MAXTHREADS) n = MR_MAXTHREADS;
  for (i = 1; i < n; i++) {
    jobs[i].work = work;
    jobs[i].ctx = ctx;
    jobs[i].i = i;
    if (pthread_create(&th[i], NULL, runjob, &jobs[i]) != 0) break;
    started = i;
  }
  work(ctx, 0);
  for (i = started + 1; i < n; i++) work(ctx, i);
  for (i = 1; i <= started; i++) pthread_join(th[i], NULL);
}

#endif
