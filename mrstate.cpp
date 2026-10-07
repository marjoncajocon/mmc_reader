/*
** mrstate.cpp
** The library state
** See Copyright Notice in mr.h
*/

#define mrstate_cpp
#define MR_CORE

#include "mrstate.h"
#include "mrmem.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>


mr_State *mr_newstate (mr_Alloc f, void *ud) {
  mr_State *R = mr_cast(mr_State *, (*f)(ud, NULL, 0, sizeof(mr_State)));
  if (R == NULL) return NULL;
  R->frealloc = f;
  R->ud = ud;
  mrB_init(&R->text);
  R->net = NULL;
  R->seq = NULL;
  R->seqwork = NULL;
  R->dpi = MR_DEFDPI;
  mrS_copystr(R->lang, sizeof(R->lang), "eng");
  R->datapath[0] = '\0';
  R->errmsg[0] = '\0';
  return R;
}


mr_State *mr_open (void) {
  return mr_newstate(mrM_defaultalloc, NULL);
}


void mr_close (mr_State *R) {
  mr_Alloc f;
  void *ud;
  if (R == NULL) return;
  mrB_free(R, &R->text);
  mrN_free(R, R->net);
  mrQ_freework(R, R->seqwork);
  mrQ_free(R, R->seq);
  f = R->frealloc;
  ud = R->ud;
  (*f)(ud, R, sizeof(mr_State), 0);
}


int mrS_error (mr_State *R, int status, const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(R->errmsg, sizeof(R->errmsg), fmt, ap);
  va_end(ap);
  return status;
}


int mrS_copystr (char *dst, size_t size, const char *src) {
  size_t l = strlen(src);
  if (l >= size) return MR_ERRARG;
  memcpy(dst, src, l + 1);
  return MR_OK;
}
