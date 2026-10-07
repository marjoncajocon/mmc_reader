/*
** mrbuf.cpp
** Growable byte/string buffer
** See Copyright Notice in mr.h
*/

#define mrbuf_cpp
#define MR_CORE

#include "mrbuf.h"
#include "mrmem.h"

#include <string.h>


#define MINBUFFER  64


void mrB_init (mr_Buffer *b) {
  b->data = NULL;
  b->len = 0;
  b->cap = 0;
}


/*
** Make room for 'extra' more bytes plus the final NUL.
*/
static int reserve (mr_State *R, mr_Buffer *b, size_t extra) {
  size_t need, newcap;
  char *newdata;
  if (extra > MR_MAXSIZE - b->len - 1)
    return MR_ERRMEM;
  need = b->len + extra + 1;
  if (need <= b->cap)
    return MR_OK;
  newcap = (b->cap == 0) ? MINBUFFER : b->cap;
  while (newcap < need) {
    if (newcap > MR_MAXSIZE / 2) {
      newcap = need;
      break;
    }
    newcap *= 2;
  }
  newdata = mr_cast(char *, mrM_realloc(R, b->data, b->cap, newcap));
  if (newdata == NULL)
    return MR_ERRMEM;
  b->data = newdata;
  b->cap = newcap;
  return MR_OK;
}


int mrB_addlstr (mr_State *R, mr_Buffer *b, const char *s, size_t l) {
  int status = reserve(R, b, l);
  if (status != MR_OK) return status;
  memcpy(b->data + b->len, s, l);
  b->len += l;
  b->data[b->len] = '\0';
  return MR_OK;
}


int mrB_addstr (mr_State *R, mr_Buffer *b, const char *s) {
  return mrB_addlstr(R, b, s, strlen(s));
}


int mrB_addchar (mr_State *R, mr_Buffer *b, char c) {
  return mrB_addlstr(R, b, &c, 1);
}


const char *mrB_cstr (const mr_Buffer *b) {
  return (b->data != NULL) ? b->data : "";
}


/*
** Empty the buffer but keep its memory for reuse.
*/
void mrB_reset (mr_Buffer *b) {
  b->len = 0;
  if (b->data != NULL)
    b->data[0] = '\0';
}


void mrB_free (mr_State *R, mr_Buffer *b) {
  if (b->data != NULL)
    mrM_free(R, b->data, b->cap);
  mrB_init(b);
}
