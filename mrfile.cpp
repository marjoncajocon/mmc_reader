/*
** mrfile.cpp
** File access with UTF-8 paths on every system
** See Copyright Notice in mr.h
*/

#define mrfile_cpp
#define MR_CORE

#include "mrfile.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif


#if defined(_WIN32)

/*
** Windows 'fopen' takes paths in the ANSI code page, so names with
** non-ASCII letters would fail. Convert to UTF-16 and use '_wfopen'.
*/
FILE *mrF_open (const char *path, const char *mode) {
  wchar_t wpath[MR_PATHSIZE];
  wchar_t wmode[16];
  if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, MR_PATHSIZE) == 0)
    return NULL;
  if (MultiByteToWideChar(CP_UTF8, 0, mode, -1, wmode, 16) == 0)
    return NULL;
  return _wfopen(wpath, wmode);
}

#else

FILE *mrF_open (const char *path, const char *mode) {
  return fopen(path, mode);
}

#endif


size_t mrF_head (const char *path, mr_byte *buf, size_t n) {
  size_t got;
  FILE *f = mrF_open(path, "rb");
  if (f == NULL) return 0;
  got = fread(buf, 1, n, f);
  fclose(f);
  return got;
}
