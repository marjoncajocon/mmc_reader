/*
** mrfile.h
** File access with UTF-8 paths on every system
** See Copyright Notice in mr.h
*/

#ifndef mrfile_h
#define mrfile_h

#include <stdio.h>

#include "mrlimits.h"


/* like fopen, but 'path' is UTF-8 also on Windows */
MRI_FUNC FILE *mrF_open (const char *path, const char *mode);

/*
** Read the first 'n' bytes of a file into 'buf'. Returns how many bytes
** were read, or 0 if the file cannot be opened.
*/
MRI_FUNC size_t mrF_head (const char *path, mr_byte *buf, size_t n);

#endif
