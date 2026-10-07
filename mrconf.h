/*
** mrconf.h
** Configuration for mmc_reader
** See Copyright Notice in mr.h
*/

#ifndef mrconf_h
#define mrconf_h

#include <stddef.h>


/*
** {======================================================
** Optional features
** Turn a feature on with -DMR_USE_<NAME>=1 (CFLAGS in build.sh).
** A module whose feature is off compiles to a stub that returns
** MR_ERRNOTSUP.
** =======================================================
*/

/* OCR engine (tesseract, through mrtess.cpp) */
#if !defined(MR_USE_TESSERACT)
#define MR_USE_TESSERACT  0
#endif

/* PDF reader */
#if !defined(MR_USE_PDF)
#define MR_USE_PDF  0
#endif

/* }====================================================== */


/*
** {======================================================
** Linkage
** =======================================================
*/

/* 'mr.h' is also usable from C, so its declarations have C linkage */
#if defined(__cplusplus)
#define MR_BEGIN_DECLS  extern "C" {
#define MR_END_DECLS  }
#else
#define MR_BEGIN_DECLS
#define MR_END_DECLS
#endif

/* functions exported by the library */
#define MR_API  extern

/* functions shared between modules but not exported */
#if defined(__GNUC__) && !defined(_WIN32)
#define MRI_FUNC  __attribute__((visibility("hidden"))) extern
#else
#define MRI_FUNC  extern
#endif

/* }====================================================== */


/*
** {======================================================
** Language helpers
** =======================================================
*/

#define mr_cast(t, exp)  ((t)(exp))

#define MR_UNUSED(x)  ((void)(x))

#if defined(__cplusplus)
#define mr_static_assert(c, msg)  static_assert(c, msg)
#define MR_NORETURN  [[noreturn]]
#else
#define mr_static_assert(c, msg)  _Static_assert(c, msg)
#define MR_NORETURN  _Noreturn
#endif

#if defined(__GNUC__)
#define MR_PRINTF(f, a)  __attribute__((format(printf, f, a)))
#else
#define MR_PRINTF(f, a)
#endif

/* }====================================================== */


/* size of the buffer that holds the last error message */
#define MR_ERRSIZE  256

#endif
