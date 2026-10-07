/*
** mmc_reader.cpp
** mmc_reader stand-alone program
** See Copyright Notice in mr.h
*/

#define mmc_reader_cpp

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <shellapi.h>
#endif

#include "mr.h"


#define EXIT_FAILED  1  /* some file could not be read */
#define EXIT_USAGE  2  /* bad command line */


static const char *progname = "mmc_reader";


typedef struct Options {
  const char *output;  /* -o: output file, NULL = stdout */
  const char *lang;  /* -l: OCR language */
  const char *datapath;  /* -d: OCR data folder */
  int dpi;  /* -r: PDF render resolution, 0 = default */
  int info;  /* -i: print file information only */
  int first;  /* index of the first file in argv */
} Options;


static void print_usage (const char *badoption) {
  if (badoption != NULL)
    fprintf(stderr, "%s: bad option '%s'\n", progname, badoption);
  fprintf(stderr,
    "usage: %s [options] file...\n"
    "Read the text in image and PDF files.\n"
    "Available options are:\n"
    "  -o file  write the text to 'file' instead of stdout\n"
    "  -l lang  OCR language, e.g. eng, fil, eng+fil (default eng)\n"
    "  -d dir   folder with the OCR language data\n"
    "  -r dpi   resolution to render PDF pages (default 300)\n"
    "  -i       print file information only, no OCR\n"
    "  -v       print version information\n"
    "  -h       print this help\n"
    "  --       stop handling options\n",
    progname);
  fflush(stderr);
}


static void l_message (const char *path, const char *msg) {
  if (path != NULL)
    fprintf(stderr, "%s: %s: %s\n", progname, path, msg);
  else
    fprintf(stderr, "%s: %s\n", progname, msg);
  fflush(stderr);
}


static int report (mr_State *R, const char *path, int status) {
  if (status != MR_OK) {
    const char *msg = mr_geterror(R);
    if (msg == NULL || msg[0] == '\0')
      msg = mr_statusname(status);
    l_message(path, msg);
  }
  return status;
}


/*
** Read the options. Returns 0 to go on, -1 to stop with success
** (after -v or -h), or EXIT_USAGE for a bad command line.
*/
static int collectargs (int argc, char **argv, Options *opt) {
  int i;
  opt->output = NULL;
  opt->lang = NULL;
  opt->datapath = NULL;
  opt->dpi = 0;
  opt->info = 0;
  for (i = 1; i < argc; i++) {
    const char *a = argv[i];
    if (a[0] != '-' || a[1] == '\0')  /* file name (or "-") */
      break;
    if (strcmp(a, "--") == 0) {
      i++;
      break;
    }
    if (a[2] != '\0') {  /* all options are one letter */
      print_usage(a);
      return EXIT_USAGE;
    }
    switch (a[1]) {
      case 'o': case 'l': case 'd': case 'r': {
        const char *val;
        if (i + 1 >= argc) {
          print_usage(a);
          return EXIT_USAGE;
        }
        val = argv[++i];
        if (a[1] == 'o') opt->output = val;
        else if (a[1] == 'l') opt->lang = val;
        else if (a[1] == 'd') opt->datapath = val;
        else opt->dpi = atoi(val);
        break;
      }
      case 'i':
        opt->info = 1;
        break;
      case 'v':
        printf("%s\n", mr_version());
        return -1;
      case 'h':
        print_usage(NULL);
        return -1;
      default:
        print_usage(a);
        return EXIT_USAGE;
    }
  }
  opt->first = i;
  return 0;
}


static int setoptions (mr_State *R, const Options *opt) {
  if (opt->lang != NULL && report(R, NULL, mr_setlang(R, opt->lang)))
    return EXIT_USAGE;
  if (opt->datapath != NULL &&
      report(R, NULL, mr_setdatapath(R, opt->datapath)))
    return EXIT_USAGE;
  if (opt->dpi != 0 && report(R, NULL, mr_setdpi(R, opt->dpi)))
    return EXIT_USAGE;
  return 0;
}


static int printinfo (mr_State *R, const char *path, FILE *out) {
  int w, h, ch, status;
  switch (mr_filekind(path)) {
    case MR_KINDIMAGE:
      status = mr_imageinfo(R, path, &w, &h, &ch);
      if (status != MR_OK) return report(R, path, status);
      fprintf(out, "%s: image %dx%d, %d channel%s\n",
              path, w, h, ch, (ch == 1) ? "" : "s");
      return MR_OK;
    case MR_KINDPDF:
      fprintf(out, "%s: pdf\n", path);
      return MR_OK;
    default:
      return report(R, path, mr_readfile(R, path));  /* gets the error */
  }
}


static int dofile (mr_State *R, const char *path, FILE *out) {
  size_t len;
  const char *text;
  int status;
  mr_cleartext(R);
  status = mr_readfile(R, path);
  if (status != MR_OK) return report(R, path, status);
  text = mr_text(R, &len);
  if (fwrite(text, 1, len, out) != len) {
    l_message(NULL, "cannot write the output");
    return MR_ERRFILE;
  }
  if (len > 0 && text[len - 1] != '\n')
    fputc('\n', out);
  return MR_OK;
}


static FILE *openoutput (const char *path) {
#if defined(_WIN32)
  wchar_t wpath[1024];
  if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wpath, 1024) == 0)
    return NULL;
  return _wfopen(wpath, L"wb");
#else
  return fopen(path, "wb");
#endif
}


static int run (int argc, char **argv) {
  Options opt;
  mr_State *R;
  FILE *out = stdout;
  int i, failed = 0;
  int status = collectargs(argc, argv, &opt);
  if (status == -1) return EXIT_SUCCESS;  /* -v or -h */
  if (status != 0) return status;
  if (opt.first >= argc) {
    print_usage(NULL);
    return EXIT_USAGE;
  }
  R = mr_open();
  if (R == NULL) {
    l_message(NULL, "cannot create state: not enough memory");
    return EXIT_FAILURE;
  }
  status = setoptions(R, &opt);
  if (status != 0) goto done;
  if (opt.output != NULL) {
    out = openoutput(opt.output);
    if (out == NULL) {
      l_message(opt.output, "cannot open for writing");
      status = EXIT_FAILURE;
      goto done;
    }
  }
  for (i = opt.first; i < argc; i++) {
    int st = opt.info ? printinfo(R, argv[i], out) : dofile(R, argv[i], out);
    if (st != MR_OK) failed = 1;
  }
  fflush(out);
  status = failed ? EXIT_FAILED : EXIT_SUCCESS;
 done:
  if (out != stdout) fclose(out);
  mr_close(R);
  return status;
}


#if defined(_WIN32)

/*
** On Windows 'argv' is in the ANSI code page, so file names with
** non-ASCII letters would be broken. Rebuild it as UTF-8 from the
** UTF-16 command line, and print UTF-8 to the console.
*/
int main (void) {
  int argc, i, status;
  size_t total = 0;
  char **argv, *p;
  wchar_t **wargv = CommandLineToArgvW(GetCommandLineW(), &argc);
  if (wargv == NULL) {
    l_message(NULL, "cannot read the command line");
    return EXIT_FAILURE;
  }
  for (i = 0; i < argc; i++)
    total += mr_cast(size_t, WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1,
                                                 NULL, 0, NULL, NULL));
  argv = mr_cast(char **, malloc(sizeof(char *) * (argc + 1) + total));
  if (argv == NULL) {
    LocalFree(wargv);
    l_message(NULL, "not enough memory");
    return EXIT_FAILURE;
  }
  p = mr_cast(char *, argv + argc + 1);
  for (i = 0; i < argc; i++) {
    int n = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, p,
                                mr_cast(int, total), NULL, NULL);
    argv[i] = p;
    p += n;
    total -= mr_cast(size_t, n);
  }
  argv[argc] = NULL;
  LocalFree(wargv);
  SetConsoleOutputCP(CP_UTF8);
  status = run(argc, argv);
  free(argv);
  return status;
}

#else

int main (int argc, char **argv) {
  return run(argc, argv);
}

#endif
