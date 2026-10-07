#!/usr/bin/env mmc
#
# build.sh - build mmc_reader (C++11)
# Run it from the mmc shell (D:\mmc-shell\mmc.exe):
#
#   ./build.sh                        release build (default)
#   ./build.sh release                optimized  -> build/release/
#   ./build.sh debug                  debug info -> build/debug/
#   ./build.sh cross-release [T...]   optimized, for each target T
#   ./build.sh cross-debug [T...]     debug info, for each target T
#   ./build.sh targets                list the default cross targets
#   ./build.sh clean                  remove the build folder
#
# Output:
#   build/<mode>/mmc_reader[.exe]             native build
#   build/<mode>/mmc_train[.exe]              model trainer
#   build/<mode>/obj/*.o
#   build/cross-<mode>/<T>/mmc_reader[.exe]   cross build
#   build/cross-<mode>/<T>/obj/*.o
#
# T is a zig target, e.g. x86_64-linux-gnu. With no T, every target in
# TARGETS is built. Native builds use $CXX (set in
# D:\mmc-shell\etc\profile, default "zig c++"); cross builds use $ZIG
# (default "zig"). Extra flags: CFLAGS=... LIBS=... ./build.sh
#

set -e

MODE="${1:-release}"
if [ $# -gt 0 ]; then
  shift
fi

CXX="${CXX:-zig c++}"
ZIG="${ZIG:-zig}"

TARGETS="x86_64-windows-gnu"
TARGETS="$TARGETS x86_64-linux-gnu aarch64-linux-gnu"
TARGETS="$TARGETS x86_64-macos aarch64-macos"

STD="-std=c++11"
WARN="-Wall -Wextra -pedantic"
# vendor/ is a system include path: no warnings from vendored code
INC="-I. -isystem vendor"
LIBS="${LIBS:-}"

RELEASE_OPT="-O2 -DNDEBUG"
DEBUG_OPT="-O0 -g -DMR_DEBUG"

# system libraries for a Windows build
WINLIBS="-lshell32"

# programs: each <name>.cpp has a main and is linked with the library
# (every other .cpp), like lua.c and luac.c
PROGS="mmc_reader mmc_train"

isprog () {
  case " $PROGS " in
    *" $1 "*) return 0 ;;
  esac
  return 1
}

# build <compiler> <outdir> <opt flags> <exe suffix> <system libs>
build () {
  comp="$1"
  out="$2"
  opt="$3"
  exe="$4"
  syslibs="$5"
  libobjs=""
  mkdir -p "$out/obj"
  for f in *.cpp; do
    [ -f "$f" ] || continue
    o="$out/obj/${f%.cpp}.o"
    echo "  CXX  $f"
    $comp $STD $WARN $opt ${CFLAGS:-} $INC -c "$f" -o "$o"
    if isprog "${f%.cpp}"; then
      :
    else
      libobjs="$libobjs $o"
    fi
  done
  for p in $PROGS; do
    [ -f "$p.cpp" ] || continue
    echo "  LINK $out/$p$exe"
    $comp "$out/obj/$p.o" $libobjs $LIBS $syslibs -o "$out/$p$exe"
  done
  echo "done: $out/"
}

# cross <mode> <opt flags> [targets...]
cross () {
  mode="$1"
  opt="$2"
  shift 2
  list="$*"
  if [ -z "$list" ]; then
    list="$TARGETS"
  fi
  for t in $list; do
    echo "[$t]"
    exe=""
    syslibs=""
    case "$t" in
      *windows*)
        exe=".exe"
        syslibs="$WINLIBS"
        ;;
    esac
    build "$ZIG c++ -target $t" "build/$mode/$t" "$opt" "$exe" "$syslibs"
  done
}

NATIVE_EXE=""
NATIVE_LIBS=""
if [ "$OS" = "Windows_NT" ]; then
  NATIVE_EXE=".exe"
  NATIVE_LIBS="$WINLIBS"
fi

case "$MODE" in
  release)
    build "$CXX" "build/release" "$RELEASE_OPT" "$NATIVE_EXE" "$NATIVE_LIBS"
    ;;
  debug)
    build "$CXX" "build/debug" "$DEBUG_OPT" "$NATIVE_EXE" "$NATIVE_LIBS"
    ;;
  cross-release)
    cross "cross-release" "$RELEASE_OPT" "$@"
    ;;
  cross-debug)
    cross "cross-debug" "$DEBUG_OPT" "$@"
    ;;
  targets)
    for t in $TARGETS; do
      echo "$t"
    done
    ;;
  clean)
    rm -rf build
    echo "cleaned"
    ;;
  *)
    echo "usage: ./build.sh [release|debug|targets|clean]"
    echo "       ./build.sh cross-release [target...]"
    echo "       ./build.sh cross-debug [target...]"
    exit 1
    ;;
esac
