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

NAME="mmc_reader"
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
INC="-I. -Ivendor"
LIBS="${LIBS:-}"

RELEASE_OPT="-O2 -DNDEBUG"
DEBUG_OPT="-O0 -g -DMR_DEBUG"

# build <compiler> <outdir> <opt flags> <exe suffix>
build () {
  comp="$1"
  out="$2"
  opt="$3"
  exe="$4"
  objs=""
  mkdir -p "$out/obj"
  for f in *.cpp; do
    [ -f "$f" ] || continue
    o="$out/obj/${f%.cpp}.o"
    echo "  CXX  $f"
    $comp $STD $WARN $opt ${CFLAGS:-} $INC -c "$f" -o "$o"
    objs="$objs $o"
  done
  if [ -z "$objs" ]; then
    echo "nothing to build: no .cpp files in the root"
    exit 1
  fi
  echo "  LINK $out/$NAME$exe"
  $comp $objs $LIBS -o "$out/$NAME$exe"
  echo "done: $out/$NAME$exe"
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
    case "$t" in
      *windows*) exe=".exe" ;;
    esac
    build "$ZIG c++ -target $t" "build/$mode/$t" "$opt" "$exe"
  done
}

NATIVE_EXE=""
if [ "$OS" = "Windows_NT" ]; then
  NATIVE_EXE=".exe"
fi

case "$MODE" in
  release)
    build "$CXX" "build/release" "$RELEASE_OPT" "$NATIVE_EXE"
    ;;
  debug)
    build "$CXX" "build/debug" "$DEBUG_OPT" "$NATIVE_EXE"
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
