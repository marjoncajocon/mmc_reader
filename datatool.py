# datatool.py - dataset helpers for getdata.sh (data tool, not part of the
# program; the only Python in the project)
#
#   python datatool.py iam <dir>          unpack IAM lines (Parquet)
#   python datatool.py fontlist <json>    print "category family styles"
#                                         of the Google Fonts we use
#   python datatool.py fontscript <json> <dir>
#                                         print a script that downloads them
#
# iam: writes <dir>/<split>/NNNNN.<ext> and <dir>/<split>.txt with one
# "file<TAB>text" line per image (UTF-8).

import json
import os
import re
import sys


# how many families of each Google Fonts category, and which styles
CHOICE = {
  "Sans Serif": (150, "400,700,400italic"),
  "Serif": (100, "400,700,400italic"),
  "Monospace": (25, "400,700,400italic"),
  "Display": (40, "400"),
  "Handwriting": (120, "400"),
}


# {======================================================
# IAM
# =======================================================

def detokenize(text):
  # IAM writes "Marie " ( Parlophone ) , a" with spaces around the
  # punctuation; the handwriting has none: "Marie" (Parlophone), a
  text = " ".join(text.split())
  text = re.sub(r" ([.,;:!?)])", r"\1", text)
  text = re.sub(r"\( ", "(", text)
  text = re.sub(r" ('(s|m|ve|ll|re|d)\b|n't\b)", r"\1", text)
  out, inside, i = [], False, 0
  while i < len(text):  # pair the double quotes: "word" not " word "
    c = text[i]
    if c == '"':
      if not inside:
        if out and out[-1] == " " and len(out) >= 2 and out[-2] == "(":
          out.pop()
        out.append(c)
        if i + 1 < len(text) and text[i + 1] == " ":
          i += 1
      else:
        if out and out[-1] == " ":
          out.pop()
        out.append(c)
      inside = not inside
    else:
      out.append(c)
    i += 1
  return "".join(out).strip()


def ext_of(data):
  if data[:2] == b"\xff\xd8":
    return "jpg"
  if data[:8] == b"\x89PNG\r\n\x1a\n":
    return "png"
  return "bin"


def iam_split(root, split):
  import pyarrow.parquet as pq
  table = pq.read_table(os.path.join(root, split + ".parquet"))
  images = table.column("image").to_pylist()
  texts = table.column("text").to_pylist()
  outdir = os.path.join(root, split)
  os.makedirs(outdir, exist_ok=True)
  with open(os.path.join(root, split + ".txt"), "w", encoding="utf-8",
            newline="\n") as lst:
    for i, (img, text) in enumerate(zip(images, texts)):
      data = img["bytes"]
      name = "%05d.%s" % (i, ext_of(data))
      with open(os.path.join(outdir, name), "wb") as f:
        f.write(data)
      lst.write(name + "\t" + detokenize(text) + "\n")
  print("%s: %d lines" % (split, len(texts)))


def iam(root):
  sys.path.insert(0, os.path.join(os.path.dirname(root), ".pylib"))
  for s in ("train", "validation", "test"):
    iam_split(root, s)

# }======================================================


# {======================================================
# Google Fonts
# =======================================================

def choose(path):
  # the most popular open source families of each category
  text = open(path, encoding="utf-8").read()
  meta = json.loads(text[text.index("{"):])
  fams = meta["familyMetadataList"]
  out = []
  for cat, (count, styles) in CHOICE.items():
    pick = [f for f in fams
            if f["category"] == cat and "latin-ext" in f["subsets"] and
            f.get("isOpenSource", False) and not f.get("isBrandFont") and
            not f.get("isNoto") and not f.get("colorCapabilities")]
    pick.sort(key=lambda f: f.get("popularity", 1 << 30))
    for f in pick[:count]:
      have = f.get("fonts", {})
      want = [s for s in styles.split(",")
              if s.replace("italic", "i") in have]
      if want:
        out.append((cat.replace(" ", ""), f["family"], want))
  return out


def fontlist(path):
  for cat, fam, want in choose(path):
    print("%s\t%s\t%s" % (cat, fam, ",".join(want)))


def fontscript(path, outdir):
  # a script of curl commands: an old browser name makes Google Fonts
  # answer with plain .ttf files; files already there are skipped
  css = "https://fonts.googleapis.com/css?family="
  print("# made by datatool.py fontscript")
  for cat, fam, want in choose(path):
    d = "%s/%s" % (outdir, cat)
    print('mkdir -p "%s"' % d)
    for style in want:
      f = '%s/%s-%s.ttf' % (d, fam.replace(" ", ""), style)
      print('if [ ! -s "%s" ]; then' % f)
      print('  u=$(curl -fsL -A "Mozilla/4.0" "%s%s:%s" | '
            'grep -o "https://[^)]*\\.ttf" | head -n 1)'
            % (css, fam.replace(" ", "+"),
               style + "&subset=latin,latin-ext"))
      print('  if [ -n "$u" ]; then curl -fsL -o "%s" "$u" || true; fi' % f)
      print('fi')
  print('echo "fonts: done"')

# }======================================================


if __name__ == "__main__":
  if len(sys.argv) == 3 and sys.argv[1] == "iam":
    iam(sys.argv[2])
  elif len(sys.argv) == 3 and sys.argv[1] == "fontlist":
    fontlist(sys.argv[2])
  elif len(sys.argv) == 4 and sys.argv[1] == "fontscript":
    fontscript(sys.argv[2], sys.argv[3])
  else:
    sys.exit("usage: python datatool.py iam <dir> | fontlist <json> | "
             "fontscript <json> <dir>")
