# iamconv.py - unpack the IAM line dataset (Parquet) for mmc_trainseq
# Run by getdata.sh; it is a data tool, not part of the program.
#
#   python iamconv.py build/data/iam
#
# Writes <dir>/<split>/NNNNN.<ext> images and <dir>/<split>.txt with one
# "file<TAB>text" line per image (UTF-8).

import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(sys.argv[1]), ".pylib"))
import pyarrow.parquet as pq  # noqa: E402


def ext_of(data):
  if data[:2] == b"\xff\xd8":
    return "jpg"
  if data[:8] == b"\x89PNG\r\n\x1a\n":
    return "png"
  return "bin"


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


def convert(root, split):
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
      text = detokenize(text)
      lst.write(name + "\t" + text + "\n")
  print("%s: %d lines" % (split, len(texts)))


if __name__ == "__main__":
  for s in ("train", "validation", "test"):
    convert(sys.argv[1], s)
