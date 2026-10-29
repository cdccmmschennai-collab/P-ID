#!/usr/bin/env python3
"""
P&ID tag extractor for SCANNED / outlined-text drawings  -  v2 (RapidOCR default)

Use this when pid_router.py says SCANNED: raster scans, or vector PDFs whose text was converted to
outlines (no text layer).  Pipeline:
    render tiles (300 dpi) -> OCR at 0 deg and 90 deg -> split lines into words -> same tag grammar,
    instrument-bubble and clean-up logic as pid_tag_extractor.py -> CSV + review list

Usage:   python pid_scan_extractor.py drawing.pdf out.csv [--engine rapidocr] [--profile mgdv]
         python pid_scan_extractor.py drawing.pdf --dump-words words.csv     # inspect raw OCR first
Needs:   pdftoppm (poppler), opencv-python-headless, numpy, pdfplumber, shapely
         + rapidocr-onnxruntime  (default; models are inside the pip package, no download)
         or easyocr / paddleocr / pytesseract (see --engine)
OCR results are cached per tile in --cache-dir, so an interrupted run resumes instead of restarting.

TEST STATUS: rapidocr = tested on a real outlined-text P&ID (4224-MGDV-6-50-2004). Tesseract = tested,
poor on this drawing style (0-1 of 9 tags on a clean render): do not use. easyocr/paddle: not run here.
"""
import argparse, csv, hashlib, json, os, re, subprocess, sys, tempfile
from collections import Counter, defaultdict

import cv2
import numpy as np
import pdfplumber

import pid_tag_extractor as core

DPI = 300
TILE, OVERLAP = 2600, 800           # px at 300 dpi; overlap must exceed the longest tag/reference (~170 pt = 700 px)
MIN_CONF = 0.55                     # rapidocr confidence floor (0-1)
ENGINE = "rapidocr"
_ENG = None


# ----------------------------------------------------------------------------- page access
class Page:
    """Render arbitrary tiles of a PDF page / image file without holding the whole raster in memory."""
    def __init__(self, path, page_no, dpi):
        self.path, self.page_no, self.dpi = path, page_no, dpi
        self.is_pdf = path.lower().endswith(".pdf")
        if self.is_pdf:
            with pdfplumber.open(path) as pdf:
                p = pdf.pages[page_no]
                self.w, self.h = int(p.width * dpi / 72), int(p.height * dpi / 72)   # as viewed (rotation applied)
        else:
            self.img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if self.img is None:
                raise SystemExit(f"cannot read {path}")
            self.h, self.w = self.img.shape

    def tile(self, x, y, w, h):
        w, h = min(w, self.w - x), min(h, self.h - y)
        if not self.is_pdf:
            return self.img[y:y + h, x:x + w]
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "t")
            subprocess.run(["pdftoppm", "-gray", "-r", str(self.dpi), "-x", str(x), "-y", str(y), "-W", str(w),
                            "-H", str(h), "-f", str(self.page_no + 1), "-l", str(self.page_no + 1),
                            "-png", "-singlefile", self.path, out], check=True)
            return cv2.imread(out + ".png", cv2.IMREAD_GRAYSCALE)


# ----------------------------------------------------------------------------- OCR engines
def run_engine(img):
    """-> [(box_pts[4][2], text, conf 0-1)] in tile pixels."""
    global _ENG
    if ENGINE == "rapidocr":
        from rapidocr_onnxruntime import RapidOCR
        _ENG = _ENG or RapidOCR()
        res, _ = _ENG(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        return [(b, t, float(c)) for b, t, c in (res or [])]
    if ENGINE == "easyocr":
        import easyocr
        _ENG = _ENG or easyocr.Reader(["en"], gpu=False)
        return [(b, t, float(c)) for b, t, c in _ENG.readtext(img)]
    if ENGINE == "paddle":
        from paddleocr import PaddleOCR                               # 2.x result layout
        _ENG = _ENG or PaddleOCR(use_angle_cls=False, lang="en", show_log=False)
        out = []
        for line in (_ENG.ocr(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), cls=False) or [[]])[0] or []:
            out.append((line[0], line[1][0], float(line[1][1])))
        return out
    if ENGINE == "tesseract":
        import pytesseract
        d = pytesseract.image_to_data(img, config="--oem 1 --psm 11", output_type=pytesseract.Output.DICT)
        return [([[d["left"][i], d["top"][i]], [d["left"][i] + d["width"][i], d["top"][i]],
                  [d["left"][i] + d["width"][i], d["top"][i] + d["height"][i]],
                  [d["left"][i], d["top"][i] + d["height"][i]]], d["text"][i].strip(), float(d["conf"][i]) / 100)
                for i in range(len(d["text"])) if d["text"][i].strip()]
    raise SystemExit(f"unknown engine {ENGINE}")


def split_line(box, text, conf, x0, y0, upright):
    """OCR engines return text LINES. Split into whitespace tokens with proportional boxes so the
    word-based grammar/bubble logic works. box is in page pixels."""
    xs = [p[0] for p in box]; ys = [p[1] for p in box]
    bx0, bx1, by0, by1 = min(xs), max(xs), min(ys), max(ys)
    toks = text.split()
    if not toks:
        return []
    total = sum(len(t) for t in toks) + (len(toks) - 1)
    horizontal = (bx1 - bx0) >= (by1 - by0)
    out, pos = [], 0
    for t in toks:
        a, b = pos / total, (pos + len(t)) / total
        pos += len(t) + 1
        if horizontal:
            wx0, wx1, wy0, wy1 = bx0 + a * (bx1 - bx0), bx0 + b * (bx1 - bx0), by0, by1
        else:   # vertical line from the rotated pass: reads bottom -> top
            wx0, wx1, wy0, wy1 = bx0, bx1, by1 - b * (by1 - by0), by1 - a * (by1 - by0)
        out.append(dict(text=t, x0=x0 + wx0, x1=x0 + wx1, top=y0 + wy0, bottom=y0 + wy1,
                        conf=conf, upright=upright))
    return out


def ocr_tile(page, x, y, vertical, cache_dir):
    key = f"{os.path.basename(page.path)}_{page.page_no}_{page.dpi}_{ENGINE}_{x}_{y}_{int(vertical)}"
    cf = os.path.join(cache_dir, hashlib.md5(key.encode()).hexdigest() + ".json") if cache_dir else None
    if cf and os.path.exists(cf):
        return json.load(open(cf))
    img = page.tile(x, y, TILE, TILE)
    if (img < 128).mean() < 0.0015:
        res = []
    else:
        h_t = img.shape[0]
        raw = run_engine(cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) if vertical else img)
        res = []
        for box, text, conf in raw:
            if conf < MIN_CONF:
                continue
            if vertical:   # rotated-tile (x', y') -> original tile (x = y', y = h_t - x')
                box = [[p[1], h_t - p[0]] for p in box]
            res += split_line(box, text, conf, x, y, not vertical)
    if cf:
        os.makedirs(cache_dir, exist_ok=True)
        json.dump(res, open(cf, "w"))
    return res


def merge_duplicates(words):
    """Overlapping tiles and the two rotation passes read the same ink twice: keep the better read."""
    words.sort(key=lambda w: -w["conf"])
    kept = []
    for w in words:
        wa = (w["x1"] - w["x0"]) * (w["bottom"] - w["top"])
        dup = False
        for k in kept:
            ix = min(w["x1"], k["x1"]) - max(w["x0"], k["x0"])
            iy = min(w["bottom"], k["bottom"]) - max(w["top"], k["top"])
            if ix > 0 and iy > 0:
                ka = (k["x1"] - k["x0"]) * (k["bottom"] - k["top"])
                if ix * iy / max(min(wa, ka), 1) > 0.5:
                    if len(w["text"]) > len(k["text"]) and k["text"].strip("-.") in w["text"]:
                        k.update(w)                    # k was a tile-edge fragment of this fuller read
                    dup = True; break
        if not dup:
            kept.append(w)
    return kept


def ocr_page(page, vertical=True, cache_dir=None, log=print, on_tile=None):
    step = TILE - OVERLAP
    xs = list(range(0, max(page.w - OVERLAP, 1), step)); ys = list(range(0, max(page.h - OVERLAP, 1), step))
    words, n, total = [], 0, len(xs) * len(ys) * (2 if vertical else 1)
    for y in ys:
        for x in xs:
            for v in ((False, True) if vertical else (False,)):
                words += ocr_tile(page, x, y, v, cache_dir)
                n += 1
                log(f"  OCR tile {n}/{total}", end="\r", flush=True) if callable(log) and log is print else None
                if on_tile:
                    on_tile(n, total)
    return merge_duplicates(words)


# ----------------------------------------------------------------------------- OCR repair + assemble
DIGIT_FIX = str.maketrans("OoIlSsBZ", "00115582")
ALPHA_FIX = str.maketrans("01584", "OISBA")

def clean_token(t):
    t = t.strip(",;:'()[]")
    t = re.sub(r"[—–_]+", "-", t.replace("''", '"').replace("”", '"').replace("“", '"'))
    return t

def repair(token):
    """(clean, 'exact'|'repaired'|None); only repairs when the result is a valid tag."""
    t = clean_token(token)
    if core.classify(t):
        return t, "exact"
    if "/" in t and core.classify(t.split("/")[-1].strip("-")):          # '01/E-V-201A' -> 'E-V-201A'
        return t.split("/")[-1].strip("-"), "repaired"
    seg = t.split("-")
    cands = set()
    if len(seg) == 3:
        cands.add("-".join([seg[0].translate(DIGIT_FIX), seg[1].translate(ALPHA_FIX), seg[2].translate(DIGIT_FIX)]))
    for c in cands:
        if core.classify(c):
            return c, "repaired"
    return None, None


CLASS_TAIL = re.compile(r'^(?:C\d{2}[A-Z]|\d{5}[A-Z])(?:-[A-Z]{1,3})?$|^[A-Z]?\d{2,5}[A-Z]?$')
PARTIAL_VALVE = re.compile(r'^[^A-Za-z0-9]?-?(BV|GV|GLV|NRV|LV|XV|DBV|BFV|CHV|NV)-(\d{3,4})$')

def join_broken_tags(words, max_dx=12.0, max_dy=16.0):
    """Tags wrapped over two lines ('2"-ETH-V057-' + '61440X', 'V-GV-' + '922'): join when the second piece
    sits right of, or just below, the first."""
    out, used = [], set()
    for i, w in enumerate(words):
        if i in used:
            continue
        if w["text"].endswith("-") and len(w["text"]) >= 3:
            wx1, wcx, wby = w["x1"], core.center(w)[0], w["bottom"]
            cand = [(j, o) for j, o in enumerate(words) if j != i and j not in used and CLASS_TAIL.match(o["text"])
                    and ((abs(core.center(o)[1] - core.center(w)[1]) < 4 and 0 <= o["x0"] - wx1 <= max_dx)
                         or (0 <= o["top"] - wby <= max_dy and abs(core.center(o)[0] - wcx) < 35))]
            if cand:
                j, o = min(cand, key=lambda p: abs(core.center(p[1])[0] - wcx) + abs(p[1]["top"] - wby))
                used.add(j)
                w = dict(w, text=w["text"] + o["text"], x1=max(w["x1"], o["x1"]), bottom=max(w["bottom"], o["bottom"]))
        out.append(w)
    return out


def extract(path, dpi=DPI, page_no=0, vertical=True, cache_dir=None, profile="auto", on_tile=None):
    page = Page(path, page_no, dpi)
    words_px = ocr_page(page, vertical, cache_dir, on_tile=on_tile)
    k = 72.0 / dpi
    words = [dict(w, x0=w["x0"]*k, x1=w["x1"]*k, top=w["top"]*k, bottom=w["bottom"]*k,
                  text=clean_token(w["text"])) for w in words_px]
    prof, _score = (core.choose_profile(words) if profile == "auto" else (profile, 0))
    core.apply_profile(prof)
    words = core.merge_equipment_suffixes(join_broken_tags(words))

    hits, review = [], []
    for w in words:
        clean, how = repair(w["text"])
        if clean:
            cls = core.classify(clean)
            cx, cy = core.center(w)
            ext = cls == "equipment" and core.is_external(w, words)
            for tag in (core.expand_list(clean) if cls == "equipment" else [clean]):
                hits.append(dict(tag=tag, cls=cls, x=cx, y=cy, rotated=not w["upright"], external_ref=ext,
                                 meaning="", read=how, conf=w["conf"]))
        elif PARTIAL_VALVE.match(w["text"]) and core.VALVE_PREFIX:
            m = PARTIAL_VALVE.match(w["text"])
            cx, cy = core.center(w)       # leading characters lost by OCR: PROPOSED tag, must be verified
            hits.append(dict(tag=f"{core.VALVE_PREFIX}-{m.group(1)}-{m.group(2)}", cls="valve", x=cx, y=cy,
                             rotated=not w["upright"], external_ref=False, meaning="PARTIAL OCR read, prefix inferred",
                             read="partial", conf=w["conf"]))
        elif re.search(r"\d", w["text"]) and len(w["text"]) >= 7 and w["text"].count("-") >= 2:
            review.append(w)
    for fn in (core.find_instruments, core.find_tie_ins, core.find_special_items):
        for h in fn(words):
            near = [w["conf"] for w in words if abs(core.center(w)[0] - h["x"]) < 30 and abs(core.center(w)[1] - h["y"]) < 30]
            h.update(read="exact", conf=min(near) if near else 0)
            hits.append(h)

    groups = defaultdict(list)
    for h in hits:
        groups[(h["tag"], h["cls"])].append(h)
    rows = []
    for (tag, cls), g in groups.items():
        rows.append(dict(tag=tag, cls=cls, occurrences=len(g), external_ref=all(h["external_ref"] for h in g),
                         rotated=any(h["rotated"] for h in g), meaning=g[0]["meaning"],
                         min_ocr_conf=round(min(h["conf"] for h in g), 2),
                         repaired=any(h["read"] == "repaired" for h in g),
                         needs_review=all(h["read"] == "partial" for h in g) or min(h["conf"] for h in g) < 0.8,
                         x=round(g[0]["x"], 1), y=round(g[0]["y"], 1)))
    rows.sort(key=lambda r: (r["cls"], r["tag"]))
    return rows, words, review, prof


def make_overlay(pdf, page_no, rows, out_png, dpi=40):
    d = tempfile.mkdtemp()
    subprocess.run(["pdftoppm", "-r", str(dpi), "-f", str(page_no + 1), "-l", str(page_no + 1), "-png",
                    "-singlefile", pdf, os.path.join(d, "o")], check=True)
    img = cv2.imread(os.path.join(d, "o.png"))
    colors = {"valve": (0, 0, 255), "instrument": (0, 160, 0), "equipment": (255, 0, 0), "line": (0, 140, 255),
              "offpage_ref": (200, 0, 200)}
    k = dpi / 72.0
    for r in rows:
        cv2.circle(img, (int(r["x"] * k), int(r["y"] * k)), 5, colors.get(r["cls"], (0, 0, 0)),
                   2 if r["needs_review"] else -1)
    cv2.imwrite(out_png, img)


def main():
    global ENGINE
    ap = argparse.ArgumentParser()
    ap.add_argument("scan"); ap.add_argument("out", nargs="?", default="scan_tags.csv")
    ap.add_argument("--engine", choices=["rapidocr", "easyocr", "paddle", "tesseract"], default="rapidocr")
    ap.add_argument("--dpi", type=int, default=DPI); ap.add_argument("--page", type=int, default=0)
    ap.add_argument("--no-vertical", action="store_true", help="skip the 90-degree pass (faster, loses vertical text)")
    ap.add_argument("--cache-dir", default="ocr_cache")
    ap.add_argument("--profile", default="auto", choices=["auto"] + list(core.PROFILES), help="tag grammar / project convention")
    ap.add_argument("--dump-words", help="write every OCR word to this CSV as well")
    ap.add_argument("--overlay", help="write a PNG with every extracted tag marked, for eyeballing misses")
    a = ap.parse_args()
    ENGINE = a.engine
    rows, words, review, prof = extract(a.scan, a.dpi, a.page, not a.no_vertical, a.cache_dir, a.profile)
    if a.dump_words:
        with open(a.dump_words, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=["text", "x0", "top", "x1", "bottom", "conf", "upright"], extrasaction="ignore")
            wr.writeheader(); wr.writerows(words)
    if a.overlay and a.scan.lower().endswith(".pdf"):
        make_overlay(a.scan, a.page, rows, a.overlay)
    if not rows:
        raise SystemExit(f"no tags matched the grammar ({len(words)} OCR words) - check the profile/grammar")
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    print(f"\nprofile {prof} | OCR words {len(words)} | {len(rows)} unique tags -> {a.out} "
          f"| needs_review: {sum(r['needs_review'] for r in rows)}")
    print("by class:", dict(Counter(r["cls"] for r in rows)))
    print("repaired:", [r["tag"] for r in rows if r["repaired"]])
    print("near-miss tokens to review:", sorted({w['text'] for w in review})[:30])

if __name__ == "__main__":
    main()
