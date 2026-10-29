#!/usr/bin/env python3
"""
P&ID router: decide, per page, whether a PDF is VECTOR (usable text layer) or SCANNED (needs OCR),
then run the matching extractor.

  vector  -> pid_tag_extractor.py   (text layer + grammar + clouds from vector arcs)
  scanned -> pid_scan_extractor.py  (rasterize + OCR + grammar + raster clouds)

Usage:
  python pid_router.py drawing.pdf out.csv                  # detect + extract every page
  python pid_router.py drawing.pdf --detect-only            # just print the verdict per page
  python pid_router.py scan.pdf out.csv --engine rapidocr   # OCR engine for scanned pages (default)
  python pid_router.py drawing.pdf out.csv --force vector   # override the detector

Image files (png/jpg/tif) are always treated as scanned.
Requires: pdfplumber (+ whatever the two extractor scripts need). Keep all three files together.
"""
import argparse, csv, json, os, sys
from datetime import datetime

import pdfplumber

IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")

# Calibration (decision thresholds)
MIN_TAG_HITS = 3          # grammar-valid tags in the text layer => text layer is genuinely usable
MIN_CHARS = 200           # fewer characters than this => essentially no text layer
SCAN_IMAGE_RATIO = 0.60   # raster images covering >= 60% of the page => page is a picture


def analyse_page(page):
    """Collect the evidence the verdict is based on."""
    import pid_tag_extractor as core
    page_area = float(page.width * page.height)
    img_area = 0.0
    for im in page.images:
        w = max(0.0, min(im["x1"], page.width) - max(im["x0"], 0))
        h = max(0.0, min(im["bottom"], page.height) - max(im["top"], 0))
        img_area += w * h
    words = page.extract_words(use_text_flow=False)
    tag_hits = sum(1 for w in words if core.classify(w["text"].strip(",;.")))
    return dict(
        chars=len(page.chars),
        words=len(words),
        tag_hits=tag_hits,
        image_ratio=round(min(img_area / page_area, 1.0), 3),
        vector_objects=len(page.curves) + len(page.lines) + len(page.rects),
    )


def classify_page(ev):
    """Return (verdict, reason). verdict is 'vector' or 'scanned'."""
    has_text = ev["chars"] >= MIN_CHARS
    is_picture = ev["image_ratio"] >= SCAN_IMAGE_RATIO

    if has_text and ev["tag_hits"] >= MIN_TAG_HITS and not is_picture:
        return "vector", "usable text layer with grammar-valid tags, page is not a full-page image"
    if is_picture and has_text:
        return "scanned", ("full-page image with a hidden text layer (a 'searchable scan'): "
                           "that layer is OCR output of unknown quality, so it is OCR'd again; "
                           "use --force vector to trust it")
    if is_picture:
        return "scanned", "full-page image, no text layer"
    if not has_text and ev["vector_objects"] > 500:
        return "scanned", ("vector drawing but text converted to outlines (no text layer): "
                           "text must be OCR'd from a rendering")
    if has_text and ev["tag_hits"] < MIN_TAG_HITS:
        return "scanned", (f"text layer exists but only {ev['tag_hits']} tag-like words: "
                           "likely a title-only or garbled layer; verify manually or use --force vector")
    return "scanned", "no usable text layer"        # unknown => OCR path is the safe default


def detect(pdf_path):
    if pdf_path.lower().endswith(IMG_EXT):
        return [dict(page=0, verdict="scanned", reason="image file", evidence={})]
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            ev = analyse_page(page)
            verdict, reason = classify_page(ev)
            out.append(dict(page=i, verdict=verdict, reason=reason, evidence=ev))
    return out


def run_vector(pdf_path, page_no, profile):
    import pid_tag_extractor as vec
    rows, _words, _clouds = vec.extract(pdf_path, page_no, profile)
    return rows


def run_scanned(path, page_no, dpi, engine, profile, cache_dir, on_tile=None):
    import pid_scan_extractor as scan
    scan.ENGINE = engine
    rows, _w, _review, _prof = scan.extract(path, dpi, page_no, True, cache_dir, profile, on_tile)
    return rows


def save_results_json(out_dir, name, job_id, pages, rows, settings):
    """Write one upload's results to out_dir/<stem>_<job_id>.json atomically; return the path.
    pages: detect() entries with the verdict actually used."""
    os.makedirs(out_dir, exist_ok=True)
    doc = dict(
        file=name, job_id=job_id, processed_at=datetime.now().isoformat(timespec="seconds"),
        settings=settings,
        pages=[dict(page=p["page"] + 1, verdict=p["verdict"], reason=p["reason"]) for p in pages],
        tag_count=len(rows), tags=rows,
    )
    path = os.path.join(out_dir, f"{os.path.splitext(name)[0]}_{job_id}.json")
    with open(path + ".tmp", "w", encoding="utf-8") as f:     # tmp + replace: never a half-written JSON
        json.dump(doc, f, ensure_ascii=False, indent=2, default=str)
    os.replace(path + ".tmp", path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf"); ap.add_argument("out", nargs="?", default="tags.csv")
    ap.add_argument("--detect-only", action="store_true")
    ap.add_argument("--force", choices=["vector", "scanned"])
    ap.add_argument("--engine", choices=["rapidocr", "easyocr", "paddle", "tesseract"], default="rapidocr")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--profile", default="auto", help="tag grammar: auto | ngl1 | mgdv (see PROFILES in pid_tag_extractor.py)")
    ap.add_argument("--cache-dir", default="ocr_cache")
    a = ap.parse_args()

    verdicts = detect(a.pdf)
    for v in verdicts:
        if a.force:
            v["verdict"], v["reason"] = a.force, "forced by --force"
        print(f"page {v['page'] + 1}: {v['verdict'].upper():8s} | {v['reason']}")
        if v["evidence"]:
            print(f"          evidence: {v['evidence']}")
    if a.detect_only:
        return

    all_rows = []
    for v in verdicts:
        rows = (run_vector(a.pdf, v["page"], a.profile) if v["verdict"] == "vector"
                else run_scanned(a.pdf, v["page"], a.dpi, a.engine, a.profile, a.cache_dir))
        for r in rows:
            r["page"] = v["page"] + 1
            r["source"] = v["verdict"]
        all_rows += rows
    if not all_rows:
        sys.exit("no tags extracted")
    fields = ["page", "source"] + [k for k in all_rows[0] if k not in ("page", "source")]
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        wr.writeheader(); wr.writerows(all_rows)
    print(f"{len(all_rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
