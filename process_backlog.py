#!/usr/bin/env python3
"""Extract every leftover upload in the web app's job folder, save the results as JSON, then delete it.

Each %TEMP%\\pid_tool_jobs\\<job_id>\\ folder holding an input.* file is detected + extracted
(existing ocr_cache is reused), written to results\\input_<job_id>.json, and the folder is removed.
A folder whose extraction fails is kept so it can be retried.

Usage:
  python process_backlog.py --dry-run     # list what would be processed
  python process_backlog.py               # extract, save JSON, delete originals
  python process_backlog.py --keep        # extract + save JSON, keep originals
"""
import argparse, glob, os, shutil, tempfile, traceback

import pid_router as router

WORK = os.path.join(tempfile.gettempdir(), "pid_tool_jobs")
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def process(folder, path, a):
    pages, rows = router.detect(path), []
    for p in pages:
        print(f"  page {p['page'] + 1}: {p['verdict']}")
        r = (router.run_vector(path, p["page"], a.profile) if p["verdict"] == "vector"
             else router.run_scanned(path, p["page"], a.dpi, a.engine, a.profile, os.path.join(folder, "ocr_cache")))
        for x in r:
            x["page"], x["source"] = p["page"] + 1, p["verdict"]
        rows += r
    return router.save_results_json(RESULTS, os.path.basename(path), os.path.basename(folder), pages, rows,
                                    dict(engine=a.engine, dpi=a.dpi, profile=a.profile)), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep", action="store_true", help="do not delete the original after saving JSON")
    ap.add_argument("--engine", choices=["rapidocr", "easyocr", "paddle", "tesseract"], default="rapidocr")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--profile", default="auto")
    a = ap.parse_args()

    jobs = []
    for folder in sorted(glob.glob(os.path.join(WORK, "*"))):
        inputs = [f for f in glob.glob(os.path.join(folder, "input.*")) if not f.endswith(".tmp")]
        if os.path.isdir(folder) and inputs:
            jobs.append((folder, inputs[0]))
    print(f"{len(jobs)} upload(s) in {WORK}")
    if a.dry_run:
        for folder, path in jobs:
            print(f"  {os.path.basename(folder)}  {os.path.basename(path)}  {os.path.getsize(path):,} bytes")
        return

    done, failed = [], []
    for i, (folder, path) in enumerate(jobs, 1):
        print(f"[{i}/{len(jobs)}] {os.path.basename(folder)}")
        try:
            out, n = process(folder, path, a)
        except Exception as e:                  # keep the folder for a retry, carry on with the rest
            traceback.print_exc()
            failed.append((folder, f"{type(e).__name__}: {e}"))
            continue
        if not a.keep:
            shutil.rmtree(folder, ignore_errors=True)
        print(f"  {n} tags -> {out}" + ("" if a.keep else "  (original deleted)"))
        done.append(out)

    print(f"\n{len(done)} saved, {len(failed)} failed")
    for folder, err in failed:
        print(f"  KEPT {folder}: {err}")


if __name__ == "__main__":
    main()
