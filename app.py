#!/usr/bin/env python3
"""Web front-end for the P&ID tag extraction pipeline.

Upload a PDF (or image) -> per-page detection (vector text layer vs scanned/OCR)
-> run the matching extractor -> table + CSV download.

Run:  python app.py      then open http://localhost:5000
"""
import csv, io, os, secrets, shutil, tempfile, threading, traceback, uuid

from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.security import check_password_hash

import pid_router as router

HERE = os.path.dirname(os.path.abspath(__file__))
USERNAME = "User"
PASSWORD_HASH = ("scrypt:32768:8:1$aerT6Pm94jOevIs0$aa8d2c39a723c7ec12a58c6ead02ed31715d797b8a21cbffdbe5f83a9a42304c"
                 "e86e58fb47564386d96ef64f5a188d7819c109443c8a2f289d0740116e36344c")


def _secret_key():
    """Persisted so logins survive a server restart."""
    p = os.path.join(HERE, ".secret_key")
    if not os.path.exists(p):
        with open(p, "w") as f:
            f.write(secrets.token_hex(32))
    return open(p).read().strip()


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024
app.config.update(SECRET_KEY=_secret_key(), SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
WORK = os.path.join(tempfile.gettempdir(), "pid_tool_jobs")
os.makedirs(WORK, exist_ok=True)
RESULTS = os.path.join(HERE, "results")
os.makedirs(RESULTS, exist_ok=True)
JOBS = {}          # job_id -> dict(path, detect, status, progress, rows, error, json, deleted)
ALLOWED = (".pdf",) + router.IMG_EXT


def _fields(rows):
    f = ["page", "source"]
    for r in rows:
        f += [k for k in r if k not in f]
    return f


def _worker(job_id, overrides, engine, dpi, profile):
    job = JOBS[job_id]
    try:
        rows_all, pages, used = [], job["detect"], []
        for n, d in enumerate(pages):
            verdict = overrides.get(str(d["page"]), d["verdict"])
            used.append(dict(d, verdict=verdict))
            job["progress"] = f"page {n + 1}/{len(pages)} ({verdict})"
            job["pct"] = 100.0 * n / len(pages)

            def on_tile(i, total, n=n):         # fill this page's share of the bar tile by tile
                job["pct"] = 100.0 * (n + i / total) / len(pages)
                job["progress"] = f"page {n + 1}/{len(pages)} - OCR tile {i}/{total}"

            if verdict == "vector":
                rows = router.run_vector(job["path"], d["page"], profile)
            else:
                rows = router.run_scanned(job["path"], d["page"], dpi, engine, profile,
                                          os.path.join(job["dir"], "ocr_cache"), on_tile)
            for r in rows:
                r["page"], r["source"] = d["page"] + 1, verdict
            rows_all += rows
        job["json"] = router.save_results_json(RESULTS, job["name"], job_id, used, rows_all,
                                               dict(engine=engine, dpi=dpi, profile=profile))
        shutil.rmtree(job["dir"], ignore_errors=True)   # results are on disk; drop the upload + OCR cache
        job["deleted"] = True
        job["rows"], job["status"], job["pct"] = rows_all, "done", 100.0
    except Exception as e:                      # surface to the UI instead of dying silently
        job["status"], job["error"] = "error", f"{type(e).__name__}: {e}"
        traceback.print_exc()


@app.before_request
def require_login():
    if request.endpoint in ("login", "static") or session.get("user"):
        return None
    if request.path.startswith("/api/"):
        return jsonify(error="Not logged in"), 401
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if (request.form.get("username") == USERNAME
                and check_password_hash(PASSWORD_HASH, request.form.get("password", ""))):
            session.clear()
            session["user"] = USERNAME
            return redirect(url_for("index"))
        error = "Invalid username or password."
    return render_template("login.html", error=error), (401 if error else 200)


@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.get("/")
def index():
    return render_template("index.html", user=session["user"])


@app.post("/api/analyze")
def analyze():
    f = request.files.get("file")
    if not f or not f.filename.lower().endswith(ALLOWED):
        return jsonify(error="Upload a PDF or image file (png/jpg/tif)."), 400
    job_id = uuid.uuid4().hex[:12]
    d = os.path.join(WORK, job_id)
    os.makedirs(d)
    path = os.path.join(d, "input" + os.path.splitext(f.filename)[1].lower())
    f.save(path)
    try:
        detect = router.detect(path)
    except Exception as e:
        shutil.rmtree(d, ignore_errors=True)
        return jsonify(error=f"Could not read file: {e}"), 400
    JOBS[job_id] = dict(path=path, dir=d, name=f.filename, detect=detect, status="detected", rows=[])
    return jsonify(job=job_id, name=f.filename, pages=detect)


@app.post("/api/process/<job_id>")
def process(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="Unknown job"), 404
    if job["status"] == "running":
        return jsonify(error="Already running"), 409
    if job.get("deleted"):
        return jsonify(error="Original file was deleted after extraction; upload it again to re-run."), 409
    b = request.get_json(silent=True) or {}
    job.update(status="running", progress="starting", pct=0.0, error=None, rows=[])
    threading.Thread(target=_worker, daemon=True, args=(
        job_id, b.get("overrides", {}), b.get("engine", "rapidocr"),
        int(b.get("dpi", 300)), b.get("profile", "auto"))).start()
    return jsonify(ok=True)


@app.get("/api/status/<job_id>")
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify(error="Unknown job"), 404
    out = dict(status=job["status"], progress=job.get("progress", ""), pct=job.get("pct", 0), error=job.get("error"))
    if job["status"] == "done":
        out["rows"] = job["rows"]
        out["fields"] = _fields(job["rows"]) if job["rows"] else []
        out["json"] = os.path.basename(job["json"])
    return jsonify(out)


@app.get("/api/json/<job_id>")
def json_download(job_id):
    job = JOBS.get(job_id)
    if not job or not job.get("json") or not os.path.exists(job["json"]):
        return "No results", 404
    return send_file(job["json"], mimetype="application/json", as_attachment=True,
                     download_name=os.path.basename(job["json"]))


@app.get("/api/csv/<job_id>")
def csv_download(job_id):
    job = JOBS.get(job_id)
    if not job or job["status"] != "done" or not job["rows"]:
        return "No results", 404
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=_fields(job["rows"]), extrasaction="ignore")
    w.writeheader(); w.writerows(job["rows"])
    name = os.path.splitext(job["name"])[0] + "_tags.csv"
    return send_file(io.BytesIO(buf.getvalue().encode()), mimetype="text/csv",
                     as_attachment=True, download_name=name)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
