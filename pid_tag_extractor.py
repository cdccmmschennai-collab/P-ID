#!/usr/bin/env python3
"""
P&ID tag extractor (vector PDFs with a text layer)  -  v2

  1. Text-layer words, rotated (vertical) text rebuilt from characters
  2. Split-tag repair: 'K-8952' + 'A' -> K-8952A,  'T-8958' + 'A/B' -> T-8958A, T-8958B
  3. Tag grammar: lines, valves, equipment, off-page / sheet refs, vessel trim, tie-ins, special items
  4. Instrument bubbles -> AREA-FUNCTION-LOOP (13 / TW / 87 -> 13-TW-87), function letters checked with isa51.py
  5. Clouds (scalloped stroke) -> in_cloud flag; de-duplication; external-reference flag; audit

Usage:  python pid_tag_extractor.py drawing.pdf [out.csv] [--page N] [--profile auto|ngl1|mgdv]
Requires: pdfplumber, shapely.   Optional: isa51.py next to this file (function-letter validation).
Calibrate per CAD template / project: the CONFIG block below.
"""
import argparse, csv, re, sys
from collections import Counter, defaultdict

import pdfplumber
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

try:
    import isa51
    ISA_OK = True
except ImportError:                     # works without the library, just less strict
    ISA_OK = False

# ============================================================================ CONFIG
CLOUD_LINEWIDTH = 0.72          # stroke width of clouds in this CAD template (check per template)
CLOUD_JOIN_TOL, CLOUD_MIN_ARCS, CLOUD_MAX_FILL = 1.5, 12, 0.85

# Equipment: any single letter, or one of these multi-letter prefixes (add yours: PK, SK, HX ...)
EQUIPMENT_PREFIXES = ["LD", "KF"]
_EQ = r'(?:[A-Z]|' + '|'.join(EQUIPMENT_PREFIXES) + r')-\d{2,5}[A-Z]?(?:/[A-Z0-9])*'
VALVE_CODES = "BV|GV|GLV|BFV|CHV|NRV|DBV|PLG|NV|PSV|XV"

_NGL1_PATTERNS = [                # first match wins
    ("sheet_ref",   re.compile(r'^SH-\d{1,3}$')),
    ("offpage_ref", re.compile(r'^[A-Z0-9]{4}-\d-\d{2}-\d{4}-\d{3}$')),
    # 12"-R13902-3117X-IP | 4"-A-89428-13032X-PP50 : size-service(+number)-class[-insulation/suffix]
    ("line",        re.compile(r'^\d{1,2}"-[A-Z]-?\d{5}-\d{4,5}[A-Z]?(?:-[A-Z]{1,3}\d{0,3})?$')),
    ("valve",       re.compile(rf'^\d{{2}}-(?:{VALVE_CODES})-\d{{3,4}}[A-Z]?$')),
    ("equipment",   re.compile(rf'^{_EQ}$')),
    ("vessel_trim", re.compile(r'^P-\d{5}-\d{5}X$')),
]
_MGDV_PATTERNS = [               # QatarEnergy Mesaieed Gas Distribution (4224-MGDV-...) conventions
    ("offpage_ref", re.compile(r'^\d{4}-[A-Z]{3,5}-\d-\d{2}-\d{4}-\d{3}$')),
    ("line",        re.compile(r'^\d{1,2}"-[A-Z]{1,4}-V\d{3}-(?:C\d{2}[A-Z]|\d{5}[A-Z])(?:-[A-Z]{1,3})?$')),
    ("valve",       re.compile(rf'^V-(?:{VALVE_CODES}|LV)-(?:\d{{3,4}}|XX\d{{3}})$')),
    ("equipment",   re.compile(r'^[A-Z]{1,2}-V-\d{3}[A-Z]?(?:/[A-Z0-9])*$')),
]
PROFILES = {
    "ngl1": dict(patterns=_NGL1_PATTERNS, instrument_layout="area",   instrument_prefix="",  valve_prefix=""),
    "mgdv": dict(patterns=_MGDV_PATTERNS, instrument_layout="noarea", instrument_prefix="V", valve_prefix="V"),
}
PROFILE = "ngl1"
TAG_PATTERNS = _NGL1_PATTERNS
INSTRUMENT_LAYOUT, INSTRUMENT_PREFIX, VALVE_PREFIX = "area", "", ""

def apply_profile(name):
    """Switch the tag grammar (project convention). Add your own entries to PROFILES."""
    global PROFILE, TAG_PATTERNS, INSTRUMENT_LAYOUT, INSTRUMENT_PREFIX, VALVE_PREFIX
    p = PROFILES[name]
    PROFILE, TAG_PATTERNS = name, p["patterns"]
    INSTRUMENT_LAYOUT, INSTRUMENT_PREFIX, VALVE_PREFIX = p["instrument_layout"], p["instrument_prefix"], p["valve_prefix"]

def choose_profile(words):
    """Pick the profile whose grammar matches the most words on this sheet."""
    best, score = PROFILE, -1
    for name, p in PROFILES.items():
        n = sum(1 for w in words if any(rx.match(w["text"].strip(",;.")) for _, rx in p["patterns"]))
        if n > score:
            best, score = name, n
    return best, score

# Instrument bubble: function letters on top, loop number below, 2-digit area up-left or stacked above
INSTRUMENT_EXCLUDE = {"LO","LDE","SP","BW","TSO","IP","RS","MW","SB","UC","FROM","TO","NEW","NOTE","HOLD",
                      "TYPE","ONLY","THE","FOR","AND","NOT","OF","IN","TT","MM","MESH","VENT","POT","DRUM",
                      "LINE","SLOPE","TRIP","RESET","FLOW","LD","REV","DWG","NO","TCV","AIR","OUT","AG","UG",
                      "DEGC","BARG","BAR","RPM","TPD","KW","HZ","PSV"}
PROJECT_FUNCTIONS = {"RV": "Relief valve (project convention)"}      # non-ISA codes used on your drawings
LOOP_RX = re.compile(r'^[A-Z]?\d{1,5}[A-Z]?$')                       # 87 | 34A | 048 | F8952A
AREA_RX = re.compile(r'^\d{2}$')
TIE_IN_FORMAT = "T-{n}"                                              # tie-in diamond 'T' over '032'
EXTERNAL_WORDS = {"FROM", "TO"}
SUFFIX_RX = re.compile(r'^/?[A-Z](?:/[A-Z0-9])*$')                   # 'A', 'A/B', '/D' next to an equipment tag
EQ_BASE_RX = re.compile(r'^(?:[A-Z]|' + '|'.join(EQUIPMENT_PREFIXES) + r')-\d{2,5}$|^[A-Z]{1,2}-V-\d{2,3}$')


# ============================================================================ words
def classify(text):
    for name, rx in TAG_PATTERNS:
        if rx.match(text):
            return name
    return None

def center(w):
    return (w["x0"] + w["x1"]) / 2, (w["top"] + w["bottom"]) / 2

def rotated_words(page, col_tol=1.5, gap_factor=0.9):
    """Vertical text (bottom->top) is stored in reverse order: sort each column by DESCENDING top."""
    ch = [c for c in page.chars if not c["upright"] and c["text"].strip()]
    ch.sort(key=lambda c: (c["x0"], -c["top"]))
    words, cur = [], []
    def flush():
        nonlocal cur
        if cur:
            words.append(dict(text="".join(c["text"] for c in cur),
                              x0=min(c["x0"] for c in cur), x1=max(c["x1"] for c in cur),
                              top=min(c["top"] for c in cur), bottom=max(c["bottom"] for c in cur),
                              upright=False))
            cur = []
    for c in ch:
        if cur:
            p = cur[-1]
            if not (abs(c["x0"] - p["x0"]) <= col_tol and -1.0 <= p["top"] - c["bottom"] <= gap_factor * c["size"]):
                flush()
        cur.append(c)
    flush()
    return words

def merge_equipment_suffixes(words, max_gap=14.0):
    """'K-8952' + 'A'  ->  'K-8952A';  'T-8958' + 'A/B'  ->  'T-8958A/B'  (same line, small gap, upright)."""
    out, used = [], set()
    for i, w in enumerate(words):
        if i in used:
            continue
        t = w["text"].strip(",;.")
        if w["upright"] and EQ_BASE_RX.match(t):
            cy = center(w)[1]
            nxt = [(j, o) for j, o in enumerate(words)
                   if j not in used and o["upright"] and abs(center(o)[1] - cy) < 4
                   and 0 <= o["x0"] - w["x1"] <= max_gap and SUFFIX_RX.match(o["text"].rstrip(".,;"))]
            if nxt:
                j, o = min(nxt, key=lambda p: p[1]["x0"])
                used.add(j)
                w = dict(w, text=t + o["text"].rstrip(".,;"), x1=o["x1"])
        out.append(w)
    return out

def read_words(page):
    upright = [w for w in page.extract_words(use_text_flow=False, extra_attrs=["upright"]) if w["upright"]]
    return merge_equipment_suffixes(upright + rotated_words(page))


# ============================================================================ instruments & tie-ins
def _function_ok(code):
    if code in INSTRUMENT_EXCLUDE or not re.fullmatch(r"[A-Z]{2,5}", code):
        return False
    return isa51.is_valid_function(code, PROJECT_FUNCTIONS) if ISA_OK else len(code) <= 4

def find_instruments_noarea(words):
    """Bubbles printed without an area number (function letters over a 3-digit loop number).
    Tag = INSTRUMENT_PREFIX-FUNCTION-LOOP, e.g. PIT / 211 -> V-PIT-211. Prefix comes from the profile."""
    nums = [dict(w, text=w["text"].rstrip(".")) for w in words if re.fullmatch(r"\d{3}[A-Z]?", w["text"].rstrip("."))]
    out, seen = [], set()
    for f in words:
        if not _function_ok(f["text"]):
            continue
        fx, fy = center(f)
        loops = [n for n in nums if abs(center(n)[0] - fx) < 9 and 6 < center(n)[1] - fy < 22]
        if not loops:
            continue
        loop = min(loops, key=lambda n: center(n)[1] - fy)
        key = (f["text"], round(fx), round(fy))
        if key in seen:
            continue
        seen.add(key)
        pre = f"{INSTRUMENT_PREFIX}-" if INSTRUMENT_PREFIX else ""
        meaning = isa51.explain(f"{f['text']}-1", PROJECT_FUNCTIONS) if ISA_OK else ""
        out.append(dict(tag=f"{pre}{f['text']}-{loop['text']}", cls="instrument", x=fx, y=fy,
                        rotated=not f["upright"], external_ref=False,
                        meaning=(meaning + (f" [prefix '{INSTRUMENT_PREFIX}' added from profile]" if pre else "")).strip()))
    return out

def find_instruments(words):
    if INSTRUMENT_LAYOUT == "noarea":
        return find_instruments_noarea(words)
    nums = [dict(w, text=w["text"].rstrip(".")) for w in words if LOOP_RX.match(w["text"].rstrip("."))]
    out, seen = [], set()
    for f in words:
        if not _function_ok(f["text"]):
            continue
        fx, fy = center(f)
        loops = [n for n in nums if abs(center(n)[0] - fx) < 9 and 6 < center(n)[1] - fy < 22]
        if not loops:
            continue
        loop = min(loops, key=lambda n: center(n)[1] - fy)
        layout_a = [n for n in nums if AREA_RX.match(n["text"]) and 8 < fx - center(n)[0] < 40
                    and -4 < fy - center(n)[1] < 22 and n is not loop]          # area up-left
        layout_b = [n for n in nums if AREA_RX.match(n["text"]) and abs(fx - center(n)[0]) < 6
                    and 8 < fy - center(n)[1] < 20 and n is not loop]            # area stacked above
        if layout_a:
            area = min(layout_a, key=lambda n: abs(fx - center(n)[0] - 22) + abs(fy - center(n)[1] - 11))
        elif layout_b:
            area = min(layout_b, key=lambda n: fy - center(n)[1])
        else:
            continue
        key = (f["text"], round(fx), round(fy))
        if key in seen:
            continue
        seen.add(key)
        meaning = isa51.explain(f"{f['text']}-1", PROJECT_FUNCTIONS) if ISA_OK else ""
        out.append(dict(tag=f"{area['text']}-{f['text']}-{loop['text']}", cls="instrument",
                        x=fx, y=fy, rotated=not f["upright"], external_ref=False, meaning=meaning))
    return out

def find_tie_ins(words):
    """Diamond with 'T' over a 2-3 digit number (tie-in point)."""
    out = []
    for t in words:
        if t["text"] != "T":
            continue
        tx, ty = center(t)
        n = [w for w in words if re.fullmatch(r"\d{2,3}", w["text"]) and abs(center(w)[0] - tx) < 6
             and 6 < center(w)[1] - ty < 20]
        if n:
            out.append(dict(tag=TIE_IN_FORMAT.format(n=n[0]["text"]), cls="tie_in", x=tx, y=ty,
                            rotated=not t["upright"], external_ref=False, meaning="Tie-in point"))
    return out


def find_special_items(words):
    """Fused 'AREA+LETTERS' word (e.g. 89SP) with a loop number just below-right -> AREA-LETTERS-NUMBER.
    Class 'special_item': format/meaning NOT confirmed, review before loading to the register."""
    out = []
    for f in words:
        m = re.fullmatch(r"(\d{2})([A-Z]{2,3})", f["text"])
        if not m:
            continue
        fx, fy = center(f)
        n = [w for w in words if re.fullmatch(r"\d{1,5}", w["text"].rstrip("."))
             and 0 <= center(w)[0] - fx < 30 and 6 < center(w)[1] - fy < 25]
        if n:
            n = min(n, key=lambda w: abs(center(w)[1] - fy))
            out.append(dict(tag=f"{m.group(1)}-{m.group(2)}-{n['text'].rstrip('.')}", cls="special_item",
                            x=fx, y=fy, rotated=not f["upright"], external_ref=False,
                            meaning=f"{m.group(2)} item (UNCONFIRMED: SP may be 'special piping item')"))
    return out


# ============================================================================ clouds
def cloud_polygons(page):
    arcs = [o for o in page.curves
            if str(o.get("stroking_color")) == "0.0" and round(o.get("linewidth") or 0, 2) == CLOUD_LINEWIDTH]
    lines = [LineString(o["pts"]) for o in arcs if len(o["pts"]) >= 2]
    if not lines:
        return []
    blobs = unary_union([l.buffer(CLOUD_JOIN_TOL) for l in lines])
    blobs = list(blobs.geoms) if hasattr(blobs, "geoms") else [blobs]
    polys = []
    for b in blobs:
        if sum(1 for l in lines if b.intersects(l)) < CLOUD_MIN_ARCS:
            continue
        poly = Polygon(b.exterior.coords).buffer(0)
        x0, y0, x1, y1 = poly.bounds
        if poly.area / ((x1 - x0) * (y1 - y0)) > CLOUD_MAX_FILL:
            continue
        polys.append(poly)
    return polys


# ============================================================================ clean-up
def expand_list(tag):
    """E-1317A/B/E/F -> E-1317A, E-1317B, ...   K-1301/2 -> K-1301, K-1302"""
    if "/" not in tag:
        return [tag]
    first, *rest = tag.split("/")
    out = [first]
    for r in rest:
        out.append(first[:-1] + r if len(r) == 1 and first[-1].isalpha() == r.isalpha() else tag)
    return out

def is_external(w, words):
    """FROM/TO just left of the tag on the same text line = reference to another sheet."""
    cx, cy = center(w)
    return any(o["text"] in EXTERNAL_WORDS and abs(center(o)[1] - cy) < 5 and 0 < cx - center(o)[0] < 130
               for o in words)

def extract(pdf_path, page_no=0, profile="auto"):
    page = pdfplumber.open(pdf_path).pages[page_no]
    words = read_words(page)
    apply_profile(choose_profile(words)[0] if profile == "auto" else profile)
    hits = []
    for w in words:
        t = w["text"].strip(",;.")
        cls = classify(t)
        if cls:
            cx, cy = center(w)
            ext = cls == "equipment" and is_external(w, words)
            for tag in (expand_list(t) if cls == "equipment" else [t]):
                hits.append(dict(tag=tag, cls=cls, x=cx, y=cy, rotated=not w["upright"],
                                 external_ref=ext, meaning=""))
    hits += find_instruments(words) + find_tie_ins(words) + find_special_items(words)

    clouds = cloud_polygons(page)
    for h in hits:
        h["cloud_id"] = next((i + 1 for i, c in enumerate(clouds) if c.contains(Point(h["x"], h["y"]))), "")

    groups = defaultdict(list)
    for h in hits:
        groups[(h["tag"], h["cls"])].append(h)
    rows = []
    for (tag, cls), g in groups.items():
        ids = sorted({h["cloud_id"] for h in g if h["cloud_id"] != ""})
        rows.append(dict(
            tag=tag, cls=cls, occurrences=len(g), in_cloud=bool(ids), cloud_ids=";".join(map(str, ids)),
            outside_cloud_too=any(h["cloud_id"] == "" for h in g) and bool(ids),
            external_ref=all(h["external_ref"] for h in g), rotated=any(h["rotated"] for h in g),
            meaning=g[0]["meaning"], x=round(g[0]["x"], 1), y=round(g[0]["y"], 1)))
    rows.sort(key=lambda r: (r["cls"], r["tag"]))
    return rows, words, clouds

def audit(rows, words):
    """Tag-like words nothing captured: REVIEW THESE, they are your likely misses."""
    captured = " ".join(r["tag"] for r in rows)
    cand = [re.compile(r'^(?:\d{1,2}"?-)?[A-Z0-9]{1,5}-\d{2,5}[A-Z]?(?:-[A-Z0-9]+)*$'),
            re.compile(r'^\d{2}[A-Z]{2,3}$|^[A-Z]{2,3}\d{2}$')]          # fused area+letters e.g. 89SP
    miss = sorted({w["text"] for w in words if any(c.match(w["text"]) for c in cand)
                   and w["text"] not in captured})
    dup = [t for t, n in Counter(r["tag"] for r in rows).items() if n > 1]
    return miss, dup


# ============================================================================ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf"); ap.add_argument("out", nargs="?", default="tags.csv")
    ap.add_argument("--page", type=int, default=0)
    ap.add_argument("--profile", default="auto", choices=["auto"] + list(PROFILES))
    a = ap.parse_args()
    rows, words, clouds = extract(a.pdf, a.page, a.profile)
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)
    miss, dup = audit(rows, words)
    print(f"profile: {PROFILE} | {len(rows)} unique tags | clouds: {len(clouds)} | in cloud: {sum(r['in_cloud'] for r in rows)} "
          f"| ISA library: {'on' if ISA_OK else 'off'}")
    print("by class:", dict(Counter(r["cls"] for r in rows)))
    print("external refs:", [r["tag"] for r in rows if r["external_ref"]])
    print("AUDIT - tag-like words NOT captured (review):", miss)

if __name__ == "__main__":
    sys.exit(main())
