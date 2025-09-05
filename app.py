# =================== Permission Backend — Precision-first face search (ArcFace optional) ===================
import os, json, uuid, base64, time, re
from io import BytesIO
from urllib.parse import urlparse, urljoin
from hashlib import sha256

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Face stacks
import numpy as np
from PIL import Image, ImageOps, ImageFilter
import imagehash

# dlib stack (fallback)
import face_recognition
import face_recognition_models  # macOS/Py3.13 shim

# Optional pro stack (auto if available)
USE_ARCFACE = False
try:
    import cv2
    from insightface.app import FaceAnalysis  # ArcFace embeddings + RetinaFace detection
    USE_ARCFACE = True
except Exception:
    USE_ARCFACE = False

# HTTP
import requests
try:
    import cloudscraper
    _SCRAPER = cloudscraper.create_scraper(browser={"custom": "chrome"})
except Exception:
    _SCRAPER = None

# HTML
from bs4 import BeautifulSoup

# .env (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ------------------- Config -------------------
UPLOAD_FOLDER = os.environ.get("PERM_UPLOAD_DIR", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}  # reference uploads only
MAX_CONTENT_MB = int(os.environ.get("PERM_MAX_UPLOAD_MB", "20"))
SAVE_ENCODINGS = os.environ.get("PERM_SAVE_ENCODINGS", "1") == "1"
PERM_DEBUG = os.environ.get("PERM_DEBUG", "0") == "1"
PERM_USE_CNN = os.environ.get("PERM_USE_CNN", "0") == "1"   # dlib CNN fallback
PERM_GL_REGIONS = [r.strip() for r in os.environ.get("PERM_GL_REGIONS", "").split(",") if r.strip()]

# Matching thresholds (strict defaults)
# dlib/face_recognition typical true-match distance ~0.40–0.55; false-matches often >0.60
DLIB_DIST_THRESHOLD = float(os.environ.get("PERM_DLIB_THRESHOLD", "0.55"))
# ArcFace (cosine distance = 1 - cosine_sim). Good strict cutoff ~0.35 (i.e., cosine_sim>=0.65)
ARCFACE_DIST_THRESHOLD = float(os.environ.get("PERM_ARCFACE_THRESHOLD", "0.35"))
# Require face bbox size (min side) in px to avoid noisy thumbnails
MIN_FACE_SIDE_PX = int(os.environ.get("PERM_MIN_FACE_SIDE", "64"))
# Per-domain cap to avoid 30 hats from same shop
MAX_SAVES_PER_DOMAIN = int(os.environ.get("PERM_MAX_PER_DOMAIN", "2"))
# Retail/CDN domain skip (heavy noise)
DEFAULT_SKIP_DOMAINS = {
    "amazon.com","m.media-amazon.com","images-na.ssl-images-amazon.com",
    "ebay.com","i.ebayimg.com",
    "walmartimages.com","i5.walmartimages.com",
    "etsy.com","i.etsystatic.com",
    "macysassets.com","slimages.macysassets.com",
    "fanatics.frgimages.com","sixersshop.com","proshop.patriots.com","mlbshop.com","fansedge.com","kustore.com",
    "hatstore.imgix.net","cdn.shopify.com","neweracap.com","media-photos.depop.com","topperzstoreusa.com"
}
# Negative keywords in URLs/titles that scream "product"
NEGATIVE_KEYS = {"hat","caps","cap","snapback","buy","cart","checkout","product","sku","price","jersey","authentic","replica","mlb","nfl","nhl","nba","store","shop"}
HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
}

# Providers / keys
SEARCH_PROVIDER = os.environ.get("SEARCH_PROVIDER", "").strip().lower()
PROVIDER = os.environ.get("PROVIDER", "").strip().lower() or SEARCH_PROVIDER or "serpapi"
SERPAPI_KEY = os.environ.get("SERPAPI_KEY", "").strip()
IMGBB_KEY = os.environ.get("IMGBB_KEY", "").strip()

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Flask
app = Flask(__name__)
CORS(app)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

# dlib model path shim
face_recognition.api.pose_predictor_model = face_recognition_models.pose_predictor_model_location()
face_recognition.api.pose_predictor_model_large = face_recognition_models.pose_predictor_model_location()

# ----------- ArcFace init (optional) -----------
ARCFACE = None
if USE_ARCFACE:
    try:
        ARCFACE = FaceAnalysis(name="buffalo_l")
        # CPU mode for older macOS
        ARCFACE.prepare(ctx_id=-1, det_size=(640, 640))
        if PERM_DEBUG: print("[ArcFace] using insightface buffalo_l (CPU)")
    except Exception as e:
        if PERM_DEBUG: print(f"[ArcFace] init failed, fallback to dlib: {e}")
        ARCFACE = None
        USE_ARCFACE = False

# ------------------- Utilities -------------------
def json_error(message: str, code: int = 400):
    return jsonify({"error": message}), code

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def np_to_list(enc):
    if enc is None: return None
    try: return enc.tolist()
    except Exception: return None

def save_sidecar(filename: str, data: dict):
    try:
        with open(os.path.join(UPLOAD_FOLDER, f"{filename}.json"), "w") as f:
            json.dump(data, f)
    except Exception:
        pass

def read_sidecar(filename: str) -> dict:
    p = os.path.join(UPLOAD_FOLDER, f"{filename}.json")
    if not os.path.exists(p): return {}
    try:
        with open(p, "r") as f: return json.load(f)
    except Exception: return {}

def list_images_only():
    return [
        f for f in os.listdir(UPLOAD_FOLDER)
        if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))
        and f.rsplit(".", 1)[-1].lower() in {"png", "jpg", "jpeg", "webp", "gif"}
    ]

def list_reference_files_sorted() -> list[str]:
    out = []
    for f in list_images_only():
        m = read_sidecar(f)
        if m.get("kind") == "reference":
            try: mt = os.path.getmtime(os.path.join(UPLOAD_FOLDER, f))
            except Exception: mt = 0
            out.append((mt, f))
    out.sort(reverse=True)
    return [f for _, f in out]

def latest_reference_filename() -> str | None:
    refs = list_reference_files_sorted()
    return refs[0] if refs else None

def domain_of(url: str) -> str:
    try: return urlparse(url).netloc.lower()
    except Exception: return ""

def image_sha256(b: bytes) -> str:
    return sha256(b).hexdigest()

def phash_of_bytes(b: bytes) -> str:
    img = Image.open(BytesIO(b)).convert("RGB")
    return str(imagehash.phash(img))

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # distance = 1 - cosine_sim
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(1.0 - (np.dot(a, b) / denom))

def pick_ext(url: str, headers: dict, b: bytes) -> str:
    ct = (headers.get("Content-Type") or "").split(";")[0].strip().lower()
    mapping = {"image/jpeg":".jpg","image/jpg":".jpg","image/png":".png","image/webp":".webp","image/gif":".gif","image/tiff":".jpg","image/tif":".jpg"}
    if ct in mapping: return mapping[ct]
    try:
        fmt = Image.open(BytesIO(b)).format
        fm = (fmt or "").upper()
        return {"JPEG":".jpg","JPG":".jpg","PNG":".png","WEBP":".webp","GIF":".gif","TIFF":".jpg","TIF":".jpg"}.get(fm) or ".jpg"
    except Exception:
        return ".jpg"

def fetch(url: str, referer: str | None = None, timeout: int = 30, stream: bool = False):
    headers = dict(HTTP_HEADERS)
    if referer: headers["Referer"] = referer
    try:
        r = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=stream)
        if r.status_code in (403, 429, 503) and _SCRAPER:
            r = _SCRAPER.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=stream)
        return r
    except Exception as e:
        if _SCRAPER:
            try: return _SCRAPER.get(url, headers=headers, timeout=timeout, allow_redirects=True, stream=stream)
            except Exception: pass
        raise e

# ------------------- Face engines -------------------
def _preprocess_for_detection(b: bytes) -> np.ndarray | None:
    """Upscale tiny images + autocontrast + sharpen → numpy array (RGB)."""
    try:
        pil = Image.open(BytesIO(b)).convert("RGB")
        w,h = pil.size
        if max(w,h) < 640:
            scale = 1024 / float(max(w,h))
            pil = pil.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        pil = ImageOps.autocontrast(pil, cutoff=1).filter(ImageFilter.SHARPEN)
        return np.array(pil)
    except Exception:
        try:
            return face_recognition.load_image_file(BytesIO(b))
        except Exception:
            return None

def _faces_dlib(arr: np.ndarray):
    # return list of (top,right,bottom,left)
    locs = face_recognition.face_locations(arr, number_of_times_to_upsample=2)
    if not locs and PERM_USE_CNN:
        locs = face_recognition.face_locations(arr, model="cnn")
    if not locs:
        locs = face_recognition.face_locations(arr, number_of_times_to_upsample=3)
    return locs

def _encodings_dlib(arr: np.ndarray, locs):
    return face_recognition.face_encodings(arr, known_face_locations=locs, num_jitters=1)

def _encodings_arcface(arr: np.ndarray):
    # returns [(bbox, embedding)]
    faces = ARCFACE.get(arr) if ARCFACE else []
    out = []
    for f in faces:
        # bbox = [x1,y1,x2,y2]
        x1,y1,x2,y2 = [int(v) for v in f.bbox]
        emb = np.array(f.normed_embedding, dtype=np.float32)
        out.append(((y1,x2,y2,x1), emb))  # convert to dlib-like (t,r,b,l)
    return out

def _bbox_ok(bb, img_wh) -> bool:
    t,r,b,l = bb
    w = max(0, r-l); h = max(0, b-t)
    if min(w,h) < MIN_FACE_SIDE_PX: return False
    # face must be at least ~6% of max side
    if max(w,h) < 0.06 * max(img_wh): return False
    # box must be inside
    return (w>0 and h>0)

def _multi_ref_embeddings():
    """Load all reference images → build an embedding pool for robust min-distance."""
    refs = list_reference_files_sorted()
    if not refs: return []
    embeds = []
    for rf in refs:
        p = os.path.join(UPLOAD_FOLDER, rf)
        try:
            arr = face_recognition.load_image_file(p)
        except Exception:
            continue
        if USE_ARCFACE and ARCFACE:
            pairs = _encodings_arcface(arr)
            for bb, emb in pairs:
                if _bbox_ok(bb, arr.shape[:2][::-1]): embeds.append(("arc", emb))
        else:
            locs = _faces_dlib(arr)
            if not locs: continue
            encs = _encodings_dlib(arr, locs)
            for e,bb in zip(encs, locs):
                if _bbox_ok(bb, arr.shape[:2][::-1]): embeds.append(("dlib", np.array(e, dtype=np.float32)))
    return embeds

def _min_distance_to_refs(emb):
    """Distance to closest ref across both engines (auto-map)."""
    global _REF_POOL
    best = 1e9
    for kind, ref in _REF_POOL:
        if kind == "arc" and emb.shape[0] == ref.shape[0]:
            d = cosine_distance(emb, ref)
        elif kind != "arc" and emb.shape[0] == ref.shape[0]:
            d = float(np.linalg.norm(emb - ref))  # not used; kept for completeness
            # map to dlib metric by using face_recognition distance when available—skip here
        else:
            # skip cross-metric compare
            continue
        if d < best: best = d
    return best

_REF_POOL = []  # [(kind, embedding)]

# ------------------- Policies & cases -------------------
POL_PATH = os.path.join(UPLOAD_FOLDER, "policies.json")
def load_policies() -> dict:
    if os.path.exists(POL_PATH):
        try:
            with open(POL_PATH, "r") as f:
                obj = json.load(f)
                obj["whitelist"] = [s.lower() for s in obj.get("whitelist", [])]
                obj["blacklist"] = [s.lower() for s in obj.get("blacklist", [])]
                return obj
        except Exception:
            return {"whitelist": [], "blacklist": []}
    return {"whitelist": [], "blacklist": []}

def save_policies(obj: dict):
    with open(POL_PATH, "w") as f:
        json.dump({"whitelist": obj.get("whitelist", []), "blacklist": obj.get("blacklist", [])}, f)

CASES_PATH = os.path.join(UPLOAD_FOLDER, "cases.json")
def load_cases() -> list[dict]:
    if os.path.exists(CASES_PATH):
        try:
            with open(CASES_PATH, "r") as f: return json.load(f)
        except Exception: return []
    return []
def save_cases(cases: list[dict]):
    with open(CASES_PATH, "w") as f: json.dump(cases, f)
def next_case_id(cases: list[dict]) -> int:
    mx = 0
    for c in cases:
        try: mx = max(mx, int(c.get("id", 0)))
        except Exception: pass
    return mx + 1

# ------------------- SerpAPI providers -------------------
class SerpApiImagesProvider:
    def __init__(self, key: str): self.key = key
    def _one(self, query: str, ijn: int = 0, gl: str = "us") -> dict:
        if not self.key: raise RuntimeError("SERPAPI_KEY missing.")
        r = requests.get("https://serpapi.com/search.json",
                         params={"engine":"google_images","q":query,"ijn":ijn,"hl":"en","gl":gl,"api_key":self.key},
                         timeout=45)
        if r.status_code != 200: raise RuntimeError(f"SerpAPI Images {r.status_code}: {r.text}")
        return r.json()
    def search_text_paged(self, q: str, max_images=30, max_pages=3, gl="us"):
        out, seen = [], set()
        for ijn in range(max_pages):
            data = self._one(q, ijn=ijn, gl=gl)
            for item in (data.get("images_results") or []):
                img = item.get("original") or item.get("image") or item.get("thumbnail")
                page = item.get("link") or item.get("source") or item.get("page_url") or item.get("thumbnail")
                if not img or not page: continue
                key = (img,page)
                if key in seen: continue
                seen.add(key)
                out.append({"image":img,"page":page})
                if len(out) >= max_images: return out
        return out

class SerpApiLensProvider:
    def __init__(self, key: str): self.key = key
    def _one(self, url: str, ijn: int = 0, gl: str = "us") -> dict:
        if not self.key: raise RuntimeError("SERPAPI_KEY missing.")
        r = requests.get("https://serpapi.com/search.json",
                         params={"engine":"google_lens","url":url,"ijn":ijn,"hl":"en","gl":gl,"api_key":self.key},
                         timeout=45)
        if r.status_code != 200: raise RuntimeError(f"SerpAPI Lens {r.status_code}: {r.text}")
        return r.json()
    def search_by_url_paged(self, image_url: str, max_images=24, max_pages=6, gls=None):
        out, seen = [], set()
        gls = gls or ["us"]
        for gl in gls:
            for ijn in range(max_pages):
                data = self._one(image_url, ijn=ijn, gl=gl)
                rows = (data.get("visual_matches") or []) or (data.get("images_results") or [])
                for item in rows:
                    img = item.get("original") or item.get("image") or item.get("thumbnail") or item.get("link")
                    page = item.get("link") or item.get("source") or item.get("page_url") or item.get("thumbnail")
                    if not img or not page: continue
                    key = (img,page)
                    if key in seen: continue
                    seen.add(key)
                    out.append({"image":img,"page":page,"gl":gl})
                    if len(out) >= max_images: return out
        return out

# ------------------- Host & index helpers -------------------
def upload_to_imgbb(image_bytes: bytes) -> str:
    if not IMGBB_KEY: raise RuntimeError("IMGBB_KEY missing.")
    b64 = base64.b64encode(image_bytes).decode("ascii")
    r = requests.post("https://api.imgbb.com/1/upload", data={"key":IMGBB_KEY,"image":b64}, timeout=45)
    if r.status_code != 200: raise RuntimeError(f"ImgBB {r.status_code}: {r.text}")
    d = (r.json() or {}).get("data") or {}
    url = (d.get("image") or {}).get("url") or d.get("display_url") or d.get("url")
    if not url: raise RuntimeError("ImgBB returned no usable URL.")
    return url

def extract_page_image_candidates(page_url: str, html: str | None = None) -> list[str]:
    try:
        if html is None:
            rsp = fetch(page_url, timeout=30)
            if rsp.status_code != 200: return []
            html = rsp.text
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        try: soup = BeautifulSoup(html or "", "html.parser")
        except Exception: return []

    cands = []
    # OG/Twitter
    for sel in [('meta[property="og:image"]', "content"),
                ('meta[name="og:image"]', "content"),
                ('meta[name="twitter:image"]', "content"),
                ('link[rel="image_src"]', "href")]:
        for tag in soup.select(sel[0]):
            u = (tag.get(sel[1]) or "").strip()
            if u: cands.append(u)
    # <img> + srcset
    def _expand_srcset(img_tag):
        srcset = img_tag.get("srcset") or ""
        for part in srcset.split(","):
            u = part.strip().split(" ")[0]
            if u: cands.append(u)
    for img in soup.select("img"):
        u = (img.get("src") or img.get("data-src") or "").strip()
        if img.get("srcset"): _expand_srcset(img)
        if u: cands.append(u)

    # Dedup + absolutize
    absu, seen = [], set()
    for u in cands:
        au = urljoin(page_url, u)
        if au not in seen:
            seen.add(au); absu.append(au)
    return absu[:12]

def _retail_like(url: str) -> bool:
    u = url.lower()
    if any(k in u for k in NEGATIVE_KEYS): return True
    return False

def download_index_scored(candidate: dict, ref_pool, whitelist: set[str], blacklist: set[str],
                          domain_counts: dict, engine: str) -> dict | None:
    img_url, page_url = candidate["image"], candidate["page"]
    dom = domain_of(page_url)

    # Skip retail/CDN
    if (dom in DEFAULT_SKIP_DOMAINS or _retail_like(page_url)) and (dom not in whitelist):
        if PERM_DEBUG: print(f"[Skip] retail/keyword: {dom} | {page_url}")
        return None

    # Per-domain cap
    if domain_counts.get(dom, 0) >= MAX_SAVES_PER_DOMAIN:
        if PERM_DEBUG: print(f"[Skip] domain cap reached: {dom}")
        return None

    # Whitelist skip
    if dom in whitelist:
        if PERM_DEBUG: print(f"[Skip] whitelisted domain: {dom}")
        return None

    # Helper: get bytes with referer & cloudscraper
    def _fetch_bytes(u):
        try:
            r = fetch(u, referer=page_url, timeout=30)
            if r.status_code == 200: return r.content
        except Exception: pass
        return None

    # Decode pipeline → faces → embeddings → score
    def _score_bytes(b: bytes):
        arr = _preprocess_for_detection(b)
        if arr is None: return None
        H,W = arr.shape[:2]
        E = []
        if USE_ARCFACE and ARCFACE and engine in ("auto","arc"):
            pairs = _encodings_arcface(arr)
            for bb, emb in pairs:
                if _bbox_ok(bb, (W,H)): E.append(("arc", emb))
        if (not E) and engine in ("auto","dlib"):
            locs = _faces_dlib(arr)
            # size/quality gate
            locs = [bb for bb in locs if _bbox_ok(bb, (W,H))]
            if locs:
                encs = _encodings_dlib(arr, locs)
                for e in encs: E.append(("dlib", np.array(e, dtype=np.float32)))
        if not E: return None

        # Compare to ref pool (same kind only)
        best = 1e9
        best_kind = None
        for kind, emb in E:
            # choose right threshold per kind
            if kind == "arc":
                d = min([cosine_distance(emb, r) for rk,r in ref_pool if rk=="arc"] or [1e9])
                if d < best: best, best_kind = d, "arc"
            else:
                # dlib distance
                d = min([float(np.linalg.norm(emb - r)) for rk,r in ref_pool if rk=="dlib"] or [1e9])
                if d < best: best, best_kind = d, "dlib"
        if best_kind is None: return None

        # Strict decision
        if best_kind == "arc":
            ok = best <= ARCFACE_DIST_THRESHOLD
        else:
            ok = best <= DLIB_DIST_THRESHOLD

        return {"ok": ok, "dist": best, "kind": best_kind}

    # 1) Try the candidate image
    b = _fetch_bytes(img_url)
    verdict = _score_bytes(b) if b else None

    # 2) Fallback: page images if needed
    if not verdict or not verdict["ok"]:
        imgs = extract_page_image_candidates(page_url)
        for u in imgs:
            b2 = _fetch_bytes(u)
            if not b2: continue
            v2 = _score_bytes(b2)
            if v2 and v2["ok"]:
                b, verdict = b2, v2
                break

    if not verdict or not verdict["ok"]:
        if PERM_DEBUG: print(f"[Skip] no pass @strict: {page_url}")
        return None

    # Save
    ext = pick_ext(img_url, {}, b or b"")
    fname = secure_filename(f"crawl_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}{ext}")
    with open(os.path.join(UPLOAD_FOLDER, fname), "wb") as f: f.write(b)

    # Sidecar
    meta = {
        "kind":"result","source_url":page_url,"image_url":img_url,"domain":dom,
        "engine_used": verdict["kind"], "distance": verdict["dist"],
        "thresholds": {"arc": ARCFACE_DIST_THRESHOLD, "dlib": DLIB_DIST_THRESHOLD}
    }
    save_sidecar(fname, meta)
    domain_counts[dom] = domain_counts.get(dom, 0) + 1
    return {"filename": fname, "source_url": page_url, "domain": dom, "engine": verdict["kind"], "distance": verdict["dist"]}

# ------------------- Routes -------------------
@app.route("/")
def home():
    return jsonify({
        "message": "✅ Permission app is running!",
        "provider": PROVIDER,
        "has_SERPAPI": bool(SERPAPI_KEY),
        "has_IMGBB": bool(IMGBB_KEY),
        "uploads_dir": UPLOAD_FOLDER,
        "engine": "arcface" if (USE_ARCFACE and ARCFACE) else "dlib",
        "arcface_threshold": ARCFACE_DIST_THRESHOLD,
        "dlib_threshold": DLIB_DIST_THRESHOLD,
        "min_face_px": MIN_FACE_SIDE_PX,
        "max_per_domain": MAX_SAVES_PER_DOMAIN,
        "regions": PERM_GL_REGIONS
    })

@app.route("/uploads/<path:filename>")
def served_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# Upload reference
@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files: return json_error("No file part", 400)
    file = request.files["image"]
    if not file or file.filename == "": return json_error("No selected file", 400)
    if not allowed_file(file.filename): return json_error("Invalid file type", 400)
    filename = secure_filename(file.filename)
    unique = f"{uuid.uuid4().hex}_{filename}"
    path = os.path.join(app.config["UPLOAD_FOLDER"], unique)
    file.save(path)
    # Store just a simple sidecar (we recompute embeddings later into a pool)
    img = face_recognition.load_image_file(path)
    locs = face_recognition.face_locations(img)
    save_sidecar(unique, {"kind":"reference","face_count":len(locs)})
    return jsonify({"filename": unique, "num_faces": len(locs), "face_locations": locs})

# List / Delete
@app.route("/list", methods=["GET"])
def list_images():
    imgs = list_images_only()
    out = {"images": imgs, "results": []}
    for f in imgs:
        m = read_sidecar(f)
        if m.get("kind") == "result":
            out["results"].append({
                "filename": f, "source_url": m.get("source_url"),
                "domain": m.get("domain"), "engine": m.get("engine_used"),
                "distance": m.get("distance", None)
            })
    return jsonify(out)

@app.route("/delete", methods=["POST"])
def delete_file():
    data = request.get_json(force=True, silent=True) or {}
    fn = data.get("filename")
    if not fn: return json_error("Filename is required", 400)
    target = os.path.join(UPLOAD_FOLDER, fn)
    if not os.path.exists(target): return json_error("File not found", 404)
    try:
        os.remove(target)
        side = f"{target}.json"
        if os.path.exists(side): os.remove(side)
        return jsonify({"message": f"{fn} deleted"})
    except Exception as e:
        return json_error(str(e), 500)

# Policies
@app.route("/policies", methods=["GET","POST"])
def policies():
    if request.method == "GET": return jsonify(load_policies())
    data = request.get_json(force=True, silent=True) or {}
    wl = [s.strip().lower() for s in data.get("whitelist", []) if s.strip()]
    bl = [s.strip().lower() for s in data.get("blacklist", []) if s.strip()]
    save_policies({"whitelist": wl, "blacklist": bl})
    return jsonify({"status":"ok"})

# SerpAPI providers
class SerpApiLensProvider:
    def __init__(self, key: str): self.key = key
    def _one(self, url: str, ijn: int = 0, gl: str = "us") -> dict:
        if not self.key: raise RuntimeError("SERPAPI_KEY missing.")
        r = requests.get("https://serpapi.com/search.json",
                         params={"engine":"google_lens","url":url,"ijn":ijn,"hl":"en","gl":gl,"api_key":self.key},
                         timeout=45)
        if r.status_code != 200: raise RuntimeError(f"SerpAPI Lens {r.status_code}: {r.text}")
        return r.json()
    def search_by_url_paged(self, image_url: str, max_images=24, max_pages=6, gls=None):
        out, seen = [], set()
        gls = gls or ["us"]
        for gl in gls:
            for ijn in range(max_pages):
                data = self._one(image_url, ijn=ijn, gl=gl)
                rows = (data.get("visual_matches") or []) or (data.get("images_results") or [])
                for item in rows:
                    img = item.get("original") or item.get("image") or item.get("thumbnail") or item.get("link")
                    page = item.get("link") or item.get("source") or item.get("page_url") or item.get("thumbnail")
                    if not img or not page: continue
                    key = (img,page)
                    if key in seen: continue
                    seen.add(key)
                    out.append({"image":img,"page":page,"gl":gl})
                    if len(out) >= max_images: return out
        return out

def upload_to_imgbb(image_bytes: bytes) -> str:
    if not IMGBB_KEY: raise RuntimeError("IMGBB_KEY missing.")
    b64 = base64.b64encode(image_bytes).decode("ascii")
    r = requests.post("https://api.imgbb.com/1/upload", data={"key":IMGBB_KEY,"image":b64}, timeout=45)
    if r.status_code != 200: raise RuntimeError(f"ImgBB {r.status_code}: {r.text}")
    d = (r.json() or {}).get("data") or {}
    url = (d.get("image") or {}).get("url") or d.get("display_url") or d.get("url")
    if not url: raise RuntimeError("ImgBB returned no usable URL.")
    return url

def to_jpeg_bytes(pil: Image.Image, q=92) -> bytes:
    buf = BytesIO(); pil.convert("RGB").save(buf, format="JPEG", quality=q, optimize=True); return buf.getvalue()

def build_variants_for_ref(ref_path: str, aggressive: bool) -> list[bytes]:
    pil = Image.open(ref_path).convert("RGB")
    vars = [to_jpeg_bytes(pil, 92), to_jpeg_bytes(pil, 80), to_jpeg_bytes(pil.resize((min(1024,pil.size[0]), int(pil.size[1]*min(1024,pil.size[0])/pil.size[0])), Image.LANCZOS), 92)]
    # Simple face crop with padding when possible
    try:
        arr = face_recognition.load_image_file(ref_path)
        locs = face_recognition.face_locations(arr)
        if locs:
            # largest face
            def area(bb): t,r,b,l=bb; return (b-t)*(r-l)
            t,r,b,l = max(locs, key=area)
            pad = int(0.25*max(b-t, r-l))
            h,w = arr.shape[:2]
            yt, yb = max(0,t-pad), min(h,b+pad); xl, xr = max(0,l-pad), min(w,r+pad)
            crop = Image.fromarray(arr[yt:yb, xl:xr])
            vars += [to_jpeg_bytes(crop, 92)]
    except Exception:
        pass
    if aggressive:
        vars += [to_jpeg_bytes(ImageOps.mirror(pil), 92)]
    return vars

# ---- Reverse crawl (precision) ----
@app.route("/reverse_crawl", methods=["POST"])
def reverse_crawl():
    if PROVIDER != "serpapi": return json_error("Reverse crawl implemented for PROVIDER=serpapi only.", 400)
    if not SERPAPI_KEY: return json_error("SERPAPI_KEY not set.", 400)
    if not IMGBB_KEY: return json_error("IMGBB_KEY not set.", 400)

    data = request.get_json(force=True, silent=True) or {}
    aggressive = bool(data.get("aggressive", True))
    max_images = int(data.get("max_images", 20))
    max_pages  = int(data.get("max_pages", 6 if aggressive else 3))

    refs = list_reference_files_sorted()
    if not refs: return json_error("No reference image found. Upload one first.", 400)

    # Build reference pool (multi-ref ensemble)
    global _REF_POOL
    _REF_POOL = _multi_ref_embeddings()
    if not _REF_POOL:
        return json_error("Could not build reference embeddings from your uploads.", 400)

    pol = load_policies(); wl = set(pol.get("whitelist", [])); bl = set(pol.get("blacklist", []))
    gls = PERM_GL_REGIONS or (["us","gb","ca","au"] if aggressive else ["us","gb"])

    lens = SerpApiLensProvider(SERPAPI_KEY)
    domain_counts = {}
    attempts = []
    saved = []

    # We try a few ref variants (JPEG qualities, crop, flip)
    ref_path = os.path.join(UPLOAD_FOLDER, refs[0])
    variants = build_variants_for_ref(ref_path, aggressive=aggressive)

    for vb in variants:
        try:
            url = upload_to_imgbb(vb)
            cands = lens.search_by_url_paged(url, max_images=max_images*4, max_pages=max_pages, gls=gls)
            attempts.append({"variant":"v", "candidates":len(cands), "regions":list({c.get("gl","us") for c in cands})})
            if PERM_DEBUG: print(f"[Reverse] candidates={len(cands)} for variant")
            if not cands: continue

            engine_mode = "auto" if USE_ARCFACE else "dlib"
            for c in cands:
                r = download_index_scored(c, _REF_POOL, wl, bl, domain_counts, engine=engine_mode)
                if r: saved.append(r)
                if len(saved) >= max_images: break
            if len(saved) >= max_images: break
        except Exception as e:
            if PERM_DEBUG: print(f"[Reverse] variant error: {e}")
            continue

    return jsonify({
        "reference": refs[0],
        "saved": saved,
        "provider": "serpapi_lens",
        "candidates_count": sum(a["candidates"] for a in attempts),
        "attempts": attempts,
        "engine": "arcface" if (USE_ARCFACE and ARCFACE) else "dlib",
        "thresholds": {"arc": ARCFACE_DIST_THRESHOLD, "dlib": DLIB_DIST_THRESHOLD}
    })

# ---- DMCA preview ----
@app.route("/dmca/preview", methods=["POST"])
def dmca_preview():
    d = request.get_json(force=True, silent=True) or {}
    full_name = (d.get("full_name") or "").strip()
    email = (d.get("email") or "").strip()
    recipient = (d.get("recipient") or "abuse@host.com").strip()
    urls = d.get("urls") or []
    if not full_name or not email or not urls:
        return json_error("full_name, email, and at least one URL are required.", 400)
    body = f"""To: {recipient}
From: {full_name} <{email}>
Subject: DMCA Takedown Notice

Hello,

I am {full_name}. I am the rights holder of the images identified below. I did not authorize their publication and request their removal pursuant to the DMCA.

Allegedly infringing material URLs:
{chr(10).join(urls)}

I have a good faith belief that the use of the material described above is not authorized by the copyright owner, its agent, or the law.
The information in this notice is accurate, and under penalty of perjury, I am the owner, or authorized to act on behalf of the owner, of an exclusive right that is allegedly infringed.

Please promptly remove or disable access to the material and notify me when completed.

Signature,
{full_name}
{email}
"""
    return jsonify({"body_text": body})

@app.route("/debug_env")
def debug_env():
    return jsonify({
        "provider": PROVIDER,
        "has_SERPAPI_KEY": bool(SERPAPI_KEY),
        "has_IMGBB_KEY": bool(IMGBB_KEY),
        "engine": "arcface" if (USE_ARCFACE and ARCFACE) else "dlib",
        "arcface_threshold": ARCFACE_DIST_THRESHOLD,
        "dlib_threshold": DLIB_DIST_THRESHOLD,
        "min_face_px": MIN_FACE_SIDE_PX,
        "max_per_domain": MAX_SAVES_PER_DOMAIN,
        "regions": PERM_GL_REGIONS
    })

# ------------------- Run -------------------
if __name__ == "__main__":
    print(f"[Permission] http://127.0.0.1:5000  | provider={PROVIDER}  | uploads={UPLOAD_FOLDER} | engine={'arcface' if (USE_ARCFACE and ARCFACE) else 'dlib'}")
    app.run(debug=True)
