# =================== Permission Backend — ArcFace-only (Render-safe) ===================
# No dlib/face_recognition imports. Uses InsightFace (ArcFace) only.
import os, io, json, uuid, time, base64, hashlib
from urllib.parse import urlparse
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import requests

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# ArcFace (InsightFace) + OpenCV (headless)
import cv2
from insightface.app import FaceAnalysis

# ------------------- Config -------------------
UPLOAD_FOLDER = os.environ.get("PERM_UPLOAD_DIR", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_CONTENT_MB = int(os.environ.get("PERM_MAX_UPLOAD_MB", "20"))

# Providers
SEARCH_PROVIDER = os.environ.get("SEARCH_PROVIDER", "").strip().lower()
PROVIDER = os.environ.get("PROVIDER", "").strip().lower() or SEARCH_PROVIDER or "serpapi"

# Keys
SERPAPI_KEY = (os.environ.get("SERPAPI_KEY") or "").strip()
IMGBB_KEY   = (os.environ.get("IMGBB_KEY") or "").strip()

# ArcFace thresholds / knobs
ARCFACE_THRESHOLD = float(os.environ.get("PERM_ARCFACE_THRESHOLD", "0.35"))
MIN_FACE_PX       = int(os.environ.get("PERM_MIN_FACE_SIDE", "64"))
MAX_PER_DOMAIN    = int(os.environ.get("PERM_MAX_PER_DOMAIN", "2"))

# Regions for Google Lens via SerpAPI (joined search/heuristics; not always honored by Lens API)
REGIONS = [r.strip() for r in os.environ.get("PERM_GL_REGIONS", "us,gb,ca,au").split(",") if r.strip()]

DEBUG = os.environ.get("PERM_DEBUG", "0") == "1"

# Flask
app = Flask(__name__)
CORS(app, supports_credentials=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024

# ------------------- Helpers -------------------
def log(msg: str):
    if DEBUG:
        print(msg, flush=True)

def allowed_url_ext(url: str) -> bool:
    p = urlparse(url).path.lower()
    return any(p.endswith(ext) for ext in ALLOWED_EXTENSIONS)

def infer_ext_from_url(url: str) -> str:
    p = urlparse(url).path.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        if p.endswith(ext):
            return ext
    return ".jpg"

def sha1_of_bytes(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()

def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    # embeddings are typically L2-normalized by InsightFace; still be safe
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    sim = float(np.dot(a, b) / (na * nb))
    return 1.0 - sim

def save_sidecar(filename: str, data: dict):
    path = os.path.join(UPLOAD_FOLDER, f"{filename}.json")
    with open(path, "w") as f:
        json.dump(data, f)

def read_sidecar(filename: str) -> dict:
    path = os.path.join(UPLOAD_FOLDER, f"{filename}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def list_images_only() -> List[str]:
    out = []
    for f in os.listdir(UPLOAD_FOLDER):
        fp = os.path.join(UPLOAD_FOLDER, f)
        if os.path.isfile(fp):
            low = f.lower()
            if any(low.endswith(ext) for ext in (".jpg", ".jpeg", ".png", ".webp")):
                out.append(f)
    return sorted(out)

def latest_reference_filename() -> Optional[str]:
    best, best_mtime = None, -1
    for f in list_images_only():
        meta = read_sidecar(f)
        if meta.get("kind") == "reference":
            m = os.path.getmtime(os.path.join(UPLOAD_FOLDER, f))
            if m > best_mtime:
                best, best_mtime = f, m
    return best

# ------------------- ArcFace init -------------------
# Use CPU provider in Render
try:
    fa = FaceAnalysis(name="buffalo_l")
    # ctx_id=-1 -> CPU; set detection size moderate
    fa.prepare(ctx_id=-1, det_size=(640, 640))
    ENGINE = "arcface"
    log("[ArcFace] using insightface buffalo_l (CPU)")
except Exception as e:
    # Extremely unlikely on Render (we pinned deps), but fallback label if init fails.
    fa = None
    ENGINE = "unavailable"
    log(f"[ArcFace] init failed: {e}")

def pil_to_bgr(pil_img: Image.Image) -> np.ndarray:
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    arr = np.array(pil_img)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def detect_and_embed(image_bgr: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple[int,int,int,int]]]:
    """
    Returns: (embeddings, boxes) where boxes are [top, right, bottom, left]
    Filters out faces smaller than MIN_FACE_PX on either side.
    """
    if fa is None:
        return [], []
    faces = fa.get(image_bgr)  # list of Face objects
    embs, boxes = [], []
    for f in faces:
        x1, y1, x2, y2 = [int(v) for v in f.bbox]  # bbox as x1,y1,x2,y2
        w, h = x2 - x1, y2 - y1
        if min(w, h) < MIN_FACE_PX:
            continue
        emb = getattr(f, "embedding", None)
        if emb is None:
            continue
        embs.append(np.array(emb, dtype=np.float32))
        # convert to [top, right, bottom, left]
        boxes.append((y1, x2, y2, x1))
    return embs, boxes

def image_bytes_to_embeddings(b: bytes) -> Tuple[List[np.ndarray], List[Tuple[int,int,int,int]]]:
    try:
        pil = Image.open(io.BytesIO(b))
    except Exception:
        return [], []
    bgr = pil_to_bgr(pil)
    return detect_and_embed(bgr)

def upload_to_imgbb(image_bytes: bytes) -> str:
    if not IMGBB_KEY:
        raise RuntimeError("IMGBB_KEY missing.")
    payload = {"key": IMGBB_KEY, "image": base64.b64encode(image_bytes).decode("ascii")}
    r = requests.post("https://api.imgbb.com/1/upload", data=payload, timeout=45)
    r.raise_for_status()
    data = r.json()
    url = (data.get("data") or {}).get("url")
    if not url:
        raise RuntimeError("ImgBB returned no URL.")
    return url

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"

def http_get(url: str, timeout: int = 30) -> bytes:
    r = requests.get(url, headers={"User-Agent": UA, "Referer": "https://www.google.com/"}, timeout=timeout)
    r.raise_for_status()
    return r.content

# ------------------- SerpAPI providers -------------------
class SerpApiImagesProvider:
    def __init__(self, key: str):
        self.key = key
    def search_text(self, query: str, max_images: int = 12) -> List[str]:
        if not self.key:
            raise RuntimeError("SERPAPI_KEY missing.")
        params = {
            "engine": "google_images",
            "q": query,
            "ijn": 0,
            "api_key": self.key
        }
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=45)
        r.raise_for_status()
        data = r.json()
        urls = []
        for item in data.get("images_results", []) or []:
            for k in ("original", "thumbnail"):
                u = item.get(k)
                if isinstance(u, str) and allowed_url_ext(u):
                    urls.append(u)
                    break
            if len(urls) >= max_images:
                break
        return urls

class SerpApiLensProvider:
    def __init__(self, key: str):
        self.key = key
    def search_by_url_paged(self, image_url: str, max_pages: int = 3) -> List[str]:
        if not self.key:
            raise RuntimeError("SERPAPI_KEY missing.")
        results = []
        seen = set()
        for page in range(max_pages):
            params = {"engine": "google_lens", "url": image_url, "api_key": self.key, "ijn": page}
            r = requests.get("https://serpapi.com/search.json", params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            # Prefer visual_matches
            bucket = data.get("visual_matches") or []
            if not bucket:
                bucket = data.get("images_results") or []
            if not bucket:
                break
            page_urls = []
            for it in bucket:
                for k in ("thumbnail", "original", "image", "source", "link"):
                    u = it.get(k)
                    if isinstance(u, str) and allowed_url_ext(u):
                        if u not in seen:
                            seen.add(u)
                            page_urls.append(u)
                        break
            if not page_urls:
                break
            results.extend(page_urls)
        return results

# ------------------- Policy store (JSON) -------------------
POL_PATH = os.path.join(UPLOAD_FOLDER, "policies.json")

def load_policies() -> dict:
    if os.path.exists(POL_PATH):
        try:
            with open(POL_PATH, "r") as f:
                obj = json.load(f)
                return {"whitelist": obj.get("whitelist", []), "blacklist": obj.get("blacklist", [])}
        except Exception:
            pass
    return {"whitelist": [], "blacklist": []}

def save_policies(obj: dict):
    with open(POL_PATH, "w") as f:
        json.dump({"whitelist": obj.get("whitelist", []), "blacklist": obj.get("blacklist", [])}, f)

def is_whitelisted(url: str, wl: List[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return any((s.lower() in url.lower()) or host.endswith(s.lower()) for s in wl)

def is_blacklisted(url: str, bl: List[str]) -> bool:
    host = urlparse(url).netloc.lower()
    return any((s.lower() in url.lower()) or host.endswith(s.lower()) for s in bl)

# ------------------- Core file ops -------------------
@app.route("/")
def home():
    return jsonify({
        "message": "✅ Permission app is running!",
        "provider": PROVIDER,
        "engine": ENGINE,
        "has_SERPAPI": bool(SERPAPI_KEY),
        "has_IMGBB": bool(IMGBB_KEY),
        "uploads_dir": UPLOAD_FOLDER,
        "arcface_threshold": ARCFACE_THRESHOLD,
        "min_face_px": MIN_FACE_PX,
        "max_per_domain": MAX_PER_DOMAIN,
        "regions": REGIONS,
    })

@app.route("/uploads/<path:filename>")
def served_upload(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

@app.route("/debug_env")
def debug_env():
    return jsonify({
        "provider": PROVIDER,
        "has_SERPAPI_KEY": bool(SERPAPI_KEY),
        "has_IMGBB_KEY": bool(IMGBB_KEY),
        "engine": ENGINE
    })

# ------------------- Upload Reference -------------------
@app.route("/upload", methods=["POST"])
def upload_image():
    if "image" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"error": "No selected file"}), 400
    raw = file.read()
    # Save to disk with unique name and original ext if present
    ext = ".jpg"
    low = file.filename.lower()
    for e in (".jpg", ".jpeg", ".png", ".webp"):
        if low.endswith(e):
            ext = e
            break
    fname = f"{uuid.uuid4().hex}_{os.path.basename(low)}"
    if not any(fname.endswith(e) for e in (".jpg", ".jpeg", ".png", ".webp")):
        fname = f"{uuid.uuid4().hex}{ext}"
    full = os.path.join(UPLOAD_FOLDER, fname)
    with open(full, "wb") as f:
        f.write(raw)
    # Detect faces + embeddings
    embs, boxes = image_bytes_to_embeddings(raw)
    side = {
        "kind": "reference",
        "face_count": len(embs),
        "encodings": [emb.tolist() for emb in embs],
        "boxes": boxes
    }
    save_sidecar(fname, side)
    return jsonify({"filename": fname, "num_faces": len(embs), "face_locations": boxes})

# ------------------- List / Delete -------------------
@app.route("/list", methods=["GET"])
def list_images():
    return jsonify({"images": list_images_only()})

@app.route("/delete", methods=["POST"])
def delete_file():
    data = request.get_json(force=True, silent=True) or {}
    filename = data.get("filename")
    if not filename:
        return jsonify({"error": "filename is required"}), 400
    target = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(target):
        return jsonify({"error": "not found"}), 404
    try:
        os.remove(target)
        side = f"{target}.json"
        if os.path.exists(side):
            os.remove(side)
        return jsonify({"message": f"{filename} deleted"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------- Policies -------------------
@app.route("/policies", methods=["GET", "POST"])
def policies():
    if request.method == "GET":
        return jsonify(load_policies())
    data = request.get_json(force=True, silent=True) or {}
    wl = [s.strip() for s in data.get("whitelist", []) if s.strip()]
    bl = [s.strip() for s in data.get("blacklist", []) if s.strip()]
    save_policies({"whitelist": wl, "blacklist": bl})
    return jsonify({"status": "ok"})

# ------------------- Text Crawl (optional) -------------------
@app.route("/crawl_images", methods=["POST"])
def crawl_images():
    data = request.get_json(force=True, silent=True) or {}
    query = (data.get("query") or "").strip()
    max_images = int(data.get("max_images", 12))
    if not query:
        return jsonify({"error": "query is required"}), 400
    if PROVIDER != "serpapi":
        return jsonify({"error": "Only serpapi is enabled here for text crawl."}), 400
    prov = SerpApiImagesProvider(SERPAPI_KEY)
    try:
        urls = prov.search_text(query, max_images=max_images)
        saved = []
        seen_hash = set()
        for u in urls:
            try:
                if not allowed_url_ext(u): 
                    continue
                b = http_get(u)
                ph = sha1_of_bytes(b)
                if ph in seen_hash: 
                    continue
                seen_hash.add(ph)
                embs, _ = image_bytes_to_embeddings(b)
                meta = {"kind": "result", "source_url": u, "face_count": len(embs), "encodings": []}
                # save file
                ext = infer_ext_from_url(u)
                fname = f"crawl_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}{ext}"
                with open(os.path.join(UPLOAD_FOLDER, fname), "wb") as f:
                    f.write(b)
                save_sidecar(fname, meta)
                saved.append({"filename": fname, "source_url": u, "faces": len(embs)})
            except Exception as e:
                log(f"[Text crawl skip] {u}: {e}")
        return jsonify({"results": saved, "provider": "serpapi"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------- Reverse Crawl (SerpAPI Lens) -------------------
@app.route("/reverse_crawl", methods=["POST"])
def reverse_crawl():
    if ENGINE != "arcface":
        return jsonify({"error": "ArcFace not initialized"}), 500
    if PROVIDER != "serpapi":
        return jsonify({"error": "Reverse crawl requires PROVIDER=serpapi"}), 400
    if not SERPAPI_KEY:
        return jsonify({"error": "SERPAPI_KEY not set."}), 400
    if not IMGBB_KEY:
        return jsonify({"error": "IMGBB_KEY not set."}), 400

    data = request.get_json(force=True, silent=True) or {}
    ref_file = data.get("reference_filename") or latest_reference_filename()
    if not ref_file:
        return jsonify({"error": "No reference image found. Upload one first."}), 400

    # Load reference encodings (from sidecar or recompute)
    ref_path = os.path.join(UPLOAD_FOLDER, ref_file)
    if not os.path.exists(ref_path):
        return jsonify({"error": "Reference file not found."}), 404
    raw_ref = open(ref_path, "rb").read()
    side = read_sidecar(ref_file)
    ref_embs = [np.array(x, dtype=np.float32) for x in side.get("encodings", [])]
    if not ref_embs:
        ref_embs, _ = image_bytes_to_embeddings(raw_ref)
        if not ref_embs:
            return jsonify({"error": "No face detected in reference image."}), 400

    # Upload reference to public URL
    try:
        public_url = upload_to_imgbb(raw_ref)
    except Exception as e:
        return jsonify({"error": f"ImgBB error: {e}"}), 500

    # Fetch candidates
    max_pages  = int(data.get("max_pages", 4))
    max_images = int(data.get("max_images", 30))
    threshold  = float(data.get("threshold", ARCFACE_THRESHOLD))
    smart_q    = (data.get("smart_query") or "").strip()

    lens = SerpApiLensProvider(SERPAPI_KEY)
    urls = lens.search_by_url_paged(public_url, max_pages=max_pages)

    # Optional: text booster
    if smart_q:
        try:
            prov = SerpApiImagesProvider(SERPAPI_KEY)
            urls = list(dict.fromkeys(urls + prov.search_text(smart_q, max_images=max_images)))
        except Exception as e:
            log(f"[Booster] text search failed: {e}")

    # Dedup + per-domain limits + policy filtering
    pol = load_policies()
    wl, bl = pol.get("whitelist", []), pol.get("blacklist", [])
    saved = []
    seen_hash = set()
    per_domain = {}

    count = 0
    for u in urls:
        if count >= max_images:
            break
        try:
            if not allowed_url_ext(u):
                continue
            if is_whitelisted(u, wl):
                log(f"[Skip wl] {u}")
                continue
            dom = urlparse(u).netloc.lower()
            if per_domain.get(dom, 0) >= MAX_PER_DOMAIN:
                continue

            b = http_get(u)
            ph = sha1_of_bytes(b)
            if ph in seen_hash:
                continue
            seen_hash.add(ph)

            embs, _ = image_bytes_to_embeddings(b)
            if not embs:
                log(f"[Skip] No faces: {u}")
                continue
            # score vs ref
            dists = [min(cosine_distance(e, r) for r in ref_embs) for e in embs]
            best = min(dists) if dists else 1.0
            if best > threshold:
                # below match bar
                continue

            # Save
            ext = infer_ext_from_url(u)
            fname = f"crawl_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}{ext}"
            with open(os.path.join(UPLOAD_FOLDER, fname), "wb") as f:
                f.write(b)
            meta = {
                "kind": "result",
                "source_url": u,
                "face_count": len(embs),
                "best_score": best,
                "scores": dists,
                "blacklisted": bool(is_blacklisted(u, bl))
            }
            save_sidecar(fname, meta)
            saved.append({"filename": fname, "source_url": u, "faces": len(embs), "best_score": best})
            per_domain[dom] = per_domain.get(dom, 0) + 1
            count += 1
        except Exception as e:
            log(f"[Skip] {u}: {e}")

    log(f"[Reverse] candidates={len(urls)} saved={len(saved)} ref={ref_file} threshold={threshold}")
    return jsonify({
        "reference": ref_file,
        "public_ref_url": public_url,
        "candidates_count": len(urls),
        "saved": saved,
        "provider": "serpapi_lens",
        "threshold": threshold
    })

# ------------------- DMCA Preview -------------------
@app.route("/dmca/preview", methods=["POST"])
def dmca_preview():
    data = request.get_json(force=True, silent=True) or {}
    full_name = (data.get("full_name") or "").strip()
    email = (data.get("email") or "").strip()
    recipient = (data.get("recipient") or "abuse@host.com").strip()
    urls = data.get("urls") or []
    if not full_name or not email or not urls:
        return jsonify({"error": "full_name, email, and at least one URL are required."}), 400

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

# ------------------- Run (local debug) -------------------
if __name__ == "__main__":
    log(f"[Permission] http://127.0.0.1:5000  | provider={PROVIDER}  | uploads={UPLOAD_FOLDER} | engine={ENGINE}")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=False)
