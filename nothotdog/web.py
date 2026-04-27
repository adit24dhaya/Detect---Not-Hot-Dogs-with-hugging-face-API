from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
import os
import threading
import time
import uuid

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()
API_URL = os.getenv("HUGGING_FACE_API_URL")
API_KEY = os.getenv("HUGGING_FACE_API_KEY")
REQUEST_HEADERS = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
HOTDOG_LABELS = {"hot_dog", "hotdog", "hot dog"}
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5MB
CACHE_SIZE = 128

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_cache = OrderedDict()
_lock = threading.Lock()
_stats = {
    "requests_total": 0,
    "inference_calls": 0,
    "cache_hits": 0,
    "errors": 0,
}


@dataclass
class PredictionResult:
    label: str
    confidence: float
    top_class: str
    top3: list
    raw: list
    cached: bool
    latency_ms: int
    request_id: str


def normalize_label(lbl: str) -> str:
    return lbl.strip().lower().replace(" ", "_").replace("-", "_")


def _session() -> requests.Session:
    retry = Retry(
        total=2,
        connect=2,
        read=2,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def query_image(image_bytes: bytes, content_type: str):
    request_headers = dict(REQUEST_HEADERS)
    request_headers["Content-Type"] = content_type or "application/octet-stream"
    with _session() as s:
        r = s.post(API_URL, headers=request_headers, data=image_bytes, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"HuggingFace inference error: HTTP {r.status_code} - {r.text}")
    payload = r.json()
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(payload["error"])
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected model response format.")
    return payload


def _cache_get(key: str):
    with _lock:
        if key not in _cache:
            return None
        _cache.move_to_end(key)
        _stats["cache_hits"] += 1
        return _cache[key]


def _cache_set(key: str, value: dict):
    with _lock:
        _cache[key] = value
        _cache.move_to_end(key)
        if len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)


def classify_image_bytes(image_bytes: bytes, content_type: str) -> PredictionResult:
    request_id = str(uuid.uuid4())
    start = time.time()
    key = sha256(image_bytes).hexdigest()

    cached_payload = _cache_get(key)
    if cached_payload is not None:
        latency = int((time.time() - start) * 1000)
        return PredictionResult(
            **cached_payload,
            cached=True,
            latency_ms=latency,
            request_id=request_id,
        )

    preds = query_image(image_bytes, content_type)
    preds = sorted(preds, key=lambda x: x.get("score", 0), reverse=True)
    if not preds:
        raise RuntimeError("Model returned no predictions")
    best = preds[0]
    binary_label = "hot dog" if normalize_label(best["label"]) in HOTDOG_LABELS else "not hot dog"
    latency = int((time.time() - start) * 1000)

    payload = {
        "label": binary_label,
        "confidence": round(float(best.get("score", 0.0)), 4),
        "top_class": best.get("label", ""),
        "top3": preds[:3],
        "raw": preds,
    }
    _cache_set(key, payload)
    return PredictionResult(**payload, cached=False, latency_ms=latency, request_id=request_id)


def _validate_request_file(f):
    if not f:
        return "file1 missing", 400
    if not API_URL or not API_KEY:
        return "Server missing Hugging Face API configuration", 500
    if f.mimetype not in ALLOWED_MIME_TYPES:
        return f"Unsupported file type: {f.mimetype}", 400
    img_bytes = f.read()
    if not img_bytes:
        return "Empty file", 400
    if len(img_bytes) > MAX_UPLOAD_BYTES:
        return "File too large (max 5MB)", 400
    return img_bytes, 200


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "configured": bool(API_URL and API_KEY),
            "max_upload_mb": 5,
            "cache_size": CACHE_SIZE,
            "stats": dict(_stats),
        }
    )


@app.route("/metrics")
def metrics():
    return jsonify(dict(_stats))


@app.route("/upload", methods=["POST"])
def upload():
    with _lock:
        _stats["requests_total"] += 1
    f = request.files.get("file1")
    validated, status = _validate_request_file(f)
    if status != 200:
        with _lock:
            _stats["errors"] += 1
        return jsonify({"error": validated}), status
    try:
        with _lock:
            _stats["inference_calls"] += 1
        result = classify_image_bytes(validated, f.mimetype or "application/octet-stream")
        return jsonify(
            {
                "label": result.label,
                "confidence": result.confidence,
                "top_class": result.top_class,
                "top3": result.top3,
                "raw": result.raw,
                "cached": result.cached,
                "latency_ms": result.latency_ms,
                "request_id": result.request_id,
            }
        )
    except Exception as exc:
        with _lock:
            _stats["errors"] += 1
        return jsonify({"error": str(exc)}), 502


@app.route("/upload-batch", methods=["POST"])
def upload_batch():
    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "files[] missing"}), 400
    results = []
    for f in files[:10]:
        validated, status = _validate_request_file(f)
        if status != 200:
            results.append({"filename": f.filename, "error": validated})
            continue
        try:
            with _lock:
                _stats["requests_total"] += 1
                _stats["inference_calls"] += 1
            out = classify_image_bytes(validated, f.mimetype or "application/octet-stream")
            results.append(
                {
                    "filename": f.filename,
                    "label": out.label,
                    "confidence": out.confidence,
                    "top_class": out.top_class,
                    "cached": out.cached,
                    "latency_ms": out.latency_ms,
                }
            )
        except Exception as exc:
            with _lock:
                _stats["errors"] += 1
            results.append({"filename": f.filename, "error": str(exc)})
    return jsonify({"count": len(results), "results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=81)