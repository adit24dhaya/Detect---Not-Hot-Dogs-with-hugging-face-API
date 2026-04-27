#web.py

from flask import Flask, render_template, request, jsonify
import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("HUGGING_FACE_API_URL")
API_KEY = os.getenv("HUGGING_FACE_API_KEY")
headers = {'Authorization': f'Bearer {API_KEY}'} if API_KEY else {}
ALLOWED_MIME_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5MB


def normalize_label(lbl: str) -> str:
    return lbl.strip().lower().replace(" ", "_").replace("-", "_")


def query_image(image_bytes, content_type: str):
    request_headers = dict(headers)
    request_headers["Content-Type"] = content_type or "application/octet-stream"
    r = requests.post(API_URL, headers=request_headers, data=image_bytes, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"HuggingFace inference error: HTTP {r.status_code} - {r.text}")
    payload = r.json()
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(payload["error"])
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected model response format.")
    return payload


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "configured": bool(API_URL and API_KEY),
            "max_upload_mb": 5,
        }
    )

@app.route("/")
def index():
    return render_template("index.html")  # don't use "./"

@app.route("/upload", methods=["POST"])

def upload():
    f = request.files.get("file1")
    if not f:
        return jsonify({"error": "file1 missing"}), 400
    if not API_URL or not API_KEY:
        return jsonify({"error": "Server missing Hugging Face API configuration"}), 500
    if f.mimetype not in ALLOWED_MIME_TYPES:
        return jsonify({"error": f"Unsupported file type: {f.mimetype}"}), 400

    img_bytes = f.read()
    if not img_bytes:
        return jsonify({"error": "Empty file"}), 400
    if len(img_bytes) > app.config["MAX_CONTENT_LENGTH"]:
        return jsonify({"error": "File too large (max 5MB)"}), 400

    try:
        preds = query_image(img_bytes, f.mimetype or "application/octet-stream")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 502

    preds = sorted(preds, key=lambda x: x.get("score", 0), reverse=True)
    if not preds:
        return jsonify({"error": "Model returned no predictions"}), 502
    best = preds[0]

    HOTDOG_LABELS = {"hot_dog", "hotdog"}

    # Decide binary label
    binary_label = "hot dog" if normalize_label(best["label"]) in HOTDOG_LABELS else "not hot dog"

    return jsonify({
        "label": binary_label,                 # hot dog / not hot dog
        "confidence": round(best["score"], 4), # confidence of top prediction
        "top_class": best["label"],            # model’s actual top class (pizza, burger, etc.)
        "top3": preds[:3],
        "raw": preds
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=81)