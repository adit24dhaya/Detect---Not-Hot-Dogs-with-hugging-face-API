#web.py

from flask import Flask, render_template, request, jsonify
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("HUGGING_FACE_API_URL")
headers = {'Authorization': f'Bearer {os.getenv("HUGGING_FACE_API_KEY")}'}

app = Flask(__name__)
def query_image(image_bytes):
    r = requests.post(API_URL, headers=headers, data=image_bytes, timeout=60)
    return r.json()  # same as json.loads(r.content)

@app.route("/")
def index():
    return render_template("index.html")  # don't use "./"

@app.route("/upload", methods=["POST"])

def upload():
    f = request.files.get("file1")
    if not f:
        return jsonify({"error": "file1 missing"}), 400

    img_bytes = f.read()
    r = requests.post(
        API_URL,
        headers={
            "Authorization": f"Bearer " + os.getenv("HUGGING_FACE_API_KEY"),
            "Content-Type": f.mimetype or "application/octet-stream",
        },
        data=img_bytes,
        timeout=60,
    )
    preds = r.json()
    preds = sorted(preds, key=lambda x: x["score"], reverse=True)
    best = preds[0]

    def norm(lbl: str) -> str:
        return lbl.strip().lower().replace(" ", "_").replace("-", "_")

    HOTDOG_LABELS = {"hot_dog", "hotdog"}

    # Decide binary label
    binary_label = "hot dog" if norm(best["label"]) in HOTDOG_LABELS else "not hot dog"

    return jsonify({
        "label": binary_label,                 # hot dog / not hot dog
        "confidence": round(best["score"], 4), # confidence of top prediction
        "top_class": best["label"],            # model’s actual top class (pizza, burger, etc.)
        "raw": preds
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=81)