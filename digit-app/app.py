import base64
import binascii
import io
import json
from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, UnidentifiedImageError

from digit_preprocessing import preprocess_pixels


BASE_DIR = Path(__file__).resolve().parent
MAX_IMAGE_BYTES = 2 * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096
CONSENSUS_MODEL = "Majority vote (experimental)"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 4 * 1024 * 1024

models = joblib.load(BASE_DIR / "models.joblib")
with (BASE_DIR / "metrics.json").open(encoding="utf-8") as metrics_file:
    meta = json.load(metrics_file)


def preprocess(data_url):
    prefix = "data:image/png;base64,"
    if not isinstance(data_url, str) or not data_url.startswith(prefix):
        raise ValueError("Expected a PNG data URL.")

    try:
        raw = base64.b64decode(data_url[len(prefix):], validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Invalid base64 image data.") from error

    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Image is empty or too large.")

    try:
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != "PNG":
                raise ValueError("Expected a PNG image.")
            if image.width > MAX_IMAGE_DIMENSION or image.height > MAX_IMAGE_DIMENSION:
                raise ValueError("Image dimensions are too large.")
            image.load()
            grayscale = image.convert("L")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise ValueError("Could not decode the PNG image.") from error

    pixels = 255 - np.asarray(grayscale, dtype=np.uint8)
    features = preprocess_pixels(pixels)
    if features is None or np.count_nonzero(pixels > 40) < 30:
        return None
    return features.reshape(1, -1)


@app.route("/")
def home():
    return render_template("index.html", meta=meta)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or "image" not in data:
        return jsonify(error="No image received"), 400
    selected_model = data.get(
        "model", meta.get("default_model", meta["best_model"])
    )
    if not isinstance(selected_model, str) or (
        selected_model != CONSENSUS_MODEL and selected_model not in models
    ):
        return jsonify(error="Unknown model selected"), 400

    try:
        features = preprocess(data["image"])
    except ValueError:
        return jsonify(error="Could not read the image"), 400

    if features is None:
        return jsonify(error="Please draw a digit first"), 400

    if selected_model == CONSENSUS_MODEL:
        votes = np.zeros(10, dtype=float)
        for classifier in models.values():
            votes[int(classifier.predict(features)[0])] += 1
        probabilities = votes / len(models)
    else:
        probabilities = models[selected_model].predict_proba(features)[0]
    return jsonify(
        prediction=int(np.argmax(probabilities)),
        probabilities=[round(float(probability) * 100, 1) for probability in probabilities],
        pixels=np.rint(features[0].reshape(8, 8)).astype(int).tolist(),
        model=selected_model,
    )


@app.route("/health")
def health():
    return jsonify(status="ok")


if __name__ == "__main__":
    app.run()
