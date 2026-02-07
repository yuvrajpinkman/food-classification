from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Load model once (VERY IMPORTANT)
print("Loading MobileNetV3...")
model = tf.keras.applications.MobileNetV3Small(weights="imagenet")
print("Model loaded.")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]

    img = Image.open(io.BytesIO(image.read())).convert("RGB")
    img = img.resize((224, 224))
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v3.preprocess_input(img)

    preds = model.predict(img)
    decoded = tf.keras.applications.mobilenet_v3.decode_predictions(preds, top=1)[0][0]

    return jsonify({
        "label": decoded[1].replace("_", " ").title(),
        "confidence": float(decoded[2])
    })

if __name__ == "__main__":
    app.run()