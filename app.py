import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import numpy as np
from PIL import Image
import io
from pymongo import MongoClient

app = Flask(__name__)

# ---------- MODEL (lazy load) ----------
model = None

def get_model():
    global model
    if model is None:
        print("Loading MobileNetV2 model...")
        model = MobileNetV2(
            weights="imagenet",
            input_shape=(160, 160, 3)
        )
    return model

# ---------- DATABASE (lazy load) ----------
client = None
foods_collection = None

def get_db():
    global client, foods_collection
    if client is None:
        client = MongoClient(os.environ.get("MONGO_URI"))
        db = client["nutrition_db"]
        foods_collection = db["foods"]
    return foods_collection

# ---------- ROUTES ----------
@app.route("/")
def index():
    return jsonify({"status": "API running"})

@app.route("/predict", methods=["POST"])
def predict():
    model = get_model()

    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]
    img = Image.open(io.BytesIO(image.read())).convert("RGB")
    img = img.resize((160, 160))
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)

    preds = model.predict(img)
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=1)[0][0]

    return jsonify({
        "label": decoded[1].replace("_", " ").title(),
        "confidence": float(decoded[2])
    })

@app.route("/db-test")
def db_test():
    foods = get_db()
    count = foods.count_documents({})
    return {"status": "connected", "documents": count}

@app.route("/db-insert-test")
def db_insert_test():
    foods = get_db()
    foods.insert_one({
        "name": "Ice Cream",
        "calories": 207,
        "protein": 3.5,
        "fat": 11,
        "source": "manual-test"
    })
    return {"status": "inserted"}