import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
import numpy as np
from PIL import Image
import io
import requests
from pymongo import MongoClient

app = Flask(__name__)

# ------------------ MODEL (lazy load) ------------------
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

# ------------------ DATABASE (lazy load) ------------------
client = None
foods_collection = None

def get_db():
    global client, foods_collection
    if client is None:
        client = MongoClient(os.environ.get("MONGO_URI"))
        db = client["nutrition_db"]
        foods_collection = db["foods"]
    return foods_collection

# ------------------ USDA NUTRITION ------------------
USDA_API_KEY = os.environ.get("USDA_API_KEY")

def get_nutrition(food_name):
    foods = get_db()
    food_key = food_name.lower()

    # 1️⃣ Check cache (exclude _id)
    cached = foods.find_one(
        {"name": food_key},
        {"_id": 0}
    )
    if cached:
        cached["source"] = "cache"
        return cached

    # 2️⃣ Call USDA API
    search_url = "https://api.nal.usda.gov/fdc/v1/foods/search"
    params = {
        "query": food_name,
        "pageSize": 1,
        "api_key": USDA_API_KEY
    }

    response = requests.get(search_url, params=params)
    data = response.json()

    if "foods" not in data or len(data["foods"]) == 0:
        return {"error": "Nutrition data not found"}

    food = data["foods"][0]

    nutrients = {
        n["nutrientName"]: n["value"]
        for n in food.get("foodNutrients", [])
    }

    nutrition = {
        "name": food_key,
        "calories": nutrients.get("Energy", 0),
        "protein": nutrients.get("Protein", 0),
        "fat": nutrients.get("Total lipid (fat)", 0),
        "carbs": nutrients.get("Carbohydrate, by difference", 0)
    }

    # 3️⃣ Store clean document
    foods.insert_one(nutrition)

    # 4️⃣ Return clean JSON-safe object
    nutrition["source"] = "usda"
    return nutrition

# ------------------ ROUTES ------------------
@app.route("/")
def index():
    return jsonify({"status": "API running"})

@app.route("/db-test")
def db_test():
    foods = get_db()
    return {
        "status": "connected",
        "documents": foods.count_documents({})
    }

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

    label = decoded[1].replace("_", " ").title()
    confidence = float(decoded[2])

    nutrition = get_nutrition(label)

    return jsonify({
        "label": label,
        "confidence": confidence,
        "nutrition": nutrition
    })