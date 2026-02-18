from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import requests

# -----------------------------------------------------
# INITIAL SETUP
# -----------------------------------------------------

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../saved_model/model.h5')
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static/uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load trained CNN model
model = load_model(MODEL_PATH)

IMG_HEIGHT = model.input_shape[1]
IMG_WIDTH = model.input_shape[2]
print(f"🔥 Model expects input size: {IMG_HEIGHT} x {IMG_WIDTH}")

# Class labels
DATASET_DIR = os.path.join(
    os.path.dirname(__file__),
    '../dataset/garbage_classification/Garbage classification/Garbage classification/train'
)
CLASSES = sorted(os.listdir(DATASET_DIR))

# -----------------------------------------------------
# DETAILED WASTE INFORMATION DATABASE
# -----------------------------------------------------

WASTE_INFO = {
    "cardboard": {
        "type": "Dry Waste",
        "renewable": "✔ Renewable resource (paper pulp)",
        "recyclable": "✔ 100% recyclable",
        "biodegradable": "✔ Decomposes in 2–3 months",
        "examples": ["Shipping boxes", "Cartons", "Paperboard"],
        "how_to_dispose": "Flatten boxes, keep dry, remove tape before recycling.",
        "after_recycling": "Converted into new cardboard, paper bags, egg trays.",
        "environmental_impact": "Recycling saves ~75% energy and reduces deforestation.",
        "did_you_know": "Over 70% of cardboard is recycled globally."
    },

    "glass": {
        "type": "Dry Waste",
        "renewable": "✘ Non-renewable (made from sand)",
        "recyclable": "✔ Infinitely recyclable",
        "biodegradable": "✘ Does not biodegrade",
        "examples": ["Bottles", "Jars", "Glass containers"],
        "how_to_dispose": "Rinse, separate by color, avoid mixing with ceramics.",
        "after_recycling": "Becomes new bottles, tiles, beads, construction material.",
        "environmental_impact": "Glass can take over 1 million years to decompose.",
        "did_you_know": "Recycling one bottle saves enough energy to power a bulb for 4 hours."
    },

    "metal": {
        "type": "Dry Waste",
        "renewable": "✘ Non-renewable (mined from ores)",
        "recyclable": "✔ Recyclable forever",
        "biodegradable": "✘ Takes 100–500 years",
        "examples": ["Aluminium cans", "Tin cans", "Scrap metal"],
        "how_to_dispose": "Rinse cans, crush if possible, keep dry.",
        "after_recycling": "Used in new cans, bicycles, vehicles, construction.",
        "environmental_impact": "Recycling aluminium saves 95% energy.",
        "did_you_know": "Most recycled metal is aluminium."
    },

    "paper": {
        "type": "Dry Waste",
        "renewable": "✔ Renewable (trees)",
        "recyclable": "✔ 5–7 recycling cycles",
        "biodegradable": "✔ Decomposes in weeks",
        "examples": ["Newspapers", "Books", "Office paper"],
        "how_to_dispose": "Keep paper clean and dry; avoid food contamination.",
        "after_recycling": "Becomes tissue paper, cardboard, notebooks.",
        "environmental_impact": "Recycling 1 ton saves 17 trees and 26,000L water.",
        "did_you_know": "Wet paper cannot be recycled."
    },

    "plastic": {
        "type": "Dry Waste",
        "renewable": "✘ Petroleum-based",
        "recyclable": "⚠ Depends on type (1, 2, 5 best)",
        "biodegradable": "✘ Takes 500–1000 years",
        "examples": ["Bottles", "Packaging", "Plastic bags"],
        "how_to_dispose": "Clean, dry, separate by plastic code.",
        "after_recycling": "Turned into fibers, benches, toys, containers.",
        "environmental_impact": "8 million tons enter oceans every year.",
        "did_you_know": "Only ~9% of plastic is recycled globally."
    },

    "trash": {
        "type": "Mixed Waste",
        "renewable": "❓ Depends on item",
        "recyclable": "✘ Not recyclable",
        "biodegradable": "❓ Partially",
        "examples": ["Diapers", "Sanitary waste", "Food wrappers"],
        "how_to_dispose": "Seal properly and dispose in general waste bin.",
        "after_recycling": "Sent to landfill or incineration.",
        "environmental_impact": "Produces methane — 28× stronger than CO₂.",
        "did_you_know": "Proper segregation can reduce landfill waste by 60%."
    }
}

# -----------------------------------------------------
# HOME PAGE
# -----------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html', company_name="EcoSort AI")

# -----------------------------------------------------
# IMAGE PREPROCESSING
# -----------------------------------------------------
def preprocess_image(filepath):
    img = load_img(filepath, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# -----------------------------------------------------
# PREDICTION ROUTE
# -----------------------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')
    if not file or file.filename == '':
        return redirect('/')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    img_array = preprocess_image(filepath)
    preds = model.predict(img_array)

    predicted_class = CLASSES[np.argmax(preds)]
    confidence = round(float(np.max(preds) * 100), 2)

    info = WASTE_INFO.get(predicted_class.lower(), {})

    return render_template(
        'result.html',
        company_name="EcoSort AI",
        filename=filename,
        prediction=predicted_class.capitalize(),
        confidence=confidence,
        info=info
    )

# -----------------------------------------------------
# DISPLAY IMAGE
# -----------------------------------------------------
@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

# -----------------------------------------------------
# ABOUT PAGE
# -----------------------------------------------------
@app.route('/about')
def about():
    return render_template('about.html', company_name="EcoSort AI")

# -----------------------------------------------------
# RECYCLING CENTER FINDER (SAFE OSM)
# -----------------------------------------------------

NOMINATIM_SERVERS = [
    "https://nominatim.openstreetmap.org/search",
    "https://nominatim.openstreetmap.de/search"
]

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "EcoSortAI/1.0 (shiwanipathak317317@gmail.com)"}

@app.route('/centers', methods=['GET', 'POST'])
def centers():
    locations = []
    city = ""
    error = None

    if request.method == 'POST':
        city = request.form.get('city', '').strip()
        if not city:
            error = "Please enter a city name."
            return render_template("centers.html", company_name="EcoSort AI", city=city, locations=[], error=error)

        geo_data = None
        params = {"q": city, "format": "json", "limit": 1}

        for server in NOMINATIM_SERVERS:
            try:
                resp = requests.get(server, params=params, headers=HEADERS, timeout=25)
                if resp.status_code == 200 and resp.text.strip():
                    geo_data = resp.json()
                    if geo_data:
                        break
            except:
                continue

        if not geo_data:
            error = "Location service is busy. Try again later."
            return render_template("centers.html", company_name="EcoSort AI", city=city, locations=[], error=error)

        lat, lon = geo_data[0]["lat"], geo_data[0]["lon"]

        query = f"""
        [out:json][timeout:25];
        (
          node(around:30000,{lat},{lon})["amenity"="recycling"];
          way(around:30000,{lat},{lon})["amenity"="recycling"];
          relation(around:30000,{lat},{lon})["amenity"="recycling"];
        );
        out center;
        """

        try:
            resp = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=30)
            if resp.status_code == 200 and resp.text.strip():
                data = resp.json()
            else:
                raise Exception
        except:
            error = "Recycling center service unavailable."
            return render_template("centers.html", company_name="EcoSort AI", city=city, locations=[], error=error)

        for el in data.get("elements", []):
            tags = el.get("tags", {})
            name = tags.get("name") or "Recycling Center"
            address = tags.get("addr:street") or "Address unavailable"

            if el["type"] == "node":
                lat_c, lon_c = el.get("lat"), el.get("lon")
            else:
                center = el.get("center", {})
                lat_c, lon_c = center.get("lat"), center.get("lon")

            locations.append({
                "name": name,
                "address": address,
                "lat": lat_c,
                "lng": lon_c
            })

        if not locations:
            error = f"No recycling centers found near {city}."

    return render_template("centers.html", company_name="EcoSort AI", city=city, locations=locations, error=error)

# -----------------------------------------------------
# RUN APP
# -----------------------------------------------------
if __name__ == '__main__':
    app.run(debug=True, port=5001)
