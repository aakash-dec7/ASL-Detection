import io
import base64
from PIL import Image
from Logger import logger
from src.s5_inference import Inference
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
inference = Inference()


@app.route("/", methods=["GET"])
def home_page():
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        image_data = data.get("image")

        if not image_data:
            raise ValueError("No image data provided")

        # Decode base64 string to PIL image
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Predict label
        prediction = inference.predict(image)

        return jsonify({"label": prediction})

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return jsonify({"label": "An error occurred. Please try again."}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3000, debug=True)
