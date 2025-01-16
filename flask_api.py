from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
from ultralytics import YOLO
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# YOLOv8 modelini yükleyin
model = YOLO('yolov8x.pt')

@app.route('/')
def index():
    return jsonify({"message": "Welcome to the YOLO Detection API"})

@app.route('/detect-image', methods=['POST'])
def detect_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    try:
        # Görüntü dosyasını kaydedin
        image_file = request.files['image']
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)

        # Görüntüyü işleyin
        image = cv2.imread(image_path)

        if image is None:
            return jsonify({"error": "Invalid image format or image not readable"}), 400

        # YOLO modeli ile tespit yapın
        results = model(image)

        # Sonuçları görsel olarak işleyin ve kaydedin
        output_image_path = os.path.join(OUTPUT_FOLDER, 'output_yolov8_image.jpg')
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                color = (0, 255, 0)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite(output_image_path, image)

        # İşlenmiş görüntü dosyasının var olup olmadığını kontrol edin ve gönderin
        if os.path.exists(output_image_path):
            return send_file(output_image_path, mimetype='image/jpeg')
        else:
            return jsonify({"error": "Processed image not found after saving"}), 404

    except Exception as e:
        return jsonify({"error": f"An error occurred during processing: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
