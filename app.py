from flask import Flask, render_template, request, send_file, url_for, jsonify
from ultralytics import YOLO
import os
from PIL import Image
import csv
import shutil

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
CSV_FOLDER = 'output'

# Create necessary folders
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(CSV_FOLDER, exist_ok=True)

# Load YOLOv8 model
model = YOLO('yolov8_model/best.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    if file:
        # Save uploaded image
        image_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(image_path)

        # Run prediction
        results = model.predict(source=image_path, save=True, save_txt=False, conf=0.25)
        boxes = results[0].boxes
        cls = boxes.cls.cpu().numpy()

        # Count detections
        occupied_count = int(sum(cls == 1))  # Assuming 1 = occupied
        empty_count = int(sum(cls == 0))     # Assuming 0 = empty
        total = occupied_count + empty_count

        # Save CSV result
        csv_path = os.path.join(CSV_FOLDER, 'result.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Total Number of Slots', 'Occupied Slots', 'Available Slots'])
            writer.writerow([total, occupied_count, empty_count])

        # Copy result image to static/results
        predicted_image_path = results[0].save_dir + '/' + os.path.basename(image_path)
        static_result_path = os.path.join(RESULT_FOLDER, 'result.jpg')
        shutil.copy(predicted_image_path, static_result_path)

        # Return JSON response for JavaScript to render dynamically
        return jsonify({
            'result_img': url_for('static', filename='results/result.jpg'),
            'csv_file': url_for('download_csv'),
            'total': total,
            'occupied': occupied_count,
            'available': empty_count
        })

    # If no file was uploaded
    return jsonify({'error': 'No image uploaded'}), 400

@app.route('/download_csv')
def download_csv():
    return send_file(os.path.join(CSV_FOLDER, 'result.csv'), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
