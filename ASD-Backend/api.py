from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import cv2
import sys
from werkzeug.utils import secure_filename

# Add project directory to path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append("/home/ec2-user/CILab-Autism-Spectrum-Disorder-main")
from test_main import process_image

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB limit

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Check if image exists in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Secure filename and save
        filename = secure_filename(file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(image_path)

        # Process image using original function
        annotated_image, final_decision = process_image(image_path, RESULTS_FOLDER)

        # Save annotated image temporarily
        temp_image_path = os.path.join(RESULTS_FOLDER, f"annotated_{filename}")
        cv2.imwrite(temp_image_path, annotated_image)

        # Convert to base64 for API Gateway compatibility
        with open(temp_image_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')

        # Clean up files
        os.remove(image_path)
        os.remove(temp_image_path)

        return jsonify({
            'output': final_decision,
            'result_image': img_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
