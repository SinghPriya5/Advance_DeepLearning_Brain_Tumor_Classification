from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model('model2.h5')

# Class names as per your model's training
class_name = ['nontumorous', 'tumorous']

# Image processing constants
IMAGE_SIZE = 240  # Corrected to match model input size
UPLOAD_FOLDER = 'static'

# Ensure the static folder exists
if not os.path.exists(UPLOAD_FOLDER):gi
    os.makedirs(UPLOAD_FOLDER)

# Function to preprocess and predict
def predict(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Add batch dimension (1, 240, 240, 3)

    predictions = model.predict(img_array)

    predicted_class = class_name[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence

# Allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route to the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file uploaded.')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', message='No file selected.')

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            file.save(filepath)

            # Predict the disease
            predicted_class, confidence = predict(filepath)

            return render_template('index.html',
                                   image_path=filepath,
                                   actual_label=predicted_class,
                                   predicted_label=predicted_class,
                                   confidence=confidence)

        else:
            return render_template('index.html', message='Invalid file type. Allowed: png, jpg, jpeg')

    # Initial page load
    return render_template('index.html', message='Upload an image')

if __name__ == '__main__':
    app.run(debug=True)
