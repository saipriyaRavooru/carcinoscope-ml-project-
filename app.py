from flask import Flask, render_template, request, redirect, url_for, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import os

app = Flask(__name__)
app.secret_key = 'carcinoscope_secret_key'

# ------------------- Model Setup -------------------
MODEL_PATH = "skin_cancer_cnn_model.h5"
model = load_model(MODEL_PATH)
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

UPLOAD_FOLDER = os.path.join('static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Simple in-memory user store (you can replace with DB later)
users = {}

# ------------------- Routes -------------------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """User signup"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users:
            return render_template('signup.html', error="Username already exists!")
        users[username] = password
        return redirect(url_for('login'))

    return render_template('signup.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            session['user'] = username
            return redirect(url_for('detect'))
        else:
            return render_template('login.html', error="Invalid username or password.")

    return render_template('login.html')


@app.route('/logout')
def logout():
    """Logout user"""
    session.pop('user', None)
    return redirect(url_for('home'))


@app.route('/detect', methods=['GET'])
def detect():
    """Detection page (requires login)"""
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('detect.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle upload and prediction"""
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Preprocess
    img = image.load_img(file_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Predict
    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    class_name = class_names[class_idx]
    confidence = float(np.max(preds[0]) * 100)

    cancerous_classes = ['akiec', 'bcc', 'mel']
    if class_name in cancerous_classes:
        status = "⚠️ Cancerous lesion detected"
    else:
        status = "✅ Benign (non-cancerous) lesion"

    return render_template(
        'detect.html',
        prediction=class_name,
        confidence=round(confidence, 2),
        status=status,
        image_path=file_path
    )


if __name__ == '__main__':
    app.run(debug=True)
