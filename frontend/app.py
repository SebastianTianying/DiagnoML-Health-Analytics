from flask import Flask, flash, redirect, request, render_template, url_for
from werkzeug.utils import secure_filename
import os
import pickle
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import coremltools as ct

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
UPLOAD_FOLDER = './static/images'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def run_model():
    new_model = tf.keras.models.load_model('../saved_model/resnet50_model')
    # new_model = pickle.load(open("./models/resnet50-model.pkl", "rb"))
    path = './static/images/BACTERIA-test.jpeg'
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = new_model.predict(images, batch_size=10)
    print("classes: ", classes)

    if classes[0] < 0.5:
        return ("image shows PNEUMONIA")
    else:
        return ("image shows NORMAL")


def create_app():
    app = Flask(__name__)
    app.secret_key = '2f44a4573531a78be7924acc'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    @app.route("/", methods=["POST", "GET"])
    def home():
        if (request.method == "POST"):
            if 'file' not in request.files:
                flash('No file part')
                return redirect(request.url)
            file = request.files['file']
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                flash(filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                print(run_model())
                # return redirect(url_for('download_file', name=filename))
                return redirect(request.url)
            
        return render_template("home.html")
    
    @app.route("/login", methods=["GET", "POST"])
    def login():
        return render_template("login.html")
    
    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        return render_template("signup.html")
    
    return app