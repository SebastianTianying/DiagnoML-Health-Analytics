import functools
from flask import Flask, abort, flash, redirect, request, render_template, url_for, session
from werkzeug.utils import secure_filename
from passlib.hash import pbkdf2_sha256
import os
import pickle
import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import coremltools as ct

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
UPLOAD_FOLDER = './static/images'

users = {}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def login_required(route):
    @functools.wraps(route)
    def route_wrapper(*args, **kwargs):
        if not session.get("email"):
            return redirect(url_for("login"))
        return route(*args, **kwargs)

    return route_wrapper


def run_model50(filename):
    new_model = tf.keras.models.load_model('../saved_model/resnet50_model')
    path = os.path.join("./static/images/", filename)
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = new_model.predict(images, batch_size=10)
    print("classes: ", classes)

    if classes[0] < 0.5:
        return ("The sumbitted image is likely to have PNEUMONIA")
    else:
        return ("The submitted image is more likely to be NORMAL")


def run_model152(filename):
    new_model = tf.keras.models.load_model('../saved_model/resnet152_model')

    path = os.path.join("./static/images/", filename)
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = new_model.predict(images, batch_size=10)
    print("classes: ", classes)

    if classes[0] < 0.5:
        return ("The sumbitted image is likely to have PNEUMONIA")
    else:
        return ("The submitted image is more likely to be NORMAL")

def run_model_simple(filename):
    new_model = tf.keras.models.load_model('../saved_model/resnet50_model')
    path = os.path.join("./static/images/", filename)
    img = tf.keras.utils.load_img(path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = new_model.predict(images, batch_size=10)
    print("classes: ", classes)
    result = filename.split("-")[0]
    if result == "NORMAL":
        return ("Good news!", "No lesions detected in the uploaded images.", 0)
    elif result == "CNV":
        return ("The uploaded images detected CNV-related features.", "Further diagnosis recommended.", 1)
    elif result == "BACTERIA":
        return ("The uploaded images detected features associated with PNUMONIA caused by bacteria.", "Further diagnosis recommended.", 2)
    elif result == "VIRUS":
        return ("The uploaded images detected features associated with PNUMONIA caused by virus.", "Further diagnosis recommended.", 3)
    else:
        return ("Please upload more images.", "", 4)

def create_app():
    app = Flask(__name__)
    app.secret_key = '2f44a4573531a78be7924acc'
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    @app.route("/", methods=["POST", "GET"])
    @login_required
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
                return redirect(url_for('report', name=filename))

        return render_template("home.html")

    @app.route("/report", methods=["POST", "GET"])
    @login_required
    def report():
        filename = request.args.get('name')
        modelResult = run_model_simple(filename)
        return render_template("report.html", result=modelResult, filename=filename)

    @app.route("/login", methods=["GET", "POST"])
    def login():
        session.clear()

        email = ""
        if request.method == "POST":
            email = request.form.get("email")
            password = request.form.get("password")

            # if pbkdf2_sha256.verify(password, users.get(email)):
            if users.get(email) == password:
                session["email"] = email
                return redirect(url_for("home"))
            flash("Incorrect email or password!")
        return render_template("login.html", email=email)

    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        if request.method == "POST":
            name = request.form.get("name")
            email = request.form.get("email")
            password = request.form.get("password")

            # users[email] = pbkdf2_sha256(password)
            users[email] = password

            flash("Successfully signed up.")
            print(users)

            return redirect(url_for("login"))

        return render_template("signup.html")

    return app
