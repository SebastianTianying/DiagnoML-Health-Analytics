from flask import Flask, flash, redirect, request, render_template
from werkzeug.utils import secure_filename
import os
import pickle

def create_app():
    app = Flask(__name__)
    app.secret_key = '2f44a4573531a78be7924acc'

    @app.route("/", methods=["POST", "GET"])
    def home():
        if (request.method == "POST"):
            flash("!!!!!!!!!!!!")
            # if 'file' not in request.files:
            #     flash('No file part')
            #     return redirect(request.url)
            # file = request.files['file']
            # if file.filename == '':
            #     flash('No selected file')
            #     return redirect(request.url)
            # if file: # also check if the file is allowed
            #     filename = secure_filename(file.filename)
            #     flash(filename)
            #     # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            #     # with open(os.path.join(app.config['UPLOAD_FOLDER'], filename)) as f:
            #     #     model = pickle.load(open("../static/resnet50-model.pkl"))
            #     #     flash(model.predict(f))
        return render_template("home.html")
    
    @app.route("/login", methods=["GET", "POST"])
    def login():
        return render_template("login.html")
    
    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        return render_template("signup.html")
    
    return app