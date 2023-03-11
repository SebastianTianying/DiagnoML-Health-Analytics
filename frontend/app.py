from flask import Flask, flash, redirect, request, render_template, url_for
from werkzeug.utils import secure_filename
import os
import pickle

ALLOWED_EXTENSIONS = ['png', 'jpg', 'jpeg']
UPLOAD_FOLDER = './static/images'


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_app():
    app = Flask(__name__)
    app.secret_key = os.urandom(12)
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