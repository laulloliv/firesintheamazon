import os
import time
from flask import render_template, Blueprint, request, redirect, url_for

from werkzeug.utils import secure_filename

from app.utils.processing import predict


routes = Blueprint('index', __name__)


@routes.route('/')
def home():
    return render_template('home.html')


@routes.route('/lab')
def lab():
    return render_template('lab.html')


@routes.route('/upload', methods=["POST"])
def upload_file():
    file = request.files['imageFile']
    if file:
        filename = secure_filename(file.filename)
        savepath = os.path.join(os.getcwd(), "app",
                                "static", "upload", filename)
        file.save(savepath)
        while not os.path.exists(savepath):
            time.sleep(1)
        return redirect(url_for("index.predict_img", filepath=savepath))
    return "Nenhum arquivo enviado."


@routes.route("/predict")
def predict_img():
    filepath = request.args.get("filepath")
    result = predict(filepath)
    if type(result) == dict:
        return render_template('predict.html', result=result)
    return render_template('lab.html', result=result, error=0)
