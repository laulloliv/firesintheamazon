from fileinput import filename
import os
from cv2 import transform
from flask import Flask, render_template, request
import cv2
import joblib
from sklearn.preprocessing import MinMaxScaler
from werkzeug.utils import secure_filename


app = Flask(__name__, static_folder='static')
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'upload')

joblib_file = "model/lg_model.sav"
joblib_model = joblib.load(joblib_file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/lab')
def lab():
    return render_template('lab.html')


@app.route('/predict', methods=["POST"])
def upload():
    file = request.files['imageFile']
    savePath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    file.save(savePath)

    img_file = "upload/" + file.filename
    imgF = 0

    if img_file:
        imgF = 1
    else:
        imgF = 0

    img = cv2.imread(img_file)
    img = cv2.resize(img, (128, 128))

    img = img.ravel()
    scaler = MinMaxScaler()
    img = [img]
    img = scaler.fit_transform(img)
    img = scaler.inverse_transform(img.reshape(1, -1))

    cls = joblib_model.predict(img)[0]
    if cls == 0:
        return render_template('lab.html', cls=cls, imgF=imgF)
    else:
        return render_template('lab.html', cls=cls, imgF=imgF)


if __name__ == '__main__':
    app.run(debug=True)
