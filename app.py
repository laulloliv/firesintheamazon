from fileinput import filename
import os
import zipfile
from cv2 import transform
from flask import Flask, render_template, request
import cv2
import joblib
import numpy as np
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


@app.route('/predict-lab')
def predictFires():
    return render_template('predict.html')


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

    tam = len(img_file)
    zipFile = img_file[:tam-3]

    if img_file[tam-3:] == "zip":

        path = img_file
        zip_object = zipfile.ZipFile(file=path, mode='r')
        zip_object.extractall('./static/upload')
        zip_object.close()

        names = []
        queimadas = []

        diretorio = './static/' + zipFile

        files = [os.path.join(diretorio, f)
                 for f in sorted(os.listdir(diretorio))]

        for img in files:

            image = cv2.imread(img)
            image = cv2.resize(image, (128, 128))
            image = image.ravel()

            scaler = MinMaxScaler()
            image = [image]
            img_test = scaler.fit_transform(image)
            img_test = scaler.inverse_transform(img_test.reshape(1, -1))

            cls = joblib_model.predict(img_test)[0]

            if cls != 0:
                x = img.find(".", 3)
                iName = img[x+2:]
                names.append(iName)
                queimadas.append(img)

        try:
            source = 'static/' + zipFile
            dest = 'static/upload/Files'
            os.rename(source, dest)
        except:
            pass

        return render_template('predict.html', qtdFocos=len(queimadas), queimadas=queimadas, names=names)

    else:

        img = cv2.imread(img_file)
        try:
            img = cv2.resize(img, (128, 128))
        except:
            return render_template('lab.html', cls=0, imgF=0, error=1)

        img = img.ravel()
        scaler = MinMaxScaler()
        img = [img]
        img = scaler.fit_transform(img)
        img = scaler.inverse_transform(img.reshape(1, -1))

        cls = joblib_model.predict(img)[0]
        if cls == 0:
            return render_template('lab.html', cls=cls, imgF=imgF, error=0)
        else:
            return render_template('lab.html', cls=cls, imgF=imgF, error=0)


if __name__ == '__main__':
    app.run(debug=True)
