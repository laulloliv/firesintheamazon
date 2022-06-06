import os

import cv2
import joblib
import zipfile
from sklearn.preprocessing import MinMaxScaler


def preprocessing(image):
    scaler = MinMaxScaler()
    img = cv2.imread(image)
    img = cv2.resize(img, (128, 128))
    img = img.ravel()
    img = scaler.fit_transform([img])
    img = scaler.inverse_transform(img.reshape(1, -1))
    return img


def predict(payload):
    type = os.path.splitext(payload)[1]
    if type == ".zip":
        result = manyfiles(payload)
    else:
        result = onefile(payload)
    return result


def onefile(payload):
    img = preprocessing(payload)
    joblib_model = joblib.load("app/lib/lg_model.sav")
    result = joblib_model.predict(img)[0]
    return result


def manyfiles(payload):
    filespath = os.path.join(os.getcwd(), "app", "static", "upload")
    with zipfile.ZipFile(payload, 'r') as zip_ref:
        zip_ref.extractall(filespath)
    os.remove(payload)

    queimadas = []
    files = os.listdir(filespath)

    for img in files:
        imgpath = os.path.join(filespath, img)
        result = onefile(imgpath)
        if result != 0:
            queimadas.append(img)

    return {"result": queimadas, "nfiles": len(files)}
