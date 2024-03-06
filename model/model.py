import cv2
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import warnings

# -------------- PREPARACAO DO DATASET

path = 'dataset'

diretorio = 'dataset'
arquivos = [os.path.join(diretorio, f) for f in sorted(os.listdir(diretorio))]

imagens = []
classes = []

for imagem_caminho in arquivos:
    try:
        imagem = cv2.imread(imagem_caminho)
        (H, W) = imagem.shape[:2]
    except:
        continue

    imagem = cv2.resize(imagem, (128, 128))

    imagem = imagem.ravel()

    imagens.append(imagem)
    nome_imagem = os.path.basename(os.path.normpath(imagem_caminho))

    if nome_imagem.startswith('n'):
        classe = 0
    else:
        classe = 1

    classes.append(classe)

X = np.asarray(imagens)
y = np.asarray(classes)


# ------------ NORMALIZACAO DOS DADOS

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# ----------- TREINAMENTO E TESTE

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=1)

# ----------- LOGISTIC REGRESSION

warnings.filterwarnings('ignore')

lg = LogisticRegression(C=0.1)
lg.fit(X_treinamento, y_treinamento)

print('Training score: ', lg.score(X_treinamento, y_treinamento))
print('Testing score: ', lg.score(X_teste, y_teste))

joblib_file = "lg_model.sav"
joblib.dump(lg, joblib_file)
