# -*- coding: utf-8 -*-
"""
Created on Wed Dec 27 23:49:18 2023

@author: User
"""

from flask import Flask, render_template, request
from tensorflow.keras.utils import load_img
import numpy as np
from keras.models import load_model
app = Flask(__name__)
model = load_model('D:/My projects/Forest fire detection/firemodelnew.h5')

@app.route('/', methods=['GET'])
def hello_word():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile= request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    img = load_img(image_path, target_size=(128, 128))
    img = np.array(img)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    #img = preprocess_input(img)
    yhat = model.predict(img)
    #label = decode_predictions(yhat)
    if (yhat<0.5):
        label='Not fire'
    else:
        label='Fire'  
    return render_template('index.html', prediction=label)


if __name__ == '__main__':
    app.run(debug=True)