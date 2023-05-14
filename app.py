from flask import Flask
from flask import render_template
from flask import request
from flask import url_for
from werkzeug.utils  import secure_filename

from datetime import datetime 

import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import random

from sklearn.cluster import KMeans

import numpy as np

import calendar;
import time;

import os

from tensorflow import keras

# Load the saved model
model = keras.models.load_model('model_checkpoint.h5')








app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/img'





@app.route('/')
def index():

    return render_template('index.html')


# @app.route('/input')
# def input():    
#     return render_template('input.html')


@app.route('/predict', methods=['POST'])
def process():

    name2= str(datetime.now().year) + str(datetime.now().month) + str(datetime.now().day)+'-'+str(datetime.now().hour)+str(datetime.now().minute)+str(datetime.now().second)
    name=name2+'.jpg'
    photo = request.files['img']
    path = os.path.join(app.config['UPLOAD_FOLDER'],name)
    photo.save(path)


    
    image_path = path
    image = keras.preprocessing.image.load_img(image_path, target_size=(112, 112))

    # Convert the image to a numpy array
    input_array = keras.preprocessing.image.img_to_array(image)
    input_array = np.expand_dims(input_array, axis=0)

    # Scale the input array
    input_array /= 255.0

    # Make a prediction using the model
    predictions = model.predict(input_array)

    # Print the predictions
    print(predictions)

    predicted_class = np.argmax(predictions)
    print('Predicted class:', predicted_class)
    

    return render_template("result.html",img=name,val=predicted_class)


# @app.errorhandler(404)
# def page_not_found(error):
#     return render_template('404.html'), 404

