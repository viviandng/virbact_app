import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# Some utilites
import numpy as np
from util import base64_to_pil

from array import array

# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.mobilenet_v2 import MobileNetV2
# model = MobileNetV2(weights='imagenet')

# print('Model loaded. Check http://127.0.0.1:5000/')

# Model saved with Keras model.save()
MODEL_PATH = 'models/Step02a_model.h5'

# Load your own trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img, model):
    # img = cv2.cvtColor(gray_img, cv.CV_GRAY2RGB)
    # tf.image.grayscale_to_rgb(
    # img)
    img = img.resize((224, 224))
    print(type(img))
    # img = array(img).reshape(150528,)

    # Preprocessing the image
    x = image.img_to_array(img)
    x_tensor = tf.convert_to_tensor(x)
    x_converted = tf.image.grayscale_to_rgb(x_tensor)
    x = x_converted.numpy()
    x = np.true_divide(x, 255)
    x = x.reshape(1,-1)

    # x = np.squeeze(x)
    # x = np.expand_dims(x, axis=0)
    # x = np.reshape(x, (224,224,3))

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(x, mode='tf')
    print('This is my shape: ' + str(x.shape))

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)

        # Save the image to ./uploads
        # img.save("./uploads/image.png")

        # Make prediction
        preds = model_predict(img, model)
        print(preds)

        # Process your result for human
        pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        print(pred_proba)

        threshold = .5
        if preds[0] > threshold:
            result = 'bacterial'
        else:
            result = 'viral'

        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        # result = str(pred_class[0][0][1])               # Convert to string
        # result = result.replace('_', ' ').capitalize()
        
        # Serialize the result, you can add additional fields
        return jsonify(result = result, probability=pred_proba)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
