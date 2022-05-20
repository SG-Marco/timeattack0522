from flask import Flask, render_template, jsonify, request, session, redirect, url_for
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')




import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
import cv2

model = load_model('cat_dog_model.h5')

@app.route('/cat_dog')
def process_and_predict():
    file_receive = request.files['file_give']
    im = Image.open(file_receive)
    width, height = im.size
    if width == height:
        im = im.resize((256,256), Image.ANTIALIAS)
    else:
        if width > height:
            left = width/2 - height/2
            right = width/2 + height/2
            top = 0
            bottom = height
            im = im.crop((left,top,right,bottom))
            im = im.resize((256,256), Image.ANTIALIAS)
        else:
            left = 0
            right = width
            top = 0
            bottom = width
            im = im.crop((left,top,right,bottom))
            im = im.resize((256,202560), Image.ANTIALIAS)
            
    ar = np.asarray(im)
    ar = ar.astype('float32')
    ar /= 255.0
    ar = ar.reshape(-1, 256, 256, 3)
    
    cat_dog = model.predict(ar)
    if cat_dog == 0:
        result = '고양이입니다'
    else:
        result = '강아지입니다'
    
    return jsonify({'result' : result})


if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)