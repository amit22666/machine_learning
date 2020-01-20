#!/usr/bin/env python
# coding: utf-8

# In[1]:
from __future__ import division, print_function

import sys

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

# coding=utf-8
#import sys
import os
import glob
import re
import numpy as np
import cv2
import requests
import io
import json




# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer


# In[4]:


# Define a flask app
app = Flask(__name__)



print('Model loaded. Check http://127.0.0.1:5000/')
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')



@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        f1 = request.form.get('dorm')

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, r'C:\Users\JAINY\Downloads\uploads', secure_filename(f.filename))
        f.save(file_path)
        img = cv2.imread(file_path)
        roi = img
        url_api = "https://api.ocr.space/parse/image"
        # img compression
        _, compressedimage = cv2.imencode(".jpg", roi, [1, 90])

        # converting into bytes format
        file_bytes = io.BytesIO(compressedimage)


        # specify the language and API key
        result = requests.post(url_api,
                               files={"kungfu.jpg": file_bytes},
                               data={"apikey": "6101f0b3ab88957",
                                     "language": f1})

        result = result.content.decode()
        result = json.loads(result)

        parsed_results = result.get("ParsedResults")[0]
        text_detected = parsed_results.get("ParsedText")
        return text_detected

        # cv2.imshow("roi", roi)
        # cv2.imshow("Img", img)


if __name__ == '__main__':
    app.run(debug=True)

