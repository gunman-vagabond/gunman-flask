from flask import Flask, request, render_template

from io import BytesIO

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import model_from_json

from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
import numpy as np
from PIL import Image
import glob
import os.path

f_model = "./GunmanPrediction/"
model_filename = "gunmanRecognition.json"
weights_filename = "gunmanRecognition.h5"

json_string = open(os.path.join(f_model, model_filename)).read()
model = model_from_json(json_string)
model.load_weights(os.path.join(f_model,weights_filename))


def gunmanPredictionUpfile(request):

    ratio_gunman=""
    ratio_ultraman=""
    ratio_rider=""
    ratio_precure=""
    ret2=""

    imgfilename="static/gray.jpg"
    if request.method == 'POST':
#        url = request.form['url']
#    else:
#        url = request.args.get('url', 'http://fc.jpn.org/ryuba/gunman/pic/Gunman.3Dmodel.jpg')

      if 'file' in request.files:
        file = request.files['file']
        image = Image.open(file)
        imgfilename = "static/" + file.filename 
        image.save(imgfilename);

        image_size = 50
        image = image.convert("RGB")
        image = image.resize((image_size, image_size))
        data = np.asarray(image)

        X = []
        X.append(data)
        X = np.array(X)

        X = X.astype('float32')
        X = X / 255.0

        predict = model.predict(X, batch_size=1)
#    ret1 = "predict ret:"+ str(ret) + " (gunman, ultraman, rider, precure)"

        bestscore = 0.0
        bestnum = 0
        for n in [0,1,2,3]:
            if bestscore < predict[0][n]:
                bestscore = predict[0][n]
                bestnum = n

#    category = ["gunman", "ultraman", "rider", "precure"]
        category = ["ガンマン", "ウルトラマン", "仮面ライダー", "プリキュア"]
        
        ret2 = "これは " + str(bestscore*100) + "% " + category[bestnum] + "だと思います"

        ratio_gunman   ='{0:.6f}'.format(predict[0][0] * 100)
        ratio_ultraman ='{0:.6f}'.format(predict[0][1] * 100)
        ratio_rider    ='{0:.6f}'.format(predict[0][2] * 100)
        ratio_precure  ='{0:.6f}'.format(predict[0][3] * 100)

      result_display = "block";
    else :
      result_display = "none";

    return render_template(
        'gunmanPredictionUpfile.html', 
#        url=url, 
#        url_select_options=url_select_options,
        gunman=ratio_gunman, 
        ultraman=ratio_ultraman, 
        rider=ratio_rider,
        precure=ratio_precure,
        result_display=result_display,
        imgfilename=imgfilename,
        ret2=ret2
    )
