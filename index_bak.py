from flask import Flask, request, render_template

import requests
from io import BytesIO

app = Flask(__name__)

import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.models import model_from_json

from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
import numpy as np
#from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import os.path

@app.route('/')
def index():
    prediction = 30
    return render_template('index.html', prediction=prediction)

@app.route('/hello')
def hello():
    return 'Hello'

from GunmanPrediction import GunmanPrediction;

@app.route('/gunmanPrediction', methods=['GET', 'POST'])
def gunmanPredictionDispatch():
    return GunmanPrediction(request, requests)


@app.route('/gunmanPrediction_save', methods=['GET', 'POST'])
def gunmanPrediction():
    if request.method == 'POST':
#    try:
        url = request.form['url']
#    except:
    else:
        url = request.args.get('url', 'http://fc.jpn.org/ryuba/gunman/pic/Gunman.3Dmodel.jpg')

    imageDataResponse = requests.get(url)
    image = Image.open(BytesIO(imageDataResponse.content))

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

    url_select_options = [
      'http://fc.jpn.org/ryuba/gunman/pic/Gunman.3Dmodel.jpg',
      'http://fc.jpn.org/ryuba/gunman/pic/Gunman.jpg',
      'http://fc.jpn.org/ryuba/gunman/pic/GunmanProfileIcon.png',
      'http://fc.jpn.org/ryuba/gunman/pic/gunrobo.MMD.pose.jpg',
      'https://shop.r10s.jp/auc-toysanta/cabinet/040her020/bs-3vfn000kjh-004.jpg',
      'https://bandai-a.akamaihd.net/bc/images/shop_top_megatrea/images/0920UA_rider01.jpg',
      'http://www.toei-anim.co.jp/tv/precure/images/top/p_mainv.png',

      'http://fc.jpn.org/ryuba/gunman/keras/gunmanRecognition/Illust.ultraman3.jpg',
      'http://fc.jpn.org/ryuba/gunman/keras/gunmanRecognition/Illust.rider3.jpg',
      'http://fc.jpn.org/ryuba/gunman/keras/gunmanRecognition/ultraman4.jpg',
      'http://fc.jpn.org/ryuba/gunman/keras/gunmanRecognition/rider4.jpg',

      'http://fc.jpn.org/ryuba/gunman/pic/Gunman.running.jpg',
      'http://fc.jpn.org/ryuba/gunman/pic/Gunman.chara.jpg',
      'http://fc.jpn.org/ryuba/gunman/pic/uma.chara.jpg'

    ]

    return render_template(
        'gunmanPrediction.html', 
        url=url, 
        url_select_options=url_select_options,
        gunman=ratio_gunman, 
        ultraman=ratio_ultraman, 
        rider=ratio_rider,
        precure=ratio_precure,
        ret2=ret2
    )

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=80)
