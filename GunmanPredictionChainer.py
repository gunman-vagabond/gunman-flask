from flask import Flask, request, render_template

import requests
from io import BytesIO

import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable,Chain,optimizers,serializers,datasets

import numpy as np

from chainer.datasets import tuple_dataset
from chainer import training, iterators
from chainer.training import extensions

from PIL import Image
import glob
import os.path

class Alex(Chain):
	# AlexNet
	def __init__(self):
	    super(Alex, self).__init__(
	        conv1 = L.Convolution2D(n_channel, 96, 11, stride=4),
	        conv2 = L.Convolution2D(96, 256, 5, pad=2),
	        conv3 = L.Convolution2D(256, 384, 3, pad=1),
	        conv4 = L.Convolution2D(384, 384, 3, pad=1),
	        conv5 = L.Convolution2D(384, 256, 3, pad=1),
	        fc6 = L.Linear(None, 4096),
	        fc7 = L.Linear(4096, 4096),
	        fc8 = L.Linear(4096, n_label),
	    )

	def __call__(self, x):
		h = F.max_pooling_2d(F.local_response_normalization(
		    F.relu(self.conv1(x))), 3, stride=2)
		h = F.max_pooling_2d(F.local_response_normalization(
		    F.relu(self.conv2(h))), 3, stride=2)
		h = F.relu(self.conv3(h))
		h = F.relu(self.conv4(h))
		h = F.max_pooling_2d(F.relu(self.conv5(h)), 2, stride=2)
		h = F.dropout(F.relu(self.fc6(h)))
		h = F.dropout(F.relu(self.fc7(h)))
		return self.fc8(h)

class DeepLearningClassifier:
	def __init__(self):
		model = Alex()
		self.model = L.Classifier(model)
		self.opt = optimizers.Adam()
		self.opt.setup(self.model)

	def fit(self,X_train, y_train):
		train_data = tuple_dataset.TupleDataset(X_train, y_train)
		train_iter = chainer.iterators.SerialIterator(train_data, batch_size)
		updater = chainer.training.StandardUpdater(train_iter, self.opt)
		self.trainer = chainer.training.Trainer(updater, (n_epoch, 'epoch'), out='result')
		self.trainer.extend(extensions.LogReport())
		self.trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy']))
		self.trainer.extend(extensions.ProgressBar())
		self.trainer.run()

	def fit_and_score(self, X_train, y_train, X_test, y_test):
		train_data = tuple_dataset.TupleDataset(X_train, y_train)
		test_data = tuple_dataset.TupleDataset(X_test, y_test)
		train_iter = chainer.iterators.SerialIterator(train_data, batch_size)
		test_iter = chainer.iterators.SerialIterator(test_data, batch_size, repeat=False, shuffle=False)
		updater=chainer.training.StandardUpdater(train_iter, self.opt)
		self.trainer = chainer.training.Trainer(updater, (n_epoch, 'epoch'), out='result')
		self.trainer.extend(extensions.Evaluator(test_iter, self.model))
		self.trainer.extend(extensions.LogReport())
		self.trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
		self.trainer.extend(extensions.ProgressBar())
		self.trainer.run()

	def predict(self, X_test):
		x=Variable(X_test)
		y=self.model.predictor(x)
		answer=y.data
		answer=np.argmax(answer, axis=1)
		return answer

	def score(self, X_test, y_test):
		y=self.predict(X_test)
		N=y_test.size
		return 1.0-np.count_nonzero(y-y_test)/N

	def predict_proba(self, X_test):
		x=Variable(X_test)
		y=self.model.predictor(x)
		y=np.exp(y.data)
		H=y.sum(1).reshape(-1,1)
		return np.exp(y)/H


# モデル設定
batch_size = 100  # バッチサイズ
#n_epoch = 100  # エポック数
n_channel = 3  # channel数（画像の奥行的な。カラー画像ならRGBなので3、モノクロなら1）
n_label = 4  # 正解ラベルの種類数


#model = Alex()
clf = DeepLearningClassifier()
#a = clf.model
#print (a)


f_model = "./GunmanPredictionChainer/"
weights_filename = "gunmanChainer.npz"
serializers.load_npz(os.path.join(f_model,weights_filename), clf.model)


def gunmanPredictionChainer(request, requests):
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

    X = X.reshape(len(X), 3, 50, 50)

    predict = clf.predict(X)

#    category = ["gunman", "ultraman", "rider", "precure"]
    category = ["ガンマン", "ウルトラマン", "仮面ライダー", "プリキュア"]
        
    ret2 = "これは " + category[int(predict)] + "だと思います"

#    ratio_gunman   ='{0:.6f}'.format(predict[0][0] * 100)
#    ratio_ultraman ='{0:.6f}'.format(predict[0][1] * 100)
#    ratio_rider    ='{0:.6f}'.format(predict[0][2] * 100)
#    ratio_precure  ='{0:.6f}'.format(predict[0][3] * 100)

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
        'gunmanPrediction.Chainer.html', 
        url=url, 
        url_select_options=url_select_options,
#        gunman=ratio_gunman, 
#        ultraman=ratio_ultraman, 
#        rider=ratio_rider,
#        precure=ratio_precure,
        ret2=ret2
    )
