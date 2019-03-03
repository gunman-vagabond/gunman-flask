from flask import Flask, request, render_template

import requests
from io import BytesIO

import numpy as np
from PIL import Image
import glob
import os.path
import sys

import tensorflow as tf
import tensorflow.python.platform
from types import *

NUM_CLASSES = 24 * 60
IMAGE_SIZE = 80 
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('readmodels', 'GunclockImageAI/models/model.ckpt', 'File name of model data')


def inference(images_placeholder, keep_prob):
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1') as scope:
#        W_fc1 = weight_variable([7*7*64, 1024])
        W_fc1 = weight_variable([20*20*64, 1024])
        b_fc1 = bias_variable([1024])
#        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 20*20*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    return y_conv

def csv2img():
    return

def test(img):
    test_image = []
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    test_image.append(np.array(img).flatten().astype(np.float32)/255.0)

    test_image = np.asarray(test_image)

    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
    labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
    keep_prob = tf.placeholder("float")

    logits = inference(images_placeholder, keep_prob)
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    saver.restore(sess,FLAGS.readmodels)

    for i in range(len(test_image)):
        pr = logits.eval(feed_dict={ 
            images_placeholder: [test_image[i]],
            keep_prob: 1.0 })[0]
        pred = np.argmax(pr)
#        print (pr)
#        print (pred)

#        print ( ('%02d' % (int(int(pred)/60))) + ":" + ('%02d' % (int(pred)%60)))
    return pred


def gunclockImageAI(request, requests):
    if request.method == 'POST':
        url = request.form['url']
    else:
        url = request.args.get('url', 'https://img.muji.net/img/item/4547315915217_1260.jpg')

    imageDataResponse = requests.get(url)
    image = Image.open(BytesIO(imageDataResponse.content))

    test(image)

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

    mytime = ('%02d' % (int(int(pred)/60))) + ":" + ('%02d' % (int(pred)%60))

    return render_template(
        'gunclockImageAI.html', 
        url=url, 
        url_select_options=url_select_options,
        mytime=mytime,
    )
