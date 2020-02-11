# ガンマン率の判定AI

画像を読取り、ガンマン率を判定します。
tensorflow で判定をしています。
flaskでWebAP化しました。

## 導入方法

    $ git clone https://github.com/gunman-vagabond/gunman-flask.git
    $ pip install -r gunman-flask/requirements.txt
    $ cd gunman-flask; python index.py  

## アクセス(例)

    http://xxxxxxx:18080/gunmanPrediction


## herokuに仕込んだ例

[画像選択版](https://gunman-flask.herokuapp.com/gunmanPrediction)

[画像upload版](https://gunman-flask.herokuapp.com/gunmanPredictionUpfile)

