from flask import Flask, request, render_template
import requests

app = Flask(__name__)

@app.route('/')
def index():
    prediction = 30
    return render_template('index.html', prediction=prediction)

@app.route('/hello')
def hello():
    return 'Hello'

from GunmanPrediction import gunmanPrediction;
@app.route('/gunmanPrediction', methods=['GET', 'POST'])
def gunmanPredictionDispatch():
    return gunmanPrediction(request, requests)

from GunclockImageAI import gunclockImageAI;
@app.route('/gunclockImageAI', methods=['GET', 'POST'])
def gunclockImageAIDispatch():
    return gunclockImageAI(request, requests)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=80)
