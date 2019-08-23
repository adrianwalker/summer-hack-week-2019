"""
curl --header "Content-Type: application/json" \
--request POST \
--data '{"pregnancies": 8, "glucose": 155, "blood_pressure": 62, "skin_thickness": 26, "insulin": 495, "bmi": 34.0, "diabetic_func": 0.543, "age": 46}' \
http://127.0.0.1:5000/predict
"""

from keras.engine import saving
from sklearn.externals import joblib
from flask import Flask, jsonify
from flask import request
import tensorflow as tf
from flask_cors import CORS

from flask import abort

MODEL_FILE = 'model_tom.dat'
SCALER_FILE = 'scaler.dat'

app = Flask(__name__)
CORS(app)
model = None
scaler = None
graph = None


def _load_model(model_file):
    global model
    global graph
    model = saving.load_model(model_file)
    model._make_predict_function()
    graph = tf.get_default_graph()


def _load_scaler(scaler_file):
    global scaler
    scaler = joblib.load(scaler_file)


def _parse_request(json):

    data = [
        int(json.get('pregnancies', 0)),
        int(json.get('glucose', 0)),
        int(json.get('blood_pressure', 0)),
        int(json.get('skin_thickness', 0)),
        int(json.get('insulin', 0)),
        float(json.get('bmi', 0)),
        float(json.get('diabetic_func', 0)),
        int(json.get('age', 0))
    ]
    print(data)
    return data


def _predict_outcome(data):
    global graph
    data_scaled = scaler.transform([data])

    with graph.as_default():
        predicted_outcome = model.predict(data_scaled)
    print(predicted_outcome)
    return predicted_outcome.tolist()


@app.route('/', methods=['GET'])
def root():
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    if not request.json:
        abort(400)

    return jsonify(_predict_outcome(_parse_request(request.json))), 200


if __name__ == '__main__':
    _load_model(MODEL_FILE)
    _load_scaler(SCALER_FILE)
    app.run(debug=True)
