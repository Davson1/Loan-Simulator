import pickle

from flask import Flask, request, render_template,jsonify
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model(1).pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # For rendering results on HTML GUI
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='Answer to the request {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():

    #For direct API calls trought request

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":

    app.run(debug=True)