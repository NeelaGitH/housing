from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

def return_prediction(model, input_json, transformer):

    input_data = pd.DataFrame([input_json])
    input_data.replace("", np.nan, inplace=True)
    data_transformed = transformer.transform(input_data)
    prediction = model.predict(data_transformed)[0]

    return prediction

with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("housing_transformer.pkl", "rb") as file:
    transformer = pickle.load(file)

@app.route("/")
def index():
    return """
    <h1> Welcome to our housing prediction dataset </h1>
    """

@app.route("/predict", methods = ['POST'])
def housing_prediction():
    content = request.json
    results = return_prediction(model, content, transformer)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug = True)

