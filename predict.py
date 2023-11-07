import pickle

from flask import Flask
from flask import request
from flask import jsonify

import xgboost as xgb


model_file = "model_xgb.bin"

with open(model_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

app = Flask("customer-satisfaction")


@app.route("/predict", methods=["POST"])
def predict():
    customer = request.get_json()

    X = dv.transform([customer])

    features = list(dv.get_feature_names_out())

    xtest = xgb.DMatrix(X, feature_names=features)

    y_pred = model.predict(xtest)[0]

    unsatisfied = y_pred >= 0.05

    result = {"probability": float(y_pred), "unsatisfied": bool(unsatisfied)}

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
