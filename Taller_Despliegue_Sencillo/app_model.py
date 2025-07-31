from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

app = Flask(__name__)

# Endpoint de bienvenida
@app.route("/", methods=["GET"])
def hello():
    return "Bienvenido a mi API del modelo advertising"

# Endpoint de predicción
@app.route("/api/v1/predict", methods=["GET"])
def predict():
    with open('ad_model.pkl', 'rb') as f:
        model = pickle.load(f)

    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    if tv is None or radio is None or newspaper is None:
        return "Args empty, not enough data to predict", 400

    prediction = model.predict([[float(tv), float(radio), float(newspaper)]])
    return jsonify({'predictions': prediction[0]})

# Endpoint de reentrenamiento con nuevos datos
@app.route("/api/v1/retrain", methods=["GET"])
def retrain():
    if os.path.exists("data/Advertising_new.csv"):
        data = pd.read_csv('data/Advertising_new.csv')

        X = data.drop(columns=['sales'])
        y = data['sales']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = Lasso(alpha=6000)
        model.fit(X_train, y_train)

        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))

        model.fit(X, y)
        with open('ad_model.pkl', 'wb') as f:
            pickle.dump(model, f)

        return f"Model retrained. New RMSE: {rmse:.2f}, MAPE: {mape:.2%}"
    else:
        return "<h2>New data for retrain NOT FOUND. Nothing done!</h2>", 404

# Ejecutar servidor local
if __name__ == "__main__":
    from os import environ
    port = int(environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

if __name__=='_main_':
    app.run(debug=True)