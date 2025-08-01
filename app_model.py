from flask import Flask, request, jsonify
import pickle
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

app = Flask(__name__)

# Cargar el modelo al iniciar la app
MODEL_PATH = "ad_model.pkl"
model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    print("Modelo no encontrado. El endpoint /predict no funcionará hasta que se entrene.")


# Endpoint de bienvenida
@app.route("/", methods=["GET"])
def hello():
    return "Bienvenido a la API del modelo Ads CTR Optimization del Grupo4 de The Bridge"

# Endpoint de predicción
@app.route("/api/v1/predict", methods=["GET"])
def predict():
    global model
    if model is None:
        return jsonify({"error": "Modelo no entrenado. Ejecuta /api/v1/retrain primero."}), 400

    tv = request.args.get('tv')
    radio = request.args.get('radio')
    newspaper = request.args.get('newspaper')

    if tv is None or radio is None or newspaper is None:
        return jsonify({"error": "Faltan parámetros: tv, radio, newspaper"}), 400

    try:
        tv = float(tv)
        radio = float(radio)
        newspaper = float(newspaper)
        prediction = model.predict([[tv, radio, newspaper]])
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Endpoint de reentrenamiento con nuevos datos
@app.route("/api/v1/retrain", methods=["GET"])
def retrain():
    if os.path.exists("data/Ads_CTR_Optimization_Dataset_new.csv"):
        data = pd.read_csv('data/Ads_CTR_Optimization_Dataset_new.csv')

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

# Endpoint adicional para mostrar estadísticas del dataset original
@app.route("/api/v1/stats", methods=["GET"])
def stats():
    try:
        data = pd.read_csv("data/Ads_CTR_Optimization_Dataset.csv")
        stats = {
            "total_rows": len(data),
            "mean_sales": round(data["sales"].mean(), 2),
            "max_sales": round(data["sales"].max(), 2),
            "min_sales": round(data["sales"].min(), 2)
        }
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": f"No se pudieron calcular las estadísticas: {str(e)}"}), 500

# URL: https://team-challenge-despliegue-grupo4.onrender.com/api/v1/stats

# Ejecutar servidor local
if __name__ == "__main__":
    from os import environ
    port = int(environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
