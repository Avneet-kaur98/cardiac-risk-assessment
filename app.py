"""
app.py
------
Flask web app for the Apple Quality Detection project.

Routes:
    /            -> Dashboard: summary stats + charts, data pulled from SQLite
    /predict     -> Form to enter apple measurements + live prediction

Run:
    python app.py
Then open:
    http://127.0.0.1:5000
"""

from flask import Flask, render_template, request
import sqlite3
import pandas as pd
import pickle
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "apple_quality.db")
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

FEATURES = ["Size", "Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", "Acidity"]

# Load the trained model once when the app starts (not on every request)
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)


def get_dashboard_stats():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM apple_quality", conn)
    conn.close()

    stats = {
        "total": len(df),
        "good_count": int((df["Quality"] == "good").sum()),
        "bad_count": int((df["Quality"] == "bad").sum()),
        "avg_by_quality": df.groupby("Quality")[FEATURES].mean().round(3).to_dict(orient="index"),
    }
    return stats


@app.route("/")
def dashboard():
    stats = get_dashboard_stats()
    return render_template("index.html", stats=stats, features=FEATURES)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None
    input_values = {feat: "" for feat in FEATURES}

    if request.method == "POST":
        try:
            input_values = {feat: float(request.form.get(feat, 0)) for feat in FEATURES}
            X = pd.DataFrame([input_values])[FEATURES]
            X_scaled = scaler.transform(X)

            pred = model.predict(X_scaled)[0]
            proba = model.predict_proba(X_scaled)[0]

            prediction = "Good" if pred == 1 else "Bad"
            confidence = round(max(proba) * 100, 1)
        except ValueError:
            prediction = "Error"
            confidence = None

    return render_template(
        "predict.html",
        prediction=prediction,
        confidence=confidence,
        input_values=input_values,
        features=FEATURES,
    )


if __name__ == "__main__":
    app.run(debug=True)
