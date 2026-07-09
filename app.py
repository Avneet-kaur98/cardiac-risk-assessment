import json
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = None
feature_info = None
insights = None
load_error = None

try:
    model = joblib.load("model_pipeline.pkl")
    with open("feature_info.json") as f:
        feature_info = json.load(f)
    with open("insights.json") as f:
        insights = json.load(f)
except Exception as exc:
    load_error = str(exc)


def top_factors_for_patient(row, top_n=3):
    if not insights:
        return []
    scored = []
    for feature, stats in insights.get("numeric_stats", {}).items():
        if feature not in row:
            continue
        value = row[feature]
        mean = stats["mean"]
        std = stats["std"] or 1.0
        importance = stats["importance"]
        z_score = (value - mean) / std
        weighted_score = z_score * importance
        scored.append({
            "feature": feature,
            "value": value,
            "direction": "up" if z_score > 0 else "down",
            "magnitude": abs(weighted_score),
        })
    scored.sort(key=lambda x: x["magnitude"], reverse=True)
    top = scored[:top_n]
    if top:
        max_mag = max(item["magnitude"] for item in top) or 1.0
        for item in top:
            item["bar_pct"] = round(min(item["magnitude"] / max_mag, 1.0) * 100, 1)
    return top


@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        features=(feature_info or {}).get("features", []),
        result=None,
        load_error=load_error,
        submitted={},
        insights=insights,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if load_error:
        return render_template(
            "index.html", features=[], result=None,
            load_error=load_error, submitted={}, insights=insights,
        )
    submitted = {}
    try:
        row = {}
        for feat in feature_info["features"]:
            raw_value = request.form.get(feat["name"], "")
            submitted[feat["name"]] = raw_value
            row[feat["name"]] = float(raw_value) if feat["type"] == "numeric" else raw_value

        input_df = pd.DataFrame([row])
        proba = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        risk_index = classes.index(1) if 1 in classes else int(proba.argmax())
        risk_probability = round(float(proba[risk_index]) * 100, 1)
        prediction = model.predict(input_df)[0]

        label_map = feature_info.get("target_labels", {})
        label = label_map.get(str(prediction), str(prediction))

        result = {
            "label": label,
            "probability": risk_probability,
            "is_high_risk": risk_probability >= 50,
            "top_factors": top_factors_for_patient(row),
        }

        return render_template(
            "index.html", features=feature_info["features"], result=result,
            load_error=None, submitted=submitted, insights=insights,
        )
    except Exception as exc:
        return render_template(
            "index.html", features=feature_info["features"],
            result={"error": f"Couldn't score that input: {exc}"},
            load_error=None, submitted=submitted, insights=insights,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
