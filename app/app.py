from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("model/model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        year = request.form.get("year")
        model_name = request.form.get("model")
        if not year or not model_name:
            return "Please provide both Year and Model."
        input_df = pd.DataFrame([{"Year": int(year), "Model": model_name}])
        prediction = model.predict(input_df)[0]
        return f"""
            <h2>Prediction: <b>{prediction}</b></h2>
            <a href="/">Try another</a>
        """
    return """
        <h2>Car Performance Prediction</h2>
        <form method="post">
            Year: <input type="number" name="year"><br><br>
            Model: <input type="text" name="model"><br><br>
            <input type="submit" value="Predict">
        </form>
    """

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])
    prediction = model.predict(input_df)
    return jsonify({"prediction": prediction[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
