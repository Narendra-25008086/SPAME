from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import string

app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Same cleaning used during training
def clean_text(text):
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    return text

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["message"]
    cleaned = clean_text(data)

    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    if prediction.lower() == "spam":
        result = "Spam 🚨"
    else:
        result = "Not Spam ✅"

    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)