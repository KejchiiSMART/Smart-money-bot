# main_web.py
import os
from flask import Flask, request, jsonify
from model_utils import load_or_create_model, fetch_oanda_data
import numpy as np

app = Flask(__name__)
model = load_or_create_model()

def generate_signal(model, threshold=0.9):
    X, _ = fetch_oanda_data()
    if len(X) == 0:
        return None
    last_data = X[-1].reshape(1, -1)
    proba = model.predict_proba(last_data)[0]
    if max(proba) >= threshold:
        return "BUY" if np.argmax(proba) == 1 else "SELL"
    return None

@app.route("/")
def home():
    return "✅ Webhook działa!"

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        signal = generate_signal(model)
        return jsonify({"signal": signal}) if signal else jsonify({"message": "Brak sygnału"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
