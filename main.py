import os
import time
import joblib
import numpy as np
from flask import Flask, request, jsonify
from sklearn.linear_model import SGDClassifier
import oandapyV20
import oandapyV20.endpoints.instruments as instruments

# Flask
app = Flask(__name__)

# Ścieżka do modelu
model_path = "smart_money_model.pkl"

def fetch_oanda_data():
    client = oandapyV20.API(access_token=os.getenv("ACCESS_TOKEN"))
    params = {
        "granularity": "M5",  # 5-minutowe świece
        "count": 100
    }
    r = instruments.InstrumentsCandles(instrument="EUR_USD", params=params)
    client.request(r)
    candles = r.response["candles"]

    X, y = [], []

    for i in range(len(candles) - 1):
        c = candles[i]
        next_c = candles[i + 1]

        open_price = float(c["mid"]["o"])
        high = float(c["mid"]["h"])
        low = float(c["mid"]["l"])
        close = float(c["mid"]["c"])
        volume = c["volume"]

        X.append([open_price, high, low, close, volume])

        next_close = float(next_c["mid"]["c"])
        y.append(int(next_close > close))

    return np.array(X), np.array(y)

def load_or_create_model():
    if os.path.exists(model_path):
        print("📦 Ładowanie istniejącego modelu...")
        try:
            return joblib.load(model_path)
        except Exception as e:
            print(f"❌ Błąd ładowania modelu: {e}")
            os.remove(model_path)
            print("🧹 Usunięto uszkodzony model, tworzenie nowego...")

    print("🧠 Tworzenie nowego modelu...")
    model = SGDClassifier(loss="log_loss")
    X, y = fetch_oanda_data()
    model.partial_fit(X, y, classes=np.array([0, 1]))
    joblib.dump(model, model_path)
    print("✅ Nowy model zapisany.")
    return model

def generate_signal(model, threshold=0.9):
    X, _ = fetch_oanda_data()
    if len(X) == 0:
        print("⚠️ Brak danych do analizy.")
        return None

    last_data = X[-1].reshape(1, -1)
    proba = model.predict_proba(last_data)[0]

    if max(proba) >= threshold:
        prediction = np.argmax(proba)
        signal = "BUY" if prediction == 1 else "SELL"
        print(f"📢 Mocny sygnał: {signal} (pewność: {max(proba):.2f})")
        return signal
    else:
        print(f"🤔 Brak silnego sygnału (pewność: {max(proba):.2f})")
        return None

def analyze_and_train(model):
    X, y = fetch_oanda_data()
    model.partial_fit(X, y, classes=np.array([0, 1]))
    joblib.dump(model, model_path)
    print("📊 Analiza i trening zakończone.")

@app.route("/")
def home():
    return "✅ Bot działa!"

@app.route("/webhook", methods=["POST"])
def webhook():
    try:
        signal = generate_signal(model)
        if signal:
            print(f"✅ Wysłano sygnał: {signal}")
            return jsonify({"signal": signal}), 200
        else:
            return jsonify({"message": "Brak sygnału."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("✅ Bot uruchomiony...")

    ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
    ACCOUNT_ID = os.getenv("ACCOUNT_ID")
    print("🔑 ACCESS_TOKEN:", bool(ACCESS_TOKEN))
    print("👤 ACCOUNT_ID:", bool(ACCOUNT_ID))

    model = load_or_create_model()

    while True:
        analyze_and_train(model)
        time.sleep(60)  # co 5 minut, by nie spamować
