import os
import time
import joblib
import pandas as pd
import numpy as np
from flask import Flask
from sklearn.linear_model import SGDClassifier

# Flask
app = Flask(__name__)

# Ścieżka do modelu
model_path = "smart_money_model.pkl"

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
    model = SGDClassifier()
    X, y = fetch_oanda_data()
    model.partial_fit(X, y, classes=np.array([0, 1]))
    joblib.dump(model, model_path)
    print("✅ Nowy model zapisany.")
    return model

import oandapyV20
import oandapyV20.endpoints.instruments as instruments

def fetch_oanda_data():
    client = oandapyV20.API(access_token=os.getenv("ACCESS_TOKEN"))
    params = {
        "granularity": "M5",  # 5-minutowe świece
        "count": 100
    }
    r = instruments.InstrumentsCandles(instrument="EUR_USD", params=params)
    client.request(r)
    candles = r.response["candles"]

    # Przekształć do numpy array: otwarcie, max, min, zamknięcie, wolumen
    X = []
    y = []

    for i in range(len(candles) - 1):
        c = candles[i]
        next_c = candles[i + 1]

        # Prosta funkcja: przewidujemy, czy cena wzrośnie w kolejnym kroku
        open_price = float(c["mid"]["o"])
        high = float(c["mid"]["h"])
        low = float(c["mid"]["l"])
        close = float(c["mid"]["c"])
        volume = c["volume"]

        X.append([open_price, high, low, close, volume])

        # y = 1 jeśli kolejna świeca zamknęła się wyżej niż obecna
        next_close = float(next_c["mid"]["c"])
        y.append(int(next_close > close))

    return np.array(X), np.array(y)


def analyze_and_train(model):
    X, y = fetch_oanda_data()
    model.partial_fit(X, y, classes=np.array([0, 1]))
    joblib.dump(model, model_path)
    print("📊 Analiza i trening zakończone.")

@app.route("/")
def home():
    return "✅ Bot działa!"

if __name__ == "__main__":
    print("✅ Bot uruchomiony...")

    ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
    ACCOUNT_ID = os.getenv("ACCOUNT_ID")
    print("🔑 ACCESS_TOKEN:", bool(ACCESS_TOKEN))
    print("👤 ACCOUNT_ID:", bool(ACCOUNT_ID))

    model = load_or_create_model()

    while True:
        analyze_and_train(model)
        time.sleep(60)
