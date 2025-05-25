
import time
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
import joblib
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from datetime import datetime, timedelta
import os

# === KONFIGURACJA ===
ACCESS_TOKEN = "YOUR_OANDA_API_KEY"
ACCOUNT_ID = "YOUR_OANDA_ACCOUNT_ID"
INSTRUMENT = "XAU_USD"
GRANULARITY = "M1"  # interwał 1 minuta

client = API(access_token=ACCESS_TOKEN)

def fetch_latest_data(count=100):
    params = {
        "count": count,
        "granularity": GRANULARITY,
        "price": "M"
    }
    r = instruments.InstrumentsCandles(instrument=INSTRUMENT, params=params)
    client.request(r)
    candles = r.response.get("candles")
    data = [{
        "time": c["time"],
        "open": float(c["mid"]["o"]),
        "high": float(c["mid"]["h"]),
        "low": float(c["mid"]["l"]),
        "close": float(c["mid"]["c"])
    } for c in candles if c["complete"]]
    return pd.DataFrame(data)

def generate_features(df):
    df["return"] = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(window=5).std()
    df = df.dropna()
    return df[["return", "volatility"]], df

def label_data(df):
    future_return = df["close"].pct_change().shift(-5)
    df["label"] = (future_return > 0.0005).astype(int)
    df = df.dropna()
    return df

def load_or_create_model(model_path="smart_money_model.pkl"):
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        model = SGDClassifier(loss="log_loss")
        model.partial_fit(np.array([[0, 0]]), np.array([0]), classes=np.array([0, 1]))
    return model

def save_signal_log(signal, time, price, filename="logs.csv"):
    with open(filename, "a") as f:
        f.write(f"{time},{price},{signal}\n")

# === GŁÓWNA PĘTLA ===
if __name__ == "__main__":
    model = load_or_create_model()
    print("✅ Bot uruchomiony i gotowy do nauki...")

    while True:
        try:
            df = fetch_latest_data()
            X_raw, df = generate_features(df)
            df = label_data(df)

            if len(df) < 10:
                print("⚠️ Zbyt mało danych do uczenia...")
                time.sleep(60)
                continue

            X = df[["return", "volatility"]].values
            y = df["label"].values

            model.partial_fit(X, y)
            joblib.dump(model, "smart_money_model.pkl")

            prediction = model.predict(X[-1].reshape(1, -1))[0]
            timestamp = df.iloc[-1]["time"]
            price = df.iloc[-1]["close"]

            if prediction == 1:
                print(f"📈 [{timestamp}] SIGNAL: AI SMART BUY at {price}")
                save_signal_log("AI SMART BUY", timestamp, price)

            time.sleep(60)

        except Exception as e:
            print("❌ Błąd:", e)
            time.sleep(60)
