import time
import pandas as pd
import numpy as np
from sklearn.linear_model import SGDClassifier
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from datetime import datetime

import os

ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")
ACCOUNT_ID = os.getenv("ACCOUNT_ID")

INSTRUMENT = "XAU_USD"
GRANULARITY = "M1"

client = API(access_token=ACCESS_TOKEN)
model = SGDClassifier()
trained = False

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
    df.dropna(inplace=True)
    return df[["return", "volatility"]], (df["close"].shift(-5) > df["close"]).astype(int)

def save_signal_log(signal, time, price, filename="logs.csv"):
    with open(filename, "a") as f:
        f.write(f"{time},{price},{signal}\n")

print("✅ Bot uruchomiony i gotowy do nauki...")

while True:
    try:
        df = fetch_latest_data()
        X, y = generate_features(df)
        y = y.iloc[-len(X):]

        if len(X) < 10:
            print("⚠️ Zbyt mało danych do uczenia...")
            time.sleep(60)
            continue

        model.partial_fit(X, y, classes=np.array([0, 1]))
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
