# model_utils.py
import os
import joblib
import numpy as np
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
from sklearn.linear_model import SGDClassifier

model_path = "smart_money_model.pkl"

def fetch_oanda_data():
    client = oandapyV20.API(access_token=os.getenv("ACCESS_TOKEN"))
    params = {"granularity": "M5", "count": 100}
    r = instruments.InstrumentsCandles(instrument="EUR_USD", params=params)
    client.request(r)
    candles = r.response["candles"]

    X, y = [], []
    for i in range(len(candles) - 1):
        c, next_c = candles[i], candles[i + 1]
        o, h, l, c_, v = map(float, (c["mid"]["o"], c["mid"]["h"], c["mid"]["l"], c["mid"]["c"])), c["volume"]
        X.append([o, h, l, c_, v])
        y.append(int(float(next_c["mid"]["c"]) > c_))

    return np.array(X), np.array(y)

def load_or_create_model():
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except:
            os.remove(model_path)
    model = SGDClassifier(loss="log_loss")
    X, y = fetch_oanda_data()
    model.partial_fit(X, y, classes=np.array([0, 1]))
    joblib.dump(model, model_path)
    return model

def analyze_and_train(model):
    X, y = fetch_oanda_data()
    model.partial_fit(X, y, classes=np.array([0, 1]))
    joblib.dump(model, model_path)
