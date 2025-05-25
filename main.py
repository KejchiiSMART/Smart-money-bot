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
    X, y = fetch_mock_data()
    model.partial_fit(X, y, classes=np.array([0, 1]))
    joblib.dump(model, model_path)
    print("✅ Nowy model zapisany.")
    return model

def fetch_mock_data():
    # Przykładowe dane symulujące rynek
    X = np.random.rand(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y

def analyze_and_train(model):
    X, y = fetch_mock_data()
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
