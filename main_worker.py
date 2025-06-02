# main_worker.py
import os
import time
from model_utils import load_or_create_model, analyze_and_train

if __name__ == "__main__":
    model = load_or_create_model()
    while True:
        analyze_and_train(model)
        print("📊 Model zaktualizowany.")
        time.sleep(60)
