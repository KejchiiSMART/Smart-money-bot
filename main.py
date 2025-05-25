import pandas as pd
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from config import API_TOKEN
import joblib

# Połącz się z OANDA
client = API(access_token=API_TOKEN)

def fetch_data(pair="XAU_USD", granularity="M5", count=100):
    params = {
        "granularity": granularity,
        "count": count,
        "price": "M"
    }
    r = instruments.InstrumentsCandles(instrument=pair, params=params)
    client.request(r)
    data = r.response['candles']
    df = pd.DataFrame([{
        "time": c['time'],
        "open": float(c['mid']['o']),
        "high": float(c['mid']['h']),
        "low": float(c['mid']['l']),
        "close": float(c['mid']['c']),
        "volume": c['volume']
    } for c in data if c['complete']])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df

def export_to_csv(df, filename="xauusd_data.csv"):
    df.to_csv(filename)
    print(f"Dane zapisane do {filename}")

def detect_smart_money(df):
    signals = []
    df['avg_vol'] = df['volume'].rolling(window=20).mean()
    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i-1]
        high_volume = row['volume'] > row['avg_vol']
        bullish_engulf = (
            row['close'] > row['open'] and
            row['open'] < prev['close'] and
            row['close'] > prev['open']
        )
        if high_volume and bullish_engulf:
            signals.append((row.name, "SMART BUY"))
    return signals

def detect_liquidity_grab(df):
    signals = []
    for i in range(2, len(df)):
        low_1 = df.iloc[i-2]['low']
        low_now = df.iloc[i]['low']
        close_now = df.iloc[i]['close']
        open_now = df.iloc[i]['open']
        volume_now = df.iloc[i]['volume']
        avg_volume = df['volume'].rolling(20).mean().iloc[i]
        liquidity_grab = (
            low_now < low_1 and
            close_now > open_now and
            close_now > df.iloc[i-1]['close'] and
            volume_now > avg_volume
        )
        if liquidity_grab:
            signals.append((df.index[i], "LIQUIDITY GRAB BUY"))
    return signals

def detect_bos(df):
    signals = []
    for i in range(2, len(df)):
        low_prev = df.iloc[i-2]['low']
        high_prev = df.iloc[i-2]['high']
        low_now = df.iloc[i]['low']
        high_now = df.iloc[i]['high']
        volume_now = df.iloc[i]['volume']
        avg_vol = df['volume'].rolling(20).mean().iloc[i]
        broke_structure = (
            df.iloc[i-1]['low'] > low_prev and
            high_now > high_prev and
            volume_now > avg_vol
        )
        if broke_structure:
            signals.append((df.index[i], "BREAK OF STRUCTURE BUY"))
    return signals

def predict_ai(df, model_path="smart_money_model.pkl"):
    model = joblib.load(model_path)
    df['body'] = df['close'] - df['open']
    df['range'] = df['high'] - df['low']
    features = df[['body', 'range', 'volume']].copy().dropna()
    df = df.loc[features.index]
    predictions = model.predict(features)
    results = []
    for i, pred in enumerate(predictions):
        if pred == 1:
            results.append((df.index[i], "AI SMART BUY"))
    return results

# Uruchomienie bota
if __name__ == "__main__":
    df = fetch_data()
    print(df.tail())
    export_to_csv(df)

    signals = (
        detect_smart_money(df) +
        detect_liquidity_grab(df) +
        detect_bos(df)
    )

    ai_signals = predict_ai(df)

    for time, signal in signals + ai_signals:
        print(f"[{time}] SIGNAL: {signal}")
        
# (...wszystkie wcześniejsze funkcje...)

def export_signals_to_pine(signals, filename="pine_signals.txt"):
    with open(filename, "w") as f:
        f.write("//@version=5\n")
        f.write('indicator("Smart Money AI Signals", overlay=true)\n\n')

        for idx, (time, signal) in enumerate(signals):
            ts = pd.to_datetime(time).strftime('%Y-%m-%dT%H:%M:%S')
            f.write(f"sig{idx} = (time == timestamp(\"{ts}\"))\n")
            f.write(f"plotshape(sig{idx} ? close : na, title=\"{signal}\", location=location.belowbar, color=color.green, style=shape.triangleup, size=size.small)\n\n")

    print("✔️ Wygenerowano plik Pine Script: pine_signals.txt")


# Uruchomienie bota
if __name__ == "__main__":
    df = fetch_data()
    print(df.tail())
    export_to_csv(df)

    signals = (
        detect_smart_money(df) +
        detect_liquidity_grab(df) +
        detect_bos(df)
    )

    ai_signals = predict_ai(df)

    export_signals_to_pine(ai_signals)

    for time, signal in signals + ai_signals:
        print(f"[{time}] SIGNAL: {signal}")
