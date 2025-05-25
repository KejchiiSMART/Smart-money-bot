import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Wczytaj dane
df = pd.read_csv("xauusd_data.csv")

# Inżynieria cech
df['body'] = df['close'] - df['open']
df['range'] = df['high'] - df['low']
df['avg_volume'] = df['volume'].rolling(20).mean()
df['high_volume'] = df['volume'] > df['avg_volume']

# Tworzenie etykiet: jeśli duża świeca i wolumen — uznajemy to za potencjalny SMART BUY
df['label'] = ((df['body'] > 0) & (df['high_volume'])).astype(int)

df.dropna(inplace=True)

# Przygotowanie danych
X = df[['body', 'range', 'volume']]
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Trening modelu
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Ocena
preds = model.predict(X_test)
print(classification_report(y_test, preds))

# Zapis modelu
joblib.dump(model, "smart_money_model.pkl")
print("✅ Model zapisany jako smart_money_model.pkl")
