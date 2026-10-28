import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Wczytanie danych
if os.path.exists('dane_nos.csv'):
    csv_file = 'dane_nos.csv'
elif os.path.exists('dane_czujnikow.csv'):
    csv_file = 'dane_czujnikow.csv'
else:
    raise FileNotFoundError("Nie znaleziono pliku z danymi ('dane_nos.csv' lub 'dane_czujnikow.csv').")

print(f"Wczytywanie danych z: {csv_file}")
df = pd.read_csv(csv_file)

# 2. Przygotowanie cech (X) i etykiet (y)
if 'Label' in df.columns:
    # Plik z app.py (dane_nos.csv)
    df = df[df['Label'].notna() & (df['Label'] != 'Brak')].copy()
    if 'TGS2610_norm' in df.columns:
        feature_cols = ['TGS2610_norm', 'TGS2603_norm', 'TGS813_norm']
    else:
        feature_cols = ['TGS2610_raw', 'TGS2603_raw', 'TGS813_raw']
    X = df[feature_cols]
    y = df['Label']
elif 'Etykieta' in df.columns:
    # Plik z Zb_z_kalibracja.py (dane_czujnikow.csv)
    df = df[df['Etykieta'].notna() & (df['Etykieta'] != 'Brak')].copy()
    feature_cols = ['TGS2610', 'TGS2603', 'TGS813']
    X = df[feature_cols]
    y = df['Etykieta']
else:
    raise ValueError(f"Nieznany format pliku. Dostępne kolumny: {list(df.columns)}")

print(f"Liczba próbek po oczyszczeniu: {len(df)}")
print(f"Rozkład klas:\n{y.value_counts()}")

# 3. Podział na zbiór treningowy (80%) i testowy (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Wybór i trening modelu
print("\nTrenowanie modelu... Proszę czekać.")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Sprawdzenie skuteczności
y_pred = model.predict(X_test)
print(f"\nSkuteczność modelu: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nSzczegółowy raport:")
print(classification_report(y_test, y_pred))

# 6. Zapisanie modelu do pliku
joblib.dump(model, 'model_nosa.pkl')
print("\nModel zapisany pomyślnie jako 'model_nosa.pkl'")