import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from nose_common import SENSOR_NAMES, extract_features

# Ścieżka do CSV z obsługą fallback
CSV_CANDIDATES = ["dane_nos.csv", "dane_nos/dane_nos.csv"]
WINDOW = "5s"                  # rozmiar okna czasowego - dopasuj do dynamiki czujników
MIN_SAMPLES_PER_WINDOW = 3
EXCLUDED_LABELS = {"Brak", "Identyfikacja"}  # etykiety pomijane przy treningu

# Agregacja pod-etykiet (np. wariantów z wiatrakiem) do głównych klas substancji
ENABLE_LABEL_AGGREGATION = True
LABEL_MAPPING = {
    "Czosnek Wiatrak": "Czosnek",
    "Kawa + Wiatrak": "Kawa",
    "Czyste Powietrze + Oczyszczenie": "Czyste Powietrze",
    "Czyste Powietrze w Pomieszczeniu": "Czyste Powietrze",
    "Czyste Powietrze Pomieszczenie Zamkniete": "Czyste Powietrze",
    "Czyste Powietrze Pomieszczenie Zamkniete z wiatrakiem": "Czyste Powietrze",
}

norm_cols = [f"{name}_norm" for name in SENSOR_NAMES]


def find_csv_path():
    for candidate in CSV_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(f"Nie znaleziono pliku CSV w żadnej z lokalizacji: {CSV_CANDIDATES}")


def main():
    csv_path = find_csv_path()
    print(f"Wczytywanie danych z: {csv_path}")
    df = pd.read_csv(csv_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp")
    df = df[~df["Label"].isin(EXCLUDED_LABELS)]

    if ENABLE_LABEL_AGGREGATION:
        df["Label"] = df["Label"].replace(LABEL_MAPPING)

    if df.empty:
        raise ValueError("Brak danych treningowych po odfiltrowaniu etykiet Brak/Identyfikacja")

    features_list, labels_list = [], []
    for label, group in df.groupby("Label"):
        group = group.set_index("Timestamp")
        for _, window in group.resample(WINDOW):
            if len(window) < MIN_SAMPLES_PER_WINDOW:
                continue
            vals = window[norm_cols].values
            feats = extract_features(vals)
            features_list.append(feats)
            labels_list.append(label)

    X = pd.DataFrame(features_list)
    y = pd.Series(labels_list)
    print(f"Liczba okien treningowych: {len(X)}")
    print("Rozkład klas:")
    print(y.value_counts())

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=300, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\n=== Wyniki na zbiorze testowym ===")
    print(classification_report(y_test, y_pred))
    print("Macierz pomyłek:")
    print(confusion_matrix(y_test, y_pred, labels=clf.classes_))
    print("Klasy:", list(clf.classes_))

    joblib.dump(
        {
            "model": clf,
            "scaler": scaler,
            "feature_columns": X.columns.tolist(),
        },
        "model_nosa.pkl",
    )
    print("\nZapisano model do model_nosa.pkl")


if __name__ == "__main__":
    main()

