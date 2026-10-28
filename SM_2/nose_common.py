"""
Wspólne funkcje dla treningu i aplikacji Flask - żeby cechy liczone
przy treningu i przy identyfikacji na żywo zawsze były spójne.
"""

import numpy as np

# Kolejność MUSI być zgodna z kolejnością v1, v2, v3 w Twoim app.py
SENSOR_NAMES = ["TGS2610", "TGS2603", "TGS813"]


def extract_features(vals_2d: np.ndarray, sensor_names=SENSOR_NAMES) -> dict:
    """
    vals_2d: tablica [n_probek, n_czujnikow] z wartościami ZNORMALIZOWANYMI
    (procentowa zmiana względem R0, tak jak norm1/norm2/norm3 w app.py).
    """
    feats = {}
    for i, name in enumerate(sensor_names):
        v = vals_2d[:, i]
        feats[f"{name}_mean"] = float(v.mean())
        feats[f"{name}_std"] = float(v.std())
        feats[f"{name}_max"] = float(v.max())
        feats[f"{name}_min"] = float(v.min())
        feats[f"{name}_slope"] = (
            float(np.polyfit(np.arange(len(v)), v, 1)[0]) if len(v) > 1 else 0.0
        )
    return feats
