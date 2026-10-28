import time
import csv
import threading
from collections import deque
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import serial
import joblib
import numpy as np
import pandas as pd

from nose_common import SENSOR_NAMES, extract_features

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*")

# Konfiguracja portu szeregowego i kalibracji
SERIAL_PORT = 'COM5'
BAUD_RATE = 9600
WARMUP_SECONDS = 600
CALIBRATION_SECONDS = 300

# Zmienne globalne stanu
current_label = "Brak"
is_warming_up = True
is_calibrating = False
warmup_start_time = None
calibration_start_time = None
r0_values = [0.0, 0.0, 0.0]
calibration_samples_sum = [0.0, 0.0, 0.0]
calibration_samples_count = 0

CSV_FILENAME = 'dane_nos.csv'

# --- Konfiguracja i stan trybu identyfikacji ---
MODEL_PATH = 'model_nosa.pkl'
IDENTIFICATION_WINDOW_SECONDS = 5.0   # ile ostatnich sekund danych bierzemy do klasyfikacji
IDENTIFICATION_MIN_SAMPLES = 3        # min. próbek w oknie, żeby w ogóle liczyć cechy
IDENTIFICATION_EMIT_INTERVAL = 1.0    # jak często (s) wysyłamy wynik do frontu

is_identifying = False
identification_buffer = deque()  # (timestamp, norm1, norm2, norm3)
last_identification_emit = 0.0

MODEL_BUNDLE = None

def load_model():
    global MODEL_BUNDLE
    try:
        MODEL_BUNDLE = joblib.load(MODEL_PATH)
        print(f"Wczytano model identyfikacji z {MODEL_PATH} (klasy: {list(MODEL_BUNDLE['model'].classes_)})")
        return True
    except FileNotFoundError:
        MODEL_BUNDLE = None
        print(f"UWAGA: nie znaleziono {MODEL_PATH} - tryb identyfikacji będzie niedostępny dopóki nie uruchomisz train_model.py")
        return False

load_model()

# Inicjalizacja pliku CSV
def init_csv():
    try:
        with open(CSV_FILENAME, 'x', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'TGS2610_raw', 'TGS2603_raw', 'TGS813_raw', 'TGS2610_norm', 'TGS2603_norm', 'TGS813_norm', 'Label'])
    except FileExistsError:
        pass # Plik już istnieje, nie robimy nic

def resolve_serial_port(preferred_port=SERIAL_PORT):
    try:
        import serial.tools.list_ports
        available_ports = [p.device for p in serial.tools.list_ports.comports()]
        if preferred_port in available_ports:
            return preferred_port
        if available_ports:
            print(f"Wybrany port {preferred_port} jest niedostępny. Wykryte porty: {available_ports}. Użycie: {available_ports[0]}")
            return available_ports[0]
    except Exception as e:
        print(f"Błąd podczas wykrywania portów szeregowych: {e}")
    return preferred_port

def read_from_serial():
    global is_warming_up, is_calibrating, warmup_start_time, calibration_start_time, r0_values, calibration_samples_sum, calibration_samples_count
    global is_identifying, identification_buffer, last_identification_emit
    
    init_csv()
    warmup_start_time = time.time()

    while True:
        ser = None
        # Pętla próby połączenia (auto-reconnect)
        while ser is None:
            port = resolve_serial_port(SERIAL_PORT)
            try:
                print(f"Próba połączenia z portem szeregowym {port}...")
                ser = serial.Serial(port, BAUD_RATE, timeout=1)
                print(f"Połączono z portem szeregowym {port}.")
            except Exception as e:
                print(f"Błąd połączenia z portem szeregowym {port}: {e}. Ponowna próba za 3 sekundy...")
                time.sleep(3)

        # Pętla odczytu danych
        while True:
            try:
                raw_line = ser.readline()
                if not raw_line:
                    continue
                line = raw_line.decode('utf-8', errors='ignore').strip()
                if not line:
                    continue
                    
                parts = line.split(',')
                if len(parts) != 3:
                    print(f"BŁĄD FORMATU DANYCH (Oczekiwano 3 wartości po przecinku): '{line}'")
                    continue
                    
                try:
                    v1, v2, v3 = float(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    print(f"BŁĄD KONWERSJI NA LICZBY: '{line}'")
                    continue # Pomiń błędne dane

                current_time = time.time()

                if is_warming_up:
                    elapsed_warmup_time = current_time - warmup_start_time
                    if elapsed_warmup_time < WARMUP_SECONDS:
                        progress = min(100, int((elapsed_warmup_time / WARMUP_SECONDS) * 100))
                        socketio.emit('warmup_progress', {'progress': progress, 'time_left': int(WARMUP_SECONDS - elapsed_warmup_time)})
                    else:
                        is_warming_up = False
                        is_calibrating = True
                        calibration_start_time = time.time()
                        socketio.emit('warmup_done', {})
                        
                elif is_calibrating:
                    elapsed_calibration_time = current_time - calibration_start_time
                    if elapsed_calibration_time < CALIBRATION_SECONDS:
                        # Zbieramy próbki
                        calibration_samples_sum[0] += v1
                        calibration_samples_sum[1] += v2
                        calibration_samples_sum[2] += v3
                        calibration_samples_count += 1
                        
                        # Powiadom front o postępie
                        progress = min(100, int((elapsed_calibration_time / CALIBRATION_SECONDS) * 100))
                        socketio.emit('calibration_progress', {'progress': progress, 'time_left': int(CALIBRATION_SECONDS - elapsed_calibration_time)})
                    else:
                        # Koniec kalibracji
                        if calibration_samples_count > 0:
                            r0_values[0] = calibration_samples_sum[0] / calibration_samples_count
                            r0_values[1] = calibration_samples_sum[1] / calibration_samples_count
                            r0_values[2] = calibration_samples_sum[2] / calibration_samples_count
                            is_calibrating = False
                            print(f"Zakończono kalibrację. R0: {r0_values}")
                            socketio.emit('calibration_done', {'r0': r0_values})
                        else:
                            # Brak danych w trakcie kalibracji (np. arduino zacięte)
                            calibration_start_time = time.time() # restart

                else:
                    # Normalny odczyt znormalizowany
                    norm1 = ((v1 - r0_values[0]) / r0_values[0]) * 100 if r0_values[0] != 0 else 0
                    norm2 = ((v2 - r0_values[1]) / r0_values[1]) * 100 if r0_values[1] != 0 else 0
                    norm3 = ((v3 - r0_values[2]) / r0_values[2]) * 100 if r0_values[2] != 0 else 0
                    
                    # Zapis do CSV
                    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                    with open(CSV_FILENAME, 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp_str, v1, v2, v3, norm1, norm2, norm3, current_label])
                    
                    # Emisja przez gniazdo
                    socketio.emit('sensor_data', {
                        'timestamp': timestamp_str,
                        'norm1': norm1,
                        'norm2': norm2,
                        'norm3': norm3,
                        'label': current_label
                    })

                    # --- Tryb identyfikacji ---
                    if is_identifying and MODEL_BUNDLE is not None:
                        identification_buffer.append((current_time, norm1, norm2, norm3))
                        # wyrzuć próbki starsze niż okno
                        while (identification_buffer and
                               current_time - identification_buffer[0][0] > IDENTIFICATION_WINDOW_SECONDS):
                            identification_buffer.popleft()

                        enough_time = (
                            len(identification_buffer) >= 2 and
                            identification_buffer[-1][0] - identification_buffer[0][0] >= 1.0
                        )
                        should_emit = current_time - last_identification_emit >= IDENTIFICATION_EMIT_INTERVAL

                        if (len(identification_buffer) >= IDENTIFICATION_MIN_SAMPLES
                                and enough_time and should_emit):
                            vals = np.array([[b[1], b[2], b[3]] for b in identification_buffer])
                            feats = extract_features(vals)
                            X = pd.DataFrame([feats])[MODEL_BUNDLE['feature_columns']]
                            X_scaled = MODEL_BUNDLE['scaler'].transform(X)
                            proba = MODEL_BUNDLE['model'].predict_proba(X_scaled)[0]
                            idx = int(np.argmax(proba))
                            pred_label = MODEL_BUNDLE['model'].classes_[idx]
                            confidence = float(proba[idx])

                            socketio.emit('identification_result', {
                                'label': pred_label,
                                'confidence': confidence,
                                'window_seconds': round(
                                    identification_buffer[-1][0] - identification_buffer[0][0], 1
                                ),
                            })
                            last_identification_emit = current_time

            except Exception as e:
                print(f"Błąd rozłączenia / odczytu z portu szeregowego: {e}. Ponowne łączenie...")
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                time.sleep(2)
                break

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_label', methods=['POST'])
def set_label():
    global current_label
    data = request.json
    new_label = data.get('label')
    if new_label is not None:
        current_label = new_label
        return jsonify({"status": "success", "label": current_label})
    return jsonify({"status": "error", "message": "Brak etykiety"}), 400


@app.route('/start_identification', methods=['POST'])
def start_identification():
    global is_identifying, current_label, identification_buffer, last_identification_emit
    if MODEL_BUNDLE is None and not load_model():
        return jsonify({
            "status": "error",
            "message": f"Brak wytrenowanego modelu ({MODEL_PATH}). Uruchom train_model.py."
        }), 400
    if is_warming_up or is_calibrating:
        return jsonify({"status": "error", "message": "Poczekaj na koniec rozgrzewki/kalibracji"}), 400

    identification_buffer.clear()
    last_identification_emit = 0.0
    is_identifying = True
    current_label = "Identyfikacja"
    return jsonify({"status": "success"})


@app.route('/stop_identification', methods=['POST'])
def stop_identification():
    global is_identifying, current_label
    is_identifying = False
    current_label = "Brak"
    return jsonify({"status": "success"})

if __name__ == '__main__':
    # Uruchomienie wątku tła
    serial_thread = threading.Thread(target=read_from_serial, daemon=True)
    serial_thread.start()
    
    # Uruchomienie aplikacji, allow_unsafe_werkzeug ze względu na wątki dla wbudowanego deweloperskiego serwera
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
