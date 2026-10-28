import os
import time
import csv
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import serial
import serial.tools.list_ports
import joblib

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
MODEL_FILENAME = 'model_nosa.pkl'

# Załadowanie modelu AI (jeśli istnieje)
ml_model = None
if os.path.exists(MODEL_FILENAME):
    try:
        ml_model = joblib.load(MODEL_FILENAME)
        print("Załadowano wytrenowany model AI (model_nosa.pkl).")
    except Exception as e:
        print(f"Błąd ładowania modelu AI: {e}")

# Inicjalizacja pliku CSV
def init_csv():
    try:
        with open(CSV_FILENAME, 'x', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp', 'TGS2610_raw', 'TGS2603_raw', 'TGS813_raw', 'TGS2610_norm', 'TGS2603_norm', 'TGS813_norm', 'Label'])
    except FileExistsError:
        pass

def get_serial_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if p.device == SERIAL_PORT:
            return SERIAL_PORT
    for p in ports:
        desc = (p.description or '').lower()
        if 'arduino' in desc or 'ch340' in desc or 'cp210' in desc or 'ftdi' in desc:
            return p.device
    for p in ports:
        if 'bluetooth' not in (p.description or '').lower():
            return p.device
    return SERIAL_PORT

def read_from_serial():
    global is_warming_up, is_calibrating, warmup_start_time, calibration_start_time
    global r0_values, calibration_samples_sum, calibration_samples_count
    
    init_csv()
    warmup_start_time = time.time()
    
    ser = None
    while True:
        if ser is None:
            active_port = get_serial_port()
            print(f"Próba połączenia z portem szeregowym {active_port}...")
            try:
                ser = serial.Serial(active_port, BAUD_RATE, timeout=1)
                print(f"Pomyślnie połączono z portem {active_port}.")
            except Exception as e:
                print(f"Błąd połączenia z portem szeregowym ({e}). Ponowna próba za 2 sekundy...")
                time.sleep(2)
                continue

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
                continue

            current_time = time.time()

            if is_warming_up:
                elapsed_warmup_time = current_time - warmup_start_time
                if elapsed_warmup_time < WARMUP_SECONDS:
                    progress = min(100, int((elapsed_warmup_time / WARMUP_SECONDS) * 100))
                    time_left = max(0, int(WARMUP_SECONDS - elapsed_warmup_time))
                    socketio.emit('warmup_progress', {
                        'progress': progress,
                        'time_left': time_left,
                        'raw': [v1, v2, v3]
                    })
                else:
                    is_warming_up = False
                    is_calibrating = True
                    calibration_start_time = time.time()
                    socketio.emit('warmup_done', {})
                    
            elif is_calibrating:
                elapsed_calibration_time = current_time - calibration_start_time
                if elapsed_calibration_time < CALIBRATION_SECONDS:
                    calibration_samples_sum[0] += v1
                    calibration_samples_sum[1] += v2
                    calibration_samples_sum[2] += v3
                    calibration_samples_count += 1
                    
                    progress = min(100, int((elapsed_calibration_time / CALIBRATION_SECONDS) * 100))
                    time_left = max(0, int(CALIBRATION_SECONDS - elapsed_calibration_time))
                    socketio.emit('calibration_progress', {
                        'progress': progress,
                        'time_left': time_left,
                        'samples': calibration_samples_count,
                        'raw': [v1, v2, v3]
                    })
                else:
                    if calibration_samples_count > 0:
                        r0_values[0] = calibration_samples_sum[0] / calibration_samples_count
                        r0_values[1] = calibration_samples_sum[1] / calibration_samples_count
                        r0_values[2] = calibration_samples_sum[2] / calibration_samples_count
                        is_calibrating = False
                        print(f"Zakończono kalibrację. R0: {r0_values}")
                        socketio.emit('calibration_done', {'r0': r0_values})
                    else:
                        calibration_start_time = time.time()

            else:
                norm1 = ((v1 - r0_values[0]) / r0_values[0]) * 100 if r0_values[0] != 0 else 0
                norm2 = ((v2 - r0_values[1]) / r0_values[1]) * 100 if r0_values[1] != 0 else 0
                norm3 = ((v3 - r0_values[2]) / r0_values[2]) * 100 if r0_values[2] != 0 else 0
                
                timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                try:
                    with open(CSV_FILENAME, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([timestamp_str, v1, v2, v3, norm1, norm2, norm3, current_label])
                except Exception as csv_err:
                    print(f"Błąd zapisu CSV: {csv_err}")
                
                # Predykcja zapachu przez model AI
                predicted_label = None
                prediction_confidence = None
                if ml_model is not None:
                    try:
                        import pandas as pd
                        x_input = pd.DataFrame([[norm1, norm2, norm3]], columns=['TGS2610_norm', 'TGS2603_norm', 'TGS813_norm'])
                        pred = ml_model.predict(x_input)
                        predicted_label = str(pred[0])
                        if hasattr(ml_model, "predict_proba"):
                            probas = ml_model.predict_proba(x_input)[0]
                            prediction_confidence = round(float(max(probas)) * 100, 1)
                    except Exception:
                        pass

                socketio.emit('sensor_data', {
                    'timestamp': timestamp_str,
                    'raw1': v1,
                    'raw2': v2,
                    'raw3': v3,
                    'norm1': norm1,
                    'norm2': norm2,
                    'norm3': norm3,
                    'label': current_label,
                    'prediction': predicted_label,
                    'confidence': prediction_confidence
                })

        except serial.SerialException as e:
            print(f"Utracono połączenie z portem szeregowym: {e}. Ponawianie za 1s...")
            try:
                ser.close()
            except Exception:
                pass
            ser = None
            time.sleep(1)
        except Exception as e:
            print(f"Błąd podczas czytania danych: {e}")
            time.sleep(0.5)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_status', methods=['GET'])
def get_status():
    current_time = time.time()
    warmup_left = max(0, int(WARMUP_SECONDS - (current_time - warmup_start_time))) if is_warming_up and warmup_start_time else 0
    cal_left = max(0, int(CALIBRATION_SECONDS - (current_time - calibration_start_time))) if is_calibrating and calibration_start_time else 0
    return jsonify({
        "current_label": current_label,
        "is_warming_up": is_warming_up,
        "is_calibrating": is_calibrating,
        "warmup_left": warmup_left,
        "calibration_left": cal_left,
        "r0": r0_values,
        "has_model": ml_model is not None
    })

@app.route('/set_label', methods=['POST'])
def set_label():
    global current_label
    data = request.json or {}
    new_label = data.get('label')
    if new_label is not None:
        current_label = new_label.strip()
        return jsonify({"status": "success", "label": current_label})
    return jsonify({"status": "error", "message": "Brak etykiety"}), 400

@app.route('/skip_warmup', methods=['POST'])
def skip_warmup():
    global is_warming_up, is_calibrating, calibration_start_time
    if is_warming_up:
        is_warming_up = False
        is_calibrating = True
        calibration_start_time = time.time()
        socketio.emit('warmup_done', {})
        return jsonify({"status": "success", "message": "Pominięto rozgrzewanie."})
    return jsonify({"status": "ignored", "message": "Rozgrzewanie nie jest aktywne."})

@app.route('/finish_calibration', methods=['POST'])
def finish_calibration():
    global is_calibrating, r0_values, calibration_samples_sum, calibration_samples_count
    if is_calibrating and calibration_samples_count > 0:
        r0_values[0] = calibration_samples_sum[0] / calibration_samples_count
        r0_values[1] = calibration_samples_sum[1] / calibration_samples_count
        r0_values[2] = calibration_samples_sum[2] / calibration_samples_count
        is_calibrating = False
        print(f"Ręcznie zakończono kalibrację. R0: {r0_values}")
        socketio.emit('calibration_done', {'r0': r0_values})
        return jsonify({"status": "success", "r0": r0_values})
    return jsonify({"status": "error", "message": "Zbyt mało próbek lub kalibracja nieaktywna."}), 400

@app.route('/recalibrate', methods=['POST'])
def recalibrate():
    global is_warming_up, is_calibrating, calibration_start_time, calibration_samples_sum, calibration_samples_count
    is_warming_up = False
    is_calibrating = True
    calibration_samples_sum = [0.0, 0.0, 0.0]
    calibration_samples_count = 0
    calibration_start_time = time.time()
    return jsonify({"status": "success", "message": "Rozpoczęto nową kalibrację."})

if __name__ == '__main__':
    serial_thread = threading.Thread(target=read_from_serial, daemon=True)
    serial_thread.start()
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)
