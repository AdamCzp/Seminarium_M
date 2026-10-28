import serial
import csv
import time

import serial.tools.list_ports

def get_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if 'arduino' in p.description.lower() or 'ch340' in p.description.lower():
            return p.device
    for p in ports:
        if p.device == 'COM5':
            return 'COM5'
    return 'COM5'

# --- KONFIGURACJA ---
port = get_arduino_port()  # Sprawdź swój port!
baud_rate = 9600
nazwa_pliku = "dane_czujnikow.csv"
czas_kalibracji = 60  # Czas w sekundach

# Pobranie etykiety od użytkownika
etykieta = input("Podaj nazwę zapachu (np. Kawa, Gaz, Powietrze): ")

try:
    ser = serial.Serial(port, baud_rate, timeout=1)
    time.sleep(2)
    print(f"Połączono z {port}.")
except Exception as e:
    print(f"BŁĄD: Nie można otworzyć portu {port} ({e}). Zamknij Serial Monitor lub inne aplikacje korzystające z tego portu!")
    exit()

# --- FAZA KALIBRACJI ---
print(f"--- ROZPOCZYNAM KALIBRACJĘ ({czas_kalibracji}s) ---")
print("Czekam, aż czujniki się rozgrzeją i ustabilizują...")

start_time = time.time()
while time.time() - start_time < czas_kalibracji:
    remaining = int(czas_kalibracji - (time.time() - start_time))
    if ser.in_waiting > 0:
        linia = ser.readline().decode('utf-8').strip()
        # Wyświetlamy odczyty, żebyś widział jak spadają, ale ich nie zapisujemy
        print(f"Kalibracja... Zostało {remaining}s | Odczyty: {linia}", end="\r")
    time.sleep(0.1)

print("\n--- KALIBRACJA ZAKOŃCZONA! ROZPOCZYNAM ZBIERANIE DANYCH ---")

# --- FAZA ZBIERANIA DANYCH ---
with open(nazwa_pliku, mode="a", newline="") as plik:
    writer = csv.writer(plik)
    if plik.tell() == 0:
        writer.writerow(["Timestamp", "TGS2610", "TGS2603", "TGS813", "Etykieta"])

    try:
        while True:
            if ser.in_waiting > 0:
                linia = ser.readline().decode('utf-8').strip()
                dane = linia.split(',')

                if len(dane) == 3:
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    pelny_wiersz = [timestamp] + dane + [etykieta]
                    writer.writerow(pelny_wiersz)
                    plik.flush()
                    print(f"Zapisano [{etykieta}]: {dane}")
    except KeyboardInterrupt:
        print("\nZatrzymano zbieranie danych.")
        ser.close()