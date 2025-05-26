import csv
import random
import math
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import PatternFill
from datetime import datetime

# --- Membaca Dataset dari file CSV ---
def load_dataset(filename, split_ratio):
    with open(filename, 'r') as file:
        lines = csv.reader(file)
        dataset = []
        next(lines)  # Lewati header
        for row in lines:
            if len(row) < 6:
                continue
            try:
                features = [float(x) for x in row[1:5]]  # Kolom 2-5 sebagai fitur
                label = row[5]  # Kolom ke-6 sebagai label
                dataset.append(features + [label])
            except ValueError:
                continue
        print(f"[DEBUG] Total data terbaca: {len(dataset)}")
        random.shuffle(dataset)
        split_index = int(split_ratio * len(dataset))
        return dataset[:split_index], dataset[split_index:]

# --- Hitung Jarak Euclidean ---
def euclidean_distance(data1, data2):
    return math.sqrt(sum((data1[i] - data2[i]) ** 2 for i in range(len(data1) - 1)))

# --- Cari K Tetangga Terdekat ---
def get_neighbors(training_set, test_instance, k):
    distances = [(train, euclidean_distance(test_instance, train)) for train in training_set]
    distances.sort(key=lambda x: x[1])
    return [distances[i][0] for i in range(k)]

# --- Voting Mayoritas ---
def predict_classification(neighbors):
    votes = {}
    for neighbor in neighbors:
        label = neighbor[-1]
        votes[label] = votes.get(label, 0) + 1
    return sorted(votes.items(), key=lambda x: x[1], reverse=True)[0][0]

# --- Hitung Akurasi ---
def calculate_accuracy(test_set, predictions):
    correct = sum(1 for i in range(len(test_set)) if test_set[i][-1] == predictions[i])
    return correct / len(test_set) * 100.0

# --- Simpan Hasil ke Excel dengan warna ---
def save_predictions_to_excel(test_set, predictions):
    wb = Workbook()
    ws = wb.active
    ws.title = "Hasil Prediksi"

    headers = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm', 'Actual', 'Predicted']
    ws.append(headers)

    fill_benar = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")  # hijau
    fill_salah = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")  # merah

    for i in range(len(test_set)):
        features = test_set[i][:-1]
        actual = test_set[i][-1]
        predicted = predictions[i]
        row = features + [actual, predicted]
        ws.append(row)

        # Tandai baris dengan warna
        for col in range(1, 7):
            ws.cell(row=i+2, column=col).fill = fill_benar if actual == predicted else fill_salah

    # Simpan dengan nama berdasarkan waktu
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename = f'hasil_prediksi_{timestamp}.xlsx'
    wb.save(filename)
    print(f"\n[INFO] Hasil prediksi disimpan ke file: {filename}")

# --- Fungsi Utama ---
def main():
    filename = 'Iris.csv'
    split_ratio = 0.7
    k = 3

    training_set, testing_set = load_dataset(filename, split_ratio)
    print(f'Training: {len(training_set)} data')
    print(f'Testing : {len(testing_set)} data\n')

    predictions = []
    for test in testing_set:
        neighbors = get_neighbors(training_set, test, k)
        result = predict_classification(neighbors)
        predictions.append(result)
        print(f'Predicted={result} \tActual={test[-1]}')

    accuracy = calculate_accuracy(testing_set, predictions)
    print(f'\nModel Accuracy: {accuracy:.2f}%')

    save_predictions_to_excel(testing_set, predictions)

if __name__ == '__main__':
    main()
