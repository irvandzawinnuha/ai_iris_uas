import csv  # Untuk membaca file CSV
import random  # Untuk mengacak urutan data
import math  # Untuk menghitung jarak Euclidean
import sqlite3  # Untuk membaca file database SQLite
import pandas as pd  # Untuk manipulasi data (opsional, bisa dihapus jika tidak digunakan)

# --- Membaca Dataset dari file CSV ---
def load_dataset(filename, split_ratio):
    """
    Membaca dataset dari file CSV dan membaginya ke data training dan testing.
    """
    with open(filename, 'r') as file:
        lines = csv.reader(file)
        dataset = []
        next(lines)  # Lewati header
        for row in lines:
            if len(row) < 6:
                continue
            try:
                features = [float(x) for x in row[1:5]]  # Ambil kolom 2-5 sebagai fitur
                label = row[5]  # Kolom ke-6 sebagai label
                dataset.append(features + [label])
            except ValueError:
                continue
        print(f"[DEBUG] Total data terbaca: {len(dataset)}")
        random.shuffle(dataset)
        split_index = int(split_ratio * len(dataset))
        training_set = dataset[:split_index]
        testing_set = dataset[split_index:]
        return training_set, testing_set

# --- Mengakses dan menampilkan isi database SQLite ---
def connect_database():
    """
    Membuka koneksi ke database SQLite dan menampilkan daftar tabel.
    """
    conn = sqlite3.connect('database.sqlite')  # File harus berada di folder yang sama
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print("[DEBUG] Tabel dalam database:", tables)
    conn.close()

# --- Menghitung Jarak Euclidean ---
def euclidean_distance(data1, data2):
    """
    Menghitung jarak Euclidean antara dua vektor fitur.
    """
    distance = 0
    for i in range(len(data1) - 1):  # Hindari label (index terakhir)
        distance += (data1[i] - data2[i]) ** 2
    return math.sqrt(distance)

# --- Menentukan K Nearest Neighbors ---
def get_neighbors(training_set, test_instance, k):
    """
    Mengambil K tetangga terdekat berdasarkan jarak.
    """
    distances = []
    for train_instance in training_set:
        dist = euclidean_distance(test_instance, train_instance)
        distances.append((train_instance, dist))
    distances.sort(key=lambda x: x[1])
    neighbors = [distances[i][0] for i in range(k)]
    return neighbors

# --- Voting berdasarkan mayoritas kelas ---
def predict_classification(neighbors):
    """
    Menentukan prediksi kelas berdasarkan suara mayoritas.
    """
    class_votes = {}
    for neighbor in neighbors:
        label = neighbor[-1]
        if label in class_votes:
            class_votes[label] += 1
        else:
            class_votes[label] = 1
    sorted_votes = sorted(class_votes.items(), key=lambda x: x[1], reverse=True)
    return sorted_votes[0][0]

# --- Menghitung Akurasi ---
def calculate_accuracy(test_set, predictions):
    """
    Menghitung akurasi prediksi dibanding data aktual.
    """
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][-1] == predictions[i]:
            correct += 1
    return correct / len(test_set) * 100.0

# --- Fungsi Utama ---
def main():
    filename = 'Iris.csv'  # Pastikan file CSV berada di folder yang sama
    split_ratio = 0.7
    k = 3

    connect_database()  # Tampilkan info database (bisa kamu hapus kalau tidak dipakai)

    training_set, testing_set = load_dataset(filename, split_ratio)
    print(f'Training: {len(training_set)} data')
    print(f'Testing : {len(testing_set)} data\n')

    predictions = []
    for test_instance in testing_set:
        neighbors = get_neighbors(training_set, test_instance, k)
        result = predict_classification(neighbors)
        predictions.append(result)
        print(f'Predicted={result} \tActual={test_instance[-1]}')

    accuracy = calculate_accuracy(testing_set, predictions)
    print(f'\nModel Accuracy: {accuracy:.2f}%')

if __name__ == '__main__':
    main()
