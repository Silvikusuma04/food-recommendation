# Food Recommendation System

## Project Overview

Sistem rekomendasi makanan menjadi kebutuhan penting dalam mendukung keputusan pengguna dalam memilih resep yang sesuai dengan selera atau kebutuhan nutrisi mereka. Proyek ini bertujuan untuk membangun sistem rekomendasi berbasis content-based filtering dan collaborative filtering menggunakan dataset dari Food.com.

Proyek ini penting karena memungkinkan personalisasi pengalaman pengguna di platform kuliner. Seiring meningkatnya volume data resep dan ulasan pengguna, pemanfaatan machine learning memungkinkan sistem merekomendasikan makanan secara otomatis, efisien, dan relevan.

## Business Understanding

### Problem Statement

Pengguna sulit menemukan resep makanan yang sesuai dengan preferensi atau kebiasaan sebelumnya karena volume data yang besar dan tidak terstruktur.

### Goals

Membangun sistem rekomendasi makanan yang dapat:

1. Memberikan rekomendasi resep mirip berdasarkan konten (nama, bahan, deskripsi).
2. Memberikan rekomendasi resep yang disukai user lain dengan pola serupa.

### Solution Approach

* **Content-Based Filtering**: Menggunakan TF-IDF dan cosine similarity untuk mencari resep yang mirip dari sisi konten.
* **Collaborative Filtering**: Menggunakan Artificial Neural Network (ANN) untuk mempelajari pola interaksi user-item dari rating pengguna.

## Data Understanding

Dataset digunakan dari [Food.com Recipes and Interactions Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions):

* recipes.csv (231,637 entri, 12 kolom)
* interactions.csv (1,132,367 entri, 5 kolom)

### Fitur:

**recipes.csv**:

* `id`: ID resep
* `name`: Nama resep
* `description`: Deskripsi pengguna
* `ingredients`: List bahan masakan
* `tags`: Kategori masakan

**interactions.csv**:

* `user_id`: ID pengguna
* `recipe_id`: ID resep
* `rating`: Rating yang diberikan
* `review`: Teks ulasan

### Exploratory Data Analysis:

1. **Distribusi Rating**
   ![Distribusi Rating](img/dis_rating.png)
   Mayoritas rating yang diberikan pengguna bernilai 5, menunjukkan bias positif umum dalam ulasan pengguna.

2. **Jumlah Rating per User**
   ![Jumlah Rating](img/jumlah_rating.png)
   Sebagian besar pengguna hanya memberikan sedikit ulasan, sebagian besar hanya satu hingga dua.

3. **Boxplot Rating (Outlier Detection)**
   ![Boxplot](img/boxplot.png)
   Boxplot menunjukkan distribusi simetris namun dengan outlier di rating bawah (1–2) yang cukup jarang.

4. **Korelasi antar Fitur Nutrisi**
   ![Korelasi](img/kor.png)
   Terlihat bahwa nilai kalori memiliki korelasi tinggi terhadap kadar karbohidrat dan gula. Ini memberikan informasi penting dalam membangun fitur tambahan untuk filtering nutrisi di masa depan.

5. **Distribusi Jumlah Bahan dalam Resep**
   ![Jumlah Bahan](img/dis_bahan_resep.png)
   Kebanyakan resep memiliki 7 hingga 13 bahan, mencerminkan masakan harian rumah tangga yang tidak terlalu kompleks.

## Data Preparation

### 1. Handling Missing Values:

* `description`, `name`, `ingredients` diisi default string.
* `review` kosong diisi "No review provided by user."

### 2. Duplicate Handling:

* Dataset dicek duplikasi dan dilakukan penghapusan jika ditemukan.

### 3. Normalisasi Rating:

* Skala rating dinormalisasi ke rentang 0–1 menggunakan Min-Max Scaler.

### 4. Encoding:

* `user_id` dan `recipe_id` dienkode ke indeks integer untuk digunakan dalam embedding layer.

### 5. TF-IDF Vectorization:

* Digunakan pada kolom gabungan `content` hasil penggabungan `name`, `description`, dan `ingredients`.
* Vektorisasi dilakukan dengan `max_features=5000` dan `stop_words='english'`.

## Modeling and Results

### Content-Based Filtering

* **Algoritma**: TF-IDF + Cosine Similarity + NearestNeighbors
* **Output**: Rekomendasi resep mirip

Contoh: 'Cream of Spinach Soup' → beberapa varian serupa muncul di hasil seperti varian rendah kalori dan dengan bahan alternatif.

### Collaborative Filtering

* Menggunakan Keras Model subclassing:
* Memprediksi rating user terhadap resep tertentu:

$$
\hat{r}_{ui} = \langle \mathbf{p}_u, \mathbf{q}_i \rangle + b_u + b_i
$$

Model:

* Embedding Layer untuk user dan item
* Dot product + bias
* Loss: MSE, Optimizer: Adam

**Hasil Evaluasi**:

* MSE: 0.2130
* RMSE: 0.4615

![Training Curve](img/plot.png)
Model menunjukkan konvergensi baik dan hasil evaluasi yang optimal. Tidak terjadi overfitting signifikan.

Contoh hasil rekomendasi User ID 8937:

1. Bacon Lattice Tomato Muffins
2. Breakfast Shepherd’s Pie
3. Mexican Stack Up

## Evaluation

### Metrik Evaluasi:

* **RMSE**: $RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{r}_i - r_i)^2}$

Hasil:

* RMSE: 0.4615 (model ANN)

## Struktur dan Deployment

* Model content-based dan collaborative disimpan:

  * `recommendasi_model.keras` untuk ANN
  * `tfidf_matrix.pkl`, `vectorizer.pkl`, `nearest_neighbors.pkl` untuk content-based

