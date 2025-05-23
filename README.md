# Food Recommendation System

## Project Overview

Sistem rekomendasi makanan menjadi kebutuhan penting dalam mendukung keputusan pengguna dalam memilih resep yang sesuai dengan selera atau kebutuhan nutrisi mereka. Proyek ini bertujuan untuk membangun sistem rekomendasi berbasis content-based filtering dan collaborative filtering menggunakan dataset dari Food.com.

Proyek ini penting karena memungkinkan personalisasi pengalaman pengguna di platform kuliner. Seiring meningkatnya volume data resep dan ulasan pengguna, pemanfaatan machine learning memungkinkan sistem merekomendasikan makanan secara otomatis, efisien, dan relevan.

---

## Referensi

1. Aulia, Rahmat, Sayed Achmady, dan Zulfa Razi. *"Pengembangan Web Pencarian Resep Masakan dengan Fitur Rekomendasi Berbasis Algoritma Machine Learning di Provinsi Aceh."* Jurnal Literasi Informatika 3.4 (2024).

2. Chow, Yi-Ying, Su-Cheng Haw, Palanichamy Naveen, Elham Abdulwahab Anaam, dan Hairulnizam Bin Mahdin. *"Food Recommender System: A Review on Techniques, Datasets and Evaluation Metrics."* Journal of System and Management Sciences, Vol. 13 No. 5, 2023, pp. 153–168. ISSN 1816-6075 (Print), 1818-0523 (Online). DOI: [10.33168/JSMS.2023.0510](https://doi.org/10.33168/JSMS.2023.0510)

3. Bondevik, Jon Nicolas, Kwabena Ebo Bennin, Önder Babur, dan Carsten Ersch. *"Food Recommendation Systems Based On Content-based and Collaborative Filtering Techniques."* 14th International Conference on Computing, Communication and Networking Technologies (ICCCNT), IIT-Delhi, October 2023. DOI: [10.1109/ICCCNT56998.2023.10307080](https://www.researchgate.net/publication/374418599_Food_Recommendation_Systems_Based_On_Content-based_and_Collaborative_Filtering_Techniques)

4. Bahri, Muhamad Naufal Syaiful, I Putu Yuda Danan Jaya, Burhanuddin Dirgantoro, Istik Mal, Umar Ali Ahmad, dan Reza Rendian Septiawan. *"Implementasi Sistem Rekomendasi Makanan pada Aplikasi EatAja Menggunakan Algoritma Collaborative Filtering."* Multinetics, Vol. 7 No. 2 (2021). Published Mar 29, 2022. DOI: [10.32722/multinetics.v7i2.4062](https://doi.org/10.32722/multinetics.v7i2.4062)

---

## Business Understanding

### Problem Statement

Pengguna sulit menemukan resep makanan yang sesuai dengan preferensi atau kebiasaan sebelumnya karena volume data yang besar dan tidak terstruktur.

---

### Goals

Membangun sistem rekomendasi makanan yang dapat:

1. Memberikan rekomendasi resep mirip berdasarkan konten (nama, bahan, deskripsi).
2. Memberikan rekomendasi resep yang disukai user lain dengan pola serupa.

---

### Solution Statement

* **Content-Based Filtering**: Menggunakan TF-IDF dan cosine similarity untuk mencari resep yang mirip dari sisi konten.
* **Collaborative Filtering**: Menggunakan Artificial Neural Network (ANN) untuk mempelajari pola interaksi user-item dari rating pengguna.

---

## Data Understanding

Dataset digunakan dari [Food.com Recipes and Interactions Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions):

* recipes.csv (231,637 entri, 12 kolom)
* interactions.csv (1,132,366 entri, 5 kolom)

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

### Data Quality:

* **Missing Values**: ditemukan pada kolom `review`, `name`, dan `description`.
* **Duplicate Rows**: tidak ditemukan data duplikat pada kedua dataset.

---

### Exploratory Data Analysis:

1. **Distribusi Rating**

   ![Distribusi Rating](img/dis_rating.png)

   Distribusi rating sangat tidak seimbang dengan dominasi rating 5. Hal ini menunjukkan bahwa pengguna cenderung hanya memberi nilai tinggi.

2. **Jumlah Rating per User**

   ![Jumlah Rating](img/jumlah_rating.png)

   Sebagian besar pengguna memberikan sedikit rating, menunjukkan ketimpangan aktivitas antar pengguna.

3. **Boxplot Rating (Outlier Detection)**

   ![Boxplot](img/boxplot.png)

   Boxplot menunjukkan adanya beberapa outlier terutama pada rating rendah (0-2), walau mayoritas rating berada pada kisaran tinggi.

4. **Korelasi antar Fitur Nutrisi**

   ![Korelasi](img/kor.png)

   Terlihat bahwa nilai kalori memiliki korelasi tinggi terhadap kadar karbohidrat dan gula. Ini memberikan informasi penting dalam membangun fitur tambahan untuk filtering nutrisi di masa depan.

5. **Distribusi Jumlah Bahan dalam Resep**

   ![Jumlah Bahan](img/dis_bahan_resep.png)

   Kebanyakan resep memiliki 7 hingga 13 bahan, mencerminkan masakan harian rumah tangga yang tidak terlalu kompleks.

---

## Data Preparation

### 1. Handling Missing Values:

* `description`, `name`, `ingredients` diisi default string.
* `review` kosong diisi "No review provided by user."

### Content-Based Filtering Preprocessing

1. **Pembuatan Fitur `content`**:

   * Tiga kolom utama yaitu `name`, `description`, dan `ingredients` digabung menjadi satu string teks panjang yang disebut sebagai `content`.
   * Kolom `ingredients` diparsing dari string list menjadi list Python menggunakan `eval`, kemudian digabung menjadi satu string dengan spasi sebagai pemisah.
   * Ini menghasilkan deskripsi utuh setiap resep yang mencakup judul, penjelasan, dan daftar bahan dalam satu format teks.

2. **Penghapusan Nilai Kosong**:

   * Setelah pengisian nilai kosong, dilakukan pengecekan ulang dan penghapusan terhadap baris-baris yang masih memiliki nilai kosong pada kolom `content` untuk menjamin kualitas input vektorisasi.

3. **TF-IDF Vectorization**:

   * Fitur `content` kemudian dikonversi ke bentuk numerik menggunakan TF-IDF (Term Frequency-Inverse Document Frequency).
   * `TfidfVectorizer` digunakan dengan parameter berikut:

     * `max_features=5000`: untuk membatasi jumlah fitur dan menjaga efisiensi komputasi.
     * `stop_words='english'`: menghapus kata-kata umum (stopwords) dari Bahasa Inggris.
   * Hasilnya adalah matriks sparse berukuran `[jumlah_resep x 5000]` yang merepresentasikan pentingnya setiap kata dalam masing-masing resep.

### Collaborative Filtering Preprocessing

1. **Encoding Identitas**:

   * Menggunakan `LabelEncoder` untuk mengubah `user_id` dan `recipe_id` menjadi ID numerik berurutan agar dapat digunakan sebagai indeks dalam embedding layer.

2. **Normalisasi Rating**:

   * Rating pengguna dinormalisasi ke rentang 0–1 menggunakan rumus Min-Max Scaling:

     ```python
     interactions['rating_normalized'] = interactions['rating'].apply(lambda x: (x - min_rating) / (max_rating - min_rating))
     ```

3. **Pemisahan Data**:

   * Fitur input (`X`) dan target (`y`) dipisahkan dan kemudian dibagi menjadi training dan validation set menggunakan `train_test_split` dengan rasio 80:20.

4. **Reshape Target**:

   * Data target `y_train` dan `y_val` direstrukturisasi menjadi vektor kolom (reshape menjadi `(-1,1)`) agar kompatibel dengan output dari model deep learning.

---

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


