# Food Recommendation System

## Project Overview

Sistem rekomendasi makanan menjadi kebutuhan penting dalam mendukung keputusan pengguna dalam memilih resep yang sesuai dengan selera atau kebutuhan nutrisi mereka. Proyek ini bertujuan untuk membangun sistem rekomendasi berbasis content-based filtering dan collaborative filtering menggunakan dataset dari Food.com.

Proyek ini penting karena memungkinkan personalisasi pengalaman pengguna di platform kuliner. Seiring meningkatnya volume data resep dan ulasan pengguna, pemanfaatan machine learning memungkinkan sistem merekomendasikan makanan secara otomatis, efisien, dan relevan.


## Referensi

1. Aulia, Rahmat, Sayed Achmady, dan Zulfa Razi. *"Pengembangan Web Pencarian Resep Masakan dengan Fitur Rekomendasi Berbasis Algoritma Machine Learning di Provinsi Aceh."* Jurnal Literasi Informatika 3.4 (2024).

2. Chow, Yi-Ying, Su-Cheng Haw, Palanichamy Naveen, Elham Abdulwahab Anaam, dan Hairulnizam Bin Mahdin. *"Food Recommender System: A Review on Techniques, Datasets and Evaluation Metrics."* Journal of System and Management Sciences, Vol. 13 No. 5, 2023, pp. 153–168. ISSN 1816-6075 (Print), 1818-0523 (Online). DOI: [10.33168/JSMS.2023.0510](https://doi.org/10.33168/JSMS.2023.0510)

3. Bondevik, Jon Nicolas, Kwabena Ebo Bennin, Önder Babur, dan Carsten Ersch. *"Food Recommendation Systems Based On Content-based and Collaborative Filtering Techniques."* 14th International Conference on Computing, Communication and Networking Technologies (ICCCNT), IIT-Delhi, October 2023. DOI: [10.1109/ICCCNT56998.2023.10307080](https://www.researchgate.net/publication/374418599_Food_Recommendation_Systems_Based_On_Content-based_and_Collaborative_Filtering_Techniques)

4. Bahri, Muhamad Naufal Syaiful, I Putu Yuda Danan Jaya, Burhanuddin Dirgantoro, Istik Mal, Umar Ali Ahmad, dan Reza Rendian Septiawan. *"Implementasi Sistem Rekomendasi Makanan pada Aplikasi EatAja Menggunakan Algoritma Collaborative Filtering."* Multinetics, Vol. 7 No. 2 (2021). Published Mar 29, 2022. DOI: [10.32722/multinetics.v7i2.4062](https://doi.org/10.32722/multinetics.v7i2.4062)

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

Dataset yang digunakan berasal dari [Food.com Recipes and Interactions Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions), yang terdiri dari dua file utama:

* `recipes.csv` (231637 entri, 12 kolom)
* `interactions.csv` (1132367 entri, 5 kolom)

### Visualisasi dan Insight:

1. **Distribusi Rating**
   ![Distribusi Rating](img/dis_rating.png)

> Terlihat bahwa mayoritas rating adalah 5, menunjukkan bias positif.

2. **Jumlah Rating per User**
   ![Jumlah Rating](img/jumlah_rating.png)

> Mayoritas user hanya memberikan satu atau dua rating.

3. **Boxplot Rating (Outlier Detection)**
   ![Boxplot](img/boxplot.png)

> Terdapat outlier di rating 0-2 yang relatif jarang muncul.

4. **Korelasi antar Fitur Nutrisi**
   ![Korelasi](img/kor.png)

> Kalori berkorelasi tinggi dengan kadar karbohidrat dan gula.

5. **Distribusi Jumlah Bahan dalam Resep**
   ![Jumlah Bahan](img/dis_bahan_resep.png)

> Sebagian besar resep memiliki 7–13 bahan, sesuai dengan tipikal masakan rumah.

## Data Preparation

1. **Handling Missing Values**:

   * Kolom deskripsi, nama, dan bahan diisi dengan nilai default.
   * Rating dinormalisasi menggunakan skala min-max.

2. **Encoding**:

   * `user_id` dan `recipe_id` diencode ke integer menggunakan `LabelEncoder` untuk digunakan di embedding layer.

3. **Text Processing**:

   * Kolom `content` dibentuk dari penggabungan `name`, `description`, dan `ingredients`.
   * Vektorisasi dilakukan menggunakan `TF-IDF` dengan `stop_words='english'` dan `max_features=5000`.

## Modeling and Results

### Content-Based Filtering

Menggunakan algoritma:

* TF-IDF Vectorizer untuk representasi fitur teks.
* Cosine Similarity + NearestNeighbors untuk menemukan resep mirip.

**Contoh hasil rekomendasi**:

* Input: "Cream of Spinach Soup"
* Output: Variasi resep soup bayam dari pengguna lain dengan bahan atau teknik mirip.

### Collaborative Filtering (Deep Learning Based)

Menggunakan arsitektur:

```math
\hat{r}_{ui} = \langle \mathbf{p}_u, \mathbf{q}_i \rangle + b_u + b_i
```

Dengan:

* \$\mathbf{p}\_u\$, \$\mathbf{q}\_i\$ = user/item embedding
* \$b\_u\$, \$b\_i\$ = bias masing-masing
* \$\langle \cdot \rangle\$ = dot product

Implementasi menggunakan `Keras Model` subclassing dan training menggunakan `Adam Optimizer` dan `MSE` loss.

**Hasil evaluasi:**

* MSE: 0.2130
* RMSE: 0.4615

![Training Curve](img/plot.png)

> Model menunjukkan konvergensi baik. Tidak terjadi overfitting signifikan.

**Contoh hasil rekomendasi untuk User ID 8937:**

1. Bacon Lattice Tomato Muffins
2. Breakfast Shepherd’s Pie
3. Mexican Stack Up

## Evaluation

### Metrics Used

* **RMSE (Root Mean Squared Error)**:
  $RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\hat{r}_i - r_i)^2}$
  Digunakan untuk mengukur deviasi prediksi model dari rating aktual.

### Insight

Model deep learning menunjukkan performa yang baik dengan RMSE 0.46, lebih baik dari baseline model matrix factorization yang diuji sebelumnya.


