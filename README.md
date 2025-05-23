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

Sistem ini memanfaatkan teknik pencarian berbasis konten untuk menemukan resep yang mirip satu sama lain berdasarkan teks deskripsi dan bahan-bahan. Proses utamanya:

1. **TF-IDF Vectorization**
   Dokumen teks `content` (gabungan dari `name`, `description`, dan `ingredients`) direpresentasikan dalam bentuk numerik menggunakan TF-IDF (Term Frequency-Inverse Document Frequency), dengan rumus umum:

$$
\text{TF-IDF}(t,d) = TF(t,d) \times \log\left(\frac{N}{DF(t)}\right)
$$

* TF(t,d): Frekuensi kemunculan kata `t` dalam dokumen `d`
* N: Jumlah total dokumen
* DF(t): Jumlah dokumen yang mengandung kata `t`

2. **Cosine Similarity**
   Digunakan untuk mengukur kemiripan antar dokumen berdasarkan sudut antar vektor:

$$
\text{cosine\_similarity}(A,B) = \frac{A \cdot B}{\|A\| \times \|B\|}
$$

   Pendekatan ini diimplementasikan melalui model `NearestNeighbors` dari `sklearn`.

3. **Rekomendasi**
   Fungsi `recommend_content()` menerima nama resep dan mengembalikan N resep terdekat berdasarkan kemiripan cosine dari TF-IDF matrix. Contoh:

   ```python
   print("Rekomendasi makanan yang mirip 'Cream of Spinach Soup':")
   print(recommend_content("cream of spinach soup"))
   ```

   Output: rekomendasi dengan nama dan deskripsi resep-resep yang paling serupa.

4. **Model Disimpan**

   * `tfidf_vectorizer.pkl` – untuk mengubah input teks baru ke vektor.
   * `tfidf_matrix.pkl` – representasi resep dalam bentuk TF-IDF.
   * `nearest_neighbors_model.pkl` – model yang bisa langsung digunakan untuk pencarian similarity.
   * `name_to_index.pkl` – mapping nama ke indeks resep.

---

### Collaborative Filtering (Deep Learning)

Model ini mempelajari interaksi antar pengguna dan resep menggunakan arsitektur Artificial Neural Network berbasis embedding.

#### Arsitektur Model

```python
class RecommenderNet(Model):
    ...
    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        item_vector = self.item_embedding(inputs[:, 1])
        dot_user_item = tf.tensordot(user_vector, item_vector, 2)
        x = dot_user_item + self.user_bias(inputs[:, 0]) + self.item_bias(inputs[:, 1])
        return x
```

Rumus prediksi:

$$
\hat{r}_{ui} = \mathbf{p}_u \cdot \mathbf{q}_i + b_u + b_i
$$

Dimana:

* $\mathbf{p}_u$: vektor embedding user
* $\mathbf{q}_i$: vektor embedding item
* $b_u, b_i$: bias user dan item
* $\hat{r}_{ui}$: prediksi rating

#### Konfigurasi Training

* **Loss Function**: `mean squared error (MSE)`
* **Optimizer**: `Adam` (learning rate 0.0001)
* **Callback**: `EarlyStopping` (patience 3, restore best weights)
* **Epochs**: 20, batch size: 64

```python
history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20, callbacks=[early_stop])
```

#### Evaluasi

```python
y_pred = model.predict(x_val)
mse = mean_squared_error(y_val, y_pred)
rmse = np.sqrt(mse)
```

**Hasil Evaluasi**:

* MSE: 0.2130
* RMSE: 0.4615

Grafik:

![Training Curve](img/plot.png)

Insight:

* Model menunjukkan konvergensi stabil.
* Tidak terjadi overfitting karena val\_loss mengikuti train\_loss.

---

## Evaluation

### Metode Evaluasi

1. **Mean Squared Error (MSE)**:

   $$
   MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
   $$
**Nilai MSE: 0.2130**

2. **Root Mean Squared Error (RMSE)**:

   $$
   RMSE = \sqrt{MSE}
   $$

   RMSE lebih mudah diinterpretasikan karena memiliki satuan yang sama dengan target (rating).

**Nilai RMSE: 0.4615**

Interpretasi:

* Rata-rata prediksi model menyimpang sebesar ±0.46 dari nilai aktual.
* Cukup baik untuk sistem rekomendasi yang bersifat subyektif dan dinamis.

---

## Inference: Collaborative Filtering

```python
hasil = recommend_for_user(8937, top_n=5)
```

Output (top-N recommendation):

| name                         | description                        |
| ---------------------------- | ---------------------------------- |
| Bacon Lattice Tomato Muffins | Ready, Set, Cook! Contest Entry... |
| Breakfast Shepherd’s Pie     | Contest Entry...                   |
| Mexican Stack Up             | Meksiko-style food layer...        |
| Ragu Shuka                   | Mediterranean flavor...            |
| Tropical Potato Salad        | Tropical salad with pineapples...  |

**Insight**:

* Resep yang direkomendasikan cenderung memiliki tema kompetisi atau rasa unik.
* Bisa diasumsikan user 8937 menyukai makanan eksperimental atau kompetitif.




