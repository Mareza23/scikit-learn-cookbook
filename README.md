# scikit-learn Cookbook

[![Python](https://img.shields.io/badge/Python-3.x-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org/)
[![Colab](https://img.shields.io/badge/Google-Colab-F9AB00)](https://colab.research.google.com/)

## Tentang Repositori Ini

Repositori ini berisi implementasi kode, penjelasan teori, dan ringkasan berdasarkan buku **scikit-learn Cookbook** terbitan O'Reilly.

Tujuan proyek ini adalah meningkatkan pemahaman konsep machine learning dan memperoleh pengalaman praktis menggunakan library Scikit-learn. Setiap bab disajikan dalam bentuk Jupyter Notebook yang berisi contoh kode, penjelasan, dan interpretasi hasil. Seluruh notebook dapat dijalankan langsung di Google Colab.

## Daftar Isi

| Bab | Topik | Notebook |
|---|---|---|
| 1 | API Scikit-learn | [ch01_sklearn_api.ipynb](ch01_sklearn_api.ipynb) |
| 2 | Alur Kerja Pra-Model dan Preprocessing Data | [ch02_preprocessing.ipynb](ch02_preprocessing.ipynb) |
| 3 | Reduksi Dimensi | [ch03_dimensionality_reduction.ipynb](ch03_dimensionality_reduction.ipynb) |
| 4 | Metrik Jarak dan k-Nearest Neighbors | [ch04_distance_metrics_knn.ipynb](ch04_distance_metrics_knn.ipynb) |
| 5 | Model Linear dan Regularisasi | [ch05_linear_models_regularization.ipynb](ch05_linear_models_regularization.ipynb) |
| 6 | Regresi Logistik Tingkat Lanjut | [ch06_Regresi_Logistik.ipynb.ipynb](ch06_Regresi_Logistik.ipynb.ipynb) |
| 7 | Support Vector Machines dan Metode Kernel | [ch07_SVM_dan_Metode_Kernel.ipynb.ipynb](ch07_SVM_dan_Metode_Kernel.ipynb.ipynb) |
| 8 | Algoritma Berbasis Pohon dan Ensemble | [ch08_tree_based_algorithms_ipynb.ipynb](ch08_tree_based_algorithms_ipynb.ipynb) |

---

## Ringkasan Tiap Bab

### Bab 1 — API Scikit-learn
Pengenalan desain API Scikit-learn yang konsisten.
- Peran **estimator**, **transformer**, dan **predictor** (`fit`, `transform`, `predict`).
- Alur kerja dasar machine learning: memuat data, membagi data latih/uji, melatih model, dan mengevaluasi hasil.
- Konsep `Pipeline` untuk merangkai langkah-langkah pemrosesan dan pemodelan.

### Bab 2 — Alur Kerja Pra-Model dan Preprocessing Data
Menyiapkan data mentah agar siap digunakan oleh model.
- Penanganan **nilai hilang** (*missing values*) dengan imputasi.
- **Penskalaan fitur** (standardisasi dan normalisasi).
- **Encoding data kategorikal** (misalnya one-hot dan ordinal).
- Menggabungkan langkah preprocessing ke dalam satu alur kerja yang rapi.

### Bab 3 — Reduksi Dimensi
Mengurangi jumlah fitur sambil mempertahankan struktur penting data.
- Motivasi reduksi dimensi: mengurangi noise, mempercepat pelatihan, dan memudahkan visualisasi.
- **PCA** dan metode terkait, termasuk memilih jumlah komponen berdasarkan varians yang dijelaskan.
- Visualisasi data berdimensi tinggi dalam 2D.

### Bab 4 — Metrik Jarak dan k-Nearest Neighbors
Bagaimana "kedekatan" antar data diukur dan dipakai untuk prediksi.
- Berbagai **metrik jarak** (Euclidean, Manhattan, dan lainnya).
- Algoritma **k-NN** untuk klasifikasi dan regresi.
- Pengaruh pemilihan nilai $k$ dan penskalaan fitur terhadap hasil.

### Bab 5 — Model Linear dan Regularisasi
Dasar model linear dan cara mengendalikan overfitting.
- **Regresi linear** dan asumsi dasarnya.
- **Regularisasi**: Ridge ($\ell_2$), Lasso ($\ell_1$), dan Elastic Net.
- Pengaruh kekuatan regularisasi terhadap koefisien dan performa model.

### Bab 6 — Regresi Logistik Tingkat Lanjut
Regresi logistik dari konsep dasar hingga pengembangannya.
- **Klasifikasi biner** dengan fungsi sigmoid pada dataset Breast Cancer (569 sampel, 30 fitur); akurasi dasar sekitar 95%.
- **Klasifikasi multikelas** pada dataset Iris: perbandingan *One-vs-Rest* (OvR) dan *multinomial*.
- **Regularisasi** $\ell_1$ dan $\ell_2$ pada loss cross-entropy, termasuk visualisasi **jalur koefisien** untuk berbagai nilai $C$. Ridge ($C=0{,}1$) menaikkan akurasi menjadi sekitar 98%.
- **Klasifikasi multilabel**: satu sampel dengan beberapa label, matriks kemunculan bersama, dan metrik exact match.
- **Metrik evaluasi**: confusion matrix, precision, recall, F1, dan kurva **ROC AUC**.

### Bab 7 — Support Vector Machines dan Metode Kernel
Klasifikasi dan regresi dengan margin maksimum.
- **Prinsip SVM**: mencari batas pemisah dengan margin terbesar; *support vectors* menentukan batas tersebut.
- **SVC** pada dataset Iris dan **SVR** (Support Vector Regression) dengan kernel RBF.
- **Perbandingan kernel** linear, polynomial, dan RBF beserta visualisasi batas keputusan; pada Iris, polynomial memperoleh akurasi sekitar 96%, linear 93%, dan RBF 91%.
- **Penyetelan parameter** dengan `GridSearchCV` (kernel, `C`, `degree`) dan pengaruh parameter $C$.
- **Data berdimensi tinggi**: dampak fitur noise dan peran pemilihan fitur atau reduksi dimensi (PCA).
- **Evaluasi** pada Breast Cancer: SVM linear dengan akurasi sekitar 95% dan AUC sekitar 0,98.

### Bab 8 — Algoritma Berbasis Pohon dan Ensemble
Dari satu pohon keputusan hingga gabungan banyak model, menggunakan dataset Iris.
- **Decision Tree**: Gini impurity, entropi, visualisasi pohon dengan `plot_tree`, serta trade-off bias–variance.
- **Random Forest**: bagging, pengacakan fitur, dan analisis *feature importance*.
- **Gradient Boosting**: model berurutan yang memperbaiki residual, pengaruh *learning rate*, dan risiko overfitting pada jumlah pohon besar.
- **Penyetelan hyperparameter** dengan `GridSearchCV` (27 kombinasi × 5 fold) dan heatmap akurasi cross-validation.
- **Stacking** (`StackingClassifier`): menggabungkan Random Forest dan Gradient Boosting dengan meta-learner Logistic Regression, lalu membandingkannya dengan model tunggal.

---

## Struktur Notebook

Setiap notebook bab berisi:

- **Ringkasan Bab:** Gambaran umum topik utama.
- **Implementasi Kode:** Reproduksi contoh kode dari buku.
- **Penjelasan Teori:** Penjelasan konsep dan metode yang digunakan.
- **Output dan Interpretasi:** Hasil eksekusi kode beserta maknanya.

## Tools dan Library

- Python
- Scikit-learn
- NumPy
- pandas
- Matplotlib / Seaborn
- Jupyter Notebook
- Google Colab

## Cara Penggunaan

1. Buka notebook bab yang ingin dipelajari (di GitHub atau Google Colab).
2. Jalankan sel kode secara berurutan (*Runtime → Run all* di Colab).
3. Baca penjelasan teori yang tersedia di notebook.
4. Amati output untuk memahami cara kerja tiap metode.

## Referensi

*scikit-learn Cookbook.* O'Reilly.

Repositori ini ditujukan untuk keperluan edukasi dan latihan machine learning.
