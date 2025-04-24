
📝 Tugas Klasifikasi Kredit Komputer - Naive Bayes

📌 Deskripsi Dataset
Dataset ini berisi data dummy terkait pengajuan kredit komputer oleh individu. Setiap baris mewakili satu individu dengan fitur seperti usia, pendapatan, status pekerjaan, dan lainnya, serta label apakah individu tersebut **layak** menerima kredit komputer atau tidak.

Tujuan
Membangun model klasifikasi menggunakan algoritma Naive Bayes untuk memprediksi kelayakan seseorang dalam menerima kredit komputer.

Tahapan Pembuatan Model

1. Import Library
Mengimpor library yang dibutuhkan:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
```

2. Load Dataset
Membaca dataset langsung dari Google Drive:
```python
file_id = "1krLRWedghy_ysJ2N6i-1GJ-ZQUmnu6eu"
url = f"https://drive.google.com/uc?id={file_id}"
data = pd.read_csv(url)
```

3. Eksplorasi Data
Menampilkan beberapa baris awal dan mengecek informasi dasar seperti kolom, tipe data, serta nilai yang hilang.

4. Pra-Pemrosesan Data
- Menggunakan **One-Hot Encoding** untuk mengubah data kategorikal menjadi numerik.
- Menentukan kolom target (`label`) secara otomatis.

5. Split Data
Membagi data menjadi data pelatihan dan pengujian (80:20):
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

6. Membangun Model Naive Bayes
Menggunakan `GaussianNB` dari `sklearn`:
```python
model = GaussianNB()
model.fit(X_train, y_train)
```

7. Evaluasi Model
Menghitung dan menampilkan:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- Akurasi model

```python
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("Akurasi:", accuracy_score(y_test, y_pred))
```

Hasil
Model memberikan hasil prediksi dengan tingkat akurasi yang cukup baik tergantung pada kualitas dan keseimbangan data.
