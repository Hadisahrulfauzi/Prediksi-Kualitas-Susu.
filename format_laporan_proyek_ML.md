# Laporan Proyek Machine Learning

### Nama : Hadi Sahrul Fauzi
### Nim : 211351060
### Kelas : Malam A

## Domain Proyek

Pada bagian ini, kamu perlu menuliskan latar belakang yang relevan dengan proyek yang diangkat.

## Business Understanding

Prediksi kualitas susu merupakan salah satu aplikasi yang dapat memberikan manfaat bagi industri pengolahan susu. Dengan adanya aplikasi ini, produsen dapat mengetahui kualitas susu sebelum diolah sehingga dapat mengambil tindakan yang tepat untuk meningkatkan kualitas susu.

### Problem Statements
Berikut adalah beberapa masalah yang dapat diatasi dengan aplikasi prediksi kualitas susu: 

- Kesulitan dalam mengidentifikasi faktor-faktor yang mempengaruhi kualitas susu, yang dapat mengarah pada manajemen kualitas yang tidak tepat.
- Kurangnya dukungan dalam mengarahkan produsen pada tindakan perbaikan yang sesuai, seperti peningkatan kualitas bahan baku, proses pengolahan, atau penyimpanan.
- Keterlambatan dalam deteksi kualitas susu yang rendah, yang berpotensi mengakibatkan penurunan kualitas produk susu.

### Goals

Berikut adalah beberapa tujuan dari aplikasi prediksi kualitas susu:

- Meningkatkan identifikasi faktor-faktor yang mempengaruhi kualitas susu untuk pengelolaan kualitas yang lebih tepat.
- Memberikan rekomendasi yang lebih tepat dalam perbaikan kualitas susu, termasuk peningkatan kualitas bahan baku, proses pengolahan, atau penyimpanan.
- Meningkatkan deteksi kualitas susu yang rendah sehingga dapat meningkatkan kualitas produk susu yang baik.

### Solution statements
- Melakukan analisis data yang mendalam untuk mengidentifikasi pola dan tren yang berkaitan dengan kualitas susu. Ini dapat mencakup analisis statistik dan penggunaan teknik seperti data mining.
- Aplikasi prediksi kualitas susu akan memanfaatkan data susu yang relevan, termasuk faktor-faktor yang mempengaruhi kualitas susu, seperti kualitas bahan baku, proses pengolahan, dan penyimpanan. Data ini akan digunakan untuk melatih model machine learning yang dapat memprediksi kualitas susu.
- Model yang dihasilkan dari datasets itu menggunakan metode Random Forest Classifier.

## Data Understanding
Dataset ini dikumpulkan secara manual dari observasi. Hal ini membantu kami membuat model pembelajaran mesin untuk memprediksi kualitas susu.
Dataset ini terdiri dari 8 variabel independen yaitu pH, Suhu, Rasa, Bau, Lemak, Kekeruhan, Warna dan Kualitas. Umumnya, Kualitas atau Kualitas susu bergantung pada parameter-parameter ini. Parameter ini memainkan peran penting dalam analisis prediktif susu. Dataset susu berisi 429 contoh kualitas buruk, 374 contoh kualitas menengah, dan 256 contoh kualitas baik.

[Milk Quality Prediction] (https://www.kaggle.com/datasets/cpluzshrijayan/milkquality).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Heart Failure Prediction Dataset adalah sebagai berikut:
- **pH**            : Fitur ini menentukan pH susu, yang berada pada kisaran 3 hingga 9,5. [float64] 
- **temperature**   : Fitur ini menentukan suhu susu, dan kisarannya adalah dari 34'C hingga 90'C.[int64] 
- **taste**         : Fitur ini mendefinisikan rasa susu dan mengambil nilai yang mungkin: 1 (baik) atau 0 (buruk).[int64] 
- **odor**          : Fitur ini mendefinisikan bau susu dan mengambil nilai yang mungkin: 1 (baik) atau 0 (buruk).[int64] 
- **fat**           : Fitur ini mendefinisikan kandungan lemak susu dan mengambil nilai yang mungkin: 1 (Tinggi) atau 0 (Rendah).[int64] 
- **turbidity**     : Fitur ini menentukan kekeruhan susu dan mengambil nilai yang mungkin: 1 (Tinggi) atau 0 (Rendah).[int64] 
- **colour**        : Fitur ini menentukan warna susu, yang berkisar antara 240 hingga 255.[int64] 
- **grade**         : Ini adalah target dan mengambil nilai: kualitas_rendah, kualitas_sedang, atau kualitas_tinggi.[object] 


## Data Preparation
Pada tahap ini, saya menggunakan metode EDA untuk melakukan preparasi data.
### Data Collection
Untuk data collection ini, saya mendapatkan dataset dari website kaggle dengan nama dataset [Milk Quality Prediction](https://www.kaggle.com/datasets/cpluzshrijayan/milkquality)., jika anda tertarik dengan datasetnya, anda bisa click link tersebut.

### Data Discovery And Profiling
Karena kita menggunakan google colab untuk mengerjakannya maka kita akan import files,
``` bash
from google.colab import files
```

Lalu mengupload token kaggle agar nanti bisa mendownload sebuah dataset dari kaggle melalui google colab,
``` bash
file.upload()
```

Setelah mengupload filenya, maka kita akan lanjut dengan membuat sebuah folder untuk menyimpan file kaggle.json yang sudah diupload tadi,
``` bash
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!ls ~/.kaggle
```
Kita mengimport semua library yang dibutuhkan, 
``` bash
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score,confusion_matrix
```
Lanjut dengan memasukkan file csv yang telah diextract pada sebuah variable
```bash
data = pd.read_csv('milkquality/milknew.csv')
```
Lalu melihat 5 data paling atas dari datasetsnya,
```bash
data.head(5)
```
Untuk melihat statistik deskriptif dari sebuah DataFrame atau struktur data,
``` bash
data.describe()
```
Kemudian saya akan melihat tipe data yang ada pada masing-masing kolom pada dataset tersebut dengan perintah,
``` bash
data.info()
```
Memeriksa nilai yang hilang (NA, Not Available) 
``` bash
data.isna().sum()
```
Memeriksa grup data berdasarkan 'Kelas'
``` bash
data.groupby('Grade').size()
```
Mengubah nilai data 'Grade'
- 0 Untuk "low" #buruk
- 1 Untuk "medium" #sedang
- 2 Untuk "high" #baik
``` bash
data['Grade']=data['Grade'].map({'low':0,'medium':1,'high':2})
data.head()
```
Buat Korelasi heatmap
``` bash
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
```
![](./assets/evaluasi.png) <br>

Lalu lakukan pemisahan data menjadi variabel dependen (target) dan variabel independen (fitur) yang umum dalam pemodelan data. Dalam hal ini, y akan menjadi target atau label, dan x akan menjadi fitur atau atribut yang digunakan untuk memprediksi target,
``` bash
X=data.drop(['Grade'],axis=1)
y=data['Grade']
```
## Modeling

kita membuat objek model Regresi Logistik (Logistic Regression) dalam library scikit-learn (sklearn) dengan beberapa parameter yang telah diatur.
``` bash
from sklearn import linear_model
lr= linear_model.LogisticRegression(random_state = 42,max_iter= 100)
```

Selanjutnya kita akan menentukan berapa persen dari datasets yang akan digunakan untuk test dan untuk train, disini kita gunakan 20% untuk test dan sisanya untuk training alias 80%,
``` bash
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
```

membuat, melatih, dan mengukur akurasi model menggunakan Random Forest Classifier (RFC) dalam scikit-learn,
``` bash
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
```

Mengukur akurasi model Random Forest Classifier (RFC) pada seluruh dataset,
``` bash
rf.score(X_test, y_test)
```
Ternyata mendapatkan score output 0.9952830188679245 99%

Saya akan coba lakukan pengetesan menggunakan data dumy seperti dibawah ini
``` bash
input_data_milk = np.array([[6.6,	35,	1, 0,	1, 0, 254]])


rf_preds=rf.predict(input_data_milk)
print('Prediksi Kualitas susu:', rf_preds)
```
Setelah pengetesan berhasil, dan modelnya sudah selesai dibuat, kita akan export sebagai sav agar nanti bisa kita gunakan pada project web streamlit kita.
``` bash
import pickle

filename = "milk.sav"
pickle.dump(rf,open(filename,'wb'))
```

## Evaluation
Matrik evaluasi yang saya gunakan disini adalah confusion matrix, karena sangat cocok untuk kasus pengkategorian seperti kasus ini. Dengan membandingkan nilai aktual dengan nilai prediksi.
``` bash
rf_preds = lr.fit(X_train, y_train).predict(X_test)
cm = confusion_matrix(y_test,rf_preds)

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
ax.xaxis.set_ticklabels(['low','medium','high']); ax.yaxis.set_ticklabels
```
![](./assets/evaluasi.png) <br>
Terlihat jelas bahwa model kita berhasil memprediksi nilai stroke yang sama dengan nilai aktualnya sebanyak 293 data.


## Deployment

[My Stroke Prediction App](https://prediksi-kualitas-susu.streamlit.app/).

![](./assets/app.png)

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

