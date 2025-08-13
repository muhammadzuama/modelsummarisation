# modelsummarisation
---

# 🚀 Proyek Google Colab

## 📌 Deskripsi Singkat

Proyek ini dapat dijalankan langsung di Google Colab untuk memudahkan eksekusi dan eksplorasi kode tanpa perlu setup lokal. Cukup klik link di bawah untuk memulai.

## 🔗 Akses Cepat ke Google Colab

Anda dapat menjalankan proyek ini langsung di Google Colab dengan mengklik tautan berikut:

👉 [Buka di Google Colab](https://colab.research.google.com/drive/160Y8aw3kF8jqRyhLB_5BhGaPWc1KkDPS?usp=sharing)

## 🛠️ Cara Menggunakan

1. Klik link di atas untuk membuka notebook di Google Colab.
2. Jalankan setiap sel kode satu per satu, mulai dari atas.
3. Pastikan Anda sudah login ke akun Google Anda untuk menyimpan salinan notebook jika ingin mengeditnya.

## 📋 Catatan

* Pastikan runtime Anda diatur ke **GPU** jika notebook membutuhkan akselerasi perangkat keras.
* Jika Anda mengalami error, periksa kembali apakah semua dependensi telah terinstall atau jalankan ulang runtime.

---

Silakan modifikasi bagian deskripsi jika proyekmu memiliki fungsi atau tujuan spesifik. Kalau butuh tambahan seperti instalasi library, badge, atau petunjuk lanjutan, saya bisa bantu juga.

## Pelatihan Model

Proyek ini telah dilatih menggunakan dataset **Liputan6 Summary**, sebuah dataset ringkasan teks berbahasa Indonesia yang komprehensif. Proses pelatihan dilakukan di **Google Colab** dengan dukungan **GPU**, memanfaatkan akselerasi perangkat keras untuk efisiensi yang lebih baik.

### Konfigurasi dan Performa Pelatihan

Kami melakukan eksperimen pelatihan dengan beberapa konfigurasi *batch size* dan presisi untuk mengoptimalkan waktu pelatihan dan performa model:

#### Konfigurasi 1: Performa Terbaik

Dengan konfigurasi ini, model menunjukkan performa terbaik dalam hal kecepatan pelatihan dan skor evaluasi:

* **`per_device_train_batch_size=4`**
* **`per_device_eval_batch_size=4`**
* **`fp16=True`**: Penggunaan *mixed precision* (presisi campuran) diaktifkan. Fitur ini sangat efektif pada GPU modern seperti NVIDIA T4, V100, atau A100. Dengan `fp16=True`, model menggunakan tipe data *floating-point* 16-bit, yang secara signifikan **mempercepat pelatihan** tanpa mengorbankan akurasi secara drastis.

**Hasil Pelatihan:**

* **Progres:** `[7500/7500 33:42, Epoch 3/3]`
* **Waktu Pelatihan Total:** 33 menit 42 detik
* **Evaluasi ROUGE (pada 1000 data test sample):**
    * `rouge1`: 38.52%
    * `rouge2`: 22.51%
    * `rougeL`: 34.40%
    * `rougeLsum`: 34.49%

#### Konfigurasi 2: Baseline (Perbandingan)

Sebagai perbandingan, kami juga menguji konfigurasi dengan *batch size* yang lebih besar tanpa `fp16`:

* **`per_device_train_batch_size=8`**
* **`per_device_eval_batch_size=8`**

**Hasil Pelatihan:**

* **Progres:** `Epoch 3` (detail langkah tidak dicatat, tetapi untuk 3 *epoch* penuh)
* **Waktu Pelatihan Total:** Sekitar 1 jam
* **Evaluasi ROUGE (pada 500 data test sample):**
    * `rouge1`: 32.94%
    * `rouge2`: 18.52%
    * `rougeL`: 29.60%
    * `rougeLsum`: 29.65%

**Kesimpulan:**

Dari hasil di atas, jelas bahwa penggunaan `per_device_train_batch_size=4` dikombinasikan dengan `fp16=True` secara signifikan **mempercepat proses pelatihan** (dari sekitar 1 jam menjadi 33 menit 42 detik) dan juga **menghasilkan skor evaluasi ROUGE yang lebih tinggi**. Peningkatan ini menunjukkan bahwa kombinasi *batch size* yang lebih kecil dan *mixed precision* optimal untuk model ini di lingkungan Google Colab dengan GPU yang kompatibel.

---
## Instalasi

Untuk menyiapkan lingkungan Anda, ikuti langkah-langkah berikut:

1.  **Kloning repositori** (jika berlaku):
    ```bash
    git clone <url-repositori-anda>
    cd <direktori-proyek-anda>
    ```

2.  **Buat lingkungan virtual** (disarankan):
    ```bash
    python -m venv venv
    source venv/bin/activate  # Di Windows, gunakan `venv\Scripts\activate`
    ```

3.  **Instal paket-paket yang diperlukan**:
    ```bash
    pip install torch==2.6.0+cu124 -f [https://download.pytorch.org/whl/cu124](https://download.pytorch.org/whl/cu124)
    pip install transformers==4.52.4
    pip install gradio==5.31.0
    pip install numpy==2.0.2
    pip install evaluate pandas
    ```
    *Catatan: Perintah instalasi PyTorch secara spesifik menargetkan versi CUDA 12.4.*

---

## Teknologi Utama dan Versi

Proyek ini mengandalkan pustaka utama berikut dan versi spesifiknya untuk memastikan kompatibilitas dan reproduktifitas:

* **PyTorch**: `2.6.0+cu124` (dengan dukungan CUDA 12.4)
* **Hugging Face Transformers**: `4.52.4`
* **Gradio**: `5.31.0`
* **NumPy**: `2.0.2`
* **evaluate**: (Terbaru yang kompatibel dengan `transformers`)
* **pandas**: (Terbaru yang kompatibel dengan `numpy`)

Tentu, saya akan menambahkan penjelasan untuk kode tersebut ke dalam dokumen `README.md` Anda.

-----

### Inisialisasi Tokenizer dan Model

Bagian kode ini bertanggung jawab untuk memuat **tokenizer** dan **model ringkasan (summarization)** yang telah dilatih sebelumnya.

```python
from transformers import BertTokenizer, EncoderDecoderModel

"""
Memuat Tokenizer.
"""
tokenizer = BertTokenizer.from_pretrained("cahya/bert2bert-indonesian-summarization")
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

"""
Memuat Model Ringkasan.
"""
model = EncoderDecoderModel.from_pretrained("cahya/bert2bert-indonesian-summarization")
model = model.to(device) # Memindahkan model ke GPU jika tersedia.
```

  * **`tokenizer = BertTokenizer.from_pretrained("cahya/bert2bert-indonesian-summarization")`**: Baris ini memuat tokenizer BERT yang sudah dilatih sebelumnya dari Hugging Face Hub. Tokenizer ini spesifik untuk model "cahya/bert2bert-indonesian-summarization", memastikan bahwa teks akan diproses (dipecah menjadi token) dengan cara yang sama seperti yang digunakan saat model dilatih.
  * **`tokenizer.bos_token = tokenizer.cls_token`** dan **`tokenizer.eos_token = tokenizer.sep_token`**: Ini mengatur token *beginning-of-sequence* (BOS) dan *end-of-sequence* (EOS) untuk tokenizer. Dalam kasus ini, token `[CLS]` (Class) digunakan sebagai token BOS dan `[SEP]` (Separator) sebagai token EOS, yang merupakan konfigurasi umum untuk model berbasis BERT dalam tugas generasi teks.
  * **`model = EncoderDecoderModel.from_pretrained("cahya/bert2bert-indonesian-summarization")`**: Baris ini memuat model ringkasan `EncoderDecoderModel` yang telah dilatih sebelumnya. Model ini dirancang khusus untuk tugas *sequence-to-sequence* seperti ringkasan teks.
  * **`model = model.to(device)`**: Ini adalah langkah krusial untuk memindahkan model ke perangkat komputasi yang ditentukan (misalnya, **GPU** jika `device` diatur ke "cuda"). Memindahkan model ke GPU dapat mempercepat proses inferensi dan pelatihan secara signifikan. Pastikan variabel `device` telah didefinisikan sebelumnya di kode Anda (misalnya, `device = "cuda" if torch.cuda.is_available() else "cpu"`).

-----

