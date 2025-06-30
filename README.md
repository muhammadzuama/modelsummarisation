# modelsummarisation
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

---