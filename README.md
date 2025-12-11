# Deteksi Emosi Wajah dengan TensorFlow.js

Aplikasi web sederhana untuk mendeteksi emosi wajah (senang, sedih, marah, takut, dan netral) langsung di browser menggunakan TensorFlow.js.

---

## 1. Fitur Utama

- **Deteksi emosi dari gambar wajah**
- **Berjalan langsung di browser** (client-side, tanpa backend khusus)
- **Menggunakan TensorFlow.js** dengan model dalam format `model.json`
- **Model fallback sederhana** bila model asli belum tersedia
- **Visualisasi hasil**:
  - Label emosi dan persentase probabilitas
  - Progress bar untuk tiap emosi

---

## 2. Struktur Project

```text
Deteksi Wajah/
├─ index.html      # Halaman utama aplikasi
├─ style.css       # Styling tampilan aplikasi
├─ script.js       # Logika deteksi emosi (TensorFlow.js)
└─ model/
   └─ model.json   # Model TensorFlow.js (bisa diganti model terlatih Anda)
```

---

## 3. Prasyarat

- **Browser modern** (Chrome, Edge, Firefox, dll.)
- **Node.js** (opsional, jika pakai `npx serve` atau server statis lain)
- Koneksi internet (untuk memuat library TensorFlow.js dari CDN).

---

## 4. Cara Instalasi & Persiapan

1. **Unduh / Clone Project**
   - Simpan project, misalnya di: `C:\Deteksi Wajah\`
   - Pastikan terdapat file:
     - `index.html`
     - `style.css`
     - `script.js`
     - Folder `model/` berisi `model.json`

2. **Cek File Model**
   - Aplikasi akan mencoba memuat model dari:
     ```js
     const MODEL_URL = "model/model.json";
     ```
   - Jika `model.json` gagal dimuat:
     - Aplikasi membuat **model fallback sederhana** (tidak terlatih).
     - Hanya untuk uji alur aplikasi, hasil **tidak akurat**.

---

## 5. Cara Menjalankan Aplikasi

> **PENTING:**  
> Jangan membuka `index.html` dengan cara double-click (URL `file://...`).  
> Jalankan melalui **server HTTP** (Live Server / `npx serve` / server statis lain).

### 5.1. Menggunakan Live Server (VS Code)

1. Buka folder project di **Visual Studio Code**.
2. Install ekstensi **Live Server** (jika belum).
3. Klik kanan `index.html` → **Open with Live Server**.
4. Browser akan terbuka, misalnya pada:
   - `http://127.0.0.1:5500/` atau
   - `http://localhost:5500/`.

### 5.2. Menggunakan `npx serve` (Node.js)

1. Buka **Terminal / Command Prompt** di folder project:
   ```bash
   cd "C:\\Deteksi Wajah"
   ```
2. Jalankan server statis:
   ```bash
   npx serve .
   ```
3. Buka alamat yang ditampilkan, misalnya:
   - `http://localhost:3000`

---

## 6. Cara Menggunakan Aplikasi

1. **Buka aplikasi** melalui URL dari server (misal `http://localhost:3000`).
2. Tunggu status di bawah tombol sampai:
   - `Model berhasil dimuat dari folder 'model'. Silakan upload gambar wajah.`  
     atau (jika gagal):
   - `Model fallback sederhana siap digunakan...` (hasil tidak akurat, hanya demo).

3. Pada bagian **"1. Upload Gambar Wajah"**:
   - Klik tombol **"Pilih Gambar"**.
   - Pilih file gambar wajah (format `JPG` / `PNG`).

4. Setelah upload berhasil:
   - Gambar muncul di bagian **Preview Gambar**.
   - Status: `Gambar berhasil dimuat. Siap untuk deteksi.`

5. Klik tombol **"Deteksi Emosi"**.

6. Lihat hasil di bagian **"4. Hasil Deteksi"**:
   - Emosi utama disorot.
   - Masing-masing emosi memiliki bar dan persentase.

7. Status di bawah tombol menampilkan:
   - Contoh: `Prediksi utama: SENANG (83.5%)`.

---

## 7. Pengaturan Model

### 7.1. Ukuran Input Model

Di `script.js`:

```js
// Ukuran input default untuk model
// 224: model RGB (mis. MobileNet 224x224)
// 48 : model grayscale 48x48 (mis. CNN FER2013)
const MODEL_INPUT_SIZE = 224;
```

- Untuk model **224x224 RGB** → biarkan `224`.
- Untuk model **48x48 grayscale** → ubah menjadi `48`.

Pra-pemrosesan gambar sudah disesuaikan:
- `48`: resize ke 48×48, diubah menjadi grayscale.
- `224`: resize ke 224×224 RGB dan dinormalisasi.

### 7.2. Label Emosi

Di `script.js`:

```js
const EMOTION_LABELS = ["senang", "sedih", "marah", "takut", "netral"];
```

- Urutan label **harus sama** dengan urutan output layer model.
- Jika Anda mengubah jumlah/urutan kelas, ubah array ini sesuai model Anda.

---

## 8. Mengganti Model dengan Model Terlatih Sendiri

Jika Anda punya model hasil training (misal dari dataset FER2013):

1. **Latih Model**
   - Gunakan TensorFlow/Keras (Python) atau TensorFlow.js (JavaScript).
   - Output layer:
     - Jumlah neuron = jumlah kelas emosi.
     - Aktivasi = `softmax`.
   - Contoh 5 kelas:
     ```text
     ["senang", "sedih", "marah", "takut", "netral"]
     ```

2. **Konversi ke TensorFlow.js** (jika melatih di Python)
   - Gunakan `tensorflowjs_converter` untuk menghasilkan:
     - `model.json`
     - File bobot `group*-shard*.bin`
   - Dokumentasi:  
     https://www.tensorflow.org/js/tutorials/conversion/import_keras

3. **Salin ke Folder `model/`**
   - Hapus/replace isi folder `model/` pada project ini.
   - Letakkan:
     - `model.json`
     - Semua file `*.bin` hasil konversi.
   - Pastikan path akhir:
     ```text
     model/model.json
     ```

4. **Sesuaikan `MODEL_INPUT_SIZE` dan `EMOTION_LABELS`** di `script.js` bila berbeda dari default.

---

## 9. Troubleshooting

- **Status: "Model belum dimuat."**
  - Pastikan aplikasi dijalankan melalui server HTTP (bukan `file://`).
  - Cek apakah `model/model.json` ada dan bisa diakses.

- **Error di console: gagal memuat model dari folder 'model'**
  - Format `model.json` tidak sesuai.
  - File bobot `*.bin` hilang atau tidak lengkap.
  - Solusi:
    - Ulangi proses konversi ke format TensorFlow.js.
    - Salin ulang semua file ke folder `model`.

- **Hasil prediksi tidak masuk akal / acak**
  - Kemungkinan masih memakai **model fallback** yang tidak terlatih.
  - Pastikan model sudah diganti dengan model terlatih yang benar.

- **Gambar tidak muncul**
  - Pastikan file benar-benar gambar (`JPG`, `PNG`, dsb.).
  - Coba dengan gambar lain atau browser lain.

---

## 10. Pengembangan Lanjutan

Beberapa ide pengembangan:

- Deteksi **multi-wajah** dalam satu gambar.
- Integrasi dengan model **face detection** untuk crop otomatis wajah.
- Menambah kelas emosi lain (misalnya: jijik, terkejut).
- Menyimpan log/riwayat hasil deteksi atau menampilkan statistik.

---

Dibangun menggunakan **HTML**, **CSS**, **JavaScript**, dan **TensorFlow.js**.
