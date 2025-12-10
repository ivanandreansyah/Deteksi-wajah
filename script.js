// Konfigurasi dasar aplikasi
// ------------------------------------------------------------
// Ukuran input default untuk model. Sesuaikan dengan model Anda:
// - 224: untuk model berbasis MobileNet (RGB 224x224)
// - 48: untuk model CNN sederhana berbasis FER2013 (48x48 grayscale)
const MODEL_INPUT_SIZE = 224;

// Label emosi harus sesuai dengan urutan output layer model Anda
const EMOTION_LABELS = ["senang", "sedih", "marah", "takut", "netral"];

// Path model TensorFlow.js (model.json harus ada di folder /model)
const MODEL_URL = "model/model.json";

// Elemen DOM
const imageUpload = document.getElementById("imageUpload");
const imagePreview = document.getElementById("imagePreview");
const detectButton = document.getElementById("detectButton");
const statusText = document.getElementById("statusText");
const resultContainer = document.getElementById("resultContainer");

// Variabel model global
let model = null;
let modelLoaded = false;
let imageLoaded = false;

// Fungsi utilitas
// ------------------------------------------------------------
function setStatus(message, type = "info") {
    statusText.textContent = message;
    statusText.style.color =
        type === "error" ? "#f87171" : type === "success" ? "#4ade80" : "#fbbf24";
}

function enableDetectIfReady() {
    detectButton.disabled = !(modelLoaded && imageLoaded);
}

// Memuat model TensorFlow.js
// ------------------------------------------------------------
async function loadModel() {
    try {
        setStatus("Memuat model dari folder 'model'...", "info");
        // Gunakan fromURL karena file disajikan oleh server statis (bukan file://)
        model = await tf.loadLayersModel(MODEL_URL);
        modelLoaded = true;
        setStatus("Model berhasil dimuat dari folder 'model'. Silakan upload gambar wajah.", "success");
        enableDetectIfReady();
    } catch (error) {
        console.error("Gagal memuat model dari folder 'model':", error);

        // Jika gagal (misalnya masih memakai placeholder model.json), buat model sederhana
        // langsung di browser sebagai fallback agar alur aplikasi tetap bisa diuji.
        try {
            setStatus("Gagal memuat model asli. Membuat model emosi sederhana di browser...", "info");
            model = createFallbackEmotionModel();
            modelLoaded = true;
            setStatus(
                "Model fallback sederhana siap digunakan. Untuk hasil akurat, ganti folder 'model' dengan model terlatih.",
                "success"
            );
            enableDetectIfReady();
        } catch (fallbackError) {
            console.error("Gagal membuat model fallback:", fallbackError);
            setStatus(
                "Tidak dapat memuat atau membuat model. Pastikan folder 'model' berisi model.json TensorFlow.js yang valid.",
                "error"
            );
        }
    }
}

// Model fallback sederhana (tidak terlatih) agar pipeline deteksi bisa diuji di browser
// tanpa perlu file model.json. Untuk penggunaan nyata, tetap gunakan model terlatih.
function createFallbackEmotionModel() {
    const channels = MODEL_INPUT_SIZE === 48 ? 1 : 3;

    const fallbackModel = tf.sequential();

    fallbackModel.add(
        tf.layers.conv2d({
            inputShape: [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, channels],
            filters: 8,
            kernelSize: 3,
            activation: "relu",
            padding: "same",
        })
    );
    fallbackModel.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

    fallbackModel.add(
        tf.layers.conv2d({
            filters: 16,
            kernelSize: 3,
            activation: "relu",
            padding: "same",
        })
    );
    fallbackModel.add(tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }));

    fallbackModel.add(tf.layers.flatten());
    fallbackModel.add(tf.layers.dense({ units: 32, activation: "relu" }));
    fallbackModel.add(
        tf.layers.dense({ units: EMOTION_LABELS.length, activation: "softmax" })
    );

    fallbackModel.compile({
        optimizer: tf.train.adam(1e-3),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"],
    });

    return fallbackModel;
}

// Pra-pemrosesan gambar untuk model
// ------------------------------------------------------------
function preprocessImage(imgElement) {
    // Bungkus dalam tf.tidy agar memori GPU/CPU otomatis dibersihkan
    return tf.tidy(() => {
        let tensor = tf.browser.fromPixels(imgElement);

        if (MODEL_INPUT_SIZE === 48) {
            // Contoh pipeline untuk model 48x48 grayscale (misalnya model FER2013 custom)
            tensor = tf.image.resizeBilinear(tensor, [48, 48]);
            // Konversi ke grayscale sederhana dengan rata-rata channel
            tensor = tensor.mean(2).expandDims(2); // [48,48] -> [48,48,1]
            tensor = tensor.toFloat().div(255.0);
        } else {
            // Default: pipeline untuk model 224x224 RGB (MobileNet / CNN RGB)
            tensor = tf.image.resizeBilinear(tensor, [MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]);
            tensor = tensor.toFloat().div(255.0);
        }

        // Tambah dimensi batch: [h,w,c] -> [1,h,w,c]
        tensor = tensor.expandDims(0);
        return tensor;
    });
}

// Menjalankan prediksi emosi
// ------------------------------------------------------------
async function runEmotionDetection() {
    if (!model || !modelLoaded) {
        setStatus("Model belum siap.", "error");
        return;
    }
    if (!imageLoaded) {
        setStatus("Silakan upload gambar terlebih dahulu.", "error");
        return;
    }

    setStatus("Melakukan prediksi emosi...", "info");
    detectButton.disabled = true;

    try {
        const inputTensor = preprocessImage(imagePreview);
        const prediction = model.predict(inputTensor);
        const probabilities = (await prediction.data()).slice();

        // Pastikan output sesuai dengan jumlah label
        if (probabilities.length !== EMOTION_LABELS.length) {
            console.warn("Jumlah output model tidak sama dengan jumlah label emosi.");
        }

        // Gabungkan label + probabilitas
        const results = EMOTION_LABELS.map((label, index) => ({
            label,
            score: probabilities[index] !== undefined ? probabilities[index] : 0,
        }));

        // Normalisasi jika total tidak 1 (opsional tapi membantu visualisasi)
        const sum = results.reduce((acc, r) => acc + r.score, 0) || 1;
        const normalized = results.map((r) => ({
            ...r,
            score: r.score / sum,
        }));

        // Urutkan dari tertinggi ke terendah
        normalized.sort((a, b) => b.score - a.score);

        renderResults(normalized);

        const top = normalized[0];
        setStatus(
            `Prediksi utama: ${top.label.toUpperCase()} (${(top.score * 100).toFixed(1)}%)`,
            "success"
        );
    } catch (error) {
        console.error("Error saat prediksi:", error);
        setStatus("Terjadi kesalahan saat melakukan deteksi emosi.", "error");
    } finally {
        enableDetectIfReady();
    }
}

// Menampilkan hasil prediksi ke UI
// ------------------------------------------------------------
function renderResults(results) {
    resultContainer.innerHTML = "";
    if (!results || results.length === 0) {
        return;
    }

    const maxScore = Math.max(...results.map((r) => r.score));

    results.forEach((r) => {
        const item = document.createElement("div");
        item.className = "result-item";
        if (r.score === maxScore) {
            item.classList.add("highlight");
        }

        const emotionEl = document.createElement("div");
        emotionEl.className = "result-emotion";
        emotionEl.textContent = r.label.toUpperCase();

        const barWrapper = document.createElement("div");
        barWrapper.className = "result-bar-wrapper";

        const barBg = document.createElement("div");
        barBg.className = "result-bar-bg";

        const barFill = document.createElement("div");
        barFill.className = "result-bar-fill";
        barFill.style.width = `${(r.score * 100).toFixed(1)}%`;

        barBg.appendChild(barFill);
        barWrapper.appendChild(barBg);

        const percentEl = document.createElement("div");
        percentEl.className = "result-percent";
        percentEl.textContent = `${(r.score * 100).toFixed(1)}%`;

        item.appendChild(emotionEl);
        item.appendChild(barWrapper);
        item.appendChild(percentEl);

        resultContainer.appendChild(item);
    });
}

// Event handler upload gambar
// ------------------------------------------------------------
imageUpload.addEventListener("change", (event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) {
        imageLoaded = false;
        enableDetectIfReady();
        return;
    }

    if (!file.type.startsWith("image/")) {
        setStatus("File yang dipilih bukan gambar.", "error");
        imageLoaded = false;
        enableDetectIfReady();
        return;
    }

    const reader = new FileReader();
    reader.onload = function (e) {
        imagePreview.src = e.target.result;
        imagePreview.style.display = "block";
        imageLoaded = true;
        setStatus("Gambar berhasil dimuat. Siap untuk deteksi.", "info");
        enableDetectIfReady();
    };

    reader.onerror = function () {
        console.error("Gagal membaca file gambar.");
        setStatus("Gagal membaca file gambar.", "error");
        imageLoaded = false;
        enableDetectIfReady();
    };

    reader.readAsDataURL(file);
});

// Event handler tombol deteksi
// ------------------------------------------------------------
detectButton.addEventListener("click", () => {
    runEmotionDetection();
});

// Inisialisasi saat halaman siap
// ------------------------------------------------------------
window.addEventListener("load", () => {
    setStatus("Memuat model, harap tunggu sebentar...", "info");
    loadModel();
});
