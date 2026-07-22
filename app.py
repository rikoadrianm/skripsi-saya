import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import time
from datetime import datetime

# ==================================
# PAGE CONFIG
# ==================================
st.set_page_config(
    page_title="Deteksi Kerusakan Kemasan",
    page_icon="📦",
    layout="wide"
)

# ==================================
# CSS (simpel, tanpa dependensi eksternal)
# ==================================
st.markdown("""
<style>
.block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 2rem; }

.main-title {
    text-align: center;
    font-size: 30px;
    font-weight: 700;
    margin-bottom: 4px;
}

.sub-title {
    text-align: center;
    font-size: 14px;
    color: #64748B;
    margin-bottom: 24px;
}

.card {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 20px;
}

div[data-testid="stMetric"] {
    background-color: var(--secondary-background-color);
    border: 1px solid rgba(128, 128, 128, 0.15);
    border-radius: 8px;
    padding: 10px 14px;
}
</style>
""", unsafe_allow_html=True)

# ==================================
# LOAD MODEL
# ==================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_kemasan.h5")

model = load_model()

# ==================================
# HEADER
# ==================================
st.markdown('<div class="main-title">📦 Sistem Deteksi Kerusakan Kemasan</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Computer Vision Berbasis Deep Learning (MobileNetV2)</div>', unsafe_allow_html=True)

# ==================================
# SIDEBAR
# ==================================
with st.sidebar:
    st.subheader("ℹ️ Informasi Model")
    st.write("**Model** : MobileNetV2")
    st.write("**Epoch** : 10")
    st.write("**Dataset** : 1365")
    st.write("**Kelas** : 3 (OK, NG, Unknown)")

    st.markdown("---")

    st.subheader("📈 Performa Model")
    st.write("**Accuracy** : 93%")
    st.write("**Precision** : 94%")
    st.write("**Recall** : 93%")
    st.write("**F1-Score** : 93%")

# ==================================
# PILIH INPUT
# ==================================
st.subheader("📷 Input Gambar")

opsi = st.radio(
    "Pilih Metode Input",
    ["Upload Gambar", "Webcam"],
    horizontal=True,
    label_visibility="collapsed"
)

img = None

# ==================================
# UPLOAD FILE
# ==================================
if opsi == "Upload Gambar":
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# ==================================
# WEBCAM
# ==================================
else:
    camera = st.camera_input("Ambil Foto")

    if camera is not None:
        file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

# ==================================
# PROSES
# ==================================
if img is not None:

    start_time = time.time()

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Gambar Input")
        st.image(img, channels="BGR", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💻 Status Sistem")

        progress = st.progress(0)
        status = st.empty()

        status.info("Memuat gambar...")
        progress.progress(20)
        time.sleep(0.5)

        status.info("Resize gambar menjadi 224x224...")
        img_resized = cv2.resize(img, (224, 224))
        progress.progress(40)
        time.sleep(0.5)

        status.info("Normalisasi nilai piksel...")
        img_normalized = img_resized / 255.0
        progress.progress(60)
        time.sleep(0.5)

        status.info("Menjalankan CNN MobileNetV2...")
        img_input = np.expand_dims(img_normalized, axis=0)
        prediction = model.predict(img_input, verbose=0)
        progress.progress(80)
        time.sleep(0.5)

        status.info("Menghitung hasil prediksi...")
        progress.progress(100)
        time.sleep(0.5)

        status.success("Proses analisis selesai")
        st.markdown('</div>', unsafe_allow_html=True)

    end_time = time.time()
    lama_proses = end_time - start_time

    # ==================================
    # HASIL
    # ==================================
    kelas = np.argmax(prediction)
    confidence = np.max(prediction)
    labels = ["NG", "OK", "Unknown"]
    hasil = labels[kelas]

    st.markdown("---")
    st.subheader("📊 Hasil Analisis")

    st.markdown('<div class="card">', unsafe_allow_html=True)
    colA, colB, colC = st.columns(3)

    with colA:
        if hasil == "OK":
            st.success("OK - Kondisi Baik")
            st.write("Kemasan terdeteksi dalam kondisi baik dan tidak ditemukan kerusakan visual.")
        elif hasil == "NG":
            st.error("NG - Terdeteksi Kerusakan")
            st.write("Kemasan terdeteksi mengalami kerusakan berdasarkan pola yang dipelajari model.")
        else:
            st.warning("Unknown - Objek Tidak Dikenal")
            st.write("Objek tidak termasuk kategori kemasan mie instan atau produk yang didukung.")

    with colB:
        st.metric("Confidence", f"{confidence*100:.2f}%")

    with colC:
        st.metric("Waktu Proses", f"{lama_proses:.3f} detik")

    st.markdown('</div>', unsafe_allow_html=True)

    # ==================================
    # PROBABILITAS
    # ==================================
    st.markdown("---")
    st.subheader("📈 Probabilitas Kelas")

    ng = float(prediction[0][0])
    ok = float(prediction[0][1])
    unknow = float(prediction[0][2])

    st.markdown('<div class="card">', unsafe_allow_html=True)

    st.write(f"NG : {ng*100:.2f}%")
    st.progress(ng)

    st.write(f"OK : {ok*100:.2f}%")
    st.progress(ok)

    st.write(f"Unknown : {unknow*100:.2f}%")
    st.progress(unknow)

    st.markdown('</div>', unsafe_allow_html=True)

    # ==================================
    # DETAIL
    # ==================================
    st.markdown("---")
    st.subheader("📋 Rincian Hasil Klasifikasi")

    st.dataframe(
        {
            "Kelas": ["NG", "OK", "Unknown"],
            "Probabilitas (%)": [
                round(ng * 100, 2),
                round(ok * 100, 2),
                round(unknow * 100, 2)
            ]
        },
        use_container_width=True
    )

    # ==================================
    # LOG PROSES
    # ==================================
    st.markdown("---")
    st.subheader("🖥️ Log Sistem")

    log_text = f"""Waktu Analisis : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

[SUCCESS] Memuat gambar selesai.
[SUCCESS] Preprocessing: Resize ke 224x224.
[SUCCESS] Preprocessing: Normalisasi piksel.
[SUCCESS] Model: Inference MobileNetV2 selesai.
[SUCCESS] Output: Hasil klasifikasi ditampilkan."""

    st.code(log_text, language=None)

# ==================================
# FOOTER
# ==================================
st.markdown("---")
st.caption("Sistem Deteksi Kerusakan Kemasan Mie Instan — Computer Vision & Deep Learning (MobileNetV2)")
