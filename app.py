import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
from PIL import Image

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return YOLO("best_model.pt")

model = load_model()

# =========================
# MAPPING NOMINAL (WAJIB SAMA SAAT TRAINING)
# =========================
nominal_map = {
    0: 1000,
    1: 2000,
    2: 5000,
    3: 10000,
    4: 20000,
    5: 50000,
    6: 100000
}

# =========================
# UI
# =========================
st.title("💰 Sistem Deteksi & Perhitungan Uang Rupiah")

mode = st.radio("Pilih Metode Input:", ["Upload Gambar", "Kamera"])

CONF_THRESHOLD = st.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# =========================
# PROCESS IMAGE
# =========================
def process_image(image):
    results = model(image)
    boxes = results[0].boxes

    # Gambar hasil deteksi
    img = results[0].plot()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    total = 0
    counter = Counter()
    details = []

    if boxes is not None:
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # FILTER CONFIDENCE
            if conf < CONF_THRESHOLD:
                continue

            counter[cls] += 1

    # HITUNG TOTAL
    for kelas in sorted(counter.keys()):
        nominal = nominal_map.get(kelas, 0)
        jumlah = counter[kelas]
        subtotal = nominal * jumlah
        total += subtotal

        details.append(f"Rp {nominal:,} → {jumlah} lembar = Rp {subtotal:,}")

    return img, details, total, counter

# =========================
# DISPLAY RESULT
# =========================
def show_result(image_np):
    result_img, details, total, counter = process_image(image_np)

    st.image(result_img, caption="Hasil Deteksi", use_column_width=True)

    st.subheader("📊 Ringkasan")

    if len(counter) == 0:
        st.warning("Tidak ada uang terdeteksi")
        return

    for d in details:
        st.write(d)

    st.success(f"💰 TOTAL UANG: Rp {total:,.0f}")

    # DEBUG (optional)
    with st.expander("🔍 Debug Info"):
        st.write(counter)

# =========================
# INPUT HANDLING
# =========================
if mode == "Upload Gambar":
    uploaded_file = st.file_uploader("Upload gambar uang", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        show_result(image_np)

else:
    camera_image = st.camera_input("Ambil foto")

    if camera_image is not None:
        image = Image.open(camera_image)
        image_np = np.array(image)

        show_result(image_np)
