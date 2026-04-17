import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter
from PIL import Image

# Load model
model = YOLO("best_model.pt")

# Mapping nominal
nominal_map = {
    0: 1000,
    1: 10000,
    2: 100000,
    3: 2000,
    4: 20000,
    5: 5000,
    6: 50000
}

st.title("💰 Sistem Deteksi dan Perhitungan Uang Rupiah")

mode = st.radio("Pilih Metode Input:", ["Upload Gambar", "Kamera"])

def process_image(image):
    results = model(image)
    boxes = results[0].boxes

    img = results[0].plot()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    classes = boxes.cls.cpu().numpy().astype(int)
    counter = Counter(classes)

    total = 0
    details = []

    for kelas, jumlah in counter.items():
        nominal = nominal_map[kelas]
        subtotal = nominal * jumlah
        total += subtotal
        details.append(f"Rp{nominal:,} → {jumlah} lembar = Rp{subtotal:,}")

    return img, details, total


if mode == "Upload Gambar":
    uploaded_file = st.file_uploader("Upload gambar uang", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        result_img, details, total = process_image(image_np)

        st.image(result_img)
        st.subheader("Detail Perhitungan")

        for d in details:
            st.write(d)

        st.success(f"TOTAL UANG: Rp{total:,.0f}")

else:
    camera_image = st.camera_input("Ambil foto")

    if camera_image is not None:
        image = Image.open(camera_image)
        image_np = np.array(image)

        result_img, details, total = process_image(image_np)

        st.image(result_img)
        st.subheader("Detail Perhitungan")

        for d in details:
            st.write(d)

        st.success(f"TOTAL UANG: Rp{total:,.0f}")