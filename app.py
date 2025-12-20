import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import os # [Baru] Import os untuk cek keberadaan file default

# Konfigurasi Halaman
st.set_page_config(page_title="Aplikasi Segmentasi YOLOv8", layout="centered")

st.title("🔍 Deteksi & Segmentasi Objek")
st.write("Upload gambar untuk melakukan segmentasi otomatis.")

# 1. Load Model (Gunakan cache agar tidak reload setiap kali ada interaksi)
@st.cache_resource
def load_model():
    # Pastikan file 'best.pt' ada di satu folder dengan app.py
    return YOLO('best.pt')

try:
    model = load_model()
except Exception as e:
    st.error(f"Error memuat model. Pastikan file 'best.pt' ada. Error: {e}")

# 2. Upload Gambar
uploaded_file = st.file_uploader("Pilih gambar...", type=["jpg", "jpeg", "png"])

# [Baru] Logika Penentuan File Input (Upload vs Default)
file_to_process = None
use_default = False
default_image_path = "default_image.png"

if uploaded_file is not None:
    file_to_process = uploaded_file
elif os.path.exists(default_image_path):
    file_to_process = default_image_path
    use_default = True

# Proses jika ada file yang dipilih (Upload atau Default)
if file_to_process is not None:
    
    # Tampilkan pesan jika menggunakan default
    if use_default:
        st.info(f"ℹ️ Tidak ada file diupload. Menggunakan gambar default: `{default_image_path}`")
    
    # Tampilkan gambar asli
    # Image.open bisa membaca path string maupun file object
    image = Image.open(file_to_process)
    
    # Buat kolom agar tampilan rapi (Kiri: Asli, Kanan: Hasil)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Gambar Asli")
        st.image(image, use_container_width=True)

    # Tombol Prediksi
    # Jika menggunakan default, user tetap bisa menekan tombol ini untuk melihat hasil
    if st.button("Lakukan Segmentasi"):
        with st.spinner('Sedang memproses...'):
            try:
                # 3. Lakukan Prediksi
                # conf=0.5 sesuai settingan Anda sebelumnya
                results = model.predict(source=image, conf=0.5)

                # 4. Visualisasi Hasil
                # Kita ambil plot dari result pertama
                res_plotted = results[0].plot()
                
                # Konversi warna: YOLO plot outputnya BGR (OpenCV format), 
                # Streamlit butuh RGB. Kita perlu balik warnanya.
                res_plotted_rgb = res_plotted[:, :, ::-1]

                with col2:
                    st.subheader("Hasil Segmentasi")
                    st.image(res_plotted_rgb, caption="Hasil Deteksi", use_container_width=True)
                
                # (Opsional) Menampilkan jumlah objek
                if results[0].masks is not None:
                    jumlah_objek = len(results[0].masks)
                    st.success(f"✅ Ditemukan {jumlah_objek} objek.")
                else:
                    st.warning("Tidak ada objek yang terdeteksi.")
            except Exception as e:
                st.error(f"Terjadi kesalahan saat memproses gambar: {e}")

elif not use_default and uploaded_file is None:
    # Hanya muncul jika tidak upload DAN file default tidak ditemukan
    st.warning("Silakan upload gambar untuk memulai.")
