from ultralytics import YOLO
import torch

def main():
    # Cek apakah GPU terdeteksi sebelum mulai
    if torch.cuda.is_available():
        print(f"✅ GPU Terdeteksi: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ WARNING: GPU tidak terdeteksi! Training akan berjalan lambat di CPU.")

    # 1. Load Model
    model = YOLO('yolov8n-seg.pt')  

    # 2. Mulai Training
    # Konfigurasi optimal untuk GTX 1660 Super (6GB VRAM)
    results = model.train(
        data='dataset/data.yaml',  # Pastikan path ini benar
        epochs=100,
        imgsz=640,
        
        # Batch size: Jumlah gambar yang diproses sekali jalan.
        # Untuk GTX 1660 Super (6GB):
        # - Model Nano (n-seg): Bisa batch=16 atau 32
        # - Model Small (s-seg): Gunakan batch=16
        # - Model Medium (m-seg): Turunkan ke batch=8
        batch=16,
        
        name='latih_gpu_1660s',    
        patience=20,
        
        # SETTING PENTING UNTUK GPU & WINDOWS:
        device=0,      # 0 artinya gunakan GPU NVIDIA pertama
        workers=4,     # Di Windows, jangan terlalu tinggi (max 4-8) agar tidak error multiprocessing
        cache=False    # False agar RAM komputer tidak penuh (hemat RAM)
    )

    print("Training selesai.")

if __name__ == '__main__':
    # Baris ini wajib ada di Windows untuk menghindari error multiprocessing
    main()
