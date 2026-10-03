#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════╗
║          GÖRÜNTÜ İŞLEME SİSTEMİ - BAŞLATICI         ║
║  Tüm algoritmalar sıfırdan, hazır fonksiyon yok!     ║
╚══════════════════════════════════════════════════════╝

KURULUM:
  pip install flask numpy opencv-python matplotlib

ÇALIŞTIRILMASI:
  python3 run.py
  → Tarayıcıda: http://localhost:5050 açılır

DOSYA YAPISI:
  run.py              → Bu başlatıcı
  server.py           → Flask REST API sunucu
  image_processor.py  → Tüm görüntü işleme algoritmaları
  static/index.html   → Web arayüzü

UYGULANAN İŞLEMLER (hepsi sıfırdan):
  ✅ Gri Dönüşüm (ITU-R 601 formülü)
  ✅ Binary Dönüşüm
  ✅ Görüntü Döndürme (bilinear interpolasyon)
  ✅ Görüntü Kırpma
  ✅ Yaklaştırma/Uzaklaştırma (bilinear interpolasyon)
  ✅ BGR→HSV Renk Uzayı Dönüşümü
  ✅ BGR→LAB Renk Uzayı Dönüşümü
  ✅ Histogram Hesaplama ve Görüntüleme
  ✅ Histogram Germe (kontrast artırma)
  ✅ Aritmetik: Çıkarma, Çarpma
  ✅ Kontrast Azaltma
  ✅ Median Filtre (konvolüsyon)
  ✅ Çift Eşikleme
  ✅ Canny Kenar Dedektörü (G.Blur + Sobel + NMS + Hysteresis)
  ✅ Salt & Pepper Gürültü Ekleme
  ✅ Mean Filtre (gürültü giderme)
  ✅ Median Denoise (S&P + Median filtre)
  ✅ Motion Blur Filtre
  ✅ Morfolojik Genişleme (Dilation)
  ✅ Morfolojik Aşınma (Erosion)
  ✅ Morfolojik Açma (Opening)
  ✅ Morfolojik Kapama (Closing)

YASAKLI FONKSIYON KULLANILMADI:
  ❌ resize, rotate, warpAffine, filter2D
  ❌ GaussianBlur, medianBlur, blur
  ❌ Canny, Sobel, Laplacian
  ❌ threshold, adaptiveThreshold, equalizeHist
  ❌ erode, dilate, rgb2hsv vb.
"""

import os
import sys
import subprocess
import webbrowser
import time

def check_deps():
    missing = []
    for pkg in ['flask', 'numpy', 'cv2', 'matplotlib']:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"Eksik paketler: {missing}")
        print("Yükleniyor...")
        pip_names = {'cv2': 'opencv-python'}
        for m in missing:
            subprocess.run([sys.executable, '-m', 'pip', 'install',
                           pip_names.get(m, m), '--break-system-packages', '-q'])

if __name__ == '__main__':
    check_deps()

    # static klasörünü ayarla
    os.makedirs('static', exist_ok=True)
    # index.html'i static'e kopyala (eğer yan yana değilse)
    if os.path.exists('index.html') and not os.path.exists('static/index.html'):
        import shutil
        shutil.copy('index.html', 'static/index.html')

    print("╔══════════════════════════════════════╗")
    print("║  Görüntü İşleme Sistemi Başlatılıyor ║")
    print("╚══════════════════════════════════════╝")
    print("→ http://localhost:5050  adresini açın")
    print("→ Durdurmak için Ctrl+C")
    print()

    # Tarayıcıyı aç (opsiyonel)
    def open_browser():
        time.sleep(1.5)
        webbrowser.open('http://localhost:5050')

    import threading
    threading.Thread(target=open_browser, daemon=True).start()

    # Sunucuyu başlat
    import server
    server.app.run(host='0.0.0.0', port=5050, debug=False)