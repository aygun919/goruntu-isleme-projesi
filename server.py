"""
Görüntü İşleme Web Sunucusu
Flask tabanlı REST API + Statik dosya sunucu
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
import numpy as np
import cv2
import base64
import io
import os
import sys
import traceback
import math

# image_processor modülünü import et
sys.path.insert(0, os.path.dirname(__file__))
import image_processor as ip

app = Flask(__name__, static_folder='static')

# ──────────────────────────────────────────────
# Yardımcı fonksiyonlar
# ──────────────────────────────────────────────
def decode_image(data_url):
    """Base64 data URL → numpy BGR array"""
    header, encoded = data_url.split(',', 1)
    data = base64.b64decode(encoded)
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def encode_image(img):
    """numpy array → base64 PNG data URL"""
    if img is None:
        raise ValueError("Boş görüntü")
    if len(img.shape) == 2:
        out = img
    else:
        out = img
    success, buf = cv2.imencode('.png', out)
    if not success:
        raise ValueError("Encoding hatası")
    b64 = base64.b64encode(buf).decode('utf-8')
    return 'data:image/png;base64,' + b64


def histogram_to_base64(hist, title="Histogram", color='#00d4ff'):
    """Histogram → base64 PNG (matplotlib kullanmadan, cv2 ile çiz)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 2.5), facecolor='#0a0a1a')
    ax.set_facecolor('#0d0d2b')
    ax.bar(range(256), hist, color=color, alpha=0.85, width=1)
    ax.set_xlim(0, 255)
    ax.set_title(title, color='white', fontsize=9, pad=4)
    ax.tick_params(colors='#aaaacc', labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333366')
    fig.tight_layout(pad=0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=90, bbox_inches='tight',
                facecolor='#0a0a1a', edgecolor='none')
    plt.close(fig)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    return 'data:image/png;base64,' + b64


# ──────────────────────────────────────────────
# Route: Ana sayfa
# ──────────────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


# ──────────────────────────────────────────────
# Route: İşlem endpoint'leri
# ──────────────────────────────────────────────
@app.route('/process', methods=['POST'])
def process():
    try:
        body = request.get_json(force=True)
        image_data = body.get('image')
        operation = body.get('operation')
        params = body.get('params', {})

        if not image_data or not operation:
            return jsonify({'error': 'image ve operation gerekli'}), 400

        img = decode_image(image_data)
        result = None
        extra = {}

        # ── İşlemler ──────────────────────────────
        if operation == 'grayscale':
            result = ip.to_grayscale_fast(img)

        elif operation == 'binary':
            threshold = int(params.get('threshold', 127))
            result = ip.to_binary(img, threshold)

        elif operation == 'rotate':
            angle = float(params.get('angle', 45))
            result = ip.rotate_image(img, angle)

        elif operation == 'crop':
            x = int(params.get('x', 0))
            y = int(params.get('y', 0))
            w = int(params.get('w', 100))
            h = int(params.get('h', 100))
            result = ip.crop_image(img, x, y, w, h)

        elif operation == 'resize':
            sx = float(params.get('scale_x', 1.5))
            sy = float(params.get('scale_y', 1.5))
            result = ip.resize_image(img, sx, sy)

        elif operation == 'hsv':
            result = ip.bgr_to_hsv_fast(img)

        elif operation == 'lab':
            result = ip.bgr_to_lab_fast(img)

        elif operation == 'histogram_stretch':
            gray = ip.to_grayscale_fast(img)
            hist_orig = ip.compute_histogram(img)
            result = ip.stretch_histogram(img)
            hist_stretched = ip.compute_histogram(result)
            extra['hist_orig'] = histogram_to_base64(hist_orig, 'Orijinal Histogram', '#ff6b6b')
            extra['hist_result'] = histogram_to_base64(hist_stretched, 'Gerilmiş Histogram', '#00d4ff')

        elif operation == 'histogram_view':
            hist = ip.compute_histogram(img)
            extra['hist_orig'] = histogram_to_base64(hist, 'Görüntü Histogramı', '#a78bfa')
            result = img  # Orijinali göster

        elif operation == 'subtract':
            img2_data = body.get('image2')
            if img2_data:
                img2 = decode_image(img2_data)
                # Boyutları eşitle
                h, w = min(img.shape[0], img2.shape[0]), min(img.shape[1], img2.shape[1])
                result = ip.subtract_images(img[:h,:w], img2[:h,:w])
            else:
                # Tek resimle: orijinalden gürültülüsünü çıkar
                noisy = ip.add_salt_pepper(img, 0.1)
                result = ip.subtract_images(img, noisy)

        elif operation == 'multiply':
            img2_data = body.get('image2')
            if img2_data:
                img2 = decode_image(img2_data)
                h, w = min(img.shape[0], img2.shape[0]), min(img.shape[1], img2.shape[1])
                result = ip.multiply_images(img[:h,:w], img2[:h,:w])
            else:
                # Kendi ile çarp (karanlık efekt)
                result = ip.multiply_images(img, img)

        elif operation == 'contrast_reduce':
            factor = float(params.get('factor', 0.5))
            result = ip.reduce_contrast(img, factor)

        elif operation == 'median_filter':
            ks = int(params.get('kernel_size', 3))
            result = ip.apply_median_filter(img, ks)

        elif operation == 'double_threshold':
            low = int(params.get('low', 50))
            high = int(params.get('high', 150))
            result = ip.double_threshold(img, low, high)

        elif operation == 'canny':
            low = int(params.get('low', 50))
            high = int(params.get('high', 150))
            result = ip.canny_edge(img, low, high)

        elif operation == 'salt_pepper':
            prob = float(params.get('prob', 0.05))
            result = ip.add_salt_pepper(img, prob)

        elif operation == 'mean_filter':
            ks = int(params.get('kernel_size', 3))
            result = ip.apply_mean_filter(img, ks)

        elif operation == 'median_denoise':
            prob = float(params.get('prob', 0.05))
            ks = int(params.get('kernel_size', 3))
            noisy = ip.add_salt_pepper(img, prob)
            result = ip.apply_median_filter(noisy, ks)
            extra['noisy'] = encode_image(noisy)

        elif operation == 'motion_blur':
            size = int(params.get('size', 15))
            angle = float(params.get('angle', 0))
            result = ip.motion_blur_filter(img, size, angle)

        elif operation == 'dilate':
            ks = int(params.get('kernel_size', 3))
            result = ip.morph_dilate(img, ks)

        elif operation == 'erode':
            ks = int(params.get('kernel_size', 3))
            result = ip.morph_erode(img, ks)

        elif operation == 'morph_open':
            ks = int(params.get('kernel_size', 3))
            result = ip.morph_open(img, ks)

        elif operation == 'morph_close':
            ks = int(params.get('kernel_size', 3))
            result = ip.morph_close(img, ks)

        else:
            return jsonify({'error': f'Bilinmeyen işlem: {operation}'}), 400

        if result is None:
            return jsonify({'error': 'İşlem sonuç döndürmedi'}), 500

        response = {'result': encode_image(result)}
        response.update(extra)
        return jsonify(response)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    print("Sunucu başlatılıyor: http://localhost:5050")
    app.run(host='0.0.0.0', port=5050, debug=False)