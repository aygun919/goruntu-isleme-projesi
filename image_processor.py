"""
Görüntü İşleme Kütüphanesi - Tüm algoritmalar sıfırdan implementasyon
Yasak fonksiyonlar kullanılmamıştır.
"""

import numpy as np
import math


# ─────────────────────────────────────────────
# 1. GRİ DÖNÜŞÜM
# ─────────────────────────────────────────────
def to_grayscale(img):
    """BGR → Gri: ITU-R 601 formülü"""
    if len(img.shape) == 2:
        return img.copy()
    gray = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b, g, r = float(img[i, j, 0]), float(img[i, j, 1]), float(img[i, j, 2])
            gray[i, j] = int(0.114 * b + 0.587 * g + 0.299 * r)
    return gray


def to_grayscale_fast(img):
    """Vektörize hızlı gri dönüşüm"""
    if len(img.shape) == 2:
        return img.copy()
    b = img[:, :, 0].astype(np.float64)
    g = img[:, :, 1].astype(np.float64)
    r = img[:, :, 2].astype(np.float64)
    return np.clip(0.114 * b + 0.587 * g + 0.299 * r, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# 2. BİNARY DÖNÜŞÜM
# ─────────────────────────────────────────────
def to_binary(img, threshold=127):
    """Gri görüntüyü binary'e çevirir"""
    gray = to_grayscale_fast(img) if len(img.shape) == 3 else img
    result = np.zeros_like(gray, dtype=np.uint8)
    result[gray >= threshold] = 255
    return result


# ─────────────────────────────────────────────
# 3. GÖRÜNTÜ DÖNDÜRME
# ─────────────────────────────────────────────
def rotate_image(img, angle_deg):
    """Bilinear interpolasyon ile görüntü döndürme"""
    angle = math.radians(angle_deg)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)

    # Yeni boyutları hesapla
    new_w = int(abs(w * cos_a) + abs(h * sin_a))
    new_h = int(abs(w * sin_a) + abs(h * cos_a))
    ncx, ncy = new_w / 2.0, new_h / 2.0

    channels = 1 if len(img.shape) == 2 else img.shape[2]
    if channels == 1:
        result = np.zeros((new_h, new_w), dtype=np.uint8)
    else:
        result = np.zeros((new_h, new_w, channels), dtype=np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            # Orijinal koordinata dönüşüm
            x = cos_a * (j - ncx) + sin_a * (i - ncy) + cx
            y = -sin_a * (j - ncx) + cos_a * (i - ncy) + cy
            x0, y0 = int(x), int(y)
            if 0 <= x0 < w - 1 and 0 <= y0 < h - 1:
                dx = x - x0
                dy = y - y0
                if channels == 1:
                    result[i, j] = int(
                        (1 - dx) * (1 - dy) * img[y0, x0] +
                        dx * (1 - dy) * img[y0, x0 + 1] +
                        (1 - dx) * dy * img[y0 + 1, x0] +
                        dx * dy * img[y0 + 1, x0 + 1]
                    )
                else:
                    for c in range(channels):
                        result[i, j, c] = int(
                            (1 - dx) * (1 - dy) * img[y0, x0, c] +
                            dx * (1 - dy) * img[y0, x0 + 1, c] +
                            (1 - dx) * dy * img[y0 + 1, x0, c] +
                            dx * dy * img[y0 + 1, x0 + 1, c]
                        )
    return result


# ─────────────────────────────────────────────
# 4. GÖRÜNTÜ KIRPMA
# ─────────────────────────────────────────────
def crop_image(img, x, y, w, h):
    """Görüntüyü kırpar"""
    img_h, img_w = img.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    x2 = max(0, min(x + w, img_w))
    y2 = max(0, min(y + h, img_h))
    return img[y:y2, x:x2].copy()


# ─────────────────────────────────────────────
# 5. GÖRÜNTÜ YAKLAŞTIRMA / UZAKLAŞTIRMA (Bilinear)
# ─────────────────────────────────────────────
def resize_image(img, scale_x, scale_y):
    """Bilinear interpolasyon ile yeniden boyutlandırma"""
    h, w = img.shape[:2]
    new_h = max(1, int(h * scale_y))
    new_w = max(1, int(w * scale_x))
    channels = 1 if len(img.shape) == 2 else img.shape[2]

    if channels == 1:
        result = np.zeros((new_h, new_w), dtype=np.uint8)
    else:
        result = np.zeros((new_h, new_w, channels), dtype=np.uint8)

    for i in range(new_h):
        for j in range(new_w):
            src_x = j * (w - 1) / (new_w - 1) if new_w > 1 else 0
            src_y = i * (h - 1) / (new_h - 1) if new_h > 1 else 0
            x0, y0 = int(src_x), int(src_y)
            x1 = min(x0 + 1, w - 1)
            y1 = min(y0 + 1, h - 1)
            dx = src_x - x0
            dy = src_y - y0
            if channels == 1:
                result[i, j] = int(
                    (1-dx)*(1-dy)*img[y0,x0] + dx*(1-dy)*img[y0,x1] +
                    (1-dx)*dy*img[y1,x0] + dx*dy*img[y1,x1]
                )
            else:
                for c in range(channels):
                    result[i, j, c] = int(
                        (1-dx)*(1-dy)*img[y0,x0,c] + dx*(1-dy)*img[y0,x1,c] +
                        (1-dx)*dy*img[y1,x0,c] + dx*dy*img[y1,x1,c]
                    )
    return result


# ─────────────────────────────────────────────
# 6. RENK UZAYI DÖNÜŞÜMLERİ
# ─────────────────────────────────────────────
def bgr_to_hsv(img):
    """BGR → HSV manuel dönüşüm"""
    result = np.zeros_like(img, dtype=np.uint8)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            b, g, r = img[i,j,0]/255.0, img[i,j,1]/255.0, img[i,j,2]/255.0
            cmax = max(r, g, b)
            cmin = min(r, g, b)
            delta = cmax - cmin
            # Hue
            if delta == 0:
                h = 0
            elif cmax == r:
                h = 60 * (((g - b) / delta) % 6)
            elif cmax == g:
                h = 60 * (((b - r) / delta) + 2)
            else:
                h = 60 * (((r - g) / delta) + 4)
            # Saturation
            s = 0 if cmax == 0 else delta / cmax
            v = cmax
            result[i,j] = [int(h/2), int(s*255), int(v*255)]
    return result


def bgr_to_hsv_fast(img):
    """Vektörize HSV dönüşümü"""
    img_f = img.astype(np.float64) / 255.0
    b, g, r = img_f[:,:,0], img_f[:,:,1], img_f[:,:,2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin

    h = np.zeros_like(cmax)
    mask_r = (cmax == r) & (delta != 0)
    mask_g = (cmax == g) & (delta != 0)
    mask_b = (cmax == b) & (delta != 0)
    h[mask_r] = 60 * (((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6)
    h[mask_g] = 60 * (((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2)
    h[mask_b] = 60 * (((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4)

    s = np.where(cmax == 0, 0, delta / cmax)
    v = cmax

    result = np.stack([h/2, s*255, v*255], axis=2)
    return np.clip(result, 0, 255).astype(np.uint8)


def bgr_to_lab_fast(img):
    """BGR → LAB (D65 illuminant)"""
    img_f = img.astype(np.float64) / 255.0
    b, g, r = img_f[:,:,0], img_f[:,:,1], img_f[:,:,2]
    # Linear
    def linearize(c):
        return np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    rl, gl, bl = linearize(r), linearize(g), linearize(b)
    # XYZ
    X = 0.4124564*rl + 0.3575761*gl + 0.1804375*bl
    Y = 0.2126729*rl + 0.7151522*gl + 0.0721750*bl
    Z = 0.0193339*rl + 0.1191920*gl + 0.9503041*bl
    # Normalize D65
    X /= 0.95047; Z /= 1.08883
    def f(t):
        return np.where(t > 0.008856, t ** (1/3), 7.787*t + 16/116)
    fx, fy, fz = f(X), f(Y), f(Z)
    L = np.clip(116*fy - 16, 0, 100)
    a = np.clip(500*(fx - fy) + 128, 0, 255)
    b_ch = np.clip(200*(fy - fz) + 128, 0, 255)
    return np.stack([L/100*255, a, b_ch], axis=2).astype(np.uint8)


# ─────────────────────────────────────────────
# 7. HİSTOGRAM ve HİSTOGRAM GERME
# ─────────────────────────────────────────────
def compute_histogram(img):
    """Gri histogram hesapla (256 bin)"""
    gray = to_grayscale_fast(img) if len(img.shape) == 3 else img
    hist = np.zeros(256, dtype=np.int64)
    for val in gray.flatten():
        hist[val] += 1
    return hist


def stretch_histogram(img):
    """Histogram germe (kontrast artırma)"""
    gray = to_grayscale_fast(img) if len(img.shape) == 3 else img
    p_min = int(gray.min())
    p_max = int(gray.max())
    if p_max == p_min:
        return gray.copy()
    result = ((gray.astype(np.float64) - p_min) / (p_max - p_min) * 255).astype(np.uint8)
    return result


# ─────────────────────────────────────────────
# 8. ARİTMETİK İŞLEMLER
# ─────────────────────────────────────────────
def subtract_images(img1, img2):
    """İki görüntü farkı"""
    a = img1.astype(np.int16)
    b = img2.astype(np.int16)
    return np.clip(np.abs(a - b), 0, 255).astype(np.uint8)


def multiply_images(img1, img2):
    """İki görüntü çarpımı (normalize)"""
    a = img1.astype(np.float64) / 255.0
    b = img2.astype(np.float64) / 255.0
    return np.clip(a * b * 255, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# 9. KONTRAST AZALTMA
# ─────────────────────────────────────────────
def reduce_contrast(img, factor=0.5):
    """Kontrastı azalt: yeni = 128 + factor*(orijinal-128)"""
    img_f = img.astype(np.float64)
    result = 128 + factor * (img_f - 128)
    return np.clip(result, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# 10. KONVOLÜSYON (Median)
# ─────────────────────────────────────────────
def apply_median_filter(img, kernel_size=3):
    """Median filtre - konvolüsyon yerine sıralama"""
    h, w = img.shape[:2]
    pad = kernel_size // 2
    channels = 1 if len(img.shape) == 2 else img.shape[2]

    if channels == 1:
        padded = np.pad(img, pad, mode='edge')
        result = np.zeros_like(img)
        for i in range(h):
            for j in range(w):
                region = padded[i:i+kernel_size, j:j+kernel_size]
                result[i, j] = int(np.median(region.flatten()))
    else:
        result = np.zeros_like(img)
        for c in range(channels):
            padded = np.pad(img[:,:,c], pad, mode='edge')
            for i in range(h):
                for j in range(w):
                    region = padded[i:i+kernel_size, j:j+kernel_size]
                    result[i, j, c] = int(np.median(region.flatten()))
    return result


def convolve2d(img_channel, kernel):
    """2D konvolüsyon (tek kanal)"""
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    h, w = img_channel.shape
    padded = np.pad(img_channel.astype(np.float64), ((ph, ph), (pw, pw)), mode='edge')
    result = np.zeros((h, w), dtype=np.float64)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            result[i, j] = np.sum(region * kernel)
    return result


# ─────────────────────────────────────────────
# 11. EŞİKLEME (Çift Eşikleme)
# ─────────────────────────────────────────────
def double_threshold(img, low=50, high=150):
    """Çift eşikleme: düşük → gri, yüksek → beyaz, arada → gri"""
    gray = to_grayscale_fast(img) if len(img.shape) == 3 else img
    result = np.zeros_like(gray, dtype=np.uint8)
    result[gray >= high] = 255
    result[(gray >= low) & (gray < high)] = 128
    return result


# ─────────────────────────────────────────────
# 12. KENAR BULMA - CANNY (Manuel)
# ─────────────────────────────────────────────
def gaussian_kernel(size=5, sigma=1.4):
    """Gaussian kernel oluştur"""
    k = size // 2
    kernel = np.zeros((size, size), dtype=np.float64)
    total = 0.0
    for x in range(-k, k+1):
        for y in range(-k, k+1):
            val = math.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel[x+k, y+k] = val
            total += val
    return kernel / total


def canny_edge(img, low=50, high=150):
    """Canny kenar dedektörü - manuel implementasyon"""
    gray = to_grayscale_fast(img) if len(img.shape) == 3 else img.copy()

    # 1. Gaussian blur
    gk = gaussian_kernel(5, 1.4)
    smoothed = convolve2d(gray, gk)

    # 2. Sobel gradyanları
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float64)
    Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float64)
    Gx = convolve2d(smoothed, Kx)
    Gy = convolve2d(smoothed, Ky)
    G = np.sqrt(Gx**2 + Gy**2)
    theta = np.arctan2(Gy, Gx) * 180 / math.pi
    theta[theta < 0] += 180

    # 3. Non-maximum suppression
    h, w = G.shape
    nms = np.zeros((h, w), dtype=np.float64)
    for i in range(1, h-1):
        for j in range(1, w-1):
            angle = theta[i, j]
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q, r = G[i, j+1], G[i, j-1]
            elif 22.5 <= angle < 67.5:
                q, r = G[i+1, j-1], G[i-1, j+1]
            elif 67.5 <= angle < 112.5:
                q, r = G[i+1, j], G[i-1, j]
            else:
                q, r = G[i-1, j-1], G[i+1, j+1]
            nms[i, j] = G[i, j] if G[i, j] >= q and G[i, j] >= r else 0

    # 4. Çift eşikleme
    result = np.zeros((h, w), dtype=np.uint8)
    strong = nms >= high
    weak = (nms >= low) & (nms < high)
    result[strong] = 255
    result[weak] = 128

    # 5. Hysteresis - zayıf kenarları güçlü komşular ile bağla
    for i in range(1, h-1):
        for j in range(1, w-1):
            if result[i, j] == 128:
                if np.any(result[i-1:i+2, j-1:j+2] == 255):
                    result[i, j] = 255
                else:
                    result[i, j] = 0
    return result


# ─────────────────────────────────────────────
# 13. GÜRÜLTÜ EKLEME (Salt & Pepper)
# ─────────────────────────────────────────────
def add_salt_pepper(img, prob=0.05):
    """Salt & Pepper gürültüsü ekle"""
    result = img.copy()
    rng = np.random.default_rng(42)
    noise = rng.random(img.shape[:2])
    result[noise < prob/2] = 0
    result[noise > 1 - prob/2] = 255
    return result


def apply_mean_filter(img, kernel_size=3):
    """Ortalama (mean) filtre"""
    h, w = img.shape[:2]
    pad = kernel_size // 2
    channels = 1 if len(img.shape) == 2 else img.shape[2]
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / (kernel_size**2)

    if channels == 1:
        padded = np.pad(img.astype(np.float64), pad, mode='edge')
        result = np.zeros((h, w), dtype=np.uint8)
        for i in range(h):
            for j in range(w):
                result[i, j] = int(np.sum(padded[i:i+kernel_size, j:j+kernel_size] * kernel))
    else:
        result = np.zeros_like(img)
        for c in range(channels):
            padded = np.pad(img[:,:,c].astype(np.float64), pad, mode='edge')
            for i in range(h):
                for j in range(w):
                    result[i, j, c] = int(np.sum(padded[i:i+kernel_size, j:j+kernel_size] * kernel))
    return result


# ─────────────────────────────────────────────
# 14. MOTİON BLUR FİLTRE
# ─────────────────────────────────────────────
def motion_blur_filter(img, size=15, angle=0):
    """Yatay motion blur filtresi"""
    kernel = np.zeros((size, size), dtype=np.float64)
    mid = size // 2
    kernel[mid, :] = 1.0 / size
    # Açı için kernel döndür
    if angle != 0:
        cos_a = math.cos(math.radians(angle))
        sin_a = math.sin(math.radians(angle))
        rot_k = np.zeros((size, size), dtype=np.float64)
        for x in range(size):
            for y in range(size):
                sx = int(cos_a*(x-mid) + sin_a*(y-mid) + mid)
                sy = int(-sin_a*(x-mid) + cos_a*(y-mid) + mid)
                if 0 <= sx < size and 0 <= sy < size:
                    rot_k[y, x] += kernel[sy, sx]
        s = rot_k.sum()
        kernel = rot_k / s if s > 0 else rot_k

    channels = 1 if len(img.shape) == 2 else img.shape[2]
    if channels == 1:
        result = np.clip(convolve2d(img.astype(np.float64), kernel), 0, 255).astype(np.uint8)
    else:
        result = np.zeros_like(img)
        for c in range(channels):
            result[:,:,c] = np.clip(convolve2d(img[:,:,c].astype(np.float64), kernel), 0, 255).astype(np.uint8)
    return result


# ─────────────────────────────────────────────
# 15. MORFOLOJİK İŞLEMLER
# ─────────────────────────────────────────────
def get_structuring_element(size=3):
    """Dikdörtgen yapısal eleman"""
    return np.ones((size, size), dtype=np.uint8)


def morph_dilate(img, kernel_size=3):
    """Genişleme (Dilation)"""
    gray = to_grayscale_fast(img) if len(img.shape) == 3 else img
    se = get_structuring_element(kernel_size)
    h, w = gray.shape
    pad = kernel_size // 2
    padded = np.pad(gray, pad, mode='constant', constant_values=0)
    result = np.zeros_like(gray)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = int(np.max(region[se == 1]))
    return result


def morph_erode(img, kernel_size=3):
    """Aşınma (Erosion)"""
    gray = to_grayscale_fast(img) if len(img.shape) == 3 else img
    se = get_structuring_element(kernel_size)
    h, w = gray.shape
    pad = kernel_size // 2
    padded = np.pad(gray, pad, mode='constant', constant_values=255)
    result = np.zeros_like(gray)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = int(np.min(region[se == 1]))
    return result


def morph_open(img, kernel_size=3):
    """Açma = Aşınma + Genişleme"""
    return morph_dilate(morph_erode(img, kernel_size), kernel_size)


def morph_close(img, kernel_size=3):
    """Kapama = Genişleme + Aşınma"""
    return morph_erode(morph_dilate(img, kernel_size), kernel_size)