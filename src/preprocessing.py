import cv2
import numpy as np

def preprocess_image(image_path, blur_method='gaussian'):
    """
    Quy trình tiền xử lý:
    1. Chuyển ảnh xám (Grayscale)
    2. Lọc nhiễu (Gaussian Blur hoặc Median Blur)
    3. Cân bằng sáng (CLAHE)
    
    Args:
        image_path: Đường dẫn đến ảnh
        blur_method: 'gaussian' hoặc 'median'
    """
    # 1. Đọc ảnh
    img = cv2.imread(image_path)
    if img is None:
        return None

    # 2. Chuyển sang ảnh xám
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3. Lọc nhiễu
    if blur_method == 'gaussian':
        # Gaussian Blur: Tốt cho nhiễu Gaussian, làm mờ đều
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    elif blur_method == 'median':
        # Median Blur: Tốt cho nhiễu muối tiêu (salt-and-pepper)
        gray = cv2.medianBlur(gray, 5)

    # 4. Tăng tương phản cục bộ (CLAHE)
    # Giúp SIFT nhìn rõ chi tiết dù ảnh tối hay sáng
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    return gray