"""
Module các hàm tiện ích
- Hiển thị kết quả
- Đánh giá performance
- Load/save data
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import time


def display_image(image, title="Image", cmap=None):
    """Hiển thị 1 ảnh"""
    plt.figure(figsize=(8, 6))
    if cmap:
        plt.imshow(image, cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def display_multiple_images(images, titles, rows=1, cols=None, figsize=(15, 5)):
    """Hiển thị nhiều ảnh"""
    if cols is None:
        cols = len(images)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    if rows * cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 2:  # Grayscale
            axes[idx].imshow(img, cmap='gray')
        else:  # Color
            axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[idx].set_title(title)
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def display_search_results(query_image_path, result_image_paths, scores, data_dir):

    n_results = len(result_image_paths)
    if n_results == 0:
        print("Không có kết quả để hiển thị.")
        return

    fig, axes = plt.subplots(1, n_results + 1, figsize=(4 * (n_results + 1), 4))
    
    # Xử lý trường hợp chỉ có 1 kết quả khiến axes không phải là mảng
    if n_results + 1 == 1:
        axes = [axes]
    
    # Hiển thị query image
    query_img = cv2.imread(query_image_path)
    if query_img is not None:
        axes[0].imshow(cv2.cvtColor(query_img, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Query Image", fontsize=12, fontweight='bold')
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, "Image not found", ha='center')
        axes[0].axis('off')
    
    # Hiển thị kết quả
    for idx, (img_rel_path, score) in enumerate(zip(result_image_paths, scores), 1):
        # Đường dẫn ảnh = data_dir + đường dẫn tương đối (đã bao gồm tên và đuôi file)
        full_path = Path(data_dir) / img_rel_path
        
        if full_path.exists():
            img = cv2.imread(str(full_path))
            if img is not None:
                axes[idx].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axes[idx].set_title(f"#{idx}\nScore: {score}", fontsize=10)
                axes[idx].axis('off')
            else:
                axes[idx].text(0.5, 0.5, "Error reading img", ha='center')
                axes[idx].axis('off')
        else:
            axes[idx].text(0.5, 0.5, "File not found", ha='center')
            axes[idx].axis('off')
            print(f"Không tìm thấy file: {full_path}")
    
    plt.tight_layout()
    plt.show()

# Các hàm benchmark 
def benchmark_feature_extraction(extractor, image_path, n_runs=10):
    times = []
    for _ in range(n_runs):
        start = time.time()
        extractor.extract_from_path(image_path)
        end = time.time()
        times.append((end - start) * 1000)
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f" Thời gian trích xuất đặc trưng:")
    print(f"   - Trung bình: {avg_time:.2f} ms")
    print(f"   - Độ lệch chuẩn: {std_time:.2f} ms")
    return avg_time

def benchmark_matching(matcher, desc1, desc2, n_runs=10):
    times = []
    for _ in range(n_runs):
        start = time.time()
        matcher.match_features(desc1, desc2)
        end = time.time()
        times.append((end - start) * 1000)
    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f" Thời gian so khớp:")
    print(f"   - Trung bình: {avg_time:.2f} ms")
    print(f"   - Độ lệch chuẩn: {std_time:.2f} ms")
    return avg_time

def count_files_in_directory(directory, extensions=['.jpg', '.jpeg', '.png', '.bmp'], recursive=True):
    """
    Đếm số file ảnh trong thư mục.
    """
    count = 0
    glob_method = Path(directory).rglob if recursive else Path(directory).glob
    for ext in extensions:
        # Chỉ cần glob 1 lần
        count += len(list(glob_method(f'*{ext}')))
    return count

if __name__ == "__main__":
    print("Module utils đã sẵn sàng!")