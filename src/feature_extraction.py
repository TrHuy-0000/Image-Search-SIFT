import cv2
import csv
import os
import numpy as np
from pathlib import Path
from preprocessing import preprocess_image

class FeatureExtractor:
    def __init__(self, algo='SIFT'):
        self.algo_name = algo
        # Khởi tạo thuật toán
        if algo == 'SIFT':
            self.algo = cv2.SIFT_create()
        elif algo == 'SURF':
            self.algo = cv2.xfeatures2d.SURF_create() 
        else:
            print("Mặc định dùng SIFT")
            self.algo = cv2.SIFT_create()

    def extract(self, img):
        #Trả về keypoints và descriptors
        if img is None:
            return None, None
        return self.algo.detectAndCompute(img, None)

    def process_dataset(self, data_dir, feature_dir):

        #Quét thư mục data, trích xuất và lưu vào feature_dir dưới dạng CSV

        data_path = Path(data_dir)
        feat_path = Path(feature_dir)
        feat_path.mkdir(parents=True, exist_ok=True)
        
        # Hỗ trợ các đuôi ảnh phổ biến
        valid_exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in valid_exts:
            image_files.extend(list(data_path.rglob(ext)))
            image_files.extend(list(data_path.rglob(ext.upper())))

        print(f"Tìm thấy {len(image_files)} ảnh. Bắt đầu trích xuất...")

        for idx, img_path in enumerate(image_files):
            # Tiền xử lý
            processed_img = preprocess_image(str(img_path))
            if processed_img is None:
                continue

            # Trích xuất đặc trưng
            kps, descs = self.extract(processed_img)

            if descs is not None:
                # Tạo tên file CSV unique để tránh trùng lặp
                rel_path = img_path.relative_to(data_path)
                safe_name = str(rel_path).replace(os.sep, '_').replace('.', '_')
                csv_name = f"{safe_name}.csv"
                save_path = feat_path / csv_name

                # Lưu đường dẫn gốc vào dòng đầu tiên của CSV để sau này tìm lại cho dễ
                with open(save_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    # Dòng 1: Đường dẫn tương đối của ảnh gốc
                    writer.writerow(["ORIGINAL_PATH", str(rel_path)])
                    # Các dòng sau: Vector đặc trưng
                    for desc in descs:
                        writer.writerow(desc)
            
            if idx % 50 == 0:
                print(f" Đã xử lý {idx}/{len(image_files)} ảnh")
        
        print(" Hoàn tất trích xuất đặc trưng!")

if __name__ == "__main__":
    # Test feature extraction
    extractor = FeatureExtractor(method='SIFT')
    
    # Test với 1 ảnh
    test_image = "../data/test.jpg"
    
    try:
        kp, desc = extractor.extract_from_path(test_image)
        print(f" Trích xuất thành công!")
        print(f"   - Số keypoints: {len(kp)}")
        print(f"   - Kích thước descriptor: {desc.shape}")
        
        # Lưu vào CSV
        extractor.save_features_to_csv(desc, "../features/test_features.csv", "test.jpg")
        print(f" Đã lưu features vào CSV")
        
    except Exception as e:
        print(f" Lỗi: {e}")
