import cv2
import csv
import os
import numpy as np
from pathlib import Path

class ImageMatcher:
    def __init__(self, method='BF'):
        self.method = method
        self.bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def match(self, query_desc, db_desc):
        if query_desc is None or db_desc is None:
            return 0
        
        matcher = self.bf if self.method == 'BF' else self.flann

        try:
            matches = matcher.knnMatch(query_desc, db_desc, k=2)
            
            good_matches = []
            # Tinh chỉnh: Nới lỏng ratio test từ 0.75 lên 0.8 để bắt được nhiều điểm hơn
            ratio_thresh = 0.8 
            for m, n in matches:
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)
            
            return len(good_matches)
        except Exception:
            return 0

    def search(self, query_desc, feature_dir, top_n=5):
        results = [] 
        feat_path = Path(feature_dir)
        csv_files = list(feat_path.glob('*.csv'))

        print(f"Đang so khớp với {len(csv_files)} files trong CSDL...")

        for csv_file in csv_files:
            descriptors = []
            original_path = ""
            try:
                with open(csv_file, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header and header[0] == "ORIGINAL_PATH":
                        original_path = header[1]
                    else:
                        continue 

                    for row in reader:
                        descriptors.append([float(x) for x in row])
                
                if not descriptors: continue

                db_desc = np.array(descriptors, dtype=np.float32)
                score = self.match(query_desc, db_desc)
                
                # Giảm ngưỡng lọc xuống > 4 matches
                if score > 4: 
                    results.append((original_path, score))

            except Exception as e:
                print(f"Lỗi: {e}")
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

if __name__ == "__main__":
    # Test matching
    from feature_extraction import FeatureExtractor
    
    matcher = ImageMatcher(method='BFMatcher')
    extractor = FeatureExtractor(method='SIFT')
    
    # Test với 1 ảnh truy vấn
    query_image = "../data/test.jpg"
    
    try:
        # Trích xuất features từ query image
        kp, desc = extractor.extract_from_path(query_image)
        print(f"Trích xuất query image: {len(kp)} keypoints")
        
        # Tìm kiếm ảnh tương tự
        results = matcher.search_similar_images(desc, "../features", top_n=5)
        
        print(f"\nTop 5 ảnh tương tự:")
        for idx, (img_name, score) in enumerate(results, 1):
            print(f"   {idx}. {img_name} - Score: {score}")
        
    except Exception as e:
        print(f"Lỗi: {e}")
