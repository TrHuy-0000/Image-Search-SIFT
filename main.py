import sys
import os
from pathlib import Path

# Thêm src vào path
sys.path.append(str(Path(__file__).parent / 'src'))

from preprocessing import preprocess_image
from feature_extraction import FeatureExtractor
from image_matching import ImageMatcher
from utils import display_search_results 

# Cấu hình
DATA_DIR = "data"           # Thư mục chứa ảnh gốc
FEATURE_DIR = "features"    # Thư mục chứa CSV
QUERY_IMG = "data/query.jpg" # Đường dẫn ảnh input 

def main():
    print("=== HỆ THỐNG TÌM KIẾM ẢNH TƯƠNG TỰ (SIFT) ===")
    
    # BƯỚC 1: TRÍCH XUẤT ĐẶC TRƯNG 
    extractor = FeatureExtractor(algo='SIFT')
    
    # Kiểm tra xem có cần trích xuất lại không
    if not os.path.exists(FEATURE_DIR) or len(os.listdir(FEATURE_DIR)) == 0:
        print("Chưa có dữ liệu đặc trưng. Đang tạo mới...")
        extractor.process_dataset(DATA_DIR, FEATURE_DIR)
    else:
        choice = input("Bạn có muốn trích xuất lại dữ liệu không? (y/n): ")
        if choice.lower() == 'y':
            extractor.process_dataset(DATA_DIR, FEATURE_DIR)

    # BƯỚC 2: TÌM KIẾM
    if not os.path.exists(QUERY_IMG):
        print(f"Không tìm thấy ảnh query tại: {QUERY_IMG}")
        return

    print(f"\nĐang xử lý ảnh query: {QUERY_IMG}")
    
    # 2.1 Tiền xử lý ảnh query
    query_processed = preprocess_image(QUERY_IMG)
    
    # 2.2 Trích xuất đặc trưng query
    kps, query_desc = extractor.extract(query_processed)
    print(f"Tìm thấy {len(kps)} keypoints trên ảnh query.")

    if query_desc is None:
        print("Không tìm thấy đặc trưng nào trên ảnh query (ảnh quá mờ hoặc trơn).")
        return

    # 2.3 So khớp
    matcher = ImageMatcher(method='BF') # Dùng BF cho chính xác
    results = matcher.search(query_desc, FEATURE_DIR, top_n=5)

    # BƯỚC 3: HIỂN THỊ
    if len(results) == 0:
        print("Không tìm thấy ảnh nào tương tự.")
    else:
        print("\nKẾT QUẢ TOP 5:")
        paths = []
        scores = []
        for path, score in results:
            print(f" - Ảnh: {path} | Điểm match: {score}")
            paths.append(path)
            scores.append(score)
        
        try:
            display_search_results(QUERY_IMG, paths, scores, DATA_DIR)
        except Exception as e:
            print(f"Lỗi hiển thị: {e}")

if __name__ == "__main__":
    main()