"""
Giao diện Web App cho hệ thống tìm kiếm ảnh tương tự (Final Fix)
"""

import sys
from pathlib import Path
import streamlit as st
from PIL import Image
import cv2
import numpy as np

# Thêm thư mục src vào path
sys.path.append(str(Path(__file__).parent / 'src'))

from feature_extraction import FeatureExtractor
from image_matching import ImageMatcher
from preprocessing import preprocess_image
from utils import count_files_in_directory

# Cấu hình trang
st.set_page_config(
    page_title="Tìm kiếm Ảnh Tương tự",
    layout="wide"
)

# CSS tùy chỉnh
st.markdown("""
    <style>
    .main {
        background-color: #f0f0f0;
    }
    .stButton>button {
        width: 100%;
        background-color: #27ae60;
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
    }
    .stButton>button:hover {
        background-color: #229954;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("Hệ thống Tìm kiếm Ảnh Tương tự")
st.markdown("---")

# Cấu hình
DATA_DIR = "data"
FEATURES_DIR = "features"

# Tạo thư mục nếu chưa tồn tại
Path(DATA_DIR).mkdir(exist_ok=True)
Path(FEATURES_DIR).mkdir(exist_ok=True)

# Sidebar - Settings
with st.sidebar:
    st.header(" Cài đặt")
    
    # Algo selection
    algo_choice = st.radio(
        "Phương pháp trích xuất:",
        ["SIFT", "SURF"],
        help="SIFT: Chính xác cao, SURF: Nhanh hơn"
    )
    
    # Matcher selection
    matcher_type = st.radio(
        "Phương pháp so khớp:",
        ["BF", "FLANN"],
        index=0,
        help="BF: Brute Force (Chính xác nhất)"
    )
    
    # Top-N
    top_n = st.slider("Số kết quả hiển thị:", 1, 20, 5)
    
    st.info("Hệ thống quét 100% dữ liệu (không sampling).")
    
    st.markdown("---")
    
    # Dataset info
    n_images = count_files_in_directory(DATA_DIR)
    n_features = len(list(Path(FEATURES_DIR).glob('*.csv')))
    
    st.metric("Số ảnh trong dataset", n_images)
    st.metric("Số features đã trích xuất", n_features)
    
    if st.button(" Trích xuất Features từ Dataset"):
        if n_images == 0:
            st.error("Không có ảnh trong dataset!")
        else:
            with st.spinner(" Đang trích xuất features..."):
                extractor = FeatureExtractor(algo=algo_choice)
                extractor.process_dataset(DATA_DIR, FEATURES_DIR)
                st.success(f" Đã trích xuất features từ {n_images} ảnh!")
                st.rerun()

# Main content
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader(" Ảnh Truy vấn")
    
    uploaded_file = st.file_uploader("Chọn ảnh:", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        # Fix warning: use_container_width -> width='stretch'
        st.image(image, caption="Query Image", width="stretch") 
        
        if st.button(" TÌM KIẾM"):
            if n_features == 0:
                st.error(" Chưa có features! Hãy nhấn nút 'Trích xuất Features' bên trái.")
            else:
                temp_path = Path("temp_query.jpg")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    with st.spinner(" Đang xử lý..."):
                        # 1. Khởi tạo extractor
                        extractor = FeatureExtractor(algo=algo_choice)
                        
                        # 2. Tiền xử lý & Trích xuất query
                        processed_query = preprocess_image(str(temp_path))
                        kps, query_desc = extractor.extract(processed_query)
                        
                        if query_desc is None:
                            st.error(" Ảnh quá mờ hoặc không có chi tiết đặc trưng.")
                        else:
                            # 3. Tìm kiếm
                            matcher = ImageMatcher(method=matcher_type)
                            results = matcher.search(query_desc, FEATURES_DIR, top_n=top_n)
                            
                            st.session_state.results = results
                            st.session_state.query_kp_count = len(kps)
                    
                    if temp_path.exists(): temp_path.unlink()
                        
                except Exception as e:
                    st.error(f" Lỗi: {e}")

with col2:
    st.subheader(" Kết quả Tìm kiếm")
    
    if 'results' in st.session_state and st.session_state.results:
        results = st.session_state.results
        
        st.info(f" Tìm thấy {len(results)} kết quả (Score > 4 match)")
        
        if len(results) == 0:
            st.warning("Không tìm thấy ảnh nào đủ giống (Score quá thấp).")
        
        for idx, (img_rel_path, score) in enumerate(results, 1):
            full_path = Path(DATA_DIR) / img_rel_path
            
            if full_path.exists():
                with st.container():
                    r_col1, r_col2 = st.columns([1, 2])
                    with r_col1:
                        result_img = Image.open(full_path)
                        st.image(result_img, width="stretch")
                    with r_col2:
                        st.markdown(f"### #{idx}")
                        st.markdown(f"**File:** `{full_path.name}`")
                        
                        # Đánh giá độ tin cậy dựa trên score SIFT
                        if score < 10:
                            quality = " Rất thấp (Nhiễu)"
                        elif score < 30:
                            quality = " Trung bình"
                        else:
                            quality = " Cao (Rất giống)"
                            
                        st.markdown(f"**Score:** {score} matches ({quality})")
                        st.progress(min(score / 100, 1.0))
                    st.markdown("---")
            else:
                st.warning(f"Không tìm thấy file: {img_rel_path}")
    else:
        st.info("Chọn ảnh và nhấn 'Tìm kiếm' để xem kết quả")
# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #7f8c8d;'>"
    "Hệ thống Tìm kiếm Ảnh Tương tự - Sử dụng SIFT/SURF"
    "</div>",
    unsafe_allow_html=True
)