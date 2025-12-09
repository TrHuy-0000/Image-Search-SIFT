# Đồ án Xử lý Ảnh - Tìm kiếm ảnh tương tự

## Mô tả
Hệ thống tìm kiếm ảnh tương tự sử dụng các đặc trưng cục bộ (SIFT/SURF) để trích xuất và so khớp ảnh.

## Cấu trúc Project
```
├── data/               # Thư mục chứa dataset ảnh
├── features/           # Lưu trữ đặc trưng đã trích xuất (.csv)
├── src/               # Source code
│   ├── preprocessing.py    # Tiền xử lý ảnh
│   ├── feature_extraction.py  # Trích xuất đặc trưng SIFT
│   ├── image_matching.py     # So khớp và tìm kiếm ảnh
│   └── utils.py             # Các hàm tiện ích
├── main.py            # File chạy chính
├── requirements.txt   # Thư viện cần thiết
└── README.md         # File này
```

## Yêu cầu
- Python 3.8+
- OpenCV
- NumPy
- Matplotlib

## Cài đặt
```bash
pip install -r requirements.txt
```

## Sử dụng

### 1. Command Line (Console)
```bash
python main.py
```

### 2. GUI Desktop App (Tkinter)
```bash
python gui_app.py
```

### 3. Web App (Streamlit)
```bash
streamlit run streamlit_app.py
```

## Tính năng
1.  Thu thập và chuẩn bị dữ liệu
2.  Tiền xử lý ảnh (Grayscale, Lọc nhiễu, CLAHE)
3.  Trích chọn đặc trưng (SIFT/SURF)
4.  So khớp và tìm kiếm ảnh tương tự (BFMatcher/FLANN)
5.  Đánh giá và hiển thị kết quả
