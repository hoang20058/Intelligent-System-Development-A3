# HỆ THỐNG QUY CHUẨN & NGUYÊN TẮC BẮT BUỘC
## HỌC PHẦN: PHÁT TRIỂN HỆ THỐNG THÔNG MINH (INTELLIGENT SYSTEM DEVELOPMENT)

### 1. NGUYÊN TẮC PHẠM VI DỮ LIỆU & BẢO MẬT WORKSPACE
- Chỉ làm việc trong thư mục được chỉ định rõ ràng cho bài tập.
- Tách bạch cấu trúc thư mục nộp bài chuẩn (Phase 1, Phase 2, Phase 3).

### 2. QUY TẮC CẤU TRÚC JUPYTER NOTEBOOK (1 CELL = 1 NHIỆM VỤ/KẾT QUẢ ĐỘC LẬP)
- Không gộp lệnh/kết quả vào chung một cell.
- Cứ mỗi cell code thực thi ra kết quả bắt buộc có ngay 1 cell Markdown giải thích bản chất toán học và phân tích kết quả.

### 3. NGUYÊN TẮC TIỀN XỬ LÝ DỮ LIỆU (5 CELLS ĐỘC LẬP)
- Cell 1: Làm sạch dữ liệu, loại bỏ triệt để rò rỉ nhãn (Leakage Prevention).
- Cell 2: Phân tách ma trận đặc trưng X và biến mục tiêu y.
- Cell 3: Mã hóa biến danh mục (One-Hot Encoding).
- Cell 4: Phân chia Train/Test Split (80% Train - 20% Test) TRƯỚC KHI trích xuất từ điển TF-IDF (NLP).
- Cell 5: Chuẩn hóa Z-score với Mean và Std tính DUY NHẤT trên tập Train.

### 4. NGUYÊN TẮC THIẾT KẾ MÔ HÌNH: ML TRUYỀN THỐNG TRƯỚC, DEEP LEARNING SAU
- Huấn luyện các mô hình Machine Learning truyền thống (Scikit-Learn) trước.
- Huấn luyện mô hình Deep Learning (MLP thuần NumPy from scratch) sau.
- Đối sánh đa chiều bằng bảng metric đầy đủ và vẽ biểu đồ cột nhóm (Grouped Bar Chart).

### 5. TÍNH TRUNG THỰC DỮ LIỆU (ZERO HARDCODING)
- 100% kết quả và số liệu thực nghiệm được sinh tự động từ quá trình chạy thực tế trên kernel.

### 6. QUY CHUẨN BÁO CÁO HỌC THUẬT WORD (.DOCX)
- Chuẩn font Times New Roman 12pt, dãn dòng 1.25, lề chuẩn A4.
- Dành riêng khung trang trọng ở đầu báo cáo để điền Link GitHub, Link Notebook và Nguồn Dataset.
- Nhúng đầy đủ biểu đồ trực quan hóa thực tế trích xuất trực tiếp từ notebook.
