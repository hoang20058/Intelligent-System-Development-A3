# Intelligent System Development - Assignment 03
## Nền Tảng Học Sâu (Deep Learning from Scratch) & Đối Sánh Machine Learning Truyền Thống

Repository chứa toàn bộ mã nguồn, dữ liệu thực nghiệm và báo cáo học thuật cho Bài tập lớn số 03 (Assignment 03) - Học phần **Phát triển Hệ thống Thông minh (Intelligent System Development)**.

---

### 📌 Cấu Trúc Dự Án

```text
├── A3_submit/
│   ├── BAO_CAO_ASSIGNMENT_03_DEEP_LEARNING.docx   # Báo cáo học thuật định dạng Microsoft Word hoàn chỉnh
│   ├── REPORT_A3_DEEP_LEARNING.md                 # Báo cáo học thuật chi tiết (Markdown)
│   ├── Phase1_Diabetes/
│   │   └── diabetes_deep_learning.ipynb          # App 1: Phân loại nguy cơ tiểu đường (57 cells)
│   ├── Phase2_HousePrice/
│   │   └── house_price_prediction.ipynb          # App 2: Dự đoán giá nhà Melbourne (57 cells)
│   └── Phase3_Ecommerce/
│       └── customer_interest_text_deeplearning.ipynb # App 3: Phân loại ý định khách hàng NLP (57 cells)
├── QUY_CHUAN_THUC_HIEN_MON_HOC.md                 # Quy chuẩn kỹ thuật & nguyên tắc bắt buộc
├── .gitignore
└── README.md
```

---

### 🚀 Tổng Quan 3 Ứng Dụng Thực Nghiệm

| Ứng Dụng (Phase) | Miền Dữ Liệu & Bài Toán | Mô Hình ML Truyền Thống | Mô Hình Deep Learning | Kết Quả Chính |
|---|---|---|---|---|
| **Phase 1: Diabetes** | Bảng y tế (Tabular) - Phân loại nhị phân | Logistic Regression, Decision Tree, Random Forest | Multi-Layer Perceptron (MLP thuần NumPy) | DL đạt **Acc: 86.91%**, **F1: 88.70%**; RF đạt **92.24%** |
| **Phase 2: House Price** | Bất động sản (Tabular) - Hồi quy giá | Linear Regression, Ridge, Random Forest Regressor | Multi-Layer Perceptron (MLP thuần NumPy) | DL đạt **$R^2$: 57.58%**, **MAE: $239k AUD** (vượt OLS 50.64%) |
| **Phase 3: E-Commerce** | Văn bản (NLP) - Phân loại cảm xúc/ý định | Naive Bayes, Logistic Regression, Random Forest | MLP thuần NumPy kết hợp TF-IDF | DL đạt **Acc: 85.12%**, **F1: 91.17%** |

---

### ⚙️ Điểm Nhấn Kỹ Thuật

- **Deep Learning thuần NumPy (From Scratch)**: Tự cài đặt toàn bộ lan truyền tiến (Forward Pass), hàm kích hoạt (ReLU/Sigmoid/Linear), hàm mất mát (Binary Cross-Entropy / MSE), lan truyền ngược qua quy tắc chuỗi (Backpropagation with Chain Rule) và thuật toán Gradient Descent.
- **Quy chuẩn 1 Cell = 1 Kết quả độc lập**: Mỗi thao tác tiền xử lý, mô hình huấn luyện, biểu đồ trực quan hóa đều được tách thành cell riêng biệt kèm markdown phân tích bản chất toán học.
- **Chống rò rỉ dữ liệu (Data Leakage Prevention)**: Chuẩn hóa Z-score tính toán thuần túy trên tập `Train` và phân chia Train/Test trước khi xây dựng từ điển TF-IDF.
- **Trực quan hóa đối sánh**: Nhúng trực tiếp Grouped Bar Chart, Loss Curve, Confusion Matrix và Residual Plots.

---

### 💻 Hướng Dẫn Cài Đặt & Khởi Chạy

1. **Clone repository**:
   ```bash
   git clone https://github.com/hoang20058/Intelligent-System-Development-A3.git
   cd Intelligent-System-Development-A3
   ```

2. **Cài đặt thư viện phụ thuộc**:
   ```bash
   pip install numpy pandas scikit-learn matplotlib seaborn jupyter python-docx
   ```

3. **Mở và chạy Notebooks**:
   ```bash
   jupyter notebook
   ```
   Sau đó mở các file notebook trong thư mục `A3_submit/` để quan sát kết quả thực nghiệm.
