# BÁO CÁO BÀI TẬP LỚN (ASSIGNMENT 03)
# HỌC PHẦN: PHÁT TRIỂN HỆ THỐNG THÔNG MINH (INTELLIGENT SYSTEM DEVELOPMENT)
## ĐỀ TÀI: NỀN TẢNG HỌC SÂU (DEEP LEARNING FROM SCRATCH) & TRIỂN KHAI THỰC NGHIỆM ĐA MIỀN DỮ LIỆU

---

**Thông tin chung:**
- **Học phần**: Phát triển Hệ thống Thông minh (Intelligent System Development)
- **Năm học**: 2026 – 2027
- **Chủ đề thực nghiệm**:
  1. Phân loại nhị phân nguy cơ tiểu đường (*Diabetes Prediction* - Dữ liệu bảng dạng Vector số/danh mục)
  2. Hồi quy định giá bất động sản (*Melbourne House Price Prediction* - Dữ liệu bảng liên tục)
  3. Phân tích ý kiến và dự đoán mức độ quan tâm của khách hàng (*E-Commerce Customer Interest Discovery* - Dữ liệu văn bản NLP)
- **Môi trường triển khai**: Python 3.13 / Jupyter Notebook (Anaconda)
- **Thư mục sản phẩm**: `A3_submit/`

---

# PHẦN I: TỔNG QUAN PHƯƠNG PHÁP LUẬN & BẢN CHẤT HỌC SÂU (DEEP LEARNING FOUNDATIONS)

> *Cơ sở lý thuyết: Bám sát các bài giảng Slide 03 (Slide 1–20 của bài giảng Foundations và Slide 1–17 của bài giảng Implementation).*

---

### 1. Khái Niệm Cốt Lõi Về Học Sâu (The Central Concept)

#### Bản chất toán học: Phép hợp thành hàm số (Function Composition)
Học sâu (Deep Learning) không phải là một "chiếc hộp đen huyền bí" mà bản chất toán học của nó là **sự hợp thành của các hàm số tham số hóa (Composition of Parameterized Functions)**:
$$\hat{y} = F_\theta(x) = (f_L \circ f_{L-1} \circ \dots \circ f_2 \circ f_1)(x)$$
Trong đó:
- $x$: Tensor đầu vào đại diện cho dữ liệu quan sát.
- $f_l(\cdot)$: Phép biến đổi phi tuyến tại tầng thứ $l$, được tham số hóa bởi ma trận trọng số $W_l$ và vector độ lệch $b_l$:
  $$h_l = f_l(h_{l-1}) = \sigma(h_{l-1} W_l + b_l)$$
- $\theta = \{W_1, b_1, W_2, b_2, \dots, W_L, b_L\}$: Toàn bộ không gian tham số có thể học được (trainable parameters) của mạng.

#### Chu trình học tập tối ưu (Learning Loop)
Một hệ thống học sâu hoàn chỉnh vận hành qua 4 giai đoạn khép kín có tính lặp:
$$\text{Forward Pass (Lan truyền xuôi)} \longrightarrow \text{Loss Computation (Tính mất mát)} \longrightarrow \text{Backpropagation (Lan truyền ngược qua Chain Rule)} \longrightarrow \text{Gradient Descent Update (Cập nhật trọng số)}$$

1. **Lan truyền xuôi (Forward Pass)**: Biến đổi dữ liệu đầu vào $x$ qua từng tầng để tạo ra các biểu diễn tiềm ẩn trung gian ($h_1, h_2, \dots$) và đưa ra giá trị dự báo $\hat{y}$.
2. **Tính toán độ mất mát (Loss Computation)**: Hàm mục tiêu $\mathcal{L}(y, \hat{y})$ lượng hóa sai số giữa dự đoán $\hat{y}$ và nhãn thực nghiệm $y$.
3. **Lan truyền ngược (Backpropagation)**: Áp dụng có hệ thống **Quy tắc chuỗi (Chain Rule)** trên đồ thị tính toán để tính đạo hàm riêng $\nabla_\theta \mathcal{L} = \frac{\partial \mathcal{L}}{\partial \theta}$.
4. **Cập nhật tham số (Gradient Descent)**: Tịnh tiến trọng số ngược chiều gradient với tốc độ học $\eta$:
   $$\theta \longleftarrow \theta - \eta \nabla_\theta \mathcal{L}$$

---

### 2. So Sánh Machine Learning Truyền Thống và Deep Learning (From Feature Engineering to Representation Learning)

#### Sự khác biệt về đường ống xử lý (Pipeline Comparison)
- **Quy trình Machine Learning truyền thống (Traditional ML Pipeline)**:
  $$x \xrightarrow{\quad \text{Kỹ sư trích xuất đặc trưng thủ công} \quad} \phi_{\text{human}}(x) \xrightarrow{\quad \text{Mô hình ML nông} \quad} y$$
  Trong ML truyền thống, chất lượng của hệ thống phụ thuộc hoàn toàn vào kỹ năng thiết kế đặc trưng của con người (Domain Expertise & Handcrafted Feature Engineering). Bản thân thuật toán (như Logistic Regression, SVM, Random Forest) chỉ học một ranh giới quyết định nông trên các đặc trưng đã được cố định sẵn.
- **Quy trình Học sâu (Deep Learning Pipeline)**:
  $$x \xrightarrow{\quad f_{\theta_1} \quad} h_1 \xrightarrow{\quad f_{\theta_2} \quad} h_2 \xrightarrow{\quad \dots \quad} h_L \xrightarrow{\quad \text{Predictor} \quad} \hat{y}$$
  Trong Deep Learning, không có sự tách rời giữa bộ trích xuất đặc trưng và bộ phân loại. Mô hình **tự động học biểu diễn phân cấp (Hierarchical Representation Learning)** cùng lúc với hàm dự báo thông qua tín hiệu phản hồi từ hàm mất mát (End-to-End Learning).

#### Câu hỏi trung tâm (Central Question - Slide 59):
> *"Where does the representation come from?"*
> - Trong ML truyền thống: Biểu diễn do **con người áp đặt từ bên ngoài** ($\phi_{\text{human}}$).
> - Trong Deep Learning: Biểu diễn do **mạng nơ-ron tự động tối ưu hóa từ dữ liệu** dựa trên bài toán mục tiêu.

---

### 3. Ý Nghĩa Của Tầng (Layer) và Không Gian Tensor (Data as Tensors)

- **Bản chất của một Tầng (Layer)**: Một tầng không đơn thuần là tập hợp các vòng tròn nơ-ron mà là một **phép biến đổi tọa độ không gian (Coordinate Transformation)**. Dữ liệu đầu vào ở hệ tọa độ cũ được ánh xạ sang một không gian tiềm ẩn mới (Latent Space) sao cho các điểm dữ liệu trở nên dễ phân tách tuyến tính hơn hoặc dễ xấp xỉ hàm mục tiêu hơn.
- **Dữ liệu dưới dạng Tensor (Tensors Across Modalities)**:
  - **Dữ liệu bảng (Tabular Data - Diabetes, House Price)**: Tensor 2 chiều $X \in \mathbb{R}^{N \times d}$ ($N$ mẫu quan sát, $d$ thuộc tính đo lường).
  - **Dữ liệu văn bản (NLP - Customer Reviews)**: Tensor 2D thưa $X \in \mathbb{R}^{N \times V}$ (TF-IDF với từ điển $V$) hoặc Tensor 3D tuần tự $X \in \mathbb{R}^{N \times T \times d}$ ($T$ bước thời gian/từ, $d$ chiều vector embedding).
  - **Dữ liệu hình ảnh (Computer Vision - Skin Lesion)**: Tensor 4 chiều $X \in \mathbb{R}^{N \times H \times W \times C}$ (Chiều cao $H$, Chiều rộng $W$, Số kênh màu $C$).

---

### 4. Tại Sao Mạng Nơ-ron Bắt Buộc Cần Hàm Kích Hoạt Phi Tuyến (Nonlinearity)?

#### Chứng minh toán học sự suy biến của mạng tuyến tính xếp chồng (Linear Collapse):
Giả sử ta xây dựng một mạng nơ-ron sâu 2 tầng nhưng **không sử dụng** bất kỳ hàm kích hoạt phi tuyến nào:
- Tầng 1: $h_1 = X W_1 + b_1$
- Tầng 2: $\hat{y} = h_1 W_2 + b_2$

Thay $h_1$ vào phương trình tầng 2:
$$\hat{y} = (X W_1 + b_1) W_2 + b_2 = X (W_1 W_2) + (b_1 W_2 + b_2)$$
Đặt $W' = W_1 W_2$ và $b' = b_1 W_2 + b_2$. Khi đó:
$$\hat{y} = X W' + b'$$
> **Kết luận toán học**: Tích của hai ma trận $W_1 W_2$ chỉ là một ma trận $W'$ mới. Một mạng nơ-ron tuyến tính dù có xếp chồng $100$ tầng thì về mặt đại số tuyến tính chỉ hoàn toàn tương đương với một mô hình hồi quy tuyến tính đơn tầng phẳng ($W'x + b'$).  
> **Do đó**: Các hàm kích hoạt phi tuyến như $\text{ReLU}(z) = \max(0, z)$ là thành phần sống còn bẻ cong không gian, giúp mạng nơ-ron có năng lực xấp xỉ vạn năng (Universal Approximation Theorem) đối với các mặt cong phức tạp.

---

### 5. Bản Chất Lan Truyền Ngược (Backpropagation) và Frameworks

- **Lan truyền ngược (Backpropagation)**: Là việc áp dụng có trật tự quy tắc chuỗi giải tích (Chain Rule) từ tầng cuối cùng ngược về tầng đầu tiên. Nhờ lưu trữ các giá trị trung gian ($Z_l, H_l$) trong bộ nhớ đệm (Cache) ở pha lan truyền xuôi, mạng tính toán đạo hàm với độ phức tạp tuyến tính $O(|\theta|)$ thay vì phải tính vi phân số học cực kỳ chậm chạp.
- **From Scratch (NumPy) vs Thư viện Tự động (PyTorch/TensorFlow)**:
  - Việc tự viết 100% bằng NumPy giúp ta hiểu sâu sắc từng phép nhân ma trận, cách định hình tensor và bản chất của đạo hàm.
  - Các framework hiện đại (PyTorch, TensorFlow, JAX) không thay đổi bản chất toán học của Deep Learning. Chúng chỉ cung cấp hai tính năng kỹ thuật: **Đồ thị vi phân tự động (Autograd)** và **Tăng tốc tính toán song song trên GPU/TPU (CUDA C++)**.

---

# PHẦN II: TRIỂN KHAI VÀ THỰC NGHIỆM 3 CHỦ ĐỀ DỮ LIỆU

---

## CHƯƠNG 1: CHỦ ĐỀ 01 – DIABETES PREDICTION (PHÂN LOẠI NHỊ PHÂN TRÊN DỮ LIỆU BẢNG)

> **Notebook**: [A3_submit/Phase1_Diabetes/diabetes_deep_learning.ipynb](file:///d:/JJin/Documents/Học/I-4/Thiet_ke_httm/jupyter/A3/A3_submit/Phase1_Diabetes/diabetes_deep_learning.ipynb) (57 cells: 25 code cells + 32 markdown cells).

---

### 1. Khám Phá & Tiền Xử Lý Dữ Liệu (5 Cells Độc Lập)

Quy trình tiền xử lý được chia tách thành 5 cells độc lập có in kết quả thực nghiệm và giải thích:
1. **Bước 1 - Loại bỏ rò rỉ nhãn (Data Leakage Removal)**: Cột `diabetes_stage` chứa nhãn giai đoạn tiến triển bệnh được loại bỏ hoàn toàn (`df_clean = df_raw.drop(columns=['diabetes_stage'])`).
2. **Bước 2 - Tách biến dự đoán $X$ và nhãn $y$**: Tách nhãn nhị phân $y \in \{0, 1\}$ ($100,000 \times 1$) và bảng thuộc tính lâm sàng $X$ ($100,000 \times 30$).
3. **Bước 3 - Mã hóa One-Hot Encoding**: Mã hóa các biến danh mục (giới tính, tình trạng hút thuốc...) với `drop_first=True`, tạo ra không gian đặc trưng số thực $d = 41$ chiều.
4. **Bước 4 - Phân chia Train/Test Split**: Hoán vị ngẫu nhiên chia $80,000$ mẫu Train ($80\%$) và $20,000$ mẫu Test ($20\%$).
5. **Bước 5 - Chuẩn hóa Z-Score**: Tính trung bình $\mu$ và độ lệch chuẩn $\sigma$ **DUY NHẤT trên tập Train**; chuẩn hóa $X_{\text{train}}$ và $X_{\text{test}}$, đồng thời tạo vector 1D cho Scikit-Learn.

---

### 2. Huấn Luyện Machine Learning Truyền Thống (Scikit-Learn)

Mỗi mô hình được huấn luyện trong 1 cell code độc lập và đánh giá trên $20,000$ bệnh nhân tập Test:
- **Mô hình 1: Logistic Regression**: Đạt Accuracy = $86.09\%$, F1 = $88.57\%$ ($0.07$ giây). Đóng vai trò là đường cơ sở tuyến tính (Linear Baseline).
- **Mô hình 2: Decision Tree Classifier**: Đạt Accuracy = $92.17\%$, Precision = $99.74\%$, F1 = $93.04\%$ ($0.30$ giây). Mô hình phân nhánh rất hiệu quả theo các ngưỡng y tế sắc nét (như chỉ số đường huyết glucose, HbA1c).
- **Mô hình 3: Random Forest Classifier**: Đạt Accuracy = $92.24\%$, Precision = $99.89\%$, F1 = $93.09\%$ ($1.28$ giây). Dẫn đầu nhóm thuật toán cây nhờ cơ chế Bagging 100 cây.

---

### 3. Xây Dựng & Huấn Luyện Deep Learning Thuần NumPy (from Scratch)

- **Kiến trúc mạng 3 tầng**: $41 \to 32 \to 16 \to 1$
  - Khởi tạo He Initialization: $W_l \sim \mathcal{N}\left(0, \sqrt{\frac{2}{d_{\text{in}}}}\right)$, $b_l = 0$.
  - Tầng ẩn dùng kích hoạt phi tuyến $\text{ReLU}(z) = \max(0, z)$.
  - Tầng ra dùng hàm kích hoạt xác suất $\text{Sigmoid}(z) = \frac{1}{1 + e^{-z}}$.
  - Hàm mất mát Binary Cross-Entropy (BCE): $\mathcal{L} = -\frac{1}{N}\sum [y \ln \hat{y} + (1-y)\ln(1-\hat{y})]$.
  - Đạo hàm tầng ra qua Chain Rule: $dZ_3 = \frac{1}{N}(\hat{y} - y)$.
- **Quá trình huấn luyện**: Chạy 500 epoch với Full-batch Gradient Descent ($\eta = 0.1$), loss giảm mượt mà từ $0.6931$ xuống $0.3087$.
- **Đánh giá trên tập kiểm thử (Test Set)**:
  - Accuracy = **$86.91\%$** | Precision = **$91.94\%$** | Recall = **$85.67\%$** | F1-Score = **$88.70\%$** (Thời gian huấn luyện: $31.93$ giây).
- **Ma trận nhầm lẫn (Confusion Matrix Heatmap)**: TP = $10,280$ ($51.4\%$), TN = $7,102$ ($35.5\%$), FP = $901$ ($4.5\%$), FN = $1,717$ ($8.6\%$).
- **Bảng trích xuất 10 bệnh nhân mẫu**: Thể hiện sự nhất quán tuyệt đối giữa trọng số thực tế và xác suất dự đoán ($\hat{y} \in [0.03, 0.98]$).

---

### 4. Bảng Đối Chuẩn & Biểu Đồ Cột Nhóm 4 Mô Hình (App 1)

| STT | Mô hình (Model) | Accuracy | Precision | Recall | F1-Score | Train Time (s) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Logistic Regression** | 86.09% | 87.32% | **89.86%** | 88.57% | **0.07s** |
| 2 | **Decision Tree** | 92.17% | 99.74% | 87.17% | 93.04% | 0.30s |
| 3 | **Random Forest** | **92.24%** | **99.89%** | 87.16% | **93.09%** | 1.28s |
| 4 | **Deep Learning (NumPy MLP)** | **86.91%** | **91.94%** | 85.67% | **88.70%** | 31.93s |

*Biểu đồ Grouped Bar Chart đã được vẽ và nhúng trực tiếp trong Cell 19 của Notebook.*

---

### 5. Chuỗi Thực Nghiệm Khảo Sát Toàn Diện (Exercises 1–7)

1. **Thực nghiệm 1 (Architecture Width)**: Mạng nhỏ $(16, 8) \to 86.87\%$; Mạng chuẩn $(32, 16) \to 86.91\%$; Mạng lớn $(64, 32) \to 86.90\%$. Cấu hình $(32, 16)$ đạt điểm bão hòa tối ưu.
2. **Thực nghiệm 2 (Learning Rate)**: $\eta = 0.001$ hội tụ quá chậm (loss $0.62$); $\eta = 0.1$ hội tụ lý tưởng (loss $0.31$); $\eta = 0.5$ có dao động nhẹ.
3. **Thực nghiệm 3 (Deep vs Shallow)**: Mạng sâu 2 tầng ẩn ($86.91\%$) vượt trội hơn mạng nông 1 tầng ẩn ($86.85\%$) nhờ khả năng tổng hợp biểu diễn phi tuyến đa cấp.
4. **Thực nghiệm 4 (Loại bỏ ReLU)**: Khi bỏ ReLU, mạng suy biến về Linear Classifier và độ chính xác tụt xuống đúng bằng mức của Logistic Regression ($86.09\%$).
5. **Thực nghiệm 5 (Ngưỡng quyết định $\tau$)**: Với $\tau = 0.2$, Recall đạt $99.2\%$ (không bỏ sót bệnh nhân); với $\tau = 0.8$, Precision đạt $98.1\%$. Ngưỡng $\tau = 0.5$ cân bằng F1 cao nhất ($88.7\%$).
6. **Thực nghiệm 6 (Khảo sát Tensor)**: $X_{5 \times 41} \to H_{1, 5 \times 32} \to H_{2, 5 \times 16} \to \hat{y}_{5 \times 1}$.

---

## CHƯƠNG 2: CHỦ ĐỀ 02 – MELBOURNE HOUSE PRICE PREDICTION (HỒI QUY ĐỊNH GIÁ BẤT ĐỘNG SẢN)

> **Notebook**: [A3_submit/Phase2_HousePrice/house_price_prediction.ipynb](file:///d:/JJin/Documents/Học/I-4/Thiet_ke_httm/jupyter/A3/A3_submit/Phase2_HousePrice/house_price_prediction.ipynb) (57 cells: 25 code cells + 32 markdown cells).

---

### 1. Khám Phá & Tiền Xử Lý Dữ Liệu (5 Cells Độc Lập)

1. **Bước 1 - Lọc missing target**: Loại bỏ $14,590$ căn nhà thiếu giá bán thực tế (`Price` = NaN), giữ lại $48,433$ giao dịch hợp lệ.
2. **Bước 2 - Trích xuất thời gian & Tách $X, y$**: Trích xuất `Year`, `Month` từ ngày bán; tách $y \in \mathbb{R}^{N \times 1}$ và các đặc trưng `Rooms`, `Distance`, `Propertycount`, `Type`, `Method`, `Regionname`.
3. **Bước 3 - One-Hot Encoding**: Mã hóa các biến danh mục tạo không gian đặc trưng số thực $d = 18$ chiều.
4. **Bước 4 - Train/Test Split**: Chia ngẫu nhiên $38,746$ căn Train ($80\%$) và $9,687$ căn Test ($20\%$).
5. **Bước 5 - Chuẩn hóa Z-Score cho cả $X$ và $y$ (Slide 27, 29)**: Tính $\mu_X, \sigma_X, \mu_y, \sigma_y$ thuần túy trên tập Train. Chuẩn hóa cả $X$ và $y$ để đưa giá trị về phân phối $\mathcal{N}(0, 1)$, ngăn chặn nổ gradient.

---

### 2. Huấn Luyện Machine Learning Hồi Quy Truyền Thống (Scikit-Learn)

- **Mô hình 1: Linear Regression (OLS)**: MAE = $\$268,373.90$ AUD, $R^2 = 0.5064$ ($50.64\%$). Giới hạn của siêu phẳng phẳng không nắm bắt được tương tác vị trí và số phòng.
- **Mô hình 2: Decision Tree Regressor**: MAE = $\$190,115.84$ AUD, $R^2 = 0.7088$ ($70.88\%$). Phân vùng siêu chữ nhật phi tuyến theo khu vực địa lý giúp giảm gần $\$80,000$ AUD sai số.
- **Mô hình 3: Random Forest Regressor**: MAE = **$\$179,825.75$ AUD**, $R^2 = \mathbf{0.7162}$ ($71.62\%$). Dẫn đầu nhờ kết hợp 100 cây làm mịn bề mặt dự đoán giá.

---

### 3. Xây Dựng & Huấn Luyện Deep Learning Hồi Quy Thuần NumPy (from Scratch)

- **Kiến trúc hồi quy (Slide 26–30)**: $18 \to 64 \to 32 \to 1$
  - Tầng ẩn kích hoạt $\text{ReLU}$, **tầng ra tuyến tính thuần túy**: $\hat{y} = Z_3 \in (-\infty, +\infty)$ (Slide 29 - Tuyệt đối không dùng Sigmoid).
  - Hàm mất mát Mean Squared Error (MSE): $\mathcal{L} = \frac{1}{N}\sum (\hat{y} - y)^2$.
  - Đạo hàm tầng ra: $dZ_3 = \frac{2}{N}(\hat{y} - y)$.
  - **Quy trình giải chuẩn hóa (Denormalization - Slide 29)**:
    $$y_{\text{real}} = \mu_y + \sigma_y \cdot \hat{y}$$
- **Kết quả đánh giá trên tập kiểm thử ($9,687$ căn nhà)**:
  - MAE = **$\$239,434.93$ AUD** | RMSE = **$\$392,993.77$ AUD** | $R^2\text{-Score} = \mathbf{0.5758}$ ($57.58\%$) (Thời gian huấn luyện: $32.00$ giây).
  - **Vượt trội hơn Linear Regression ($50.64\%$)**: Giảm gần $\$29,000$ AUD sai số tuyệt đối trung bình.
- **Đồ thị phân tán (Scatter Plot)**: Điểm dữ liệu bám sát đường lý tưởng $y = x$ trong phân khúc $0.5 - 2.5$ triệu AUD.
- **Bảng 10 căn nhà mẫu**: Sai số thực tế dao động từ $4.3\% - 19.5\%$ ở các phân khúc nhà phổ thông.

---

### 4. Bảng Đối Chuẩn & Biểu Đồ Cột Nhóm 4 Mô Hình (App 2)

| STT | Mô hình (Model) | MAE (AUD) | RMSE (AUD) | $R^2$-Score | Train Time (s) |
|:---:|---|:---:|:---:|:---:|:---:|
| 1 | **Linear Regression (OLS)** | $268,373.90 AUD | $423,902.66 AUD | 0.5064 (50.64%) | **0.01s** |
| 2 | **Decision Tree Regressor** | $190,115.84 AUD | $325,582.64 AUD | 0.7088 (70.88%) | 0.05s |
| 3 | **Random Forest Regressor** | **$179,825.75 AUD** | **$321,413.87 AUD** | **0.7162 (71.62%)** | 1.22s |
| 4 | **Deep Learning (NumPy MLP)** | **$239,434.93 AUD** | **$392,993.77 AUD** | **0.5758 (57.58%)** | 32.00s |

*Biểu đồ Grouped Bar Chart đối chiếu đồng thời $R^2$ (%) và MAE (nghìn AUD) được nhúng sẵn trong Cell 19.*

---

### 5. Chuỗi Thực Nghiệm Khảo Sát Deep Learning Hồi Quy

1. **Thực nghiệm 1 (Width)**: $(32, 16) \to R^2 = 0.4905$; $(64, 32) \to R^2 = 0.5463$; $(128, 64) \to R^2 = 0.5619$. Cấu hình $(64, 32)$ là điểm cân bằng tối ưu.
2. **Thực nghiệm 2 (Learning Rate)**: $\eta = 0.01$ đem lại đường cong suy giảm MSE ổn định nhất.
3. **Thực nghiệm 3 (Deep vs Shallow)**: Mạng sâu ($R^2 = 54.63\%$) vượt mạng nông ($R^2 = 52.52\%$).
4. **Thực nghiệm 4 (Loại bỏ ReLU)**: Khi bỏ ReLU, $R^2$ tụt từ $54.63\%$ xuống đúng $50.41\%$ (suy biến hoàn toàn về Linear Regression).
5. **Thực nghiệm 5 (Tầm quan trọng của Target Normalization)**:
   - Khi có chuẩn hóa $y$: MSE loss ổn định $\approx 0.40$, gradient được kiểm soát hoàn hảo.
   - Khi để $y$ thô (tính bằng triệu AUD): MSE loss ban đầu vọt lên tới **$1.35 \times 10^{12}$**, gradient bùng nổ làm tràn số hệ thống!
6. **Thực nghiệm 6 (Khảo sát Tensor)**: $X_{5 \times 18} \to H_{1, 5 \times 64} \to H_{2, 5 \times 32} \to \hat{y}_{5 \times 1}$.

---

## CHƯƠNG 3: CHỦ ĐỀ 03 – E-COMMERCE REVIEWS & CUSTOMER INTEREST (XỬ LÝ NGÔN NGỮ TỰ NHIÊN - NLP)

> **Notebook**: [A3_submit/Phase3_Ecommerce/customer_interest_text_deeplearning.ipynb](file:///d:/JJin/Documents/Học/I-4/Thiet_ke_httm/jupyter/A3/A3_submit/Phase3_Ecommerce/customer_interest_text_deeplearning.ipynb) (57 cells: 25 code cells + 32 markdown cells).

---

### 1. Khám Phá & Tiền Xử Lý Dữ Liệu NLP (5 Cells Độc Lập)

1. **Bước 1 - Lọc missing review text**: Loại bỏ $845$ bản ghi thiếu nội dung phản hồi, giữ lại $22,641$ đánh giá văn bản hợp lệ.
2. **Bước 2 - Ghép văn bản & Tách nhãn**: Kết hợp `Title` và `Review Text` thành chuỗi văn bản hoàn chỉnh; tách nhãn mục tiêu `Recommended IND`.
3. **Bước 3 - Phân chia Train/Test Split TRƯỚC KHI trích xuất từ vựng**: Chia tập Train ($18,112$ mẫu) và Test ($4,529$ mẫu) trước để chống rò rỉ từ điển (Data Snooping).
4. **Bước 4 - Trích xuất TF-IDF (Slide 31-32)**: Xây dựng từ điển $V = 250$ từ/cụm từ cảm xúc cốt lõi, loại bỏ stop words.
5. **Bước 5 - Chuẩn hóa Z-Score**: Chuẩn hóa ma trận đặc trưng TF-IDF trên tập Train và tạo vector 1D cho Scikit-Learn.

---

### 2. Huấn Luyện Machine Learning Truyền Thống (Scikit-Learn)

- **Mô hình 1: Logistic Regression**: Đạt Accuracy = **$87.46\%$**, Precision = $89.89\%$, Recall = $95.27\%$, F1 = **$92.50\%$** ($0.07$ giây). Đây là mô hình cực kỳ mạnh mẽ trên không gian từ vựng Bag-of-Words/TF-IDF.
- **Mô hình 2: Decision Tree Classifier**: Đạt Accuracy = $82.78\%$, F1 = $89.94\%$ ($0.65$ giây).
- **Mô hình 3: Random Forest Classifier**: Đạt Accuracy = $83.82\%$, Recall = $99.35\%$, F1 = $90.89\%$ ($0.61$ giây).

---

### 3. Xây Dựng & Huấn Luyện Deep Learning Phân Loại Văn Bản Thuần NumPy

- **Kiến trúc mạng NLP 3 tầng (Slide 31–36, 56)**: $250 \to 64 \to 32 \to 1$
  - Kích hoạt ẩn $\text{ReLU}$, tầng ra $\text{Sigmoid}$, hàm mất mát BCE.
  - Lan truyền ngược qua Chain Rule tính đạo hàm chính xác: $dZ_3 = \frac{1}{N}(\hat{y} - y)$.
  - Huấn luyện 400 chu kỳ với Gradient Descent ($\eta = 0.05$), loss giảm mượt từ $0.7818$ xuống $0.2982$.
- **Kết quả kiểm thử ($4,529$ nhận xét)**:
  - Accuracy = **$85.12\%$** | Precision = **$87.97\%$** | Recall = **$94.62\%$** | F1-Score = **$91.17\%$** (Thời gian huấn luyện: $18.34$ giây).
  - Vượt trội hơn cả Decision Tree ($82.78\%$) và tương đương Random Forest ($83.82\%$).
- **Ma trận nhầm lẫn**: TP = $3,481$ ($76.9\%$), TN = $374$ ($8.3\%$), FP = $476$ ($10.5\%$), FN = $198$ ($4.4\%$).
- **Bảng 10 nhận xét mẫu**: Dự đoán chuẩn xác cả các câu khen ngợi ($\hat{y} > 90\%$) và phàn nàn ($\hat{y} < 30\%$).

---

### 4. Bảng Đối Chuẩn & Biểu Đồ Cột Nhóm 4 Mô Hình (App 3)

| STT | Mô hình (Model) | Accuracy | Precision | Recall | F1-Score | Train Time (s) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| 1 | **Logistic Regression** | **87.46%** | **89.89%** | 95.27% | **92.50%** | **0.07s** |
| 2 | **Decision Tree** | 82.78% | 85.59% | 94.75% | 89.94% | 0.65s |
| 3 | **Random Forest** | 83.82% | 83.75% | **99.35%** | 90.89% | 0.61s |
| 4 | **Deep Learning (NumPy MLP)** | **85.12%** | **87.97%** | 94.62% | **91.17%** | 18.34s |

*Biểu đồ Grouped Bar Chart so sánh trực quan Accuracy và F1-Score được nhúng sẵn trong Cell 19.*

---

### 5. Giải Quyết Chi Tiết Toàn Diện Bài Tập Exercise 3 – Customer Comment (Slide 66)

Xét câu nhận xét kinh điển trong bài giảng:
> **"The camera is excellent but the battery is poor."**

Nhóm trả lời trọn vẹn 6 câu hỏi bản chất của Thầy:
1. **Đầu vào thô (Raw input) là gì?**  
   $\to$ Là chuỗi ký tự thô (Raw text string) hoặc danh sách các token: `["The", "camera", "is", "excellent", "but", "the", "battery", "is", "poor"]`.
2. **TF-IDF biểu diễn câu này như thế nào?**  
   $\to$ Ánh xạ câu thành một vector thưa $x \in \mathbb{R}^V$. Trong đó, các vị trí ứng với các từ `"camera"`, `"excellent"`, `"battery"`, `"poor"` mang giá trị trọng số thực dương, tất cả các từ còn lại trong từ điển bằng $0$.
3. **TF-IDF làm mất đi thông tin gì? (Hạn chế của Bag-of-Words - Slide 33)**  
   $\to$ **Mất hoàn toàn thứ tự từ (Word Order) và quan hệ cấu trúc cú pháp**. TF-IDF không biết tính từ `"excellent"` bổ nghĩa cho danh từ `"camera"` hay danh từ `"battery"`. Đặc biệt, mô hình túi từ bị vô hiệu hóa trước hiện tượng phủ định: câu *"The product is good"* và *"The product is not good"* có vector gần như giống hệt nhau dù ý nghĩa đối nghịch hoàn toàn.
4. **Tại sao Word Embeddings (Slide 34) lại vượt trội hơn?**  
   $\to$ Embeddings ánh xạ mỗi từ thành một vector dày đặc trong không gian liên tục $e \in \mathbb{R}^d$ ($d = 100$ hoặc $300$). Không gian này bảo toàn tính tương đồng ngữ nghĩa: khoảng cách giữa `excellent` và `great` rất gần nhau, giải quyết triệt để sự rời rạc của One-Hot/TF-IDF.
5. **Tại sao Ngữ cảnh (Context) lại quan trọng?**  
   $\to$ Một từ có ý nghĩa thay đổi tùy ngữ cảnh: từ `"poor"` đi với `"battery"` mang nghĩa thời lượng pin ngắn (phần cứng), khác với nghĩa nghèo nàn trong kinh tế.
6. **Mô hình Deep Learning nâng cao nào có thể xử lý tốt chuỗi này? (Slide 36)**  
   $\to$ **RNN / LSTM** (xử lý tuần tự có cổng nhớ) hoặc **Transformer / Self-Attention** (như BERT, RoBERTa, GPT). Cơ chế Self-Attention cho phép từ `"excellent"` chú ý trực tiếp đến `"camera"` và `"poor"` chú ý trực tiếp đến `"battery"`, phân tách rõ ràng hai khía cạnh (Aspect-Based Sentiment Analysis).

---

# PHẦN III: TỔNG HỢP SO SÁNH LIÊN CHỦ ĐỀ & KẾT LUẬN SƯ PHẠM

---

### 1. Bảng Khái Quát Hóa 3 Dạng Dữ Liệu (Bám Sát Slide 56)

| Tiêu Chí So Sánh | Chủ Đề 01: Diabetes Prediction | Chủ Đề 02: House Price Prediction | Chủ Đề 03: Customer Interest NLP |
|---|---|---|---|
| **Dạng dữ liệu (Modality - Slide 56)** | Vector bảng số học & phân loại có cấu trúc | Vector bảng thuộc tính nhà & địa lý liên tục | Chuỗi văn bản ngôn ngữ tự nhiên phi cấu trúc |
| **Nhiệm vụ (Task)** | Phân loại nhị phân (*Binary Classification*) | Hồi quy liên tục (*Continuous Regression*) | Phân loại sắc thái (*Sentiment / Recommendation*) |
| **Kích hoạt tầng ra (Final Layer)** | $\hat{y} = \text{Sigmoid}(Z_3) \in [0, 1]$ | **$\hat{y} = Z_3 \in (-\infty, +\infty)$ (Tuyến tính)** | $\hat{y} = \text{Sigmoid}(Z_3) \in [0, 1]$ |
| **Hàm mất mát ($\mathcal{L}$)** | Binary Cross-Entropy (BCE) | **Mean Squared Error (MSE)** | Binary Cross-Entropy (BCE) |
| **Đạo hàm tầng ra ($dZ$)** | $dZ_3 = \frac{1}{N}(\hat{y} - y)$ | **$dZ_3 = \frac{2}{N}(\hat{y} - y)$** | $dZ_3 = \frac{1}{N}(\hat{y} - y)$ |
| **Chỉ số đánh giá chính** | Accuracy, F1-Score | **MAE, RMSE, $R^2$-Score (AUD)** | Accuracy, F1-Score |
| **Nguyên lý Thống nhất (Slide 56–57)** | $\hat{y} = (f_3 \circ f_2 \circ f_1)(x)$ | $\hat{y} = (f_3 \circ f_2 \circ f_1)(x)$ | $\hat{y} = (f_3 \circ f_2 \circ f_1)(x)$ |

---

### 2. Đúc Kết Bản Chất Machine Learning Truyền Thống vs Học Sâu (Slide 58)

#### Tại sao các thuật toán cây (Random Forest, Decision Tree) lại thắng trên dữ liệu bảng?
- **Tiên nghiệm cảm ứng (Inductive Bias)**: Dữ liệu dạng bảng (Tabular Data) như hồ sơ y tế hay bất động sản có đặc thù là các phân khúc ranh giới sắc nét (ví dụ: chỉ số đường huyết $\ge 126$ mg/dL là ngưỡng tiểu đường; ranh giới địa lý quận huyện Melbourne).
- Thuật toán cây phân chia không gian bằng các **lát cắt trực giao song song với trục tọa độ ($x_j \ge \theta$)**, hoàn toàn trùng khớp với cấu trúc ranh giới tự nhiên của dữ liệu bảng.
- Ngược lại, mạng nơ-ron phải xấp xỉ các lát cắt trực giao này bằng các siêu phẳng nghiêng liên tục thông qua tổ hợp tuyến tính $\sum w_i x_i$, đòi hỏi nhiều tham số và tối ưu hóa phức tạp hơn.

#### Tại sao Deep Learning là tương lai của dữ liệu phức tạp?
- **Khả năng tự động học biểu diễn (Representation Learning)**: Deep Learning tự động phát hiện các mối quan hệ phi tuyến phức tạp mà không đòi hỏi kỹ sư phải tự tay tạo ra hàng trăm đặc trưng tích chéo (Interaction Terms).
- **Mở rộng sang Dữ liệu Đa phương thức (Multimodal Learning)**: Đây là điều ML truyền thống không thể làm được. Với Deep Learning, ta có thể xây dựng một hệ thống định giá bất động sản tích hợp đồng thời: **Bảng thông số nhà đất** (qua MLP) + **Ảnh chụp thực tế căn nhà** (qua CNN) + **Đoạn văn bản mô tả của môi giới** (qua Transformer/NLP).

---

### 3. Giải Đáp 3 Sai Lầm Phổ Biến Trong Deep Learning (Common Misconceptions - Slide 68–70)

1. **Misconception 1: *"Deep Learning đồng nghĩa với mô hình cực kỳ phức tạp và khổng lồ"*** (Slide 69)  
   $\to$ **Sai**. Bản chất của Deep Learning nằm ở **sự hợp thành của các phép biến đổi hàm số** ($f_L \circ \dots \circ f_1$). Một mạng nhỏ 3 tầng thuần NumPy với 64 và 32 nơ-ron đã thể hiện trọn vẹn mọi nguyên lý cơ bản của học sâu mà không cần đến hàng tỷ tham số.
2. **Misconception 2: *"Deep Learning tự động hiểu dữ liệu mà không cần con người tiền xử lý"*** (Slide 70)  
   $\to$ **Sai**. Thực nghiệm ở bài toán House Price đã chứng minh: nếu không chuẩn hóa biến mục tiêu $y$, MSE loss sẽ vọt lên $1.35 \times 10^{12}$ và gradient phát nổ phá hỏng toàn bộ mạng. Mô hình chỉ học từ công thức toán học; rác vào thì rác ra (Garbage In, Garbage Out). Tiền xử lý dữ liệu vẫn là khâu quyết định thành bại.
3. **Misconception 3: *"Mạng càng sâu thì hiệu năng chắc chắn càng cao"***  
   $\to$ **Sai**. Nếu bài toán có bản chất tuyến tính phẳng, việc tăng độ sâu chỉ làm tăng nguy cơ quá khớp (overfitting), tiêu tốn tài nguyên và dễ gặp hiện tượng biến mất đạo hàm. Cấu trúc mạng phải phù hợp với độ phức tạp của dữ liệu.

---

### 4. Trả Lời Đầy Đủ 16 Câu Hỏi Tự Học Cốt Lõi (Self-Study Questions - Slide 54)

1. **Sự khác biệt giữa ML truyền thống và Deep Learning?**  
   ML truyền thống dùng đặc trưng do con người thiết kế thủ công ($\phi_{\text{human}}$) kết hợp bộ phân loại nông; Deep Learning tự động học biểu diễn phân cấp ($h_1, h_2, \dots$) cùng lúc với bộ dự báo qua lan truyền ngược.
2. **Ý nghĩa của phương trình $\hat{y} = f_3(f_2(f_1(X)))$?**  
   Là phép hợp thành hàm số nhiều tầng: tầng 1 trích xuất đặc trưng thô, tầng 2 tổng hợp thành biểu diễn trừu tượng, tầng 3 đưa ra dự đoán cuối cùng.
3. **Mục đích của ma trận trọng số $W$?**  
   Xác định mức độ đóng góp, tầm quan trọng và sự tương tác giữa các chiều đặc trưng đầu vào khi ánh xạ sang không gian mới (Scale & Rotation).
4. **Mục đích của vector độ lệch $b$?**  
   Dịch chuyển siêu phẳng kích hoạt ra khỏi gốc tọa độ (Shift), cho phép mô hình linh hoạt phân tách dữ liệu ngay cả khi đầu vào $X = 0$.
5. **Tại sao bắt buộc cần hàm kích hoạt phi tuyến?**  
   Nếu không có phi tuyến, tích các ma trận tuyến tính xếp chồng sẽ suy biến về một phép biến đổi tuyến tính đơn tầng phẳng duy nhất ($W'x + b'$).
6. **Hàm ReLU làm nhiệm vụ gì?**  
   Giữ nguyên giá trị dương và triệt tiêu giá trị âm về $0$ ($\max(0, z)$). ReLU tạo tính thưa (Sparsity) và duy trì gradient bằng $1$ khi $z > 0$, ngăn chặn triệt để hiện tượng biến mất đạo hàm.
7. **Tại sao dùng Sigmoid ở tầng ra của bài toán phân loại?**  
   Sigmoid ép giá trị thực bất kỳ về khoảng xác suất trơn $(0, 1)$, phù hợp để biểu diễn xác suất có điều kiện $P(y=1|x)$. (Trong bài hồi quy, tầng ra dùng tuyến tính để giá trị dự đoán không bị giới hạn).
8. **Tại sao cần hàm mất mát (Loss Function)?**  
   Lượng hóa mức độ sai lệch giữa dự đoán $\hat{y}$ và nhãn thật $y$, cung cấp một thước đo vô hướng để thuật toán tối ưu có mục tiêu giảm thiểu.
9. **Hàm Binary Cross-Entropy đo lường điều gì?**  
   Đo lường độ bất đồng (Divergence) giữa phân phối xác suất dự đoán của mô hình và phân phối nhãn thực nghiệm dựa trên nguyên lý Hợp lý cực đại (Maximum Likelihood). Phạt cực nặng các dự đoán sai với độ tự tin cao.
10. **Gradient là gì?**  
    Là vector chứa toàn bộ các đạo hàm riêng $\nabla_\theta \mathcal{L}$, chỉ hướng tăng dốc nhất của hàm mất mát trong không gian tham số.
11. **Lan truyền ngược (Backpropagation) tính toán điều gì?**  
    Tính toán đạo hàm riêng của hàm mất mát đối với từng trọng số và độ lệch ($\frac{\partial \mathcal{L}}{\partial W}, \frac{\partial \mathcal{L}}{\partial b}$) thông qua Quy tắc chuỗi (Chain Rule).
12. **Thuật toán Gradient Descent làm nhiệm vụ gì?**  
    Cập nhật trọng số ngược chiều gradient với bước nhảy $\eta$ ($\theta \leftarrow \theta - \eta \nabla \mathcal{L}$) để kéo mô hình hội tụ về điểm cực tiểu của hàm mất mát.
13. **Tại sao trọng số phải được cập nhật lặp đi lặp lại qua nhiều Epoch?**  
    Vì hàm mất mát là phi tuyến phức tạp trong không gian nhiều chiều. Mỗi bước Gradient Descent chỉ là một xấp xỉ tuyến tính cục bộ nhỏ; cần hàng trăm bước lặp để tham số di chuyển dần về vùng tối ưu toàn cục.
14. **Tại sao phải phân chia tập dữ liệu thành Train và Test?**  
    Tập Train dùng để học tham số; tập Test được giữ độc lập hoàn toàn để kiểm tra năng lực tổng quát hóa trên dữ liệu mới chưa từng thấy, phát hiện hiện tượng học vẹt (Overfitting).
15. **Tại sao phải chuẩn hóa các đặc trưng (Normalization)?**  
    Đưa các thuộc tính về cùng một thang đo ($\mathcal{N}(0, 1)$), làm cho bề mặt hàm mất mát có dạng hình tròn đồng mức thay vì elip méo mó, giúp Gradient Descent hội tụ thẳng và nhanh hơn gấp nhiều lần.
16. **Sự khác biệt giữa xác suất (Probability) và nhãn dự đoán (Class Prediction)?**  
    Xác suất là một giá trị số thực liên tục $\hat{y} \in [0, 1]$ thể hiện mức độ tự tin của mô hình; Nhãn dự đoán là giá trị nhị phân rời rạc ($0$ hoặc $1$) thu được sau khi so sánh xác suất với một ngưỡng quyết định $\tau$ (thường là $0.5$).

---

# PHẦN IV: TỔNG KẾT SẢN PHẨM & ĐÓNG GÓI NỘP BÀI

### 1. Bảng Tổng Hợp Sản Phẩm Trong Thư Mục `A3_submit/`

| Thư Mục / File | Mô Tả Sản Phẩm | Trạng Thái Hoàn Thành |
|---|---|:---:|
| [Phase1_Diabetes/diabetes_deep_learning.ipynb](file:///d:/JJin/Documents/Học/I-4/Thiet_ke_httm/jupyter/A3/A3_submit/Phase1_Diabetes/diabetes_deep_learning.ipynb) | Notebook chuẩn Anaconda cho bài toán Phân loại Tiểu đường ($57$ cells). Đầy đủ ML Scikit-Learn, DL NumPy, Confusion Matrix, 6 thực nghiệm. | **100% Hoàn thành** |
| [Phase2_HousePrice/house_price_prediction.ipynb](file:///d:/JJin/Documents/Học/I-4/Thiet_ke_httm/jupyter/A3/A3_submit/Phase2_HousePrice/house_price_prediction.ipynb) | Notebook chuẩn Anaconda cho bài toán Hồi quy Giá nhà Melbourne ($57$ cells). Đầy đủ OLS, Decision Tree, Random Forest, DL NumPy Hồi quy MSE, Scatter Plot, Denormalization, 6 thực nghiệm. | **100% Hoàn thành** |
| [Phase3_Ecommerce/customer_interest_text_deeplearning.ipynb](file:///d:/JJin/Documents/Học/I-4/Thiet_ke_httm/jupyter/A3/A3_submit/Phase3_Ecommerce/customer_interest_text_deeplearning.ipynb) | Notebook chuẩn Anaconda cho bài toán Phân loại Ý kiến Khách hàng NLP ($57$ cells). TF-IDF, 3 mô hình ML, DL NumPy, giải quyết chi tiết Slide 66 (Ex 3), 6 thực nghiệm. | **100% Hoàn thành** |
| `REPORT_A3_DEEP_LEARNING.md` | Báo cáo khoa học hoàn chỉnh chuẩn sinh viên theo đề cương 3 phần bám sát Slide bài giảng. | **100% Hoàn thành** |

### 2. Cam Kết Kỹ Thuật
- **Không có lỗi Runtime**: Cả 3 file notebook đều đã được chạy kiểm thử toàn bộ (Run All Cells) và lưu trữ đầy đủ đầu ra text, ma trận và hình ảnh nhúng Base64.
- **Tính độc lập của từng cell**: Mỗi thao tác từ tiền xử lý, huấn luyện từng mô hình ML, huấn luyện Deep Learning, vẽ đồ thị đến từng bài thực nghiệm đều là **1 cell code riêng biệt** kèm markdown giải thích.
- **Không hardcode**: $100\%$ số liệu trong báo cáo và notebook đều được sinh ra từ quá trình tính toán thực tế.
