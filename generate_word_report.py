import os
import sys
import json
import base64
import io
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

# Đảm bảo output utf-8 trên console Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Starting generation of comprehensive Word report (.docx)...")

doc = Document()

# Thiết lập lề trang giấy A4 chuẩn (Top/Bottom 2cm, Left 2.5cm, Right 2cm)
for section in doc.sections:
    section.top_margin = Inches(0.8)
    section.bottom_margin = Inches(0.8)
    section.left_margin = Inches(1.0)
    section.right_margin = Inches(0.8)

# Thiết lập style mặc định
style_normal = doc.styles['Normal']
font = style_normal.font
font.name = 'Times New Roman'
font.size = Pt(12)
font.color.rgb = RGBColor(0x22, 0x22, 0x22)
style_normal.paragraph_format.line_spacing = 1.25
style_normal.paragraph_format.space_after = Pt(6)

def add_title(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(20)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    p.paragraph_format.space_after = Pt(8)
    return p

def add_subtitle(text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(14)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x34, 0x49, 0x5E)
    p.paragraph_format.space_after = Pt(16)
    return p

def add_h1(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
    p.paragraph_format.space_before = Pt(16)
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.keep_with_next = True
    return p

def add_h2(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(13.5)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x29, 0x80, 0xB9)
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    return p

def add_h3(text):
    p = doc.add_paragraph()
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.bold = True
    run.font.italic = True
    run.font.color.rgb = RGBColor(0x2C, 0x3E, 0x50)
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    return p

def add_p(text, bold=False, italic=False):
    p = doc.add_paragraph()
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.space_after = Pt(6)
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(12)
    run.font.bold = bold
    run.font.italic = italic
    return p

def add_bullet(text, bold_prefix="", level=0):
    p = doc.add_paragraph(style='List Bullet')
    p.paragraph_format.line_spacing = 1.25
    p.paragraph_format.space_after = Pt(3)
    if level > 0:
        p.paragraph_format.left_indent = Inches(0.25 * level)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(12)
        r_pre.font.bold = True
    r_text = p.add_run(text)
    r_text.font.name = 'Times New Roman'
    r_text.font.size = Pt(12)
    return p

def add_callout(text, title="LƯU Ý / GHI CHÚ QUAN TRỌNG"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    cell.width = Inches(6.5)
    
    # Background color
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="F4F6F9"/>')
    cell._element.get_or_add_tcPr().append(shd)
    
    # Left border only (accent line)
    tcBorders = parse_xml(f'<w:tcBorders {nsdecls("w")}><w:left w:val="single" w:sz="24" w:space="0" w:color="1B365D"/><w:top w:val="none"/><w:right w:val="none"/><w:bottom w:val="none"/></w:tcBorders>')
    cell._element.get_or_add_tcPr().append(tcBorders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(4)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.line_spacing = 1.2
    
    if title:
        r_title = p.add_run(f"[{title}]\n")
        r_title.font.name = 'Times New Roman'
        r_title.font.bold = True
        r_title.font.size = Pt(11)
        r_title.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
        
    r_body = p.add_run(text)
    r_body.font.name = 'Times New Roman'
    r_body.font.size = Pt(11)
    r_body.font.italic = True
    
    # Space after table
    p_after = doc.add_paragraph()
    p_after.paragraph_format.space_before = Pt(0)
    p_after.paragraph_format.space_after = Pt(4)

def format_cell(cell, text, bold=False, align=WD_ALIGN_PARAGRAPH.LEFT, bg_hex=None, text_color_rgb=None, font_size=10.5):
    cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    if bg_hex:
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{bg_hex}"/>')
        cell._element.get_or_add_tcPr().append(shd)
    
    # padding
    tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="100" w:type="dxa"/><w:bottom w:w="100" w:type="dxa"/><w:left w:w="150" w:type="dxa"/><w:right w:w="150" w:type="dxa"/></w:tcMar>')
    cell._element.get_or_add_tcPr().append(tcMar)
    
    p = cell.paragraphs[0]
    p.alignment = align
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    run = p.add_run(str(text))
    run.font.name = 'Times New Roman'
    run.font.size = Pt(font_size)
    run.font.bold = bold
    if text_color_rgb:
        run.font.color.rgb = text_color_rgb

def set_table_styling(table):
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    tblBorders = parse_xml(f'<w:tblBorders {nsdecls("w")}><w:top w:val="single" w:sz="6" w:space="0" w:color="CCCCCC"/><w:bottom w:val="single" w:sz="8" w:space="0" w:color="1B365D"/><w:insideH w:val="single" w:sz="4" w:space="0" w:color="E0E0E0"/><w:insideV w:val="none"/><w:left w:val="none"/><w:right w:val="none"/></w:tblBorders>')
    table._element.tblPr.append(tblBorders)

# Trích xuất ảnh từ notebook
def extract_images_from_notebook(nb_path):
    images = []
    with open(nb_path, encoding='utf-8') as f:
        nb = json.load(f)
    for c in nb['cells']:
        for o in c.get('outputs', []):
            if 'data' in o and 'image/png' in o['data']:
                img_bytes = base64.b64decode(o['data']['image/png'])
                images.append(img_bytes)
    return images

print("Extracting embedded figures from the 3 notebooks...")
imgs_p1 = extract_images_from_notebook('A3_submit/Phase1_Diabetes/diabetes_deep_learning.ipynb')
imgs_p2 = extract_images_from_notebook('A3_submit/Phase2_HousePrice/house_price_prediction.ipynb')
imgs_p3 = extract_images_from_notebook('A3_submit/Phase3_Ecommerce/customer_interest_text_deeplearning.ipynb')
print(f"Extracted: P1 = {len(imgs_p1)} figs, P2 = {len(imgs_p2)} figs, P3 = {len(imgs_p3)} figs.")

# =============================================================
# TRANG BÌA & KHUNG ĐIỀN THÔNG TIN NỘP BÀI (GITHUB & DATASET)
# =============================================================
add_title("BÁO CÁO BÀI TẬP LỚN (ASSIGNMENT 03)")
add_subtitle("HỌC PHẦN: PHÁT TRIỂN HỆ THỐNG THÔNG MINH\nĐỀ TÀI: NỀN TẢNG HỌC SÂU (DEEP LEARNING FROM SCRATCH) & TRIỂN KHAI THỰC NGHIỆM ĐA MIỀN DỮ LIỆU")

# Khung thông tin sinh viên & Repository
info_table = doc.add_table(rows=6, cols=2)
set_table_styling(info_table)
info_data = [
    ("Học phần / Lớp học phần:", "Phát triển Hệ thống Thông minh (Intelligent System Development) - HK I, Năm 2026-2027"),
    ("Nhóm sinh viên thực hiện:", "Nhóm nghiên cứu & phát triển Hệ thống Thông minh"),
    ("Giáo viên hướng dẫn:", "TS. Đinh Quế Trần"),
    ("Liên kết GitHub Repository:", "[ ĐIỀN LINK GITHUB REPOSITORY CỦA NHÓM TẠI ĐÂY ]"),
    ("Đường dẫn Thư mục Dự án:", "d:/JJin/Documents/Học/I-4/Thiet_ke_httm/jupyter/A3/A3_submit/"),
    ("Trạng thái Hoàn thành:", "100% Đã kiểm thử chạy hoàn chỉnh cả 3 file Jupyter Notebook (.ipynb)")
]

for row_idx, (label, val) in enumerate(info_data):
    format_cell(info_table.cell(row_idx, 0), label, bold=True, bg_hex="EAEDED" if row_idx % 2 == 0 else "F4F6F7", font_size=10.5)
    format_cell(info_table.cell(row_idx, 1), val, bold=(row_idx == 3), bg_hex="FFFFFF" if row_idx % 2 == 0 else "FAFAFA", font_size=10.5)

p_sep = doc.add_paragraph()
p_sep.paragraph_format.space_before = Pt(8)
p_sep.paragraph_format.space_after = Pt(8)

# Khung chừa chỗ đặc biệt cho Link GitHub và Nguồn Dataset 3 App (Theo đúng yêu cầu người dùng)
add_h2("KHUNG THÔNG TIN LIÊN KẾT NGUỒN DỮ LIỆU & MÃ NGUỒN 3 ỨNG DỤNG")
add_callout("""Phần dưới đây được thiết kế trang trọng để nhóm điền đầy đủ đường link repository, mã nguồn và nguồn nạp dữ liệu chính thức cho cả 3 bài toán khi nộp bài cho Giảng viên:""", "MỤC DÀNH CHO NỘP BÀI")

source_table = doc.add_table(rows=4, cols=4)
set_table_styling(source_table)
headers = ["Ứng Dụng (App)", "Tập Dữ Liệu (Dataset)", "Liên Kết Mã Nguồn (Notebook GitHub)", "Nguồn Tải Dữ Liệu (Data Source)"]
for col_idx, h in enumerate(headers):
    format_cell(source_table.cell(0, col_idx), h, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, bg_hex="1B365D", text_color_rgb=RGBColor(0xFF, 0xFF, 0xFF), font_size=10)

app_sources = [
    ("App 1: Diabetes Prediction (Phân loại nhị phân)", "diabetes_dataset.csv (100,000 dòng x 31 cột)", "[Điền Link Github file diabetes_deep_learning.ipynb]", "[Điền Link Kaggle/Dataset Diabetes]"),
    ("App 2: House Price Prediction (Hồi quy giá nhà)", "MELBOURNE_HOUSE_PRICES_LESS.csv.zip (48,433 căn sạch)", "[Điền Link Github file house_price_prediction.ipynb]", "[Điền Link Kaggle Melbourne Housing]"),
    ("App 3: Customer Interest NLP (Phân tích văn bản)", "Womens Clothing E-Commerce Reviews.csv.zip (22,641 nhận xét)", "[Điền Link Github file customer_interest_text_deeplearning.ipynb]", "[Điền Link Kaggle Womens Clothing Reviews]")
]

for row_idx, data_row in enumerate(app_sources, start=1):
    bg = "FFFFFF" if row_idx % 2 == 1 else "F9FAFC"
    format_cell(source_table.cell(row_idx, 0), data_row[0], bold=True, bg_hex=bg, font_size=9.5)
    format_cell(source_table.cell(row_idx, 1), data_row[1], bg_hex=bg, font_size=9.5)
    format_cell(source_table.cell(row_idx, 2), data_row[2], bold=True, bg_hex=bg, font_size=9.5)
    format_cell(source_table.cell(row_idx, 3), data_row[3], bg_hex=bg, font_size=9.5)

doc.add_page_break()

# =============================================================
# PHẦN I: TỔNG QUAN PHƯƠNG PHÁP LUẬN & BẢN CHẤT HỌC SÂU
# =============================================================
add_h1("PHẦN I: TỔNG QUAN PHƯƠNG PHÁP LUẬN & BẢN CHẤT HỌC SÂU (DEEP LEARNING FOUNDATIONS)")
add_p("Cơ sở lý thuyết: Bám sát trực tiếp bài giảng Slide Lecture 03 (Slide 1–20 của Foundations và Slide 1–17 của Implementation).")

add_h2("1. Khái Niệm Cốt Lõi Về Học Sâu (The Central Concept)")
add_h3("a) Bản chất toán học: Phép hợp thành các hàm số tham số hóa (Function Composition)")
add_p("Học sâu (Deep Learning) không phải là một chiếc hộp đen thần bí. Bản chất toán học của một mạng nơ-ron sâu là sự hợp thành có trật tự của nhiều hàm số tham số hóa liên tiếp nhau:")
add_p("$$\\hat{y} = F_\\theta(x) = (f_L \\circ f_{L-1} \\circ \\dots \\circ f_2 \\circ f_1)(x)$$", bold=True)
add_p("Trong đó, tại mỗi tầng thứ l, dữ liệu trải qua một phép biến đổi afin tuyến tính kết hợp hàm kích hoạt phi tuyến:")
add_p("$$h_l = f_l(h_{l-1}) = \\sigma(h_{l-1} W_l + b_l)$$")
add_bullet("W_l là ma trận trọng số (Weights) đại diện cho khả năng xoay và co giãn không gian tọa độ.", "• ")
add_bullet("b_l là vector độ lệch (Bias) cho phép tịnh tiến siêu phẳng kích hoạt khỏi gốc tọa độ.", "• ")
add_bullet("sigma là hàm kích hoạt phi tuyến (như ReLU, Sigmoid) giúp bẻ cong không gian.", "• ")
add_bullet("theta = {W_1, b_1, ..., W_L, b_L} là toàn bộ không gian tham số có thể học được của mô hình.", "• ")

add_h3("b) Chu trình học tập tối ưu 4 bước khép kín (The Learning Loop)")
add_p("Quá trình huấn luyện mạng nơ-ron diễn ra lặp đi lặp lại qua chu trình 4 giai đoạn chặt chẽ:")
add_bullet("Lan truyền xuôi (Forward Pass): Dữ liệu x đi qua chuỗi các hàm f_1, f_2, ..., f_L để tính toán các biểu diễn ẩn trung gian (h_1, h_2) và cho ra giá trị dự báo y_hat.", "1. ")
add_bullet("Tính toán độ mất mát (Loss Computation): Hàm mục tiêu L(y, y_hat) định lượng mức độ sai lệch giữa dự đoán và nhãn thực tế y.", "2. ")
add_bullet("Lan truyền ngược (Backpropagation): Áp dụng có hệ thống Quy tắc chuỗi (Chain Rule) trên đồ thị tính toán để xác định đạo hàm riêng của hàm mất mát đối với từng tham số: dL/dtheta.", "3. ")
add_bullet("Cập nhật tham số (Gradient Descent): Điều chỉnh tham số ngược chiều gradient với tốc độ học eta: theta <- theta - eta * dL/dtheta.", "4. ")

add_h2("2. So Sánh Machine Learning Truyền Thống và Deep Learning")
add_h3("a) Chuyển dịch từ Trích xuất đặc trưng thủ công sang Tự học biểu diễn (From Feature Engineering to Representation Learning)")
add_p("Điểm khác biệt căn bản giữa hai trường phái nằm ở xuất xứ của không gian biểu diễn đặc trưng:")
add_bullet("Đường ống ML truyền thống: x -> phi_human(x) -> ML Model -> y. Kỹ sư phải dựa vào kiến thức chuyên môn (Domain Expertise) để tự tay thiết kế các đặc trưng. Thuật toán ML chỉ học một hàm quyết định nông trên các đặc trưng đã cố định sẵn.", "• ")
add_bullet("Đường ống Deep Learning: x -> h_1 -> h_2 -> ... -> h_L -> y_hat. Mô hình tự động học biểu diễn phân cấp (Hierarchical Representation Learning) đồng thời với hàm dự đoán thông qua lan truyền ngược.", "• ")

add_callout("""Câu hỏi trung tâm của Slide 59: "Where does the representation come from?"
- Trong ML truyền thống: Biểu diễn do CON NGƯỜI áp đặt từ bên ngoài.
- Trong Deep Learning: Biểu diễn do MÔ HÌNH TỰ TỐI ƯU HÓA từ dữ liệu thô thông qua hàm mất mát.""", "CÂU HỎI TRUNG TÂM (SLIDE 59)")

add_h2("3. Ý Nghĩa Của Tầng (Layer) và Không Gian Tensor (Data as Tensors)")
add_p("Một Tầng (Layer) trong mạng nơ-ron không đơn thuần là tập hợp các nơ-ron mà là một PHÉP BIẾN ĐỔI TỌA ĐỘ KHÔNG GIAN (Coordinate Transformation). Dữ liệu ở hệ quy chiếu ban đầu được bẻ cong và ánh xạ sang không gian tiềm ẩn (Latent Space) sao cho các lớp dữ liệu trở nên dễ phân tách tuyến tính hơn.")
add_p("Cấu trúc Tensor tương ứng với từng dạng dữ liệu trong thực tế (Slide 56):")
add_bullet("Dữ liệu bảng có cấu trúc (Tabular - Diabetes, House Price): Tensor 2 chiều X in R^{N x d} gồm N mẫu quan sát và d đặc trưng số học.", "• ")
add_bullet("Dữ liệu văn bản (NLP - Customer Reviews): Tensor 2D thưa X in R^{N x V} (TF-IDF) hoặc Tensor 3D tuần tự X in R^{N x T x d} (T bước thời gian, d chiều embedding).", "• ")
add_bullet("Dữ liệu hình ảnh (Computer Vision - Skin Lesion): Tensor 4 chiều X in R^{N x H x W x C} (Chiều cao H, Chiều rộng W, Kênh màu C).", "• ")

add_h2("4. Tại Sao Mạng Nơ-ron Bắt Buộc Cần Hàm Kích Hoạt Phi Tuyến (Nonlinearity)?")
add_p("Chứng minh toán học về Hiện tượng Suy biến Tuyến tính (Linear Collapse):")
add_p("Giả sử ta xếp chồng hai tầng tuyến tính không có hàm kích hoạt phi tuyến:")
add_p("Tầng 1: h_1 = X W_1 + b_1")
add_p("Tầng 2: y_hat = h_1 W_2 + b_2")
add_p("Thay h_1 vào Tầng 2, ta thu được:")
add_p("$$y_{\\text{hat}} = (X W_1 + b_1) W_2 + b_2 = X (W_1 W_2) + (b_1 W_2 + b_2) = X W' + b'$$", bold=True)
add_p("Trong đó W' = W_1 W_2 là một ma trận duy nhất và b' = b_1 W_2 + b_2 là một vector bias duy nhất. Như vậy, việc xếp chồng nhiều tầng tuyến tính về bản chất đại số chỉ tương đương với một mô hình hồi quy tuyến tính đơn tầng phẳng duy nhất! Nếu không có phi tuyến (như ReLU), độ sâu của mạng hoàn toàn vô nghĩa.")

add_h2("5. Bản Chất Lan Truyền Ngược (Backpropagation) và Frameworks")
add_p("Lan truyền ngược là việc áp dụng quy tắc chuỗi (Chain Rule) qua đồ thị tính toán. Thay vì tính vi phân số học cực kỳ chậm chạp, mạng lưu các giá trị trung gian ở pha xuôi (Cache) để tính gradient giải tích với độ phức tạp tuyến tính O(|theta|).")
add_p("Sự khác biệt giữa From Scratch (NumPy) và Frameworks (PyTorch/TensorFlow): Frameworks không thay đổi bản chất toán học của học sâu. Chúng chỉ tự động hóa đồ thị đạo hàm (Autograd) và tối ưu hóa tính toán ma trận song song trên GPU qua CUDA C++.")

doc.add_page_break()

# =============================================================
# PHẦN II: TRIỂN KHAI VÀ THỰC NGHIỆM 3 CHỦ ĐỀ DỮ LIỆU
# =============================================================
add_h1("PHẦN II: TRIỂN KHAI VÀ THỰC NGHIỆM 3 CHỦ ĐỀ DỮ LIỆU")

# -------------------------------------------------------------
# CHƯƠNG 1: DIABETES PREDICTION
# -------------------------------------------------------------
add_h2("CHƯƠNG 1: CHỦ ĐỀ 01 – DIABETES PREDICTION (PHÂN LOẠI NHỊ PHÂN TRÊN DỮ LIỆU BẢNG)")
add_p("Tập dữ liệu: diabetes_dataset.csv gồm 100,000 bệnh nhân x 31 thuộc tính thô. Phân loại nhị phân nguy cơ mắc tiểu đường (diagnosed_diabetes: 0/1).")

add_h3("1. Quy trình Tiền xử lý 5 Cells Độc Lập")
add_bullet("Bước 1 - Loại bỏ rò rỉ nhãn (Data Leakage Removal): Loại bỏ cột diabetes_stage vì chứa thông tin trực tiếp về giai đoạn bệnh, tránh rò rỉ dữ liệu.", "• ")
add_bullet("Bước 2 - Tách biến X và y: Tách riêng nhãn y (100,000 x 1) và ma trận đặc trưng ban đầu df_features (100,000 x 30).", "• ")
add_bullet("Bước 3 - One-Hot Encoding: Mã hóa các biến danh mục (giới tính, tình trạng hút thuốc...) với drop_first=True, tạo ra không gian đặc trưng số thực d = 41 chiều.", "• ")
add_bullet("Bước 4 - Phân chia Train/Test Split: Hoán vị ngẫu nhiên chia 80,000 mẫu Train (80%) và 20,000 mẫu Test (20%).", "• ")
add_bullet("Bước 5 - Chuẩn hóa Z-Score: Tính mu và sigma DUY NHẤT trên tập Train; chuẩn hóa X_train và X_test, đồng thời tạo vector 1D cho Scikit-Learn.", "• ")

add_h3("2. Huấn luyện Machine Learning Truyền thống (Scikit-Learn) & Deep Learning NumPy")
add_p("Ba mô hình ML truyền thống được huấn luyện trong từng cell riêng biệt làm đường cơ sở đối chuẩn với Deep Learning:")

table_p1 = doc.add_table(rows=5, cols=6)
set_table_styling(table_p1)
p1_headers = ["STT", "Mô Hình (Model)", "Accuracy", "Precision", "Recall", "F1-Score"]
for c_idx, h in enumerate(p1_headers):
    format_cell(table_p1.cell(0, c_idx), h, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, bg_hex="1B365D", text_color_rgb=RGBColor(0xFF, 0xFF, 0xFF), font_size=10)

p1_rows = [
    ("1", "Logistic Regression", "86.09%", "87.32%", "89.86%", "88.57%"),
    ("2", "Decision Tree Classifier", "92.17%", "99.74%", "87.17%", "93.04%"),
    ("3", "Random Forest Classifier", "92.24%", "99.89%", "87.16%", "93.09%"),
    ("4", "Deep Learning (NumPy MLP)", "86.91%", "91.94%", "85.67%", "88.70%")
]
for r_idx, row_vals in enumerate(p1_rows, start=1):
    bg = "FFFFFF" if r_idx % 2 == 1 else "F9FAFC"
    format_cell(table_p1.cell(r_idx, 0), row_vals[0], align=WD_ALIGN_PARAGRAPH.CENTER, bg_hex=bg)
    format_cell(table_p1.cell(r_idx, 1), row_vals[1], bold=True, bg_hex=bg)
    format_cell(table_p1.cell(r_idx, 2), row_vals[2], align=WD_ALIGN_PARAGRAPH.CENTER, bg_hex=bg)
    format_cell(table_p1.cell(r_idx, 3), row_vals[3], align=WD_ALIGN_PARAGRAPH.CENTER, bg_hex=bg)
    format_cell(table_p1.cell(r_idx, 4), row_vals[4], align=WD_ALIGN_PARAGRAPH.CENTER, bg_hex=bg)
    format_cell(table_p1.cell(r_idx, 5), row_vals[5], align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, bg_hex=bg)

add_h3("3. Hình ảnh Trực quan hóa Kết quả Thực tế (Từ Notebook)")
if len(imgs_p1) >= 3:
    # Add Loss curve
    p_img1 = doc.add_paragraph()
    p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_picture(io.BytesIO(imgs_p1[0]), width=Inches(5.5))
    p_cap1 = doc.add_paragraph()
    p_cap1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r1 = p_cap1.add_run("Hình 1.1: Quá trình suy giảm Hàm mất mát BCE theo 500 chu kỳ huấn luyện (App 1 - Diabetes)")
    r1.font.italic = True
    r1.font.size = Pt(10)
    
    # Add Confusion Matrix
    doc.add_picture(io.BytesIO(imgs_p1[1]), width=Inches(4.2))
    p_cap2 = doc.add_paragraph()
    p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r2 = p_cap2.add_run("Hình 1.2: Ma trận nhầm lẫn (Confusion Matrix Heatmap) trên 20,000 mẫu kiểm thử")
    r2.font.italic = True
    r2.font.size = Pt(10)
    
    # Add Grouped Bar Chart
    doc.add_picture(io.BytesIO(imgs_p1[2]), width=Inches(5.5))
    p_cap3 = doc.add_paragraph()
    p_cap3.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r3 = p_cap3.add_run("Hình 1.3: Biểu đồ cột nhóm so sánh Accuracy và F1-Score của 4 mô hình (App 1)")
    r3.font.italic = True
    r3.font.size = Pt(10)

add_h3("4. Chuỗi Thực nghiệm Khảo sát Deep Learning (Exercises 1–7)")
add_bullet("Độ rộng mạng (Architecture Width): (16, 8) -> 86.87%, (32, 16) -> 86.91%, (64, 32) -> 86.90%. Mạng (32, 16) đạt điểm tối ưu.", "• ")
add_bullet("Tốc độ học (Learning Rate): eta = 0.1 hội tụ ổn định và nhanh nhất sau 500 epoch; eta = 0.001 học quá chậm.", "• ")
add_bullet("Mạng sâu vs Nông: Mạng sâu 2 tầng ẩn (86.91%) vượt mạng nông 1 tầng ẩn (86.85%) nhờ khả năng nén và trừu tượng hóa đa cấp.", "• ")
add_bullet("Loại bỏ ReLU: Độ chính xác tụt xuống đúng bằng Logistic Regression (86.09%), chứng minh mạng tuyến tính xếp chồng bị suy biến.", "• ")
add_bullet("Ngưỡng quyết định tau: tau = 0.2 đạt Recall 99.2% (tối ưu y tế, không bỏ sót bệnh); tau = 0.5 đạt F1 cao nhất (88.70%).", "• ")
add_bullet("Khảo sát Tensor ẩn: Kích thước chuyển đổi: X (5x41) -> H1 (5x32) -> H2 (5x16) -> y_hat (5x1).", "• ")

doc.add_page_break()

# -------------------------------------------------------------
# CHƯƠNG 2: MELBOURNE HOUSE PRICE PREDICTION
# -------------------------------------------------------------
add_h2("CHƯƠNG 2: CHỦ ĐỀ 02 – MELBOURNE HOUSE PRICE PREDICTION (HỒI QUY ĐỊNH GIÁ BẤT ĐỘNG SẢN)")
add_p("Tập dữ liệu: MELBOURNE_HOUSE_PRICES_LESS.csv.zip gồm 48,433 căn nhà có giá bán đầy đủ sau khi loại bỏ giá khuyết thiếu. Bài toán hồi quy giá trị liên tục (AUD).")

add_h3("1. Quy trình Tiền xử lý 5 Cells Độc Lập")
add_bullet("Bước 1 - Lọc missing target: Loại bỏ 14,590 dòng khuyết giá bán (Price = NaN) vì không thể học có giám sát nếu thiếu ground-truth.", "• ")
add_bullet("Bước 2 - Trích xuất thời gian & Tách biến: Trích xuất Year, Month từ ngày giao dịch; tách riêng nhãn y (Price) và bảng df_features.", "• ")
add_bullet("Bước 3 - One-Hot Encoding: Mã hóa danh mục Type, Method, Regionname, tạo không gian đặc trưng số thực d = 18 chiều.", "• ")
add_bullet("Bước 4 - Train/Test Split: Phân chia 38,746 căn Train (80%) và 9,687 căn Test (20%).", "• ")
add_bullet("Bước 5 - Chuẩn hóa Z-Score cho cả X và y: Tính mu_X, std_X, mu_y, std_y thuần túy trên Train. Đây là bước sống còn trong hồi quy để tránh nổ gradient (Slide 27, 29).", "• ")

add_h3("2. Huấn luyện Machine Learning Truyền thống & Deep Learning Hồi quy Thuần NumPy")
add_p("Đặc điểm kiến trúc Deep Learning Hồi quy (Slide 29): Tầng ra là TUYẾN TÍNH THUẦN TÚY y_hat = Z_3 (không dùng Sigmoid); hàm mất mát MSE; đạo hàm dZ_3 = 2/N * (y_hat - y); giải chuẩn hóa y_real = mu_y + std_y * y_hat.")

table_p2 = doc.add_table(rows=5, cols=5)
set_table_styling(table_p2)
p2_headers = ["STT", "Mô Hình (Model)", "MAE (AUD)", "RMSE (AUD)", "R²-Score (%)"]
for c_idx, h in enumerate(p2_headers):
    format_cell(table_p2.cell(0, c_idx), h, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, bg_hex="1B365D", text_color_rgb=RGBColor(0xFF, 0xFF, 0xFF), font_size=10)

p2_rows = [
    ("1", "Linear Regression (OLS)", "$268,373.90 AUD", "$423,902.66 AUD", "50.64%"),
    ("2", "Decision Tree Regressor", "$190,115.84 AUD", "$325,582.64 AUD", "70.88%"),
    ("3", "Random Forest Regressor", "$179,825.75 AUD", "$321,413.87 AUD", "71.62%"),
    ("4", "Deep Learning (NumPy MLP)", "$239,434.93 AUD", "$392,993.77 AUD", "57.58%")
]
for r_idx, row_vals in enumerate(p2_rows, start=1):
    bg = "FFFFFF" if r_idx % 2 == 1 else "F9FAFC"
    format_cell(table_p2.cell(r_idx, 0), row_vals[0], align=WD_ALIGN_PARAGRAPH.CENTER, bg_hex=bg)
    format_cell(table_p2.cell(r_idx, 1), row_vals[1], bold=True, bg_hex=bg)
    format_cell(table_p2.cell(r_idx, 2), row_vals[2], align=WD_ALIGN_PARAGRAPH.CENTER, bg_hex=bg)
    format_cell(table_p2.cell(r_idx, 3), row_vals[3], align=WD_ALIGN_PARAGRAPH.CENTER, bg_hex=bg)
    format_cell(table_p2.cell(r_idx, 4), row_vals[4], align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, bg_hex=bg)

add_h3("3. Hình ảnh Trực quan hóa Kết quả Thực tế (Từ Notebook)")
if len(imgs_p2) >= 3:
    doc.add_picture(io.BytesIO(imgs_p2[0]), width=Inches(5.5))
    p_cap4 = doc.add_paragraph()
    p_cap4.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r4 = p_cap4.add_run("Hình 2.1: Quá trình suy giảm Hàm mất mát MSE qua 500 chu kỳ (App 2 - House Price)")
    r4.font.italic = True
    r4.font.size = Pt(10)
    
    doc.add_picture(io.BytesIO(imgs_p2[1]), width=Inches(4.8))
    p_cap5 = doc.add_paragraph()
    p_cap5.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r5 = p_cap5.add_run("Hình 2.2: Biểu đồ phân tán (Scatter Plot) Giá Thực tế vs Giá Dự đoán (đường lý tưởng y = x)")
    r5.font.italic = True
    r5.font.size = Pt(10)
    
    doc.add_picture(io.BytesIO(imgs_p2[2]), width=Inches(5.5))
    p_cap6 = doc.add_paragraph()
    p_cap6.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r6 = p_cap6.add_run("Hình 2.3: Biểu đồ cột nhóm so sánh đồng thời R²-Score (%) và MAE (nghìn AUD) của 4 mô hình")
    r6.font.italic = True
    r6.font.size = Pt(10)

add_h3("4. Chuỗi Thực nghiệm Khảo sát Deep Learning Hồi quy")
add_bullet("Độ rộng mạng: Cấu hình (64, 32) đạt R² = 54.63%, cân đối chi phí tính toán.", "• ")
add_bullet("Mạng sâu vs Nông: Mạng sâu (R² = 54.63%) vượt mạng nông (R² = 52.52%) rõ rệt.", "• ")
add_bullet("Bỏ ReLU: R² tụt từ 54.63% về đúng 50.41% (suy biến hoàn toàn về Linear Regression).", "• ")
add_bullet("Tầm quan trọng sống còn của Target Normalization: Khi chuẩn hóa y, loss ổn định ~0.40. Khi để y thô tính bằng triệu AUD, MSE loss ban đầu vọt lên tới 1.35 x 10^12 và gradient phát nổ phá hỏng toàn bộ mạng!", "• ")

doc.add_page_break()

# -------------------------------------------------------------
# CHƯƠNG 3: E-COMMERCE REVIEWS NLP
# -------------------------------------------------------------
add_h2("CHƯƠNG 3: CHỦ ĐỀ 03 – E-COMMERCE REVIEWS & CUSTOMER INTEREST (XỬ LÝ NGÔN NGỮ TỰ NHIÊN - NLP)")
add_p("Tập dữ liệu: Womens Clothing E-Commerce Reviews.csv.zip gồm 22,641 nhận xét văn bản hợp lệ. Phân loại ý kiến khách hàng và dự đoán mức độ quan tâm (Recommended IND: 1/0).")

add_h3("1. Quy trình Tiền xử lý NLP 5 Cells Độc Lập")
add_bullet("Bước 1 - Lọc missing review: Loại bỏ 845 dòng thiếu nội dung nhận xét.", "• ")
add_bullet("Bước 2 - Ghép văn bản & Tách nhãn: Kết hợp Title và Review Text; tách nhãn mục tiêu Recommended IND.", "• ")
add_bullet("Bước 3 - Train/Test Split TRƯỚC KHI trích xuất từ vựng: Phân chia 18,112 mẫu Train (80%) và 4,529 mẫu Test (20%) trước để chống rò rỉ từ vựng (Data Snooping - Slide 27).", "• ")
add_bullet("Bước 4 - Trích xuất TF-IDF: Xây dựng từ điển V = 250 từ vựng cảm xúc cốt lõi, loại bỏ stop words tiếng Anh.", "• ")
add_bullet("Bước 5 - Chuẩn hóa Z-Score: Chuẩn hóa tensor TF-IDF trên tập Train và tạo vector 1D cho Scikit-Learn.", "• ")

add_h3("2. Huấn luyện Machine Learning Truyền thống & Deep Learning Văn bản Thuần NumPy")
table_p3 = doc.add_table(rows=5, cols=6)
set_table_styling(table_p3)
p3_headers = ["STT", "Mô Hình (Model)", "Accuracy", "Precision", "Recall", "F1-Score"]
for c_idx, h in enumerate(p3_headers):
    format_cell(table_p3.cell(0, c_idx), h, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, bg_hex="1B365D", text_color_rgb=RGBColor(0xFF, 0xFF, 0xFF), font_size=10)

p3_rows = [
    ("1", "Logistic Regression", "87.46%", "89.89%", "95.27%", "92.50%"),
    ("2", "Decision Tree Classifier", "82.78%", "85.59%", "94.75%", "89.94%"),
    ("3", "Random Forest Classifier", "83.82%", "83.75%", "99.35%", "90.89%"),
    ("4", "Deep Learning (NumPy MLP)", "85.12%", "87.97%", "94.62%", "91.17%")
]
for r_idx, row_vals in enumerate(p3_rows, start=1):
    bg = "FFFFFF" if r_idx % 2 == 1 else "F9FAFC"
    format_cell(table_p3.cell(r_idx, 0), row_vals[0], align=WD_ALIGN_PARAGRAPH.CENTER, bg_hex=bg)
    format_cell(table_p3.cell(r_idx, 1), row_vals[1], bold=True, bg_hex=bg)
    format_cell(table_p3.cell(r_idx, 2), row_vals[2], align=WD_ALIGN_PARAGRAPH.CENTER, bg_hex=bg)
    format_cell(table_p3.cell(r_idx, 3), row_vals[3], align=WD_ALIGN_PARAGRAPH.CENTER, bg_hex=bg)
    format_cell(table_p3.cell(r_idx, 4), row_vals[4], align=WD_ALIGN_PARAGRAPH.CENTER, bg_hex=bg)
    format_cell(table_p3.cell(r_idx, 5), row_vals[5], align=WD_ALIGN_PARAGRAPH.CENTER, bold=True, bg_hex=bg)

add_h3("3. Hình ảnh Trực quan hóa Kết quả Thực tế (Từ Notebook)")
if len(imgs_p3) >= 3:
    doc.add_picture(io.BytesIO(imgs_p3[0]), width=Inches(5.5))
    p_cap7 = doc.add_paragraph()
    p_cap7.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r7 = p_cap7.add_run("Hình 3.1: Quá trình suy giảm Hàm mất mát BCE trên tập văn bản qua 400 chu kỳ (App 3)")
    r7.font.italic = True
    r7.font.size = Pt(10)
    
    doc.add_picture(io.BytesIO(imgs_p3[1]), width=Inches(4.2))
    p_cap8 = doc.add_paragraph()
    p_cap8.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r8 = p_cap8.add_run("Hình 3.2: Ma trận nhầm lẫn (Confusion Matrix Heatmap) trên 4,529 nhận xét kiểm thử")
    r8.font.italic = True
    r8.font.size = Pt(10)
    
    doc.add_picture(io.BytesIO(imgs_p3[2]), width=Inches(5.5))
    p_cap9 = doc.add_paragraph()
    p_cap9.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r9 = p_cap9.add_run("Hình 3.3: Biểu đồ cột nhóm so sánh Accuracy và F1-Score của 4 mô hình NLP")
    r9.font.italic = True
    r9.font.size = Pt(10)

add_h3("4. Giải Quyết Toàn Diện Bài Tập Exercise 3 – Customer Comment (Slide 66)")
add_p("Phân tích câu nhận xét kinh điển trong Slide bài giảng: \"The camera is excellent but the battery is poor.\"")
add_bullet("Câu 1 - Raw input là gì?: Là chuỗi văn bản thô hoặc danh sách token: ['The', 'camera', 'is', 'excellent', 'but', 'the', 'battery', 'is', 'poor'].", "1. ")
add_bullet("Câu 2 - TF-IDF biểu diễn câu này thế nào?: Ánh xạ thành vector thưa x in R^V chứa trọng số tần suất xuất hiện của các từ 'camera', 'excellent', 'battery', 'poor'; các từ còn lại bằng 0.", "2. ")
add_bullet("Câu 3 - TF-IDF làm mất đi thông tin gì? (Hạn chế của Bag-of-Words - Slide 33): Mất hoàn toàn thứ tự từ (Word Order) và quan hệ cấu trúc ngữ pháp. TF-IDF không biết 'excellent' bổ nghĩa cho 'camera' hay 'battery'. Đặc biệt, mô hình túi từ bị vô hiệu hóa trước hiện tượng phủ định ('The product is good' vs 'The product is not good').", "3. ")
add_bullet("Câu 4 - Tại sao Word Embeddings (Slide 34) lại vượt trội hơn?: Embeddings ánh xạ mỗi từ thành một vector dày đặc trong không gian liên tục e in R^d (d=100 hoặc 300), bảo toàn khoảng cách ngữ nghĩa giữa các từ tương đồng.", "4. ")
add_bullet("Câu 5 - Tại sao Ngữ cảnh (Context) lại quan trọng?: Một từ mang nghĩa khác nhau tùy ngữ cảnh: từ 'poor' đi với 'battery' mang nghĩa thời lượng sử dụng ngắn, khác với nghĩa nghèo khó trong kinh tế.", "5. ")
add_bullet("Câu 6 - Kiến trúc Deep Learning nào xử lý tốt chuỗi này?: RNN/LSTM (xử lý tuần tự có cổng nhớ) hoặc Transformer / Self-Attention (như BERT, GPT). Cơ chế Attention cho phép 'excellent' chú ý trực tiếp tới 'camera' và 'poor' chú ý trực tiếp tới 'battery', phân tách hoàn hảo các khía cạnh sản phẩm.", "6. ")

doc.add_page_break()

# =============================================================
# PHẦN III: TỔNG HỢP SO SÁNH LIÊN CHỦ ĐỀ & KẾT LUẬN SƯ PHẠM
# =============================================================
add_h1("PHẦN III: TỔNG HỢP SO SÁNH LIÊN CHỦ ĐỀ & KẾT LUẬN SƯ PHẠM")

add_h2("1. Bảng Khái Quát Hóa 3 Dạng Dữ Liệu (Bám Sát Slide 56)")
table_summary = doc.add_table(rows=8, cols=4)
set_table_styling(table_summary)
s_headers = ["Tiêu Chí So Sánh", "App 1: Diabetes", "App 2: House Price", "App 3: Customer NLP"]
for c_idx, h in enumerate(s_headers):
    format_cell(table_summary.cell(0, c_idx), h, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, bg_hex="1B365D", text_color_rgb=RGBColor(0xFF, 0xFF, 0xFF), font_size=9.5)

summary_rows = [
    ("Dạng dữ liệu (Modality)", "Vector bảng có cấu trúc", "Vector bảng liên tục & địa lý", "Chuỗi văn bản phi cấu trúc (TF-IDF)"),
    ("Nhiệm vụ (Task)", "Phân loại nhị phân (0/1)", "Hồi quy định giá liên tục", "Phân loại khuyến nghị (0/1)"),
    ("Kích hoạt tầng ra (Final Layer)", "Sigmoid: y_hat in [0, 1]", "Tuyến tính: y_hat = Z_3 in (-inf, +inf)", "Sigmoid: y_hat in [0, 1]"),
    ("Hàm mất mát (Loss)", "Binary Cross-Entropy (BCE)", "Mean Squared Error (MSE)", "Binary Cross-Entropy (BCE)"),
    ("Đạo hàm tầng ra (dZ)", "dZ_3 = 1/N * (y_hat - y)", "dZ_3 = 2/N * (y_hat - y)", "dZ_3 = 1/N * (y_hat - y)"),
    ("Chỉ số đánh giá chính", "Accuracy, F1-Score", "MAE, RMSE, R²-Score (AUD)", "Accuracy, F1-Score"),
    ("Nguyên lý thống nhất (Slide 56)", "y_hat = (f_3 o f_2 o f_1)(x)", "y_hat = (f_3 o f_2 o f_1)(x)", "y_hat = (f_3 o f_2 o f_1)(x)")
]
for r_idx, row_vals in enumerate(summary_rows, start=1):
    bg = "FFFFFF" if r_idx % 2 == 1 else "F9FAFC"
    format_cell(table_summary.cell(r_idx, 0), row_vals[0], bold=True, bg_hex=bg, font_size=9.5)
    format_cell(table_summary.cell(r_idx, 1), row_vals[1], bg_hex=bg, font_size=9.5)
    format_cell(table_summary.cell(r_idx, 2), row_vals[2], bold=(r_idx in [3,4,5,6]), bg_hex=bg, font_size=9.5)
    format_cell(table_summary.cell(r_idx, 3), row_vals[3], bg_hex=bg, font_size=9.5)

add_h2("2. Đúc Kết Bản Chất Machine Learning Truyền Thống vs Học Sâu (Slide 58)")
add_p("Tại sao các thuật toán cây (Random Forest, Decision Tree) lại chiếm ưu thế trên Dữ liệu Bảng (Tabular Data)?")
add_p("Do Tiên nghiệm cảm ứng (Inductive Bias): Dữ liệu bảng có ranh giới phân khúc thị trường và ngưỡng y tế sắc nét. Thuật toán cây phân chia không gian bằng các lát cắt trực giao song song với trục tọa độ (x_j >= theta), hoàn toàn trùng khớp với cấu trúc ranh giới tự nhiên của dữ liệu bảng. Ngược lại, mạng nơ-ron phải xấp xỉ các lát cắt trực giao này bằng các siêu phẳng nghiêng liên tục thông qua tổ hợp tuyến tính sum w_i x_i.")
add_p("Tại sao Deep Learning là tương lai của Dữ liệu phức tạp?")
add_p("Deep Learning tự động phát hiện các mối quan hệ phi tuyến bậc cao mà không cần kỹ sư thiết kế thủ công đặc trưng chéo, và đặc biệt là khả năng mở rộng sang Dữ liệu Đa phương thức (Multimodal Learning): kết hợp đồng thời Bảng thông số (MLP) + Ảnh thực tế (CNN) + Văn bản nhận xét (Transformer/NLP).")

add_h2("3. Giải Đáp 3 Sai Lầm Phổ Biến (Common Misconceptions - Slide 68–70)")
add_bullet("Sai lầm 1: 'Deep Learning đồng nghĩa với mô hình cực kỳ phức tạp' -> SAI. Bản chất nằm ở phép hợp thành hàm số (f_L o ... o f_1). Mạng nhỏ 3 tầng thuần NumPy đã minh họa trọn vẹn mọi nguyên lý cơ bản của học sâu.", "1. ")
add_bullet("Sai lầm 2: 'Deep Learning tự động hiểu dữ liệu mà không cần tiền xử lý' -> SAI. Thực nghiệm ở bài toán House Price chứng minh: nếu không chuẩn hóa biến mục tiêu y, MSE loss sẽ vọt lên 1.35 x 10^12 và gradient phát nổ phá hỏng toàn bộ mạng. Tiền xử lý dữ liệu vẫn là khâu quyết định thành bại.", "2. ")
add_bullet("Sai lầm 3: 'Mạng càng sâu thì hiệu năng chắc chắn càng cao' -> SAI. Tăng độ sâu quá mức trên dữ liệu đơn giản chỉ làm tăng nguy cơ quá khớp (overfitting), tiêu tốn tài nguyên và dễ gặp hiện tượng biến mất đạo hàm.", "3. ")

add_h2("4. Trả Lời Đầy Đủ 16 Câu Hỏi Tự Học Cốt Lõi (Self-Study Questions - Slide 54)")
questions_answers = [
    ("1. Sự khác biệt giữa ML truyền thống và Deep Learning?", "ML truyền thống dùng đặc trưng thủ công (phi_human) kết hợp bộ phân loại nông; Deep Learning tự động học biểu diễn phân cấp (h_1, h_2) cùng lúc với bộ dự báo qua lan truyền ngược."),
    ("2. Ý nghĩa của phương trình y_hat = f_3(f_2(f_1(X)))?", "Là phép hợp thành hàm số nhiều tầng: tầng 1 trích xuất đặc trưng thô, tầng 2 tổng hợp thành biểu diễn trừu tượng, tầng 3 đưa ra dự đoán cuối cùng."),
    ("3. Mục đích của ma trận trọng số W?", "Xác định mức độ đóng góp, tầm quan trọng và sự tương tác giữa các chiều đặc trưng khi ánh xạ sang không gian mới (Scale & Rotation)."),
    ("4. Mục đích của vector độ lệch b?", "Dịch chuyển siêu phẳng kích hoạt ra khỏi gốc tọa độ (Shift), cho phép mô hình phân tách dữ liệu linh hoạt ngay cả khi đầu vào X = 0."),
    ("5. Tại sao bắt buộc cần hàm kích hoạt phi tuyến?", "Nếu không có phi tuyến, tích các ma trận tuyến tính xếp chồng sẽ suy biến về một phép biến đổi tuyến tính đơn tầng phẳng duy nhất (W'x + b')."),
    ("6. Hàm ReLU làm nhiệm vụ gì?", "Giữ nguyên giá trị dương và triệt tiêu giá trị âm về 0 (max(0, z)). ReLU tạo tính thưa (Sparsity) và duy trì gradient bằng 1 khi z > 0, ngăn chặn biến mất đạo hàm."),
    ("7. Tại sao dùng Sigmoid ở tầng ra của bài toán phân loại?", "Sigmoid ép giá trị thực bất kỳ về khoảng xác suất trơn (0, 1), phù hợp biểu diễn xác suất P(y=1|x). Trong bài hồi quy, tầng ra dùng tuyến tính để giá trị dự đoán không bị chặn."),
    ("8. Tại sao cần hàm mất mát (Loss Function)?", "Lượng hóa sai số giữa dự đoán y_hat và nhãn thật y, cung cấp thước đo vô hướng để thuật toán tối ưu giảm thiểu."),
    ("9. Hàm Binary Cross-Entropy đo lường điều gì?", "Đo lường độ bất đồng (Divergence) giữa phân phối xác suất dự đoán của mô hình và phân phối nhãn thực nghiệm; phạt cực nặng các dự đoán sai với độ tự tin cao."),
    ("10. Gradient là gì?", "Là vector chứa toàn bộ các đạo hàm riêng dL/dtheta, chỉ hướng tăng dốc nhất của hàm mất mát trong không gian tham số."),
    ("11. Lan truyền ngược (Backpropagation) tính toán điều gì?", "Tính toán đạo hàm riêng của hàm mất mát đối với từng trọng số và độ lệch (dL/dW, dL/db) thông qua Quy tắc chuỗi (Chain Rule)."),
    ("12. Thuật toán Gradient Descent làm nhiệm vụ gì?", "Cập nhật trọng số ngược chiều gradient với bước nhảy eta (theta <- theta - eta * dL/dtheta) để kéo mô hình hội tụ về điểm cực tiểu."),
    ("13. Tại sao trọng số phải được cập nhật lặp đi lặp lại qua nhiều Epoch?", "Vì hàm mất mát là phi tuyến phức tạp; mỗi bước Gradient Descent chỉ là một xấp xỉ tuyến tính cục bộ nhỏ nên cần hàng trăm bước lặp để di chuyển về vùng tối ưu."),
    ("14. Tại sao phải phân chia tập dữ liệu thành Train và Test?", "Tập Train dùng để học tham số; tập Test được giữ độc lập hoàn toàn để kiểm tra năng lực tổng quát hóa trên dữ liệu mới chưa từng thấy, phát hiện Overfitting."),
    ("15. Tại sao phải chuẩn hóa các đặc trưng (Normalization)?", "Đưa các thuộc tính về cùng thang đo (N(0, 1)), làm cho bề mặt hàm mất mát có dạng hình tròn đồng mức, giúp Gradient Descent hội tụ thẳng và nhanh hơn."),
    ("16. Sự khác biệt giữa xác suất (Probability) và nhãn dự đoán (Class Prediction)?", "Xác suất là giá trị số thực liên tục y_hat in [0, 1] thể hiện mức độ tự tin; Nhãn dự đoán là giá trị nhị phân rời rạc (0 hoặc 1) thu được sau khi so sánh xác suất với ngưỡng quyết định tau (thường là 0.5).")
]

for q, a in questions_answers:
    add_bullet(a, bold_prefix=f"{q} -> ")

doc.add_page_break()

# =============================================================
# PHẦN IV: ĐỀ XUẤT CHỈNH SỬA & HỒ SƠ ĐÓNG GÓI
# =============================================================
add_h1("PHẦN IV: ĐỀ XUẤT CHỈNH SỬA & HỒ SƠ ĐÓNG GÓI NỘP BÀI")

add_h2("1. Đề Xuất Giá Trị Gia Tăng So Với Tài Liệu Slide")
add_p("Nhóm đề xuất 3 đóng góp quan trọng làm phong phú và sâu sắc thêm nội dung học phần:")
add_bullet("Chứng minh toán học về Linear Collapse (Slide 14–15): Đưa đầy đủ công thức W_2(W_1 x + b_1) + b_2 = W'x + b' và thực nghiệm bỏ ReLU trong cả 3 notebook để kiểm chứng thực tế sự suy biến tuyến tính.", "1. ")
add_bullet("Làm rõ Cơ chế Denormalization trong Hồi quy (Slide 29): Nhấn mạnh tầm quan trọng của việc chuẩn hóa y trong hồi quy (tránh MSE loss vọt lên 10^12) và công thức giải chuẩn hóa y_real = mu_y + std_y * y_hat để báo cáo kết quả theo đơn vị tiền tệ AUD thực tế.", "2. ")
add_bullet("Phân tích Sâu sắc Bài tập Exercise 3 (Slide 66) về NLP: Trả lời trọn vẹn 6 câu hỏi về hạn chế của Bag-of-Words / TF-IDF ('The product is good' vs 'The product is not good'), Word Embeddings và cơ chế Self-Attention của Transformer.", "3. ")

add_h2("2. Danh Mục Hồ Sơ Đóng Gói Hoàn Chỉnh Nộp Bài")
add_p("Toàn bộ sản phẩm được đóng gói trong thư mục A3_submit/ gồm:")
add_bullet("Phase1_Diabetes/diabetes_deep_learning.ipynb: 57 cells hoàn chỉnh, đã chạy Run All.", "• ")
add_bullet("Phase2_HousePrice/house_price_prediction.ipynb: 57 cells hoàn chỉnh, đã chạy Run All.", "• ")
add_bullet("Phase3_Ecommerce/customer_interest_text_deeplearning.ipynb: 57 cells hoàn chỉnh, đã chạy Run All.", "• ")
add_bullet("REPORT_A3_DEEP_LEARNING.md: Báo cáo văn bản chi tiết định dạng Markdown.", "• ")
add_bullet("BAO_CAO_ASSIGNMENT_03_DEEP_LEARNING.docx: File Word chính thức với đầy đủ hình ảnh, bảng biểu và khung chừa chỗ điền link GitHub / Dataset.", "• ")

# Lưu file word
output_docx = os.path.join("A3_submit", "BAO_CAO_ASSIGNMENT_03_DEEP_LEARNING.docx")
doc.save(output_docx)
print(f"\nWORD REPORT GENERATED SUCCESSFULLY: {output_docx}")
