import math
import random

# ==========================================
# 1. PHÉP TÍNH THỨ 5: HÀM KÍCH HOẠT PHI TUYẾN (SIGMOID)
# ==========================================
# Bẻ cong không gian và ép giá trị về khoảng [0, 1]
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(min(x, 500), -500)))

# Đạo hàm của Sigmoid (để phục vụ phép TRỪ sai số khi học)
def sigmoid_derivative(y):
    return y * (1.0 - y)

# ==========================================
# 2. KHỞI TẠO CẤU TRÚC 3 NƠ-RON
# ==========================================
# Đầu vào: 2 tín hiệu (x1, x2)
# Lớp ẩn: 2 nơ-ron (N1, N2)
# Đầu ra: 1 nơ-ron (N3)

# Khởi tạo trọng số (Weights) và ngưỡng (Bias) ngẫu nhiên
random.seed(42)

# Trọng số từ Đầu vào -> 2 Nơ-ron ẩn
w_in_hidden = [[random.uniform(-1, 1) for _ in range(2)] for _ in range(2)]
b_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]

# Trọng số từ 2 Nơ-ron ẩn -> Nơ-ron đầu ra
w_hidden_out = [random.uniform(-1, 1), random.uniform(-1, 1)]
b_out = random.uniform(-1, 1)

# Tập dữ liệu huấn luyện XOR
du_lieu = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0)
]

# Tốc độ học (Learning Rate)
toc_do_hoc = 0.5

# ==========================================
# 3. VÒNG LẶP TIẾN HÓA: "FOR VÀ TIẾP TỤC"
# ==========================================
print("🚀 Đang khởi động mạng nơ-ron...")

for epoch in range(10000): # Chạy 10.000 vòng lặp tự sửa sai
    tong_sai_so = 0
    
    for x, y_that in du_lieu:
        # --- BƯỚC A: LAN TRUYỀN TIẾN (NHÂN + CỘNG + BẺ CONG PHI TUYẾN) ---
        # 1. Tính toán ở 2 nơ-ron ẩn:
        hidden_outputs = []
        for j in range(2):
            # Nhân trọng số + Cộng ngưỡng: (x0*w0 + x1*w1) + b
            tong_an = (x[0] * w_in_hidden[0][j]) + (x[1] * w_in_hidden[1][j]) + b_hidden[j]
            # Kích hoạt phi tuyến:
            hidden_outputs.append(sigmoid(tong_an))
            
        # 2. Tính toán ở nơ-ron đầu ra:
        tong_ra = (hidden_outputs[0] * w_hidden_out[0]) + (hidden_outputs[1] * w_hidden_out[1]) + b_out
        du_doan = sigmoid(tong_ra)
        
        # --- BƯỚC B: TÍNH SAI SỐ (HỖN LOẠN / NHIỄU) ---
        sai_so = y_that - du_doan
        tong_sai_so += sai_so ** 2
        
        # --- BƯỚC C: SỬA SAI BẰNG PHÉP TRỪ (BACKPROPAGATION) ---
        # Tính độ dốc (Gradient) ở đầu ra:
        delta_out = sai_so * sigmoid_derivative(du_doan)
        
        # Tính độ dốc dội ngược về 2 nơ-ron ẩn:
        delta_hidden = [
            delta_out * w_hidden_out[j] * sigmoid_derivative(hidden_outputs[j])
            for j in range(2)
        ]
        
        # CẬP NHẬT TRỌNG SỐ BẰNG PHÉP CỘNG/TRỪ ĐI SAI SỐ:
        # Cập nhật tầng ra:
        for j in range(2):
            w_hidden_out[j] += toc_do_hoc * delta_out * hidden_outputs[j]
        b_out += toc_do_hoc * delta_out
        
        # Cập nhật tầng ẩn:
        for i in range(2):
            for j in range(2):
                w_in_hidden[i][j] += toc_do_hoc * delta_hidden[j] * x[i]
            b_hidden[j] += toc_do_hoc * delta_hidden[j]

    # Cứ mỗi 2000 vòng lặp thì báo cáo độ giảm sai số
    if (epoch + 1) % 2000 == 0:
        print(f"Vòng lặp {epoch + 1}: Tổng sai số = {tong_sai_so:.5f}")

# ==========================================
# 4. THỬ NGHIỆM KẾT QUẢ ĐÃ THẤU SUỐT
# ==========================================
print("\n✅ KẾT QUẢ DỰ ĐOÁN SAU KHI HỌC:")
for x, y_that in du_lieu:
    # Chạy thử 1 lần:
    h = [sigmoid((x[0]*w_in_hidden[0][j]) + (x[1]*w_in_hidden[1][j]) + b_hidden[j]) for j in range(2)]
    ket_qua = sigmoid((h[0]*w_hidden_out[0]) + (h[1]*w_hidden_out[1]) + b_out)
    print(f"Đầu vào: {x} -> Thực tế: {y_that} -> Mạng nơ-ron đoán: {ket_qua:.4f} (Làm tròn: {round(ket_qua)})")
