import math
import random
import json

# Hàm kích hoạt phi tuyến
def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(min(x, 500), -500)))

def sigmoid_derivative(y):
    return y * (1.0 - y)

# ==========================================
# 1. KHỞI TẠO BỘ NÃO NHIỀU NƠ-RON (2 -> 16 -> 1)
# ==========================================
random.seed(42)
SO_DAU_VAO = 2
SO_NO_RON_AN = 16  # Mở rộng lên 16 nơ-ron trung gian!
SO_DAU_RA = 1

# Ma trận trọng số (weights) và ngưỡng (bias)
w_an = [[random.uniform(-1, 1) for _ in range(SO_NO_RON_AN)] for _ in range(SO_DAU_VAO)]
b_an = [random.uniform(-1, 1) for _ in range(SO_NO_RON_AN)]

w_ra = [random.uniform(-1, 1) for _ in range(SO_NO_RON_AN)]
b_ra = random.uniform(-1, 1)

du_lieu = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
toc_do_hoc = 0.3

# ==========================================
# 2. VÒNG LẶP HUẤN LUYỆN
# ==========================================
print(f"🚀 Đang huấn luyện Bộ não quy mô {SO_NO_RON_AN} nơ-ron...")

for epoch in range(10000):
    for x, y_that in du_lieu:
        # Lan truyền tiến qua 16 nơ-ron
        h = []
        for j in range(SO_NO_RON_AN):
            tong = sum(x[i] * w_an[i][j] for i in range(SO_DAU_VAO)) + b_an[j]
            h.append(sigmoid(tong))
            
        tong_ra = sum(h[j] * w_ra[j] for j in range(SO_NO_RON_AN)) + b_ra
        du_doan = sigmoid(tong_ra)
        
        # Lan truyền ngược sửa sai
        sai_so = y_that - du_doan
        delta_ra = sai_so * sigmoid_derivative(du_doan)
        
        delta_an = [delta_ra * w_ra[j] * sigmoid_derivative(h[j]) for j in range(SO_NO_RON_AN)]
        
        # Cập nhật trọng số
        for j in range(SO_NO_RON_AN):
            w_ra[j] += toc_do_hoc * delta_ra * h[j]
            for i in range(SO_DAU_VAO):
                w_an[i][j] += toc_do_hoc * delta_an[j] * x[i]
            b_an[j] += toc_do_hoc * delta_an[j]
        b_ra += toc_do_hoc * delta_ra

# ==========================================
# 3. XUẤT KÝ ỨC THÀNH TỆP JSON
# ==========================================
ky_uc = {
    "mo_hinh": "Deep-Neuron-16",
    "so_no_ron_an": SO_NO_RON_AN,
    "weights_tang_an": w_an,
    "bias_tang_an": b_an,
    "weights_tang_ra": w_ra,
    "bias_tang_ra": b_ra
}

with open("brain_weights.json", "w", encoding="utf-8") as f:
    json.dump(ky_uc, f, indent=4)

print("💾 Đã lưu ký ức nơ-ron thành công vào tệp: brain_weights.json")

# Tự kiểm tra kết quả
for x, y_that in du_lieu:
    h = [sigmoid(sum(x[i] * w_an[i][j] for i in range(SO_DAU_VAO)) + b_an[j]) for j in range(SO_NO_RON_AN)]
    ket_qua = sigmoid(sum(h[j] * w_ra[j] for j in range(SO_NO_RON_AN)) + b_ra)
    print(f"Đầu vào: {x} -> Dự đoán: {ket_qua:.4f} (Chuẩn: {round(ket_qua)})")
