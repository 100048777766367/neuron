import math
import json
import sys

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(min(x, 500), -500)))

# 1. NẠP KÝ ỨC TỪ TỆP JSON
print("📂 Đang nạp ký ức từ brain_weights.json...")
try:
    with open("brain_weights.json", "r", encoding="utf-8") as f:
        ky_uc = json.load(f)
except FileNotFoundError:
    print("❌ Lỗi: Chưa tìm thấy tệp ký ức brain_weights.json! Hãy chạy huấn luyện trước.")
    sys.exit(1)

w_an = ky_uc["weights_tang_an"]
b_an = ky_uc["bias_tang_an"]
w_ra = ky_uc["weights_tang_ra"]
b_ra = ky_uc["bias_tang_ra"]
so_an = ky_uc["so_no_ron_an"]

print(f"✅ Đã nạp thành công mô hình: {ky_uc['mo_hinh']} ({so_an} nơ-ron)")

# 2. HÀM DỰ ĐOÁN SIÊU TỐC (KHÔNG CẦN HUẤN LUYỆN)
def suy_luan(x1, x2):
    x = [x1, x2]
    # Lan truyền qua tầng ẩn
    h = []
    for j in range(so_an):
        tong = (x[0] * w_an[0][j]) + (x[1] * w_an[1][j]) + b_an[j]
        h.append(sigmoid(tong))
    # Lan truyền qua tầng ra
    tong_ra = sum(h[j] * w_ra[j] for j in range(so_an)) + b_ra
    xac_suat = sigmoid(tong_ra)
    return xac_suat

# 3. THỬ NGHIỆM DỰ ĐOÁN TỨC THÌ
cac_cau_hoi = [
    (0, 0),
    (0, 1),
    (1, 0),
    (1, 1),
    (0.9, 0.1) # Thử một số lạ xem nơ-ron phản ứng thế nào!
]

print("\n⚡ KẾT QUẢ SUY LUẬN SIÊU TỐC:")
for c1, c2 in cac_cau_hoi:
    ket_qua = suy_luan(c1, c2)
    print(f"Hỏi: [{c1}, {c2}] -> Nơ-ron phán: {ket_qua:.4f} (Làm tròn: {round(ket_qua)})")
