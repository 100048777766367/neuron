import math
import json
import sys

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(min(x, 500), -500)))

# 1. NẠP KÝ ỨC
with open("brain_weights.json", "r", encoding="utf-8") as f:
    ky_uc = json.load(f)

w_an = ky_uc["weights_tang_an"]
b_an = ky_uc["bias_tang_an"]
w_ra = ky_uc["weights_tang_ra"]
b_ra = ky_uc["bias_tang_ra"]
so_an = ky_uc["so_no_ron_an"]

# 2. ĐỌC DỮ LIỆU ĐẦU VÀO TỪ DÒNG LỆNH
# Mặc định là [1, 0] nếu người dùng không truyền gì
x1 = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
x2 = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0

# 3. SUY LUẬN
h = [sigmoid((x1 * w_an[0][j]) + (x2 * w_an[1][j]) + b_an[j]) for j in range(so_an)]
tong_ra = sum(h[j] * w_ra[j] for j in range(so_an)) + b_ra
xac_suat = sigmoid(tong_ra)
ket_qua = round(xac_suat)

# In kết quả theo định dạng đẹp để Bot lấy gửi cho người hỏi
print(f"Đầu vào: [{x1}, {x2}]")
print(f"Dự đoán: {ket_qua} (Độ tin cậy: {xac_suat*100:.2f}%)")
