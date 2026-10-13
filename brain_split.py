import math
import json

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(min(x, 500), -500)))

def sigmoid_derivative(y):
    return y * (1.0 - y)

# ==========================================================
# BỘ NÃO CHUẨN SPLIT LEARNING: SUY LUẬN + BẮN ĐẠO HÀM VỀ MẮT
# ==========================================================
def run_brain_cycle(target_y=1, lr=0.5):
    # 1. ĐÓN TÍN HIỆU THỊ GIÁC TỪ MẮT GỬI SANG
    print("📡 [BỘ NÃO] Đang đọc tín hiệu từ synapse_forward.json...")
    try:
        with open("synapse_forward.json", "r") as f:
            data = json.load(f)
            features = data["features"] # Nhận [f1, f2]
    except FileNotFoundError:
        print("❌ Chưa nhận được tín hiệu từ Mắt! Dùng tín hiệu giả lập [0.8, 0.2].")
        features = [0.8, 0.2]

    # 2. NẠP BỘ NHỚ NÃO
    with open("brain_weights.json", "r") as f:
        ky_uc = json.load(f)

    w_an = ky_uc["weights_tang_an"]
    b_an = ky_uc["bias_tang_an"]
    w_ra = ky_uc["weights_tang_ra"]
    b_ra = ky_uc["bias_tang_ra"]
    so_an = ky_uc["so_no_ron_an"]

    # 3. LAN TRUYỀN TIẾN (FORWARD PASS)
    h = []
    for j in range(so_an):
        s = (features[0] * w_an[0][j]) + (features[1] * w_an[1][j]) + b_an[j]
        h.append(sigmoid(s))

    tong_ra = sum(h[j] * w_ra[j] for j in range(so_an)) + b_ra
    pred = sigmoid(tong_ra)
    print(f"🧠 [BỘ NÃO] Dự đoán: {pred:.4f} | Nhãn thực tế: {target_y}")

    # 4. TÍNH SAI SỐ VÀ ĐẠO HÀM TẦNG RA
    sai_so = target_y - pred
    delta_ra = sai_so * sigmoid_derivative(pred)

    # 5. TÍNH ĐẠO HÀM TẦNG ẨN
    delta_an = [delta_ra * w_ra[j] * sigmoid_derivative(h[j]) for j in range(so_an)]

    # ==========================================================
    # 🔥 BƯỚC ĐỘT PHÁ: TÍNH ĐẠO HÀM ĐẦU VÀO ĐỂ BẮN VỀ MẮT!
    # dL/df_i = sum(delta_an_j * w_an[i][j])
    # ==========================================================
    grad_to_eye = [
        sum(delta_an[j] * w_an[0][j] for j in range(so_an)),
        sum(delta_an[j] * w_an[1][j] for j in range(so_an))
    ]

    # 6. CẬP NHẬT TRỌNG SỐ NỘI BỘ CỦA NÃO
    for j in range(so_an):
        w_ra[j] += lr * delta_ra * h[j]
        w_an[0][j] += lr * delta_an[j] * features[0]
        w_an[1][j] += lr * delta_an[j] * features[1]
        b_an[j] += lr * delta_an[j]
    b_ra += lr * delta_ra

    # Lưu lại bộ não mới
    with open("brain_weights.json", "w") as f:
        json.dump(ky_uc, f, indent=4)

    # 7. ĐÓNG GÓI XUNG ĐẠO HÀM BẮN TRẢ VỀ CHO MẮT!
    packet_backward = {
        "gradients_for_eye": grad_to_eye,
        "loss": round(sai_so**2, 5)
    }
    with open("synapse_backward.json", "w") as f:
        json.dump(packet_backward, f, indent=4)
        
    print(f"⚡ [BỘ NÃO] ĐÃ BẮN XUNG ĐẠO HÀM VỀ CHO MẮT: {grad_to_eye}")

if __name__ == "__main__":
    run_brain_cycle(target_y=1)
