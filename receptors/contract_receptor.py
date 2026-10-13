import json
import os

RAW_SIGNAL = "synapse_forward.json"
VERIFIED_SIGNAL = "verified_synapse.json"

def verify_and_purify():
    if not os.path.exists(RAW_SIGNAL):
        return False

    try:
        with open(RAW_SIGNAL, "r") as f:
            data = json.load(f)

        features = data.get("features", [])
        # KIỂM ĐỊNH Ổ KHÓA: Phải là mảng có ít nhất 2 đặc trưng số
        if not isinstance(features, list) or len(features) < 2:
            print(f"⛔ [HỆ MIỄN DỊCH CHẶN] Tín hiệu sai cấu trúc hợp đồng: {data}")
            return False

        # Tinh chế dữ liệu: Ép chặt vào khoảng an toàn [0.0, 1.0]
        clean_features = [max(0.0, min(1.0, float(f))) for f in features[:2]]
        
        purified_packet = {
            "contract": "bodhi.vision.v1",
            "timestamp": data.get("timestamp", 0),
            "features": clean_features
        }

        with open(VERIFIED_SIGNAL, "w") as f:
            json.dump(purified_packet, f, indent=4)

        print(f"🛡️ [LỌC THÀNH CÔNG] Tín hiệu tinh khiết đã sẵn sàng: {clean_features}")
        return True
    except Exception as e:
        print(f"⚠️ [ĐỘC TỐ] Tệp tín hiệu bị hỏng: {e}")
        return False

if __name__ == "__main__":
    verify_and_purify()
