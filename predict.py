import sys
import re
from stigmergy_engine import StigmergyEngine

def main():
    # 1. Khởi tạo động cơ bầy đàn từ các túi nhớ trong synapses/
    engine = StigmergyEngine()
    
    # 2. Xử lý dữ liệu đầu vào từ dòng lệnh
    raw_str = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "1 0"
    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", raw_str)
    
    x1 = float(numbers[0]) if len(numbers) > 0 else 1.0
    x2 = float(numbers[1]) if len(numbers) > 1 else 0.0

    # 3. Phán đoán bầy đàn
    # Từng nơ-ron nhận tín hiệu
    signals = [cell.forward([x1, x2]) for cell in engine.swarm]
    prob = engine.leader.forward(signals)
    
    prediction = 1 if prob >= 0.5 else 0
    confidence = prob * 100 if prediction == 1 else (1.0 - prob) * 100

    print("----------------------------------------")
    print(f"🎯 ĐẦU VÀO     : [{x1}, {x2}]")
    print(f"🔮 DỰ ĐOÁN BẦY : {prediction}")
    print(f"📊 ĐỘ TIN CẬY  : {confidence:.2f}% (Xác suất: {prob:.4f})")
    print("----------------------------------------")

if __name__ == "__main__":
    main()
