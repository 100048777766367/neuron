import sys
import re
from swarm_engine import SwarmEngine

def main():
    swarm = SwarmEngine()
    
    # Lấy đầu vào từ dòng lệnh
    raw_str = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "1 0"
    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", raw_str)
    x1 = float(numbers[0]) if len(numbers) > 0 else 1.0
    x2 = float(numbers[1]) if len(numbers) > 1 else 0.0

    prob = swarm.predict([x1, x2])
    prediction = 1 if prob >= 0.5 else 0
    confidence = prob * 100 if prediction == 1 else (1.0 - prob) * 100

    print("-" * 40)
    print(f"🎯 ĐẦU VÀO     : [{x1}, {x2}]")
    print(f"🔮 DỰ ĐOÁN BẦY : {prediction}")
    print(f"📊 ĐỘ TIN CẬY  : {confidence:.2f}% (Xác suất: {prob:.4f})")
    print("-" * 40)

if __name__ == "__main__":
    main()
