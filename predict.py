import sys
import re
from brain_mesh import BrainMesh

def parse_inputs(args):
    """
    Tự động xử lý mọi định dạng đầu vào:
    Hỗ trợ: "1 0", "[1, 0]", "1,0", "0.8 0.2", v.v.
    """
    if len(args) <= 1:
        # Mặc định nếu không truyền tham số nào
        return [1.0, 0.0]
    
    # Gộp toàn bộ tham số lại thành 1 chuỗi
    raw_str = " ".join(args[1:])
    
    # Dùng Regular Expression lọc lấy tất cả các số (kể cả số thập phân âm/dương)
    numbers = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", raw_str)
    
    if len(numbers) < 2:
        raise ValueError(f"Cần tối thiểu 2 số đầu vào, nhưng chỉ nhận được: {numbers}")
        
    return [float(numbers[0]), float(numbers[1])]

def main():
    try:
        # 1. NẠP MẠNG LƯỚI NƠ-RON NGUYÊN TỬ
        mesh = BrainMesh()
        
        # 2. XỬ LÝ ĐẦU VÀO TỪ DÒNG LỆNH
        inputs = parse_inputs(sys.argv)
        x1, x2 = inputs[0], inputs[1]

        # 3. SUY LUẬN SIÊU TỐC
        prob = mesh.predict([x1, x2])
        prediction = 1 if prob >= 0.5 else 0

        # Tính độ tin cậy thực tế (Confidence score)
        if prediction == 1:
            confidence = prob * 100.0
        else:
            confidence = (1.0 - prob) * 100.0

        # 4. XUẤT KẾT QUẢ ĐỊNH DẠNG ĐẸP
        print("-" * 40)
        print(f"🎯 ĐẦU VÀO     : [{x1}, {x2}]")
        print(f"🔮 DỰ ĐOÁN     : {prediction}")
        print(f"📊 ĐỘ TIN CẬY  : {confidence:.2f}% (Xác suất gốc: {prob:.4f})")
        print("-" * 40)

    except Exception as e:
        print(f"❌ LỖI SUY LUẬN: {str(e)}")
        print("💡 Hướng dẫn: Nhập đúng 2 con số, ví dụ: python predict.py 1 0")
        sys.exit(1)

if __name__ == "__main__":
    main()
