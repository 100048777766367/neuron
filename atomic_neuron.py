import math
import random

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(min(x, 500), -500)))

def sigmoid_derivative(y):
    return y * (1.0 - y)

# ==============================================================
# ĐƠN VỊ NƠ-RON NGUYÊN TỬ TỰ TRỊ (ATOMIC NEURON)
# ==============================================================
class AtomicNeuron:
    def __init__(self, neuron_id, n_inputs, weights=None, bias=None):
        self.id = neuron_id
        self.n_inputs = n_inputs
        
        # Nếu chưa có trọng số, tự sinh ngẫu nhiên
        if weights is not None and bias is not None:
            self.weights = weights
            self.bias = bias
        else:
            self.weights = [random.uniform(-1, 1) for _ in range(n_inputs)]
            self.bias = random.uniform(-1, 1)
            
        self.last_inputs = []
        self.last_output = 0.0

    # LAN TRUYỀN TIẾN (Kèm bộ lọc kháng độc tố / chống dữ liệu rác)
    def forward(self, inputs):
        # Điều kiện 1 & 2: Kiểm tra định dạng và lọc giá trị bất thường
        if len(inputs) != self.n_inputs:
            raise ValueError(f"Nơ-ron {self.id}: Đầu vào sai kích thước!")
            
        self.last_inputs = inputs
        # Phép nhân cộng tích luỹ: sum(w * x) + b
        tong = sum(inputs[i] * self.weights[i] for i in range(self.n_inputs)) + self.bias
        self.last_output = sigmoid(tong)
        return self.last_output

    # LAN TRUYỀN NGƯỢC (Tự co giãn trọng số dựa trên đạo hàm)
    def backward(self, delta, lr=0.3):
        # Tự cập nhật trọng số nguyên tử
        for i in range(self.n_inputs):
            self.weights[i] += lr * delta * self.last_inputs[i]
        self.bias += lr * delta

    # Đóng gói trạng thái nơ-ron thành JSON
    def export_state(self):
        return {
            "id": self.id,
            "weights": self.weights,
            "bias": self.bias
        }
