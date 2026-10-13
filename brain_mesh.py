import json
import os
from atomic_neuron import AtomicNeuron, sigmoid_derivative

STATE_FILE = "brain_weights.json"

class BrainMesh:
    def __init__(self, n_in=2, n_hidden=16):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.hidden_layer = []
        self.output_neuron = None
        
        # TỰ ĐỘNG KHỞI TẠO HOẶC NẠP KÝ ỨC CŨ
        if os.path.exists(STATE_FILE):
            self.load_mesh()
        else:
            print("🌱 [KHỞI SINH] Tạo mạng lưới nơ-ron nguyên tử mới...")
            self.init_mesh()
            self.train_baseline()

    def init_mesh(self):
        # Tạo 16 nơ-ron tầng ẩn độc lập
        self.hidden_layer = [
            AtomicNeuron(f"hidden_{j}", self.n_in) for j in range(self.n_hidden)
        ]
        # Tạo 1 nơ-ron tầng ra độc lập
        self.output_neuron = AtomicNeuron("output", self.n_hidden)

    # 1. BÀI TOÁN TỰ SINH NỀN TẢNG (Cho mọi bản Fork mới)
    def train_baseline(self, epochs=5000, lr=0.3):
        print(f"🚀 Tự động huấn luyện mạng lưới ({self.n_hidden} nơ-ron nguyên tử)...")
        data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
        
        for _ in range(epochs):
            for x, y_that in data:
                # Lan truyền qua từng nơ-ron nguyên tử
                h = [neuron.forward(x) for neuron in self.hidden_layer]
                out = self.output_neuron.forward(h)
                
                # Tính sai số và đạo hàm
                delta_out = (y_that - out) * sigmoid_derivative(out)
                deltas_h = [
                    delta_out * self.output_neuron.weights[j] * sigmoid_derivative(h[j])
                    for j in range(self.n_hidden)
                ]
                
                # Từng nơ-ron tự cập nhật độc lập
                self.output_neuron.backward(delta_out, lr)
                for j, neuron in enumerate(self.hidden_layer):
                    neuron.backward(deltas_h[j], lr)
                    
        self.save_mesh()
        print("✅ Mạng lưới đã hoàn thành bài học nền tảng!")

    # 2. SUY LUẬN SIÊU TỐC
    def predict(self, x):
        h = [neuron.forward(x) for neuron in self.hidden_layer]
        return self.output_neuron.forward(h)

    # 3. CẢM ỨNG MÔI TRƯỜNG (STIGMERGY & BẮN ĐẠO HÀM VỀ MẮT)
    def process_stigmergy(self, forward_file="synapse_forward.json", target_y=1, lr=0.5):
        if not os.path.exists(forward_file):
            print("🍃 [MÔI TRƯỜNG TĨNH] Không phát hiện tín hiệu mới từ Mắt.")
            return

        with open(forward_file, "r") as f:
            features = json.load(f).get("features", [0.5, 0.5])

        print(f"📡 [CẢM ỨNG] Nhận tín hiệu: {features}")
        h = [neuron.forward(features) for neuron in self.hidden_layer]
        pred = self.output_neuron.forward(h)

        delta_out = (target_y - pred) * sigmoid_derivative(pred)
        deltas_h = [
            delta_out * self.output_neuron.weights[j] * sigmoid_derivative(h[j])
            for j in range(self.n_hidden)
        ]

        # Tính đạo hàm để bắn ngược lại môi trường
        grad_to_eye = [
            sum(deltas_h[j] * self.hidden_layer[j].weights[0] for j in range(self.n_hidden)),
            sum(deltas_h[j] * self.hidden_layer[j].weights[1] for j in range(self.n_hidden))
        ]

        # Tự cập nhật
        self.output_neuron.backward(delta_out, lr)
        for j, neuron in enumerate(self.hidden_layer):
            neuron.backward(deltas_h[j], lr)

        self.save_mesh()

        # Để lại vệt đạo hàm ra môi trường
        with open("synapse_backward.json", "w") as f:
            json.dump({"gradients_for_eye": grad_to_eye, "loss": round((target_y - pred)**2, 5)}, f, indent=4)
        print(f"⚡ [XUNG ĐẠO HÀM] Đã gửi vệt đạo hàm cho Mắt: {grad_to_eye}")

    def save_mesh(self):
        state = {
            "hidden_neurons": [n.export_state() for n in self.hidden_layer],
            "output_neuron": self.output_neuron.export_state()
        }
        with open(STATE_FILE, "w") as f:
            json.dump(state, f, indent=4)

    def load_mesh(self):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
        self.hidden_layer = [
            AtomicNeuron(n["id"], self.n_in, n["weights"], n["bias"])
            for n in state["hidden_neurons"]
        ]
        out_state = state["output_neuron"]
        self.output_neuron = AtomicNeuron(out_state["id"], self.n_hidden, out_state["weights"], out_state["bias"])
        print(f"🧠 [KÝ ỨC] Đã nạp thành công mạng lưới {len(self.hidden_layer)} nơ-ron nguyên tử.")

if __name__ == "__main__":
    mesh = BrainMesh()
    mesh.process_stigmergy()
