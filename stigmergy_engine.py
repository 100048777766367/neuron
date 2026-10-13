import os
import json
import time
from cells.base_neuron import CellNeuron, sigmoid_derivative

NUM_CELLS = 16
STATE_LOG = "synapses/stigmergy_state.json"

class StigmergyEngine:
    def __init__(self):
        # 1. KHỞI TẠO BẦY NƠ-RON TỰ TRỊ (CÓ CƠ CHẾ HA TỰ PHỤC HỒI)
        self.swarm = []
        for i in range(NUM_CELLS):
            try:
                self.swarm.append(CellNeuron(f"h_{i}", n_inputs=2))
            except Exception as e:
                print(f"⚠️ [HA] Nơ-ron h_{i} gặp sự cố, tự phục hồi khẩn cấp: {e}")
                self.swarm.append(CellNeuron(f"h_{i}", n_inputs=2))

        self.leader = CellNeuron("out", n_inputs=NUM_CELLS)

    # 2. ĐỌC NHẬT KÝ MÔI TRƯỜNG ĐỂ BIẾT ĐÃ XỬ LÝ ĐẾN ĐÂU
    def get_last_processed_id(self):
        if os.path.exists(STATE_LOG):
            try:
                with open(STATE_LOG, "r") as f:
                    return json.load(f).get("last_processed_timestamp", 0)
            except Exception:
                return 0
        return 0

    def mark_processed(self, timestamp):
        os.makedirs("synapses", exist_ok=True)
        with open(STATE_LOG, "w") as f:
            json.dump({
                "last_processed_timestamp": timestamp,
                "updated_at": time.time(),
                "status": "healthy_and_waiting"
            }, f, indent=4)

    # 3. TRÁI TIM BẤT ĐỒNG BỘ: NGỦ DẬY -> KIỂM TRA MÔI TRƯỜNG -> XỬ LÝ TIẾP
    def process_inbox(self, signal_file="synapse_forward.json"):
        if not os.path.exists(signal_file):
            print("🍃 [MÔI TRƯỜNG TĨNH] Không tìm thấy tệp tín hiệu nào trong Inbox. Tiếp tục ngủ.")
            return

        try:
            with open(signal_file, "r") as f:
                data = json.load(f)
        except Exception as e:
            print(f"❌ [LỖI TỆP] Tệp tín hiệu bị hỏng hoặc chưa ghi xong: {e}")
            return

        signal_timestamp = data.get("timestamp", 0)
        last_timestamp = self.get_last_processed_id()

        # NẾU TÍN HIỆU ĐÃ TỪNG XỬ LÝ TRƯỚC KHI NGỦ -> BỎ QUA ĐỂ TRÁNH TRÙNG LẶP
        if signal_timestamp > 0 and signal_timestamp <= last_timestamp:
            print(f"💤 [BỎ QUA] Tín hiệu (id: {signal_timestamp}) đã được xử lý từ trước. Không cần làm lại.")
            return

        print(f"⚡ [THỨC DẬY XỬ LÝ] Phát hiện tín hiệu mới từ Mắt (Timestamp: {signal_timestamp})!")
        features = data.get("features", [0.5, 0.5])

        # --- GIAI ĐOẠN SUY LUẬN (FORWARD) ---
        signals = [cell.forward(features) for cell in self.swarm]
        pred = self.leader.forward(signals)
        print(f"🔮 [BẦY PHÁN QUYẾT] Đầu vào {features} -> Kết quả dự đoán: {pred:.4f}")

        # --- GIAI ĐOẠN ĐẠO HÀM TRẢ VỀ CHO MẮT (BACKWARD) ---
        target_y = 1
        delta_out = (target_y - pred) * sigmoid_derivative(pred)
        deltas_swarm = [
            delta_out * self.leader.weights[j] * sigmoid_derivative(signals[j])
            for j in range(NUM_CELLS)
        ]

        grad_eye = [
            sum(deltas_swarm[j] * self.swarm[j].weights[0] for j in range(NUM_CELLS)),
            sum(deltas_swarm[j] * self.swarm[j].weights[1] for j in range(NUM_CELLS))
        ]

        # Từng tế bào tự lưu lại túi nhớ của riêng nó
        self.leader.backward(delta_out)
        self.leader.save_self()
        for j, cell in enumerate(self.swarm):
            cell.backward(deltas_swarm[j])
            cell.save_self()

        # --- ĐỂ LẠI VỆT MÙI MỚI RA MÔI TRƯỜNG CHO MẮT ---
        backward_packet = {
            "reply_to_timestamp": signal_timestamp,
            "processed_at": time.time(),
            "gradients_for_eye": grad_eye,
            "node_status": "synced"
        }
        with open("synapse_backward.json", "w") as f:
            json.dump(backward_packet, f, indent=4)

        # Ghi chú lại là đã xử lý xong phiên này
        self.mark_processed(signal_timestamp)
        print("✅ [HOÀN TẤT CHU KỲ] Đã để lại tệp synapse_backward.json trên môi trường và lưu trạng thái!")

if __name__ == "__main__":
    engine = StigmergyEngine()
    
    # Đảm bảo nếu chưa có túi nhớ nền tảng thì tự tạo trước
    if not os.path.exists("synapses/out.json"):
        print("🌱 [KHỞI SINH] Tạo túi nhớ nền tảng...")
        # Tạo nhanh trạng thái mặc định
        engine.leader.save_self()
        for cell in engine.swarm:
            cell.save_self()

    # Chạy kiểm tra môi trường bất đồng bộ
    engine.process_inbox()
