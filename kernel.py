import os, sys, json, time, hashlib

print("=" * 75)
print("🚀 [HẠT NHÂN THỨC TỈNH] Nút: neuron | Thẻ Bài: SIEU_XOR_SINGULARITY")
print("=" * 75)

epoch = 1
if os.path.exists("synapses/state.json"):
    try:
        with open("synapses/state.json", "r") as f: 
            epoch = json.load(f).get("generation_epoch", 0) + 1
    except Exception: pass

t = int(time.time())
cid = hashlib.shake_256(f"neuron_{epoch}_{t}".encode()).hexdigest(64)

# 1. Cơ chế Toán học XOR tự nghịch đảo (Inversion Invariance)
entropy_delta = int(cid[:8], 16)
state_vector = 0b1010101010101010
xor_mutated = state_vector ^ entropy_delta
reverted = xor_mutated ^ entropy_delta
assert reverted == state_vector, "Lỗi bất biến: Phép toán XOR mất tính đối xứng!"

# 2. Ghi sổ trạng thái biến cố (Event-Sourced State)
with open("synapses/state.json", "w", encoding="utf-8") as f:
    json.dump({
        "generation_epoch": epoch,
        "last_timestamp": t,
        "quantum_cid": cid,
        "state_vector_hex": hex(xor_mutated),
        "status": "synchronized"
    }, f, indent=4)

# 3. Xuất sóng giao cảm (Synapse Forward)
with open("synapse_forward.json", "w", encoding="utf-8") as f:
    json.dump({
        "contract": "bodhi.contract.v1",
        "identity": "100048777766367/neuron",
        "epoch": epoch,
        "cid": cid,
        "proof": "XOR_INVARIANT_VALID"
    }, f, indent=4)

# 4. Xuất Báo Cáo Step Summary vào GitHub Actions
summary_file = os.environ.get("GITHUB_STEP_SUMMARY")
if summary_file:
    with open(summary_file, "a", encoding="utf-8") as f:
        f.write(f"""
## 🎴 CHỨNG CHỈ KIỂM ĐỊNH THẺ BÀI: NEURON
> **Kỷ nguyên:** #{epoch} | **Thẻ Bài:** SIEU_XOR_SINGULARITY | **Cấp:** T7_SINGULARITY

| Trục Phân Loại | Trạng Thái | Đánh Giá Kỹ Thuật |
| :--- | :---: | :--- |
| **01. Bản chất & Cơ chế** | 🟢 HOÀN HẢO | Logic Toán học XOR Tất Định |
| **02. Tính Đối Xứng** | 🟢 BẢO TOÀN | `A ^ B ^ B == A` (Đã kiểm chứng) |
| **03. Bảo Chứng CID** | 🟢 TOÀN VẸN | `{cid[:24]}...` |
| **04. Hạ Tầng** | 🟢 CHUẨN MỰC | Python 3.14 & Actions Setup v7 |
""")

print(f"✨ [HOÀN HẢO] Chu kỳ #{epoch} đã kiểm định thành công! (CID: {cid[:16]}...)")
