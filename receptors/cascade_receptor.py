import os
import subprocess

OWNER = os.environ.get("REPO_OWNER", "")
PARENT = os.environ.get("REPO_PARENT", "")
UPSTREAM_SEED = "100048777766367"  # Cội nguồn hạt mầm ban đầu

TARGET_FILE = "synapse_forward.json"

def try_fetch(target_owner):
    if not target_owner:
        return False
    url = f"https://raw.githubusercontent/{target_owner}/retina/main/synapse_forward.json"
    print(f"📡 [XÚC TU DÒ TÌM] Đang vươn vòi tìm Mắt tại: {target_owner}/retina...")
    res = subprocess.run(["curl", "-s", "-f", url, "-o", TARGET_FILE], capture_output=True)
    if res.returncode == 0 and os.path.exists(TARGET_FILE):
        print(f"🎯 [BẮT ĐƯỢC SÓNG] Đã hút thành công xung thị giác từ {target_owner}!")
        return True
    return False

def main():
    # BẬC 1: Dò tìm Mắt bản địa của chính tài khoản này
    if OWNER and try_fetch(OWNER):
        return

    # BẬC 2: Nếu là Fork, tự động vươn vòi bú sữa từ Mắt của Parent
    if PARENT and try_fetch(PARENT):
        return

    # BẬC 3: Dò tìm về Mắt cội nguồn nguyên thủy
    if try_fetch(UPSTREAM_SEED):
        return

    print("🍃 [MÔI TRƯỜNG TĨNH] Không tìm thấy Mắt ở bất kỳ nhánh phả hệ nào.")

if __name__ == "__main__":
    main()
