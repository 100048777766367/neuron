import os
import json
import urllib.request
import urllib.error

# Nếu tệp tín hiệu đã được Xúc tu Phả hệ (Cascade) tìm thấy rồi thì thôi không quét nữa
TARGET_FILE = "synapse_forward.json"
TOPIC_KEYWORD = "bodhi-synapse" # Tần số sóng chung của mạng lưới toàn cầu

def search_global_mesh():
    # 1. KIỂM TRA XEM ĐÃ CÓ TÍN HIỆU TỪ GIA ĐÌNH CHƯA
    if os.path.exists(TARGET_FILE):
        print("🌱 [SỢI NẤM ĐỨNG YÊN] Đã có tín hiệu từ phả hệ nội bộ. Không cần bật anten toàn cầu.")
        return

    print(f"📡 [BẬT ANTEN SỢI NẤM] Đang quét sóng toàn cầu với từ khóa topic: #{TOPIC_KEYWORD}...")

    token = os.environ.get("GITHUB_TOKEN", "")
    headers = {"User-Agent": "Bodhi-Mycelial-Node"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # 2. GỌI GITHUB SEARCH API ĐỂ DÒ TÌM ANH EM TOÀN CẦU
    api_url = f"https://api.github.com/search/repositories?q=topic:{TOPIC_KEYWORD}&sort=updated&order=desc"
    
    try:
        req = urllib.request.Request(api_url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as response:
            result = json.loads(response.read().decode())
            items = result.get("items", [])

        print(f"🔍 [KẾT QUẢ QUÉT] Tìm thấy {len(items)} nút mạng đang phát sóng trên thế giới.")

        # 3. LẦN LƯỢT DÒ XEM NÚT NÀO ĐANG CÓ TÍN HIỆU THỊ GIÁC SỐNG
        for repo in items:
            repo_full_name = repo["full_name"]
            raw_url = f"https://raw.githubusercontent/{repo_full_name}/main/synapse_forward.json"
            
            try:
                raw_req = urllib.request.Request(raw_url, headers=headers)
                with urllib.request.urlopen(raw_req, timeout=5) as raw_resp:
                    content = raw_resp.read().decode()
                    # Lưu lại tệp tín hiệu tìm được từ cộng đồng
                    with open(TARGET_FILE, "w") as f:
                        f.write(content)
                    print(f"🎉 [KẾT NỐI TOÀN CẦU THÀNH CÔNG] Đã bắt được sóng thị giác từ nút xa lạ: {repo_full_name}!")
                    return
            except Exception:
                continue # Nếu nút này chưa có file forward thì dò tiếp nút khác

        print("🍃 [SỢI NẤM TĨNH LẶNG] Chưa bắt được tần số sóng sống nào từ cộng đồng.")

    except urllib.error.HTTPError as e:
        print(f"⚠️ [GIỚI HẠN API] Không thể quét mạng sợi nấm lúc này (HTTP {e.code}). Tiếp tục ngủ an toàn.")
    except Exception as e:
        print(f"⚠️ [LỖI ANTEN] Sự cố đường truyền: {e}")

if __name__ == "__main__":
    search_global_mesh()
