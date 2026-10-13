# Sử dụng nền tảng Python 3.10 siêu nhẹ và bảo mật
FROM python:3.14-slim

# Thiết lập thư mục làm việc bên trong chiếc hộp
WORKDIR /app

# Tối ưu hóa: Ngăn Python sinh rác (.pyc) và xuất log lập tức
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Sao chép toàn bộ thế giới nơ-ron phân tán vào chiếc hộp
COPY cells/ ./cells/
COPY synapses/ ./synapses/
COPY stigmergy_engine.py .
COPY predict.py .

# Mặc định khi bật hộp lên: Kích hoạt động cơ bầy đàn cảm ứng môi trường
CMD ["python", "stigmergy_engine.py"]
