FROM python:3.14-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
COPY . .
CMD ["sh", "-c", "if [ -f kernel.py ]; then python -B kernel.py; else python -B engine.py; fi"]
