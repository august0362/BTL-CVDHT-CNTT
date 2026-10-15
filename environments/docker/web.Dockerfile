# Tiến trình web: Streamlit + điều phối + POSTER/MA3D-Net/YOLO/RealESRGAN/3DDFA-V2 trên CPU (ADR-012).
# Build từ gốc dự án:  docker compose build web   (hoặc: docker build -f environments/docker/web.Dockerfile .)
# Trọng số KHÔNG nằm trong image: compose mount từ backend/model/* (tải bằng environments/docker/download_weights.py).
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    YOLO_CONFIG_DIR=/tmp/ultralytics \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# torch bản CPU trước (PyPI mặc định kéo bản CUDA vài GB); phiên bản trùng environments/web/requirements-real.txt.
RUN pip install --no-cache-dir torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cpu

COPY environments/web/requirements.txt environments/web/requirements-real.txt environments/web/
# scipy/spandrel: frontalization và RealESRGAN import nhưng chưa có trong requirements-real.txt (bản đã kiểm trong venv dự án).
RUN pip install --no-cache-dir -r environments/web/requirements-real.txt scipy==1.18.1 spandrel==0.4.1

# 3DDFA-V2 upstream ghim commit (chỉ dùng models/mobilenet_v1.py; trọng số mount riêng vào backend/model/3ddfa_v2).
RUN git clone --quiet https://github.com/cleardusk/3DDFA_V2.git backend/third_party/3DDFA_V2 \
    && git -C backend/third_party/3DDFA_V2 -c advice.detachedHead=false checkout --quiet 1b6c67601abffc1e9f248b291708aef0e43b55ae \
    && rm -rf backend/third_party/3DDFA_V2/.git

COPY backend backend
COPY frontend frontend

EXPOSE 8599
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=10 \
    CMD curl -fs http://127.0.0.1:8599/_stcore/health || exit 1

CMD ["streamlit", "run", "frontend/app.py", "--server.address=0.0.0.0", "--server.port=8599", "--server.headless=true"]
