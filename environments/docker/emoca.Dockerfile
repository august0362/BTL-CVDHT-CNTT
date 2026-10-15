# Dịch vụ EMOCA thật (GPU NVIDIA): cùng các bước với backend/services/emoca_svc/wsl/setup_emoca_wsl.sh (ADR-018),
# chuyển sang image. Build từ gốc dự án:  docker compose build emoca   (~30-60 phút, image ~15 GB).
# FLAME/EMOCA KHÔNG nằm trong image: entrypoint tải vào volume `emoca_assets` khi ACCEPT_FLAME_EMOCA_LICENSES=1.
# torch 1.11 + CUDA 11.5 (conda) => GPU kiến trúc tới sm_86 (RTX 20xx/30xx/40xx); RTX 50xx chưa hỗ trợ.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CONDA_DIR=/opt/conda \
    EMOCA_DIR=/opt/emoca \
    EMOCA_COMMIT=e0be0db
ENV PATH=$CONDA_DIR/bin:$PATH

RUN apt-get update \
    && apt-get install -y --no-install-recommends wget curl git git-lfs unzip ca-certificates build-essential \
       python3 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Miniconda + kênh như setup_emoca_wsl.sh (chang conda_channels).
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh \
    && bash /tmp/mc.sh -b -p $CONDA_DIR && rm /tmp/mc.sh \
    && conda config --add channels defaults && conda config --add channels conda-forge \
    && (conda tos accept --channel https://repo.anaconda.com/pkgs/main \
        && conda tos accept --channel https://repo.anaconda.com/pkgs/r || true) \
    && conda config --set channel_priority flexible

# EMOCA upstream ghim commit + patch của dự án (không sửa upstream tại chỗ — bất biến §1.3).
RUN GIT_LFS_SKIP_SMUDGE=1 git -c core.autocrlf=false clone https://github.com/radekd91/emoca.git $EMOCA_DIR \
    && git -C $EMOCA_DIR -c advice.detachedHead=false checkout $EMOCA_COMMIT \
    && cd $EMOCA_DIR && (GIT_LFS_SKIP_SMUDGE=1 bash pull_submodules.sh || true) \
    && for s in external/emonet external/SwinTransformer external/face-parsing.PyTorch; do \
         git -C "$s" rev-parse --verify -q HEAD >/dev/null || { echo "submodule $s hỏng"; exit 1; }; done
COPY backend/services/emoca_svc/patches /tmp/patches
RUN sed -i 's/\r$//' /tmp/patches/*.sh /tmp/patches/*.patch && bash /tmp/patches/apply.sh $EMOCA_DIR

# Env conda work38 (khoá đúng file đã dựng được trên WSL2/Colab).
COPY environments/emoca_wsl/environment_work38.yml /tmp/env.yml
RUN sed -i -e 's/\r$//' -e '/^prefix:/d' /tmp/env.yml \
    && conda create -y -n work38 python=3.8 \
    && conda env update -n work38 --file /tmp/env.yml --prune \
    && conda clean -afy
ENV PATH=$CONDA_DIR/envs/work38/bin:$PATH

# Các chang pip của setup_emoca_wsl.sh, cùng thứ tự và phiên bản.
RUN cd $EMOCA_DIR \
    && pip install --no-cache-dir --upgrade pip==24.0 \
    && sed -E 's/^(onnx|youtube-dl|mediapipe|mmcv-full|pyzmq)/# \1/' requirements38.txt > /tmp/req38.txt \
    && pip install --no-cache-dir -r /tmp/req38.txt \
    && pip install --no-cache-dir Cython==0.29 \
    && pip install --no-cache-dir onnx==1.10.2 onnxruntime-gpu 'youtube-dl~=2021.6.6' mediapipe pyzmq==26.2.1 \
    && pip install --no-cache-dir mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu115/torch1.11/index.html \
    && pip install --no-cache-dir -e . \
    && pip uninstall -y opencv-python opencv-contrib-python opencv-contrib-python-headless \
    && pip install --no-cache-dir opencv-python-headless==4.5.5.64 \
    && pip install --no-cache-dir git+https://github.com/mattloper/chumpy \
    && pip install --no-cache-dir tensorboard==2.9.1 \
    && pip install --no-cache-dir --no-deps mediapipe==0.10.11 \
    && python -c "import torch, pytorch3d, face_alignment, gdl; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Mã dịch vụ giữ đúng bố cục dự án: server.py tính ROOT = backend/.
WORKDIR /app
COPY backend/contracts backend/contracts
COPY backend/configs backend/configs
COPY backend/services/emoca_svc backend/services/emoca_svc
COPY environments/docker/emoca_entrypoint.sh /usr/local/bin/emoca_entrypoint.sh
RUN sed -i 's/\r$//' /usr/local/bin/emoca_entrypoint.sh && chmod +x /usr/local/bin/emoca_entrypoint.sh

HEALTHCHECK --interval=15s --timeout=5s --start-period=120s --retries=20 \
    CMD curl -fs http://127.0.0.1:8765/v1/health || exit 1

ENTRYPOINT ["/usr/local/bin/emoca_entrypoint.sh"]
