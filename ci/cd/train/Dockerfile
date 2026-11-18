# 訓練用的 image：官方 Ultralytics image 已內含 CUDA 版 PyTorch。
# 實際使用時請把 latest 換成固定的版本標籤，訓練結果才能重現。
FROM ultralytics/ultralytics:latest

COPY train/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Jenkins 會用非 root 身分執行 container，這些套件的設定檔改寫到一定可寫的位置。
ENV YOLO_CONFIG_DIR=/tmp/ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib \
    HOME=/tmp
