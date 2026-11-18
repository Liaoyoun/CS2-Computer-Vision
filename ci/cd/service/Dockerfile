# 偵測服務的 image。由部署流程建置，建置前 model/model.onnx 必須已經從 MLflow 下載好。
#
#   docker build -f service/Dockerfile --build-arg MODEL_VERSION=3 -t detection-service:v3 .
FROM python:3.12-slim

# OpenCV（Ultralytics 的相依套件）需要的系統函式庫。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 正式環境用 CPU 推論，先裝 CPU 版 PyTorch，image 會比預設的 CUDA 版小很多。
COPY service/requirements.txt ./service/requirements.txt
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r service/requirements.txt

COPY service/ ./service/

# 模型直接打包進 image：服務啟動時不必連 MLflow，回滾就是換回舊的 image。
COPY model/model.onnx ./model/model.onnx

ARG MODEL_VERSION=unknown
ENV MODEL_VERSION=${MODEL_VERSION} \
    MODEL_PATH=/app/model/model.onnx \
    YOLO_CONFIG_DIR=/tmp/ultralytics

RUN useradd --create-home --uid 10001 appuser
USER 10001

EXPOSE 8000
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
