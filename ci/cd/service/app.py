"""偵測服務：收一張圖片，回傳邊界框與類別。"""

import io
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from PIL import Image, UnidentifiedImageError

MODEL_PATH = os.environ.get("MODEL_PATH", "model/model.onnx")
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")


def load_model():
    # 放在函式裡匯入：測試時會把這個函式換成假模型，就不需要安裝 Ultralytics。
    from ultralytics import YOLO

    return YOLO(MODEL_PATH, task="detect")


@asynccontextmanager
async def lifespan(app):
    # 模型在服務啟動時載入一次；載入完成前健康檢查不會通過，K8s 就不會把流量導進來。
    app.state.model = load_model()
    yield


app = FastAPI(title="偵測服務", lifespan=lifespan)


def to_detections(result):
    """把一張圖的推論結果轉成可以直接回傳的 list。"""
    detections = []
    for x1, y1, x2, y2, confidence, class_id in result.boxes.data.tolist():
        detections.append(
            {
                "class_id": int(class_id),
                "class_name": result.names[int(class_id)],
                "confidence": round(float(confidence), 4),
                "box": [round(float(value), 1) for value in (x1, y1, x2, y2)],
            }
        )
    return detections


@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": MODEL_VERSION}


@app.post("/detect")
def detect(
    request: Request,
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.0, le=1.0, description="信心門檻"),
):
    try:
        image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail="上傳的檔案不是可辨識的圖片") from error

    results = request.app.state.model.predict(image, conf=conf, verbose=False)

    return {
        "model_version": MODEL_VERSION,
        "width": image.width,
        "height": image.height,
        "detections": to_detections(results[0]),
    }
