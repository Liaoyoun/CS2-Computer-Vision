"""冒煙測試：在偵測服務的 container 裡執行，確認這個 image 真的能回應。

部署流程在推送 image 之前會跑：

    docker exec <container> python -m service.smoke_test

只用標準函式庫加 Pillow（image 裡本來就有），不需要另外安裝東西。
"""

import io
import json
import os
import sys
import time
import urllib.error
import urllib.request

from PIL import Image

BASE_URL = os.environ.get("SMOKE_URL", "http://127.0.0.1:8000")
EXPECTED_VERSION = os.environ.get("MODEL_VERSION", "unknown")
WAIT_SECONDS = 90


def wait_until_healthy():
    deadline = time.monotonic() + WAIT_SECONDS
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE_URL}/healthz", timeout=3) as response:
                return json.load(response)
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(2)
    raise SystemExit(f"偵測服務在 {WAIT_SECONDS} 秒內沒有通過健康檢查")


def post_sample_image():
    buffer = io.BytesIO()
    Image.new("RGB", (640, 640), color=(114, 114, 114)).save(buffer, format="JPEG")

    boundary = "smoke-test-boundary"
    body = (
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="smoke.jpg"\r\n'
            "Content-Type: image/jpeg\r\n\r\n"
        ).encode()
        + buffer.getvalue()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    request = urllib.request.Request(
        f"{BASE_URL}/detect",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    health = wait_until_healthy()
    if health.get("model_version") != EXPECTED_VERSION:
        sys.exit(f"模型版本不符：預期 {EXPECTED_VERSION}，實際 {health.get('model_version')}")

    # 灰色圖片上偵測不到東西是正常的；這裡只確認模型載得起來、推論跑得完、回傳格式正確。
    payload = post_sample_image()
    if not isinstance(payload.get("detections"), list):
        sys.exit(f"回傳格式不正確：{payload}")

    print(f"冒煙測試通過，模型版本 {health['model_version']}")


if __name__ == "__main__":
    main()
