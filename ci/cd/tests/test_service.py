import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from service import app as service_app


class FakeBoxes:
    def __init__(self, rows):
        self.data = np.array(rows, dtype=float).reshape(-1, 6)


class FakeResult:
    names = {0: "person", 1: "dog"}

    def __init__(self, rows):
        self.boxes = FakeBoxes(rows)


class FakeModel:
    """取代真正的 YOLO，讓 CI 不需要模型檔也不需要 PyTorch。"""

    def __init__(self, rows):
        self.rows = rows
        self.last_conf = None

    def predict(self, image, conf, verbose):
        self.last_conf = conf
        return [FakeResult(self.rows)]


def make_jpeg():
    buffer = io.BytesIO()
    Image.new("RGB", (320, 240), color=(10, 20, 30)).save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel([[10, 20, 110, 220, 0.9, 1], [5, 5, 50, 50, 0.4, 0]])
    monkeypatch.setattr(service_app, "load_model", lambda: model)
    return model


@pytest.fixture
def client(fake_model):
    with TestClient(service_app.app) as test_client:
        yield test_client


def test_healthz_reports_model_version(client):
    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": service_app.MODEL_VERSION}


def test_detect_returns_boxes_with_class_names(client):
    response = client.post("/detect", files={"file": ("a.jpg", make_jpeg(), "image/jpeg")})

    assert response.status_code == 200
    body = response.json()
    assert (body["width"], body["height"]) == (320, 240)
    assert body["detections"][0] == {
        "class_id": 1,
        "class_name": "dog",
        "confidence": 0.9,
        "box": [10.0, 20.0, 110.0, 220.0],
    }
    assert len(body["detections"]) == 2


def test_detect_passes_confidence_threshold_to_model(client, fake_model):
    client.post("/detect?conf=0.6", files={"file": ("a.jpg", make_jpeg(), "image/jpeg")})

    assert fake_model.last_conf == 0.6


def test_detect_rejects_file_that_is_not_an_image(client):
    response = client.post("/detect", files={"file": ("a.txt", b"not an image", "text/plain")})

    assert response.status_code == 400


def test_detect_rejects_out_of_range_confidence(client):
    response = client.post("/detect?conf=1.5", files={"file": ("a.jpg", make_jpeg(), "image/jpeg")})

    assert response.status_code == 422


def test_no_detections_returns_empty_list(monkeypatch):
    monkeypatch.setattr(service_app, "load_model", lambda: FakeModel([]))

    with TestClient(service_app.app) as test_client:
        response = test_client.post(
            "/detect", files={"file": ("a.jpg", make_jpeg(), "image/jpeg")}
        )

    assert response.json()["detections"] == []
