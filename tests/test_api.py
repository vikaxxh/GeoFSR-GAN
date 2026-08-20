import io
import base64
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def create_dummy_image_bytes(size=(32, 32), color=(100, 150, 200)):
    img = Image.new("RGB", size, color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "1.0.0"
    assert "device" in data
    assert "model_loaded" in data


def test_super_resolve_endpoint():
    img_bytes = create_dummy_image_bytes(size=(24, 24))
    files = {"file": ("test.png", img_bytes, "image/png")}

    response = client.post("/super-resolve", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    # Verify returned image resolution is 4x (24x4 = 96)
    out_img = Image.open(io.BytesIO(response.content))
    assert out_img.size == (96, 96)


def test_segment_endpoint():
    img_bytes = create_dummy_image_bytes(size=(48, 48))
    files = {"file": ("test.png", img_bytes, "image/png")}

    response = client.post("/segment", files=files)
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"

    out_img = Image.open(io.BytesIO(response.content))
    assert out_img.size == (48, 48)


def test_metrics_endpoint():
    sr_bytes = create_dummy_image_bytes(size=(64, 64), color=(100, 100, 100))
    hr_bytes = create_dummy_image_bytes(size=(64, 64), color=(100, 100, 100))

    payload = {
        "sr_image_b64": base64.b64encode(sr_bytes).decode("utf-8"),
        "hr_image_b64": base64.b64encode(hr_bytes).decode("utf-8")
    }

    response = client.post("/metrics", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "psnr_db" in data
    assert "ssim" in data
    assert "miou" in data
    assert data["psnr_db"] > 30.0  # Identical images should yield high PSNR
