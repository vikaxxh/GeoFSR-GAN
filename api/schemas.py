from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    version: str = Field(..., example="1.0.0")
    device: str = Field(..., example="cpu")
    model_loaded: bool = Field(..., example=True)
    scale_factor: int = Field(..., example=4)


class MetricsRequest(BaseModel):
    sr_image_b64: str = Field(..., description="Base64 encoded PNG/JPEG image of Super-Resolved prediction")
    hr_image_b64: str = Field(..., description="Base64 encoded PNG/JPEG image of Ground-Truth reference")


class MetricsResponse(BaseModel):
    psnr_db: float = Field(..., example=22.16)
    ssim: float = Field(..., example=0.3471)
    miou: float = Field(..., example=1.0000)
