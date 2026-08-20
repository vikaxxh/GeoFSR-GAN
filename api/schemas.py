from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class HealthResponse(BaseModel):
    status: str = Field(..., json_schema_extra={"example": "healthy"})
    version: str = Field(..., json_schema_extra={"example": "1.0.0"})
    device: str = Field(..., json_schema_extra={"example": "cpu"})
    model_loaded: bool = Field(..., json_schema_extra={"example": True})
    scale_factor: int = Field(..., json_schema_extra={"example": 4})


class MetricsRequest(BaseModel):
    sr_image_b64: str = Field(..., description="Base64 encoded PNG/JPEG image of Super-Resolved prediction")
    hr_image_b64: str = Field(..., description="Base64 encoded PNG/JPEG image of Ground-Truth reference")


class MetricsResponse(BaseModel):
    psnr_db: float = Field(..., json_schema_extra={"example": 22.16})
    ssim: float = Field(..., json_schema_extra={"example": 0.3471})
    miou: float = Field(..., json_schema_extra={"example": 1.0000})
