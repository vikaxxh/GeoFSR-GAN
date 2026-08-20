import io
import base64
import os
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from models import GeoFSRGenerator, LightweightSegmentationUNet
from evaluation import calculate_psnr, calculate_ssim, compute_miou, tensor_to_numpy
from .schemas import HealthResponse, MetricsRequest, MetricsResponse

app = FastAPI(
    title="GeoFSR-GAN API",
    description="Production REST API for GeoFSR-GAN Satellite Super-Resolution & Downstream Segmentation",
    version="1.0.0"
)

# Enable CORS for Streamlit / React frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GENERATOR = None
SEG_HEAD = None


def load_models():
    global GENERATOR, SEG_HEAD
    if GENERATOR is None:
        GENERATOR = GeoFSRGenerator(scale=4, in_channels=3, out_channels=3, num_features=32, num_spatial_blocks=2, fusion_type="concat").to(DEVICE)
        ckpt_path = "experiments/cpu_debug/checkpoints/geofsr_generator_latest.pth"
        if os.path.exists(ckpt_path):
            GENERATOR.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            print(f"[API] Loaded generator weights from '{ckpt_path}'.")
        GENERATOR.eval()

    if SEG_HEAD is None:
        SEG_HEAD = LightweightSegmentationUNet(in_channels=3, num_classes=1, num_features=16).to(DEVICE)
        SEG_HEAD.eval()


@app.on_event("startup")
def startup_event():
    load_models()


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        device=str(DEVICE),
        model_loaded=(GENERATOR is not None),
        scale_factor=4
    )


@app.post("/super-resolve")
async def super_resolve(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file.")

    contents = await file.read()
    try:
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse image file: {str(e)}")

    lr_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(DEVICE)

    load_models()
    with torch.no_grad():
        sr_tensor = GENERATOR(lr_tensor)

    sr_np = (tensor_to_numpy(sr_tensor[0]) * 255).astype("uint8")
    sr_pil = Image.fromarray(sr_np)

    buf = io.BytesIO()
    sr_pil.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/segment")
async def segment_building_footprints(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image file.")

    contents = await file.read()
    try:
        img_pil = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse image file: {str(e)}")

    img_tensor = TF.to_tensor(img_pil).unsqueeze(0).to(DEVICE)

    load_models()
    with torch.no_grad():
        logits = SEG_HEAD(img_tensor)
        probs = torch.sigmoid(logits)
        mask = (probs > 0.5).float()

    mask_np = (tensor_to_numpy(mask[0]).squeeze(-1) * 255).astype("uint8")
    mask_pil = Image.fromarray(mask_np, mode="L")

    buf = io.BytesIO()
    mask_pil.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


@app.post("/metrics", response_model=MetricsResponse)
def compute_image_metrics(req: MetricsRequest):
    try:
        sr_bytes = base64.b64decode(req.sr_image_b64)
        hr_bytes = base64.b64decode(req.hr_image_b64)

        sr_pil = Image.open(io.BytesIO(sr_bytes)).convert("RGB")
        hr_pil = Image.open(io.BytesIO(hr_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

    sr_tensor = TF.to_tensor(sr_pil).unsqueeze(0).to(DEVICE)
    hr_tensor = TF.to_tensor(hr_pil).unsqueeze(0).to(DEVICE)

    if sr_tensor.shape != hr_tensor.shape:
        sr_tensor = torch.nn.functional.interpolate(sr_tensor, size=(hr_tensor.shape[-2], hr_tensor.shape[-1]), mode="bicubic", align_corners=False)

    load_models()
    with torch.no_grad():
        psnr_val = calculate_psnr(sr_tensor, hr_tensor)
        ssim_val = calculate_ssim(sr_tensor, hr_tensor)

        mask_sr = torch.sigmoid(SEG_HEAD(sr_tensor))
        mask_hr = torch.sigmoid(SEG_HEAD(hr_tensor))
        miou_val = compute_miou(mask_sr, mask_hr)

    return MetricsResponse(
        psnr_db=round(psnr_val, 2),
        ssim=round(ssim_val, 4),
        miou=round(miou_val, 4)
    )
