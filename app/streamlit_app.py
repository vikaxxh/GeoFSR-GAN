import os
import sys
import glob
import torch
import numpy as np
import streamlit as st
from PIL import Image
import torchvision.transforms.functional as TF

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models import BicubicBaseline, SimpleSpatialSR, GeoFSRGenerator, LightweightSegmentationUNet, DWT2D
from losses.sobel_loss import SobelEdgeFilter
from evaluation import calculate_psnr, calculate_ssim, compute_miou, tensor_to_numpy


# Page Config
st.set_page_config(
    page_title="GeoSR | Spatial-Frequency Satellite Super-Resolution",
    page_icon="🛰️",
    layout="wide"
)

# Custom CSS Theme
st.markdown("""
    <style>
    .main {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    .stMetric {
        background: rgba(30, 41, 59, 0.7);
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #334155;
    }
    .css-1r6slb0, .css-12oz5g7 {
        border-radius: 12px;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_models():
    device = torch.device("cpu")
    bicubic = BicubicBaseline(scale=4).to(device).eval()
    spatial_sr = SimpleSpatialSR(scale=4, num_features=32).to(device).eval()

    geofsr = GeoFSRGenerator(scale=4, in_channels=3, out_channels=3, num_features=32, num_spatial_blocks=2, fusion_type="concat").to(device).eval()
    ckpt_path = "experiments/cpu_debug/checkpoints/geofsr_generator_latest.pth"
    if os.path.exists(ckpt_path):
        geofsr.load_state_dict(torch.load(ckpt_path, map_location=device))

    seg_head = LightweightSegmentationUNet(in_channels=3, num_classes=1, num_features=16).to(device).eval()
    dwt = DWT2D(in_channels=3).to(device)
    sobel = SobelEdgeFilter().to(device)

    return bicubic, spatial_sr, geofsr, seg_head, dwt, sobel


bicubic_model, spatial_sr_model, geofsr_model, seg_head, dwt_module, sobel_filter = load_all_models()


# Sidebar Controls
st.sidebar.title("🛰️ GeoSR Workspace")
st.sidebar.subheader("Input Selection")

sample_images = sorted(glob.glob("data/sample_dataset/*.png") + glob.glob("data/sample_dataset/*.jpg"))
selection_mode = st.sidebar.radio("Image Source", ["Preset Sample", "Upload Image"])

input_pil = None
if selection_mode == "Preset Sample" and sample_images:
    selected_sample = st.sidebar.selectbox("Choose Satellite Imagery", sample_images, format_func=lambda x: os.path.basename(x))
    input_pil = Image.open(selected_sample).convert("RGB")
else:
    uploaded_file = st.sidebar.file_uploader("Upload Low-Res Satellite Patch", type=["png", "jpg", "tif"])
    if uploaded_file is not None:
        input_pil = Image.open(uploaded_file).convert("RGB")

if input_pil is None:
    st.info("👈 Please select or upload a satellite image from the sidebar to begin!")
    st.stop()

# Ensure image dimensions are multiples of 16 for DWT and 4x scale alignment
w, h = input_pil.size
w_crop = (w // 16) * 16
h_crop = (h // 16) * 16
if w_crop > 0 and h_crop > 0 and (w != w_crop or h != h_crop):
    input_pil = input_pil.crop((0, 0, w_crop, h_crop))

# Prepare Tensors
hr_tensor = TF.to_tensor(input_pil).unsqueeze(0)
# Create synthetic LR image by downsampling
lr_tensor = torch.nn.functional.interpolate(hr_tensor, scale_factor=0.25, mode="bicubic", align_corners=False)

# Header
st.title("🌐 GeoFSR-GAN: Dual-Domain Satellite Super-Resolution & GIS Analytics")
st.caption("High-Frequency Sub-Band Synthesis | Differentiable Sobel Guidance | Downstream Land-Cover Segmentation")

# Inference
with torch.no_grad():
    sr_bicubic = bicubic_model(lr_tensor)
    sr_spatial = spatial_sr_model(lr_tensor)
    sr_geofsr = geofsr_model(lr_tensor)

    # Metrics vs HR Ground-Truth
    psnr_bicubic = calculate_psnr(sr_bicubic, hr_tensor)
    ssim_bicubic = calculate_ssim(sr_bicubic, hr_tensor)

    psnr_spatial = calculate_psnr(sr_spatial, hr_tensor)
    ssim_spatial = calculate_ssim(sr_spatial, hr_tensor)

    psnr_geofsr = calculate_psnr(sr_geofsr, hr_tensor)
    ssim_geofsr = calculate_ssim(sr_geofsr, hr_tensor)

    # Segmentation
    mask_hr = torch.sigmoid(seg_head(hr_tensor))
    mask_geofsr = torch.sigmoid(seg_head(sr_geofsr))
    miou_score = compute_miou(mask_geofsr, mask_hr)

# Metrics Summary Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Bicubic Baseline PSNR", f"{psnr_bicubic:.2f} dB", f"SSIM: {ssim_bicubic:.4f}")
with col2:
    st.metric("Spatial SR PSNR", f"{psnr_spatial:.2f} dB", f"SSIM: {ssim_spatial:.4f}")
with col3:
    st.metric("GeoFSR-GAN PSNR", f"{psnr_geofsr:.2f} dB", f"+{psnr_geofsr - psnr_bicubic:.2f} dB vs Bicubic")
with col4:
    st.metric("Downstream Seg mIoU", f"{miou_score:.4f}", "Building Footprint Match")

st.markdown("---")

# Workspace Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🖼️ Visual Comparison Grid",
    "🌊 DWT Frequency Sub-Bands",
    "📐 Sobel Edge Diagnostics",
    "🏘️ Downstream GIS Segmentation"
])

# Tab 1: Visual Comparison Grid
with tab1:
    st.subheader("Interactive 4x Super-Resolution Side-by-Side Inspection")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(tensor_to_numpy(lr_tensor[0]), caption="LR Input (Downscaled 4x)", use_container_width=True)
    with c2:
        st.image(tensor_to_numpy(sr_bicubic[0]), caption=f"Bicubic (x4)\nPSNR: {psnr_bicubic:.2f} dB", use_container_width=True)
    with c3:
        st.image(tensor_to_numpy(sr_geofsr[0]), caption=f"GeoFSR-GAN (x4)\nPSNR: {psnr_geofsr:.2f} dB", use_container_width=True)
    with c4:
        st.image(tensor_to_numpy(hr_tensor[0]), caption="Ground-Truth HR Target", use_container_width=True)

# Tab 2: DWT Wavelet Sub-Bands
with tab2:
    st.subheader("2D Discrete Wavelet Transform (Haar) Sub-band Decomposition")
    with torch.no_grad():
        ll, lh, hl, hh = dwt_module(sr_geofsr)

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.image(tensor_to_numpy(ll[0]), caption="LL (Low-Pass Approximation)", use_container_width=True)
    with f2:
        st.image(tensor_to_numpy(lh[0]), caption="LH (Horizontal Detail)", use_container_width=True)
    with f3:
        st.image(tensor_to_numpy(hl[0]), caption="HL (Vertical Detail)", use_container_width=True)
    with f4:
        st.image(tensor_to_numpy(hh[0]), caption="HH (Diagonal Detail)", use_container_width=True)

# Tab 3: Sobel Edge Diagnostics
with tab3:
    st.subheader("Differentiable Sobel Edge Map Comparison")
    with torch.no_grad():
        edge_lr = sobel_filter(torch.nn.functional.interpolate(lr_tensor, size=(hr_tensor.shape[-2], hr_tensor.shape[-1]), mode="bicubic"))
        edge_bicubic = sobel_filter(sr_bicubic)
        edge_geofsr = sobel_filter(sr_geofsr)
        edge_hr = sobel_filter(hr_tensor)

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        st.image(tensor_to_numpy(edge_lr[0]), caption="Edge: LR (Bicubic Up)", use_container_width=True)
    with e2:
        st.image(tensor_to_numpy(edge_bicubic[0]), caption="Edge: Bicubic Baseline", use_container_width=True)
    with e3:
        st.image(tensor_to_numpy(edge_geofsr[0]), caption="Edge: GeoFSR-GAN", use_container_width=True)
    with e4:
        st.image(tensor_to_numpy(edge_hr[0]), caption="Edge: Ground-Truth HR", use_container_width=True)

# Tab 4: Downstream Building Footprint Segmentation
with tab4:
    st.subheader("Downstream Land-Cover Building Footprint Prediction Overlay")
    m1, m2, m3 = st.columns(3)
    with m1:
        st.image(tensor_to_numpy(sr_geofsr[0]), caption="GeoFSR-GAN Output", use_container_width=True)
    with m2:
        mask_np = tensor_to_numpy(mask_geofsr[0]).squeeze(-1)
        st.image(mask_np, caption=f"Predicted Mask (mIoU: {miou_score:.4f})", use_container_width=True)
    with m3:
        mask_hr_np = tensor_to_numpy(mask_hr[0]).squeeze(-1)
        st.image(mask_hr_np, caption="Ground-Truth Mask", use_container_width=True)
