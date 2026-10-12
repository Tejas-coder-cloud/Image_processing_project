"""Interactive Streamlit Web Application for Traffic Sign Image Processing Pipeline.

Replicates and extends Stage 1 (Enhancement) and Stage 2 (Color Segmentation)
interactive workflows depicted in project slides.
"""

from pathlib import Path
from typing import List, Optional
import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from src.config import Stage1Config, Stage2Config, DatasetConfig
from src.pipeline import run_stage1, run_stage2

# Page configuration
st.set_page_config(
    page_title="Traffic Sign Vision Pipeline: Stage 1 & 2",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom modern styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94A3B8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        background-color: #1E293B;
        border-radius: 6px 6px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_dataset_samples(dataset_dir: Path, pattern: str, limit: int = 20) -> List[Path]:
    """Helper to locate sample images."""
    if not dataset_dir.exists():
        return []
    files = sorted(list(dataset_dir.glob(pattern)))
    return files[:limit]


# Sidebar Controls
st.sidebar.markdown("## 🚦 Image Input Source")

input_mode = st.sidebar.radio(
    "Select Image Source:",
    ["GTSDB Road Scene (Detection)", "GTSRB Sign Crop (Recognition)", "Upload Custom Image"],
)

root_dir = Path(__file__).resolve().parent
default_paths = DatasetConfig()
gtsdb_dir = root_dir / default_paths.gtsdb_dir if not default_paths.gtsdb_dir.is_absolute() else default_paths.gtsdb_dir
if not gtsdb_dir.exists():
    gtsdb_dir = root_dir / "gtsdb_dataset" / "train" / "images"

gtsrb_dir = root_dir / default_paths.gtsrb_dir if not default_paths.gtsrb_dir.is_absolute() else default_paths.gtsrb_dir
if not gtsrb_dir.exists():
    gtsrb_dir = root_dir / "gtsrb_dataset" / "Train"

image_bgr: Optional[np.ndarray] = None
image_name: str = "custom_image"

if input_mode == "GTSDB Road Scene (Detection)":
    # Prioritize 122_jpg, 124_jpg, 104_jpg
    samples = load_dataset_samples(gtsdb_dir, "122_jpg*.jpg") + load_dataset_samples(gtsdb_dir, "124_jpg*.jpg") + load_dataset_samples(gtsdb_dir, "*.jpg", limit=15)
    # Deduplicate while preserving order
    seen = set()
    unique_samples = []
    for s in samples:
        if s.name not in seen:
            seen.add(s.name)
            unique_samples.append(s)

    if unique_samples:
        choice = st.sidebar.selectbox("Choose GTSDB Sample:", [s.name for s in unique_samples])
        selected_path = next(s for s in unique_samples if s.name == choice)
        image_bgr = cv2.imread(str(selected_path))
        image_name = selected_path.stem
    else:
        st.sidebar.warning("No GTSDB images found. Check gtsdb_dataset path.")

elif input_mode == "GTSRB Sign Crop (Recognition)":
    gtsrb_samples: List[Path] = []
    for cls_folder in ["1", "14", "0", "13", "12"]:
        p = gtsrb_dir / cls_folder
        if p.exists():
            gtsrb_samples.extend(load_dataset_samples(p, "*.png", limit=3))

    if gtsrb_samples:
        choice = st.sidebar.selectbox("Choose GTSRB Sign Sample:", [f"Class {s.parent.name}/{s.name}" for s in gtsrb_samples])
        selected_path = next(s for s in gtsrb_samples if f"Class {s.parent.name}/{s.name}" == choice)
        image_bgr = cv2.imread(str(selected_path))
        image_name = f"gtsrb_{selected_path.parent.name}_{selected_path.stem}"
    else:
        st.sidebar.warning("No GTSRB images found. Check gtsrb_dataset path.")

else:
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "bmp"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_name = Path(uploaded_file.name).stem

if image_bgr is None:
    st.info("👈 Please select or upload an image from the sidebar to start the pipeline.")
    st.stop()

# Header
st.markdown('<div class="main-title">Traffic Sign Vision Pipeline</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Stage 1: Image Enhancement &nbsp;|&nbsp; Stage 2: Color Segmentation (HSV & CIELAB)</div>',
    unsafe_allow_html=True,
)

# Tabs
tab1, tab2, tab3 = st.tabs(["⚡ Stage 1: Image Enhancement", "🎨 Stage 2: Color Segmentation", "📊 Full Pipeline Summary"])

# ==========================================
# STAGE 1 TAB
# ==========================================
with tab1:
    st.markdown("### Stage 1: Image Enhancement Controls")
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

    with col_ctrl1:
        st.markdown("**1. Denoising Filter**")
        denoise_type = st.selectbox("Algorithm", ["Adaptive (Edge-Preserving)", "Manual Bilateral", "Non-Local Means (NLM)"])
        if denoise_type == "Manual Bilateral":
            d_val = st.slider("Filter Diameter (d)", 3, 11, 5, 2)
            sigma_c = st.slider("Color Sigma", 10.0, 80.0, 25.0, 5.0)
            sigma_s = st.slider("Space Sigma", 5.0, 40.0, 10.0, 2.0)
            nlm_h = 6.0
        elif denoise_type == "Non-Local Means (NLM)":
            nlm_h = st.slider("NLM Filter Strength (h)", 2.0, 20.0, 6.0, 1.0)
            d_val, sigma_c, sigma_s = 5, 25.0, 10.0
        else:
            d_val, sigma_c, sigma_s, nlm_h = 5, 25.0, 10.0, 6.0
            st.caption("Automatically sets gentle parameters based on estimated noise to avoid blur.")

    with col_ctrl2:
        st.markdown("**2. CLAHE (CIELAB L* Channel)**")
        clahe_clip = st.slider("Clip Limit", 1.0, 6.0, 2.5, 0.2)
        grid_dim = st.select_slider("Tile Grid Size", options=[4, 8, 12, 16], value=8)

    with col_ctrl3:
        st.markdown("**3. Exposure & Sharpening**")
        gamma_val = st.slider("Gamma Factor", 0.6, 2.0, 1.15, 0.05)
        sharpen_val = st.slider("Edge Sharpening (Unsharp Mask)", 0.0, 0.8, 0.35, 0.05)
        st.caption("Sharpening restores crisp text & sign contours without blurring.")

    # Configure Stage 1
    method_name = "nlm" if denoise_type.startswith("Non-Local") else ("bilateral" if denoise_type.startswith("Manual") else "adaptive")
    s1_cfg = Stage1Config(
        denoise_method=method_name,
        bilateral_d=d_val,
        bilateral_sigma_color=sigma_c,
        bilateral_sigma_space=sigma_s,
        nlm_h=nlm_h,
        nlm_h_color=nlm_h,
        clahe_clip_limit=clahe_clip,
        clahe_grid_size=(grid_dim, grid_dim),
        gamma=gamma_val,
        sharpen_amount=sharpen_val,
    )

    stage1_res = run_stage1(image_bgr, s1_cfg)
    raw_bgr = stage1_res["raw_image"]
    enhanced_bgr = stage1_res["enhanced_image"]
    denoised_bgr = stage1_res["denoised_image"]
    q_raw = stage1_res["quality_raw"]
    q_enh = stage1_res["quality_enhanced"]

    st.markdown("#### Quality Assessment & Noise Analysis")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Noise Std (Immerkaer)", f"{q_raw['noise_sigma']:.2f}")
    m_col2.metric("Blur Score (Laplacian)", f"{q_raw['blur_laplacian_var']:.1f} -> {q_enh['blur_laplacian_var']:.1f}")
    m_col3.metric("RMS Contrast", f"{q_raw['rms_contrast']:.1f} -> {q_enh['rms_contrast']:.1f}")
    m_col4.metric("Dynamic Range", f"{q_raw['dynamic_range']} -> {q_enh['dynamic_range']}")

    disp_col1, disp_col2, disp_col3 = st.columns(3)
    with disp_col1:
        st.image(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB), caption="1. Input Road Image", use_container_width=True)
    with disp_col2:
        st.image(cv2.cvtColor(denoised_bgr, cv2.COLOR_BGR2RGB), caption="2. Edge-Preserving Denoised", use_container_width=True)
    with disp_col3:
        st.image(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB), caption="3. Stage 1 Enhanced (CLAHE+Sharpening)", use_container_width=True)

    p_col1, p_col2 = st.columns(2)
    with p_col1:
        fig_ps, ax_ps = plt.subplots(figsize=(6, 3))
        ax_ps.plot(stage1_res["radial_frequencies"], stage1_res["radial_power"], color="#38BDF8", lw=2)
        ax_ps.fill_between(stage1_res["radial_frequencies"], stage1_res["radial_power"], color="#38BDF8", alpha=0.25)
        ax_ps.set_title("Power Spectrum Analysis (Frequency Domain)", fontsize=10)
        ax_ps.set_xlabel("Normalized Spatial Frequency")
        ax_ps.set_ylabel("Radial Power")
        ax_ps.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig_ps)
        plt.close(fig_ps)

    with p_col2:
        fig_hist, ax_h = plt.subplots(figsize=(6, 3))
        h_before = stage1_res["hist_raw"]["gray"]
        h_after = stage1_res["hist_enhanced"]["gray"]
        ax_h.plot(h_before, label="Before CLAHE", color="#94A3B8", lw=1.5, linestyle="--")
        ax_h.plot(h_after, label="After CLAHE", color="#38BDF8", lw=2.0)
        ax_h.set_title("Luminance Histogram Equalization", fontsize=10)
        ax_h.set_xlim([0, 255])
        ax_h.grid(True, linestyle="--", alpha=0.4)
        ax_h.legend(loc="upper right")
        st.pyplot(fig_hist)
        plt.close(fig_hist)

# ==========================================
# STAGE 2 TAB
# ==========================================
with tab2:
    st.markdown("### Stage 2: Color Segmentation (HSV & CIELAB)")

    with st.expander("🛠️ Interactive Threshold & Vehicle Rejection Sliders", expanded=True):
        t_col1, t_col2 = st.columns(2)

        with t_col1:
            st.markdown("**HSV Saturated Color Controls**")
            red_s_min = st.slider("Red Saturation Min (excludes asphalt/cars)", 40, 150, 65)
            red_h_band1 = st.slider("Red Hue Band 1 (Lower)", 0, 20, (0, 12))
            red_h_band2 = st.slider("Red Hue Band 2 (Upper Wrap)", 150, 180, (155, 180))
            blue_h = st.slider("Blue Hue Range", 80, 150, (95, 135))
            yellow_h = st.slider("Yellow Hue Range", 10, 50, (14, 38))

        with t_col2:
            st.markdown("**CIELAB & Vehicle Rejection Filters**")
            lab_a_min = st.slider("Lab a* Min (Red Chrominance)", 130, 160, 138)
            reject_vehicles = st.checkbox("Enable Traffic Sign Verification (Rejects vehicles & brake lights)", value=True)
            min_sign_dim = st.slider("Min Box Dimension (px)", 6, 30, 10)
            road_lane_cutoff = st.slider("Road Lane Y-Horizon Cutoff (%)", 70, 95, 86)

    # Configure Stage 2
    s2_cfg = Stage2Config(
        hsv_red_lower1=(red_h_band1[0], red_s_min, 40),
        hsv_red_upper1=(red_h_band1[1], 255, 255),
        hsv_red_lower2=(red_h_band2[0], red_s_min, 40),
        hsv_red_upper2=(red_h_band2[1], 255, 255),
        hsv_blue_lower=(blue_h[0], 65, 40),
        hsv_blue_upper=(blue_h[1], 255, 255),
        hsv_yellow_lower=(yellow_h[0], 70, 50),
        hsv_yellow_upper=(yellow_h[1], 255, 255),
        lab_red_lower=(20, lab_a_min, 105),
        min_box_dim=min_sign_dim,
        max_road_y_ratio=road_lane_cutoff / 100.0,
        require_sign_structure=reject_vehicles,
    )

    stage2_res = run_stage2(enhanced_bgr, s2_cfg)

    st.markdown("#### Color Channel Planes (HSV & CIELAB)")
    ch_col1, ch_col2, ch_col3, ch_col4, ch_col5, ch_col6 = st.columns(6)
    channels = stage2_res["channels"]
    ch_col1.image(channels["H"], caption="Hue (H)", use_container_width=True)
    ch_col2.image(channels["S"], caption="Saturation (S)", use_container_width=True)
    ch_col3.image(channels["V"], caption="Value (V)", use_container_width=True)
    ch_col4.image(channels["L"], caption="Lightness (L*)", use_container_width=True)
    ch_col5.image(channels["a"], caption="Red-Green (a*)", use_container_width=True)
    ch_col6.image(channels["b"], caption="Yellow-Blue (b*)", use_container_width=True)

    st.markdown("#### Segmentation Masks & Candidate Localizations")
    mask_c1, mask_c2, mask_c3, mask_c4 = st.columns(4)
    mask_c1.image(stage2_res["cleaned_red"], caption="Red Signs Mask", use_container_width=True)
    mask_c2.image(stage2_res["cleaned_blue"], caption="Blue Signs Mask", use_container_width=True)
    mask_c3.image(stage2_res["cleaned_yellow"], caption="Yellow Signs Mask", use_container_width=True)
    mask_c4.image(stage2_res["binary_mask_cleaned"], caption="Final Unified Binary Mask", use_container_width=True)

    st.image(
        cv2.cvtColor(stage2_res["overlay"], cv2.COLOR_BGR2RGB),
        caption=f"Candidate Traffic Signs Localized ({len(stage2_res['bounding_boxes'])} sign candidates)",
        use_container_width=True,
    )

    st.markdown("#### Export Results")
    down_c1, down_c2, down_c3 = st.columns(3)
    _, enh_png = cv2.imencode(".png", stage1_res["enhanced_image"])
    _, mask_png = cv2.imencode(".png", stage2_res["binary_mask_cleaned"])
    _, overlay_png = cv2.imencode(".png", stage2_res["overlay"])

    down_c1.download_button(
        "💾 Download Enhanced Image",
        data=enh_png.tobytes(),
        file_name=f"{image_name}_stage1_enhanced.png",
        mime="image/png",
    )
    down_c2.download_button(
        "💾 Download Binary Mask",
        data=mask_png.tobytes(),
        file_name=f"{image_name}_stage2_mask.png",
        mime="image/png",
    )
    down_c3.download_button(
        "💾 Download Overlay Image",
        data=overlay_png.tobytes(),
        file_name=f"{image_name}_detected_overlay.png",
        mime="image/png",
    )

# ==========================================
# STAGE 3: FULL SUMMARY TAB
# ==========================================
with tab3:
    st.markdown("### End-to-End Pipeline Progression")
    summary_c1, summary_c2, summary_c3, summary_c4 = st.columns(4)

    summary_c1.image(cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB), caption="1. Raw Input Image", use_container_width=True)
    summary_c2.image(cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB), caption="2. Stage 1 Enhanced", use_container_width=True)
    summary_c3.image(stage2_res["binary_mask_cleaned"], caption="3. Stage 2 Binary Mask", use_container_width=True)
    summary_c4.image(cv2.cvtColor(stage2_res["overlay"], cv2.COLOR_BGR2RGB), caption="4. Final Candidate Detections", use_container_width=True)

    st.success(
        f"Pipeline executed successfully! Localized {len(stage2_res['bounding_boxes'])} candidate traffic sign regions (vehicle interference eliminated)."
    )
