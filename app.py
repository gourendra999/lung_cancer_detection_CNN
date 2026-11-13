from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError
import onnxruntime as ort


# -----------------------------------------------------------------------------
# App configuration and paths
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_CANDIDATES = [
    BASE_DIR / "models" / "lung_cancer_model.onnx",
    BASE_DIR / "lung_cancer_model.onnx",
]
METADATA_CANDIDATES = [
    BASE_DIR / "models" / "metadata.json",
    BASE_DIR / "metadata.json",
]
MAX_UPLOAD_MB = 15
FALLBACK_IMAGE_SIZE = (150, 150)

st.set_page_config(
    page_title="LungAI | CT Image Classification",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------------------------------------------------------
# Styling - retain the original dark medical/biotech visual direction
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
    :root {
        --bg: #080d12; --panel: #111a22; --panel2: #17232d;
        --border: #263846; --cyan: #23d7c2; --blue: #46a8ef;
        --green: #35d39a; --red: #ff637d; --amber: #ffc65c;
        --text: #e5eff5; --muted: #9aafbd;
    }
    .stApp { background: radial-gradient(circle at 80% 0%, #102631 0%, var(--bg) 46%); color: var(--text); }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    #MainMenu, footer { visibility: hidden; }
    section[data-testid="stSidebar"] { background: #0b141c; border-right: 1px solid var(--border); }
    section[data-testid="stSidebar"] * { color: var(--text); }
    .brand { padding: 1.2rem 0 1rem; border-bottom: 1px solid var(--border); margin-bottom: 1rem; }
    .brand-title { font-size: 1.55rem; font-weight: 800; color: #fff; letter-spacing: -0.8px; }
    .brand-title span { color: var(--cyan); }
    .brand-sub { color: var(--muted); font: 0.68rem 'JetBrains Mono', monospace; letter-spacing: 1.4px; text-transform: uppercase; }
    .hero { padding: 2rem 2.2rem; border: 1px solid var(--border); border-radius: 18px;
        background: linear-gradient(120deg, rgba(10,30,40,.98), rgba(12,25,38,.92));
        margin: 0.5rem 0 1.2rem; position: relative; overflow: hidden; }
    .hero:after { content: ''; position: absolute; width: 300px; height: 300px; right: -70px; top: -150px;
        background: radial-gradient(circle, rgba(35,215,194,.18), transparent 68%); }
    .hero-kicker { color: var(--cyan); letter-spacing: 2px; font: 0.7rem 'JetBrains Mono', monospace; text-transform: uppercase; }
    .hero h1 { color: #fff; font-size: 2.5rem; font-weight: 800; letter-spacing: -1.3px; margin: .5rem 0; }
    .hero h1 span { color: var(--cyan); }
    .hero p { color: #b3c5d0; max-width: 720px; margin: 0; line-height: 1.6; }
    .panel { padding: 1.1rem 1.2rem; border: 1px solid var(--border); border-radius: 14px; background: rgba(17,26,34,.88); }
    .panel-heading { color: var(--cyan); text-transform: uppercase; letter-spacing: 1.8px; font: 0.7rem 'JetBrains Mono', monospace; margin-bottom: .8rem; }
    .result { padding: 1.4rem; border-radius: 14px; border: 1px solid var(--border); background: var(--panel); margin: .5rem 0 1rem; }
    .result-normal { border-color: rgba(53,211,154,.55); background: linear-gradient(130deg, #0c241d, #111a22); }
    .result-cancer { border-color: rgba(255,99,125,.55); background: linear-gradient(130deg, #2a121a, #111a22); }
    .result-title { font-size: 1.45rem; font-weight: 800; margin: .25rem 0; }
    .result-score { font: 2.2rem 'JetBrains Mono', monospace; color: var(--cyan); }
    .muted { color: var(--muted); font-size: .86rem; }
    .footer { border-top: 1px solid var(--border); margin-top: 1.5rem; padding-top: 1rem; color: var(--muted); text-align: center; font-size: .76rem; }
    div.stButton > button { border: 0; border-radius: 10px; color: #041316; font-weight: 800;
        background: linear-gradient(100deg, #23d7c2, #46a8ef); min-height: 2.8rem; }
    div.stButton > button:hover { filter: brightness(1.08); transform: translateY(-1px); }
    [data-testid="stFileUploader"] section { background: rgba(23,35,45,.65); border: 1px dashed #456174; border-radius: 12px; }
    [data-testid="stMetricValue"] { color: var(--cyan); font-family: 'JetBrains Mono', monospace; }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    .stAlert { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Model metadata and human-readable class names
# -----------------------------------------------------------------------------
FOLDER_TO_DISPLAY = {
    "adenocarcinoma_left.lower.lobe_t2_n0_m0_ib": "Adenocarcinoma",
    "large.cell.carcinoma_left.hilum_t2_n2_m0_iiia": "Large Cell Carcinoma",
    "normal": "Normal",
    "squamous.cell.carcinoma_left.hilum_t1_n2_m0_iiia": "Squamous Cell Carcinoma",
}
DISPLAY_DETAILS = {
    "Adenocarcinoma": "The model assigned this image to the adenocarcinoma image class.",
    "Large Cell Carcinoma": "The model assigned this image to the large-cell carcinoma image class.",
    "Normal": "The model assigned this image to the normal image class.",
    "Squamous Cell Carcinoma": "The model assigned this image to the squamous-cell carcinoma image class.",
}
CLASS_COLORS = {
    "Adenocarcinoma": "#ff637d",
    "Large Cell Carcinoma": "#ff9b61",
    "Normal": "#35d39a",
    "Squamous Cell Carcinoma": "#ffc65c",
}
# This order is taken from the notebook's recorded flow_from_directory class_indices.
# The training script writes metadata.json and overrides this fallback automatically.
FALLBACK_CLASS_NAMES = [
    "adenocarcinoma_left.lower.lobe_T2_N0_M0_Ib",
    "large.cell.carcinoma_left.hilum_T2_N2_M0_IIIa",
    "normal",
    "squamous.cell.carcinoma_left.hilum_T1_N2_M0_IIIa",
]


def first_existing(paths: list[Path]) -> Path | None:
    return next((path for path in paths if path.is_file()), None)


def to_display_label(folder_name: str) -> str:
    key = str(folder_name).strip().lower()
    return FOLDER_TO_DISPLAY.get(key, str(folder_name).replace("_", " ").replace(".", " ").title())


@st.cache_resource(show_spinner="Loading the CNN inference engine...")
def load_onnx_session(model_path: str) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        model_path,
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )


def load_metadata(metadata_path: Path | None) -> tuple[dict[str, Any], bool]:
    if metadata_path is None:
        return {
            "class_names": FALLBACK_CLASS_NAMES,
            "image_size": list(FALLBACK_IMAGE_SIZE),
            "normalization": "divide_by_255",
            "interpolation": "nearest",
            "input_layout": "NHWC",
        }, False

    with metadata_path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    if not isinstance(metadata.get("class_names"), list) or not metadata["class_names"]:
        raise ValueError("metadata.json must contain a non-empty 'class_names' list.")
    metadata.setdefault("image_size", list(FALLBACK_IMAGE_SIZE))
    metadata.setdefault("normalization", "divide_by_255")
    metadata.setdefault("interpolation", "nearest")
    metadata.setdefault("input_layout", "NHWC")
    return metadata, True


def prepare_image(image_bytes: bytes, input_shape: list[Any], metadata: dict[str, Any]) -> tuple[Image.Image, np.ndarray, tuple[int, int]]:
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file is not a valid readable image. Please upload a JPG or PNG CT image.") from exc

    if image.width < 32 or image.height < 32:
        raise ValueError("The image is too small to process. Please upload a higher-resolution image.")
    if image.width * image.height > 50_000_000:
        raise ValueError("The image dimensions are unusually large. Please resize the image and try again.")

    default_h, default_w = metadata.get("image_size", FALLBACK_IMAGE_SIZE)
    is_nchw = len(input_shape) == 4 and input_shape[1] == 3 and input_shape[-1] != 3
    if is_nchw:
        height = input_shape[2] if isinstance(input_shape[2], int) else default_h
        width = input_shape[3] if isinstance(input_shape[3], int) else default_w
    else:
        height = input_shape[1] if len(input_shape) == 4 and isinstance(input_shape[1], int) else default_h
        width = input_shape[2] if len(input_shape) == 4 and isinstance(input_shape[2], int) else default_w

    interpolation_name = str(metadata.get("interpolation", "nearest")).lower()
    resampling = Image.Resampling.NEAREST if interpolation_name == "nearest" else Image.Resampling.BILINEAR
    resized = image.resize((int(width), int(height)), resample=resampling)
    array = np.asarray(resized, dtype=np.float32)

    if metadata.get("normalization") == "divide_by_255":
        array /= 255.0
    elif metadata.get("normalization") != "none":
        raise ValueError(f"Unsupported model normalization setting: {metadata.get('normalization')}")

    array = np.expand_dims(array, axis=0)
    if is_nchw:
        array = np.transpose(array, (0, 3, 1, 2))
    return image, np.ascontiguousarray(array, dtype=np.float32), (int(width), int(height))


def predict_probabilities(session: ort.InferenceSession, input_tensor: np.ndarray, number_of_classes: int) -> np.ndarray:
    model_input = session.get_inputs()[0]
    output_values = session.run(None, {model_input.name: input_tensor})
    if not output_values:
        raise RuntimeError("The ONNX model returned no output tensors.")

    scores = np.asarray(output_values[0], dtype=np.float32)
    if scores.ndim == 2 and scores.shape[0] == 1:
        scores = scores[0]
    elif scores.ndim != 1:
        raise RuntimeError(f"Unexpected model output shape: {scores.shape}")

    if scores.size != number_of_classes:
        raise RuntimeError(
            f"Model returned {scores.size} class scores, but metadata lists {number_of_classes} classes. "
            "Check that the ONNX model and metadata.json come from the same training run."
        )
    if not np.all(np.isfinite(scores)):
        raise RuntimeError("The model returned non-finite scores. Check the ONNX export.")

    # The notebook's final layer is softmax. This also tolerates an export that
    # exposes logits instead of probabilities.
    total = float(scores.sum())
    if np.any(scores < 0) or np.any(scores > 1) or not np.isclose(total, 1.0, atol=1e-3):
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted)
        scores = exp_scores / exp_scores.sum()
    else:
        scores = scores / total
    return scores.astype(np.float32)


# -----------------------------------------------------------------------------
# Load assets
# -----------------------------------------------------------------------------
model_path = first_existing(MODEL_CANDIDATES)
metadata_path = first_existing(METADATA_CANDIDATES)

st.sidebar.markdown(
    """
    <div class="brand">
      <div class="brand-title">Lung<span>AI</span></div>
      <div class="brand-sub">CT image classification · CNN / ONNX</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model_path is None:
    st.error(
        "ONNX model file not found. Place `lung_cancer_model.onnx` inside the `models/` folder. "
        "See README.md for the required deployment structure."
    )
    st.stop()

try:
    metadata, metadata_verified = load_metadata(metadata_path)
    session = load_onnx_session(str(model_path))
    input_meta = session.get_inputs()[0]
    class_names_raw = metadata["class_names"]
    class_labels = [to_display_label(name) for name in class_names_raw]
    if len(set(class_labels)) != len(class_labels):
        raise ValueError("Class metadata maps two or more folder names to the same display label.")
except Exception as exc:
    st.error(f"Unable to initialize the model: {type(exc).__name__}: {exc}")
    st.stop()

if not metadata_verified:
    st.sidebar.warning("Using fallback class order from the supplied notebook. Add models/metadata.json from the training script before publishing.")

st.sidebar.success("ONNX model loaded")
st.sidebar.caption(f"Model file: {model_path.name}")
st.sidebar.markdown("**Model classes**")
for label in class_labels:
    st.sidebar.markdown(f"- {label}")
st.sidebar.markdown("---")
st.sidebar.markdown("**How to use**")
for index, step in enumerate(
    ["Upload a lung CT image in JPG or PNG format.", "Select Analyze image.", "Review the predicted image class and model scores."],
    start=1,
):
    st.sidebar.markdown(f"**{index}.** {step}")
st.sidebar.warning("Educational demonstration only. This model is not a medical device and must not be used to diagnose, screen, or guide treatment.")


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

if st.sidebar.button("↺ Reset session", use_container_width=True):
    st.session_state.scan_history = []
    st.session_state.last_result = None
    st.rerun()


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-kicker">AI-assisted image analysis · ONNX inference</div>
      <h1>Lung CT <span>Image Classifier</span></h1>
      <p>Upload an image to view the CNN model's four-class output scores. Results are experimental image classifications, not clinical findings.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

stat_cols = st.columns(4)
stat_cols[0].metric("Image classes", len(class_labels))
stat_cols[1].metric("Architecture", "CNN")
stat_cols[2].metric("Inference", "ONNX Runtime")
stat_cols[3].metric("Session scans", len(st.session_state.scan_history))

left_col, right_col = st.columns([0.95, 1.15], gap="large")

with left_col:
    st.markdown('<div class="panel-heading">01 · CT image input</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload CT image",
        type=["jpg", "jpeg", "png"],
        help=f"Maximum file size: {MAX_UPLOAD_MB} MB. Use an image from the dataset's expected image domain.",
    )

    current_hash = None
    if uploaded_file is not None:
        image_bytes = uploaded_file.getvalue()
        current_hash = hashlib.sha256(image_bytes).hexdigest()
        if len(image_bytes) > MAX_UPLOAD_MB * 1024 * 1024:
            st.error(f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.")
        else:
            try:
                preview_image, input_tensor, resized_size = prepare_image(
                    image_bytes, input_meta.shape, metadata
                )
                st.image(preview_image, caption="Uploaded CT image", use_container_width=True)
                info_cols = st.columns(3)
                info_cols[0].metric("Width", f"{preview_image.width}px")
                info_cols[1].metric("Height", f"{preview_image.height}px")
                info_cols[2].metric("Model input", f"{resized_size[0]}×{resized_size[1]}")
                if st.button("Analyze image", use_container_width=True, type="primary"):
                    try:
                        with st.spinner("Running CNN inference..."):
                            scores = predict_probabilities(session, input_tensor, len(class_labels))
                        best_index = int(np.argmax(scores))
                        result = {
                            "image_hash": current_hash,
                            "filename": uploaded_file.name,
                            "predicted_label": class_labels[best_index],
                            "scores": scores.tolist(),
                        }
                        st.session_state.last_result = result
                        st.session_state.scan_history.append(
                            {"label": result["predicted_label"], "score": float(scores[best_index])}
                        )
                        st.rerun()
                    except Exception as exc:
                        st.error(f"Inference failed: {type(exc).__name__}: {exc}")
            except Exception as exc:
                st.error(f"Image validation failed: {type(exc).__name__}: {exc}")
    else:
        st.markdown(
            """
            <div class="panel" style="text-align:center; padding:3rem 1.2rem;">
              <div style="font-size:2.5rem;">🩻</div>
              <div style="font-size:1.05rem; font-weight:700; margin:.5rem 0;">Awaiting image</div>
              <div class="muted">Choose a JPG or PNG CT image to begin.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right_col:
    st.markdown('<div class="panel-heading">02 · Model output</div>', unsafe_allow_html=True)
    result = st.session_state.last_result
    if result is not None and current_hash is not None and result.get("image_hash") == current_hash:
        predicted_label = result["predicted_label"]
        scores = np.asarray(result["scores"], dtype=float)
        best_score = float(np.max(scores))
        is_normal = predicted_label == "Normal"
        result_class = "result-normal" if is_normal else "result-cancer"
        result_color = CLASS_COLORS.get(predicted_label, "#46a8ef")

        st.markdown(
            f"""
            <div class="result {result_class}">
              <div class="muted">Predicted image class</div>
              <div class="result-title" style="color:{result_color};">{predicted_label}</div>
              <div class="result-score">{best_score * 100:.1f}%</div>
              <div class="muted">Highest model softmax score — not a calibrated clinical probability.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.info("This output is a machine-learning class prediction only. It cannot confirm or exclude cancer. A qualified clinician must interpret medical imaging and determine any next steps.")

        plot_df = pd.DataFrame({"Class": class_labels, "Model score (%)": scores * 100})
        plot_df = plot_df.sort_values("Model score (%)", ascending=True)
        fig = go.Figure(
            go.Bar(
                x=plot_df["Model score (%)"],
                y=plot_df["Class"],
                orientation="h",
                marker_color=[CLASS_COLORS.get(label, "#46a8ef") for label in plot_df["Class"]],
                text=[f"{value:.1f}%" for value in plot_df["Model score (%)"]],
                textposition="outside",
            )
        )
        fig.update_layout(
            title="Model scores by image class",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#dce8ef"},
            xaxis={"range": [0, 110], "ticksuffix": "%", "gridcolor": "#263846"},
            yaxis={"title": "", "gridcolor": "rgba(0,0,0,0)"},
            margin={"l": 10, "r": 35, "t": 45, "b": 15},
            height=300,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        score_table = pd.DataFrame(
            {
                "Image class": class_labels,
                "Model score": [f"{score * 100:.2f}%" for score in scores],
                "Selected": ["Predicted" if label == predicted_label else "—" for label in class_labels],
            }
        )
        st.dataframe(score_table, use_container_width=True, hide_index=True)
        with st.expander("About this class output"):
            st.write(DISPLAY_DETAILS.get(predicted_label, "The model selected this image class."))
    elif uploaded_file is not None:
        st.markdown(
            '<div class="panel" style="padding:2.5rem 1.2rem; text-align:center;"><div style="font-size:2rem;">📊</div><b>Ready for analysis</b><div class="muted">Select Analyze image to run inference.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="panel" style="padding:3.5rem 1.2rem; text-align:center;"><div style="font-size:2.5rem;">📈</div><b>Results will appear here</b><div class="muted">Upload an image to view model output scores.</div></div>',
            unsafe_allow_html=True,
        )

if st.session_state.scan_history:
    with st.expander("Recent scans in this browser session"):
        history_df = pd.DataFrame(st.session_state.scan_history[-10:][::-1])
        history_df.index = np.arange(1, len(history_df) + 1)
        history_df.columns = ["Predicted class", "Highest model score"]
        history_df["Highest model score"] = history_df["Highest model score"].map(lambda value: f"{value * 100:.1f}%")
        st.dataframe(history_df, use_container_width=True)

st.markdown(
    """
    <div class="footer">
      <b>LungAI</b> · CNN image classification · ONNX Runtime<br>
      Educational research prototype only. Not validated for clinical use.
    </div>
    """,
    unsafe_allow_html=True,
)
