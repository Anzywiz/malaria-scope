"""
MalariaScope — Streamlit-only edition.

Runs YOLO inference directly inside the Streamlit process (no FastAPI
backend, no HTTP calls). Point MODEL_PATH (or the settings field in the
app) at your trained weights file.

Install once:
    pip install streamlit ultralytics opencv-python-headless pillow numpy pandas

Run with:
    streamlit run app.py
"""
import base64
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

st.set_page_config(
    page_title="MalariaScope",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ----------------------------------------------------------------------------
# Model config — update this default to wherever your weights live
# ----------------------------------------------------------------------------
DEFAULT_MODEL_PATH = "best.pt"

CLASS_STYLE = {
    "unparasitized": {"color": (255, 0, 0), "font_scale": 0.6, "thickness": 2},
    "parasitized":   {"color": (0, 200, 200), "font_scale": 0.6, "thickness": 2},
}

# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "dark"          # default = black, per spec
if "model_path" not in st.session_state:
    st.session_state.model_path = DEFAULT_MODEL_PATH
if "conf_threshold" not in st.session_state:
    st.session_state.conf_threshold = 0.25

# ----------------------------------------------------------------------------
# Theme tokens
# ----------------------------------------------------------------------------
THEMES = {
    "dark": {
        "bg": "#0a0d0e",
        "bg_elevated": "#12171a",
        "surface": "#151b1e",
        "border": "#242e31",
        "border_soft": "#1b2224",
        "text": "#e7edee",
        "text_muted": "#8fa1a4",
        "text_faint": "#57696c",
        "accent": "#35d0c1",
        "accent_soft": "rgba(53, 208, 193, 0.12)",
        "shadow": "0 1px 2px rgba(0,0,0,0.4)",
        "sev_low": "#4ade80",
        "sev_elevated": "#fbbf24",
        "sev_high": "#fb923c",
        "sev_vhigh": "#f87171",
        "alert_bg": "#1c1712",
        "alert_border": "#4a3418",
        "alert_text": "#e8b876",
        "metric_label": "#aebcbe",
    },
    "light": {
        "bg": "#f5f7f7",
        "bg_elevated": "#ffffff",
        "surface": "#ffffff",
        "border": "#e1e6e7",
        "border_soft": "#edf1f1",
        "text": "#12191b",
        "text_muted": "#5c6c6f",
        "text_faint": "#8fa1a4",
        "accent": "#0e988a",
        "accent_soft": "rgba(14, 152, 138, 0.10)",
        "shadow": "0 1px 2px rgba(20,30,32,0.06)",
        "sev_low": "#15803d",
        "sev_elevated": "#b45309",
        "sev_high": "#c2410c",
        "sev_vhigh": "#b91c1c",
        "alert_bg": "#fdf5ea",
        "alert_border": "#f0d5a8",
        "alert_text": "#7c4a12",
        "metric_label": "#5c6c6f",
    },
}


def inject_css(theme_name: str) -> None:
    t = THEMES[theme_name]
    st.markdown(
        f"""
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            :root {{
                --bg: {t['bg']};
                --bg-elevated: {t['bg_elevated']};
                --surface: {t['surface']};
                --border: {t['border']};
                --border-soft: {t['border_soft']};
                --text: {t['text']};
                --text-muted: {t['text_muted']};
                --text-faint: {t['text_faint']};
                --accent: {t['accent']};
                --accent-soft: {t['accent_soft']};
            }}

            html, body, .stApp {{
                background-color: var(--bg) !important;
                color: var(--text);
                font-family: 'IBM Plex Sans', sans-serif;
            }}

            #MainMenu, footer, header[data-testid="stHeader"] {{
                background: transparent;
            }}
            header[data-testid="stHeader"] {{ background-color: var(--bg) !important; }}

            .main .block-container {{
                padding-top: 1.75rem;
                padding-bottom: 3rem;
                max-width: 980px;
            }}

            /* ---- nameplate / header ---- */
            .ms-eyebrow {{
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.72rem;
                letter-spacing: 0.16em;
                color: var(--accent);
                text-transform: uppercase;
                margin-bottom: 0.3rem;
            }}
            .ms-title {{
                font-size: 1.9rem;
                font-weight: 600;
                color: var(--text);
                margin: 0;
                line-height: 1.2;
            }}
            .ms-subtitle {{
                color: var(--text-muted);
                font-size: 0.92rem;
                margin-top: 0.35rem;
            }}
            .ms-scanline {{
                height: 1px;
                width: 100%;
                margin: 1.1rem 0 1.5rem 0;
                background: linear-gradient(90deg, var(--accent) 0%, var(--border) 35%, var(--border) 100%);
                opacity: 0.6;
            }}

            /* ---- theme toggle ---- */
            div[data-testid="stButton"] button {{
                background-color: var(--surface);
                border: 1px solid var(--border);
                color: var(--text-muted);
                border-radius: 8px;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.78rem;
                padding: 0.35rem 0.8rem;
            }}
            div[data-testid="stButton"] button:hover {{
                border-color: var(--accent);
                color: var(--accent);
            }}
            div[data-testid="stButton"] button p {{
                font-family: 'IBM Plex Mono', monospace !important;
                font-size: 0.78rem !important;
            }}

            /* primary run button stands apart */
            div[data-testid="stButton"] button[kind="primary"] {{
                background-color: var(--accent);
                border: 1px solid var(--accent);
                color: #06110f;
                font-weight: 600;
            }}
            div[data-testid="stButton"] button[kind="primary"]:hover {{
                filter: brightness(1.08);
                color: #06110f;
            }}
            div[data-testid="stButton"] button[kind="primary"] p {{ color: #06110f !important; }}

            /* ---- download button (separate testid from stButton) ---- */
            div[data-testid="stDownloadButton"] button {{
                background-color: var(--surface) !important;
                border: 1px solid var(--border) !important;
                color: var(--text) !important;
                border-radius: 8px;
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.78rem;
            }}
            div[data-testid="stDownloadButton"] button p,
            div[data-testid="stDownloadButton"] button span {{
                color: var(--text) !important;
            }}
            div[data-testid="stDownloadButton"] button:hover {{
                border-color: var(--accent) !important;
                color: var(--accent) !important;
            }}
            div[data-testid="stDownloadButton"] button:hover p,
            div[data-testid="stDownloadButton"] button:hover span {{
                color: var(--accent) !important;
            }}

            /* ---- cards / expanders ---- */
            div[data-testid="stExpander"] {{
                background-color: var(--surface);
                border: 1px solid var(--border);
                border-radius: 10px;
            }}
            div[data-testid="stExpander"] summary,
            div[data-testid="stExpander"] details > summary {{
                background-color: var(--surface) !important;
                color: var(--text) !important;
            }}
            div[data-testid="stExpander"] summary:hover,
            div[data-testid="stExpander"] summary:focus,
            div[data-testid="stExpander"] details > summary:hover {{
                background-color: var(--bg-elevated) !important;
                color: var(--accent) !important;
            }}
            div[data-testid="stExpander"] summary span,
            div[data-testid="stExpander"] summary p {{
                color: inherit !important;
            }}
            div[data-testid="stExpander"] summary svg {{
                fill: var(--text-muted) !important;
            }}
            div[data-testid="stExpander"] summary:hover svg {{
                fill: var(--accent) !important;
            }}
            div[data-testid="stExpander"] summary {{
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.85rem;
            }}

            /* ---- metrics ---- */
            div[data-testid="stMetric"] {{
                background-color: var(--surface);
                border: 1px solid var(--border);
                border-radius: 10px;
                padding: 0.85rem 1rem 0.7rem 1rem;
            }}
            div[data-testid="stMetricLabel"],
            div[data-testid="stMetricLabel"] * {{
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.72rem;
                letter-spacing: 0.04em;
                color: {t['metric_label']} !important;
                text-transform: uppercase;
                opacity: 1 !important;
            }}
            div[data-testid="stMetricValue"] {{
                font-family: 'IBM Plex Mono', monospace;
                color: var(--text) !important;
                opacity: 1 !important;
            }}

            /* ---- severity badge ---- */
            .severity-badge {{
                display: inline-block;
                padding: 0.3rem 0.85rem;
                border-radius: 999px;
                font-weight: 600;
                font-size: 0.85rem;
                font-family: 'IBM Plex Mono', monospace;
                margin-bottom: 0.4rem;
                border: 1px solid transparent;
            }}
            .ms-caption {{
                color: var(--text-muted);
                font-size: 0.85rem;
            }}
            .ms-reliability {{
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.78rem;
                color: var(--text-faint);
                margin-top: 0.15rem;
            }}

            /* ---- section labels ---- */
            .ms-section-label {{
                font-family: 'IBM Plex Mono', monospace;
                font-size: 0.75rem;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                color: var(--text-faint);
                margin: 1.6rem 0 0.6rem 0;
                border-bottom: 1px solid var(--border-soft);
                padding-bottom: 0.4rem;
            }}

            /* ---- disclaimer ---- */
            .ms-disclaimer {{
                background-color: {t['alert_bg']};
                border: 1px solid {t['alert_border']};
                color: {t['alert_text']};
                border-radius: 10px;
                padding: 0.85rem 1.05rem;
                font-size: 0.83rem;
                margin-top: 1.1rem;
            }}

            /* ---- inputs ---- */
            div[data-testid="stTextInput"] input {{
                background-color: var(--surface) !important;
                border: 1px solid var(--border) !important;
                color: var(--text) !important;
            }}

            /* ---- file uploader ---- */
            div[data-testid="stFileUploader"] section {{
                background-color: var(--surface);
                border: 1px dashed var(--border);
                border-radius: 10px;
            }}
            div[data-testid="stFileUploader"] section > div,
            div[data-testid="stFileUploaderDropzoneInstructions"] {{
                color: var(--text) !important;
            }}
            div[data-testid="stFileUploaderDropzoneInstructions"] span,
            div[data-testid="stFileUploaderDropzoneInstructions"] small {{
                color: var(--text-muted) !important;
            }}
            div[data-testid="stFileUploaderDropzoneInstructions"] svg {{
                fill: var(--text-muted) !important;
            }}
            div[data-testid="stFileUploader"] button {{
                background-color: var(--surface) !important;
                border: 1px solid var(--border) !important;
                color: var(--text) !important;
            }}
            div[data-testid="stFileUploader"] button p,
            div[data-testid="stFileUploader"] button span {{
                color: var(--text) !important;
            }}
            div[data-testid="stFileUploader"] button:hover {{
                border-color: var(--accent) !important;
                color: var(--accent) !important;
            }}

            /* ---- alerts (st.error / st.info / st.warning) ---- */
            div[data-testid="stAlert"] {{
                background-color: var(--surface);
                border: 1px solid var(--border);
                border-radius: 10px;
                color: var(--text);
            }}

            /* ---- dataframe ---- */
            div[data-testid="stDataFrame"] {{
                border: 1px solid var(--border);
                border-radius: 10px;
            }}

            hr {{ border-color: var(--border-soft); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css(st.session_state.theme)

# ----------------------------------------------------------------------------
# Model loading + inference (all in-process — no API calls)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    return YOLO(model_path)


def draw_custom(img_rgb: np.ndarray, result) -> np.ndarray:
    img = img_rgb.copy()
    for box in result.boxes:
        cls_name = result.names[int(box.cls[0])]
        style = CLASS_STYLE.get(cls_name, {"color": (0, 255, 0), "font_scale": 0.6, "thickness": 2})
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{cls_name} {float(box.conf[0]):.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), style["color"], style["thickness"])
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, style["font_scale"], style["thickness"])
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw, y1), style["color"], -1)
        cv2.putText(img, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    style["font_scale"], (255, 255, 255), style["thickness"], cv2.LINE_AA)
    return img


def run_inference(model, image_rgb: np.ndarray, conf_threshold: float):
    result = model.predict(source=image_rgb, conf=conf_threshold, verbose=False)[0]
    counts = {"unparasitized": 0, "parasitized": 0}
    for box in result.boxes:
        cls_name = result.names[int(box.cls[0])]
        counts[cls_name] = counts.get(cls_name, 0) + 1
    annotated = draw_custom(image_rgb, result)
    return counts, annotated


def confidence_label(n: int) -> str:
    return "low" if n < 500 else "moderate" if n < 2000 else "high"


def aggregate_parasitemia(uploaded_files, model, conf_threshold: float) -> dict:
    total_unparasitized = 0
    total_parasitized = 0
    per_image_results = []
    originals = {}

    for f in uploaded_files:
        try:
            img = Image.open(f).convert("RGB")
            img_rgb = np.array(img)
            counts, annotated = run_inference(model, img_rgb, conf_threshold)
        except Exception as e:
            per_image_results.append({"image": f.name, "error": str(e)})
            continue

        total_unparasitized += counts.get("unparasitized", 0)
        total_parasitized += counts.get("parasitized", 0)
        originals[f.name] = img_rgb

        buf = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))[1]
        annotated_b64 = base64.b64encode(buf.tobytes()).decode("utf-8")

        per_image_results.append({
            "image": f.name,
            "unparasitized": counts.get("unparasitized", 0),
            "parasitized": counts.get("parasitized", 0),
            "annotated_image_base64": annotated_b64,
        })

    total_rbc = total_unparasitized + total_parasitized
    parasitemia_pct = round((total_parasitized / total_rbc * 100) if total_rbc > 0 else 0.0, 3)

    return {
        "num_images": len(uploaded_files),
        "per_image_results": per_image_results,
        "total_unparasitized": total_unparasitized,
        "total_parasitized": total_parasitized,
        "total_rbc_counted": total_rbc,
        "parasitemia_pct": parasitemia_pct,
        "confidence": confidence_label(total_rbc),
        "originals": originals,
    }


# ----------------------------------------------------------------------------
# Severity classification — grounded in published CDC / WHO thresholds
# ----------------------------------------------------------------------------
def classify_severity(pct: float) -> dict:
    """
    Bands are synthesized from published sources, since CDC and WHO use
    DIFFERENT thresholds for "severe" and there is no single universally
    agreed 3-tier mild/moderate/severe scale in the literature:

    - CDC: parasitemia >= 5% is one criterion for severe disease requiring
      aggressive IV antimalarial treatment.
    - WHO: parasitemia > 10% infected erythrocytes is the hyperparasitemia
      threshold for severe P. falciparum malaria.
    - Non-endemic-setting clinical protocols commonly flag >2% as an early
      "elevated" marker, since non-immune patients decompensate faster.

    These bands are informational context only. Severe malaria is a
    clinical diagnosis based on parasitemia AND clinical signs together —
    not parasitemia percentage alone.
    """
    t = THEMES[st.session_state.theme]
    if pct < 2:
        return {
            "label": "Low parasitemia",
            "color": t["sev_low"],
            "note": "Below the 2% threshold used in non-endemic clinical protocols as an early monitoring marker.",
        }
    elif pct < 5:
        return {
            "label": "Elevated parasitemia",
            "color": t["sev_elevated"],
            "note": "Above 2% (non-endemic-setting monitoring threshold), below the CDC's 5% severe-disease criterion.",
        }
    elif pct < 10:
        return {
            "label": "High parasitemia — meets CDC severe-disease threshold",
            "color": t["sev_high"],
            "note": "At or above 5% — one of the CDC's criteria for severe disease, alongside clinical signs.",
        }
    else:
        return {
            "label": "Very high parasitemia — meets WHO hyperparasitemia threshold",
            "color": t["sev_vhigh"],
            "note": "Above 10% infected erythrocytes — WHO's hyperparasitemia threshold for severe P. falciparum malaria.",
        }


def confidence_note(confidence: str, total_rbc: int) -> str:
    benchmarks = {
        "low": "Fewer than 500 RBCs counted — treat this result as preliminary.",
        "moderate": "500–1,999 RBCs counted — a reasonable sample, though more fields would tighten the estimate.",
        "high": "2,000+ RBCs counted — a solid sample size for this estimate.",
    }
    return f"{total_rbc:,} RBCs counted. {benchmarks.get(confidence, '')}"


# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
head_col, toggle_col = st.columns([6, 1])
with head_col:
    st.markdown('<div class="ms-eyebrow">Thin blood smear · parasitemia estimation</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="ms-title">MalariaScope</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ms-subtitle">Detects and counts parasitized vs. unparasitized red blood cells '
        'across microscope field images and estimates aggregate parasitemia.</div>',
        unsafe_allow_html=True,
    )
with toggle_col:
    st.write("")
    label = "☾ Dark" if st.session_state.theme == "dark" else "☀ Light"
    if st.button(label, key="theme_toggle"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

st.markdown('<div class="ms-scanline"></div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Settings — collapsed, no sidebar
# ----------------------------------------------------------------------------
with st.expander("⚙ Detection settings", expanded=False):
    s1, s2 = st.columns(2)
    with s1:
        st.session_state.model_path = st.text_input("Model weights path (.pt)", value=st.session_state.model_path)
    with s2:
        st.session_state.conf_threshold = st.slider(
            "Detection confidence threshold", 0.0, 1.0, st.session_state.conf_threshold, 0.05
        )

model_path = st.session_state.model_path
conf_threshold = st.session_state.conf_threshold

if not Path(model_path).exists():
    st.warning(
        f"Model weights not found at `{model_path}`. Update the path above (Detection settings) "
        "to point to your trained `best.pt`."
    )

# ----------------------------------------------------------------------------
# Upload + run
# ----------------------------------------------------------------------------
st.markdown('<div class="ms-section-label">Upload</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader(
    "Upload microscope field images (multiple fields from the same slide recommended)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

run = st.button(
    "Run analysis", type="primary",
    disabled=not uploaded_files or not Path(model_path).exists(),
)

if run and uploaded_files:
    with st.spinner(f"Running inference on {len(uploaded_files)} image(s)..."):
        model = load_model(model_path)
        result = aggregate_parasitemia(uploaded_files, model, conf_threshold)
        st.session_state["result"] = result
        st.session_state["originals"] = result.pop("originals")

# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
if "result" in st.session_state:
    data = st.session_state["result"]
    originals = st.session_state.get("originals", {})
    severity = classify_severity(data["parasitemia_pct"])

    st.markdown('<div class="ms-section-label">Summary</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Parasitemia", f"{data['parasitemia_pct']}%")
    col2.metric("Unparasitized RBCs", f"{data['total_unparasitized']:,}")
    col3.metric("Parasitized RBCs", f"{data['total_parasitized']:,}")
    col4.metric("Fields analyzed", data["num_images"])

    st.markdown(
        f'<span class="severity-badge" style="background:{severity["color"]}22;'
        f'color:{severity["color"]};border-color:{severity["color"]}55">{severity["label"]}</span>',
        unsafe_allow_html=True,
    )
    st.markdown(f'<div class="ms-caption">{severity["note"]}</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="ms-reliability">Sample reliability — {confidence_note(data["confidence"], data["total_rbc_counted"])}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="ms-disclaimer">⚠ <b>Not a diagnostic device.</b> This is a research/portfolio '
        'prototype. Results are not validated for clinical use and must not replace confirmed '
        'laboratory diagnosis by a qualified clinician or certified microscopist.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("How this analysis is calculated"):
        st.markdown(
            """
Counts across all uploaded fields are **summed first, then divided once** to get the
aggregate parasitemia percentage — the standard CDC/WHO method, rather than averaging
each field's percentage separately.

**Clinical reference:**
- **CDC** — parasitemia ≥ 5%, combined with clinical signs, is one criterion for severe
  disease requiring aggressive IV antimalarial therapy.
- **WHO** — parasitemia > 10% infected erythrocytes is the hyperparasitemia threshold for
  severe *P. falciparum* malaria.
- **Non-endemic-setting protocols** — commonly flag > 2% as an early monitoring marker,
  since non-immune patients can decompensate faster than in endemic populations.

**Sample-size confidence in this app:** fewer than 500 RBCs counted is labeled *low*,
500–1,999 is *moderate*, and 2,000+ is *high* — more fields analyzed together generally
means a tighter estimate.

*These bands are informational — severe malaria is diagnosed clinically, using
parasitemia together with symptoms and lab findings, not percentage alone.*
            """
        )

    st.markdown('<div class="ms-section-label">Per-field detail</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ms-caption" style="margin-bottom:0.75rem;">Compare each field\'s original image against detected bounding boxes.</div>',
        unsafe_allow_html=True,
    )

    for item in data["per_image_results"]:
        if item.get("error"):
            with st.expander(f"{item['image']} — error"):
                st.warning(f"Could not process this image: {item['error']}")
            continue

        label = f"{item['image']} — {item['unparasitized']} unparasitized · {item['parasitized']} parasitized"
        with st.expander(label):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown('<div class="ms-caption">Original</div>', unsafe_allow_html=True)
                if item["image"] in originals:
                    st.image(originals[item["image"]], use_container_width=True)
                else:
                    st.info("Original not available.")

            with c2:
                st.markdown('<div class="ms-caption">Detected</div>', unsafe_allow_html=True)
                if item.get("annotated_image_base64"):
                    st.image(
                        base64.b64decode(item["annotated_image_base64"]),
                        use_container_width=True,
                        caption=f"{item['unparasitized']} unparasitized · {item['parasitized']} parasitized",
                    )
                else:
                    st.info("Annotated image not returned.")

    st.markdown('<div class="ms-section-label">Export</div>', unsafe_allow_html=True)
    df_rows = [r for r in data["per_image_results"] if not r.get("error")]
    if df_rows:
        df = pd.DataFrame([{k: v for k, v in r.items() if k != "annotated_image_base64"} for r in df_rows])
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download per-image results (CSV)",
            df.to_csv(index=False).encode("utf-8"),
            file_name="parasitemia_results.csv",
            mime="text/csv",
        )
else:
    st.info("Upload one or more slide field images, then click **Run analysis**.")