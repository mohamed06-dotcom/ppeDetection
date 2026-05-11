import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import streamlit as st
import pandas as pd
import cv2

from src.detection import YOLOInference, Postprocess
from tabs import media, live, alerts, evaluation

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(BASE_DIR, "runs")

st.set_page_config(
    page_title="Industrial PPE Monitor",
    layout="wide",
    initial_sidebar_state="expanded"
)


@st.cache_resource
def load_detector():
    return YOLOInference(confidence=0.25)


detector  = load_detector()
processor = Postprocess(detector.class_names)


if "alert_logs" not in st.session_state:
    st.session_state.alert_logs = pd.DataFrame(
        columns=["Timestamp", "Source", "Person", "Missing Equipment"]
    )
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False
if "last_per_person" not in st.session_state:
    st.session_state.last_per_person = []
if "last_annotated" not in st.session_state:
    st.session_state.last_annotated = None


def log_violation(source: str, person_id: int, missing_items: list):
    if missing_items:
        from datetime import datetime
        new_log = pd.DataFrame({
            "Timestamp":         [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            "Source":            [source],
            "Person":            [f"Person {person_id + 1}"],
            "Missing Equipment": [", ".join(missing_items)],
        })
        st.session_state.alert_logs = pd.concat(
            [new_log, st.session_state.alert_logs], ignore_index=True
        )


def draw_roi(frame, roi_rect):
    overlay = frame.copy()
    x, y, w, h = roi_rect
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.putText(overlay, "Active Monitoring Zone",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    return cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)


def format_image_for_display(img_array, max_height=450):
    h, w = img_array.shape[:2]
    if h > max_height:
        scale = max_height / h
        return cv2.resize(img_array, (int(w * scale), max_height))
    return img_array


def render_per_person_report(per_person: list, source: str = "Manual Media Upload"):
    for p in per_person:
        pid = p["person_id"] + 1
        with st.container(border=True):
            if p["compliant"]:
                st.success(f"Person {pid} — Fully Compliant")
            else:
                st.error(f"Person {pid} — Non-Compliant")
                st.markdown("**Missing:**")
                for item in p["missing"]:
                    st.markdown(f"- {item}")

            c1, c2, c3 = st.columns(3)
            c1.write("Helmet: " + ("Detected" if p["has_helmet"] else "Missing"))
            c2.write("Vest: "   + ("Detected" if p["has_vest"]   else "Missing"))
            c3.write("Boots: "  + ("Detected" if p["has_boots"]  else "Missing"))

            if not p["compliant"]:
                btn_key = f"log_btn_{source}_{pid}"
                if st.button(
                    f"Log Violation — Person {pid}",
                    key=btn_key,
                    type="primary",
                    use_container_width=True,
                ):
                    log_violation(source, p["person_id"], p["missing"])
                    st.success(f"Violation logged for Person {pid}.")



with st.sidebar:
    st.subheader("System Controls")
    st.caption("Configure global detection parameters.")
    global_confidence = st.slider(
        "Detection Confidence", 0.1, 1.0, 0.25, step=0.05,
        help="Higher values reduce false positives."
    )
    detector.set_confidence(global_confidence)

    st.divider()
    st.subheader("Live Feed Settings")

    from tabs.live import get_available_cameras
    available_cams = get_available_cameras()
    cam_index = st.selectbox(
        "Select Camera Source", options=available_cams,
        format_func=lambda x: f"Camera {x}"
    )
    auto_log = st.toggle("Auto-log Violations", value=True)
    use_roi  = st.toggle("Enable ROI Overlay",  value=False)
    show_fps = st.toggle("Show FPS Counter",    value=True)


st.title("Industrial PPE Monitoring System")
st.markdown("Real-time computer vision system for occupational safety and compliance tracking.")
st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "Media Analysis",
    "Live Surveillance",
    "Alerts & Analytics",
    "Model Evaluation",
])

with tab1:
    media.render(
        detector, processor,
        log_violation, format_image_for_display, render_per_person_report
    )

with tab2:
    live.render(
        detector, processor,
        log_violation, format_image_for_display,
        cam_index, auto_log, use_roi, show_fps, draw_roi
    )

with tab3:
    alerts.render()

with tab4:
    evaluation.render(RESULTS_PATH)