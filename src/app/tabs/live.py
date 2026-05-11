import streamlit as st
import cv2
from datetime import datetime


@st.cache_data(ttl=60)
def get_available_cameras(max_check: int = 3) -> list:
    available = []
    for i in range(max_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available.append(i)
            cap.release()
    return available if available else [0]


def render(detector, processor, log_violation, format_image_for_display,
           cam_index, auto_log, use_roi, show_fps, draw_roi):

    col_feed, col_status = st.columns([3, 1], gap="large")

    with col_status:
        st.subheader("Live Status")
        col_start, col_stop = st.columns(2)
        with col_start:
            if st.button("Start Feed", use_container_width=True, type="primary"):
                st.session_state.camera_running = True
        with col_stop:
            if st.button("Stop Feed", use_container_width=True):
                st.session_state.camera_running = False

        st.divider()
        live_panel = st.empty()

    with col_feed:
        FRAME_WINDOW = st.empty()

        if st.session_state.camera_running:
            cap = cv2.VideoCapture(cam_index)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
            cap.set(cv2.CAP_PROP_FPS,            30)

            if not cap.isOpened():
                st.error(f"Cannot access Camera {cam_index}.")
                st.session_state.camera_running = False
            else:
                st.toast(f"Connected to Camera {cam_index}")
                frame_count = 0
                prev_time   = datetime.now()

                while st.session_state.camera_running:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Stream interrupted.")
                        break

                    h, w = frame.shape[:2]
                    if use_roi:
                        roi_rect = (int(w * 0.15), int(h * 0.15),
                                    int(w * 0.70), int(h * 0.70))
                        frame = draw_roi(frame, roi_rect)

                    results, annotated_rgb = detector.run_inference_annotated(frame)

                    now     = datetime.now()
                    elapsed = (now - prev_time).total_seconds()
                    fps     = 1.0 / elapsed if elapsed > 0 else 0
                    prev_time = now

                    if show_fps:
                        annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
                        cv2.putText(annotated_bgr, f"FPS: {fps:.1f}",
                                    (15, 35), cv2.FONT_HERSHEY_DUPLEX,
                                    0.8, (0, 255, 0), 1)
                        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

                    FRAME_WINDOW.image(
                        format_image_for_display(annotated_rgb, max_height=600),
                        channels="RGB"
                    )

                    if frame_count % 30 == 0:
                        per_person = processor.check_ppe_per_person(results)

                        panel_md = ""
                        for p in per_person:
                            pid    = p["person_id"] + 1
                            color  = "green" if p["compliant"] else "red"
                            status = "Compliant" if p["compliant"] else "Non-Compliant"
                            helmet = "Detected" if p["has_helmet"] else "**Missing**"
                            vest   = "Detected" if p["has_vest"]   else "**Missing**"
                            boots  = "Detected" if p["has_boots"]  else "**Missing**"
                            panel_md += (
                                f"**Person {pid}** — :{color}[{status}]  \n"
                                f"Helmet: {helmet} | Vest: {vest} | Boots: {boots}\n\n"
                            )

                        live_panel.markdown(
                            panel_md if panel_md else "No persons detected."
                        )

                        if auto_log:
                            for p in per_person:
                                if not p["compliant"]:
                                    log_violation(
                                        f"Live Cam {cam_index}",
                                        p["person_id"],
                                        p["missing"]
                                    )

                    frame_count += 1

                cap.release()
                FRAME_WINDOW.empty()
                st.info("Camera offline.")

        else:
            FRAME_WINDOW.markdown(
                """<div style="border:1px solid rgba(255,255,255,0.15);border-radius:8px;
                height:450px;display:flex;align-items:center;justify-content:center;
                flex-direction:column;gap:8px;opacity:0.4;">
                    <h3 style="margin:0;font-weight:500;">Camera Offline</h3>
                    <p style="margin:0;font-size:14px;">
                        Press <b>Start Feed</b> to initialize live surveillance.
                    </p>
                </div>""",
                unsafe_allow_html=True
            )