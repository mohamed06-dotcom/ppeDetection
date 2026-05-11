import streamlit as st
from PIL import Image
import cv2
import tempfile
import os


@st.dialog("Warning , person not detected")
def warn_dialog(message:str):
    st.write(message)

    if st.button("Yes , proceed"):
        st.write("Proceeding")

def render(detector, processor, log_violation, format_image_for_display, render_per_person_report):
    col_upload, col_report = st.columns([2, 1], gap="large")

    with col_upload:
        st.subheader("Inspection Media")
        analysis_type = st.segmented_control(
            "Select Media Type", ["Image", "Video"], default="Image"
        )

        if analysis_type == "Image":
            uploaded_file = st.file_uploader(
                "Upload static image for inspection", type=["jpg", "png", "jpeg"]
            )
            if uploaded_file:
                image = Image.open(uploaded_file)
                with st.spinner("Running inference..."):
                    results, annotated_rgb = detector.run_inference_annotated(image)

                st.session_state.last_per_person = processor.check_ppe_per_person(results)
                st.session_state.last_annotated  = annotated_rgb

                st.image(
                    format_image_for_display(annotated_rgb, max_height=500),
                    caption="Inference Output"
                )

            elif st.session_state.last_annotated is not None:
                st.image(
                    format_image_for_display(st.session_state.last_annotated, max_height=500),
                    caption="Last Inference Output"
                )

        elif analysis_type == "Video":
            uploaded_video = st.file_uploader(
                "Upload recorded surveillance footage", type=["mp4", "avi", "mov"]
            )
            if uploaded_video:
                suffix = "." + uploaded_video.name.split(".")[-1].lower() if "." in uploaded_video.name else ".mp4"

                tfile = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                try:
                    tfile.write(uploaded_video.read())
                    tfile.flush()
                    tfile.close()

                    vf = cv2.VideoCapture(tfile.name)
                    stframe = st.empty()
                    stop_video = st.button("Stop Playback", type="secondary")

                    frame_count = 0
                    while vf.isOpened() and not stop_video:
                        ret, frame = vf.read()
                        if not ret:
                            break

                        results, annotated_rgb = detector.run_inference_annotated(frame)
                        stframe.image(
                            format_image_for_display(annotated_rgb, max_height=500),
                            channels="RGB"
                        )

                        if frame_count % 30 == 0:
                            st.session_state.last_per_person = processor.check_ppe_per_person(results)
                            for p in st.session_state.last_per_person:
                                if not p["compliant"]:
                                    log_violation("Video Analysis", p["person_id"], p["missing"])

                        frame_count += 1

                    vf.release()
                    st.success(f"Processing complete — {frame_count} frames analysed.")
                finally:
                    try:
                        os.unlink(tfile.name)
                    except PermissionError:
                        pass

    with col_report:
        st.subheader("Compliance Report")
        per_person = st.session_state.last_per_person

        if per_person:
            source = "Image Upload" if analysis_type == "Image" else "Video Analysis"
            render_per_person_report(per_person, source=source)
        else:
            st.info("Upload media to generate a compliance report.")