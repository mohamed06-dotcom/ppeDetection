import streamlit as st
import pandas as pd
from datetime import datetime

def render():
    col_title, col_btn = st.columns([4, 1])
    with col_title:
        st.subheader("Incident Analytics & Logs")
    with col_btn:
        if st.button("Clear Logs", use_container_width=True):
            st.session_state.alert_logs = pd.DataFrame(
                columns=["Timestamp", "Source", "Person", "Missing Equipment"]
            )
            st.rerun()

    if st.session_state.alert_logs.empty:
        st.info("No violations recorded in the current session.")
    else:
        col_metric, col_chart = st.columns([1, 2])

        with col_metric:
            st.metric("Total Recorded Violations", len(st.session_state.alert_logs))
            csv = st.session_state.alert_logs.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Export Logs (CSV)",
                data=csv,
                file_name=f"ppe_logs_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

        with col_chart:
            all_missing = (
                st.session_state.alert_logs["Missing Equipment"]
                .str.split(", ").explode()
            )
            st.bar_chart(all_missing.value_counts(), color="#1a73e8")

        st.markdown("### Detailed Log Register")
        st.dataframe(
            st.session_state.alert_logs,
            use_container_width=True,
            hide_index=True,
            height=300,
        )