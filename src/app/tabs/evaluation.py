import streamlit as st
import pandas as pd
import os
import plotly.graph_objects as go


def render(results_path: str):
    st.subheader("Model Performance Metrics")

    with st.expander("View Metric Definitions"):
        st.latex(r"Precision = \frac{TP}{TP + FP}")
        st.latex(r"Recall = \frac{TP}{TP + FN}")
        st.latex(r"mAP@50 = \frac{1}{N}\sum_{i=1}^{N} AP_i")

    if os.path.exists(results_path):
        try:
            df_results = pd.read_csv(results_path)
            df_results.columns = df_results.columns.str.strip()
            last = df_results.iloc[-1]

            st.markdown("### Current Epoch Performance")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("mAP@50",    f"{last.get('metrics/mAP50(B)',    0):.4f}",
                          f"Epoch {int(last.get('epoch', 0))}")
            col_m2.metric("mAP@50-95", f"{last.get('metrics/mAP50-95(B)', 0):.4f}")
            col_m3.metric("Precision", f"{last.get('metrics/precision(B)', 0):.4f}")
            col_m4.metric("Recall",    f"{last.get('metrics/recall(B)',    0):.4f}")

            loss_cols = [c for c in df_results.columns
                         if "loss" in c.lower() and "train" in c.lower()]
            if loss_cols:
                st.divider()
                st.markdown("### Training Loss Trajectory")

                df_plot = df_results[loss_cols].reset_index(drop=True)
                df_plot.index = df_plot.index + 1

                fig = go.Figure()
                colors = {
                    "train/box_loss": "#1a73e8",
                    "train/cls_loss": "#34a853",
                    "train/dfl_loss": "#ea4335",
                }
                for col in loss_cols:
                    fig.add_trace(go.Scatter(
                        x=df_plot.index,
                        y=df_plot[col],
                        mode="lines",
                        name=col.replace("train/", "").replace("_", " ").title(),
                        line=dict(color=colors.get(col, "#5f6368"), width=2),
                        hovertemplate="Epoch %{x}<br>Loss: %{y:.4f}<extra></extra>",
                    ))

                fig.update_layout(
                    xaxis_title="Epoch",
                    yaxis_title="Loss Value",
                    hovermode="x unified",
                    height=350,
                    margin=dict(l=20, r=20, t=30, b=20),
                    xaxis=dict(tickmode="linear", dtick=5),
                    legend=dict(orientation="h", yanchor="bottom",
                                y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("View Raw Training Data"):
                st.dataframe(df_results, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"Error parsing results file: {e}")
    else:
        st.warning(f"Results file not found at: `{results_path}`")