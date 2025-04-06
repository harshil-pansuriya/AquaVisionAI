# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from src.main import MarineMonitoringSystem

st.set_page_config(layout="wide", page_title="AquaVisionAI")
system = MarineMonitoringSystem()

st.markdown("<h1 style='text-align: center; color: #1E90FF;'>AquaVisionAI - Marine Ecosystem Monitoring</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>Upload underwater images to analyze species, coral health, and pollution.</p>", unsafe_allow_html=True)

with st.container():
    uploaded_files = st.file_uploader(
        "Upload Underwater Images",
        type=["jpg", "png"],
        accept_multiple_files=True,
        help="Select one or more images to process."
    )

    if uploaded_files:
        st.markdown("---")
        for uploaded_file in uploaded_files:
            image = np.array(Image.open(uploaded_file))
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

            species, species_conf, coral_status, coral_conf, pollution = system.process_image(image)

            col1, col2 = st.columns([2, 1])
            with col1:
                st.image(image, caption=f"Original Image: {uploaded_file.name}")
                if pollution:
                    img_with_boxes = image.copy()
                    for lbl, box, conf in pollution:
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img_with_boxes, f"{lbl} ({conf:.2f})", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    st.image(img_with_boxes, caption="Pollution Detected")

            with col2:
                st.markdown("#### Analysis Results")
                st.write(f"**Species**: {species}  \nConfidence: {species_conf:.2f}", unsafe_allow_html=True)
                st.write(f"**Coral Health**: {coral_status}  \nConfidence: {coral_conf:.2f}", unsafe_allow_html=True)
                pollution_text = ', '.join([f'{lbl} ({conf:.2f})' for lbl, _, conf in pollution]) if pollution else 'None'
                st.write(f"**Pollution**: {pollution_text}", unsafe_allow_html=True)

            st.markdown("---")

st.markdown("<p style='text-align: center; color: #999; font-size: 12px;'>Powered by xAI - Built for Marine Conservation</p>", unsafe_allow_html=True)