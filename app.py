import streamlit as st
from ultralytics import YOLO
import torch
from PIL import Image
import cv2
import tempfile
import time

@st.cache(allow_output_mutation=True)
def load_model():
    model = YOLO("(link unavailable)")
    return model

model = load_model()

st.title("Pest Detection Model")


st.subheader("Upload Media")
media_type = st.selectbox("Select Media Type", ["Image", "Video"])
uploaded_file = st.file_uploader("Choose Media", type=["jpg", "jpeg", "png", "mp4"])


confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

if st.button("Run Inference"):
    if uploaded_file is not None:
        if media_type == "Image":

            image = Image.open(uploaded_file)


            results = model.predict(image, save=True, imgsz=640, conf=confidence_threshold)


            st.subheader("Detection Results")
            for i, r in enumerate(results):
                im_bgr = r.plot()
                im_rgb = Image.fromarray(im_bgr[..., ::-1])
                st.image(im_rgb, caption=f"Detection {i+1}")
                st.write(r)

        elif media_type == "Video":

            temp_file = tempfile.NamedTemporaryFile(suffix=".mp4")
            temp_file.write(uploaded_file.read())
            temp_file.seek(0)


            cap = cv2.VideoCapture(temp_file.name)


            st.subheader("Video Inference Results")
            st.write("Processing video...")


            progress_bar = st.progress(0)

            frame_count = 0
            while cap.isOpened():
                success, frame = cap.read()
                if success:
   
                    results = model(frame)
 
                    annotated_frame = results[0].plot()

                    st.image(annotated_frame, caption=f"Frame {frame_count}")


                    frame_count += 1
                    progress_bar.progress(min(frame_count / cap.get(cv2.CAP_PROP_FRAME_COUNT), 1))

                    if frame_count >= cap.get(cv2.CAP_PROP_FRAME_COUNT):
                        break

                else:
                    break


            cap.release()

            temp_file.close()

            st.write("Video processing complete")

    else:
        st.write("Please upload media")
