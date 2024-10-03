import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import tempfile
import time
import os

def load_model(x):
    try:
        model = YOLO(x)
        return model
    except Exception as e:
        st.error(f"Model loading error: {e}")
        return None

def process_image(image, model, confidence_threshold):
    try:
        results = model.predict(image, save=True, imgsz=640, conf=confidence_threshold)
        print(f"Results: {results}")
        return results
    except Exception as e:
        print(f"Error during prediction: {e}")
        st.error(f"Image processing error: {e}")
        return None

def main():
    st.title("Pest Detection Model")
    st.subheader("Upload Media")

    media_type = st.selectbox("Select Media Type", ["Image", "Video"])
    uploaded_file = st.file_uploader("Choose Media", type=["jpg", "jpeg", "png", "mp4"])
    confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

    if st.button("Run Inference"):
        if uploaded_file is not None:
            model = load_model("best.pt")  
            if model is not None:
                if media_type == "Image":
                    image = Image.open(uploaded_file)
                    image.save("temp.jpg")  # Save image to disk
                    results = process_image("temp.jpg", model, confidence_threshold)
                    if results is not None and len(results) > 0:
                        st.subheader("Detection Results")
                        for i, r in enumerate(results):
                            im_bgr = r.plot()
                            im_rgb = Image.fromarray(im_bgr[..., ::-1])
                            st.image(im_rgb, caption=f"Detection {i+1}")
                            st.write(r)
                    else:
                        st.error("No detections found")
                elif media_type == "Video":
                    # Video processing code
            else:
                st.error("Model loading failed")
        else:
            st.write("Please upload media")

main()

