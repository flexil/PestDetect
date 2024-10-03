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

def process_video(video_file, model, confidence_threshold):
    cap = cv2.VideoCapture(video_file)
    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            results = model(frame)
            annotated_frame = results[0].plot()
            yield annotated_frame, frame_count
            frame_count += 1
        else:
            break
    cap.release()

def save_predicted_images(save_path, frame):
    try:
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        cv2.imwrite(save_path, frame)
    except Exception as e:
        print(f"Error saving image: {e}")
        print(f"Save Path: {save_path}")
        print(f"Image Shape: {frame.shape}")

def main():
    st.title("Pest Detection Model")
    st.subheader("Upload Media")

    media_type = st.selectbox("Select Media Type", ["Image", "Video"])
    uploaded_file = st.file_uploader("Choose Media", type=["jpg", "jpeg", "png", "mp4"])
    confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

    if st.button("Run Inference"):
        if uploaded_file is not None:
            model = load_model("best.pt")  # Replace with actual model path
            if model is not None:
                if media_type == "Image":
                    image = Image.open(uploaded_file)
                    results = process_image(image, model, confidence_threshold)
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
                    temp_file = tempfile.NamedTemporaryFile(suffix=".mp4")
                    temp_file.write(uploaded_file.read())
                    temp_file.seek(0)
                    st.subheader("Video Inference Results")
                    st.write("Processing video...")
                    progress_bar = st.progress(0)
                    frame_count = 0
                    for annotated_frame, frame_count in process_video(temp_file.name, model, confidence_threshold):
                        st.image(annotated_frame, caption=f"Frame {frame_count}")
                        progress_bar.progress(min(frame_count / cv2.VideoCapture(temp_file.name).get(cv2.CAP_PROP_FRAME_COUNT), 1))
                    temp_file.close()
                    st.write("Video processing complete")
            else:
                st.error("Model loading failed")
        else:
            st.write("Please upload media")

main()
