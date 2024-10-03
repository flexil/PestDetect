
import streamlit as st
from ultralytics import YOLO
from PIL import Image

@st.cache_data(ttl=250)
def load_model(x):
    model = YOLO(x)
    return model

model = load_model('best.pt')

st.title("Pest Detection Model")
st.subheader("Upload an image")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

if st.button("Run Inference"):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        results = model.predict(image, save=False, imgsz=640, conf=confidence_threshold)
        st.subheader("Detection Results")
        for i, r in enumerate(results):
            im_bgr = r.plot()
            im_rgb = Image.fromarray(im_bgr[..., ::-1])
            st.image(im_rgb, caption="Detections")
            
            # Iterate over detections
            for box in r.boxes:
                x1, y1, x2, y2, confidence, class_id = box
                class_name = model.names[int(class_id)]
                st.write(f"Class ID: {class_id}, Class: {class_name}, Confidence: {confidence:.2f}")
                st.write(f"Detection Box: ({x1}, {y1}, {x2}, {y2})")
    else:
        st.write("Please upload an image")
