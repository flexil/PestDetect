import streamlit as st
from ultralytics import YOLO
from PIL import Image

@st.cache(allow_output_mutation=True)
def load_model(x):
    model = YOLO(x)
    return model

model = load_model('(link unavailable)')

st.title("Pest Detection Model")
st.subheader("Upload an image")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)

if st.button("Run Inference"):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        results = model.predict(image, save=True, imgsz=640, conf=confidence_threshold)
        st.subheader("Detection Results")
        for i, r in enumerate(results):
            im_bgr = r.plot()
            im_rgb = Image.fromarray(im_bgr[..., ::-1])
            st.image(im_rgb, caption=f"Detection {i+1}")
            class_id = r.boxes.xyxy[0][5].int().item()  # Get class ID
            class_name = results.names[int(class_id)]  # Get class name
            st.write(f"Class: {class_name}")
    else:
        st.write("Please upload an image")
