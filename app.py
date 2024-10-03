import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO("best.pt")

# Define class names
class_names = [
    'Aphids', 
    'Early Blight', 
    'Healthy Leaf', 
    'Leaf Curl', 
    'Leafhoppers and Jassids', 
    'Molds', 
    'Mosaic Virus', 
    'Septoria', 
    'Bacterial Canker', 
    'Bacterial Spot', 
    'Flea Beetle', 
    'Late Blight', 
    'Leafminer', 
    'Powdery Mildew', 
    'Yellow Curl Virus'
]

# Streamlit app
st.title("Tomato Leaf Pest and Disease Detection")
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
            
            # Display image and label
            col1, col2 = st.columns([4, 1])
            with col1:
                st.image(im_rgb, caption=f"Detection {i+1}")
            with col2:
                for detection in r.boxes.xyxy:
                    x1, y1, x2, y2, confidence, class_id = detection
                    class_name = class_names[int(class_id)]
                    st.write(f"**Class:** {class_name}")
                    st.write(f"**Confidence:** {confidence:.2f}")
                    st.write(f"**Detection Box:** ({x1}, {y1}, {x2}, {y2})")
    else:
        st.write("Please upload an image")
