
import streamlit as st
from ultralytics import YOLO
from PIL import Image

@st.cache_data(ttl=250)
def load_model(x):
    model = YOLO(x)
    return model

model = load_model('best.pt')
class_names = {
    0: "Aphids",
    1: "Early Blight",
    2: "Healthy Leaf",
    3: "Leaf Curl",
    4: "Leafhoppers and Jassids",
    5: "Molds",
    6: "Mosaic Virus",
    7: "Septoria",
    8: "Bacterial Canker",
    9: "Bacterial Spot",
    10: "Flea Beetle",
    11: "Late Blight",
    12: "Leafminer",
    13: "Powdery Mildew",
    14: "Yellow Curl Virus"
}


st.title("Tomato leaves Pest Detection")
st.write("@maxtekAI")

st.image("banner.jpeg", caption="Tomato leaves disease")

st.subheader(" Disease Model Detection Capabilities")
# Calculate number of columns
num_cols = 2
col_width = len(class_names) // num_cols

# Create columns
cols = st.columns(num_cols)

# Display detection capabilities in columns
for i, (class_id, class_name) in enumerate(class_names.items()):
    with cols[i % num_cols]:
        st.write(f"{class_id}: {class_name}")
    
st.subheader("Upload an image [look for image sample of tomato leaf disease listed above and upload for inference]")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "webp"])
confidence_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)


    
if st.button("Run Inference"):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        results = model.predict(image, save=False, imgsz=640, conf=confidence_threshold)
        st.subheader("Detection Results")
        for i, r in enumerate(results):
            im_bgr = r.plot()
            im_rgb = Image.fromarray(im_bgr[..., ::-1])
            st.image(im_rgb, caption=f"Detection {i+1}")
            for detection in r.boxes.xyxy:
                x1, y1, x2, y2 = detection
                st.write(f"Detection Box: ({x1}, {y1}, {x2}, {y2})")
                # Get the class name from the class_names dictionary
                class_id = int(r.classes[0])  # Assuming r.classes[0] gives the predicted class index
                class_name = class_names[class_id]
                st.write(f"Class: {class_name}")
    else:
        st.write("Please upload an image")
